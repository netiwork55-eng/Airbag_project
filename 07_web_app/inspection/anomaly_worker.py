"""Isolated official Anomalib adapter. No anomaly algorithm is implemented here.

Only local state_dict artifacts created by this application are loaded, weights_only=True.
No image, logging or model request goes to cloud services.
"""
import os
os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
import base64
import contextlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import traceback
import zlib
from itertools import chain
import cv2
import numpy as np
import torch
from torchvision.transforms import v2
from inspection.pipeline import prepare, crop, input_signature
from inspection.schema import Recipe
from inspection.store import Store, atomic_json, file_hash, now

torch.set_num_threads(4)
TRANSFORM = v2.Compose([v2.ToImage(), v2.Resize((224,224), antialias=True),
                        v2.ToDtype(torch.float32,scale=True),
                        v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])


def tensor(frame):
    return TRANSFORM(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).unsqueeze(0)


def model_for(algorithm):
    if algorithm == "padim":
        from anomalib.models.image.padim.torch_model import PadimModel
        return PadimModel(backbone="resnet18",layers=["layer1","layer2","layer3"],pre_trained=False)
    if algorithm == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        return PatchcoreModel(backbone="resnet18",layers=["layer2","layer3"],pre_trained=False)
    raise ValueError("Supported Anomalib methods: padim, patchcore")


def backbone(model):
    # The same official TorchVision ResNet18 V1 weights already used on this computer.
    path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    if not path.is_file() or not file_hash(path).startswith("f37072fd"):
        raise ValueError("Verified official TorchVision ResNet18 weights are not cached; run the documented setup first")
    state = torch.load(path,map_location="cpu",weights_only=True)
    extractor = model.feature_extractor.feature_extractor
    expected = extractor.state_dict()
    # Older official ResNet weights omit BatchNorm counters (not learned parameters).
    extractor.load_state_dict({key:(torch.zeros_like(value) if key.endswith("num_batches_tracked")
                                   and key not in state else state[key])
                               for key,value in expected.items()},strict=True)
    return file_hash(path)


def predict(model, image):
    with torch.inference_mode():
        output = model(tensor(image))
    score = float(output.pred_score.reshape(-1)[0])
    amap = output.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
    if not np.isfinite(score) or not np.isfinite(amap).all():
        raise ValueError("Anomalib returned non-finite output")
    return score,amap


def onnx_session(path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])


def export_onnx(store, key, model, samples, threshold):
    """Use the official PyTorch exporter; adopt only after score/map/decision parity."""
    import onnx
    class Outputs(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, image):
            result = self.inner(image)
            return result.pred_score,result.anomaly_map
    path = store.path("models",key,".onnx")
    wrapper = Outputs(model).eval()
    samples = iter(samples)
    first = next(samples)
    torch.onnx.export(wrapper,tensor(first),str(path),input_names=["image"],
        output_names=["score","anomaly_map"],opset_version=18,dynamo=False,external_data=False)
    onnx.checker.check_model(str(path))
    session = onnx_session(path)
    differences = []
    outcome = lambda score: "NG" if score>threshold*1.05 else "PASS" if score<threshold*.95 else "REVIEW"
    for image in chain([first],samples):
        expected,expected_map = predict(model,image)
        score,amap = session.run(None,{"image":tensor(image).numpy()})
        score = float(np.asarray(score).reshape(-1)[0])
        amap = np.asarray(amap).squeeze()
        if not np.isfinite(score) or not np.isfinite(amap).all() or not np.isclose(score,expected,rtol=1e-4,atol=1e-4) or not np.allclose(amap,expected_map,rtol=1e-4,atol=1e-4):
            raise ValueError("ONNX score/anomaly-map parity failed; native Anomalib runtime retained")
        if outcome(score)!=outcome(expected):
            raise ValueError("ONNX changes a calibration decision; native runtime retained")
        differences.append(abs(score-expected))
    return {"runtime":"onnxruntime","onnx_sha256":file_hash(path),
            "onnx_parity":{"images":len(differences),"max_score_difference":max(differences),
                           "relative_tolerance":1e-4,"absolute_tolerance":1e-4},
            "onnxruntime_version":importlib.metadata.version("onnxruntime")}


def split_training_groups(images, reference_group):
    """Keep the reference and entire capture groups out of calibration."""
    groups = sorted({i["group"] for i in images})
    if len(groups)<5:
        raise ValueError("At least five independent capture groups are required")
    random.Random(42).shuffle(groups)
    # Exclude the reference before selecting; discarding it afterwards can
    # empty calibration despite other independent groups being available.
    eligible_groups = [group for group in groups if group != reference_group]
    cal_groups = set(eligible_groups[:max(1,round(len(groups)*.2))])
    fit = [i for i in images if i["group"] not in cal_groups]
    cal = [i for i in images if i["group"] in cal_groups]
    if len(fit)<10 or len(cal)<4:
        raise ValueError("Need at least 10 fit and 4 calibration images in separate groups")
    return groups, fit, cal


def train(store, request):
    saved = store.recipe(request["recipe_id"])
    recipe = Recipe.model_validate(saved["config"])
    area = next(a for a in recipe.areas if a.id == request["area_id"])
    algorithm = request["algorithm"]
    images = [store.read("images",i) for i in dict.fromkeys(request["image_ids"])]
    from inspection.store import label_confirmed, group_confirmed
    reference = store.read('images',recipe.reference_id)
    if reference['label']!='GOOD' or not label_confirmed(reference) or not group_confirmed(reference):
        raise ValueError('Confirm the GOOD reference and its actual capture group before training')
    if any(i['label']!='GOOD' or not label_confirmed(i) or not group_confirmed(i) for i in images):
        raise ValueError("Anomaly training requires human-confirmed GOOD images with explicit capture groups")
    if len(images)<20:
        raise ValueError("Use at least 20 distinct GOOD images; no image-count ceiling is imposed")
    ref = store.image(recipe.reference_id)
    groups, fit, cal = split_training_groups(images, reference["group"])
    torch.manual_seed(42)
    np.random.seed(42)
    model = model_for(algorithm)
    backbone_hash = backbone(model)
    model.train()
    def prepared(item):
        frame = store.image(item["id"])
        aligned,visible,_,_ = prepare(frame,ref,recipe)
        if np.any(crop(visible,area.region) != 255):
            raise ValueError("Training area is not fully visible after alignment")
        return crop(aligned,area.region)
    with torch.no_grad():
        for item in fit:
            model(tensor(prepared(item)))
    # Official library routines perform fitting/coreset selection and anomaly-map generation.
    if algorithm == "padim":
        model.fit()
    else:
        model.subsample_embedding(sampling_ratio=.01)
    model.eval()
    scores = [predict(model,prepared(i))[0] for i in cal]
    threshold = float(max(scores))
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError("Could not calibrate a finite positive threshold")
    key = request["model_id"]
    path = store.path("models",key,".pt")
    torch.save(model.state_dict(),path)
    metadata = {"id":key,"created_at":now(),"framework":"anomalib",
        "framework_version":importlib.metadata.version("anomalib"),"algorithm":algorithm,
        "backbone":"torchvision ResNet18 IMAGENET1K_V1","backbone_sha256":backbone_hash,
        "input_size":[224,224],"input_signature":input_signature(recipe,area),
        "recipe_id":saved["id"],"area_id":area.id,"sha256":file_hash(path),
        "fit_ids":[i["id"] for i in fit],"calibration_ids":[i["id"] for i in cal],
        "used_groups":groups,"calibration_scores":scores,"threshold":threshold,
        "review_margin":threshold*.05,"production_ready":False,
        "threshold_note":"Good-only calibration; defect recall is unmeasured until independent NG validation"}
    try:
        metadata.update(export_onnx(store,key,model,(prepared(i) for i in cal),threshold))
    except Exception as exc:
        store.path("models",key,".onnx").unlink(missing_ok=True)
        metadata.update(runtime="anomalib",onnx_export_error=str(exc)[:600])
    atomic_json(store.path("models",key),metadata)
    return metadata


def serve(root):
    store = Store(root)
    cache = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["action"] == "train":
                    response = train(store,request)
                elif request["action"] == "train_detection":
                    from .detection_worker import train as train_detection
                    response = train_detection(store,request)
                elif request["action"] == "predict":
                    key = request["model_id"]
                    metadata = store.model(key)
                    token = (key,metadata["sha256"])
                    if token not in cache:
                        if metadata.get("framework")=="torchvision":
                            from .detection_worker import make_model
                            model,_=make_model(metadata['kind'],metadata['labels'],input_size=metadata['input_size'])
                            model.load_state_dict(torch.load(store.path('models',key,'.pt'),map_location='cpu',weights_only=True),strict=True)
                            model.eval()
                        elif metadata.get("runtime")=="onnxruntime":
                            model = onnx_session(store.path("models",key,".onnx"))
                        else:
                            model = model_for(metadata["algorithm"])
                            model.load_state_dict(torch.load(store.path("models",key,".pt"),map_location="cpu",weights_only=True),strict=True)
                            model.eval()
                        cache[token] = model
                        while len(cache)>4:
                            del cache[next(iter(cache))]
                    raw = base64.b64decode(request["image"],validate=True)
                    image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if metadata.get('framework')=='torchvision':
                        from .detection_worker import infer
                        response=infer(cache[token],image,metadata['labels'])
                    elif metadata.get("runtime")=="onnxruntime":
                        score,amap = cache[token].run(None,{"image":tensor(image).numpy()})
                        score,amap = float(np.asarray(score).reshape(-1)[0]),np.asarray(amap).squeeze().astype(np.float32)
                    else:
                        score,amap = predict(cache[token],image)
                    if metadata.get('framework')!='torchvision':
                        response = {"score":score,"shape":list(amap.shape),
                                    "map":base64.b64encode(zlib.compress(amap.tobytes())).decode()}
                else:
                    raise ValueError("Unknown worker action")
            print(json.dumps({"ok":True,"result":response},allow_nan=False),flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"ok":False,"error":str(exc)}),flush=True)


if __name__ == "__main__":
    serve(sys.argv[1])
