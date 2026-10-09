"""Official TorchVision detectors. No detection/segmentation algorithm is reimplemented."""
import importlib.metadata
import base64
from pathlib import Path
import random
from functools import partial
from urllib.parse import urlparse
import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN,MaskRCNN,
    FasterRCNN_ResNet50_FPN_Weights,MaskRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d
from .annotations import read_dataset,area_sample
from .schema import Recipe
from .pipeline import input_signature
from .store import atomic_json,file_hash,now


def image_tensor(image):
    return torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()/255


def make_model(kind,labels,*,pretrained=False,input_size=512):
    segment=kind=='segmentation'
    backbone=resnet_fpn_backbone(backbone_name='resnet50',weights=None,
        norm_layer=partial(FrozenBatchNorm2d,eps=0.0),trainable_layers=3)
    model=(MaskRCNN if segment else FasterRCNN)(backbone,num_classes=91,min_size=input_size,max_size=input_size,
                  box_score_thresh=.001,box_detections_per_img=100)
    provenance=None
    if pretrained:
        weight=(MaskRCNN_ResNet50_FPN_Weights if segment else FasterRCNN_ResNet50_FPN_Weights).COCO_V1
        path=Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weight.url).path).name
        expected=path.stem.rsplit('-',1)[-1]
        if not path.is_file() or not file_hash(path).startswith(expected):
            raise ValueError("Official TorchVision detection weights are not cached; prepare them with the setup tool")
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        provenance={'source':weight.url,'sha256':file_hash(path)}
    classes=len(labels)+1
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,classes)
    if segment:
        model.roi_heads.mask_predictor=MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels,256,classes)
    return model,provenance


def infer(model,image,labels):
    with torch.inference_mode():out=model([image_tensor(image)])[0]
    if any(not torch.isfinite(out[k]).all() for k in ('boxes','scores')):
        raise ValueError("Detector returned non-finite output")
    detections=[]
    for index,(b,s,l) in enumerate(zip(out['boxes'],out['scores'],out['labels'])):
        label=int(l)
        if not 1<=label<=len(labels):raise ValueError("Detector returned an unknown class")
        row={'box':b.tolist(),'score':float(s),'label':labels[label-1]}
        if 'masks' in out:
            mask=out['masks'][index,0].cpu().numpy()
            if not np.isfinite(mask).all():raise ValueError("Invalid segmentation mask")
            binary=(mask>=.5).astype(np.uint8)
            ok,encoded=cv2.imencode('.png',binary)
            if not ok:raise ValueError('Could not preserve segmentation evidence')
            row['mask_png']=base64.b64encode(encoded).decode('ascii')
            # Boundary extraction for evidence only; TorchVision produces the mask.
            contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            row['polygons']=[c.reshape(-1,2).tolist() for c in contours if len(c)>=3]
        detections.append(row)
    return {'detections':detections}


def train(store,request):
    recipe=Recipe.model_validate(store.recipe(request['recipe_id'])['config'])
    area=next(a for a in recipe.areas if a.id==request['area_id'])
    if area.method.kind not in ('detection','segmentation'):raise ValueError("Select a labelled inspection method")
    data=read_dataset(store,request['dataset_id'])
    labels=sorted(set(area.method.target_labels))
    if not set(labels)<=set(data['labels']):raise ValueError("Selected labels are missing from the annotation set")
    rows=data['records'];groups=sorted({r['group'] for r in rows})
    if len(rows)<10 or len(groups)<5:raise ValueError("Use at least 10 reviewed images from five independent capture groups")
    random.Random(42).shuffle(groups)
    calibration=set(groups[:max(1,round(len(groups)*.2))])
    calibration.discard(store.read('images',recipe.reference_id)['group'])
    fit=[r for r in rows if r['group'] not in calibration]
    cal=[r for r in rows if r['group'] in calibration]
    if len(fit)<5 or len(cal)<2:raise ValueError("Need five fit and two separate calibration images")
    epochs=int(request.get('epochs',10));size=int(request.get('input_size',512))
    if not 1<=epochs<=100 or size not in (320,512,640):raise ValueError("Invalid training settings")
    torch.manual_seed(42);np.random.seed(42)
    reference=store.image(recipe.reference_id)
    # Validate every transformed annotation before allocating/training the network.
    positives=set()
    for row in rows:
        _,objects=area_sample(store,recipe,area,row,reference)
        if row in fit:positives.update(o['label'] for o in objects)
    if positives!=set(labels):raise ValueError("Every target label needs positive fit examples inside this area")
    model,weights=make_model(area.method.kind,labels,pretrained=True,input_size=size)
    # Fine-tune the established model. Freeze BatchNorm statistics for single-image batches.
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9,weight_decay=.0005)
    losses=[]
    for epoch in range(epochs):
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        ordered=list(fit);random.Random(42+epoch).shuffle(ordered);total=0.
        for row in ordered:
            image,objects=area_sample(store,recipe,area,row,reference)
            target={'boxes':torch.tensor([o['box'] for o in objects],dtype=torch.float32).reshape(-1,4),
                    'labels':torch.tensor([labels.index(o['label'])+1 for o in objects],dtype=torch.int64)}
            if area.method.kind=='segmentation':
                masks=np.stack([o['mask'] for o in objects]) if objects else np.empty((0,*image.shape[:2]),np.uint8)
                target['masks']=torch.from_numpy(masks)
            optimizer.zero_grad(set_to_none=True)
            loss=sum(model([image_tensor(image)],[target]).values())
            if not torch.isfinite(loss):raise ValueError("Training loss is not finite")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            total+=float(loss.detach())
        losses.append(total/len(fit))
    model.eval()
    # Exercise held-out images, but never call these fit/calibration images independent validation.
    for row in cal:infer(model,area_sample(store,recipe,area,row,reference)[0],labels)
    key=request['model_id'];path=store.path('models',key,'.pt');torch.save(model.state_dict(),path)
    metadata={'id':key,'created_at':now(),'framework':'torchvision','framework_version':importlib.metadata.version('torchvision'),
        'algorithm':'maskrcnn_resnet50_fpn' if area.method.kind=='segmentation' else 'fasterrcnn_resnet50_fpn',
        'kind':area.method.kind,'labels':labels,'input_size':size,'pretrained':weights,
        'input_signature':input_signature(recipe,area),'sha256':file_hash(path),'runtime':'torchvision',
        'dataset_id':data['id'],'dataset_hash':data['hash'],'fit_ids':[r['image_id'] for r in fit],
        'calibration_ids':[r['image_id'] for r in cal],'used_groups':groups,'epochs':epochs,'losses':losses,
        'threshold':area.method.threshold,'review_margin':area.method.review_margin,'production_ready':False,
        'runtime_note':'Native TorchVision retained until variable detection/mask export parity is verified'}
    atomic_json(store.path('models',key),metadata)
    return metadata
