"""Local immutable recipes, image/model provenance and atomic production pointer."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import threading
import uuid
import cv2
import numpy as np
from .schema import Recipe
from . import ENGINE_VERSION


def now():
    return datetime.now(timezone.utc).isoformat()


def label_confirmed(image):
    return image.get('label') in ('GOOD','NG') and bool(image.get('confirmed_at'))


def group_confirmed(image):
    return bool(image.get('group')) and image.get('group_confirmed') is True


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self._verified = {}
        for folder in ("images", "recipes", "models", "tests", "evidence", "reviews", "training", "datasets"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def path(self, folder, key, suffix=".json"):
        patterns = {"images": r"image_[0-9a-f]{64}", "recipes": r"recipe_[0-9a-f]{32}",
                    "models": r"model_[0-9a-f]{32}", "tests": r"test_[0-9a-f]{32}",
                    "evidence": r"event_[0-9a-f]{32}", "reviews": r"event_[0-9a-f]{32}",
                    "training": r"train_[0-9a-f]{32}", "datasets": r"data_[0-9a-f]{32}"}
        if folder not in patterns or not re.fullmatch(patterns[folder], key):
            raise ValueError("Invalid record ID")
        return self.root / folder / (key + suffix)

    def read(self, folder, key):
        return json.loads(self.path(folder, key).read_text(encoding="utf-8"))

    def records(self, folder, limit=None):
        paths = sorted((self.root / folder).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if limit is not None:
            paths = paths[:limit]
        return [json.loads(p.read_text(encoding="utf-8")) for p in
                paths]

    def evidence_page(self,limit=100,before=None):
        paths=sorted((self.root/'evidence').glob('*.json'),key=lambda p:(p.stat().st_mtime_ns,p.name),reverse=True)
        start=0
        if before:
            cursor=self.path('evidence',before)
            try:start=paths.index(cursor)+1
            except ValueError as exc:raise ValueError('Evidence cursor is missing; refresh the list') from exc
        selected=paths[start:start+limit]
        events=[]
        for path in selected:
            item=json.loads(path.read_text(encoding='utf-8'))
            review=self.path('reviews',item['id'])
            item['review']=json.loads(review.read_text(encoding='utf-8')) if review.exists() else None
            events.append(item)
        return {'events':events,'next_cursor':selected[-1].stem if selected and start+limit<len(paths) else None}

    def put_image(self, raw, name="image", *, label="UNKNOWN", group=""):
        if not isinstance(group,str) or len(group)>120:
            raise ValueError('Capture group must be text up to 120 characters')
        group = group.strip()
        if label not in ("GOOD", "NG", "UNKNOWN"):
            raise ValueError("Image label must be GOOD, NG or UNKNOWN")
        if not 0 < len(raw) <= 20 * 1024 * 1024:
            raise ValueError("Image must be between 1 byte and 20 MB")
        # Check decompressed dimensions before OpenCV allocates the frame.
        from PIL import Image
        from io import BytesIO
        try:
            with Image.open(BytesIO(raw)) as source:
                if source.format not in {"PNG", "JPEG", "BMP", "TIFF"} or source.width * source.height > 24_000_000:
                    raise ValueError("Unsupported image or image larger than 24 megapixels")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Upload one frame per image")
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Image could not be decoded safely") from exc
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or min(frame.shape[:2]) < 16:
            raise ValueError("Image cannot be decoded or is too small")
        # Pixel identity detects renamed/re-encoded duplicates as well as identical files.
        pixel_hash = hashlib.sha256(str(frame.shape).encode() + frame.tobytes()).hexdigest()
        key = "image_" + pixel_hash
        with self.lock:
            path = self.path("images", key)
            if path.exists():
                existing = self.read("images", key)
                if group and group_confirmed(existing) and existing['group'] != group:
                    raise ValueError('Identical image already belongs to a different capture group; confirm its source group first')
                if label != "UNKNOWN" and existing["label"] not in (label, "UNKNOWN"):
                    raise ValueError("Identical image already has a conflicting label")
                if label != "UNKNOWN":
                    existing["label"] = label
                    existing['confirmed_at'] = now()
                if group:
                    existing.update(group=group,group_confirmed=True)
                if label != 'UNKNOWN' or group:
                    atomic_json(path, existing)
                return existing
            ok, png = cv2.imencode(".png", frame)
            if not ok:
                raise ValueError("Image encoding failed")
            self.path("images", key, ".png").write_bytes(png.tobytes())
            item = {"id": key, "name": Path(name).name[:200], "label": label,
                    "group": str(group or pixel_hash)[:120], "width": frame.shape[1],
                    "height": frame.shape[0], "pixel_sha256": pixel_hash,
                    "sha256": file_hash(self.path("images", key, ".png")), "created_at": now()}
            item.update(group_confirmed=bool(group),source_sha256=hashlib.sha256(raw).hexdigest())
            if label != 'UNKNOWN':
                item['confirmed_at'] = now()
            atomic_json(path, item)
            return item

    def image(self, key):
        meta = self.read("images", key)
        path = self.path("images", key, ".png")
        if file_hash(path) != meta["sha256"]:
            raise ValueError("Source image has changed")
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

    def save_recipe(self, raw, parent=None):
        recipe = Recipe.model_validate(raw)
        if parent:
            self.recipe(parent)
        reference = self.read("images", recipe.reference_id)
        if reference["label"] != "GOOD" or not label_confirmed(reference):
            raise ValueError("Confirm that the reference image is GOOD before saving a recipe")
        self.image(recipe.reference_id)
        models = {}
        for area in recipe.areas:
            if area.method.model_id:
                models[area.method.model_id] = self.model(area.method.model_id)["sha256"]
        item = {"id": "recipe_" + uuid.uuid4().hex, "created_at": now(), "parent": parent,
                "config": recipe.model_dump(), "reference_sha256": reference["sha256"], "models": models,
                "model_metadata":{key:digest(self.model(key)) for key in models}}
        item["hash"] = digest({k: item[k] for k in ("config", "reference_sha256", "models","model_metadata")})
        atomic_json(self.path("recipes", item["id"]), item)
        return item

    def recipe(self, key):
        item = self.read("recipes", key)
        Recipe.model_validate(item["config"])
        fields = ("config", "reference_sha256", "models") + (("model_metadata",) if "model_metadata" in item else ())
        if digest({k: item[k] for k in fields}) != item["hash"]:
            raise ValueError("Recipe has changed after it was saved")
        self.image(item["config"]["reference_id"])
        if self.read("images", item["config"]["reference_id"])["sha256"] != item["reference_sha256"]:
            raise ValueError("Reference does not match recipe")
        for key, sha in item["models"].items():
            metadata = self.model(key)
            if metadata["sha256"] != sha or ("model_metadata" in item and digest(metadata)!=item["model_metadata"][key]):
                raise ValueError("Model does not match recipe")
        return item

    def model(self, key):
        item = self.read("models", key)
        path = self.path("models", key, ".pt")
        stat = path.stat()
        token = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, item["sha256"])
        if self._verified.get(key) != token and file_hash(path) != item["sha256"]:
            raise ValueError("Model is missing or has changed")
        self._verified[key] = token
        if item.get("runtime")=="onnxruntime":
            path = self.path("models",key,".onnx")
            stat = path.stat()
            token = (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,item["onnx_sha256"])
            if self._verified.get(key+".onnx") != token and file_hash(path)!=item["onnx_sha256"]:
                raise ValueError("ONNX model is missing or has changed")
            self._verified[key+".onnx"] = token
        return item

    def production(self):
        path = self.root / "production.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def validated_recipe(self, recipe_id, report_id):
        with self.lock:
            recipe = self.recipe(recipe_id)
            report = self.read("tests", report_id)
            if report["recipe_hash"] != recipe["hash"] or not report.get("eligible"):
                raise ValueError("This exact recipe needs a passing independent validation report")
            if report.get("report_hash") != digest({k:v for k,v in report.items() if k!="report_hash"}):
                raise ValueError("Validation report has changed")
            if report.get('engine_version')!=ENGINE_VERSION:
                raise ValueError('Inspection software changed; run validation again before Production')
            if recipe['config']['locator']['method']=='fixed':
                raise ValueError("Continuous place-and-remove Production needs a part locator")
            if not report.get('source_provenance'):
                raise ValueError('Validation source provenance is missing; run validation again')
            for key,source in report['source_provenance'].items():
                current = self.read('images',key)
                if any(current.get(field)!=value for field,value in source.items()):
                    raise ValueError('Reference/training/calibration confirmation or capture group changed; validate again')
            if report.get('object_metrics'):
                from .annotations import read_dataset
                metrics=report['object_metrics']
                if read_dataset(self,metrics['dataset_id'])['hash']!=metrics['dataset_hash']:
                    raise ValueError('Validation annotations changed')
            for row in report["rows"]:
                image = self.read("images",row["image_id"])
                self.image(row["image_id"])
                if image["label"]!=row["label"] or image["group"]!=row["group"]:
                    raise ValueError("Validation image labels or capture groups have changed")
                if not row.get('excluded') and (not label_confirmed(image) or not group_confirmed(image)):
                    raise ValueError('Validation image confirmation is missing')
            return recipe

    def activate(self, recipe_id, report_id):
        with self.lock:
            recipe = self.validated_recipe(recipe_id, report_id)
            old = self.production()
            item = {"engine_version":ENGINE_VERSION,"recipe_id": recipe_id, "recipe_hash": recipe["hash"], "report_id": report_id,
                    "activated_at": now(), "previous": ({k: old[k] for k in
                    ("recipe_id", "recipe_hash", "report_id", "activated_at")} if old else None)}
            atomic_json(self.root / "production.json", item)
            return item

    def rollback(self):
        old = self.production()
        if not old or not old.get("previous"):
            raise ValueError("No previous Production recipe")
        prev = old["previous"]
        return self.activate(prev["recipe_id"], prev["report_id"])
