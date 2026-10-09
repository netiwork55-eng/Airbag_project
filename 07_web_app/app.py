r"""
Backend หลักของ Generic Visual Inspection Workspace (FastAPI)

ขอบเขตระบบในไฟล์นี้:
  - Project/Class contract และ metadata ของ dataset workspace
  - CVAT integration สำหรับ annotation; duplicate Annotation Studio ถูกถอดแล้ว
  - upload/install/restore dataset โดยป้องกัน path traversal และแก้ข้อมูลระหว่าง train
  - วิเคราะห์ dataset, preflight, คำนวณค่าที่แนะนำ และควบคุม subprocess train
  - สร้าง Normal Reference จากภาพ Known-good และตรวจ OK/NG แบบไม่จำแนกชนิดตำหนิ
  - ระบุตำแหน่งความผิดปกติด้วยกรอบแดง พร้อม Confidence และ Brand ID เสริม
  - คง legacy detector/training endpoints ไว้เฉพาะเพื่ออ่านข้อมูลย้อนหลังระหว่าง migration

หลักการสำคัญสำหรับผู้รับช่วง:
  - เส้นทางหลักใหม่ใช้ inspection/ (recipe, pipeline, decision, runtime); normal_reference ยังรองรับ legacy live runtime ที่รอ field validation
  - งานกล้องและงาน train ใช้ shared state + lock; การแก้ state ต้องรักษาขอบเขต lock เดิม
  - endpoint ที่แก้ dataset ต้องเรียก dataset_mutation_blocked_response ระหว่าง train
  - upload ZIP ต้องผ่าน staging และ extract_zip_safely ห้าม extract path จากผู้ใช้โดยตรง
  - annotation ต้นฉบับต้องไม่ถูก augmentation แก้ทับ ให้สร้าง export ใหม่เสมอ
  - หน้าเว็บดึงภาพล่าสุดจาก /latest_frame.jpg; Stop ต้องหยุด worker และ cap.release() จริง

ไฟล์นี้มีหลาย subsystem เพื่อคง deployment เดิมไว้ คอมเมนต์หัวข้อแต่ละช่วงด้านล่างคือ
แผนที่สำหรับค้นหา code path ก่อนแก้ หากจะแยกโมดูลต้องแยกพร้อม state/lock และทดสอบ API
contract เดิม ไม่ควรย้ายฟังก์ชันทีละส่วนโดยไม่ตาม dependency

รัน:
  ..\venv\Scripts\python.exe 07_web_app\app.py
"""

from __future__ import annotations

import sys
import csv
import threading
import time
import shutil
import subprocess
import zipfile
import json
import hashlib
import uuid
import re
import os
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import wraps
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest, urlopen

import cv2
import numpy as np
import yaml
from fastapi import Body, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from inspection_area import checked_area, FULL_AREA, checked_areas
from inspection_cycles import PartCycle, save_cycle
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from ultralytics import YOLO
import uvicorn


WEB_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_APP_DIR.parent
CAMERA_CODE_DIR = PROJECT_ROOT / "01_camera"
TRAINING_CODE_DIR = PROJECT_ROOT / "02_training"
BRAND_CODE_DIR = PROJECT_ROOT / "06_brand_classification"
MODEL_DIR = PROJECT_ROOT / "models"
NORMAL_REFERENCE_MODEL_DIR = MODEL_DIR / "normal_reference"
INSPECTION_REVIEW_DIR = PROJECT_ROOT / "03_dataset" / "inspection_reviews"
CVAT_EXPORT_DIR = PROJECT_ROOT / "03_dataset" / "cvat_exports"
CVAT_ANNOTATED_DIR = PROJECT_ROOT / "03_dataset" / "cvat_annotated_ng"

# โฟลเดอร์ปลายทางที่สคริปต์ train เดิมอ่านอยู่แล้ว
AIRBAG_EXPORTS_DIR = PROJECT_ROOT / "05_train_new_object" / "01_yolov8_exports"
AIRBAG_BACKGROUND_DIR = PROJECT_ROOT / "05_train_new_object" / "02_background_images"
BRAND_EXPORTS_DIR = BRAND_CODE_DIR / "01_brand_exports"
BRAND_BACKGROUND_DIR = BRAND_CODE_DIR / "02_unknown_background_images"
UPLOAD_STAGING_DIR = PROJECT_ROOT / "03_dataset" / "web_upload_staging"
TRAIN_LOG_DIR = WEB_APP_DIR / "logs" / "train_jobs"
ANNOTATION_ROOT = PROJECT_ROOT / "03_dataset" / "annotation_studio"
PROJECT_CONFIG_PATH = WEB_APP_DIR / "project_config.json"
DELETED_DATASETS_DIR = PROJECT_ROOT / "03_dataset" / "deleted_dataset_folders"
ANNOTATION_SESSIONS_DIR = ANNOTATION_ROOT / "sessions"
ANNOTATION_EXPORTS_DIR = ANNOTATION_ROOT / "exports"
PYTHON_EXE = PROJECT_ROOT.parent / "venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path(sys.executable)

# ดึง logic เดิมจากไฟล์กล้องหลัก โดยไม่แก้ไฟล์นั้น
sys.path.insert(0, str(WEB_APP_DIR))
sys.path.insert(0, str(CAMERA_CODE_DIR))
sys.path.insert(0, str(TRAINING_CODE_DIR))
import airbag_webcam_detection as detector  # noqa: E402
import train_from_yolov8_exports as locator_training  # noqa: E402
from background_workflow import (  # noqa: E402
    MIN_EVAL_IMAGES as MIN_BACKGROUND_EVAL_IMAGES,
    background_images as split_background_images,
    background_integrity as detection_background_integrity,
    background_summary as detection_background_summary,
    install_background_upload,
)
from normal_reference import (  # noqa: E402
    DecisionStabilizer,
    InspectionResult,
    NormalReferenceInspector,
    iter_images,
    yolo_primary_box,
)
from deep_patch_reference import DeepPatchReference, load_inspection_reference  # noqa: E402
from inspection_contract import RecipeBoundInspector, build_inspection_contract  # noqa: E402
from inspection_jobs import make_job, job_path, checked_settings  # noqa: E402

JOB_DIR = PROJECT_ROOT / "jobs"


# ===== วงจรชีวิตแอปและทรัพยากรร่วม =====
# startup เตรียม cache/model ที่จำเป็น ส่วน shutdown ต้องปิดกล้องและ process train
# ให้ครบ เพื่อไม่ให้กล้องถูกล็อกหรือมี Python process ค้างหลังปิดเว็บ
@asynccontextmanager
async def lifespan(_: FastAPI):
    set_latest_frame(make_status_frame("Ready", "Press Start Stream"))
    # Inspection starts from an explicitly selected recipe; legacy models load on demand.
    # CVAT is an on-demand connector after human review. Starting inspection
    # must not start annotation polling or perform remote annotation writes.
    try:
        yield
    finally:
        vision_service.close()
        REFERENCE_TRAIN_STOP_EVENT.set()
        stop_camera_worker()
        with TRAIN_LOCK:
            if TRAIN_PROCESS is not None:
                stop_train_process_locked()


app = FastAPI(title="Industrial Visual Inspection", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
app.mount("/static", StaticFiles(directory=WEB_APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_APP_DIR / "templates")


MODEL_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()

SHARED_MODELS: tuple[YOLO, YOLO | None, YOLO | None, YOLO | None] | None = None
SHARED_MODEL_ERROR: str | None = None
SHARED_LOCATOR: tuple[object, int] | None = None
SHARED_LOCATOR_KEY = ""

WORKER_THREAD: threading.Thread | None = None
STOP_EVENT = threading.Event()
CVAT_BRIDGE_LOCK = threading.RLock()
CURRENT_CAP = None

STATE = {
    "running": False,
    "status": "idle",
    "error": "",
    "camera_index": detector.DEFAULT_CAMERA_INDEX,
    "frame_count": 0,
    "raw_frame_count": 0,
    "fps": 0.0,
    "started_at": None,
    "stopped_at": None,
}

NG_CAPTURE_INTERVAL_SECONDS = 1.0
NG_EVENT_RELEASE_SECONDS = 1.25

LATEST_JPEG: bytes | None = None
TRAIN_LOCK = threading.RLock()
JSON_WRITE_LOCK = threading.RLock()
ANALYSIS_CACHE_LOCK = threading.RLock()
ANALYSIS_CACHE: dict[str, tuple[float, object]] = {}
TRAIN_RUN_CACHE_LOCK = threading.RLock()
TRAIN_RUN_CACHE: dict[str, object] = {"key": None, "records": []}
EVALUATION_PREVIEW_CACHE_LOCK = threading.RLock()
EVALUATION_PREVIEW_CACHE: dict[str, object] = {"key": None, "tiles": ()}
TRAIN_PROCESS: subprocess.Popen | None = None
TRAIN_LOG_HANDLE = None
TRAIN_STATE = {
    "running": False,
    "status": "idle",
    "train_type": "",
    "started_at": None,
    "finished_at": None,
    "stop_requested_at": None,
    "returncode": None,
    "log_path": "",
    "command": [],
}

# Normal-reference training runs inside this process, but never inside the HTTP
# request that starts it. Keeping backend-owned state lets the browser poll,
# refresh, or reconnect without making the job appear to vanish.
# Both trainers share one lock: mutual exclusion and one lock order.
REFERENCE_TRAIN_LOCK = TRAIN_LOCK
REFERENCE_TRAIN_STOP_EVENT = threading.Event()
WORKFLOW_LOCK = threading.RLock()


class ReferenceTrainingCancelled(Exception):
    """Cooperative cancellation; never report an interrupted build as finished."""


def check_reference_cancelled() -> None:
    if REFERENCE_TRAIN_STOP_EVENT.is_set():
        raise ReferenceTrainingCancelled("Reference training stopped by user.")


def serialized_workflow(function):
    """Serialize camera/train transitions without blocking status polling."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        from inspection.ownership import InspectionBusy
        with WORKFLOW_LOCK:
            try:
                with vision_service.runtime.ownership.hold('Web legacy operation'):
                    return function(*args, **kwargs)
            except InspectionBusy as exc:
                return JSONResponse({'ok':False,'error':str(exc)},status_code=409)
    return wrapped

REFERENCE_TRAIN_THREAD: threading.Thread | None = None
REFERENCE_TRAIN_STATE = {
    "job_id": "",
    "running": False,
    "status": "idle",
    "stage": "",
    "stage_detail": "",
    "progress": 0.0,
    "epoch": 0,
    "epochs": 0,
    "last_progress_at": None,
    "total_images": 0,
    "started_at": None,
    "finished_at": None,
    "error": "",
    "dataset": "",
    "result": {},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTO_SPLIT_SEED = 42
AUTO_SPLIT_RATIOS = {"train": 0.80, "valid": 0.15, "test": 0.05}
AUTO_SPLIT_MANIFEST = "auto_split_manifest.json"
YOLO_CLASS_NAME_FILES = {"classes.txt", "obj.names", "names.txt", "labels.txt"}
MAX_INSPECTION_IMAGE_BYTES = 32 * 1024 * 1024
MAX_UPLOAD_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_MEMBERS = 100_000
MAX_ZIP_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_ANNOTATION_COLORS = [
    "#20f3a5",
    "#00eaff",
    "#ff4fd8",
    "#ffd166",
    "#8d6cff",
    "#ff7a59",
]
ANNOTATION_SHAPE_TYPES = {"bbox", "polygon"}
ANNOTATION_LABEL_TOOLS = {"bbox", "polygon", "ai_select"}
ANNOTATION_EXPORT_FORMATS = {"bbox", "segmentation"}
ANNOTATION_AUGMENTATIONS = {
    "flip_horizontal",
    "flip_vertical",
    "rotate90",
    "rotate_minus90",
    "rotate180",
    "rotate15",
    "rotate_minus15",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "blur",
}
UPLOAD_CATEGORIES = {
    "airbag": {
        "label": "Known-good Reference Dataset",
        "description": "ภาพชิ้นงานปกติที่ใช้สร้างแบบอ้างอิงสำหรับตัดสิน OK / NG",
    },
    "toyota": {
        "label": "Toyota Brand Reference",
        "description": "ภาพอ้างอิงสำหรับการระบุแบรนด์ Toyota (ตัวเลือกเสริม)",
    },
    "honda": {
        "label": "Honda Brand Reference",
        "description": "ภาพอ้างอิงสำหรับการระบุแบรนด์ Honda (ตัวเลือกเสริม)",
    },
    "background": {
        "label": "Negative / Background Reference",
        "description": "ภาพที่ไม่ใช่ชิ้นงานเป้าหมายสำหรับลด false positive (ตัวเลือกเสริม)",
    },
}

# เว็บต้องให้กล้องลื่นก่อน แล้วให้ AI อัปเดตกรอบเป็นจังหวะ
# ถ้าให้ YOLO predict ทุกเฟรมใน thread เดียว กล้องจะค้างเพราะต้องรอ AI ทุกครั้ง
WEB_AI_MIN_INTERVAL_SECONDS = 0.42
WEB_DISPLAY_MAX_FPS = 12.0
WEB_AIRBAG_IMAGE_SIZE = 512
WEB_GENERAL_IMAGE_SIZE = 416
WEB_GENERAL_DETECT_EVERY_N_AI_FRAMES = 8
BACKGROUND_MAX_FALSE_POSITIVE_RATE = 0.10

DEFAULT_PROJECT_CONFIG = {
    "project_name": "Visual Inspection Project",
    "brand": "",
    "model": "",
    "target_classes": ["OK", "NG"],
    "primary_class": "OK",
    "detection_mode": "normal_reference",
    "inspection_strategy": "normal_reference",
    "inspection_modules": {
        "part_locator": True,
        "normal_reference": True,
        "legacy_defect_assist": False,
        "brand_classifier": True,
    },
    "normal_reference": {
        "active_model": "",
        "source_dataset": "",
        "inspection_size": 512,
        "min_good_images": 5,
        "min_region_ratio": 0.002,
        "trained_at": "",
        "training_images": 0,
    },
    "class_aliases": {},
    "class_display_names": {},
    "primary_confidence": 0.30,
    "secondary_confidence": 0.40,
    "camera_index": detector.DEFAULT_CAMERA_INDEX,
    "use_classifier": True,
    "active_detector_model": "",
    "active_locator_model": "",
    "previous_detector_model": "",
    "active_brand_model": "",
    "last_train_type": "",
    "last_train_log": "",
    "last_train_status": "",
    "last_model_validation": {},
    "use_legacy_defect_signal": False,
    "augmentation_profile": {
        "mode": "recommended",
        "values": {},
    },
}

AUGMENTATION_LIMITS = {
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 20.0),
    "translate": (0.0, 0.25),
    "scale": (0.0, 0.5),
    "shear": (0.0, 10.0),
    "fliplr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "erasing": (0.0, 0.8),
}

AUGMENTATION_PRESETS = {
    "off": {
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "erasing": 0.0,
    },
    "recommended": {
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 5.0,
        "translate": 0.08,
        "scale": 0.18,
        "shear": 1.0,
        "fliplr": 0.15,
        "mosaic": 0.35,
        "erasing": 0.10,
    },
}


# ===== Augmentation contract ของหน้า Training =====
# normalize ค่า UI ให้เป็น schema กลางก่อนส่งไปสร้าง dataset/train เพื่อไม่ให้ frontend
# แต่ละเวอร์ชันส่ง key หรือช่วงตัวเลขต่างกันแล้วได้ผล train ที่คาดเดาไม่ได้
def normalize_augmentation_profile(value: object) -> dict:
    """Validate the project augmentation profile before it reaches a trainer."""
    incoming = value if isinstance(value, dict) else {}
    mode = str(incoming.get("mode") or "recommended").strip().lower()
    if mode not in {"off", "recommended", "custom"}:
        mode = "recommended"
    source = incoming.get("values") if isinstance(incoming.get("values"), dict) else {}
    base = dict(AUGMENTATION_PRESETS["off" if mode == "off" else "recommended"])
    if mode == "custom":
        for key, (minimum, maximum) in AUGMENTATION_LIMITS.items():
            try:
                base[key] = max(minimum, min(maximum, float(source.get(key, base[key]))))
            except (TypeError, ValueError):
                pass
    return {
        "mode": mode,
        "values": {key: round(float(base[key]), 4) for key in AUGMENTATION_LIMITS},
    }


# ===== สถานะภาพล่าสุดที่ browser อ่าน =====
# worker กล้องเขียน JPEG ล่าสุดไว้ใน shared state; endpoint อ่านสำเนาภายใต้ lock
# ห้ามส่ง numpy frame ก้อนเดียวกันข้าม thread โดยไม่ copy เพราะกำลังถูกวาดทับได้
def encode_frame(frame, quality: int = 74) -> bytes | None:
    """แปลง OpenCV frame เป็น JPEG bytes สำหรับส่งขึ้นเว็บ."""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return encoded.tobytes()


def make_status_frame(message: str, detail: str = ""):
    """สร้างภาพสถานะไว้แสดงแทนจอดำก่อนกล้อง/AI พร้อม."""
    frame = np.full(
        (detector.CAMERA_HEIGHT, detector.CAMERA_WIDTH, 3),
        (18, 24, 31),
        dtype=np.uint8,
    )
    cv2.putText(
        frame,
        message,
        (28, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (0, 234, 255),
        2,
        cv2.LINE_AA,
    )
    if detail:
        cv2.putText(
            frame,
            detail[:72],
            (28, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 235, 245),
            1,
            cv2.LINE_AA,
        )
    return frame


def set_latest_frame(frame) -> None:
    """อัปเดตรูปล่าสุดที่หน้าเว็บจะดึงไปแสดง."""
    global LATEST_JPEG
    jpeg = encode_frame(frame)
    if jpeg is None:
        return
    with STATE_LOCK:
        LATEST_JPEG = jpeg


def clear_inspection_state() -> None:
    """Invalidate a previous decision when no current inspection is available."""
    with STATE_LOCK:
        STATE.update({"decision": "NOT_INSPECTED", "review_required": False,
                      "confidence": 0.0, "anomaly_score": 0.0, "regions": [],
                      "brand": None, "brand_confidence": None, "fps": 0.0, "part_cycle": None, "frame_decision": "NOT_INSPECTED", "part_present": False})


def set_status(status: str, error: str = "") -> None:
    """อัปเดตสถานะให้หน้าเว็บรู้ว่าตอนนี้กำลังทำอะไร."""
    with STATE_LOCK:
        STATE["status"] = status
        STATE["error"] = error
        if status != "running":
            clear_inspection_state()


# ===== Utility ด้านชื่อไฟล์, path และ JSON =====
# input จาก browser ทุกจุดต้องผ่าน safe_name/safe_relative_parts ก่อนประกอบ Path
# เพื่อกันชื่อชน, path traversal และการเขียนไฟล์ออกนอก workspace
def timestamp_name() -> str:
    """สร้างชื่อรอบอัปโหลด/รอบ train ให้อ่านง่ายและไม่ชนกัน."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(text: str) -> str:
    """ทำชื่อไฟล์/โฟลเดอร์ให้ปลอดภัยก่อนเขียนลง disk."""
    keep = []
    for char in text.strip():
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "upload"


def safe_relative_parts(filename: str) -> list[str]:
    """แปลงชื่อไฟล์จาก browser/zip ให้เป็น relative path ที่ไม่หลุดโฟลเดอร์."""
    parts = []
    for part in Path(filename.replace("\\", "/")).parts:
        if part in {"", ".", ".."}:
            continue
        parts.append(safe_name(part))
    return parts or ["upload.bin"]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if is_image_file(item))


def count_yolov8_exports(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("data.yaml") if item.is_file())


# กลุ่มฟังก์ชัน Annotation Studio: แยก session/image/label ออกจาก dataset train จริง
def json_response_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so a crash cannot leave a half-written config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with JSON_WRITE_LOCK:
        try:
            temporary.write_text(text, encoding="utf-8")
            # Windows readers/scanners may briefly deny an atomic replacement.
            # Retry only sharing/access errors; never truncate the destination.
            for attempt in range(6):
                try:
                    os.replace(temporary, path)
                    break
                except OSError as error:
                    if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 5:
                        raise
                    time.sleep(0.05 * (2 ** attempt))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    """Write OpenCV images safely even when a Windows path contains non-ASCII text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"Could not encode image: {path.name}")
    encoded.tofile(str(path))


def read_image_file(path: Path) -> np.ndarray:
    """Read an image safely when its Windows path contains non-ASCII text."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or unreadable image: {path}")
    return image


def review_evidence_quality(source: np.ndarray, anomaly_score: float, threshold: float) -> float:
    """Score a frame so an event keeps its clearest, strongest evidence image."""
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 800.0)
    anomaly_strength = min(2.0, float(anomaly_score) / max(float(threshold), 1e-6)) / 2.0
    return round(sharpness * 0.55 + anomaly_strength * 0.45, 6)


def reference_review_scope(reference: dict | None) -> str:
    """Conservative scope: same dataset IDs and recipe metadata, not brand alone."""
    reference = reference or {}
    datasets = reference.get("recipe_datasets") or (reference.get("inspection_contract") or {}).get("datasets") or []
    if not datasets or any(not isinstance(item, dict) or not item.get("id") for item in datasets):
        return ""
    items = [{"id": str(item["id"]), "recipe": item.get("recipe") or {}} for item in datasets]
    items.sort(key=lambda item: json.dumps(item, sort_keys=True))
    area = checked_area(reference.get("inspection_area"))
    if area != FULL_AREA:
        items.append({"inspection_area": area})
    areas = checked_areas(reference.get("inspection_areas"), area)
    if areas != [area]:
        items.append({"inspection_areas": areas})
    if reference.get("alignment_reference"):
        items.append({"alignment_reference": reference["alignment_reference"]})
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_inspection_review(
    source: np.ndarray,
    annotated: np.ndarray,
    result: InspectionResult,
    *,
    object_box: tuple[int, int, int, int],
    object_label: str,
    object_confidence: float,
    source_name: str = "camera",
    existing_review_id: str = "",
    reference: dict | None = None,
) -> dict:
    """Create or update one physical inspection event for human review and CVAT."""
    visible_decision = "NG" if result.status == "NG" else "REVIEW"
    clean_existing_id = safe_name(existing_review_id) if existing_review_id else ""
    review_id = clean_existing_id or f"{visible_decision.lower()}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    review_dir = INSPECTION_REVIEW_DIR / review_id
    manifest = review_dir / "review.json"
    existing = read_json(manifest, {}) if clean_existing_id and manifest.is_file() else {}
    scope = reference_review_scope(reference)
    model_hash = ((reference or {}).get("inspection_contract") or {}).get("model_sha256", "")
    if existing and (existing.get("evaluation_scope", "") != scope or existing.get("model_sha256", "") != model_hash):
        return save_inspection_review(source, annotated, result, object_box=object_box,
            object_label=object_label, object_confidence=object_confidence,
            source_name=source_name, reference=reference)
    if existing and existing.get("status") != "pending":
        return existing

    source_file = review_dir / "source.jpg"
    annotated_file = review_dir / "preview.jpg"
    evidence_quality = review_evidence_quality(source, result.anomaly_score, result.threshold)
    previous_quality = float(existing.get("evidence_quality") or -1.0)
    replace_evidence = not existing or evidence_quality > previous_quality
    now_text = now_iso()
    frame_count = int(existing.get("event_frame_count") or 0) + 1
    record = {
        **existing,
        "id": review_id,
        "status": "pending",
        "decision": "NG" if visible_decision == "NG" or existing.get("decision") == "NG" else "REVIEW",
        "review_required": bool(result.review_required),
        "created_at": existing.get("created_at") or now_text,
        "last_seen_at": now_text,
        "event_frame_count": frame_count,
        "source_name": safe_name(source_name),
        "evaluation_scope": scope,
        "model_sha256": model_hash,
        "source_image": str(source_file.resolve()),
        "preview_image": str(annotated_file.resolve()),
        "normality_confidence": round(float(result.confidence), 6),
        "anomaly_score": round(float(result.anomaly_score), 6),
        "max_anomaly_score": round(max(float(existing.get("max_anomaly_score") or 0.0), float(result.anomaly_score)), 6),
        "threshold": round(float(result.threshold), 6),
        "object": {
            "label": object_label,
            "confidence": round(float(object_confidence), 6),
            "box": [int(value) for value in object_box],
        },
        "brand": result.brand,
        "brand_confidence": result.brand_confidence,
        "evidence_quality": max(previous_quality, evidence_quality),
        "regions": [
            {
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "score": round(float(region.score), 6),
            }
            for region in result.regions
        ],
    }
    if not replace_evidence:
        for key in ("review_required", "normality_confidence", "anomaly_score", "threshold",
                    "object", "brand", "brand_confidence", "regions", "source_name"):
            record[key] = existing.get(key)
    if replace_evidence:
        # A replaced frame without a heatmap must not retain an older heatmap.
        record.pop("heatmap_image", None)
        write_jpeg(source_file, source)
        write_jpeg(annotated_file, annotated)
        if result.anomaly_map:
            heatmap = np.asarray(result.anomaly_map, dtype=np.float32)
            if heatmap.ndim == 2 and heatmap.size:
                normalized = np.clip(heatmap / max(float(result.threshold), 1e-6), 0.0, 2.0) / 2.0
                colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
                colored = cv2.resize(colored, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_CUBIC)
                heatmap_file = review_dir / "heatmap.jpg"
                write_jpeg(heatmap_file, colored, quality=94)
                record["heatmap_image"] = str(heatmap_file.resolve())
    write_json(manifest, record)
    return record

def inspection_review_records(status: str = "") -> list[dict]:
    records: list[dict] = []
    if not INSPECTION_REVIEW_DIR.exists():
        return records
    for manifest in INSPECTION_REVIEW_DIR.glob("*/review.json"):
        try:
            record = read_json(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if status and record.get("status") != status:
            continue
        records.append(record)
    return sorted(records, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def normal_reference_training_fingerprint(paths: list[Path], settings: dict, *, backbone_path: str = "") -> str:
    """Identify the exact reviewed OK data and settings used by a Reference.

    The fingerprint prevents a button click from rebuilding an identical model
    and presenting that no-op as AI improvement. File content is hashed instead
    of timestamps so moving or touching a Dataset does not create a false change.
    """
    digest = hashlib.sha256()
    relevant_settings = {
        "inspection_size": int(settings["inspection_size"]),
        "min_good_images": int(settings["min_good_images"]),
        "min_region_ratio": float(settings["min_region_ratio"]),
        "pipeline_version": f"deep-patch-{DeepPatchReference.VERSION}",
        "inspection_area": checked_area(settings.get("inspection_area")),
        "inspection_areas": checked_areas(settings.get("inspection_areas"), settings.get("inspection_area")),
        "alignment_reference": settings.get("alignment_reference"),
    }
    digest.update(json.dumps(relevant_settings, sort_keys=True).encode("utf-8"))
    if backbone_path and Path(backbone_path).is_file():
        digest.update(file_sha256(Path(backbone_path)).encode("ascii"))
    content_hashes = []
    for path in paths:
        paired = file_sha256(path)
        parts = list(path.parts)
        image_index = next((i for i, part in enumerate(parts) if part.lower() == "images"), None)
        if image_index is not None:
            label = Path(*parts[:image_index], "labels", *parts[image_index + 1:]).with_suffix(".txt")
            if label.is_file():
                paired += ":" + file_sha256(label)
        content_hashes.append(paired)
    content_hashes.sort()
    digest.update(b"normal-reference-corpus-v2\0")
    for content_hash in content_hashes:
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_reference_on_reviewed_cases(inspector: NormalReferenceInspector, reference: dict | None = None) -> dict:
    """Evaluate with human-reviewed production cases kept outside OK training.

    Confirmed NG captures and false alarms are never added to the normal
    reference here. Missing or different scopes are excluded; this alone does
    not prove independence from images imported manually into training.
    """
    groups = {
        "confirmed_ng": inspection_review_records("confirmed_ng"),
        "false_alarm": inspection_review_records("false_alarm"),
    }
    counts = {"confirmed_ng": 0, "detected_ng": 0, "reviewed_ok": 0, "accepted_ok": 0, "false_ok": 0, "ng_review": 0, "ng_uninspectable": 0}
    scope = reference_review_scope(reference)
    excluded_scope = 0
    invalid_images = 0
    for expected, records in groups.items():
        for record in records:
            if not scope or record.get("evaluation_scope") != scope:
                excluded_scope += 1
                continue
            source_path = Path(str(record.get("source_image") or ""))
            raw_box = (record.get("object") or {}).get("box")
            if not source_path.is_file() or not isinstance(raw_box, list) or len(raw_box) != 4:
                invalid_images += 1
                continue
            try:
                result = inspector.inspect(read_image_file(source_path), object_box=tuple(raw_box))
            except (OSError, ValueError):
                invalid_images += 1
                continue
            if expected == "confirmed_ng":
                counts["confirmed_ng"] += 1
                counts["detected_ng"] += int(result.status == "NG" and not result.input_issue)
                counts["false_ok"] += int(result.status == "OK" and not result.review_required and not result.input_issue)
                counts["ng_uninspectable"] += int(bool(result.input_issue) or result.status == "NO_PART")
                counts["ng_review"] += int((result.review_required or result.status == "REVIEW") and not result.input_issue)
            else:
                counts["reviewed_ok"] += 1
                counts["accepted_ok"] += int(result.status == "OK" and not result.review_required and not result.input_issue)

    confirmed_ng = counts["confirmed_ng"]
    reviewed_ok = counts["reviewed_ok"]
    ng_recall = counts["detected_ng"] / confirmed_ng if confirmed_ng else None
    reviewed_ok_acceptance = counts["accepted_ok"] / reviewed_ok if reviewed_ok else None
    balanced_accuracy = (
        (ng_recall + reviewed_ok_acceptance) / 2.0
        if ng_recall is not None and reviewed_ok_acceptance is not None else None
    )
    return {
        "confirmed_ng_images": confirmed_ng,
        "detected_ng_images": counts["detected_ng"],
        "ng_recall": round(ng_recall, 4) if ng_recall is not None else None,
        "false_ok_rate": round(counts["false_ok"] / confirmed_ng, 4) if confirmed_ng else None,
        "false_ok_images": counts["false_ok"],
        "confirmed_ng_review_images": counts["ng_review"],
        "confirmed_ng_uninspectable_images": counts["ng_uninspectable"],
        "reviewed_ok_images": reviewed_ok,
        "accepted_reviewed_ok_images": counts["accepted_ok"],
        "reviewed_ok_acceptance_rate": (
            round(reviewed_ok_acceptance, 4) if reviewed_ok_acceptance is not None else None
        ),
        "balanced_accuracy": round(balanced_accuracy, 4) if balanced_accuracy is not None else None,
        "evaluation_scope": scope,
        "excluded_review_scope_images": excluded_scope,
        "invalid_review_images": invalid_images,
        "evaluation_note": "Human-reviewed captures from the same saved dataset scope; historical unscoped captures excluded. Training overlap has not been verified.",
    }


def evaluate_reference_holdout(inspector, manifest: dict) -> dict:
    """Score one immutable set of images and ROIs for model comparison."""
    entries = manifest.get("validation") or []
    if not entries:
        raise ValueError("Candidate has no recorded validation images")
    counts = {"OK": 0, "REVIEW": 0, "NG": 0, "UNINSPECTABLE": 0}
    signature = hashlib.sha256()
    for entry in entries:
        path = Path(entry["path"])
        actual_hash = file_sha256(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Validation image changed during training: {path.name}")
        box = tuple(entry["box"])
        signature.update(json.dumps([actual_hash, box], separators=(",", ":")).encode())
        result = inspector.inspect(read_image_file(path), object_box=box)
        decision = ("UNINSPECTABLE" if result.input_issue or result.status == "NO_PART" else
                    "REVIEW" if result.review_required or result.status == "REVIEW" else result.status)
        if decision not in counts:
            raise ValueError(f"Unsupported inspection decision: {decision}")
        counts[decision] += 1
    total = len(entries)
    return {"validation_images": total,
            "independent_validation": bool(manifest.get("independent_validation")),
            "normal_acceptance_rate": round(counts["OK"] / total, 4),
            "strict_ok_rate": round(counts["OK"] / total, 4),
            "review_rate": round(counts["REVIEW"] / total, 4),
            "false_ng_rate": round(counts["NG"] / total, 4),
            "uninspectable_rate": round(counts["UNINSPECTABLE"] / total, 4),
            "holdout_decisions": counts, "comparison_set_sha256": signature.hexdigest()}


def current_reference_quality_metrics(reference: dict, model_path: Path | None) -> dict:
    """Merge saved OK holdout metrics with the latest human review evaluation."""
    quality = dict(reference.get("quality_metrics") or {})
    if model_path is None or not model_path.is_file():
        return quality
    manifests = sorted(INSPECTION_REVIEW_DIR.glob("*/review.json"))
    review_signature = ":".join(
        f"{path.parent.name}:{path.stat().st_mtime_ns}" for path in manifests
    )
    cache_key = f"reference-review-eval:{model_path}:{model_path.stat().st_mtime_ns}:{reference_review_scope(reference)}:{review_signature}"

    def build() -> dict:
        inspector = load_inspection_reference(model_path)
        return evaluate_reference_on_reviewed_cases(inspector, reference)

    operational = cached_analysis(cache_key, build, ttl_seconds=60.0)
    quality.update(operational)
    confirmed_ng = int(operational.get("confirmed_ng_images") or 0)
    quality["defect_metrics_available"] = confirmed_ng > 0
    quality["defect_metrics_reason"] = (
        "Measured on human-confirmed NG production captures."
        if confirmed_ng else "No human-confirmed NG evaluation images are available yet."
    )
    return quality


def compare_reference_candidate(candidate: dict, incumbent: dict | None = None) -> dict:
    """Record basic model checks. These checks never authorize activation."""
    incumbent = dict(incumbent or {})
    checks: list[dict] = []

    def add(name: str, passed: bool, candidate_value, incumbent_value=None, detail: str = "") -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "candidate": candidate_value,
            "incumbent": incumbent_value,
            "detail": detail,
        })

    add("independent_holdout_required",
        candidate.get("independent_validation") is True and int(candidate.get("validation_images") or 0) > 0,
        candidate.get("independent_validation"), True,
        "Training-set scores cannot authorize automatic model activation.")
    if "strict_ok_rate" in candidate:
        add("clear_ok_acceptance_minimum", float(candidate["strict_ok_rate"]) >= 0.90,
            candidate["strict_ok_rate"], 0.90,
            "REVIEW is not a clean OK decision.")
    if candidate.get("comparison_set_sha256") and incumbent:
        add("same_comparison_images_and_roi",
            candidate["comparison_set_sha256"] == incumbent.get("comparison_set_sha256"),
            candidate["comparison_set_sha256"], incumbent.get("comparison_set_sha256"))
    candidate_acceptance = float(candidate.get("normal_acceptance_rate") or 0.0)
    incumbent_acceptance = incumbent.get("normal_acceptance_rate")
    add("known_good_acceptance_minimum", candidate_acceptance >= 0.90, candidate_acceptance, 0.90)
    if incumbent_acceptance is not None:
        incumbent_acceptance = float(incumbent_acceptance)
        add(
            "known_good_not_regressed",
            candidate_acceptance + 0.03 >= incumbent_acceptance,
            candidate_acceptance,
            incumbent_acceptance,
            "Candidate may be at most 3 percentage points lower on Known-good holdout.",
        )

    confirmed_ng = int(candidate.get("confirmed_ng_images") or 0)
    reviewed_ok = int(candidate.get("reviewed_ok_images") or 0)
    comparable_ng = confirmed_ng > 0 and incumbent.get("ng_recall") is not None
    comparable_ok = reviewed_ok > 0 and incumbent.get("reviewed_ok_acceptance_rate") is not None
    if comparable_ng:
        candidate_recall = float(candidate.get("ng_recall") or 0.0)
        incumbent_recall = float(incumbent.get("ng_recall") or 0.0)
        add("ng_recall_not_regressed", candidate_recall + 0.02 >= incumbent_recall, candidate_recall, incumbent_recall)
    if comparable_ok:
        candidate_reviewed_ok = float(candidate.get("reviewed_ok_acceptance_rate") or 0.0)
        incumbent_reviewed_ok = float(incumbent.get("reviewed_ok_acceptance_rate") or 0.0)
        add(
            "reviewed_ok_not_regressed",
            candidate_reviewed_ok + 0.02 >= incumbent_reviewed_ok,
            candidate_reviewed_ok,
            incumbent_reviewed_ok,
        )

    checks_passed = all(check["passed"] for check in checks)
    decision = "needs_test" if checks_passed else "rejected"
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "promote": False,
        "checks_passed": checks_passed,
        "decision": decision,
        "provisional": True,
        "checks": checks,
        "failed_checks": failed,
        "candidate_metrics": candidate,
        "incumbent_metrics": incumbent,
        "reason": (
            "Basic checks passed. Open Jobs to test this model. The active model has not changed."
            if checks_passed
            else f"This model needs more work. Failed checks: {', '.join(failed)}. The active model has not changed."
        ),
    }


def json_file_is_valid(path: Path) -> bool:
    try:
        return isinstance(read_json(path), dict)
    except Exception:
        return False


def clear_analysis_cache() -> None:
    with ANALYSIS_CACHE_LOCK:
        ANALYSIS_CACHE.clear()


def cached_analysis(key: str, builder, ttl_seconds: float = 300.0):
    now = time.monotonic()
    with ANALYSIS_CACHE_LOCK:
        cached = ANALYSIS_CACHE.get(key)
        if cached and now - cached[0] <= ttl_seconds:
            return cached[1]
    value = builder()
    with ANALYSIS_CACHE_LOCK:
        ANALYSIS_CACHE[key] = (time.monotonic(), value)
    return value


def parse_class_names_text(text: str | list | tuple | None) -> list[str]:
    """Normalize user-entered class names while keeping order stable."""
    raw_items: list[str] = []
    if isinstance(text, (list, tuple)):
        raw_items = [str(item or "") for item in text]
    else:
        raw_items = re.split(r"[,;\n]+", str(text or ""))
    seen = set()
    names = []
    for item in raw_items:
        name = str(item or "").strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name[:80])
    return names


def normalized_class_key(name: str) -> str:
    return re.sub(r"[\s_-]+", "_", str(name or "").strip().lower()).strip("_")


def normalize_class_aliases(value: object, target_classes: list[str]) -> dict[str, list[str]]:
    """Keep aliases deterministic and prevent one alias from targeting multiple classes."""
    if not isinstance(value, dict):
        return {}
    target_lookup = {normalized_class_key(name): name for name in target_classes}
    claimed = set(target_lookup)
    aliases: dict[str, list[str]] = {}
    for raw_target, raw_aliases in value.items():
        target = target_lookup.get(normalized_class_key(raw_target))
        if not target:
            continue
        items = raw_aliases if isinstance(raw_aliases, (list, tuple)) else re.split(r"[,;\n]+", str(raw_aliases or ""))
        clean_items = []
        for raw_alias in items:
            alias = str(raw_alias or "").strip()[:80]
            key = normalized_class_key(alias)
            if not alias or key in claimed:
                continue
            claimed.add(key)
            clean_items.append(alias)
        if clean_items:
            aliases[target] = clean_items
    return aliases


def normalize_class_display_names(value: object, target_classes: list[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    incoming = {normalized_class_key(key): str(display or "").strip()[:80] for key, display in value.items()}
    return {
        target: incoming[normalized_class_key(target)]
        for target in target_classes
        if incoming.get(normalized_class_key(target))
    }


def project_class_lookup(config: dict) -> dict[str, str]:
    """Map dataset/model names to the project's canonical class names."""
    targets = list(config.get("target_classes") or [])
    lookup = {normalized_class_key(name): name for name in targets}
    for canonical, aliases in (config.get("class_aliases") or {}).items():
        canonical_name = lookup.get(normalized_class_key(canonical))
        if not canonical_name:
            continue
        for alias in aliases:
            lookup[normalized_class_key(alias)] = canonical_name
    # Preserve compatibility with historical Airbag datasets.
    airbag_target = lookup.get("airbag")
    if airbag_target:
        for alias in ("airbagv2", "airbag_v2", "airbag-v2"):
            lookup[normalized_class_key(alias)] = airbag_target
    return lookup


def normalize_project_config(payload: dict) -> dict:
    """Single source of truth for project-level setup used by Training, Edit, and detection."""
    try:
        existing = read_json(PROJECT_CONFIG_PATH, DEFAULT_PROJECT_CONFIG)
    except Exception:
        backup_path = PROJECT_CONFIG_PATH.with_suffix(".json.bak")
        try:
            existing = read_json(backup_path, DEFAULT_PROJECT_CONFIG)
        except Exception:
            existing = dict(DEFAULT_PROJECT_CONFIG)
    config = {**DEFAULT_PROJECT_CONFIG, **(existing if isinstance(existing, dict) else {})}
    incoming = payload if isinstance(payload, dict) else {}

    # The public decision remains OK/NG, while the proven detector and brand
    # models stay available as supporting modules.  Do not erase those models
    # during migration: the detector supplies the part ROI and can contribute a
    # strong NG signal for historical Mark/Stain classes.
    legacy_targets = {normalized_class_key(name) for name in config.get("target_classes") or []}
    if not ({"target_classes", "target_classes_text"} & incoming.keys()) and legacy_targets in ({"airbag", "mark"}, {"object", "defect"}):
        config["target_classes"] = ["OK", "NG"]
        config["primary_class"] = "OK"
        config["class_aliases"] = {}
        config["class_display_names"] = {}
    config["detection_mode"] = "normal_reference"
    config["inspection_strategy"] = "normal_reference"

    for field in ("project_name", "brand", "model", "primary_class", "last_train_type", "last_train_log", "last_train_status"):
        if field in incoming:
            config[field] = str(incoming.get(field) or "").strip()[:240]
    # The decision strategy is fixed, but supporting detector/brand modules are
    # intentionally preserved.
    config["detection_mode"] = "normal_reference"
    config["inspection_strategy"] = "normal_reference"
    for field, default, minimum, maximum in (
        ("primary_confidence", 0.30, 0.01, 0.99),
        ("secondary_confidence", 0.40, 0.01, 0.99),
    ):
        if field in incoming:
            try:
                config[field] = max(minimum, min(maximum, float(incoming[field])))
            except (TypeError, ValueError):
                config[field] = default
    if "camera_index" in incoming:
        try:
            config["camera_index"] = max(0, min(10, int(incoming["camera_index"])))
        except (TypeError, ValueError):
            config["camera_index"] = detector.DEFAULT_CAMERA_INDEX
    if "use_classifier" in incoming:
        config["use_classifier"] = bool(incoming.get("use_classifier"))

    # OK/NG is the stable decision contract. Brand identification is an
    # independent optional module and is not represented as an NG class.
    config["target_classes"] = ["OK", "NG"]
    config["primary_class"] = "OK"

    if not config.get("project_name"):
        config["project_name"] = DEFAULT_PROJECT_CONFIG["project_name"]
    config["target_classes"] = ["OK", "NG"]
    config["primary_class"] = "OK"

    alias_source = incoming.get("class_aliases", config.get("class_aliases"))
    display_source = incoming.get("class_display_names", config.get("class_display_names"))
    config["class_aliases"] = normalize_class_aliases(alias_source, config["target_classes"])
    config["class_display_names"] = normalize_class_display_names(display_source, config["target_classes"])

    for field in ("active_detector_model", "active_locator_model", "previous_detector_model", "active_brand_model"):
        if field in incoming:
            raw_path = str(incoming.get(field) or "").strip()
            config[field] = str(Path(raw_path).resolve()) if raw_path else ""
    if "last_model_validation" in incoming and isinstance(incoming.get("last_model_validation"), dict):
        config["last_model_validation"] = incoming["last_model_validation"]
    reference_source = incoming.get("normal_reference", config.get("normal_reference"))
    reference_defaults = dict(DEFAULT_PROJECT_CONFIG["normal_reference"])
    if isinstance(reference_source, dict):
        reference_defaults.update(reference_source)
    reference_defaults["inspection_size"] = max(160, min(1280, int(reference_defaults.get("inspection_size") or 512)))
    reference_defaults["min_good_images"] = max(3, min(1000, int(reference_defaults.get("min_good_images") or 5)))
    reference_defaults["min_region_ratio"] = max(0.0001, min(0.25, float(reference_defaults.get("min_region_ratio") or 0.002)))
    config["normal_reference"] = reference_defaults
    module_source = incoming.get("inspection_modules", config.get("inspection_modules"))
    module_defaults = dict(DEFAULT_PROJECT_CONFIG["inspection_modules"])
    if isinstance(module_source, dict):
        for key in module_defaults:
            if key in module_source:
                module_defaults[key] = bool(module_source[key])
    module_defaults["normal_reference"] = True
    module_defaults["part_locator"] = True
    config["inspection_modules"] = module_defaults
    # Mark/Stain classes from historical detectors are shadow telemetry only.
    # They must never promote an otherwise normal Good-only result to NG.
    config["use_legacy_defect_signal"] = False
    config["inspection_modules"]["legacy_defect_assist"] = False
    config["inspection_modules"]["brand_classifier"] = bool(config.get("use_classifier", True))
    augmentation_source = incoming.get("augmentation_profile", config.get("augmentation_profile"))
    config["augmentation_profile"] = normalize_augmentation_profile(augmentation_source)

    config["target_classes_text"] = ", ".join(config["target_classes"])
    # Detector runs remain historical training records, but their active model
    # is still the locator/defect-assist module used by OK-first inspection.
    if not str(config.get("active_locator_model") or "").strip():
        config["active_locator_model"] = str(config.get("active_detector_model") or "")
    config["updated_at"] = now_iso()
    if not config.get("created_at"):
        config["created_at"] = config["updated_at"]
    return config


def read_project_config() -> dict:
    config = normalize_project_config({})
    detector.configure_detection_target(
        config.get("primary_class"),
        config.get("target_classes"),
        config.get("class_display_names"),
        config.get("detection_mode"),
    )
    if not PROJECT_CONFIG_PATH.exists():
        write_json(PROJECT_CONFIG_PATH, config)
    return config


def write_project_config(payload: dict) -> dict:
    config = normalize_project_config(payload)
    if PROJECT_CONFIG_PATH.exists() and json_file_is_valid(PROJECT_CONFIG_PATH):
        shutil.copy2(PROJECT_CONFIG_PATH, PROJECT_CONFIG_PATH.with_suffix(".json.bak"))
    write_json(PROJECT_CONFIG_PATH, config)
    detector.configure_detection_target(
        config.get("primary_class"),
        config.get("target_classes"),
        config.get("class_display_names"),
        config.get("detection_mode"),
    )
    clear_analysis_cache()
    return config


def training_is_running() -> bool:
    with TRAIN_LOCK:
        refresh_train_state_locked()
        process_running = TRAIN_PROCESS is not None
    with REFERENCE_TRAIN_LOCK:
        reference_running = bool(REFERENCE_TRAIN_STATE.get("running"))
    return process_running or reference_running


def dataset_mutation_blocked_response() -> JSONResponse | None:
    if training_is_running() or ("vision_service" in globals() and vision_service.busy()):
        return JSONResponse(
            {
                "ok": False,
                "error": "Dataset is locked while training is running. Stop training before adding, editing, deleting, or restoring data.",
            },
            status_code=409,
        )
    return None


def latest_detector_model_path() -> Path | None:
    config = read_project_config()
    configured = str(config.get("active_detector_model") or "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path
    candidates = sorted(MODEL_DIR.glob("*_best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def active_locator_model_path(config: dict | None = None) -> Path | None:
    config = config or read_project_config()
    configured = str(config.get("active_locator_model") or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    return latest_detector_model_path()


def normal_reference_runtime_status(config: dict | None = None) -> dict:
    """Return the authoritative readiness state for generic OK/NG inspection."""
    config = config or read_project_config()
    reference = dict(config.get("normal_reference") or {})
    raw_model = str(reference.get("active_model") or "").strip()
    model_path = Path(raw_model) if raw_model else None
    ready = bool(model_path and model_path.is_file())
    model_version = 0
    engine = str(reference.get("anomaly_engine") or "statistical_reference")
    inspection_size = int(reference.get("inspection_size") or 512)
    dependency_error = ""
    if ready and model_path:
        try:
            stat = model_path.stat()
            def inspect_metadata():
                loaded = load_inspection_reference(model_path)
                deep = isinstance(loaded, DeepPatchReference)
                return {"version": loaded.model_version, "deep": deep,
                        "size": loaded.input_size if deep else loaded.size[0],
                        "backbone": loaded.backbone_path if deep else ""}
            # Cache scalar metadata, not the large memory bank or mutable YOLO.
            metadata = cached_analysis(f"reference-metadata:{model_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}", inspect_metadata)
            model_version = metadata["version"]
            inspection_size = metadata["size"]
            if metadata["deep"]:
                engine = "deep_patch"
                ready = Path(metadata["backbone"]).is_file()
                if not ready:
                    dependency_error = "Reference backbone is missing; restore the matching Part Locator model."
            else:
                ready = model_version >= NormalReferenceInspector.VERSION
        except Exception:
            ready = False
    reason = ""
    if engine != "deep_patch" and model_path and model_path.is_file() and model_version < NormalReferenceInspector.VERSION:
        reason = "Normal Reference รุ่นเดิมยังไม่แยก ROI/พื้นหลัง กรุณาสร้าง Reference ใหม่จาก Known-good Dataset"
    elif not ready:
        reason = (
            "ยังไม่มี Normal Reference ที่พร้อมใช้งาน: เลือก Dataset ของชิ้นงานปกติ "
            "ตรวจยืนยันว่าเป็น Known-good แล้วสร้าง Reference ก่อนเปิดกล้อง"
        )
    return {
        "inspection_size": inspection_size,
        "strategy": "normal_reference",
        "ready": ready,
        "model": str(model_path.resolve()) if ready and model_path else "",
        "source_dataset": str(reference.get("source_dataset") or "").strip(),
        "training_images": int(reference.get("training_images") or 0),
        "trained_at": str(reference.get("trained_at") or ""),
        "pipeline_version": model_version,
        "anomaly_engine": engine,
        "reason": dependency_error or reason,
    }


def clear_shared_model_cache() -> None:
    global SHARED_MODELS, SHARED_MODEL_ERROR, SHARED_LOCATOR, SHARED_LOCATOR_KEY
    with MODEL_LOCK:
        SHARED_MODELS = None
        SHARED_MODEL_ERROR = ""
        SHARED_LOCATOR = None
        SHARED_LOCATOR_KEY = ""


# สร้าง session ใหม่พร้อม manifest เพื่อให้ย้อนกลับมาแก้ label ได้
# Annotation Studio implementation removed. CVAT owns annotation and annotation editing.

def copytree_unique(source: Path, destination_root: Path, name: str) -> Path:
    """copy โฟลเดอร์ไปปลายทางโดยเติมเลขถ้าชื่อซ้ำ."""
    destination_root.mkdir(parents=True, exist_ok=True)
    base = destination_root / safe_name(name)
    destination = base
    index = 2
    while destination.exists():
        destination = destination_root / f"{base.name}_{index}"
        index += 1
    shutil.copytree(source, destination)
    return destination


def extract_zip_safely(zip_path: Path, destination: Path) -> int:
    """แตก zip โดยกัน path traversal เช่น ../../file."""
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            raise ValueError(f"ZIP contains too many files ({len(members):,}).")
        total_size = sum(max(0, member.file_size) for member in members)
        if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP expands beyond the allowed size.")
        for member in members:
            if member.is_dir():
                continue
            parts = safe_relative_parts(member.filename)
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    return extracted


async def save_uploads_to_staging(files: list[UploadFile], category: str) -> tuple[Path, int]:
    """รับไฟล์จากเว็บแล้วเก็บลง staging ก่อนคัดแยกไปโฟลเดอร์ train จริง."""
    if not files:
        raise ValueError("No files uploaded.")

    staging_dir = UPLOAD_STAGING_DIR / f"{timestamp_name()}_{uuid.uuid4().hex}_{safe_name(category)}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    total_bytes = 0

    for upload in files:
        original_name = upload.filename or "upload.bin"
        parts = safe_relative_parts(original_name)
        destination = staging_dir.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)

        file_bytes = 0
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                file_bytes += len(chunk)
                total_bytes += len(chunk)
                if file_bytes > MAX_UPLOAD_FILE_BYTES or total_bytes > MAX_UPLOAD_TOTAL_BYTES:
                    raise ValueError("Upload exceeds the allowed size.")
                output.write(chunk)
        saved_count += 1

        if destination.suffix.lower() == ".zip":
            extract_dir = staging_dir / f"extracted_{safe_name(destination.stem)}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            saved_count += extract_zip_safely(destination, extract_dir)

    return staging_dir, saved_count


# ===== Auto Split สำหรับ YOLO Dataset ที่โปรแกรมตีกรอบไม่ได้แบ่งชุดมาให้ =====
# กติกาสำคัญ: ถ้ามี train/val ที่ใช้งานได้อยู่แล้วต้องรักษาของเดิม, ถ้ายังไม่มีจึงคัดลอก
# รูป+label ไปสร้าง export ใหม่แบบ deterministic และไม่แก้ไฟล์ที่ผู้ใช้อัปโหลดเข้ามา
def uploaded_yolo_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    """คืน images/labels ของ split มาตรฐาน โดยรองรับทั้ง valid และ val."""
    aliases = ("valid", "val") if split == "valid" else (split,)
    for alias in aliases:
        candidates = (
            (dataset_root / alias / "images", dataset_root / alias / "labels"),
            (dataset_root / "images" / alias, dataset_root / "labels" / alias),
        )
        for images_dir, labels_dir in candidates:
            if images_dir.exists() or labels_dir.exists():
                return images_dir, labels_dir
    alias = aliases[0]
    return dataset_root / alias / "images", dataset_root / alias / "labels"


def yaml_dataset_path(dataset_root: Path, key: str) -> Path | None:
    """อ่าน path ของ split จาก data.yaml โดยไม่เชื่อ path ที่หลุดออกนอก dataset root."""
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return None
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        raw_path = str(data.get(key) or "").strip()
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        resolved = candidate.resolve()
        root = dataset_root.resolve()
        if resolved != root and root not in resolved.parents:
            return None
        return resolved
    except Exception:
        return None


def dataset_has_usable_split(dataset_root: Path) -> bool:
    """ถือว่าแบ่งแล้วเมื่อมีรูปทั้ง Train และ Validation; Test เป็นส่วนเสริมจึงไม่บังคับ."""
    train_path = yaml_dataset_path(dataset_root, "train")
    valid_path = yaml_dataset_path(dataset_root, "val")
    if train_path is None:
        train_path, _ = uploaded_yolo_split_dirs(dataset_root, "train")
    if valid_path is None:
        valid_path, _ = uploaded_yolo_split_dirs(dataset_root, "valid")
    return count_images(train_path) > 0 and count_images(valid_path) > 0


def read_uploaded_class_names(staging_dir: Path, fallback_names: list[str]) -> list[str]:
    """อ่านชื่อ class จาก data.yaml/classes.txt ก่อนใช้ Project config เป็น fallback."""
    discovered: list[list[str]] = []
    for data_yaml in sorted(staging_dir.rglob("data.yaml")):
        try:
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = data.get("names", [])
        if isinstance(names, dict):
            ordered = []
            for key, value in names.items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    index = len(ordered)
                ordered.append((index, str(value).strip()))
            values = [value for _, value in sorted(ordered) if value]
        elif isinstance(names, list):
            values = [str(value).strip() for value in names if str(value).strip()]
        else:
            values = []
        if values:
            discovered.append(values)

    for class_file in sorted(
        path for path in staging_dir.rglob("*") if path.is_file() and path.name.lower() in YOLO_CLASS_NAME_FILES
    ):
        try:
            values = [line.strip() for line in class_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        except Exception:
            continue
        if values:
            discovered.append(values)

    unique = []
    for names in discovered:
        if names not in unique:
            unique.append(names)
    if len(unique) > 1:
        raise ValueError("Uploaded dataset contains conflicting class-name lists. Keep one data.yaml/classes.txt contract.")
    if unique:
        return unique[0]
    return [str(name).strip() for name in fallback_names if str(name).strip()]


def structural_label_candidates(image_path: Path, staging_dir: Path) -> list[Path]:
    """สร้างตำแหน่ง label ที่เป็นไปได้จากโครงสร้าง images/labels และไฟล์ที่อยู่คู่กัน."""
    candidates = [image_path.with_suffix(".txt")]
    try:
        relative = image_path.relative_to(staging_dir)
    except ValueError:
        relative = Path(image_path.name)
    parts = list(relative.parts)
    for index, part in enumerate(parts):
        if part.lower() != "images":
            continue
        label_parts = parts.copy()
        label_parts[index] = "labels"
        candidates.append(staging_dir.joinpath(*label_parts).with_suffix(".txt"))
    return candidates


def parse_uploaded_yolo_label(label_path: Path) -> set[int]:
    """ตรวจ syntax ขั้นต้นและคืน class IDs; label ว่างถือเป็น negative image ที่ถูกต้อง."""
    class_ids: set[int] = set()
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        raise ValueError(f"Cannot read label file: {label_path.name}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5 and len(parts) < 7:
            raise ValueError(f"Invalid YOLO label at {label_path.name}:{line_number}")
        try:
            class_value = float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric YOLO label at {label_path.name}:{line_number}") from exc
        class_id = int(class_value)
        if class_value != class_id or class_id < 0:
            raise ValueError(f"Invalid class ID at {label_path.name}:{line_number}")
        if len(parts) >= 7 and len(coordinates) % 2 != 0:
            raise ValueError(f"Invalid polygon coordinates at {label_path.name}:{line_number}")
        class_ids.add(class_id)
    return class_ids


def uploaded_pair_fingerprint(image_path: Path, label_path: Path) -> str:
    """Hash รูปและ label แบบอ่านทีละก้อน เพื่อไม่ดึงไฟล์ภาพขนาดใหญ่ทั้งหมดเข้า RAM."""
    digest = hashlib.sha256()
    for path in (image_path, label_path):
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def discover_unsplit_yolo_pairs(staging_dir: Path, excluded_roots: list[Path]) -> tuple[list[dict], list[str]]:
    """จับคู่รูปกับ label จาก Dataset ที่ยังไม่แบ่ง พร้อมแจ้งรูป/label ที่จับคู่ไม่ได้."""
    excluded = [path.resolve() for path in excluded_roots]

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in excluded)

    images = sorted(
        path for path in staging_dir.rglob("*")
        if is_image_file(path) and not is_excluded(path)
    )
    label_files = sorted(
        path for path in staging_dir.rglob("*.txt")
        if path.name.lower() not in YOLO_CLASS_NAME_FILES and not is_excluded(path)
    )
    labels_by_stem: dict[str, list[Path]] = {}
    for label_path in label_files:
        labels_by_stem.setdefault(label_path.stem.lower(), []).append(label_path)

    pairs: list[dict] = []
    used_labels: set[Path] = set()
    missing_labels: list[str] = []
    duplicate_pairs = 0
    seen_fingerprints: set[str] = set()
    for image_path in images:
        label_path = next((path for path in structural_label_candidates(image_path, staging_dir) if path.is_file()), None)
        if label_path is None:
            same_stem = labels_by_stem.get(image_path.stem.lower(), [])
            if len(same_stem) == 1:
                label_path = same_stem[0]
        if label_path is None:
            missing_labels.append(str(image_path.relative_to(staging_dir)))
            continue

        label_path = label_path.resolve()
        class_ids = parse_uploaded_yolo_label(label_path)
        fingerprint = uploaded_pair_fingerprint(image_path, label_path)
        if fingerprint in seen_fingerprints:
            duplicate_pairs += 1
            used_labels.add(label_path)
            continue
        seen_fingerprints.add(fingerprint)
        used_labels.add(label_path)
        relative = str(image_path.relative_to(staging_dir)).replace("\\", "/")
        pairs.append(
            {
                "image": image_path,
                "label": label_path,
                "relative": relative,
                "classes": class_ids,
                "fingerprint": fingerprint,
            }
        )

    # รายงานเฉพาะ orphan ที่อยู่ในโฟลเดอร์ labels เพื่อไม่ตีความ README.txt เป็น YOLO label
    orphan_labels = [
        str(path.relative_to(staging_dir))
        for path in label_files
        if path.resolve() not in used_labels and "labels" in {part.lower() for part in path.parts}
    ]
    errors = []
    if label_files and missing_labels:
        sample = ", ".join(missing_labels[:3])
        errors.append(f"{len(missing_labels)} image(s) have no matching label: {sample}")
    if orphan_labels:
        sample = ", ".join(orphan_labels[:3])
        errors.append(f"{len(orphan_labels)} label(s) have no matching image: {sample}")
    if errors:
        raise ValueError("; ".join(errors))
    notes = [f"Skipped {duplicate_pairs} exact duplicate pair(s)."] if duplicate_pairs else []
    return pairs, notes


def auto_split_capacities(total: int) -> dict[str, int]:
    """กำหนดจำนวนแต่ละ split โดยรักษา Validation อย่างน้อยหนึ่งรูปเมื่อมีข้อมูลพอ."""
    if total <= 1:
        return {"train": total, "valid": 0, "test": 0}
    if total < 10:
        return {"train": total - 1, "valid": 1, "test": 0}
    valid = max(1, round(total * AUTO_SPLIT_RATIOS["valid"]))
    test = max(1, round(total * AUTO_SPLIT_RATIOS["test"]))
    if valid + test >= total:
        test = 0
        valid = 1
    return {"train": total - valid - test, "valid": valid, "test": test}


def split_yolo_pairs_class_aware(pairs: list[dict]) -> dict[str, list[dict]]:
    """แบ่งแบบ deterministic โดยพยายามกระจาย class และจำนวนรูปให้ใกล้เป้าหมาย."""
    capacities = auto_split_capacities(len(pairs))
    split_names = ("train", "valid", "test")
    class_totals: dict[int, int] = {}
    for pair in pairs:
        for class_id in pair["classes"]:
            class_totals[class_id] = class_totals.get(class_id, 0) + 1

    targets = {
        split: {
            class_id: round(total * capacities[split] / max(1, len(pairs)))
            for class_id, total in class_totals.items()
        }
        for split in split_names
    }
    current = {split: {class_id: 0 for class_id in class_totals} for split in split_names}
    result: dict[str, list[dict]] = {split: [] for split in split_names}
    ordered = sorted(
        pairs,
        key=lambda pair: (
            -sum(1 / max(1, class_totals[class_id]) for class_id in pair["classes"]),
            hashlib.sha256(f"{AUTO_SPLIT_SEED}:{pair['relative']}".encode("utf-8")).hexdigest(),
        ),
    )

    for pair in ordered:
        available = [split for split in split_names if len(result[split]) < capacities[split]]
        if not available:
            break
        missing_from_train = any(current["train"].get(class_id, 0) == 0 for class_id in pair["classes"])
        if missing_from_train and "train" in available:
            chosen = "train"
        else:
            def score(split: str) -> tuple[float, str]:
                class_need = sum(
                    (targets[split].get(class_id, 0) - current[split].get(class_id, 0))
                    / max(1, targets[split].get(class_id, 0))
                    for class_id in pair["classes"]
                )
                capacity_need = (capacities[split] - len(result[split])) / max(1, capacities[split])
                tie = hashlib.sha256(f"{AUTO_SPLIT_SEED}:{pair['relative']}:{split}".encode("utf-8")).hexdigest()
                return class_need + capacity_need * 0.25, tie

            chosen = max(available, key=score)
        result[chosen].append(pair)
        for class_id in pair["classes"]:
            current[chosen][class_id] = current[chosen].get(class_id, 0) + 1
    return result


def write_auto_split_data_yaml(export_dir: Path, class_names: list[str], has_test: bool) -> None:
    """เขียน data.yaml แบบ absolute path ให้ trainer หา split ได้จากทุก working directory."""
    lines = [
        f"train: {export_dir / 'train' / 'images'}",
        f"val: {export_dir / 'valid' / 'images'}",
    ]
    if has_test:
        lines.append(f"test: {export_dir / 'test' / 'images'}")
    names_text = "[" + ", ".join(json.dumps(name, ensure_ascii=False) for name in class_names) + "]"
    lines.extend(["", f"nc: {len(class_names)}", f"names: {names_text}", ""])
    (export_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def create_auto_split_export(
    pairs: list[dict],
    destination_root: Path,
    class_names: list[str],
    notes: list[str] | None = None,
) -> dict:
    """คัดลอกคู่รูป/label ไป export ใหม่และบันทึกหลักฐานการแบ่งสำหรับตรวจย้อนหลัง."""
    if not pairs:
        raise ValueError("No image/label pairs were found for Auto Split.")
    if len(pairs) < 2:
        raise ValueError("Auto Split needs at least 2 labeled images so Train and Validation are not the same image.")
    max_class_id = max((class_id for pair in pairs for class_id in pair["classes"]), default=0)
    if not class_names:
        if max_class_id == 0:
            class_names = ["Object"]
        else:
            raise ValueError("Class names are missing. Add data.yaml/classes.txt or set Project detection classes first.")
    if max_class_id >= len(class_names):
        raise ValueError(
            f"Label class ID {max_class_id} exceeds the {len(class_names)} configured class name(s)."
        )

    export_dir = unique_child_path(destination_root, f"{timestamp_name()}_auto_split")
    export_dir.mkdir(parents=True, exist_ok=True)
    split_map = split_yolo_pairs_class_aware(pairs)
    manifest_items = []
    class_counts = {split: {name: 0 for name in class_names} for split in split_map}
    for split, split_pairs in split_map.items():
        images_dir = export_dir / split / "images"
        labels_dir = export_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index, pair in enumerate(split_pairs, start=1):
            stem = safe_name(Path(pair["image"]).stem)
            unique_suffix = pair["fingerprint"][:10]
            image_name = f"{index:06d}_{stem}_{unique_suffix}{Path(pair['image']).suffix.lower()}"
            label_name = f"{Path(image_name).stem}.txt"
            shutil.copy2(pair["image"], images_dir / image_name)
            shutil.copy2(pair["label"], labels_dir / label_name)
            for class_id in pair["classes"]:
                class_counts[split][class_names[class_id]] += 1
            manifest_items.append(
                {
                    "source": pair["relative"],
                    "split": split,
                    "image": image_name,
                    "classes": sorted(pair["classes"]),
                }
            )

    counts = {split: len(items) for split, items in split_map.items()}
    write_auto_split_data_yaml(export_dir, class_names, counts["test"] > 0)
    manifest = {
        "version": 1,
        "mode": "automatic_class_aware_split",
        "created_at": now_iso(),
        "seed": AUTO_SPLIT_SEED,
        "requested_ratios": AUTO_SPLIT_RATIOS,
        "counts": counts,
        "class_names": class_names,
        "class_counts": class_counts,
        "notes": notes or [],
        "items": manifest_items,
    }
    write_json(export_dir / AUTO_SPLIT_MANIFEST, manifest)
    return {"path": str(export_dir), **manifest}


def prepare_detection_upload(staging_dir: Path, destination_root: Path) -> dict:
    """ติดตั้ง export ที่พร้อมใช้และ Auto Split เฉพาะข้อมูลที่ยังไม่มี Train/Val."""
    destination_root.mkdir(parents=True, exist_ok=True)
    yaml_roots = sorted({path.parent for path in staging_dir.rglob("data.yaml")})
    ready_roots = [root for root in yaml_roots if dataset_has_usable_split(root)]
    installed_paths: list[str] = []
    copied_files = 0
    for dataset_root in ready_roots:
        destination = copytree_unique(
            dataset_root,
            destination_root,
            f"{timestamp_name()}_{safe_name(dataset_root.name)}",
        )
        installed_paths.append(str(destination))
        copied_files += sum(1 for path in destination.rglob("*") if path.is_file())

    project_config = read_project_config()
    fallback_names = [str(name) for name in project_config.get("target_classes", [])]
    pairs, notes = discover_unsplit_yolo_pairs(staging_dir, ready_roots)
    auto_split = None
    if pairs:
        class_names = read_uploaded_class_names(staging_dir, fallback_names)
        auto_split = create_auto_split_export(pairs, destination_root, class_names, notes)
        installed_paths.append(auto_split["path"])
        copied_files += sum(1 for path in Path(auto_split["path"]).rglob("*") if path.is_file())

    return {
        "installed_paths": installed_paths,
        "copied_files": copied_files,
        "ready_exports": len(ready_roots),
        "auto_split": auto_split,
    }


def copy_images_from_staging(staging_dir: Path, destination_root: Path, prefix: str) -> int:
    """copy เฉพาะรูปไปปลายทาง ใช้กับ background/negative images."""
    destination_dir = unique_child_path(destination_root, f"{timestamp_name()}_{safe_name(prefix)}")
    copied = 0
    for image_path in sorted(path for path in staging_dir.rglob("*") if is_image_file(path)):
        try:
            relative = image_path.relative_to(staging_dir)
        except ValueError:
            relative = Path(image_path.name)
        target = destination_dir.joinpath(*safe_relative_parts(str(relative)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target)
        copied += 1
    return copied


# ===== Dataset Workspace และ metadata =====
# dataset_meta.json เป็นตัวบอกประเภท Known-good/Brand Reference และชื่อแสดงผลของ folder
# ห้ามเดาประเภทจากชื่อ folder อย่างเดียวเมื่อมี metadata เพราะชื่อสามารถซ้ำหรือเปลี่ยนได้
def dataset_summary() -> dict:
    """สรุปข้อมูลที่พร้อมใช้ train จากโฟลเดอร์โปรเจกต์."""
    background_split = detection_background_summary(AIRBAG_BACKGROUND_DIR)
    detection_folders: list[Path] = []
    if AIRBAG_EXPORTS_DIR.exists():
        detection_folders = [
            folder_path
            for folder_path in sorted(path for path in AIRBAG_EXPORTS_DIR.iterdir() if path.is_dir())
            if ensure_dataset_meta(folder_path).get("dataset_type") == "Known-good Reference"
        ]
    summary = {
        "airbag": {
            "label": UPLOAD_CATEGORIES["airbag"]["label"],
            "exports": sum(count_yolov8_exports(folder_path) for folder_path in detection_folders),
            "images": sum(count_yolov8_export_images(folder_path) for folder_path in detection_folders),
            "path": str(AIRBAG_EXPORTS_DIR),
        },
        "toyota": {
            "label": UPLOAD_CATEGORIES["toyota"]["label"],
            "exports": count_yolov8_exports(BRAND_EXPORTS_DIR / "Toyota"),
            "images": count_images(BRAND_EXPORTS_DIR / "Toyota"),
            "path": str(BRAND_EXPORTS_DIR / "Toyota"),
        },
        "honda": {
            "label": UPLOAD_CATEGORIES["honda"]["label"],
            "exports": count_yolov8_exports(BRAND_EXPORTS_DIR / "Honda"),
            "images": count_images(BRAND_EXPORTS_DIR / "Honda"),
            "path": str(BRAND_EXPORTS_DIR / "Honda"),
        },
        "background": {
            "label": UPLOAD_CATEGORIES["background"]["label"],
            "images": background_split["total"],
            "train_images": background_split["train"],
            "eval_valid_images": background_split["eval_valid"],
            "eval_test_images": background_split["eval_test"],
            "split_ready": background_split["ready"],
            "brand_unknown_images": count_images(BRAND_BACKGROUND_DIR),
            "path": str(AIRBAG_BACKGROUND_DIR),
        },
    }
    if AIRBAG_EXPORTS_DIR.exists():
        for folder_path in sorted(path for path in AIRBAG_EXPORTS_DIR.iterdir() if path.is_dir()):
            meta = ensure_dataset_meta(folder_path)
            if meta.get("dataset_type") != "Brand Reference":
                continue
            class_name = classification_class_name_from_meta(meta, folder_display_name(folder_path))
            if not class_name:
                continue
            key = f"classification_{safe_name(class_name).lower()}"
            existing = summary.get(key, {})
            summary[key] = {
                "label": class_name,
                "exports": int(existing.get("exports") or 0) + count_yolov8_exports(folder_path),
                "images": int(existing.get("images") or 0) + count_images(folder_path),
                "path": str(folder_path),
                "source": "workspace_folder",
            }
    return summary


def count_yolov8_export_images(path: Path) -> int:
    """นับเฉพาะรูปที่อยู่ใต้ YOLOv8 export ที่มี data.yaml."""
    if not path.exists():
        return 0
    dataset_roots = {item.parent for item in path.rglob("data.yaml") if item.is_file()}
    return sum(count_images(dataset_root) for dataset_root in dataset_roots)


def dataset_folder_path(folder_name: str) -> Path:
    """แปลงชื่อ folder จากหน้าเว็บให้เป็น path ที่อยู่ใน AIRBAG_EXPORTS_DIR เท่านั้น."""
    clean_folder = safe_name(folder_name)
    folder_path = (AIRBAG_EXPORTS_DIR / clean_folder).resolve()
    exports_root = AIRBAG_EXPORTS_DIR.resolve()
    if folder_path == exports_root or exports_root not in folder_path.parents:
        raise ValueError("Invalid dataset folder.")
    return folder_path


def dataset_meta_path(folder_path: Path) -> Path:
    return folder_path / "dataset_meta.json"


def normalize_dataset_meta(payload: dict) -> dict:
    allowed_fields = {
        "dataset_name": "Dataset Name",
        "brand": "Brand",
        "model": "Model",
        "target": "Target / Object",
        "dataset_type": "Dataset Type",
        "notes": "Notes",
    }
    meta = {}
    for key, label in allowed_fields.items():
        value = str(payload.get(key) or "").strip()
        meta[key] = value[:1200] if key == "notes" else value[:160]
        meta[f"{key}_label"] = label
    type_aliases = {
        "Detection": "Known-good Reference",
        "Classification": "Brand Reference",
    }
    meta["dataset_type"] = type_aliases.get(meta.get("dataset_type"), meta.get("dataset_type"))
    allowed_dataset_types = {"Known-good Reference", "Brand Reference", "Negative / Background"}
    if meta.get("dataset_type") not in allowed_dataset_types:
        meta["dataset_type"] = ""
    if meta.get("dataset_type") == "Brand Reference":
        class_name = " ".join(part for part in [meta.get("brand"), meta.get("model")] if part).strip()
        if class_name:
            meta["dataset_name"] = class_name[:160]
    meta["updated_at"] = now_iso()
    return meta


def read_dataset_meta(folder_path: Path) -> dict:
    try:
        meta = read_json(dataset_meta_path(folder_path), {})
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def folder_display_name(path: Path) -> str:
    return path.name.replace("_", " ")


def infer_dataset_meta(folder_path: Path) -> dict:
    display_name = folder_display_name(folder_path)
    lowered_name = folder_path.name.lower()
    project_config = read_project_config()
    has_exports = count_yolov8_exports(folder_path) > 0
    has_images = count_images(folder_path) > 0
    if has_exports:
        inferred_type = "Known-good Reference"
    elif any(word in lowered_name for word in ["background", "negative", "unknown"]):
        inferred_type = "Negative / Background"
    else:
        inferred_type = "Known-good Reference" if has_images else ""
    meta = normalize_dataset_meta(
        {
            "dataset_name": display_name,
            "brand": project_config.get("brand", ""),
            "model": project_config.get("model", ""),
            "target": project_config.get("model", "") if has_images else "",
            "dataset_type": inferred_type,
            "notes": "Auto-filled from existing workspace folder.",
        }
    )
    meta["auto_filled"] = True
    meta["review_status"] = "needs_info"
    return meta


def ensure_dataset_meta(folder_path: Path) -> dict:
    """Resolve metadata for display without writing or confirming dataset images."""
    meta = read_dataset_meta(folder_path)
    if not meta:
        return infer_dataset_meta(folder_path) if count_yolov8_exports(folder_path) or count_images(folder_path) else {}
    meta = dict(meta)
    meta["dataset_type"] = {"Detection": "Known-good Reference", "Classification": "Brand Reference"}.get(
        meta.get("dataset_type"), meta.get("dataset_type"))
    return meta


def write_dataset_meta(folder_path: Path, payload: dict) -> dict:
    meta = normalize_dataset_meta(payload)
    existing = read_dataset_meta(folder_path)
    meta["created_at"] = existing.get("created_at") or now_iso()
    meta["auto_filled"] = False
    meta["review_status"] = "reviewed"
    meta["reviewed_at"] = now_iso()
    write_json(dataset_meta_path(folder_path), meta)
    return meta


def classification_class_name_from_meta(meta: dict, fallback: str = "") -> str:
    parts = [meta.get("brand"), meta.get("model")]
    class_name = " ".join(str(part or "").strip() for part in parts if str(part or "").strip()).strip()
    return class_name or str(meta.get("dataset_name") or fallback).strip()


def dataset_folder_record(path: Path) -> dict:
    exports = count_yolov8_exports(path)
    images = count_images(path)
    updated_at = path.stat().st_mtime if path.exists() else 0
    meta = ensure_dataset_meta(path)
    background_split = None
    if meta.get("dataset_type") == "Negative / Background":
        # Folder Explorer เป็นหน้าต่างเดียวของ Background แม้ไฟล์จริงถูกแยกไว้ใน
        # 02_background_images เพื่อไม่ให้ผู้ใช้ต้องรู้จักโครงสร้างภายในระบบ
        background_split = detection_background_summary(AIRBAG_BACKGROUND_DIR)
        images = background_split["total"]
    is_reviewed = meta.get("review_status") == "reviewed" and meta.get("auto_filled") is False
    is_auto_filled = not is_reviewed and (
        bool(meta.get("auto_filled")) or str(meta.get("notes") or "").startswith("Auto-filled")
    )
    needs_info = not meta or not is_reviewed or is_auto_filled
    label = meta.get("dataset_name") or folder_display_name(path)
    subtitle_parts = [meta.get("brand"), meta.get("model"), meta.get("target")]
    subtitle = " / ".join(part for part in subtitle_parts if part)
    return {
        "id": path.name,
        "name": path.name,
        "label": label,
        "folder_label": folder_display_name(path),
        "subtitle": subtitle,
        "meta": meta,
        "needs_info": needs_info,
        "review_status": "needs_info" if needs_info else "reviewed",
        "images": images,
        "exports": exports,
        "path": str(path),
        "updated_at": updated_at,
        "status": "ready" if images or exports else "empty",
        "background_split": background_split,
    }


def list_dataset_folder_records() -> list[dict]:
    AIRBAG_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    folders = [path for path in sorted(AIRBAG_EXPORTS_DIR.iterdir()) if path.is_dir()]
    return [dataset_folder_record(path) for path in folders]


def reviewed_known_good_corpus(dataset_ids: set[str] | None = None) -> dict:
    """Collect every confirmed Known-good Dataset and remove exact copies."""
    datasets: list[dict] = []
    unique_paths: list[Path] = []
    seen_hashes: set[str] = set()
    source_images = 0
    for record in list_dataset_folder_records():
        if dataset_ids is not None and str(record.get("id") or "") not in dataset_ids:
            continue
        meta = dict(record.get("meta") or {})
        if meta.get("dataset_type") != "Known-good Reference":
            continue
        if record.get("review_status") != "reviewed" or record.get("needs_info"):
            continue
        folder_paths = iter_images(dataset_folder_path(str(record["id"])))
        source_images += len(folder_paths)
        unique_in_dataset = 0
        for path in sorted(folder_paths, key=lambda item: str(item).lower()):
            content_hash = file_sha256(path)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            unique_paths.append(path)
            unique_in_dataset += 1
        datasets.append({
            "id": str(record["id"]),
            "label": str(record.get("label") or record.get("folder_label") or record["id"]),
            "images": len(folder_paths),
            "unique_contribution": unique_in_dataset,
            "recipe": {key: str(meta.get(key) or "").strip() for key in ("brand", "model", "target")},
        })
    return {
        "paths": unique_paths,
        "datasets": datasets,
        "dataset_count": len(datasets),
        "source_images": source_images,
        "unique_images": len(unique_paths),
        "duplicates_removed": max(0, source_images - len(unique_paths)),
    }


def validate_reference_recipe(datasets: list[dict]) -> None:
    """Do not silently combine unrelated products into one alignment template."""
    if len(datasets) <= 1:
        return
    recipes = [item.get("recipe") or {} for item in datasets]
    signatures = {tuple(str(recipe.get(key) or "").strip().casefold() for key in ("brand", "model", "target")) for recipe in recipes}
    if any(not str(recipe.get("model") or "").strip() for recipe in recipes) or len(signatures) != 1:
        raise ValueError("เลือกรูปชิ้นงาน/มุมตรวจเดียวกันก่อน Train: หลาย Dataset ต้องระบุ Brand, Model และ Target ให้สอดคล้องกัน และต้องมี Model ห้ามรวมภาพด้านหน้า/ด้านหลังหรือคนละรุ่นเป็น Reference เดียว")


def deleted_dataset_records() -> list[dict]:
    if not DELETED_DATASETS_DIR.exists():
        return []
    records = []
    for path in sorted(
        (item for item in DELETED_DATASETS_DIR.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        deletion_meta = read_json(path / "deletion.json", {})
        records.append(
            {
                "id": path.name,
                "original_name": deletion_meta.get("original_name") or path.name,
                "deleted_at": deletion_meta.get("deleted_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "images": count_images(path),
                "exports": count_yolov8_exports(path),
            }
        )
    return records


def unique_child_path(destination_root: Path, name: str) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    base = destination_root / safe_name(name)
    destination = base
    index = 2
    while destination.exists():
        destination = destination_root / f"{base.name}_{index}"
        index += 1
    return destination


def copy_folder_contents(source: Path, destination: Path) -> int:
    copied = 0
    for item in sorted(path for path in source.rglob("*") if path.is_file()):
        try:
            relative = item.relative_to(source)
        except ValueError:
            relative = Path(item.name)
        target = destination.joinpath(*safe_relative_parts(str(relative)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        copied += 1
    return copied


def install_staging_to_dataset_folder(staging_dir: Path, target_folder: Path) -> dict:
    target_folder.mkdir(parents=True, exist_ok=True)
    prepared = prepare_detection_upload(staging_dir, target_folder)
    installed_paths = prepared["installed_paths"]
    copied_files = int(prepared["copied_files"])
    # ถ้าไม่มี label เลย ให้เก็บเป็นรูปดิบสำหรับนำไป Annotation แทนการเดาว่าเป็น negative
    if not installed_paths:
        raw_destination = unique_child_path(target_folder / "raw_images", timestamp_name())
        copied_files += copy_images_from_staging(staging_dir, raw_destination, "images")
        installed_paths.append(str(raw_destination))
    return {
        "installed_paths": installed_paths,
        "copied_files": copied_files,
        "ready_exports": prepared["ready_exports"],
        "auto_split": prepared["auto_split"],
        "exports": count_yolov8_exports(target_folder),
        "images": count_images(target_folder),
    }


# ===== การประเมินขนาดงานและค่าที่แนะนำ =====
# ค่าที่คืนเป็นคำแนะนำตามจำนวนภาพ/ทรัพยากร ไม่ใช่ค่าบังคับ และต้องคำนวณแยกระหว่าง
# detector กับ classifier เพราะรูปแบบ dataset และต้นทุน train ต่างกัน
def detection_train_stats() -> dict:
    folders = [
        item
        for item in list_dataset_folder_records()
        if str(item.get("meta", {}).get("dataset_type") or "").strip() == "Detection"
    ]
    return {
        "folders": len(folders),
        "exports": sum(int(item.get("exports") or 0) for item in folders),
        "images": sum(int(item.get("images") or 0) for item in folders),
    }


def brand_train_stats() -> dict:
    summary = dataset_summary()
    class_items = [
        item
        for key, item in summary.items()
        if key != "airbag" and key != "background" and int(item.get("images") or 0) > 0
    ]
    return {
        "classes": len(class_items),
        "images": sum(int(item.get("images") or 0) for item in class_items),
        "class_names": [str(item.get("label") or "") for item in class_items],
        "class_counts": {
            str(item.get("label") or ""): int(item.get("images") or 0)
            for item in class_items
        },
        "background_images": int(summary.get("background", {}).get("brand_unknown_images") or 0),
    }


# ===== วิเคราะห์ความถูกต้องของ YOLO export =====
# อ่าน class ID/พิกัดจาก label โดยไม่แก้ไฟล์ต้นทาง ผลวิเคราะห์ถูก cache เพื่อให้หน้าเว็บ
# ตอบเร็ว; ทุก mutation ของ dataset ต้อง clear_analysis_cache มิฉะนั้น UI จะแสดงข้อมูลเก่า
YOLO_SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}


def yolo_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    aliases = YOLO_SPLIT_ALIASES.get(split, (split,))
    candidates: list[tuple[Path, Path]] = []
    for alias in aliases:
        candidates.append((dataset_root / alias / "images", dataset_root / alias / "labels"))
        candidates.append((dataset_root / "images" / alias, dataset_root / "labels" / alias))

    for images_dir, labels_dir in candidates:
        if images_dir.exists() or labels_dir.exists():
            return images_dir, labels_dir

    alias = aliases[0]
    return dataset_root / alias / "images", dataset_root / alias / "labels"


def read_yolo_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return []
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    names = data.get("names", [])
    if isinstance(names, dict):
        items = []
        for key, value in names.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                index = len(items)
            items.append((index, str(value)))
        return [value for _, value in sorted(items)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def yolo_export_detail(dataset_root: Path) -> dict:
    split_details = {}
    total_images = 0
    total_labels = 0
    missing_labels = 0
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = yolo_split_dirs(dataset_root, split)
        images = sorted(path for path in images_dir.rglob("*") if is_image_file(path)) if images_dir.exists() else []
        labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        split_missing = len(image_stems - label_stems)
        if images or labels:
            split_details[split] = {
                "images": len(images),
                "labels": len(labels),
                "missing_labels": split_missing,
            }
        total_images += len(images)
        total_labels += len(labels)
        missing_labels += split_missing
    return {
        "path": str(dataset_root),
        "name": dataset_root.name,
        "has_data_yaml": (dataset_root / "data.yaml").exists(),
        "images": total_images,
        "labels": total_labels,
        "missing_labels": missing_labels,
        "splits": split_details,
    }


def yolo_export_integrity(dataset_root: Path) -> dict:
    names = read_yolo_class_names(dataset_root)
    class_boxes = {name: 0 for name in names}
    bad_lines = 0
    invalid_class_ids = 0
    sanitized_coordinates = 0
    empty_labels = 0
    split_stems: dict[str, set[str]] = {}
    orphan_labels = 0
    for split in ("train", "valid", "test"):
        images_dir, labels_dir = yolo_split_dirs(dataset_root, split)
        images = [path for path in images_dir.rglob("*") if is_image_file(path)] if images_dir.exists() else []
        labels = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        split_stems[split] = image_stems
        orphan_labels += len(label_stems - image_stems)
        for label_path in labels:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                parts = line.split()
                try:
                    values = [float(part) for part in parts]
                except ValueError:
                    bad_lines += 1
                    continue
                if len(values) < 5 or not values[0].is_integer():
                    bad_lines += 1
                    continue
                class_id = int(values[0])
                if class_id < 0 or class_id >= len(names):
                    invalid_class_ids += 1
                    continue
                coords = values[1:]
                if any(value < 0.0 or value > 1.0 for value in coords):
                    sanitized_coordinates += 1
                if len(values) == 5 and (values[3] <= 0 or values[4] <= 0):
                    bad_lines += 1
                    continue
                if len(values) > 5 and len(coords) < 6:
                    bad_lines += 1
                    continue
                class_boxes[names[class_id]] = class_boxes.get(names[class_id], 0) + 1
    leakage = {
        "train_valid": len(split_stems.get("train", set()) & split_stems.get("valid", set())),
        "train_test": len(split_stems.get("train", set()) & split_stems.get("test", set())),
        "valid_test": len(split_stems.get("valid", set()) & split_stems.get("test", set())),
    }
    return {
        "classes": names,
        "class_boxes": class_boxes,
        "bad_lines": bad_lines,
        "invalid_class_ids": invalid_class_ids,
        "sanitized_coordinates": sanitized_coordinates,
        "empty_labels": empty_labels,
        "orphan_labels": orphan_labels,
        "split_name_leakage": leakage,
    }


def folder_yolo_export_details(folder_path: Path) -> list[dict]:
    if not folder_path.exists():
        return []
    roots = sorted({path.parent for path in folder_path.rglob("data.yaml") if path.is_file()})
    return [yolo_export_detail(root) for root in roots]


def image_sample_record(folder_path: Path, image_path: Path) -> dict | None:
    try:
        relative = image_path.relative_to(folder_path).as_posix()
    except ValueError:
        return None
    return {
        "name": image_path.name,
        "relative_path": relative,
        "url": f"/api/dataset-folders/{quote(folder_path.name, safe='')}/image?path={quote(relative, safe='/')}",
    }


def image_samples(folder_path: Path, limit: int = 12) -> list[dict]:
    samples = []
    for image_path in sorted(path for path in folder_path.rglob("*") if is_image_file(path)):
        sample = image_sample_record(folder_path, image_path)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


LATEST_UPLOAD_MANIFEST = ".latest_upload.json"


def record_latest_dataset_upload(folder_path: Path, installed_paths: list[str]) -> dict:
    """บันทึกรูปของ upload รอบล่าสุดเท่านั้น เพื่อส่งต่อจาก Upload ไปหน้า Train โดยไม่ปนรูปเก่า."""
    folder_root = folder_path.resolve()
    relative_paths: list[str] = []
    undo_files = {}
    for raw_path in installed_paths:
        installed_path = Path(str(raw_path)).resolve()
        if installed_path != folder_root and folder_root not in installed_path.parents:
            continue
        for item in ([installed_path] if installed_path.is_file() else installed_path.rglob('*')):
            if item.is_file() and not item.is_symlink() and folder_root in item.resolve().parents:
                undo_files[item.relative_to(folder_root).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
        candidates = [installed_path] if is_image_file(installed_path) else sorted(
            path for path in installed_path.rglob("*") if is_image_file(path)
        )
        for image_path in candidates:
            try:
                relative_paths.append(image_path.relative_to(folder_root).as_posix())
            except ValueError:
                continue
    relative_paths = sorted(dict.fromkeys(relative_paths))
    manifest = {
        "folder": folder_path.name,
        "uploaded_at": now_iso(),
        "count": len(relative_paths),
        "relative_paths": relative_paths,
        "undo_files": undo_files,
        "upload_id": uuid.uuid4().hex,
    }
    write_json(folder_path / LATEST_UPLOAD_MANIFEST, manifest)
    return manifest


def latest_dataset_upload_page(folder_path: Path, offset: int = 0, limit: int = 48) -> dict:
    manifest = read_json(folder_path / LATEST_UPLOAD_MANIFEST, {})
    folder_root = folder_path.resolve()
    samples: list[dict] = []
    for relative in manifest.get("relative_paths") or []:
        target = (folder_root / Path(str(relative))).resolve()
        if target == folder_root or folder_root not in target.parents or not is_image_file(target):
            continue
        sample = image_sample_record(folder_path, target)
        if sample is not None:
            samples.append(sample)
    total = len(samples)
    page = samples[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "folder": folder_path.name,
        "uploaded_at": manifest.get("uploaded_at") or "",
        "upload_id": manifest.get("upload_id"),
        "can_undo": bool(manifest.get("undo_files")),
        "samples": page,
        "total": total,
        "offset": offset,
        "next_offset": next_offset,
        "has_more": next_offset < total,
    }


def dataset_folder_preview(folder_path: Path) -> dict:
    record = dataset_folder_record(folder_path)
    yolo_exports = folder_yolo_export_details(folder_path)
    return {
        "folder": record,
        "images": count_images(folder_path),
        # Count only YOLO annotation files inside split/labels directories.
        # A raw rglob("*.txt") also counts README.dataset.txt and
        # README.roboflow.txt, which made the UI over-report label coverage.
        "labels": sum(int(export.get("labels") or 0) for export in yolo_exports),
        "exports": len(yolo_exports),
        "yolo_exports": yolo_exports,
        "samples": image_samples(folder_path),
    }


def combined_label_relationship_stats(target_classes: list[str]) -> dict | None:
    if len(target_classes) <= 1:
        return None
    dataset_name = safe_name("_".join(target_classes))
    dataset_dir = AIRBAG_EXPORTS_DIR.parent / "03_output" / f"{dataset_name}_yolov8_combined"
    if not dataset_dir.exists():
        return None

    stats = {
        "dataset": str(dataset_dir),
        "label_files": 0,
        "empty_files": 0,
        "single_class_files": 0,
        "multi_class_files": 0,
    }
    for label_file in dataset_dir.glob("*/*/*.txt"):
        stats["label_files"] += 1
        text = label_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            stats["empty_files"] += 1
            continue
        class_ids = set()
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_ids.add(int(float(parts[0])))
            except ValueError:
                continue
        if len(class_ids) > 1:
            stats["multi_class_files"] += 1
        elif len(class_ids) == 1:
            stats["single_class_files"] += 1
    return stats


def combined_dataset_quality_stats(target_classes: list[str]) -> dict | None:
    """ตรวจคุณภาพ combined dataset ที่ใช้ train จริง ไม่ใช่ดูแค่ export ต้นทาง.

    จุดนี้ตั้งใจให้ preflight มองเห็นปัญหาที่ทำให้ metric หลอกตาได้ เช่น
    - test split ไม่มี class สำคัญบางตัว
    - กล่อง full-frame เยอะจนวัด localization จริงไม่ได้
    - ภาพซ้ำ/คล้ายกันรั่วข้าม train/valid/test
    """
    if not target_classes:
        return None

    dataset_name = safe_name("_".join(target_classes))
    dataset_dir = AIRBAG_EXPORTS_DIR.parent / "03_output" / f"{dataset_name}_yolov8_combined"
    if not dataset_dir.exists():
        return None

    source_records = []
    for path in sorted(dataset_dir.glob("*/*/*")):
        if not path.is_file() or (not is_image_file(path) and path.suffix.lower() != ".txt"):
            continue
        stat = path.stat()
        source_records.append(
            {
                "path": path.relative_to(dataset_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    source_fingerprint = hashlib.sha256(
        json.dumps(source_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report_path = dataset_dir / "quality_report.json"
    cached_report = read_json(report_path, {})
    if cached_report.get("source_fingerprint") == source_fingerprint:
        return cached_report

    names = read_yolo_class_names(dataset_dir) or list(target_classes)
    split_names = ("train", "valid", "test")
    splits: dict[str, dict] = {}
    image_hashes: list[tuple[str, str, int]] = []

    def image_dhash(image_path: Path) -> int | None:
        image = cv2.imread(str(image_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if image is None:
            return None
        resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        bits = resized[:, :-1] > resized[:, 1:]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bool(bit))
        return value

    for split in split_names:
        images_dir, labels_dir = yolo_split_dirs(dataset_dir, split)
        images = sorted(path for path in images_dir.rglob("*") if is_image_file(path)) if images_dir.exists() else []
        labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        class_counts = {name: 0 for name in names}
        empty_labels = 0
        bad_rows = 0
        full_frame_boxes = 0
        edge_touch_boxes = 0
        rows = 0

        for image_path in images:
            digest = image_dhash(image_path)
            if digest is not None:
                image_hashes.append((split, image_path.name, digest))

        for label_file in labels:
            text = label_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                empty_labels += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    bad_rows += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x_center, y_center, width, height = [float(part) for part in parts[1:]]
                except ValueError:
                    bad_rows += 1
                    continue
                rows += 1
                if 0 <= class_id < len(names):
                    class_counts[names[class_id]] = class_counts.get(names[class_id], 0) + 1
                else:
                    bad_rows += 1
                if width >= 0.98 and height >= 0.98:
                    full_frame_boxes += 1
                if (
                    x_center - width / 2 <= 0.0001
                    or y_center - height / 2 <= 0.0001
                    or x_center + width / 2 >= 0.9999
                    or y_center + height / 2 >= 0.9999
                ):
                    edge_touch_boxes += 1

        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_label_images": len(image_stems - label_stems),
            "empty_labels": empty_labels,
            "rows": rows,
            "class_counts": class_counts,
            "bad_rows": bad_rows,
            "full_frame_boxes": full_frame_boxes,
            "edge_touch_boxes": edge_touch_boxes,
        }

    near_pairs = 0
    exact_pairs = 0
    samples: list[dict] = []
    for index, left in enumerate(image_hashes):
        left_split, left_name, left_hash = left
        for right_split, right_name, right_hash in image_hashes[index + 1 :]:
            if left_split == right_split:
                continue
            distance = int((left_hash ^ right_hash).bit_count())
            if distance == 0:
                exact_pairs += 1
            if distance <= 4:
                near_pairs += 1
                if len(samples) < 8:
                    samples.append(
                        {
                            "distance": distance,
                            "left": {"split": left_split, "image": left_name},
                            "right": {"split": right_split, "image": right_name},
                        }
                    )

    report = {
        "source_fingerprint": source_fingerprint,
        "dataset": str(dataset_dir),
        "classes": names,
        "splits": splits,
        "cross_split_duplicates": {
            "exact_pairs": exact_pairs,
            "near_pairs": near_pairs,
            "threshold": 4,
            "samples": samples,
        },
    }
    write_json(report_path, report)
    return report


def background_visual_assist_risk_stats(limit: int = 12) -> dict:
    """วัดว่า Visual Assist มีแนวโน้มตี background เป็น Airbag หรือไม่."""
    images = [
        path
        for path in AIRBAG_BACKGROUND_DIR.rglob("*")
        if is_image_file(path)
    ] if AIRBAG_BACKGROUND_DIR.exists() else []
    source_records = []
    for path in sorted(images):
        stat = path.stat()
        source_records.append(
            {
                "path": path.relative_to(AIRBAG_BACKGROUND_DIR).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    source_fingerprint = hashlib.sha256(
        json.dumps(
            {"gate_version": 2, "files": source_records},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report_path = AIRBAG_BACKGROUND_DIR / "visual_assist_risk_report.json"
    cached_report = read_json(report_path, {})
    if cached_report.get("source_fingerprint") == source_fingerprint:
        return cached_report

    hit_images = 0
    samples: list[dict] = []
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        candidates = detector.find_visual_airbag_candidates(frame, [])
        selected = detector.select_airbag_detections(candidates, frame.shape, []) if candidates else []
        if not selected:
            continue
        hit_images += 1
        if len(samples) < limit:
            samples.append(
                {
                    "image": str(image_path),
                    "detections": [
                        {"box": [int(value) for value in xyxy], "confidence": round(float(confidence), 5)}
                        for xyxy, confidence in selected
                    ],
                }
            )
    rate = hit_images / len(images) if images else 0.0
    report = {
        "source_fingerprint": source_fingerprint,
        "images": len(images),
        "hit_images": hit_images,
        "hit_rate": round(rate, 5),
        "samples": samples,
    }
    write_json(report_path, report)
    return report


def dataset_class_compatibility(config: dict, detection_folders: list[dict] | None = None) -> dict:
    """Report exactly how every source dataset class will map before training."""
    folders = detection_folders
    if folders is None:
        folders = [
            folder
            for folder in list_dataset_folder_records()
            if str(folder.get("meta", {}).get("dataset_type") or "").strip() == "Detection"
        ]
    lookup = project_class_lookup(config)
    source_records = []
    unmapped = set()
    mapped_pairs = set()
    for folder in folders:
        folder_path = dataset_folder_path(folder["id"])
        for detail in folder_yolo_export_details(folder_path):
            root = Path(detail["path"])
            for source_name in read_yolo_class_names(root):
                target_name = lookup.get(normalized_class_key(source_name))
                record = {
                    "source": source_name,
                    "target": target_name or "",
                    "folder": folder.get("label") or folder.get("id"),
                    "export": detail.get("name") or root.name,
                    "status": "mapped" if target_name else "unmapped",
                }
                source_records.append(record)
                if target_name:
                    mapped_pairs.add((source_name, target_name))
                else:
                    unmapped.add(source_name)
    return {
        "targets": list(config.get("target_classes") or []),
        "mapped": [{"source": source, "target": target} for source, target in sorted(mapped_pairs)],
        "unmapped": sorted(unmapped, key=str.lower),
        "sources": source_records,
    }


# ===== Preflight ก่อน train =====
# รวมผล dataset integrity, class compatibility, disk และ model contract เป็นคำตอบเดียว
# api_start_train ต้องไม่ข้ามผล error ของขั้นนี้ แม้ผู้ใช้ส่ง request ตรงโดยไม่ผ่านหน้าเว็บ
def _compute_train_preflight(train_type: str) -> dict:
    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []
    class_report: dict | None = None
    combined_quality: dict | None = None
    visual_risk: dict | None = None
    if train_type == "airbag_detector":
        detection_folders = [
            folder
            for folder in list_dataset_folder_records()
            if str(folder.get("meta", {}).get("dataset_type") or "").strip() == "Detection"
        ]
        total_exports = sum(int(folder.get("exports") or 0) for folder in detection_folders)
        total_images = sum(int(folder.get("images") or 0) for folder in detection_folders)
        checks.append({"label": "Detection folders", "status": "pass" if detection_folders else "fail", "detail": f"{len(detection_folders)} folders"})
        checks.append({"label": "YOLOv8 exports", "status": "pass" if total_exports else "fail", "detail": f"{total_exports} exports"})
        checks.append({"label": "Images", "status": "pass" if total_images else "fail", "detail": f"{total_images} images"})
        if not detection_folders or not total_exports:
            errors.append("ยังไม่มี Detection dataset ที่มี data.yaml สำหรับ train")
        background_split = detection_background_summary(AIRBAG_BACKGROUND_DIR)
        background_integrity = detection_background_integrity(AIRBAG_BACKGROUND_DIR)
        checks.extend(
            [
                {
                    "label": "Background train",
                    "status": "pass" if background_split["train"] > 0 else "fail",
                    "detail": f"{background_split['train']} images",
                },
                {
                    "label": "Background eval",
                    "status": "pass" if background_split["ready"] else "fail",
                    "detail": f"valid {background_split['eval_valid']} / test {background_split['eval_test']}",
                },
                {
                    "label": "Background isolation",
                    "status": "pass" if background_integrity["ok"] else "fail",
                    "detail": f"exact {background_integrity['exact_pairs']} / near {background_integrity['near_pairs']}",
                },
            ]
        )
        if background_split["train"] <= 0:
            errors.append("Detection training needs background/negative images in the Train split.")
        if background_split["eval_valid"] < MIN_BACKGROUND_EVAL_IMAGES or background_split["eval_test"] < MIN_BACKGROUND_EVAL_IMAGES:
            errors.append(
                "Background evaluation is incomplete: upload enough diverse background images so "
                f"eval_valid and eval_test contain at least {MIN_BACKGROUND_EVAL_IMAGES} images each."
            )
        if not background_integrity["ok"]:
            errors.append("Background images are duplicated or visually too similar across train/eval splits.")
        project_config = read_project_config()
        target_classes = list(project_config.get("target_classes") or [])
        class_report = dataset_class_compatibility(project_config, detection_folders)
        checks.append(
            {
                "label": "Class mapping",
                "status": "fail" if class_report["unmapped"] else "pass",
                "detail": (
                    f"{len(class_report['unmapped'])} unmapped"
                    if class_report["unmapped"]
                    else f"{len(class_report['mapped'])} source mappings"
                ),
            }
        )
        if class_report["unmapped"]:
            errors.append(
                "Dataset classes are not mapped to this project: "
                + ", ".join(class_report["unmapped"])
                + ". Open Manage Classes and add aliases before training."
            )
        relationship_stats = combined_label_relationship_stats(target_classes)
        if relationship_stats and relationship_stats["multi_class_files"] == 0:
            primary_class = str(project_config.get("primary_class") or (target_classes[0] if target_classes else "primary object"))
            secondary_classes = [name for name in target_classes if name.lower() != primary_class.lower()]
            secondary_text = ", ".join(secondary_classes) if secondary_classes else "secondary/defect classes"
            warnings.append(
                "The latest combined dataset has no image whose label file contains multiple classes "
                f"({', '.join(target_classes)}). For main-object + defect work, label both "
                f"{primary_class} and {secondary_text} in the same image before training for stable joint detection."
            )

        combined_quality = combined_dataset_quality_stats(target_classes)
        if combined_quality:
            test_counts = (combined_quality["splits"].get("test") or {}).get("class_counts") or {}
            missing_test_classes = [
                name
                for name in target_classes
                if int(test_counts.get(name) or 0) <= 0
            ]
            duplicate_stats = combined_quality["cross_split_duplicates"]
            total_near_duplicates = int(duplicate_stats.get("near_pairs") or 0)
            total_exact_duplicates = int(duplicate_stats.get("exact_pairs") or 0)
            all_rows = sum(int(split.get("rows") or 0) for split in combined_quality["splits"].values())
            full_frame_boxes = sum(int(split.get("full_frame_boxes") or 0) for split in combined_quality["splits"].values())
            test_rows = int((combined_quality["splits"].get("test") or {}).get("rows") or 0)
            test_full_frame = int((combined_quality["splits"].get("test") or {}).get("full_frame_boxes") or 0)
            full_frame_ratio = full_frame_boxes / all_rows if all_rows else 0.0
            test_full_frame_ratio = test_full_frame / test_rows if test_rows else 0.0

            checks.extend(
                [
                    {
                        "label": "Test class coverage",
                        "status": "warn" if missing_test_classes else "pass",
                        "detail": (
                            "missing " + ", ".join(missing_test_classes)
                            if missing_test_classes
                            else "all target classes present"
                        ),
                    },
                    {
                        "label": "Cross-split duplicates",
                        "status": "warn" if total_near_duplicates else "pass",
                        "detail": f"exact {total_exact_duplicates} / near {total_near_duplicates}",
                    },
                    {
                        "label": "Full-frame labels",
                        "status": "warn" if test_full_frame_ratio > 0.5 or full_frame_ratio > 0.45 else "pass",
                        "detail": f"test {test_full_frame}/{test_rows}, all {full_frame_boxes}/{all_rows}",
                    },
                ]
            )
            if missing_test_classes:
                warnings.append(
                    "The latest combined test split does not contain these target classes: "
                    + ", ".join(missing_test_classes)
                    + ". Metrics for those classes cannot be trusted until the split is rebuilt."
                )
            if total_near_duplicates:
                warnings.append(
                    f"The latest combined dataset has {total_near_duplicates} visually similar image pair(s) across train/valid/test "
                    "and should be rebuilt with duplicate-aware splitting for honest validation."
                )
            if test_full_frame_ratio > 0.5:
                warnings.append(
                    "Most test labels are full-frame boxes, so Airbag metrics may look stronger than real camera localization."
                )

        visual_risk = background_visual_assist_risk_stats()
        checks.append(
            {
                "label": "Visual Assist background",
                "status": "warn" if visual_risk["hit_images"] else "pass",
                "detail": f"{visual_risk['hit_images']} / {visual_risk['images']} background hits",
            }
        )
        if visual_risk["hit_images"]:
            warnings.append(
                "Visual Assist can still propose Airbag boxes on background images; tighten fallback gating before trusting camera output."
            )

        total_bad = 0
        integrity_jobs: list[tuple[dict, dict]] = []
        for folder in detection_folders:
            folder_path = dataset_folder_path(folder["id"])
            details = folder_yolo_export_details(folder_path)
            label_count = sum(item["labels"] for item in details)
            missing_count = sum(item["missing_labels"] for item in details)
            if label_count <= 0:
                errors.append(f"{folder['label']} ไม่มีไฟล์ label สำหรับ train")
            elif missing_count:
                warnings.append(f"{folder['label']} มีรูปที่ยังไม่มี label {missing_count} ไฟล์")
            integrity_jobs.extend((folder, detail) for detail in details)

        def inspect_integrity(job: tuple[dict, dict]) -> tuple[dict, dict, dict]:
            folder, detail = job
            return folder, detail, yolo_export_integrity(Path(detail["path"]))

        max_workers = min(8, max(1, len(integrity_jobs)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            integrity_results = list(executor.map(inspect_integrity, integrity_jobs))
        for folder, detail, integrity in integrity_results:
                total_bad += integrity["bad_lines"] + integrity["invalid_class_ids"] + integrity["orphan_labels"]
                if integrity["bad_lines"] or integrity["invalid_class_ids"] or integrity["orphan_labels"]:
                    errors.append(
                        f"{folder['label']} / {detail['name']} label integrity failed: "
                        f"bad={integrity['bad_lines']}, invalid class={integrity['invalid_class_ids']}, "
                        f"orphan={integrity['orphan_labels']}"
                    )
                if integrity["sanitized_coordinates"]:
                    warnings.append(
                        f"{folder['label']} / {detail['name']} has "
                        f"{integrity['sanitized_coordinates']} label rows whose coordinates will be clamped during merge."
                    )
                leakage_count = sum(integrity["split_name_leakage"].values())
                if leakage_count:
                    warnings.append(
                        f"{folder['label']} / {detail['name']} reuses {leakage_count} image names across train/validation/test."
                    )
        checks.append(
            {
                "label": "Label integrity",
                "status": "fail" if total_bad else "pass",
                "detail": f"{total_bad} invalid rows" if total_bad else "class IDs and rows valid",
            }
        )
    elif train_type == "brand_classifier":
        stats = brand_train_stats()
        unknown_low = bool(stats["images"] and stats["background_images"] < max(50, int(stats["images"] * 0.05)))
        checks.append({"label": "Classification classes", "status": "pass" if stats["classes"] else "fail", "detail": f"{stats['classes']} classes"})
        checks.append({"label": "Class images", "status": "pass" if stats["images"] else "fail", "detail": f"{stats['images']} images"})
        checks.append({"label": "Unknown/background", "status": "warn" if stats["background_images"] == 0 or unknown_low else "pass", "detail": f"{stats['background_images']} images"})
        if stats["classes"] <= 0:
            errors.append("ยังไม่มี class สำหรับ Brand/Type Classification")
        if stats["images"] <= 0:
            errors.append("ยังไม่มีรูปภาพสำหรับ train classifier")
        if stats["background_images"] <= 0:
            warnings.append("ยังไม่มี unknown/background สำหรับช่วยลดการเดา class มั่ว")
        if stats["background_images"] > 0 and unknown_low:
            warnings.append(
                "Unknown/background images for Brand classifier are low compared with known brand images; "
                "brand labels may look overconfident on unknown or non-target objects."
            )
        low_classes = {
            name: count
            for name, count in stats.get("class_counts", {}).items()
            if count < 30
        }
        if low_classes:
            errors.append(
                "Classification classes need at least 30 images each: "
                + ", ".join(f"{name}={count}" for name, count in low_classes.items())
            )
        class_counts = list(stats.get("class_counts", {}).values())
        if class_counts and min(class_counts) > 0 and max(class_counts) / min(class_counts) > 3:
            warnings.append("Classification dataset is imbalanced by more than 3x between classes.")
    else:
        errors.append("Unknown train type.")
    disk = shutil.disk_usage(PROJECT_ROOT)
    checks.append(
        {
            "label": "Disk space",
            "status": "pass" if disk.free >= 5 * 1024**3 else "fail",
            "detail": f"{disk.free / 1024**3:.1f} GB free",
        }
    )
    if disk.free < 5 * 1024**3:
        errors.append("Free disk space is below 5 GB; training could stop mid-run and corrupt outputs.")
    return {
        "ok": not errors,
        "train_type": train_type,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "class_report": class_report if train_type == "airbag_detector" else None,
        "combined_quality": combined_quality if train_type == "airbag_detector" else None,
        "background_visual_assist": visual_risk if train_type == "airbag_detector" else None,
        "checked_at": now_iso(),
    }


def train_preflight(train_type: str) -> dict:
    return cached_analysis(
        f"preflight:{train_type}",
        lambda: _compute_train_preflight(train_type),
    )


# ===== Model registry, run history และ train process state =====
# ก่อน activate/copy weight ต้อง inspect class/type เสมอ เพื่อกัน COCO detector หรือ
# classifier ถูกนำมาแทน Airbag detector; TRAIN_STATE ถูกแก้ภายใต้ TRAIN_LOCK เท่านั้น
def model_registry() -> list[dict]:
    config = read_project_config()
    # Inspection uses the locator resolver; the legacy detector selection can
    # remain configured after switching to Normal Reference inspection.
    locator = active_locator_model_path(config)
    active_detector = str(locator) if locator else ""
    active_brand = str(config.get("active_brand_model") or "")
    try:
        active_detector = str(Path(active_detector).resolve()) if active_detector else ""
    except Exception:
        active_detector = ""
    try:
        active_brand = str(Path(active_brand).resolve()) if active_brand else ""
    except Exception:
        active_brand = ""
    candidates: list[tuple[str, Path, str]] = []
    for path in sorted((PROJECT_ROOT / "models").glob("*.pt")):
        candidates.append(("Detection", path, "models"))
    for path in sorted((BRAND_CODE_DIR / "04_trained_models").glob("*.pt")):
        candidates.append(("Classification", path, "brand_trained_models"))
    seen = set()
    models = []
    for model_type, path, source in candidates:
        resolved = str(path.resolve())
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        stat = path.stat()
        is_active = resolved == active_detector or resolved == active_brand
        if not active_brand and path.name == "airbag_brand_best.pt":
            is_active = True
        try:
            metadata = read_json(path.with_suffix(".metadata.json"), {})
        except Exception:
            metadata = {}
        try:
            background_evaluation = read_json(path.with_suffix(".background_eval.json"), {})
        except Exception:
            background_evaluation = {}
        models.append(
            {
                "name": path.name,
                "type": model_type,
                "source": source,
                "path": str(path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "active": is_active,
                "classes": metadata.get("classes") or [],
                "validation": metadata.get("validation") or {},
                "background_evaluation": background_evaluation or (metadata.get("validation") or {}).get("background_evaluation") or {},
            }
        )
    return sorted(models, key=lambda item: item["updated_at"], reverse=True)


def read_log_for_metrics(log_path: str, max_bytes: int = 2_000_000) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    with path.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(max(0, size - max_bytes))
        data = file.read()
    return data.decode("utf-8", errors="replace")


def parse_loss_history(log_path: str) -> list[dict]:
    text = read_log_for_metrics(log_path)
    if not text:
        return []
    clean_text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    history: dict[int, dict] = {}
    pattern = re.compile(
        r"^\s*(\d+)\s*/\s*(\d+)\s+\S+\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        re.MULTILINE,
    )
    for match in pattern.finditer(clean_text):
        epoch = int(match.group(1))
        total_epochs = int(match.group(2))
        box_loss = float(match.group(3))
        cls_loss = float(match.group(4))
        dfl_loss = float(match.group(5))
        history[epoch] = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "box_loss": round(box_loss, 6),
            "cls_loss": round(cls_loss, 6),
            "dfl_loss": round(dfl_loss, 6),
            "total_loss": round(box_loss + cls_loss + dfl_loss, 6),
        }
    return [history[epoch] for epoch in sorted(history)]


def parse_duration_token(value: str) -> float:
    """แปลงเวลาจาก log ของ Ultralytics เช่น 51.8s, 5:14, 1:02:03 เป็นวินาที."""
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("s"):
        try:
            return float(text[:-1])
        except ValueError:
            return 0.0
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return 0.0
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return 0.0


def training_log_started_at(log_path: str | Path) -> float:
    """Read the start time embedded in a train-log filename, never its file mtime."""
    match = re.match(r"(\d{8}_\d{6})_", Path(log_path).name)
    if not match:
        return 0.0
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return 0.0


def train_log_sort_key(log_path: Path) -> tuple[int, float, str]:
    """Keep log history in run order even after copying the project to another PC."""
    started_at = training_log_started_at(log_path)
    return (1 if started_at else 0, started_at, log_path.name)


def training_progress_line_parts(line: str) -> dict | None:
    """Parse an Ultralytics *training* tqdm row and reject validation tqdm rows."""
    epoch_match = re.search(r"^\s*(\d+)\s*/\s*(\d+)\b", line)
    pair_matches = list(re.finditer(r"(\d+)\s*/\s*(\d+)", line))
    percent_match = re.search(r":\s*(\d{1,3})%", line)
    if not epoch_match or len(pair_matches) < 2 or not percent_match:
        return None
    return {
        "epoch": int(epoch_match.group(1)),
        "total_epochs": int(epoch_match.group(2)),
        "batch": int(pair_matches[-1].group(1)),
        "total_batches": int(pair_matches[-1].group(2)),
        "epoch_percent": float(percent_match.group(1)),
    }


def logged_training_duration_summary(log_path: str) -> tuple[float, int]:
    """Return logged training seconds and completed epochs without relying on file metadata.

    Detector logs have one 100% tqdm row per epoch.  Classifier logs provide a
    final ``N epochs completed in X hours`` line.  Both survive a file copy,
    unlike ``st_mtime``.
    """
    text = read_log_for_metrics(log_path)
    if not text:
        return 0.0, 0
    clean_text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    completed_epoch_seconds: dict[int, float] = {}
    final_duration_pattern = re.compile(r"(?:^|\s)(\d+:\d+(?::\d+)?|[\d.]+s)\s*$")
    for line in clean_text.splitlines():
        parts = training_progress_line_parts(line.strip())
        if not parts or parts["epoch_percent"] < 100:
            continue
        duration_match = final_duration_pattern.search(line)
        duration = parse_duration_token(duration_match.group(1)) if duration_match else 0.0
        if duration > 0:
            completed_epoch_seconds[parts["epoch"]] = duration
    if completed_epoch_seconds:
        return sum(completed_epoch_seconds.values()), len(completed_epoch_seconds)

    completed_match = re.findall(
        r"\b(\d+)\s+epochs?\s+completed\s+in\s+([\d.]+)\s+hours?\b",
        clean_text,
        flags=re.IGNORECASE,
    )
    if completed_match:
        epochs_text, hours_text = completed_match[-1]
        try:
            epochs = int(epochs_text)
            seconds = float(hours_text) * 3600
            if epochs > 0 and seconds > 0:
                return seconds, epochs
        except ValueError:
            pass
    return 0.0, 0


def parse_training_progress(log_path: str, started_at: float | None = None, status: str = "") -> dict:
    """อ่าน progress ล่าสุดจาก log จริงเพื่อคำนวณ ETA ให้แม่นกว่าสูตร elapsed/percent ฝั่งหน้าเว็บ."""
    text = read_log_for_metrics(log_path)
    if not text:
        return {}
    clean_text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    progress_line = ""
    progress_parts: dict | None = None
    for line in reversed(lines):
        parts = training_progress_line_parts(line)
        if parts:
            progress_line = line
            progress_parts = parts
            break
    if not progress_line or not progress_parts:
        return {}

    epoch = progress_parts["epoch"]
    total_epochs = progress_parts["total_epochs"]
    batch = progress_parts["batch"]
    total_batches = progress_parts["total_batches"]
    epoch_percent = progress_parts["epoch_percent"]
    if not epoch_percent and batch and total_batches:
        epoch_percent = (batch / total_batches) * 100
    speed_match = re.search(r"([\d.]+)\s*(s/it|it/s|ms/it)", progress_line)
    seconds_per_batch = 0.0
    speed_text = speed_match.group(0) if speed_match else ""
    if speed_match:
        speed_value = float(speed_match.group(1))
        speed_unit = speed_match.group(2)
        if speed_unit == "s/it":
            seconds_per_batch = speed_value
        elif speed_unit == "it/s" and speed_value > 0:
            seconds_per_batch = 1.0 / speed_value
        elif speed_unit == "ms/it":
            seconds_per_batch = speed_value / 1000.0
    time_match = re.search(r"(\d+:\d+(?::\d+)?|[\d.]+s)\s*<\s*(\d+:\d+(?::\d+)?|[\d.]+s)", progress_line)
    epoch_elapsed_from_log = parse_duration_token(time_match.group(1)) if time_match else 0.0
    epoch_remaining_from_log = parse_duration_token(time_match.group(2)) if time_match else 0.0

    epoch_fraction = max(0.0, min(1.0, (batch / total_batches) if total_batches else epoch_percent / 100))
    completed_epoch_units = max(0.0, (epoch - 1) + epoch_fraction)
    overall_percent = (completed_epoch_units / total_epochs) * 100 if total_epochs else 0.0
    elapsed_seconds = 0.0
    if started_at:
        elapsed_seconds = max(0.0, time.time() - float(started_at))
    else:
        elapsed_seconds, _ = logged_training_duration_summary(log_path)

    logged_elapsed_seconds, logged_epochs = logged_training_duration_summary(log_path)
    observed_seconds_per_epoch = (
        logged_elapsed_seconds / logged_epochs if logged_elapsed_seconds and logged_epochs else 0.0
    )
    history = parse_loss_history(log_path)
    completed_epoch_time = observed_seconds_per_epoch
    if not completed_epoch_time and history and elapsed_seconds:
        completed_epoch_time = elapsed_seconds / max(1, len(history))

    current_epoch_remaining = 0.0
    if seconds_per_batch and total_batches and batch:
        current_epoch_remaining = max(0.0, (total_batches - batch) * seconds_per_batch)
    elif epoch_remaining_from_log:
        current_epoch_remaining = epoch_remaining_from_log

    future_epochs = max(0, total_epochs - epoch)
    future_epoch_seconds = completed_epoch_time or (
        (total_batches * seconds_per_batch) if total_batches and seconds_per_batch else 0.0
    )
    remaining_seconds = current_epoch_remaining + future_epochs * future_epoch_seconds
    if status == "finished" or overall_percent >= 100:
        remaining_seconds = 0.0

    confidence = "low"
    if seconds_per_batch and total_batches and completed_epoch_units >= 1:
        confidence = "high"
    elif seconds_per_batch and total_batches:
        confidence = "medium"
    elif completed_epoch_time:
        confidence = "medium"

    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "epoch_percent": round(max(0.0, min(100.0, epoch_percent)), 2),
        "overall_percent": round(max(0.0, min(100.0, overall_percent)), 2),
        "seconds_per_batch": round(seconds_per_batch, 4) if seconds_per_batch else 0.0,
        "seconds_per_epoch": round(future_epoch_seconds, 2) if future_epoch_seconds else 0.0,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "remaining_seconds": round(max(0.0, remaining_seconds), 2),
        "epoch_remaining_seconds": round(max(0.0, current_epoch_remaining), 2),
        "speed_text": speed_text,
        "line": progress_line[-220:],
        "confidence": confidence,
    }


def latest_detector_results_csv() -> Path | None:
    """Locate the CSV produced by the newest detector run without trusting old absolute paths in logs."""
    if not PROJECT_ROOT.exists():
        return None
    latest_log = latest_train_log_path()
    run_name = ""
    if latest_log:
        log_text = read_log_for_metrics(str(latest_log), max_bytes=400_000)
        matches = re.findall(r"\bsave_dir=([^,\r\n]+)", log_text)
        if matches:
            run_name = Path(matches[-1].strip()).name

    candidates = [
        path
        for path in PROJECT_ROOT.glob("**/results.csv")
        if path.is_file() and "detect" in {part.lower() for part in path.parts}
    ]
    if not candidates:
        return None
    if run_name:
        matching_run = [path for path in candidates if path.parent.name == run_name]
        if matching_run:
            candidates = matching_run
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_detector_results_csv(results_csv: Path) -> list[dict]:
    """Return a bounded, numeric history from Ultralytics results.csv.

    Rows are intentionally parsed defensively: a partially-written CSV during a
    live run is common and must not make the Results page fail.
    """
    rows: list[dict] = []
    try:
        with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw_row in csv.DictReader(handle):
                row: dict[str, float | int | None] = {}
                for key, raw_value in raw_row.items():
                    if not key:
                        continue
                    value = str(raw_value or "").strip()
                    if not value:
                        row[key.strip()] = None
                        continue
                    try:
                        number = float(value)
                        row[key.strip()] = int(number) if key.strip() == "epoch" and number.is_integer() else number
                    except ValueError:
                        row[key.strip()] = None
                if isinstance(row.get("epoch"), int):
                    rows.append(row)
    except (OSError, csv.Error):
        return []
    return rows[-240:]


def latest_detector_evaluation() -> dict:
    """Build one stable, frontend-friendly snapshot from the latest YOLO detector run."""
    results_csv = latest_detector_results_csv()
    if not results_csv:
        return {"ok": False, "error": "No detector results.csv is available yet."}
    rows = read_detector_results_csv(results_csv)
    if not rows:
        return {"ok": False, "error": "The latest detector results.csv has no complete metric rows yet."}

    latest = rows[-1]
    precision = latest.get("metrics/precision(B)")
    recall = latest.get("metrics/recall(B)")
    map50 = latest.get("metrics/mAP50(B)")
    map50_95 = latest.get("metrics/mAP50-95(B)")
    f1 = None
    if isinstance(precision, (float, int)) and isinstance(recall, (float, int)) and precision + recall:
        f1 = round((2 * precision * recall) / (precision + recall), 6)

    active_model = latest_detector_model_path()
    relative_run = results_csv.parent.relative_to(PROJECT_ROOT).as_posix()
    available_assets = {
        name: name in {path.name for path in results_csv.parent.iterdir() if path.is_file()}
        for name in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "val_batch0_pred.jpg")
    }
    metric_keys = (
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
        "metrics/mAP50-95(B)", "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
    )
    history = []
    for row in rows:
        history.append({"epoch": row.get("epoch"), **{key: row.get(key) for key in metric_keys}})
    return {
        "ok": True,
        "run": {
            "id": relative_run,
            "name": results_csv.parent.name,
            "updated_at": datetime.fromtimestamp(results_csv.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "epochs": latest.get("epoch"),
            "model": active_model.name if active_model else "Detector model unavailable",
            "classes": list(read_project_config().get("target_classes") or []),
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "f1": f1,
        },
        "history": history,
        "assets": available_assets,
    }


def latest_train_log_path() -> Path | None:
    if not TRAIN_LOG_DIR.exists():
        return None
    logs = [path for path in TRAIN_LOG_DIR.glob("*.log") if path.is_file()]
    if not logs:
        return None
    return max(logs, key=train_log_sort_key)


def infer_train_type_from_log_path(log_path: str) -> str:
    name = Path(log_path).name.lower()
    if "brand_classifier" in name:
        return "brand_classifier"
    if "airbag_detector" in name:
        return "airbag_detector"
    return ""


def infer_train_status_from_log(log_path: str) -> dict:
    path = Path(log_path) if log_path else None
    if not path or not path.exists():
        return {"status": "idle", "returncode": None, "status_detail": "ยังไม่มี log สำหรับงาน train"}
    text = read_log_for_metrics(str(path))
    clean_text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    lower_text = clean_text.lower()
    modified_at = path.stat().st_mtime
    if "[training stopped]" in lower_text or "user requested stop" in lower_text:
        return {"status": "stopped", "returncode": -1, "finished_at": modified_at, "status_detail": "ผู้ใช้สั่งหยุด train จากหน้าเว็บ"}
    complete_markers = ["training complete:", "copied trained model to:", "training complete"]
    if any(marker in lower_text for marker in complete_markers):
        early_stopped = "earlystopping:" in lower_text or "training stopped early" in lower_text
        if early_stopped:
            completed_match = re.findall(r"(\d+)\s+epochs?\s+completed", lower_text)
            completed_epochs = int(completed_match[-1]) if completed_match else None
            detail = "Training เสร็จด้วย Early Stopping และบันทึกโมเดลที่ดีที่สุดแล้ว"
            if completed_epochs:
                detail = f"Training เสร็จด้วย Early Stopping ที่ epoch {completed_epochs} และบันทึกโมเดลที่ดีที่สุดแล้ว"
            return {
                "status": "finished",
                "returncode": 0,
                "finished_at": modified_at,
                "status_detail": detail,
                "early_stopped": True,
                "completion_reason": "early_stopping",
            }
        return {
            "status": "finished",
            "returncode": 0,
            "finished_at": modified_at,
            "status_detail": "Training เสร็จสมบูรณ์และบันทึกโมเดลผลลัพธ์แล้ว",
            "early_stopped": False,
            "completion_reason": "completed",
        }
    failure_markers = [
        "traceback (most recent call last):",
        "runtimeerror:",
        "valueerror:",
        "filenotfounderror:",
        "modulenotfounderror:",
        "refusing to copy a model whose classes do not match the target",
        "training failed",
        "error:",
    ]
    if any(marker in lower_text for marker in failure_markers):
        return {"status": "failed", "returncode": 1, "finished_at": modified_at, "status_detail": "พบ error ใน log งาน train"}
    if time.time() - modified_at <= 30:
        return {"status": "training", "returncode": None, "finished_at": None, "status_detail": "log ยังถูกอัปเดตล่าสุด จึงถือว่างาน train อาจกำลังทำงานอยู่"}
    return {"status": "idle", "returncode": None, "finished_at": modified_at, "status_detail": "log ล่าสุดยังไม่มีหลักฐานว่าเสร็จ ล้มเหลว หรือกำลังรัน"}


def extract_model_path_from_log(log_path: str) -> str:
    text = read_log_for_metrics(log_path)
    matches = re.findall(r"Copied (?:trained |brand )?model to:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"Copied trained model to:\s*(.+)", text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def train_run_records(limit: int = 30) -> list[dict]:
    if not TRAIN_LOG_DIR.exists():
        return []
    log_paths = sorted(TRAIN_LOG_DIR.glob("*.log"), key=train_log_sort_key, reverse=True)[:limit]
    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in log_paths
        if path.is_file()
    )
    cache_key = (int(limit), signature)
    with TRAIN_RUN_CACHE_LOCK:
        if TRAIN_RUN_CACHE.get("key") == cache_key:
            return list(TRAIN_RUN_CACHE.get("records") or [])
    records = []
    for log_path in log_paths:
        inferred = infer_train_status_from_log(str(log_path))
        history = parse_loss_history(str(log_path))
        progress = parse_training_progress(str(log_path), status=inferred.get("status", "idle"))
        model_path = extract_model_path_from_log(str(log_path))
        background_evaluation = read_json(Path(model_path).with_suffix(".background_eval.json"), {}) if model_path and Path(model_path).exists() else {}
        status = inferred.get("status", "idle")
        overall_percent = float(progress.get("overall_percent") or 0)
        if status == "finished":
            overall_percent = 100.0
        records.append(
            {
                "id": log_path.name,
                "log_path": str(log_path),
                "train_type": infer_train_type_from_log_path(str(log_path)),
                "status": status,
                "status_detail": inferred.get("status_detail", ""),
                "early_stopped": bool(inferred.get("early_stopped", False)),
                "completion_reason": inferred.get("completion_reason", ""),
                "updated_at": datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "loss": history[-1]["total_loss"] if history else None,
                "epochs": len(history),
                "total_epochs": int(progress.get("total_epochs") or len(history)),
                "progress": round(max(0.0, min(100.0, overall_percent)), 2),
                "elapsed_seconds": round(float(progress.get("elapsed_seconds") or 0), 2),
                "model_path": model_path,
                "background_evaluation": background_evaluation,
            }
        )
    with TRAIN_RUN_CACHE_LOCK:
        TRAIN_RUN_CACHE["key"] = cache_key
        TRAIN_RUN_CACHE["records"] = list(records)
    return records


def normal_reference_run_records(limit: int = 30) -> list[dict]:
    """Return saved Normal Reference models as first-class training history."""
    if not NORMAL_REFERENCE_MODEL_DIR.exists():
        return []
    config = read_project_config()
    active_text = str((config.get("normal_reference") or {}).get("active_model") or "").strip()
    active_path = Path(active_text).resolve() if active_text and Path(active_text).is_file() else None
    model_paths = sorted(
        NORMAL_REFERENCE_MODEL_DIR.glob("*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    records = []
    for model_path in model_paths:
        modified_at = datetime.fromtimestamp(model_path.stat().st_mtime)
        is_active = bool(active_path and model_path.resolve() == active_path)
        engine = "Deep Patch" if "deep_patch" in model_path.stem.lower() else "Statistical Reference"
        comparison = read_json(model_path.with_suffix(".comparison.json"), {})
        rejected = comparison.get("decision") == "rejected"
        comparison_label = " · ต้องตรวจแก้" if rejected else " · รอทดสอบใน Jobs" if comparison and not comparison.get("activated") else ""
        records.append(
            {
                "id": model_path.name,
                "log_path": "",
                "train_type": "normal_reference",
                "module": "normal_reference",
                "status": "finished",
                "status_detail": f"{engine} พร้อมใช้งาน" + (" · โมเดลที่ใช้งานอยู่" if is_active else comparison_label),
                "early_stopped": False,
                "completion_reason": "completed",
                "updated_at": modified_at.strftime("%Y-%m-%d %H:%M:%S"),
                "loss": None,
                "epochs": 0,
                "total_epochs": 0,
                "progress": 100.0,
                "elapsed_seconds": 0.0,
                "model_path": str(model_path),
                "active": is_active,
                "candidate_rejected": rejected,
                "candidate_comparison": comparison,
                "engine": "deep_patch" if "deep_patch" in model_path.stem.lower() else "statistical_reference",
            }
        )
    return records


def read_log_tail(log_path: str, max_chars: int = 6000) -> str:
    """อ่าน log ช่วงท้ายเพื่อแสดงบนเว็บโดยไม่โหลดทั้งไฟล์ใหญ่."""
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    max_bytes = max(max_chars * 4, 8192)
    with path.open("rb") as file:
        file.seek(0, 2)
        size = file.tell()
        file.seek(max(0, size - max_bytes))
        data = file.read()
    text = data.decode("utf-8", errors="replace")
    return text[-max_chars:]


def close_train_log_handle_locked() -> None:
    """ปิดไฟล์ log ของงาน train ปัจจุบันแบบไม่ทำให้ backend ล้มถ้า handle ปิดไปแล้ว."""
    global TRAIN_LOG_HANDLE
    if TRAIN_LOG_HANDLE is None:
        return
    try:
        TRAIN_LOG_HANDLE.flush()
        TRAIN_LOG_HANDLE.close()
    except Exception:
        pass
    TRAIN_LOG_HANDLE = None


def inspect_detector_model(model_path: Path, expected_classes: list[str] | None = None) -> dict:
    """Load one detector once and verify its embedded class contract."""
    try:
        model = YOLO(str(model_path))
        names = detector.model_names_list(model.names)
    except Exception as exc:
        return {
            "ok": False,
            "path": str(model_path),
            "classes": [],
            "expected_classes": expected_classes or [],
            "error": str(exc),
            "checked_at": now_iso(),
        }
    expected = expected_classes or []
    actual_keys = [normalized_class_key(name) for name in names]
    expected_keys = [normalized_class_key(name) for name in expected]
    matches = not expected or actual_keys == expected_keys
    return {
        "ok": matches,
        "path": str(model_path),
        "classes": names,
        "expected_classes": expected,
        "error": "" if matches else "Model classes do not match the project class order.",
        "checked_at": now_iso(),
    }


def evaluate_detector_backgrounds(model_path: Path, config: dict) -> dict:
    """วัด False Positive บน Background ที่ไม่เคยเข้า Train ก่อนอนุมัติโมเดล."""
    model = YOLO(str(model_path))
    primary_key = normalized_class_key(str(config.get("primary_class") or ""))
    primary_threshold = float(config.get("primary_confidence") or 0.30)
    secondary_threshold = float(config.get("secondary_confidence") or 0.40)
    predict_threshold = max(0.01, min(primary_threshold, secondary_threshold))

    split_reports = {}
    visual_assist_reports = {}
    for split in ("eval_valid", "eval_test"):
        images = split_background_images(AIRBAG_BACKGROUND_DIR, split)
        false_positive_images = 0
        detections = 0
        max_confidence = 0.0
        examples = []
        visual_hit_images = 0
        visual_examples = []
        if images:
            results = model.predict(
                source=[str(path) for path in images],
                conf=predict_threshold,
                imgsz=WEB_AIRBAG_IMAGE_SIZE,
                device=detector.DEVICE,
                verbose=False,
                stream=False,
            )
            for image_path, result in zip(images, results):
                image_hits = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = str(result.names.get(class_id, class_id))
                        threshold = primary_threshold if normalized_class_key(class_name) == primary_key else secondary_threshold
                        if confidence < threshold:
                            continue
                        detections += 1
                        max_confidence = max(max_confidence, confidence)
                        image_hits.append({"class": class_name, "confidence": round(confidence, 5)})
                if image_hits:
                    false_positive_images += 1
                    if len(examples) < 10:
                        examples.append({"image": str(image_path), "detections": image_hits})
            for image_path in images:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                visual_candidates = detector.find_visual_airbag_candidates(frame, [])
                visual_selected = detector.select_airbag_detections(visual_candidates, frame.shape, []) if visual_candidates else []
                if not visual_selected:
                    continue
                visual_hit_images += 1
                if len(visual_examples) < 10:
                    visual_examples.append(
                        {
                            "image": str(image_path),
                            "detections": [
                                {"box": [int(value) for value in xyxy], "confidence": round(float(confidence), 5)}
                                for xyxy, confidence in visual_selected
                            ],
                        }
                    )
        image_count = len(images)
        rate = false_positive_images / image_count if image_count else 1.0
        split_reports[split] = {
            "images": image_count,
            "false_positive_images": false_positive_images,
            "false_positive_rate": round(rate, 5),
            "detections": detections,
            "max_confidence": round(max_confidence, 5),
            "examples": examples,
        }
        visual_rate = visual_hit_images / image_count if image_count else 1.0
        visual_assist_reports[split] = {
            "images": image_count,
            "hit_images": visual_hit_images,
            "hit_rate": round(visual_rate, 5),
            "examples": visual_examples,
        }

    enough_images = all(split_reports[name]["images"] >= MIN_BACKGROUND_EVAL_IMAGES for name in split_reports)
    rates_ok = all(split_reports[name]["false_positive_rate"] <= BACKGROUND_MAX_FALSE_POSITIVE_RATE for name in split_reports)
    visual_assist_ok = all(visual_assist_reports[name]["hit_images"] == 0 for name in visual_assist_reports)
    return {
        "ok": enough_images and rates_ok and visual_assist_ok,
        "minimum_images_per_split": MIN_BACKGROUND_EVAL_IMAGES,
        "maximum_false_positive_rate": BACKGROUND_MAX_FALSE_POSITIVE_RATE,
        "thresholds": {"primary": primary_threshold, "secondary": secondary_threshold},
        "splits": split_reports,
        "visual_assist": visual_assist_reports,
        "error": "" if enough_images and rates_ok and visual_assist_ok else (
            "Background evaluation does not have enough images."
            if not enough_images
            else (
                "False-positive rate exceeds the activation limit."
                if not rates_ok
                else "Visual Assist proposes Airbag boxes on background images."
            )
        ),
        "checked_at": now_iso(),
    }


def inspect_classifier_model(model_path: Path) -> dict:
    try:
        model = YOLO(str(model_path))
        names = detector.model_names_list(model.names)
        task = str(getattr(model, "task", "") or "")
    except Exception as exc:
        return {"ok": False, "path": str(model_path), "classes": [], "task": "", "error": str(exc), "checked_at": now_iso()}
    valid = task == "classify" and len(names) >= 2
    return {
        "ok": valid,
        "path": str(model_path),
        "classes": names,
        "task": task,
        "error": "" if valid else "Model is not a usable multi-class classifier.",
        "checked_at": now_iso(),
    }


def write_model_metadata(model_path: Path, payload: dict) -> None:
    metadata_path = model_path.with_suffix(".metadata.json")
    write_json(metadata_path, payload)


def finalize_successful_train_config_locked() -> None:
    """Promote the completed training artifact into the project config."""
    train_type = str(TRAIN_STATE.get("train_type") or "")
    log_path = str(TRAIN_STATE.get("log_path") or "")
    update_payload = {
        "last_train_type": train_type,
        "last_train_log": log_path,
        "last_train_status": str(TRAIN_STATE.get("status") or ""),
    }
    if train_type == "airbag_detector":
        model_path = extract_model_path_from_log(log_path)
        if not model_path:
            latest_model = latest_detector_model_path()
            model_path = str(latest_model) if latest_model else ""
        if model_path and Path(model_path).exists():
            config = read_project_config()
            train_project = TRAIN_STATE.get("project") or config
            validation = inspect_detector_model(
                Path(model_path),
                list(train_project.get("target_classes") or []),
            )
            contract_ok = bool(validation["ok"])
            validation["model_contract_ok"] = contract_ok
            if contract_ok:
                try:
                    background_evaluation = evaluate_detector_backgrounds(Path(model_path), config)
                except Exception as exc:
                    background_evaluation = {
                        "ok": False,
                        "error": f"Background evaluation failed: {exc}",
                        "checked_at": now_iso(),
                    }
            else:
                background_evaluation = {
                    "ok": False,
                    "error": "Background evaluation was skipped because the model class contract failed.",
                    "checked_at": now_iso(),
                }
            validation["background_evaluation"] = background_evaluation
            validation["ok"] = contract_ok and bool(background_evaluation.get("ok"))
            if not validation["ok"] and not validation.get("error"):
                validation["error"] = str(background_evaluation.get("error") or "Model activation gate failed.")
            write_json(Path(model_path).with_suffix(".background_eval.json"), background_evaluation)
            update_payload["last_model_validation"] = validation
            if validation["ok"]:
                previous_model = str(config.get("active_detector_model") or "")
                if previous_model and Path(previous_model).resolve() != Path(model_path).resolve():
                    update_payload["previous_detector_model"] = previous_model
                update_payload["active_detector_model"] = model_path
                write_model_metadata(
                    Path(model_path),
                    {
                        "model_path": str(Path(model_path).resolve()),
                        "classes": validation["classes"],
                        "project_name": config.get("project_name"),
                        "train_log": log_path,
                        "previous_detector_model": previous_model,
                        "activated_at": now_iso(),
                        "validation": validation,
                    },
                )
                clear_shared_model_cache()
            else:
                update_payload["last_train_status"] = "finished_not_activated"
    elif train_type == "brand_classifier":
        model_path = extract_model_path_from_log(log_path)
        if model_path and Path(model_path).exists():
            validation = inspect_classifier_model(Path(model_path))
            update_payload["last_model_validation"] = validation
            if validation["ok"]:
                update_payload["active_brand_model"] = model_path
                write_model_metadata(
                    Path(model_path),
                    {
                        "model_path": str(Path(model_path).resolve()),
                        "classes": validation["classes"],
                        "project_name": read_project_config().get("project_name"),
                        "train_log": log_path,
                        "activated_at": now_iso(),
                        "validation": validation,
                    },
                )
                clear_shared_model_cache()
            else:
                update_payload["last_train_status"] = "finished_not_activated"
    write_project_config(update_payload)


def refresh_train_state_locked() -> None:
    """อัปเดตสถานะ train process ถ้ามันจบแล้ว."""
    global TRAIN_PROCESS, TRAIN_LOG_HANDLE
    if TRAIN_PROCESS is None:
        return

    returncode = TRAIN_PROCESS.poll()
    if returncode is None:
        return

    close_train_log_handle_locked()

    TRAIN_STATE["running"] = False
    TRAIN_STATE["returncode"] = returncode
    TRAIN_STATE["finished_at"] = time.time()
    if TRAIN_STATE.get("stop_requested_at"):
        TRAIN_STATE["status"] = "stopped"
    else:
        TRAIN_STATE["status"] = "finished" if returncode == 0 else "failed"
    if TRAIN_STATE["status"] == "finished":
        finalize_successful_train_config_locked()
    TRAIN_PROCESS = None


def wait_for_train_completion(process):
    """Keep project ownership until the existing child and finalization finish."""
    process.wait()
    with TRAIN_LOCK:
        if TRAIN_PROCESS is process:
            refresh_train_state_locked()


def stop_train_process_locked() -> tuple[bool, str]:
    """หยุด process train ที่กำลังทำงานอยู่ และฆ่า process ลูกบน Windows ให้ครบ."""
    global TRAIN_PROCESS
    refresh_train_state_locked()
    if TRAIN_PROCESS is None:
        return False, "No training process is running."

    process = TRAIN_PROCESS
    TRAIN_STATE["status"] = "stopping"
    TRAIN_STATE["stop_requested_at"] = time.time()
    if TRAIN_LOG_HANDLE is not None:
        try:
            TRAIN_LOG_HANDLE.write("\n\n[Training stopped] User requested stop from Training page.\n")
            TRAIN_LOG_HANDLE.flush()
        except Exception:
            pass

    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=8,
            )
        else:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception as exc:
        return False, f"Stop training failed: {exc}"

    try:
        returncode = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False, "Training process did not stop in time."

    close_train_log_handle_locked()
    TRAIN_PROCESS = None
    TRAIN_STATE["running"] = False
    TRAIN_STATE["status"] = "stopped"
    TRAIN_STATE["finished_at"] = time.time()
    TRAIN_STATE["returncode"] = returncode
    return True, "Training stopped."


# ===== Shared model cache =====
# กล้อง, Auto Label และงานวิเคราะห์ใช้โมเดลร่วมกันเพื่อลด RAM/เวลาโหลด เมื่อเปลี่ยน
# active weight ต้อง clear_shared_model_cache ก่อน request ถัดไปจึงจะเห็นโมเดลใหม่
def load_airbag_model() -> YOLO:
    """โหลด Airbag model จากลำดับเดียวกับไฟล์กล้องเดิม."""
    skipped = []
    configured_model = latest_detector_model_path()
    dynamic_candidates = sorted(
        MODEL_DIR.glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidate_paths = []
    seen_paths = set()
    preferred_paths = [configured_model] if configured_model else []
    for candidate_path in [*preferred_paths, *dynamic_candidates, *detector.MODEL_CANDIDATES]:
        if candidate_path is None:
            continue
        resolved = candidate_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_paths.append(candidate_path)

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue

        model = YOLO(str(candidate_path))
        if detector.model_names_are_inspection_detector(model.names):
            model.to(detector.DEVICE)
            model.fuse()
            print(f"Web app loaded inspection detector model: {candidate_path}")
            return model

        skipped.append((candidate_path, model.names))

    checked = "\n".join(f"  - {path}" for path in candidate_paths)
    skipped_text = "\n".join(f"  - skipped {path}: {names}" for path, names in skipped)
    raise FileNotFoundError(
        "ไม่พบ Airbag model สำหรับเว็บ\n"
        f"Checked:\n{checked}\n\n{skipped_text}"
    )


def load_brand_model() -> YOLO | None:
    """โหลดโมเดลแยกแบรนด์ ถ้ามี."""
    config = read_project_config()
    configured = str(config.get("active_brand_model") or "").strip()
    dynamic = sorted(
        (BRAND_CODE_DIR / "04_trained_models").glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = ([Path(configured)] if configured else []) + dynamic + detector.BRAND_MODEL_CANDIDATES
    seen = set()
    for brand_model_path in candidates:
        resolved = brand_model_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not brand_model_path.exists():
            continue

        model = YOLO(str(brand_model_path))
        if getattr(model, "task", "") == "classify" and detector.model_names_are_brand_classifier(model.names):
            model.to(detector.DEVICE)
            print(f"Web app loaded brand model: {brand_model_path}")
            return model

    print("Web app warning: brand classifier not found.")
    return None


def load_general_model() -> YOLO | None:
    """โหลด COCO guard model ถ้า logic เดิมเปิดใช้งานอยู่."""
    if not (detector.USE_GENERAL_NOT_AIRBAG_DETECTOR or detector.USE_COCO_NEGATIVE_GUARD):
        return None

    general_model_path = next(
        (path for path in detector.GENERAL_MODEL_CANDIDATES if path.exists()),
        None,
    )
    if general_model_path is None:
        print("Web app warning: yolov8n.pt not found. COCO guard disabled.")
        return None

    model = YOLO(str(general_model_path))
    model.to(detector.DEVICE)
    model.fuse()
    print(f"Web app loaded general guard model: {general_model_path}")
    return model


def load_primary_fallback_model(active_model: YOLO) -> YOLO | None:
    """Load an Airbag-only model to recover the main object when multi-class misses it."""
    if detector.model_names_are_airbag_only(active_model.names):
        return None

    for fallback_path in detector.MODEL_CANDIDATES:
        if not fallback_path.exists():
            continue

        model = YOLO(str(fallback_path))
        if detector.model_names_are_airbag_only(model.names):
            model.to(detector.DEVICE)
            model.fuse()
            print(f"Web app loaded Airbag primary fallback model: {fallback_path}")
            return model

    print("Web app warning: Airbag primary fallback model not found.")
    return None


def get_shared_models() -> tuple[YOLO, YOLO | None, YOLO | None, YOLO | None]:
    """โหลดโมเดลครั้งเดียว แล้ว cache ไว้ใช้ซ้ำทุกครั้งที่กด Start."""
    global SHARED_MODELS, SHARED_MODEL_ERROR

    if SHARED_MODELS is not None:
        return SHARED_MODELS
    if SHARED_MODEL_ERROR:
        raise RuntimeError(SHARED_MODEL_ERROR)

    with MODEL_LOCK:
        if SHARED_MODELS is not None:
            return SHARED_MODELS
        if SHARED_MODEL_ERROR:
            raise RuntimeError(SHARED_MODEL_ERROR)

        try:
            airbag_model = load_airbag_model()
            SHARED_MODELS = (
                airbag_model,
                load_brand_model(),
                load_general_model(),
                load_primary_fallback_model(airbag_model),
            )
            return SHARED_MODELS
        except Exception as exc:
            SHARED_MODEL_ERROR = str(exc)
            raise


def get_shared_locator(config: dict | None = None) -> tuple[object, int]:
    """Load and cache the generic part locator used to crop the inspection ROI."""
    global SHARED_LOCATOR, SHARED_LOCATOR_KEY

    config = config or read_project_config()
    locator_path = active_locator_model_path(config)
    locator_key = (
        f"{locator_path.resolve()}:{locator_path.stat().st_mtime_ns}:{locator_path.stat().st_size}"
        if locator_path else ""
    )
    if SHARED_LOCATOR is not None and SHARED_LOCATOR_KEY == locator_key:
        return SHARED_LOCATOR

    with MODEL_LOCK:
        if SHARED_LOCATOR is not None and SHARED_LOCATOR_KEY == locator_key:
            return SHARED_LOCATOR
        locator_model, locator_class_id = detector.load_primary_object_locator(config)
        if locator_model is None or locator_class_id is None:
            raise RuntimeError(
                "ไม่พบโมเดลสำหรับหาตำแหน่งชิ้นงาน (Part locator) "
                "กรุณาเลือกหรือ Train โมเดลตรวจจับชิ้นงานก่อนเปิดกล้อง"
            )
        SHARED_LOCATOR = (locator_model, int(locator_class_id))
        SHARED_LOCATOR_KEY = locator_key
        return SHARED_LOCATOR


# ===== Camera inference worker =====
# มี worker ได้เพียงหนึ่งตัว ใช้ stop_event เป็นสัญญาณจบ และต้อง release กล้องใน finally
# การเพิ่มงานหนักใน loop จะทำให้ภาพหน้าเว็บกระตุก จึงแยก general detection เป็นช่วงเวลา
def open_camera_like_detector(camera_index: int):
    """เปิดกล้องที่ใช้งานได้ โดยให้ index ที่ตั้งค่าไว้มีลำดับแรก."""
    started = time.perf_counter()
    cap, selected_camera_index = detector.open_available_camera(camera_index, max_index=10)
    elapsed = time.perf_counter() - started

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, detector.CAMERA_FPS)
    with STATE_LOCK:
        STATE["camera_index"] = selected_camera_index
    if selected_camera_index != camera_index:
        write_project_config({"camera_index": selected_camera_index})
    print(
        "Web app opened camera via airbag_webcam_detection.py "
        f"requested={camera_index} selected={selected_camera_index} open_s={elapsed:.3f}"
    )
    return cap, selected_camera_index


def refresh_general_detections(general_model, frame):
    """รัน COCO guard เหมือนไฟล์กล้องเดิม."""
    if general_model is None:
        return []

    general_result = general_model.predict(
        frame,
        conf=min(detector.GENERAL_CONFIDENCE_THRESHOLD, detector.COCO_NEGATIVE_GUARD_CONFIDENCE),
        imgsz=WEB_GENERAL_IMAGE_SIZE,
        max_det=12,
        device=detector.DEVICE,
        half=detector.USE_HALF,
        verbose=False,
    )[0]

    detections = []
    if general_result.boxes is None:
        return detections

    for box in general_result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = general_result.names.get(class_id, str(class_id))
        detections.append((detector.get_xyxy(box), class_name, confidence))
    return detections


def find_airbag_candidates(airbag_model, frame, airbag_confidence: float, debug_detection: bool):
    """Run the inspection detector and split main Airbag vs defect/feature detections."""
    predict_confidence = (
        detector.DEBUG_AIRBAG_CONFIDENCE_THRESHOLD
        if debug_detection
        else detector.RAW_AIRBAG_CONFIDENCE_THRESHOLD
    )
    result = airbag_model.predict(
        frame,
        conf=predict_confidence,
        imgsz=WEB_AIRBAG_IMAGE_SIZE,
        iou=detector.AIRBAG_NMS_IOU_THRESHOLD,
        max_det=detector.MAX_AIRBAG_DETECTIONS,
        device=detector.DEVICE,
        half=detector.USE_HALF,
        verbose=False,
    )[0]

    candidates = []
    secondary_detections = []
    best_raw_confidence = 0.0
    if result.boxes is None:
        return candidates, secondary_detections, best_raw_confidence

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names.get(class_id, str(class_id))
        xyxy = detector.get_xyxy(box)

        if not detector.is_airbag_class(class_id, class_name):
            if confidence >= detector.SECONDARY_CONFIDENCE_THRESHOLD or debug_detection:
                secondary_detections.append((xyxy, class_name, confidence))
            continue

        best_raw_confidence = max(best_raw_confidence, confidence)
        if confidence < airbag_confidence and not debug_detection:
            continue
        candidates.append((xyxy, confidence))

    return candidates, secondary_detections, best_raw_confidence


def draw_processed_result(
    frame,
    airbag_model,
    primary_fallback_model,
    brand_model,
    general_model,
    airbag_confidence: float,
    debug_detection: bool,
    frame_index: int,
    cached_general_detections,
    last_airbag_detections,
    missed_airbag_frames: int,
):
    """ประมวลผล 1 frame ด้วย logic เดิม แล้ววาดกรอบลงภาพ."""
    if general_model is not None and frame_index % detector.GENERAL_DETECT_EVERY_N_FRAMES == 0:
        cached_general_detections = refresh_general_detections(general_model, frame)

    raw_candidates, secondary_detections, best_raw_confidence = find_airbag_candidates(
        airbag_model,
        frame,
        airbag_confidence,
        debug_detection,
    )

    current_airbag_detections = detector.select_airbag_detections(
        raw_candidates,
        frame.shape,
        cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
    )
    used_visual_fallback = False
    if not current_airbag_detections and primary_fallback_model is not None:
        fallback_candidates, fallback_best_confidence = detector.find_airbag_only_candidates(
            primary_fallback_model,
            frame,
            airbag_confidence,
            debug_detection,
            image_size=WEB_AIRBAG_IMAGE_SIZE,
        )
        best_raw_confidence = max(best_raw_confidence, fallback_best_confidence)
        current_airbag_detections = detector.select_airbag_detections(
            fallback_candidates,
            frame.shape,
            cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
        )
    if not current_airbag_detections and detector.primary_supports_airbag_assists():
        visual_candidates = detector.find_visual_airbag_candidates(frame, secondary_detections)
        current_airbag_detections = detector.select_airbag_detections(
            visual_candidates,
            frame.shape,
            cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
        )
        used_visual_fallback = bool(current_airbag_detections)

    if current_airbag_detections:
        if used_visual_fallback:
            current_airbag_detections = [
                (xyxy, None) for xyxy, _ in current_airbag_detections
            ]
        last_airbag_detections = current_airbag_detections
        missed_airbag_frames = 0
        detections_to_draw = current_airbag_detections
    elif missed_airbag_frames < detector.AIRBAG_HOLD_FRAMES:
        missed_airbag_frames += 1
        detections_to_draw = last_airbag_detections
    else:
        missed_airbag_frames += 1
        detections_to_draw = []

    airbag_boxes = []
    for xyxy, confidence in detections_to_draw:
        brand_result = detector.classify_airbag_brand(brand_model, frame, xyxy)
        if detector.should_reject_by_brand(brand_result, xyxy, cached_general_detections):
            continue

        airbag_boxes.append(xyxy)
        detector.draw_detection(frame, xyxy, detector.ACTIVE_PRIMARY_CLASS, confidence)
        if brand_result is not None:
            brand_name, brand_confidence = brand_result
            detector.draw_brand_label(frame, xyxy, brand_name, brand_confidence)

    if debug_detection:
        color = (
            (0, 255, 0)
            if best_raw_confidence >= airbag_confidence
            else (0, 255, 255)
            if best_raw_confidence > 0
            else (0, 0, 255)
        )
        detector.draw_status(frame, f"Raw {detector.ACTIVE_PRIMARY_CLASS} confidence: {best_raw_confidence:.2f}", color)

    if detector.USE_GENERAL_NOT_AIRBAG_DETECTOR and general_model is not None:
        for xyxy, class_name, confidence in cached_general_detections:
            if detector.overlaps_airbag(xyxy, airbag_boxes):
                continue
            detector.draw_detection(frame, xyxy, class_name, confidence)

    secondary_detections = detector.filter_secondary_detections(
        secondary_detections,
        airbag_boxes,
        debug_detection=debug_detection,
    )
    for xyxy, class_name, confidence in secondary_detections:
        detector.draw_detection(frame, xyxy, class_name, confidence)

    return cached_general_detections, last_airbag_detections, missed_airbag_frames


def camera_worker(
    camera_index: int,
    airbag_confidence: float,
    debug_detection: bool,
    use_brand: bool,
) -> None:
    """worker หลักของเว็บ: อ่านกล้องให้ลื่น และให้ AI ทำงานแยกเป็นจังหวะ."""
    global CURRENT_CAP

    cap = None
    ai_thread: threading.Thread | None = None
    try:
        set_status("loading_models")
        airbag_model, brand_model, general_model, primary_fallback_model = get_shared_models()
        if not use_brand:
            brand_model = None

        set_status("opening_camera")
        set_latest_frame(make_status_frame("Opening camera", f"Camera index {camera_index}"))
        cap, camera_index = open_camera_like_detector(camera_index)
        with STATE_LOCK:
            CURRENT_CAP = cap

        # อ่าน frame ดิบทันที เพื่อให้หน้าเว็บเห็นภาพก่อน AI frame แรกเสร็จ
        set_status("warming_camera")
        first_frame = None
        for _ in range(20):
            if STOP_EVENT.is_set():
                return
            ok, frame = cap.read()
            if ok and frame is not None:
                first_frame = frame
                set_latest_frame(frame)
                with STATE_LOCK:
                    STATE["raw_frame_count"] += 1
                break
            time.sleep(0.05)

        if first_frame is None:
            raise RuntimeError("เปิดกล้องได้ แต่ยังอ่านภาพจากกล้องไม่ได้")

        set_status("running")

        raw_lock = threading.Lock()
        overlay_lock = threading.Lock()

        raw_frame_for_ai = first_frame.copy()
        raw_frame_seq = 1
        overlay_airbags = []
        overlay_brands = []
        overlay_general_detections = []
        overlay_debug_text = None

        cached_general_detections = []
        last_airbag_detections = []
        missed_airbag_frames = detector.AIRBAG_HOLD_FRAMES + 1
        ai_frame_index = 0
        fps = 0.0
        last_time = time.perf_counter()
        last_publish_time = 0.0
        min_publish_interval = 1.0 / WEB_DISPLAY_MAX_FPS

        def ai_detection_loop() -> None:
            """รัน YOLO แยกจาก loop อ่านกล้อง เพื่อลดอาการภาพค้างบนเว็บ."""
            nonlocal cached_general_detections
            nonlocal last_airbag_detections
            nonlocal missed_airbag_frames
            nonlocal ai_frame_index
            nonlocal overlay_airbags
            nonlocal overlay_brands
            nonlocal overlay_general_detections
            nonlocal overlay_debug_text

            last_processed_seq = -1
            last_ai_time = 0.0

            while not STOP_EVENT.is_set():
                with raw_lock:
                    current_seq = raw_frame_seq
                    if raw_frame_for_ai is None or current_seq == last_processed_seq:
                        frame_for_ai = None
                    else:
                        frame_for_ai = raw_frame_for_ai.copy()

                if frame_for_ai is None:
                    time.sleep(0.01)
                    continue

                now = time.perf_counter()
                if now - last_ai_time < WEB_AI_MIN_INTERVAL_SECONDS:
                    time.sleep(0.015)
                    continue

                last_processed_seq = current_seq
                last_ai_time = now

                try:
                    if (
                        general_model is not None
                        and ai_frame_index % WEB_GENERAL_DETECT_EVERY_N_AI_FRAMES == 0
                    ):
                        cached_general_detections = refresh_general_detections(general_model, frame_for_ai)

                    raw_candidates, secondary_detections, best_raw_confidence = find_airbag_candidates(
                        airbag_model,
                        frame_for_ai,
                        airbag_confidence,
                        debug_detection,
                    )

                    current_airbag_detections = detector.select_airbag_detections(
                        raw_candidates,
                        frame_for_ai.shape,
                        cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
                    )
                    used_visual_fallback = False
                    if not current_airbag_detections and primary_fallback_model is not None:
                        fallback_candidates, fallback_best_confidence = detector.find_airbag_only_candidates(
                            primary_fallback_model,
                            frame_for_ai,
                            airbag_confidence,
                            debug_detection,
                            image_size=WEB_AIRBAG_IMAGE_SIZE,
                        )
                        best_raw_confidence = max(best_raw_confidence, fallback_best_confidence)
                        current_airbag_detections = detector.select_airbag_detections(
                            fallback_candidates,
                            frame_for_ai.shape,
                            cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
                        )
                    if not current_airbag_detections and detector.primary_supports_airbag_assists():
                        visual_candidates = detector.find_visual_airbag_candidates(
                            frame_for_ai,
                            secondary_detections,
                        )
                        current_airbag_detections = detector.select_airbag_detections(
                            visual_candidates,
                            frame_for_ai.shape,
                            cached_general_detections if detector.USE_COCO_NEGATIVE_GUARD else [],
                        )
                        used_visual_fallback = bool(current_airbag_detections)

                    if current_airbag_detections:
                        if used_visual_fallback:
                            current_airbag_detections = [
                                (xyxy, None) for xyxy, _ in current_airbag_detections
                            ]
                        last_airbag_detections = current_airbag_detections
                        missed_airbag_frames = 0
                        detections_to_draw = current_airbag_detections
                    elif missed_airbag_frames < detector.AIRBAG_HOLD_FRAMES:
                        missed_airbag_frames += 1
                        detections_to_draw = last_airbag_detections
                    else:
                        missed_airbag_frames += 1
                        detections_to_draw = []

                    new_airbags = []
                    new_brands = []
                    airbag_boxes = []
                    for xyxy, confidence in detections_to_draw:
                        brand_result = detector.classify_airbag_brand(brand_model, frame_for_ai, xyxy)
                        if detector.should_reject_by_brand(
                            brand_result,
                            xyxy,
                            cached_general_detections,
                        ):
                            continue

                        airbag_boxes.append(xyxy)
                        new_airbags.append((xyxy, confidence))
                        if brand_result is not None:
                            brand_name, brand_confidence = brand_result
                            new_brands.append((xyxy, brand_name, brand_confidence))

                    new_general_detections = []
                    new_secondary_detections = detector.filter_secondary_detections(
                        secondary_detections,
                        airbag_boxes,
                        debug_detection=debug_detection,
                    )
                    if detector.USE_GENERAL_NOT_AIRBAG_DETECTOR and general_model is not None:
                        for xyxy, class_name, confidence in cached_general_detections:
                            if detector.overlaps_airbag(xyxy, airbag_boxes):
                                continue
                            new_general_detections.append((xyxy, class_name, confidence))

                    if debug_detection:
                        new_debug_text = best_raw_confidence
                    else:
                        new_debug_text = None

                    with overlay_lock:
                        overlay_airbags = new_airbags
                        overlay_brands = new_brands
                        overlay_general_detections = [*new_secondary_detections, *new_general_detections]
                        overlay_debug_text = new_debug_text

                    ai_frame_index += 1
                except Exception as exc:
                    set_status("error", str(exc))
                    print(f"Web app AI detection error: {exc}")
                    break

        ai_thread = threading.Thread(target=ai_detection_loop, daemon=True)
        ai_thread.start()

        while not STOP_EVENT.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                set_status("waiting_frame")
                time.sleep(0.05)
                continue

            with STATE_LOCK:
                STATE["raw_frame_count"] += 1

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
            last_time = now

            with raw_lock:
                raw_frame_for_ai = frame.copy()
                raw_frame_seq += 1

            if now - last_publish_time < min_publish_interval:
                continue

            display_frame = frame.copy()
            with overlay_lock:
                local_airbags = list(overlay_airbags)
                local_brands = list(overlay_brands)
                local_general_detections = list(overlay_general_detections)
                local_debug_text = overlay_debug_text

            for xyxy, confidence in local_airbags:
                detector.draw_detection(display_frame, xyxy, detector.ACTIVE_PRIMARY_CLASS, confidence)

            for xyxy, brand_name, brand_confidence in local_brands:
                detector.draw_brand_label(display_frame, xyxy, brand_name, brand_confidence)

            for xyxy, class_name, confidence in local_general_detections:
                detector.draw_detection(display_frame, xyxy, class_name, confidence)

            if local_debug_text is not None:
                color = (
                    (0, 255, 0)
                    if local_debug_text >= airbag_confidence
                    else (0, 255, 255)
                    if local_debug_text > 0
                    else (0, 0, 255)
                )
                detector.draw_status(display_frame, f"Raw {detector.ACTIVE_PRIMARY_CLASS} confidence: {local_debug_text:.2f}", color)

            detector.draw_fps(display_frame, fps)
            set_latest_frame(display_frame)
            last_publish_time = now

            with STATE_LOCK:
                STATE["frame_count"] += 1
                STATE["fps"] = round(fps, 2)
                STATE["status"] = "running"

    except Exception as exc:
        message = str(exc)
        set_status("error", message)
        set_latest_frame(make_status_frame("Camera error", message))
        print(f"Web app worker error: {message}")
    finally:
        if ai_thread is not None and ai_thread.is_alive():
            ai_thread.join(timeout=1.5)
        if cap is not None:
            cap.release()
        with STATE_LOCK:
            CURRENT_CAP = None
            STATE["running"] = False
            STATE["stopped_at"] = time.time()
            if STATE["status"] not in {"error", "model_error"}:
                STATE["status"] = "stopped"
        print("Web app camera worker stopped and camera released.")


@serialized_workflow
def stop_camera_worker(join_timeout: float = 3.0) -> bool:
    """Request stop; retain ownership until the worker's finally releases capture."""
    global WORKER_THREAD, CURRENT_CAP
    STOP_EVENT.set()
    worker = WORKER_THREAD
    if worker is not None and worker.is_alive():
        set_status("stopping")
        set_latest_frame(make_status_frame("Stopping camera"))
        worker.join(timeout=join_timeout)
        if worker.is_alive():
            return False
    WORKER_THREAD = None
    with STATE_LOCK:
        CURRENT_CAP = None
        STATE["running"] = False
        clear_inspection_state()
        if STATE["status"] not in {"error", "model_error"}:
            STATE["status"] = "stopped"
            set_latest_frame(make_status_frame("Camera stopped"))
    return True


def classify_product_brand(
    brand_model: object,
    image: np.ndarray,
    object_box: tuple[int, int, int, int] | None = None,
) -> tuple[str, float] | None:
    """Call an optional product/brand classifier through a generic contract."""
    if brand_model is None or image is None:
        return None
    return detector.classify_airbag_brand(
        brand_model,
        image,
        object_box or (0, 0, image.shape[1], image.shape[0]),
    )


def normal_reference_camera_worker(camera_index: int, use_brand: bool) -> None:
    """Live generic OK/NG inspection without loading the legacy defect detector."""
    global CURRENT_CAP
    cap = None
    cycle = PartCycle()
    reference_config = {}
    try:
        config = read_project_config()
        reference_config = dict(config.get("normal_reference") or {})
        model_path = Path(str(reference_config.get("active_model") or ""))
        if not model_path.is_file():
            raise FileNotFoundError("สร้าง Normal Reference จาก Dataset ภาพปกติก่อนเปิดตรวจสอบสด")
        inspector = RecipeBoundInspector(load_inspection_reference(model_path), model_path, reference_config)
        locator_model, locator_class_id = get_shared_locator(config)
        brand_model = None
        if use_brand and str(config.get("active_brand_model") or "").strip():
            try:
                brand_model = load_brand_model()
            except Exception as exc:
                print(f"Optional brand model unavailable: {exc}")

        set_status("opening_camera")
        cap, camera_index = open_camera_like_detector(camera_index)
        with STATE_LOCK:
            CURRENT_CAP = cap
        set_status("running")
        fps = 0.0
        last_time = time.perf_counter()
        last_result = None
        last_annotated = None
        last_inspection = 0.0
        last_ng_capture = 0.0
        last_anomaly_seen = 0.0
        active_review_id = ""
        stabilizer = DecisionStabilizer()
        frame_unavailable = False

        while not STOP_EVENT.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                set_status("waiting_frame")
                if not frame_unavailable:
                    set_latest_frame(make_status_frame("Camera frame unavailable", "Waiting for a fresh image"))
                if not frame_unavailable:
                    save_cycle(PROJECT_ROOT, cycle.finish("camera_unavailable"), reference_config, f"camera_{camera_index}")
                frame_unavailable = True
                stabilizer.reset()
                last_result = last_annotated = None
                last_inspection = last_anomaly_seen = last_ng_capture = 0.0
                active_review_id = ""
                fps = 0.0
                last_time = time.perf_counter()
                time.sleep(0.04)
                continue
            frame_unavailable = False
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                fps = fps * 0.85 + (1.0 / elapsed) * 0.15

            if last_annotated is None or now - last_inspection >= WEB_AI_MIN_INTERVAL_SECONDS:
                previous_segment = stabilizer.segment
                located = detector.analyze_primary_object(locator_model, locator_class_id, frame)
                if located is not None:
                    brand_name = None
                    brand_confidence = None
                    if brand_model is not None:
                        try:
                            brand_result = classify_product_brand(brand_model, frame, located["box"])
                            if brand_result:
                                brand_name, brand_confidence = brand_result
                        except Exception:
                            pass
                    measured = inspector.inspect(
                        frame,
                        object_box=located["box"],
                        brand=brand_name,
                        brand_confidence=brand_confidence,
                    )
                    last_result = stabilizer.update(measured, object_box=located["box"])
                else:
                    stabilizer.reset()
                    last_result = None
                if stabilizer.segment != previous_segment:
                    active_review_id = ""
                    last_anomaly_seen = last_ng_capture = 0.0
                last_annotated = inspector.draw_result(
                    frame,
                    last_result or InspectionResult("NO_PART", 0.0, 0.0, inspector.threshold, []),
                    object_box=located["box"] if located else None,
                    object_label=located["label"] if located else "Part",
                    object_confidence=located["confidence"] if located else None,
                )
                anomaly_visible = bool(
                    located is not None
                    and last_result is not None
                    and not last_result.input_issue
                    and (last_result.status == "NG" or last_result.review_required)
                )
                if anomaly_visible:
                    if active_review_id and now - last_anomaly_seen > NG_EVENT_RELEASE_SECONDS:
                        active_review_id = ""
                    last_anomaly_seen = now
                    if not active_review_id or now - last_ng_capture >= NG_CAPTURE_INTERVAL_SECONDS:
                        try:
                            review = save_inspection_review(
                                frame,
                                last_annotated,
                                last_result,
                                object_box=located["box"],
                                object_label=located["label"],
                                object_confidence=located["confidence"],
                                source_name=f"camera_{camera_index}",
                                existing_review_id=active_review_id,
                                reference=reference_config,
                            )
                            active_review_id = str(review.get("id") or active_review_id)
                            last_ng_capture = now
                        except OSError as exc:
                            print(f"Could not save NG review: {exc}")
                elif active_review_id and now - last_anomaly_seen > NG_EVENT_RELEASE_SECONDS:
                    active_review_id = ""
                closed = cycle.observe(located is not None, last_result, now=now, frame=frame)
                completed = save_cycle(PROJECT_ROOT, closed, reference_config, f"camera_{camera_index}")
                with STATE_LOCK:
                    STATE["part_cycle"] = dict(cycle.current) if cycle.current else None
                    if completed:
                        STATE["last_part_result"] = completed
                last_inspection = now
            display_frame = last_annotated.copy() if last_annotated is not None else frame
            cycle.draw(display_frame)
            detector.draw_fps(display_frame, fps)
            set_latest_frame(display_frame)
            with STATE_LOCK:
                STATE["frame_count"] += 1
                STATE["raw_frame_count"] += 1
                STATE["fps"] = round(fps, 2)
                STATE["status"] = "running"
                if last_result is not None:
                    STATE["decision"] = (
                        "REVIEW" if last_result.review_required and last_result.status != "NG"
                        else last_result.status
                    )
                    STATE["review_required"] = last_result.review_required
                    STATE["confidence"] = last_result.confidence
                    STATE["anomaly_score"] = last_result.anomaly_score
                    STATE["regions"] = [
                        {
                            "x": region.x,
                            "y": region.y,
                            "width": region.width,
                            "height": region.height,
                            "score": region.score,
                        }
                        for region in last_result.regions
                    ]
                    STATE["brand"] = last_result.brand
                    STATE["brand_confidence"] = last_result.brand_confidence
                else:
                    STATE["decision"] = "NO_PART"
                    STATE["review_required"] = False
                    STATE["confidence"] = 0.0
                    STATE["anomaly_score"] = 0.0
                    STATE["regions"] = []
                    STATE["brand"] = None
                    STATE["brand_confidence"] = None
                STATE["frame_decision"] = STATE["decision"]
                STATE["part_present"] = cycle.missing == 0 and cycle.current is not None
                if cycle.current:
                    STATE["decision"] = cycle.current["decision"]
                    STATE["review_required"] = cycle.current["decision"] == "REVIEW"
    except Exception as exc:
        message = str(exc)
        set_status("error", message)
        set_latest_frame(make_status_frame("Inspection error", message))
        print(f"Normal-reference camera worker error: {message}")
    finally:
        try:
            save_cycle(PROJECT_ROOT, cycle.finish("camera_stopped"), reference_config, f"camera_{camera_index}")
        except OSError as exc:
            set_status("error", f"Could not save part result: {exc}")
        if cap is not None:
            cap.release()
        with STATE_LOCK:
            CURRENT_CAP = None
            STATE["running"] = False
            clear_inspection_state()
            STATE["stopped_at"] = time.time()
            if STATE["status"] != "error":
                STATE["status"] = "stopped"
                set_latest_frame(make_status_frame("Camera stopped"))


@serialized_workflow
def start_camera_worker(
    camera_index: int,
    airbag_confidence: float,
    debug_detection: bool,
    use_brand: bool,
) -> None:
    """เริ่ม worker ใหม่ โดยหยุดตัวเก่าก่อนเสมอ."""
    global WORKER_THREAD

    config = read_project_config()
    if config.get("inspection_strategy") == "normal_reference":
        runtime = normal_reference_runtime_status(config)
        if not runtime["ready"]:
            raise RuntimeError(runtime["reason"])
        get_shared_locator(config)

    if training_is_running():
        raise RuntimeError("Training is running; stop it before opening the camera.")
    if not stop_camera_worker():
        raise RuntimeError("Camera is still stopping; wait for the existing worker to exit.")
    STOP_EVENT.clear()

    with STATE_LOCK:
        clear_inspection_state()
        STATE.update(
            {
                "running": True,
                "status": "starting",
                "error": "",
                "camera_index": camera_index,
                "frame_count": 0,
                "raw_frame_count": 0,
                "fps": 0.0,
                "started_at": time.time(),
                "stopped_at": None,
            }
        )
    set_latest_frame(make_status_frame("Starting camera", f"Camera index {camera_index}"))

    if config.get("inspection_strategy") == "normal_reference":
        target = normal_reference_camera_worker
        args = (camera_index, use_brand)
    else:
        target = camera_worker
        args = (camera_index, airbag_confidence, debug_detection, use_brand)
    WORKER_THREAD = vision_service.runtime.ownership.start_thread(target,args,daemon=True)


# ===== HTTP routes: หน้าเว็บและ Annotation API =====
# route ทำหน้าที่ validate request/จัด response; งานไฟล์หรือ AI หลักอยู่ใน helper ด้านบน
# เพื่อลดเวลาถือ lock และให้ทดสอบ logic ได้โดยไม่ต้องสร้าง HTTP request
@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> RedirectResponse:
    """หน้าเว็บหลัก."""
    return RedirectResponse(url="/inspection", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/detect", include_in_schema=False)
def detect_page_redirect() -> RedirectResponse:
    return RedirectResponse(url="/inspection", status_code=307)


@app.get("/annotation", include_in_schema=False)
def annotation_page_redirect() -> RedirectResponse:
    return RedirectResponse(url=cvat_connection_config()["base_url"], status_code=307)


# Annotation is handled by the existing CVAT installation. Duplicate Studio routes removed.

@app.get("/api/dataset-folders")
def api_dataset_folders() -> JSONResponse:
    """ส่งรายการ folder จริงใน workspace ให้ Folder Explorer."""
    folders = list_dataset_folder_records()
    return JSONResponse(
        {
            "ok": True,
            "folders": folders,
            "totals": {
                "folders": len(folders),
                "exports": sum(item["exports"] for item in folders),
                "images": sum(item["images"] for item in folders),
            },
        }
    )


@app.get("/api/dataset-folders/{folder_name}/preview")
def api_dataset_folder_preview(folder_name: str) -> JSONResponse:
    """ส่ง preview ของ folder จาก disk จริง เช่นจำนวนรูป label export และตัวอย่างรูป."""
    try:
        folder_path = dataset_folder_path(folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse({"ok": False, "error": "Folder not found."}, status_code=404)
        return JSONResponse({"ok": True, "preview": dataset_folder_preview(folder_path)})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/dataset-folders/{folder_name}/image")
def api_dataset_folder_image(folder_name: str, path: str = Query(...)) -> Response:
    """ส่งรูป preview โดยบังคับ path ให้อยู่ใน folder นั้นเท่านั้น."""
    try:
        folder_path = dataset_folder_path(folder_name).resolve()
        target = (folder_path / Path(path)).resolve()
        if target == folder_path or folder_path not in target.parents:
            return JSONResponse({"ok": False, "error": "Invalid image path."}, status_code=400)
        if not is_image_file(target):
            return JSONResponse({"ok": False, "error": "Image not found."}, status_code=404)
        return FileResponse(target)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/dataset-folders/{folder_name}/latest-upload")
def api_dataset_folder_latest_upload(
    folder_name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(48, ge=1, le=96),
) -> JSONResponse:
    """ส่งเฉพาะรูปจาก upload รอบล่าสุดแบบแบ่งหน้า ไม่ปนรูปเดิมใน Dataset."""
    try:
        folder_path = dataset_folder_path(folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse({"ok": False, "error": "Folder not found."}, status_code=404)
        return JSONResponse({"ok": True, **latest_dataset_upload_page(folder_path, offset=offset, limit=limit)})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/dataset-folders")
def api_create_dataset_folder(payload: dict = Body(...)) -> JSONResponse:
    """สร้าง folder ใหม่ใน workspace จากชื่อที่ user ระบุ."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    clear_analysis_cache()
    folder_name = str(payload.get("name") or "").strip()
    if not folder_name:
        return JSONResponse({"ok": False, "error": "Folder name is required."}, status_code=400)
    try:
        folder_path = dataset_folder_path(folder_name)
        if folder_path.exists():
            return JSONResponse({"ok": False, "error": "Folder already exists."}, status_code=409)
        folder_path.mkdir(parents=True, exist_ok=False)
        return JSONResponse({"ok": True, "folder": dataset_folder_record(folder_path), "folders": list_dataset_folder_records()})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.delete("/api/dataset-folders/{folder_name}")
def api_delete_dataset_folder(folder_name: str) -> JSONResponse:
    """Soft-delete a dataset folder so accidental removal remains recoverable."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    clear_analysis_cache()
    try:
        folder_path = dataset_folder_path(folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse({"ok": False, "error": "Folder not found."}, status_code=404)
        if folder_path.parent.resolve() != AIRBAG_EXPORTS_DIR.resolve():
            return JSONResponse({"ok": False, "error": "Only top-level dataset folders can be deleted."}, status_code=400)
        DELETED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        deleted_name = f"{timestamp_name()}_{safe_name(folder_path.name)}"
        destination = unique_child_path(DELETED_DATASETS_DIR, deleted_name)
        shutil.move(str(folder_path), str(destination))
        write_json(
            destination / "deletion.json",
            {
                "original_name": folder_path.name,
                "original_path": str(folder_path),
                "deleted_at": now_iso(),
            },
        )
        return JSONResponse(
            {
                "ok": True,
                "deleted": folder_name,
                "deleted_path": str(destination),
                "permanent": False,
                "folders": list_dataset_folder_records(),
                "deleted_folders": deleted_dataset_records(),
            }
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/deleted-dataset-folders")
def api_deleted_dataset_folders() -> JSONResponse:
    return JSONResponse({"ok": True, "folders": deleted_dataset_records()})


def cleanup_deleted_dataset_artifacts(original_name: str) -> list[str]:
    """Invalidate rebuildable combined inputs; preserve trained runs and models."""
    output_root = (AIRBAG_EXPORTS_DIR.parent / "03_output").resolve()
    targets = []
    for manifest in output_root.glob("*_yolov8_combined/source_manifest.json"):
        data = read_json(manifest, {})
        sources = {str(row.get("path", "")).replace("\\", "/").split("/")[0]
                   for row in data.get("files", []) if row.get("group") == "exports"}
        if original_name not in sources:
            continue
        for name in ("train", "valid", "test", "data.yaml", "source_manifest.json",
                     "quality_report.json", "full_frame_rejections.json", "split_rebalance_report.json"):
            path = manifest.parent / name
            if not path.exists():
                continue
            if path.resolve().parent.parent != output_root or path.is_symlink() or os.path.isjunction(path):
                raise ValueError("Unsafe generated dataset path")
            if path.is_dir():
                for directory, folders, files in os.walk(path, followlinks=False):
                    if any((Path(directory) / child).is_symlink() or os.path.isjunction(Path(directory) / child)
                           for child in folders + files):
                        raise ValueError("Generated dataset contains filesystem links")
            targets.append(path)
    removed = []
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    return removed


@app.delete("/api/deleted-dataset-folders/{deleted_id}")
def api_permanently_delete_dataset(deleted_id: str, payload: dict = Body(...)) -> JSONResponse:
    """Permanently remove one explicitly confirmed, direct trash entry."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    if payload.get("confirm_id") != deleted_id:
        return JSONResponse({"ok": False, "error": "ยืนยันรายการที่จะลบก่อน"}, status_code=400)
    root = DELETED_DATASETS_DIR.resolve()
    source = root / deleted_id
    if (not deleted_id or deleted_id in {".", ".."}
            or any(char in deleted_id for char in '/\\:')
            or source.resolve().parent != root
            or source.is_symlink() or os.path.isjunction(source)):
        return JSONResponse({"ok": False, "error": "Invalid trash entry."}, status_code=400)
    if not source.is_dir():
        return JSONResponse({"ok": False, "error": "ไม่พบรายการนี้ในถังขยะ"}, status_code=404)
    try:
        # Reject redirected descendants before a recursive filesystem operation.
        for directory, folders, files in os.walk(source, followlinks=False):
            for name in folders + files:
                child = Path(directory) / name
                if child.is_symlink() or os.path.isjunction(child):
                    return JSONResponse({"ok": False, "error": "Trash entry contains filesystem links."}, status_code=400)
        original_name = str(read_json(source / "deletion.json", {}).get("original_name") or "")
        removed_artifacts = cleanup_deleted_dataset_artifacts(original_name) if original_name else []
        config = read_project_config()
        reset_name = bool(original_name and config.get("project_name") == original_name)
        if reset_name:
            write_project_config({"project_name": DEFAULT_PROJECT_CONFIG["project_name"]})
        shutil.rmtree(source)
        clear_analysis_cache()
        return JSONResponse({"ok": True, "deleted": deleted_id, "permanent": True,
                             "removed_artifacts": removed_artifacts, "project_name_reset": reset_name})
    except (OSError, ValueError):
        return JSONResponse({"ok": False, "error": "ลบไม่ครบ ไฟล์อาจกำลังถูกใช้งาน กรุณาลองใหม่"}, status_code=409)


@app.post("/api/deleted-dataset-folders/{deleted_id}/restore")
def api_restore_dataset_folder(deleted_id: str) -> JSONResponse:
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    clear_analysis_cache()
    try:
        clean_id = safe_name(deleted_id)
        source = (DELETED_DATASETS_DIR / clean_id).resolve()
        deleted_root = DELETED_DATASETS_DIR.resolve()
        if source == deleted_root or deleted_root not in source.parents or not source.is_dir():
            return JSONResponse({"ok": False, "error": "Deleted dataset not found."}, status_code=404)
        deletion_meta = read_json(source / "deletion.json", {})
        original_name = safe_name(str(deletion_meta.get("original_name") or source.name))
        destination = unique_child_path(AIRBAG_EXPORTS_DIR, original_name)
        deletion_file = source / "deletion.json"
        if deletion_file.exists():
            deletion_file.unlink()
        shutil.move(str(source), str(destination))
        return JSONResponse(
            {
                "ok": True,
                "restored_path": str(destination),
                "folder": dataset_folder_record(destination),
                "folders": list_dataset_folder_records(),
                "deleted_folders": deleted_dataset_records(),
            }
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/dataset-folders/upload")
async def api_upload_dataset_folder(
    folder: str = Form(""),
    destination_mode: str = Form("append"),
    new_folder_name: str = Form(""),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    """รับไฟล์จากหน้า Folder Explorer แล้วติดตั้งเข้า folder ที่เลือกหรือ folder ใหม่."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    clear_analysis_cache()
    try:
        folder_name = new_folder_name if destination_mode == "create" else folder
        if not folder_name.strip():
            return JSONResponse({"ok": False, "error": "Choose or create a destination folder."}, status_code=400)
        target_folder = dataset_folder_path(folder_name)
        created_folder = not target_folder.exists()
        target_folder.mkdir(parents=True, exist_ok=True)
        staging_dir, saved_count = await save_uploads_to_staging(files, safe_name(target_folder.name))
        target_meta = ensure_dataset_meta(target_folder)
        if target_meta.get("dataset_type") == "Negative / Background":
            # Upload จาก Folder Explorer ต้องเข้าทางเดียวกับปุ่ม Background หลัก
            # ไม่เช่นนั้นรูปจะกลายเป็น raw_images และไม่ถูกนำไปแบ่ง Eval
            split_result = install_background_upload(staging_dir, AIRBAG_BACKGROUND_DIR)
            brand_images = copy_images_from_staging(staging_dir, BRAND_BACKGROUND_DIR, "unknown")
            installed = {
                "installed_paths": split_result.get("installed_paths") or [],
                "copied_files": split_result["installed"],
                "ready_exports": 0,
                "auto_split": None,
                "exports": 0,
                "images": split_result["total"],
                "background_split": split_result,
                "brand_unknown_images": brand_images,
            }
        else:
            installed = install_staging_to_dataset_folder(staging_dir, target_folder)
        latest_upload = record_latest_dataset_upload(target_folder, installed.get("installed_paths") or [])
        shutil.rmtree(staging_dir, ignore_errors=True)
        return JSONResponse(
            {
                "ok": True,
                "saved_files": saved_count,
                "created_folder": created_folder,
                "installed": installed,
                "latest_upload": {
                    "folder": latest_upload["folder"],
                    "uploaded_at": latest_upload["uploaded_at"],
                    "upload_id": latest_upload["upload_id"],
                    "count": latest_upload["count"],
                },
                "folder": dataset_folder_record(target_folder),
                "folders": list_dataset_folder_records(),
            }
        )
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/dataset-folders/undo-upload")
@serialized_workflow
def api_undo_dataset_folder_upload(payload: dict = Body(...)) -> JSONResponse:
    """Undo only the server-recorded latest batch, preserving it in existing Trash."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    try:
        folder = dataset_folder_path(str(payload.get("folder") or ""))
        manifest_path = folder / LATEST_UPLOAD_MANIFEST
        manifest = read_json(manifest_path, {})
        if not manifest.get("upload_id") or payload.get("upload_id") != manifest["upload_id"]:
            raise ValueError("Upload has changed or predates safe Undo. Refresh and use Trash for older data.")
        files = manifest.get("undo_files") or {}
        if not files:
            raise ValueError("This upload cannot be undone safely; use the dataset Trash controls.")
        targets = []
        for relative, expected in files.items():
            candidate = folder / relative
            target = candidate.resolve()
            if folder not in target.parents or not target.is_file() or candidate.is_symlink():
                raise ValueError("Upload files have changed; nothing moved.")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Upload files have changed; nothing moved.")
            targets.append((target, target.relative_to(folder)))
        trash = unique_child_path(DELETED_DATASETS_DIR, "undo_" + manifest["upload_id"])
        trash.mkdir()
        moved = []
        try:
            for target, relative in targets:
                dest = trash / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(dest))
                moved.append((target, dest))
            write_json(trash / "deletion.json", {"original_name": folder.name + "_upload_" + manifest["upload_id"][:8],
                "deleted_at": now_iso(), "reason": "undo_upload", "source_folder": folder.name})
            manifest_path.unlink()
        except Exception:
            for target, dest in reversed(moved):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(target))
            raise
        clear_analysis_cache()
        return JSONResponse({"ok": True, "moved": len(moved), "trash_id": trash.name,
                             "folders": list_dataset_folder_records()})
    except (OSError, ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/dataset-folders/{folder_name}/metadata")
def api_update_dataset_folder_metadata(folder_name: str, payload: dict = Body(...)) -> JSONResponse:
    """บันทึกข้อมูลอธิบาย dataset หลัง upload สำเร็จจาก popup."""
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    clear_analysis_cache()
    try:
        folder_path = dataset_folder_path(folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            return JSONResponse({"ok": False, "error": "Folder not found."}, status_code=404)
        if folder_path.parent.resolve() != AIRBAG_EXPORTS_DIR.resolve():
            return JSONResponse({"ok": False, "error": "Only top-level dataset folders can be updated."}, status_code=400)
        write_dataset_meta(folder_path, payload)
        return JSONResponse({"ok": True, "folder": dataset_folder_record(folder_path), "folders": list_dataset_folder_records()})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


# ===== HTTP routes: Training, Models และ Project config =====
# start สร้าง subprocess พร้อม log/state; stop หยุดเฉพาะ process ที่ backend ควบคุมอยู่
# Resume ต้องผูกกับ combined dataset และ source manifest เดิมเสมอ
@app.get("/api/train/preflight")
def api_train_preflight(train_type: str = Query("airbag_detector")) -> JSONResponse:
    """ตรวจความพร้อมก่อน train จาก backend จริง."""
    if train_type == "normal_reference":
        config = read_project_config()
        runtime = normal_reference_runtime_status(config)
        locator = active_locator_model_path()
        free_bytes = shutil.disk_usage(PROJECT_ROOT).free
        checks = [
            {
                "label": "Normal Reference engine",
                "status": "pass",
                "detail": "พร้อมเรียนรู้ภาพ Known-good และแยก Holdout",
            },
            {
                "label": "Part Locator (ROI only)",
                "status": "pass" if locator else "fail",
                "detail": f"{Path(locator).name} · Mark ไม่ร่วมตัดสิน" if locator else "ยังไม่มีโมเดลหา ROI ชิ้นงาน",
            },
            {
                "label": "Active Reference",
                "status": "pass" if runtime["ready"] else "warn",
                "detail": Path(runtime["model"]).name if runtime["ready"] else "จะสร้างเมื่อเริ่ม Train",
            },
            {
                "label": "Disk space",
                "status": "pass" if free_bytes >= 1024**3 else "fail",
                "detail": f"เหลือ {free_bytes / 1024**3:.1f} GB",
            },
        ]
        errors = []
        if free_bytes < 1024**3:
            errors.append("Free disk space is below 1 GB.")
        if not locator:
            errors.append("Part Locator model is required before training.")
        warnings = []
        return JSONResponse({
            "ok": not errors,
            "train_type": train_type,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "checked_at": now_iso(),
        })
    if train_type not in {"airbag_detector", "brand_classifier"}:
        return JSONResponse({"ok": False, "error": "Unknown train type."}, status_code=400)
    return JSONResponse(train_preflight(train_type))


@app.get("/api/train/runs")
def api_train_runs() -> JSONResponse:
    """Return unified history while preserving each model module identity."""
    runs = normal_reference_run_records() + train_run_records()
    runs.sort(key=lambda run: str(run.get("updated_at") or ""), reverse=True)
    module_counts: dict[str, int] = {}
    for run in runs:
        module = str(run.get("module") or run.get("train_type") or "other")
        module_counts[module] = module_counts.get(module, 0) + 1
    return JSONResponse({"ok": True, "runs": runs[:60], "module_counts": module_counts})


@app.get("/api/train/evaluation")
def api_train_evaluation() -> JSONResponse:
    """Return the latest detector evaluation without exposing filesystem paths."""
    snapshot = latest_detector_evaluation()
    status_code = 200 if snapshot.get("ok") else 404
    return JSONResponse(snapshot, status_code=status_code, headers={"Cache-Control": "no-store"})


@app.get("/api/train/evaluation/assets/{asset_name}")
def api_train_evaluation_asset(asset_name: str) -> Response:
    """Serve only known visual artifacts from the latest measured detector run."""
    allowed_assets = {"results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "val_batch0_pred.jpg"}
    if asset_name not in allowed_assets:
        return JSONResponse({"ok": False, "error": "Unknown evaluation asset."}, status_code=404)
    results_csv = latest_detector_results_csv()
    asset_path = results_csv.parent / asset_name if results_csv else None
    if not asset_path or not asset_path.is_file():
        return JSONResponse({"ok": False, "error": "Evaluation asset is not available for the latest run."}, status_code=404)
    return FileResponse(asset_path, headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/train/evaluation/preview/{tile_index}")
def api_train_evaluation_preview(tile_index: int) -> Response:
    """Serve clean validation samples cropped once from YOLO's prediction sheet."""
    # Put three inspection-friendly crops first for the page preview, while
    # still exposing every cell from YOLO's 4x4 validation sheet to View all.
    tile_positions = (
        (0, 3), (1, 2), (1, 3),
        (2, 2), (2, 3), (3, 2), (3, 3),
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
        (2, 0), (2, 1), (3, 0), (3, 1),
    )
    if tile_index < 0 or tile_index >= len(tile_positions):
        return JSONResponse({"ok": False, "error": "Unknown validation preview tile."}, status_code=404)
    results_csv = latest_detector_results_csv()
    preview_path = results_csv.parent / "val_batch0_pred.jpg" if results_csv else None
    if not preview_path or not preview_path.is_file():
        return JSONResponse({"ok": False, "error": "Validation preview is not available for the latest run."}, status_code=404)
    stat = preview_path.stat()
    cache_key = (str(preview_path.resolve()), stat.st_mtime_ns, stat.st_size)
    with EVALUATION_PREVIEW_CACHE_LOCK:
        cached_tiles = EVALUATION_PREVIEW_CACHE.get("tiles", ())
        if EVALUATION_PREVIEW_CACHE.get("key") != cache_key or len(cached_tiles) != len(tile_positions):
            image = cv2.imread(str(preview_path))
            if image is None or image.size == 0:
                return JSONResponse({"ok": False, "error": "Validation preview could not be read."}, status_code=500)

            rows, columns = 4, 4
            height, width = image.shape[:2]
            generated_tiles: list[bytes] = []
            for row_index, column_index in tile_positions:
                top = round(row_index * height / rows)
                bottom = round((row_index + 1) * height / rows)
                left = round(column_index * width / columns)
                right = round((column_index + 1) * width / columns)
                tile = image[top:bottom, left:right]

                # Ultralytics places the source filename above the image and
                # pads the rest of a sheet cell with a flat gray canvas. Keep
                # only the real prediction so cards do not look compressed.
                filename_strip = min(64, max(0, tile.shape[0] // 6))
                if filename_strip:
                    tile = tile[filename_strip:, :]
                if tile.size:
                    background = np.median(tile[-min(14, tile.shape[0]) :, :], axis=(0, 1))
                    distance = np.max(np.abs(tile.astype(np.int16) - background.astype(np.int16)), axis=2)
                    active_rows = np.flatnonzero(np.mean(distance > 18, axis=1) > 0.035)
                    if active_rows.size:
                        row_runs = np.split(active_rows, np.flatnonzero(np.diff(active_rows) > 1) + 1)
                        image_rows = max(row_runs, key=len)
                        crop_top = max(0, int(image_rows[0]) - 6)
                        crop_bottom = min(tile.shape[0], int(image_rows[-1]) + 7)
                        if crop_bottom - crop_top >= 96:
                            tile = tile[crop_top:crop_bottom, :]

                encoded, jpeg = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not encoded:
                    return JSONResponse({"ok": False, "error": "Validation preview could not be encoded."}, status_code=500)
                generated_tiles.append(jpeg.tobytes())

            cached_tiles = tuple(generated_tiles)
            EVALUATION_PREVIEW_CACHE["key"] = cache_key
            EVALUATION_PREVIEW_CACHE["tiles"] = cached_tiles
    return Response(
        content=cached_tiles[tile_index],
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600, immutable"},
    )


@app.get("/api/models/registry")
def api_models_registry() -> JSONResponse:
    """ส่งรายการโมเดลที่ระบบรู้จักและ path ของไฟล์จริง."""
    return JSONResponse({"ok": True, "models": model_registry(), "inspection_models": vision_service.store.records("models")})


@app.post("/api/models/activate")
@serialized_workflow
def api_activate_model(payload: dict = Body(...)) -> JSONResponse:
    if training_is_running():
        return JSONResponse(
            {"ok": False, "error": "A model cannot be activated while training is running."},
            status_code=409,
        )
    if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
        return JSONResponse(
            {"ok": False, "error": "Stop camera detection before changing the active model."},
            status_code=409,
        )
    raw_path = str(payload.get("path") or "").strip()
    model_kind = str(payload.get("kind") or "detection").strip().lower()
    if not raw_path:
        return JSONResponse({"ok": False, "error": "Model path is required."}, status_code=400)
    model_path = Path(raw_path).resolve()
    project_root = PROJECT_ROOT.resolve()
    if model_path.suffix.lower() != ".pt" or (model_path != project_root and project_root not in model_path.parents):
        return JSONResponse({"ok": False, "error": "Model path is outside this project."}, status_code=400)
    if not model_path.exists():
        return JSONResponse({"ok": False, "error": "Model file not found."}, status_code=404)

    config = read_project_config()
    if model_kind == "classification":
        validation = inspect_classifier_model(model_path)
    else:
        validation = inspect_detector_model(model_path, list(config.get("target_classes") or []))
        contract_ok = bool(validation["ok"])
        validation["model_contract_ok"] = contract_ok
        if contract_ok:
            try:
                background_evaluation = evaluate_detector_backgrounds(model_path, config)
            except Exception as exc:
                background_evaluation = {"ok": False, "error": f"Background evaluation failed: {exc}", "checked_at": now_iso()}
            validation["background_evaluation"] = background_evaluation
            validation["ok"] = bool(background_evaluation.get("ok"))
            if not validation["ok"]:
                validation["error"] = str(background_evaluation.get("error") or "Background activation gate failed.")
    if not validation["ok"]:
        return JSONResponse(
            {
                "ok": False,
                "error": validation["error"],
                "validation": validation,
            },
            status_code=400,
        )
    if model_kind == "classification":
        update = {
            "active_brand_model": str(model_path),
            "last_model_validation": validation,
        }
    else:
        current = str(config.get("active_detector_model") or "")
        update = {
            "active_detector_model": str(model_path),
            "last_model_validation": validation,
        }
        if current and Path(current).resolve() != model_path:
            update["previous_detector_model"] = current
    config = write_project_config(update)
    write_model_metadata(
        model_path,
        {
            "model_path": str(model_path),
            "classes": validation["classes"],
            "project_name": config.get("project_name"),
            "train_log": config.get("last_train_log"),
            "previous_detector_model": config.get("previous_detector_model") if model_kind != "classification" else "",
            "kind": model_kind,
            "activated_at": now_iso(),
            "validation": validation,
        },
    )
    clear_shared_model_cache()
    return JSONResponse({"ok": True, "config": config, "validation": validation})


@app.get("/api/project/config")
def api_get_project_config() -> JSONResponse:
    return JSONResponse({"ok": True, "config": read_project_config()})


def save_job(name: str, reference: dict, locator: str, *, backbone: str,
             brand: str = "", check_brand: bool = False) -> dict:
    job = make_job(PROJECT_ROOT, name, reference, locator, backbone_path=backbone,
                   brand_path=brand, check_brand=check_brand)
    write_json(job_path(JOB_DIR, job["id"]), job)
    return job


def job_change_blocked():
    if training_is_running() or (WORKER_THREAD is not None and WORKER_THREAD.is_alive()):
        return JSONResponse({"ok": False, "error": "Stop Train and the camera before changing jobs."}, status_code=409)
    return None


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request):
    return RedirectResponse(url="/inspection#test", status_code=307)


@app.get("/api/jobs")
def api_jobs():
    reference = read_project_config().get("normal_reference") or {}
    jobs = []
    for path in sorted(JOB_DIR.glob("job_*.json"), reverse=True):
        try:
            job = read_json(path)
            jobs.append({"id": job["id"], "name": job["name"], "created_at": job["created_at"],
                         "active": reference.get("job_id") == job["id"], "use": "test",
                         "check_brand": job["settings"].get("use_classifier", False),
                         "checks_passed": job["settings"]["normal_reference"].get("candidate_comparison", {}).get("checks_passed"),
                         "metrics": job["settings"]["normal_reference"].get("quality_metrics", {})})
        except (OSError, ValueError, KeyError, TypeError):
            jobs.append({"id": path.stem, "name": path.stem, "error": "Job file cannot be read"})
    return JSONResponse({"ok": True, "jobs": sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True),
                         "active_model": Path(str(reference.get("active_model") or "")).name,
                         "test_reports": [{"id": p.stem} for p in sorted((WEB_APP_DIR / "logs" / "model_tests").glob("test_*.json"), reverse=True)
                                          if re.fullmatch(r"test_[0-9a-f]{32}", p.stem)]})


@app.post("/api/jobs/save-current")
@serialized_workflow
def api_save_current_job(payload: dict = Body(...)):
    blocked = job_change_blocked()
    if blocked is not None:
        return blocked
    try:
        config = read_project_config()
        reference = dict(config.get("normal_reference") or {})
        model = load_inspection_reference(reference["active_model"])
        # Earlier models stored the selected data in the contract only.
        reference.setdefault("recipe_datasets", (reference.get("inspection_contract") or {}).get("datasets", []))
        job = save_job(str(payload.get("name") or ""), reference,
                       str(config.get("active_locator_model") or ""), backbone=model.backbone_path,
                       brand=str(config.get("active_brand_model") or ""),
                       check_brand=payload.get("check_brand", False))
        return JSONResponse({"ok": True, "job_id": job["id"]})
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


def load_job_test_context(job_id):
    job = read_json(job_path(JOB_DIR, job_id))
    if job.get("id") != job_id:
        raise ValueError("Job ID does not match its file")
    settings = checked_settings(PROJECT_ROOT, job)
    reference = settings["normal_reference"]
    model = load_inspection_reference(reference["active_model"])
    if Path(model.backbone_path).resolve() != Path(job["files"]["backbone"]["path"]):
        raise ValueError("Job backbone does not match the model")
    inspector = RecipeBoundInspector(model, reference["active_model"], reference)
    if inspector.contract_issue:
        raise ValueError(f"Job model check failed: {inspector.contract_issue}")
    locator, class_id = detector.load_primary_object_locator(settings)
    brand_model = YOLO(settings["active_brand_model"]) if settings.get("use_classifier") else None
    return job, inspector, locator, class_id, brand_model


def run_job_test_image(context, image, *, preview=True):
    started = time.perf_counter()
    job, inspector, locator, class_id, brand_model = context
    located = detector.analyze_primary_object(locator, class_id, image)
    if located:
        brand = classify_product_brand(brand_model, image, located["box"]) if brand_model is not None else None
        brand = brand or (None, None)
        result = inspector.inspect(image, object_box=located["box"], brand=brand[0], brand_confidence=brand[1])
    else:
        result = InspectionResult("NO_PART", 0., 0., inspector.threshold, [])
    payload = {"ok": True, **result.to_dict(), "job_id": job["id"],
        "decision": "REVIEW" if result.review_required and result.status != "NG" else result.status,
        "model_sha256": job["files"]["model"]["sha256"], "use": "test",
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    if preview:
        annotated = inspector.draw_result(image, result, object_box=located["box"] if located else None)
        ok, encoded = cv2.imencode(".jpg", annotated)
        payload["preview"] = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}" if ok else None
    else:
        payload.pop("anomaly_map", None)
    payload["test_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return payload


@serialized_workflow
def test_job_image(job_id: str, image: np.ndarray):
    """Test without activating a job or writing production review evidence."""
    blocked = job_change_blocked()
    if blocked is not None:
        return blocked
    try:
        return JSONResponse(run_job_test_image(load_job_test_context(job_id), image))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/jobs/{job_id}/test")
async def api_test_job(job_id: str, file: UploadFile = File(...)):
    data = await file.read(20 * 1024 * 1024 + 1)
    if not data or len(data) > 20 * 1024 * 1024:
        return JSONResponse({"ok": False, "error": "Choose an image smaller than 20 MB."}, status_code=400)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"ok": False, "error": "Image cannot be read."}, status_code=400)
    return await run_in_threadpool(test_job_image, job_id, image)


@serialized_workflow
def compare_job_images(job_ids, samples):
    from inspection_testing import summarize_test
    blocked = job_change_blocked()
    if blocked is not None:
        return blocked
    report = {"id": "test_" + uuid.uuid4().hex, "created_at": now_iso(),
              "labels_source": "user", "production_ready": False,
              "scope": "uploaded_test_images", "models": []}
    try:
        for job_id in job_ids:
            context = load_job_test_context(job_id)
            job = context[0]
            model_path = Path(job["files"]["model"]["path"])
            manifest_path = model_path.with_suffix(".training.json")
            known_hashes = set()
            manifest_known = manifest_path.is_file()
            if manifest_known:
                manifest = read_json(manifest_path)
                manifest_known = all(isinstance(manifest.get(key), list) for key in ("fit", "calibration", "validation")) and bool(manifest.get("fit"))
                known_hashes = {entry["sha256"] for split in ("fit", "calibration", "validation")
                                for entry in manifest.get(split, [])}
            rows = []
            for sample in samples:
                row = {key: sample[key] for key in ("name", "expected", "sha256")}
                row["development_image"] = sample["sha256"] in known_hashes
                row["overlap_check_available"] = manifest_known
                try:
                    row.update(run_job_test_image(context, sample["image"], preview=False))
                except Exception as exc:
                    row.update({"ok": False, "decision": "ERROR", "error": str(exc)})
                rows.append(row)
            report["models"].append({"id": job_id, "name": job["name"],
                "model_sha256": job["files"]["model"]["sha256"], "rows": rows,
                "summary": summarize_test(rows)})
        write_json(WEB_APP_DIR / "logs" / "model_tests" / (report["id"] + ".json"), report)
        return JSONResponse({"ok": True, **report})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/jobs/compare")
async def api_compare_jobs(job_ids: str = Form(...), labels: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        ids, expected = json.loads(job_ids), json.loads(labels)
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or len(set(ids)) != len(ids):
            raise ValueError("Choose 1 to 3 different models")
        for value in ids:
            job_path(JOB_DIR, value)
        if not 1 <= len(files) <= 50 or not isinstance(expected, list) or len(expected) != len(files):
            raise ValueError("Choose 1 to 50 images with one label per image")
        if any(value not in ("OK", "NG", "UNKNOWN") for value in expected):
            raise ValueError("Labels must be OK, NG or UNKNOWN")
        samples, seen, total = [], set(), 0
        for file, label in zip(files, expected):
            data = await file.read(20 * 1024 * 1024 + 1)
            total += len(data)
            if not data or len(data) > 20 * 1024 * 1024 or total > 100 * 1024 * 1024:
                raise ValueError("Each image must be under 20 MB; total under 100 MB")
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                raise ValueError("Duplicate image in test set; remove duplicates")
            seen.add(digest)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Image cannot be read: {Path(file.filename or 'image').name}")
            samples.append({"name": Path(file.filename or "image").name, "expected": label,
                            "sha256": digest, "image": image})
        return await run_in_threadpool(compare_job_images, ids, samples)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/api/jobs/test-report/{report_id}")
def api_test_report(report_id: str):
    if not re.fullmatch(r"test_[0-9a-f]{32}", report_id):
        return JSONResponse({"ok": False, "error": "Invalid report ID"}, status_code=400)
    path = WEB_APP_DIR / "logs" / "model_tests" / (report_id + ".json")
    if not path.is_file():
        return JSONResponse({"ok": False, "error": "Report not found"}, status_code=404)
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/api/jobs/{job_id}/use")
@serialized_workflow
def api_use_job(job_id: str):
    blocked = job_change_blocked()
    if blocked is not None:
        return blocked
    try:
        job = read_json(job_path(JOB_DIR, job_id))
        if job.get("id") != job_id:
            raise ValueError("Job ID does not match its file")
        settings = checked_settings(PROJECT_ROOT, job)
        reference = settings["normal_reference"]
        model = load_inspection_reference(reference["active_model"])
        if Path(model.backbone_path).resolve() != Path(job["files"]["backbone"]["path"]):
            raise ValueError("Job backbone does not match the model")
        guard = RecipeBoundInspector(model, reference["active_model"], reference)
        if guard.contract_issue:
            raise ValueError("Job model check failed")
        updated = write_project_config(settings)
        clear_shared_model_cache()
        return JSONResponse({"ok": True, "job_id": job_id, "use": "test", "production_ready": False,
                             "model": updated["normal_reference"]["active_model"]})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/project/config")
@serialized_workflow
def api_update_project_config(payload: dict = Body(...)) -> JSONResponse:
    # This endpoint edits settings only. Artifact identity and measured results
    # belong to the training/activation workflow, never to a browser payload.
    editable_fields = {
        "project_name",
        "brand",
        "model",
        "target_classes",
        "target_classes_text",
        "primary_class",
        "class_aliases",
        "class_display_names",
        "detection_mode",
        "augmentation_profile",
        "inspection_strategy",
        "primary_confidence",
        "secondary_confidence",
        "camera_index",
        "use_classifier",
        "normal_reference",
    }
    if training_is_running():
        return JSONResponse(
            {
                "ok": False,
                "error": "Project settings are locked while training is running.",
            },
            status_code=409,
        )
    if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
        return JSONResponse({"ok": False, "error": "Stop camera detection before changing project settings."}, status_code=409)
    unknown = set(payload) - editable_fields
    if unknown:
        return JSONResponse({"ok": False, "error": "These fields require the model workflow and cannot be edited as settings.",
                             "fields": sorted(unknown)}, status_code=400)
    update = dict(payload)
    if "normal_reference" in update:
        settings = update["normal_reference"]
        allowed = {"inspection_size", "min_good_images", "min_region_ratio"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            return JSONResponse({"ok": False, "error": "Only Reference training settings may be edited here; model identity and evaluation are read-only."}, status_code=400)
        # A partial settings update must not erase the active model/manifest.
        reference = dict(read_project_config().get("normal_reference") or {})
        update["normal_reference"] = {**reference, **settings}
    config = write_project_config(update)
    return JSONResponse({"ok": True, "config": config})


def run_normal_reference_training(
    source_datasets: list[str],
    paths: list[Path],
    settings: dict,
    training_fingerprint: str,
    source_images: int,
    duplicates_removed: int,
) -> None:
    """Build a saved test job without changing the active inspection model."""
    with REFERENCE_TRAIN_LOCK:
        REFERENCE_TRAIN_STATE["stage"] = "extracting_normal_features"
        REFERENCE_TRAIN_STATE["stage_detail"] = "preparing_calibration"
        REFERENCE_TRAIN_STATE["progress"] = 1.0

    deep_progress_start = 1.0

    def report_progress(value: float, detail: str) -> None:
        check_reference_cancelled()
        percent = round(deep_progress_start + max(0.0, min(0.99, float(value))) * (99.0 - deep_progress_start), 1)
        stage = {
            "screening_part_roi": "extracting_deep_features",
            "learning_alignment_and_foreground": "extracting_deep_features",
            "extracting_multiscale_features": "extracting_deep_features",
            "extracting_deep_features": "extracting_deep_features",
            "calibrating_threshold": "calibrating_threshold",
            "validating_holdout": "validating_holdout",
            "building_production_memory": "building_production_reference",
            "deep_patch_ready": "building_production_reference",
        }.get(str(detail), "extracting_deep_features")
        with REFERENCE_TRAIN_LOCK:
            REFERENCE_TRAIN_STATE.update({
                "stage": stage,
                "stage_detail": str(detail),
                "progress": percent,
                "last_progress_at": time.time(),
            })
    try:
        config = read_project_config()
        locator_path = str(config.get("active_locator_model") or config.get("active_detector_model") or "").strip()
        if not locator_path or not Path(locator_path).is_file():
            raise FileNotFoundError("Part Locator model is required before Good-only deep anomaly training.")
        locator_names = YOLO(locator_path).names
        locator_classes = [
            str(locator_names[index]) for index in sorted(locator_names)
        ] if isinstance(locator_names, dict) else [str(name) for name in locator_names]
        locator_retrained = [name.strip().lower() for name in locator_classes] != ["part"]
        if locator_retrained:
            def report_locator_progress(fraction: float, epoch: int, epochs: int, detail: str) -> None:
                check_reference_cancelled()
                percent = round(2.0 + max(0.0, min(1.0, fraction)) * 32.0, 1)
                with REFERENCE_TRAIN_LOCK:
                    REFERENCE_TRAIN_STATE.update({
                        "stage": "training_part_locator",
                        "stage_detail": f"epoch {epoch}/{epochs} · {detail}",
                        "progress": percent,
                        "epoch": int(epoch),
                        "epochs": int(epochs),
                        "last_progress_at": time.time(),
                    })

            with REFERENCE_TRAIN_LOCK:
                REFERENCE_TRAIN_STATE.update({
                    "stage": "training_part_locator",
                    "stage_detail": "mapping_airbag_boxes_to_generic_part",
                    "progress": 2.0,
                    "epoch": 0,
                    "epochs": max(8, min(50, int(settings.get("locator_epochs") or 20))),
                    "last_progress_at": time.time(),
                })
            dataset_dir, target_names = locator_training.build_combined_dataset(
                exports_dir=AIRBAG_EXPORTS_DIR,
                background_dir=AIRBAG_BACKGROUND_DIR,
                output_root=PROJECT_ROOT / "05_train_new_object" / "03_output",
                class_name="Part",
                class_aliases={"Part": ["Airbag", "AirbagV2", "Object"]},
                force_rebuild=True,
                include_folders=set(source_datasets),
            )
            locator_path = str(locator_training.train_model(
                dataset_dir,
                target_names,
                epochs=max(8, min(50, int(settings.get("locator_epochs") or 20))),
                imgsz=max(320, min(640, int(settings.get("locator_image_size") or 416))),
                augmentation={
                    "hsv_h": 0.0, "hsv_s": 0.35, "hsv_v": 0.25,
                    "degrees": 5.0, "translate": 0.08, "scale": 0.18,
                    "shear": 0.0, "fliplr": 0.0, "flipud": 0.0,
                    "mosaic": 0.15, "mixup": 0.0, "perspective": 0.0,
                    "erasing": 0.05, "close_mosaic": 5,
                },
                base_model=locator_path,
                progress_callback=report_locator_progress,
            ).resolve())
            deep_progress_start = 35.0
            with REFERENCE_TRAIN_LOCK:
                REFERENCE_TRAIN_STATE.update({
                    "stage": "extracting_deep_features",
                    "stage_detail": "part_locator_ready",
                    "progress": 35.0,
                    "epoch": 0,
                    "epochs": 0,
                    "last_progress_at": time.time(),
                })
        inspector, quality_metrics = DeepPatchReference.fit_with_evaluation(
            paths,
            backbone_path=locator_path,
            inspection_size=int(settings["inspection_size"]),
            min_images=int(settings["min_good_images"]),
            min_region_ratio=float(settings["min_region_ratio"]),
            inspection_area=checked_area(settings.get("inspection_area")),
            inspection_areas=checked_areas(settings.get("inspection_areas"), settings.get("inspection_area")),
            alignment_reference=settings.get("alignment_reference"),
            progress_callback=report_progress,
        )
        check_reference_cancelled()
        operational_metrics = evaluate_reference_on_reviewed_cases(inspector, settings)
        quality_metrics.update(operational_metrics)
        quality_metrics["defect_metrics_available"] = bool(
            operational_metrics["confirmed_ng_images"]
        )
        quality_metrics["defect_metrics_reason"] = (
            "Measured on human-confirmed NG production captures."
            if operational_metrics["confirmed_ng_images"]
            else "No human-confirmed NG evaluation images are available yet."
        )
        incumbent_reference = dict(config.get("normal_reference") or {})
        incumbent_path = Path(str(incumbent_reference.get("active_model") or ""))
        incumbent_metrics: dict = {}
        manifest = getattr(inspector, "training_manifest", None)
        if manifest is None:
            raise ValueError("Cannot compare candidate without a recorded validation split")
        quality_metrics.update(evaluate_reference_holdout(inspector, manifest))
        candidate_jobs = settings.get("recipe_datasets") or []
        active_jobs = incumbent_reference.get("recipe_datasets") or (incumbent_reference.get("inspection_contract") or {}).get("datasets") or []
        def work_types(items):
            return {tuple(str((item.get("recipe") or {}).get(key) or "").strip().casefold()
                          for key in ("brand", "model", "target")) for item in items}
        same_work = bool(candidate_jobs and active_jobs) and work_types(candidate_jobs) == work_types(active_jobs)
        if incumbent_path.is_file() and same_work:
            incumbent = load_inspection_reference(incumbent_path)
            # Old published percentages may describe another split or bank.
            # Compare both inspectors on exactly the candidate holdout and ROI.
            incumbent_metrics = evaluate_reference_holdout(incumbent, manifest)
            incumbent_metrics.update(evaluate_reference_on_reviewed_cases(incumbent, settings))
            incumbent_metrics["independent_validation"] = False
            incumbent_metrics["evaluation_note"] = "Paired candidate holdout; incumbent may have seen these images during its historical training."
        comparison = compare_reference_candidate(quality_metrics, incumbent_metrics)

        with REFERENCE_TRAIN_LOCK:
            REFERENCE_TRAIN_STATE.update({
                "stage": "comparing_candidate",
                "stage_detail": comparison["decision"],
                "progress": 99.0,
                "last_progress_at": time.time(),
            })
        selected_name = "_".join(str(item.get("label") or item.get("id") or "") for item in settings.get("recipe_datasets", []))
        model_name = f"{safe_name(selected_name or 'inspection')[:80]}_{datetime.now():%Y%m%d_%H%M%S}_deep_patch.npz"
        check_reference_cancelled()
        model_path = inspector.save(NORMAL_REFERENCE_MODEL_DIR / model_name)
        training_manifest = getattr(inspector, "training_manifest", None)
        if training_manifest is not None:
            write_json(model_path.with_suffix(".training.json"), training_manifest)
        training_fingerprint = normal_reference_training_fingerprint(paths, settings, backbone_path=locator_path)
        reference = {
            **settings,
            "effective_inspection_size": inspector.input_size,
            "min_region_ratio": float(settings["min_region_ratio"]),
            "active_model": str(model_path.resolve()),
            "source_dataset": "reviewed_known_good",
            "source_datasets": source_datasets,
            "source_images": source_images,
            "duplicates_removed": duplicates_removed,
            "trained_at": now_iso(),
            "training_images": int(inspector.training_count),
            "training_fingerprint": training_fingerprint,
            "anomaly_engine": "deep_patch",
            "pipeline_version": DeepPatchReference.VERSION,
            "backbone_model": locator_path,
            "locator_retrained": locator_retrained,
            "background_handling": "aligned_part_mask_only",
            "recipe_name": ", ".join(str(item.get("label") or item["id"]) for item in settings.get("recipe_datasets", []))[:120] or "New job",
            "quality_metrics": quality_metrics,
            "inspection_contract": build_inspection_contract(model_path, settings.get("recipe_datasets", [])),
            "candidate_comparison": comparison,
        }
        comparison_record = {
            **comparison,
            "candidate_model": str(model_path.resolve()),
            "incumbent_model": str(incumbent_path.resolve()) if incumbent_path.is_file() else "",
            "measured_at": now_iso(),
            "activated": False,
        }
        write_json(model_path.with_suffix(".comparison.json"), comparison_record)

        with REFERENCE_TRAIN_LOCK:
            check_reference_cancelled()
            job = save_job(reference["recipe_name"], reference, locator_path, backbone=locator_path)
            result = {
                "model": str(model_path),
                "active_model": str(incumbent_path) if incumbent_path.is_file() else "",
                "activated": False,
                "job_id": job["id"],
                "comparison": comparison,
                "training_images": int(inspector.training_count),
                "source_images": source_images,
                "duplicates_removed": duplicates_removed,
                "source_datasets": source_datasets,
                "threshold": float(inspector.threshold),
                "quality_metrics": quality_metrics,
                "config": config,
            }
            completion_stage = "needs_test" if comparison["checks_passed"] else "candidate_rejected"
            with REFERENCE_TRAIN_LOCK:
                REFERENCE_TRAIN_STATE.update({
                    "running": False,
                    "status": "finished",
                    "stage": completion_stage,
                    "stage_detail": comparison["reason"],
                    "progress": 100.0,
                    "last_progress_at": time.time(),
                    "finished_at": time.time(),
                    "error": "",
                    "result": result,
                })
    except ReferenceTrainingCancelled as exc:
        with REFERENCE_TRAIN_LOCK:
            REFERENCE_TRAIN_STATE.update({
                "running": False, "status": "stopped", "stage": "stopped",
                "stage_detail": str(exc), "finished_at": time.time(), "error": "", "result": {},
            })
    except Exception as exc:
        with REFERENCE_TRAIN_LOCK:
            REFERENCE_TRAIN_STATE.update({
                "running": False,
                "status": "failed",
                "stage": "failed",
                "stage_detail": "failed",
                "last_progress_at": time.time(),
                "finished_at": time.time(),
                "error": str(exc),
                "result": {},
            })


def reference_training_size(value) -> int:
    """Never silently reduce a requested image size."""
    try:
        numeric = float(value)
        size = int(numeric)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("inspection_size must be a multiple of 32 between 256 and 1024")
    if numeric != size or size < 256 or size > 1024 or size % 32:
        raise ValueError("inspection_size must be a multiple of 32 between 256 and 1024")
    return size


def reference_region_ratio(value) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Minimum defect area must be between 0.0001 and 0.25")
    if isinstance(value, bool) or not np.isfinite(ratio) or not .0001 <= ratio <= .25:
        raise ValueError("Minimum defect area must be between 0.0001 and 0.25")
    return ratio


@app.post("/api/inspection/reference/train")
@serialized_workflow
def api_train_normal_reference(payload: dict = Body(...)) -> JSONResponse:
    """Start a durable, pollable normal-reference build from known-good images."""
    global REFERENCE_TRAIN_THREAD
    if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
        return JSONResponse({"ok": False, "error": "Stop camera detection before starting training."}, status_code=409)
    try:
        requested_ids = payload.get("dataset_ids")
        selected_ids = None
        if isinstance(requested_ids, list):
            selected_ids = {str(item).strip() for item in requested_ids if str(item).strip()}
            if not selected_ids:
                raise ValueError("Select at least one Known-good Dataset for this inspection recipe.")
        corpus = reviewed_known_good_corpus(selected_ids)
        validate_reference_recipe(corpus["datasets"])
        paths = list(corpus["paths"])
        source_datasets = [str(item["id"]) for item in corpus["datasets"]]
        if not source_datasets:
            return JSONResponse(
                {"ok": False, "error": "Select and confirm at least one Dataset as Known-good Reference before training."},
                status_code=409,
            )
        config = read_project_config()
        settings = dict(config.get("normal_reference") or {})
        settings["recipe_datasets"] = [{"id": item["id"], "label": item.get("label") or item["id"], "recipe": dict(item.get("recipe") or {})} for item in corpus["datasets"]]
        settings["inspection_area"] = checked_area(payload.get("inspection_area"))
        settings["inspection_areas"] = checked_areas(payload.get("inspection_areas"), settings["inspection_area"])
        anchor = payload.get("alignment_reference")
        if anchor is not None:
            allowed = {str(Path(item).resolve()) for item in paths}
            if not isinstance(anchor, dict) or str(Path(anchor.get("path", "")).resolve()) not in allowed:
                raise ValueError("Reference image must belong to the selected datasets")
            if file_sha256(Path(anchor["path"])) != anchor.get("sha256"):
                raise ValueError("Reference image changed; load the preview again")
        settings["alignment_reference"] = anchor
        settings.update({
            "inspection_size": reference_training_size(payload.get("inspection_size", settings.get("inspection_size") or 512)),
            "min_good_images": max(3, min(5000, int(payload.get("min_good_images") or settings.get("min_good_images") or 5))),
            "min_region_ratio": reference_region_ratio(payload.get("min_region_ratio", settings.get("min_region_ratio", .00035))),
        })
        if len(paths) < int(settings["min_good_images"]):
            raise ValueError(f"Known-good Dataset requires at least {settings['min_good_images']} images.")
        training_fingerprint = normal_reference_training_fingerprint(
            paths, settings, backbone_path=str(config.get("active_locator_model") or config.get("active_detector_model") or ""))
        current_reference = dict(config.get("normal_reference") or {})
        current_model = Path(str(current_reference.get("active_model") or ""))
        exact_fingerprint_match = (
            current_reference.get("training_fingerprint") == training_fingerprint
        )
        legacy_reference_is_current = (
            not current_reference.get("training_fingerprint")
            and current_model.is_file()
            and int(current_reference.get("training_images") or 0) == len(paths)
            and int(current_reference.get("inspection_size") or 0) == int(settings["inspection_size"])
            and int(current_reference.get("min_good_images") or 0) == int(settings["min_good_images"])
            and abs(float(current_reference.get("min_region_ratio") or 0) - float(settings["min_region_ratio"])) < 1e-9
            and str(current_reference.get("anomaly_engine") or "") == "deep_patch"
            and int(current_reference.get("pipeline_version") or 0) == int(DeepPatchReference.VERSION)
            and checked_area(current_reference.get("inspection_area")) == settings["inspection_area"]
            and checked_areas(current_reference.get("inspection_areas"), current_reference.get("inspection_area")) == settings["inspection_areas"]
            and current_reference.get("alignment_reference") == settings.get("alignment_reference")
            and all(path.stat().st_mtime <= current_model.stat().st_mtime for path in paths)
        )
        if (
            current_model.is_file()
            and list(current_reference.get("source_datasets") or []) == source_datasets
            and (exact_fingerprint_match or legacy_reference_is_current)
        ):
            return JSONResponse({
                "ok": True,
                "accepted": False,
                "unchanged": True,
                "message": "Dataset and training settings have not changed. The active Reference is already current.",
                "quality_metrics": dict(current_reference.get("quality_metrics") or {}),
                "corpus": {key: value for key, value in corpus.items() if key != "paths"},
            })
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    with REFERENCE_TRAIN_LOCK:
        if REFERENCE_TRAIN_STATE.get("running"):
            return JSONResponse({"ok": False, "error": "Reference training is already running."}, status_code=409)
        with TRAIN_LOCK:
            refresh_train_state_locked()
            if TRAIN_PROCESS is not None:
                return JSONResponse({"ok": False, "error": "Another training job is running."}, status_code=409)
        REFERENCE_TRAIN_STOP_EVENT.clear()
        job_id = f"reference-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        REFERENCE_TRAIN_STATE.update({
            "job_id": job_id,
            "running": True,
            "status": "training",
            "stage": "queued",
            "stage_detail": "queued",
            "progress": 0.0,
            "epoch": 0,
            "epochs": 0,
            "last_progress_at": time.time(),
            "total_images": len(paths),
            "source_images": int(corpus["source_images"]),
            "duplicates_removed": int(corpus["duplicates_removed"]),
            "datasets": source_datasets,
            "started_at": time.time(),
            "finished_at": None,
            "error": "",
            "dataset": "reviewed_known_good",
            "result": {},
        })
        try:
            REFERENCE_TRAIN_THREAD = vision_service.runtime.ownership.start_thread(
                run_normal_reference_training,
                (source_datasets, paths, settings, training_fingerprint, int(corpus["source_images"]), int(corpus["duplicates_removed"])),
                daemon=True,name='normal-reference-training')
        except Exception as exc:
            REFERENCE_TRAIN_THREAD = None
            REFERENCE_TRAIN_STATE.update({"running": False, "status": "failed", "stage": "failed",
                                          "error": str(exc), "finished_at": time.time()})
            return JSONResponse({"ok": False, "error": "Could not start Reference worker."}, status_code=500)
        state = dict(REFERENCE_TRAIN_STATE)
    return JSONResponse({"ok": True, "accepted": True, "state": state}, status_code=202)


@app.get("/api/inspection/reference/corpus")
def api_normal_reference_corpus() -> JSONResponse:
    corpus = reviewed_known_good_corpus()
    return JSONResponse({"ok": True, **{key: value for key, value in corpus.items() if key != "paths"}})


@app.post("/api/inspection/reference/corpus/preview")
def api_reference_corpus_preview(payload: dict = Body(...)) -> JSONResponse:
    ids = payload.get("dataset_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        return JSONResponse({"ok": False, "error": "dataset_ids must be a list of Dataset IDs"}, status_code=400)
    selected = {item.strip() for item in ids if item.strip()}
    corpus = reviewed_known_good_corpus(selected)
    available = {item["id"] for item in corpus["datasets"]}
    issue = ""
    if selected != available:
        issue = "บาง Dataset ถูกลบหรือยังไม่ได้ยืนยัน Known-good กรุณาตรวจรายการที่เลือก"
    else:
        try:
            validate_reference_recipe(corpus["datasets"])
        except ValueError as exc:
            issue = str(exc)
    return JSONResponse({"ok": True, "selection_valid": not issue, "selection_error": issue,
                         **{key: value for key, value in corpus.items() if key != "paths"}})


@app.post("/api/inspection/reference/area-preview")
@serialized_workflow
def api_area_preview(payload: dict = Body(...)):
    blocked = job_change_blocked()
    if blocked is not None:
        return blocked
    try:
        ids = payload.get("dataset_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x,str) for x in ids):
            raise ValueError("Select datasets first")
        corpus = reviewed_known_good_corpus(set(ids))
        if set(ids) != {item["id"] for item in corpus["datasets"]}:
            raise ValueError("Dataset selection has changed")
        validate_reference_recipe(corpus["datasets"])
        locator, class_id = get_shared_locator(read_project_config())
        for path in sorted(corpus["paths"], key=str):
            path = Path(path)
            image = cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                continue
            located = detector.analyze_primary_object(locator,class_id,image)
            if located is None:
                continue
            x1,y1,x2,y2 = [int(round(float(v))) for v in located["box"]]
            height,width = image.shape[:2]
            if x1<=1 or y1<=1 or x2>=width-1 or y2>=height-1:
                continue
            roi = cv2.resize(image[y1:y2,x1:x2],(512,512),interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".png",roi)
            if not ok:
                raise ValueError("Could not create reference preview")
            return JSONResponse({"ok":True,"reference":{"path":str(path.resolve()),"sha256":file_sha256(path),"box":[x1,y1,x2,y2]},
                "preview":"data:image/png;base64,"+base64.b64encode(encoded).decode('ascii')})
        raise ValueError("No complete part found in the selected datasets")
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return JSONResponse({"ok":False,"error":str(exc)},status_code=400)


@app.get("/api/inspection/reference/validation-samples")
def api_normal_reference_validation_samples(limit: int = Query(12, ge=3, le=24)) -> JSONResponse:
    """Replay this model's recorded validation images and ROIs, with hash checks."""
    config = read_project_config()
    runtime = normal_reference_runtime_status(config)
    if not runtime["ready"]:
        return JSONResponse({"ok": False, "error": runtime["reason"]}, status_code=409)
    reference_config = dict(config.get("normal_reference") or {})
    try:
        manifest = read_json(Path(runtime["model"]).with_suffix(".training.json"))
        entries = manifest.get("validation") or []
        if not entries:
            raise ValueError("This model has no saved test images")
        for entry in entries:
            path = Path(entry["path"]).resolve()
            if not path.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError("Test image is outside the project")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Test image changed: {path.name}")
            box = entry["box"]
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and np.isfinite(x) for x in box):
                raise ValueError("Saved test area is invalid")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": f"Saved test set unavailable: {exc}"}, status_code=409)
    sample_count = min(limit, len(entries))
    indices = np.linspace(0, len(entries) - 1, sample_count, dtype=int)
    inspector = load_inspection_reference(Path(runtime["model"]))
    samples: list[dict] = []
    for index in indices:
        entry = entries[int(index)]
        path = Path(entry["path"])
        image = read_image_file(path)
        sample_box = tuple(entry["box"])
        result = inspector.inspect(image, object_box=sample_box)
        annotated = inspector.draw_result(
            image, result, object_box=sample_box, object_label="Part", object_confidence=None,
        )
        preview_width = min(640, annotated.shape[1])
        preview_height = max(1, int(round(annotated.shape[0] * preview_width / annotated.shape[1])))
        preview = cv2.resize(annotated, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        preview_ok, preview_bytes = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 84])
        samples.append({
            "name": path.name,
            "dataset": "saved_test_set",
            "url": f"data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode('ascii')}" if preview_ok else None,
            "annotated_image": (
                f"data:image/jpeg;base64,{base64.b64encode(preview_bytes).decode('ascii')}"
                if preview_ok else None
            ),
            "decision": "REVIEW" if result.review_required and result.status != "NG" else result.status,
            "anomaly_score": result.anomaly_score,
            "threshold": result.threshold,
            "confidence": result.confidence,
            "input_issue": result.input_issue,
        })
        if len(samples) >= sample_count:
            break
    return JSONResponse({
        "ok": True,
        "evaluation_scope": "saved_model_validation_replay",
        "independent_metric": False,
        "validation_images": len(entries),
        "samples": samples,
    })


@app.get("/api/inspection/reference/status")
def api_normal_reference_status() -> JSONResponse:
    """Return only the active generic inspection contract; legacy YOLO runs are archival."""
    config = read_project_config()
    reference = dict(config.get("normal_reference") or {})
    runtime = normal_reference_runtime_status(config)
    model_path = Path(runtime["model"]) if runtime["model"] else None
    source_dataset = runtime["source_dataset"]
    folder_record = None
    if source_dataset:
        try:
            folder = dataset_folder_path(source_dataset)
            if folder.is_dir():
                folder_record = dataset_folder_record(folder)
        except ValueError:
            folder_record = None
    brand_enabled = bool(config.get("use_classifier")) and Path(
        str(config.get("active_brand_model") or "")
    ).is_file()
    with REFERENCE_TRAIN_LOCK:
        training_state = dict(REFERENCE_TRAIN_STATE)
    return JSONResponse({
        "ok": True,
        "strategy": "normal_reference",
        "anomaly_engine": runtime.get("anomaly_engine", "statistical_reference"),
        "ready": runtime["ready"],
        "model": runtime["model"],
        "reason": runtime["reason"],
        "source_dataset": source_dataset,
        "source_datasets": list(reference.get("source_datasets") or []),
        "source_images": int(reference.get("source_images") or reference.get("training_images") or 0),
        "duplicates_removed": int(reference.get("duplicates_removed") or 0),
        "training_images": int(reference.get("training_images") or (folder_record or {}).get("images") or 0),
        "trained_at": str(reference.get("trained_at") or ""),
        "inspection_size": int(runtime.get("inspection_size") or reference.get("inspection_size") or 512),
        "min_good_images": int(reference.get("min_good_images") or 5),
        "decisions": ["OK", "NG"],
        "ng_localization": "red_bounding_box",
        "confidence": "normality_confidence",
        "brand_identification": "enabled" if brand_enabled else "optional",
        "workflow": "ok_first_hybrid",
        "modules": {
            "part_locator": bool(active_locator_model_path(config)),
            "normal_reference": runtime["ready"],
            "legacy_defect_assist": False,
            "brand_classifier": brand_enabled,
        },
        "quality_metrics": current_reference_quality_metrics(reference, model_path),
        "dataset": folder_record,
        "training": training_state,
    })


@app.get("/api/inspection/parts")
def api_part_results(limit: int = Query(30, ge=1, le=100)):
    folder = PROJECT_ROOT / "03_dataset" / "inspection_cycles"
    rows = []
    errors = 0
    for path in sorted(folder.glob("part_*.json"), key=lambda item:item.stat().st_mtime, reverse=True)[:limit]:
        try:
            item = read_json(path)
            rows.append({key:value for key,value in item.items() if key != "evidence_result"})
        except (OSError,ValueError,TypeError):
            errors += 1
    return JSONResponse({"ok":True,"parts":rows,"unreadable":errors})


@app.get("/api/inspection/parts/{part_id}/image")
def api_part_image(part_id: str):
    if not re.fullmatch(r"part_[0-9a-f]{32}",part_id):
        return JSONResponse({"ok":False,"error":"Invalid part ID"},status_code=400)
    path = PROJECT_ROOT / "03_dataset" / "inspection_cycles" / (part_id+".jpg")
    if not path.is_file():
        return JSONResponse({"ok":False,"error":"Image not found"},status_code=404)
    return FileResponse(path,media_type="image/jpeg")


@app.get("/api/inspection/reviews")
def api_inspection_reviews(status: str = Query("")) -> JSONResponse:
    """List captured NG events; they remain pending until a human reviews them."""
    allowed = {"", "pending", "confirmed_ng", "false_alarm", "needs_review"}
    if status not in allowed:
        return JSONResponse({"ok": False, "error": "Invalid review status."}, status_code=400)
    records = inspection_review_records(status)
    return JSONResponse({"ok": True, "count": len(records), "reviews": records})


@app.get("/api/inspection/reviews/{review_id}/{image_kind}")
def api_inspection_review_image(review_id: str, image_kind: str) -> Response:
    clean_id = safe_name(review_id)
    if clean_id != review_id or image_kind not in {"source", "preview", "heatmap"}:
        return json_response_error("Invalid review image.", 400)
    image_path = INSPECTION_REVIEW_DIR / clean_id / f"{image_kind}.jpg"
    if not image_path.is_file():
        return json_response_error("Review image not found.", 404)
    return FileResponse(image_path, media_type="image/jpeg")


@app.post("/api/inspection/reviews/{review_id}")
def api_update_inspection_review(review_id: str, payload: dict = Body(...)) -> JSONResponse:
    """Record a human verdict without changing the immutable captured image."""
    clean_id = safe_name(review_id)
    if clean_id != review_id:
        return JSONResponse({"ok": False, "error": "Invalid review id."}, status_code=400)
    manifest = INSPECTION_REVIEW_DIR / clean_id / "review.json"
    if not manifest.is_file():
        return JSONResponse({"ok": False, "error": "Review not found."}, status_code=404)
    status = str(payload.get("status") or "").strip()
    if status not in {"confirmed_ng", "false_alarm", "needs_review"}:
        return JSONResponse({"ok": False, "error": "Invalid review status."}, status_code=400)
    record = read_json(manifest)
    record.update({
        "status": status,
        "reviewed_at": now_iso(),
        "note": str(payload.get("note") or "").strip()[:1000],
    })
    write_json(manifest, record)
    return JSONResponse({"ok": True, "review": record})


def cvat_connection_config() -> dict:
    base_url = str(os.environ.get("CVAT_URL") or "http://localhost:8080").strip().rstrip("/")
    token = str(os.environ.get("CVAT_ACCESS_TOKEN") or os.environ.get("CVAT_TOKEN") or "").strip()
    project_text = str(os.environ.get("CVAT_PROJECT_ID") or "").strip()
    return {
        "configured": bool(base_url and token),
        "base_url": base_url,
        "token": token,
        "project_id": int(project_text) if project_text.isdigit() else None,
    }


def cvat_request(path: str, *, method: str = "GET", payload: dict | None = None,
                 body: bytes | None = None, content_type: str = "application/json", timeout: float = 30.0):
    config = cvat_connection_config()
    if not config["configured"]:
        raise RuntimeError("CVAT is not configured. Set CVAT_URL and CVAT_ACCESS_TOKEN before starting the server.")
    data = body if body is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
    token = config["token"]
    authorization = token if token.lower().startswith(("token ", "bearer ")) else f"Token {token}"
    request = UrlRequest(
        f"{config['base_url']}{path}",
        data=data,
        method=method,
        headers={"Accept": "application/vnd.cvat+json", "Authorization": authorization, "Content-Type": content_type},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            return json.loads(response_body.decode("utf-8")) if response_body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"CVAT returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to CVAT: {exc.reason}") from exc


CVAT_HEALTH_CACHE = {"key": None, "expires": 0.0, "result": {}}
CVAT_HEALTH_LOCK = threading.Lock()


def cvat_connection_health() -> dict:
    """Check authentication, not merely whether connection settings exist."""
    config = cvat_connection_config()
    if not config["configured"]:
        return {"connected": False, "connection_status": "not_configured", "connection_error": "CVAT is not configured."}
    key = (config["base_url"], config["token"])
    with CVAT_HEALTH_LOCK:
        if CVAT_HEALTH_CACHE["key"] == key and time.monotonic() < CVAT_HEALTH_CACHE["expires"]:
            return dict(CVAT_HEALTH_CACHE["result"])
        try:
            user = cvat_request("/api/users/self", timeout=3.0)
            if not isinstance(user, dict) or not user.get("id"):
                raise ValueError("Unexpected authentication response")
            result = {"connected": True, "connection_status": "connected", "connection_error": ""}
        except (RuntimeError, OSError, ValueError):
            result = {"connected": False, "connection_status": "unavailable", "connection_error": "Cannot connect or authenticate to CVAT. Check server and credentials."}
        CVAT_HEALTH_CACHE.update(key=key, expires=time.monotonic() + 15.0, result=result)
        return dict(result)


def cvat_multipart(images: list[Path]) -> tuple[bytes, str]:
    boundary = f"----inspection-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
        ])

    field("image_quality", "95")
    field("sorting_method", "natural")
    for frame_index, image_path in enumerate(images):
        upload_name = f"{frame_index:06d}_{safe_name(image_path.stem)}{image_path.suffix.lower()}"
        mime = "image/png" if image_path.suffix.lower()==".png" else "image/jpeg"
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"client_files[{frame_index}]\"; filename=\"{upload_name}\"\r\nContent-Type: {mime}\r\n\r\n".encode("utf-8")
        )
        chunks.append(image_path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def wait_for_cvat_request(request_id: str, timeout_seconds: float = 120.0) -> None:
    if not request_id:
        return
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = cvat_request(f"/api/requests/{quote(request_id, safe='')}", timeout=15.0)
        status = str(state.get("status") or state.get("state") or "").lower()
        if status in {"finished", "completed"}:
            return
        if status in {"failed", "canceled", "cancelled"}:
            raise RuntimeError(f"CVAT media processing failed: {state.get('message') or status}")
        time.sleep(1.0)
    raise RuntimeError("CVAT media processing did not finish within 120 seconds.")


def cvat_task_labels(task_id: int, task: dict) -> list[dict]:
    labels=task.get("labels")
    if isinstance(labels,list):
        return labels
    labels=[];page=1
    while True:
        batch=cvat_request(f"/api/labels?task_id={int(task_id)}&page={page}&page_size=100")
        labels.extend(batch["results"])
        if not batch.get("next"):
            return labels
        page+=1


def add_cvat_draft_annotations(task_id: int, records: list[dict]) -> int:
    task = cvat_request(f"/api/tasks/{task_id}")
    labels = cvat_task_labels(task_id,task)
    ng_label = next((label for label in labels if str(label.get("name") or "").strip().lower() == "ng"), None)
    if not ng_label:
        return 0
    shapes: list[dict] = []
    for frame, record in enumerate(records):
        regions = list(record.get("regions") or [])
        if not regions:
            raw_box = (record.get("object") or {}).get("box") or []
            if len(raw_box) == 4:
                x1, y1, x2, y2 = [float(value) for value in raw_box]
                regions = [{"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}]
        for region in regions:
            x1, y1 = float(region.get("x") or 0), float(region.get("y") or 0)
            x2 = x1 + max(1.0, float(region.get("width") or 0))
            y2 = y1 + max(1.0, float(region.get("height") or 0))
            shapes.append({
                "type": "rectangle",
                "frame": frame,
                "label_id": int(ng_label["id"]),
                "points": [x1, y1, x2, y2],
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "attributes": [],
                "source": "auto",
            })
    if shapes:
        cvat_request(
            f"/api/tasks/{task_id}/annotations/",
            method="PUT",
            payload={"version": 0, "tags": [], "shapes": shapes, "tracks": []},
            timeout=60.0,
        )
    return len(shapes)


def sync_confirmed_reviews_to_cvat(
    review_ids: list[str] | None = None,
    *,
    include_pending: bool = False,
) -> dict:
    with CVAT_BRIDGE_LOCK:
        return _sync_confirmed_reviews_to_cvat(review_ids, include_pending=include_pending)


def _sync_confirmed_reviews_to_cvat(
    review_ids: list[str] | None = None,
    *,
    include_pending: bool = False,
) -> dict:
    """Persist each remote step; resume known tasks without creating duplicates.

    A timeout during a mutating HTTP call has an unknown outcome. Those cases
    require reconciliation, rather than blindly replaying a POST every 5 seconds.
    """
    requested = {safe_name(item) for item in (review_ids or []) if safe_name(item)}
    records = inspection_review_records("confirmed_ng")
    if include_pending:
        records.extend(inspection_review_records("pending"))
    if requested:
        records = [record for record in records if record.get("id") in requested]
    records = [record for record in records
               if (record.get("cvat_sync") or {}).get("status") not in {"synced", "annotations_imported", "needs_attention", "creating", "uploading", "annotating"}]
    config = cvat_connection_config()
    if not records:
        return {"ok": True, "synced": 0, "message": "No retryable confirmed NG events."}
    if not config["configured"]:
        raise RuntimeError("CVAT is not configured. Set CVAT_URL and CVAT_ACCESS_TOKEN before starting the server.")

    # Resume a persisted batch with its original frame order, including members
    # not named in an interactive retry. New records form a separate batch.
    retry = next((record for record in records if (record.get("cvat_sync") or {}).get("task_id")), None)
    if retry:
        task_id = int(retry["cvat_sync"]["task_id"])
        records = sorted([record for record in inspection_review_records("")
                          if (record.get("cvat_sync") or {}).get("task_id") == task_id],
                         key=lambda record: int(record["cvat_sync"].get("frame_index", 0)))
        sync = dict(records[0]["cvat_sync"])
    else:
        records = records[:100]
        sync = {}
        task_id = None
    images = [Path(str(record.get("source_image") or "")) for record in records]
    if not all(path.is_file() for path in images):
        raise RuntimeError("One or more confirmed NG source images are missing.")

    def persist(stage: str, **values) -> None:
        sync.update(values)
        sync.update({"status": stage, "updated_at": now_iso()})
        # One local transaction keeps review edits from being lost during save.
        with JSON_WRITE_LOCK:
            for frame_index, record in enumerate(records):
                manifest = INSPECTION_REVIEW_DIR / str(record["id"]) / "review.json"
                latest = read_json(manifest, record)
                latest["cvat_sync"] = {**sync, "frame_index": frame_index}
                write_json(manifest, latest)

    try:
        if task_id is None:
            task_payload = {"name": f"{safe_name(read_project_config().get('project_name') or 'Inspection')} NG {datetime.now():%Y-%m-%d %H:%M:%S}"}
            if config["project_id"] is not None:
                task_payload["project_id"] = config["project_id"]
            else:
                shape_type = "any" if any(r.get("inspection_engine")=="area_recipe" for r in records) else "rectangle"
                task_payload["labels"] = [{"name": "NG", "color": "#ff3344", "type": shape_type}]
            persist("creating")
            task = cvat_request("/api/tasks", method="POST", payload=task_payload)
            task_id = int(task["id"])
            persist("created", task_id=task_id, task_url=f"{config['base_url']}/tasks/{task_id}")
        if sync.get("status") == "created":
            multipart, content_type = cvat_multipart(images)
            persist("uploading")
            upload = cvat_request(f"/api/tasks/{task_id}/data/", method="POST", body=multipart,
                                  content_type=content_type, timeout=120.0)
            persist("processing", request_id=str(upload.get("rq_id") or "") if isinstance(upload, dict) else "")
        if sync.get("status") == "processing":
            wait_for_cvat_request(str(sync.get("request_id") or ""))
            persist("drafts_pending")
        if sync.get("status") == "drafts_pending":
            # Draft writes are mutating too: do not replay an ambiguous failure.
            persist("annotating")
            draft_boxes = add_cvat_draft_annotations(task_id, records)
            persist("synced", draft_boxes=draft_boxes, synced_at=now_iso(), error="")
        if sync.get("status") != "synced":
            raise RuntimeError("CVAT transfer needs reconciliation before retrying.")
    except Exception as exc:
        stage = str(sync.get("status") or "")
        persist("needs_attention" if stage in {"creating", "uploading", "annotating"} else stage,
                error=str(exc)[:500], failed_at=now_iso())
        raise
    return {"ok": True, "synced": len(records), "task_id": task_id,
            "task_url": sync["task_url"], "draft_boxes": int(sync.get("draft_boxes") or 0)}


def cvat_annotation_label_map(task: dict) -> dict[int, str]:
    return {
        int(label["id"]): str(label.get("name") or "NG").strip() or "NG"
        for label in (task.get("labels") or [])
        if label.get("id") is not None
    }


def store_cvat_annotation(record: dict, shapes: list[dict], task: dict) -> dict:
    """Persist human CVAT boxes as an accumulating, train-ready NG archive."""
    review_id = safe_name(str(record.get("id") or ""))
    source = Path(str(record.get("source_image") or ""))
    if not review_id or not source.is_file():
        raise RuntimeError("Review source image is missing.")
    image = read_image_file(source)
    height, width = image.shape[:2]
    label_map = cvat_annotation_label_map(task)
    annotations: list[dict] = []
    for shape in shapes:
        if str(shape.get("type") or "") != "rectangle" or bool(shape.get("outside")):
            continue
        points = list(shape.get("points") or [])
        if len(points) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in points]
        x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        label_id = int(shape.get("label_id") or 0)
        annotations.append({
            "label": label_map.get(label_id, "NG"),
            "cvat_label_id": label_id,
            "box": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            "source": str(shape.get("source") or "manual"),
        })
    if not annotations:
        return record

    current_manifest = INSPECTION_REVIEW_DIR / review_id / "review.json"
    current_record = read_json(current_manifest, record)
    if (
        current_record.get("cvat_annotations") == annotations
        and (current_record.get("cvat_sync") or {}).get("status") == "annotations_imported"
    ):
        return current_record

    with JSON_WRITE_LOCK:
        image_dir = CVAT_ANNOTATED_DIR / "images"
        label_dir = CVAT_ANNOTATED_DIR / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image_target = image_dir / f"{review_id}.jpg"
        if not image_target.exists() or image_target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            shutil.copy2(source, image_target)

        class_path = CVAT_ANNOTATED_DIR / "classes.json"
        class_payload = read_json(class_path, {"names": []})
        class_names = [str(item) for item in (class_payload.get("names") or [])]
        for annotation in annotations:
            if annotation["label"] not in class_names:
                class_names.append(annotation["label"])
        write_json(class_path, {"names": class_names, "updated_at": now_iso()})

        yolo_rows: list[str] = []
        for annotation in annotations:
            x1, y1, x2, y2 = annotation["box"]
            class_id = class_names.index(annotation["label"])
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            yolo_rows.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        label_path = label_dir / f"{review_id}.txt"
        label_path.write_text("\n".join(yolo_rows) + "\n", encoding="utf-8")

        latest = read_json(INSPECTION_REVIEW_DIR / review_id / "review.json", record)
        sync = dict(latest.get("cvat_sync") or {})
        sync.update({
            "status": "annotations_imported",
            "annotation_count": len(annotations),
            "imported_at": now_iso(),
        })
        latest.update({
            "status": "confirmed_ng",
            "reviewed_at": latest.get("reviewed_at") or now_iso(),
            "cvat_sync": sync,
            "cvat_annotations": annotations,
            "annotation_image": str(image_target.resolve()),
            "annotation_label": str(label_path.resolve()),
        })
        write_json(INSPECTION_REVIEW_DIR / review_id / "review.json", latest)

        archive_records = []
        for manifest in INSPECTION_REVIEW_DIR.glob("*/review.json"):
            item = read_json(manifest, {})
            if item.get("cvat_annotations"):
                archive_records.append({
                    "review_id": item.get("id"),
                    "image": item.get("annotation_image"),
                    "label_file": item.get("annotation_label"),
                    "annotations": item.get("cvat_annotations"),
                    "cvat_task_id": (item.get("cvat_sync") or {}).get("task_id"),
                    "imported_at": (item.get("cvat_sync") or {}).get("imported_at"),
                })
        write_json(CVAT_ANNOTATED_DIR / "index.json", {
            "updated_at": now_iso(),
            "images": len(archive_records),
            "classes": class_names,
            "records": sorted(archive_records, key=lambda item: str(item.get("review_id") or "")),
        })
    return latest


def pull_cvat_annotations(review_ids: list[str] | None = None) -> dict:
    with CVAT_BRIDGE_LOCK:
        return _pull_cvat_annotations(review_ids)


def _pull_cvat_annotations(review_ids: list[str] | None = None) -> dict:
    """Pull operator annotations from CVAT and merge them into the local NG archive."""
    requested = {safe_name(item) for item in (review_ids or []) if safe_name(item)}
    candidates = inspection_review_records("")
    candidates = [item for item in candidates if item.get('inspection_engine')!='area_recipe'
                  and not (str(item.get('id','')).startswith('event_') and item.get('recipe_hash'))]
    if requested:
        candidates = [item for item in candidates if item.get("id") in requested]
    task_records: dict[int, list[dict]] = {}
    for record in candidates:
        sync = dict(record.get("cvat_sync") or {})
        task_id = sync.get("task_id")
        if task_id is None or sync.get("status") not in {"synced", "annotations_imported"}:
            continue
        task_records.setdefault(int(task_id), []).append(record)

    imported = 0
    boxes = 0
    for task_id, records in task_records.items():
        task = cvat_request(f"/api/tasks/{task_id}")
        payload = cvat_request(f"/api/tasks/{task_id}/annotations/")
        task_shapes = list(payload.get("shapes") or [])
        label_map = cvat_annotation_label_map(task)
        for record in records:
            frame_index = int((record.get("cvat_sync") or {}).get("frame_index") or 0)
            frame_shapes = [
                shape for shape in task_shapes
                if int(shape.get("frame") or 0) == frame_index and not bool(shape.get("outside"))
            ]
            # Ignore untouched machine drafts. A manually created/edited shape or
            # a user-defined defect label is treated as the operator verdict.
            reviewed_shapes = [
                shape for shape in frame_shapes
                if str(shape.get("source") or "").lower() != "auto"
                or label_map.get(int(shape.get("label_id") or 0), "NG").strip().lower() not in {"ng", "defect"}
            ]
            if not reviewed_shapes:
                continue
            latest = store_cvat_annotation(record, reviewed_shapes, task)
            imported += 1
            boxes += len(latest.get("cvat_annotations") or [])
    return {"ok": True, "tasks_checked": len(task_records), "imported": imported, "boxes": boxes}


@app.get("/api/inspection/cvat/status")
def api_cvat_status() -> JSONResponse:
    config = cvat_connection_config()
    archive = read_json(CVAT_ANNOTATED_DIR / "index.json", {})
    return JSONResponse({
        "ok": True,
        "configured": config["configured"],
        **cvat_connection_health(),
        "base_url": config["base_url"],
        "project_id": config["project_id"],
        "mode": "direct_sync" if config["configured"] else "zip_export",
        "automatic": False,
        "attention_required": sum(
            1 for record in inspection_review_records("")
            if (record.get("cvat_sync") or {}).get("status") in {"needs_attention", "creating", "uploading", "annotating"}
        ),
        "annotated_images": int(archive.get("images") or 0),
        "annotation_classes": list(archive.get("classes") or []),
    })


@app.post("/api/inspection/cvat/sync")
def api_sync_inspection_reviews_to_cvat(payload: dict = Body(default={})) -> JSONResponse:
    if not payload.get("review_ids"):
        return JSONResponse({"ok": False, "error": "Select the review images to send to CVAT."}, status_code=400)
    try:
        result = sync_confirmed_reviews_to_cvat([str(item) for item in (payload.get("review_ids") or [])])
        return JSONResponse(result)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)


@app.post("/api/inspection/cvat/pull")
def api_pull_inspection_annotations_from_cvat(payload: dict = Body(default={})) -> JSONResponse:
    try:
        result = pull_cvat_annotations([str(item) for item in (payload.get("review_ids") or [])])
        return JSONResponse(result)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)


@app.post("/api/inspection/cvat/export")
def api_export_inspection_reviews_to_cvat(payload: dict = Body(default={})) -> Response:
    """Create a CVAT-ready image archive from human-confirmed NG captures."""
    requested_ids = {safe_name(str(item)) for item in (payload.get("review_ids") or [])}
    records = inspection_review_records("confirmed_ng")
    if requested_ids:
        records = [record for record in records if record.get("id") in requested_ids]
    if not records:
        return json_response_error("No confirmed NG images are ready for CVAT export.", 409)
    CVAT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = CVAT_EXPORT_DIR / f"confirmed_ng_{timestamp_name()}.zip"
    manifest_rows: list[dict] = []
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            review_id = str(record["id"])
            source_path = INSPECTION_REVIEW_DIR / review_id / "source.jpg"
            if not source_path.is_file():
                continue
            archive.write(source_path, f"images/{review_id}.jpg")
            manifest_rows.append({
                "image": f"images/{review_id}.jpg",
                "review_id": review_id,
                "captured_at": record.get("created_at"),
                "brand": record.get("brand"),
                "normality_confidence": record.get("normality_confidence"),
                "anomaly_score": record.get("anomaly_score"),
            })
        archive.writestr(
            "manifest.json",
            json.dumps({"task": "Generic NG annotation", "images": manifest_rows}, ensure_ascii=False, indent=2),
        )
    if not manifest_rows:
        export_path.unlink(missing_ok=True)
        return json_response_error("Confirmed NG source images were not found.", 404)
    return FileResponse(export_path, media_type="application/zip", filename=export_path.name)


@app.post("/api/inspection/inspect")
async def api_inspect_against_normal(file: UploadFile = File(...)) -> JSONResponse:
    """Inspect one image and return OK/NG, confidence, and red-box regions."""
    config = read_project_config()
    reference = dict(config.get("normal_reference") or {})
    model_path = Path(str(reference.get("active_model") or ""))
    if not model_path.is_file():
        return JSONResponse({"ok": False, "error": "Train a normal reference before inspection."}, status_code=409)
    raw = await file.read(MAX_INSPECTION_IMAGE_BYTES + 1)
    if not raw or len(raw) > MAX_INSPECTION_IMAGE_BYTES:
        return JSONResponse({"ok": False, "error": "Invalid or oversized image."}, status_code=400)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"ok": False, "error": "Unsupported image file."}, status_code=400)
    # Decode/inference and model I/O must not monopolize the ASGI event loop.
    return await run_in_threadpool(inspect_uploaded_image, image, config, reference,
                                   model_path, file.filename or "upload")


def inspect_uploaded_image(image: np.ndarray, config: dict, reference: dict,
                           model_path: Path, source_name: str) -> JSONResponse:
    """Synchronous inference runs in the server's bounded worker pool."""
    try:
        inspector = RecipeBoundInspector(load_inspection_reference(model_path), model_path, reference)
        locator_model, locator_class_id = detector.load_primary_object_locator(config)
        located = detector.analyze_primary_object(locator_model, locator_class_id, image)
        if located is None:
            return JSONResponse(
                {"ok": True, "decision": "NO_PART", "confidence": 0.0, "regions": []},
                status_code=200,
            )
        brand_name = None
        brand_confidence = None
        if bool(config.get("use_classifier")) and str(config.get("active_brand_model") or "").strip():
            try:
                brand_model = load_brand_model()
                brand_result = classify_product_brand(brand_model, image, located["box"])
                if brand_result:
                    brand_name, brand_confidence = brand_result
            except Exception as exc:
                print(f"Optional brand classification skipped: {exc}")
        result = inspector.inspect(
            image,
            object_box=located["box"],
            brand=brand_name,
            brand_confidence=brand_confidence,
        )
        annotated = inspector.draw_result(
            image,
            result,
            object_box=located["box"],
            object_label=located["label"],
            object_confidence=located["confidence"],
        )
        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        payload = result.to_dict()
        payload.update({
            "ok": True,
            "decision": result.status,
            "part": {
                "label": located["label"],
                "confidence": round(float(located["confidence"]), 4),
                "box": [int(value) for value in located["box"]],
            },
            "evidence": {
                "normal_reference": True,
                "anomaly_engine": str(reference.get("anomaly_engine") or "deep_patch"),
                "pipeline_version": int(reference.get("pipeline_version") or getattr(inspector, "model_version", 0)),
                "recipe_id": str(reference.get("recipe_id") or "default"),
                "background_handling": str(reference.get("background_handling") or "part_locator_roi_only"),
                "legacy_defect_assist": False,
                "legacy_mark_shadow_confidence": round(float(located.get("defect_confidence") or 0.0), 4),
            },
            "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}" if ok else None,
        })
        payload["decision"] = "REVIEW" if result.review_required and result.status != "NG" else result.status
        if not result.input_issue and (result.status == "NG" or result.review_required):
            review = save_inspection_review(
                image,
                annotated,
                result,
                object_box=located["box"],
                object_label=located["label"],
                object_confidence=located["confidence"],
                source_name=source_name,
                reference=reference,
            )
            payload["review_id"] = review["id"]
        return JSONResponse(payload)
    except (ValueError, OSError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/train/start")
@serialized_workflow
def api_start_train(
    train_type: str = Form(...),
    epochs: int = Form(50),
    imgsz: int = Form(640),
    resume: bool = Form(False),
    target_class: str = Form(""),
    target_classes: str = Form(""),
    project_name: str = Form(""),
    project_brand: str = Form(""),
    project_model: str = Form(""),
    augmentation_mode: str = Form("recommended"),
    augmentation_config: str = Form("{}"),
) -> JSONResponse:
    """เริ่ม train เป็น background process เพื่อไม่ให้หน้าเว็บค้าง."""
    global TRAIN_PROCESS, TRAIN_LOG_HANDLE

    if train_type not in {"airbag_detector", "brand_classifier"}:
        return JSONResponse({"ok": False, "error": "Unknown train type."}, status_code=400)

    with TRAIN_LOCK:
        refresh_train_state_locked()
        if TRAIN_PROCESS is not None or REFERENCE_TRAIN_STATE.get("running"):
            return JSONResponse(
                {"ok": False, "error": "Training is already running.", "state": TRAIN_STATE},
                status_code=409,
            )
    if WORKER_THREAD is not None and WORKER_THREAD.is_alive():
        return JSONResponse(
            {
                "ok": False,
                "error": "Stop camera detection before starting training to avoid unstable CPU/GPU and memory usage.",
            },
            status_code=409,
        )

    epochs = max(1, min(int(epochs), 300))
    imgsz = max(160, min(int(imgsz), 1280))
    try:
        augmentation_values = json.loads(augmentation_config or "{}")
    except json.JSONDecodeError:
        return JSONResponse({"ok": False, "error": "Invalid augmentation configuration."}, status_code=400)
    augmentation_profile = normalize_augmentation_profile(
        {"mode": augmentation_mode, "values": augmentation_values}
    )
    project_payload = {
        "project_name": project_name,
        "brand": project_brand,
        "model": project_model,
        "last_train_type": train_type,
        "augmentation_profile": augmentation_profile,
    }
    if target_classes or target_class:
        project_payload["target_classes_text"] = target_classes or target_class
        project_payload["primary_class"] = target_class
    preflight = train_preflight(train_type)
    if not preflight["ok"]:
        return JSONResponse(
            {
                "ok": False,
                "error": "ข้อมูลยังไม่พร้อม train",
                "preflight": preflight,
            },
            status_code=400,
        )
    if train_type == "airbag_detector":
        detection_stats = detection_train_stats()
        if detection_stats["exports"] <= 0:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "ยังไม่มี Detection dataset ที่พร้อม train: ต้องมี folder ประเภท Detection ที่มี data.yaml/labels ก่อน",
                    "readiness": detection_stats,
                },
                status_code=400,
            )

    # The frontend already saves the Project/Class Contract before this call.
    # Do not mutate config again until every backend guard has passed; otherwise
    # a rejected start request could leave a misleading "training" state.
    project_config = write_project_config(
        {
            **project_payload,
            "last_train_status": "training",
        }
    )
    with TRAIN_LOCK:
        refresh_train_state_locked()
        if TRAIN_PROCESS is not None:
            return JSONResponse(
                {"ok": False, "error": "Training is already running.", "state": TRAIN_STATE},
                status_code=409,
            )

        TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = TRAIN_LOG_DIR / f"{timestamp_name()}_{train_type}.log"

        if train_type == "airbag_detector":
            clean_target_class = target_class.strip()
            clean_target_classes = target_classes.strip()
            command = [
                str(PYTHON_EXE),
                str(TRAINING_CODE_DIR / "train_from_yolov8_exports.py"),
                "--epochs",
                str(epochs),
                "--imgsz",
                str(imgsz),
            ]
            if clean_target_classes:
                command.extend(["--class-names", clean_target_classes])
            elif clean_target_class:
                command.extend(["--class-name", clean_target_class])
            command.extend(
                [
                    "--class-aliases-json",
                    json.dumps(project_config.get("class_aliases") or {}, ensure_ascii=False),
                ]
            )
            if resume:
                command.append("--resume")
            else:
                command.append("--force-rebuild")
                command.extend(
                    [
                        "--augmentation-profile",
                        augmentation_profile["mode"],
                        "--augmentation-json",
                        json.dumps(augmentation_profile["values"], ensure_ascii=True),
                    ]
                )
        else:
            command = [
                str(PYTHON_EXE),
                str(BRAND_CODE_DIR / "train_brand_classifier.py"),
                "--epochs",
                str(epochs),
                "--imgsz",
                str(min(imgsz, 320)),
                "--force-rebuild",
                "--augmentation-profile",
                augmentation_profile["mode"],
                "--augmentation-json",
                json.dumps(augmentation_profile["values"], ensure_ascii=True),
            ]

        TRAIN_LOG_HANDLE = log_path.open("w", encoding="utf-8", buffering=1)
        TRAIN_LOG_HANDLE.write("Command:\n")
        TRAIN_LOG_HANDLE.write(" ".join(command) + "\n\n")
        TRAIN_LOG_HANDLE.write("Project Config:\n")
        TRAIN_LOG_HANDLE.write(json.dumps(project_config, ensure_ascii=False, indent=2) + "\n\n")
        TRAIN_LOG_HANDLE.flush()

        try:
            TRAIN_PROCESS = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=TRAIN_LOG_HANDLE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            close_train_log_handle_locked()
            TRAIN_PROCESS = None
            write_project_config(
                {
                    "last_train_type": train_type,
                    "last_train_log": str(log_path),
                    "last_train_status": "failed_to_start",
                }
            )
            return JSONResponse(
                {"ok": False, "error": f"Failed to start training process: {exc}"},
                status_code=500,
            )
        TRAIN_STATE.update(
            {
                "running": True,
                "status": "training",
                "train_type": train_type,
                "started_at": time.time(),
                "finished_at": None,
                "stop_requested_at": None,
                "returncode": None,
                "log_path": str(log_path),
                "command": command,
                "project": project_config,
                "augmentation_profile": (
                    {"mode": "checkpoint", "values": {}}
                    if resume
                    else augmentation_profile
                ),
            }
        )
        try:
            vision_service.runtime.ownership.start_thread(wait_for_train_completion,(TRAIN_PROCESS,),
                daemon=True,name='legacy-training-completion')
        except Exception:
            stop_train_process_locked()
            raise

    return JSONResponse({"ok": True, "state": TRAIN_STATE})


@app.post("/api/train/stop")
def api_stop_train() -> JSONResponse:
    """หยุดงาน train ที่กำลังรันอยู่จากหน้าเว็บ."""
    with TRAIN_LOCK:
        if REFERENCE_TRAIN_STATE.get("running"):
            REFERENCE_TRAIN_STOP_EVENT.set()
            REFERENCE_TRAIN_STATE.update({"status": "stopping", "stage_detail": "Stop requested; finishing current operation."})
            return JSONResponse({"ok": True, "message": "Reference stop requested.", "state": dict(REFERENCE_TRAIN_STATE)})
        stopped, message = stop_train_process_locked()
        payload = dict(TRAIN_STATE)
        log_path = str(payload.get("log_path", ""))
        payload["project"] = read_project_config()
        payload["log_tail"] = read_log_tail(log_path)
        payload["loss_history"] = parse_loss_history(log_path)
    status_code = 200 if stopped else 409
    return JSONResponse({"ok": stopped, "message": message, "state": payload}, status_code=status_code)


@app.get("/api/train/status")
def api_train_status() -> JSONResponse:
    """ส่งสถานะ train ล่าสุดและ log ช่วงท้าย."""
    with TRAIN_LOCK:
        refresh_train_state_locked()
        payload = dict(TRAIN_STATE)
        project = read_project_config()
        reference_runtime = normal_reference_runtime_status(project)
        log_path = str(payload.get("log_path", ""))
        # The active workflow is Normal Reference inspection.  Old YOLO logs are
        # still available from the history endpoint, but must never masquerade as
        # the current job after an application restart.
        if (
            not log_path
            and not payload.get("running")
            and str(project.get("inspection_strategy") or "") != "normal_reference"
        ):
            latest_log = latest_train_log_path()
            if latest_log:
                log_path = str(latest_log)
                payload["log_path"] = log_path
                payload["historical_log"] = True
        if log_path and not payload.get("running"):
            inferred = infer_train_status_from_log(log_path)
            payload["status"] = inferred["status"]
            payload["returncode"] = inferred.get("returncode")
            payload["finished_at"] = inferred.get("finished_at") or payload.get("finished_at")
            payload["status_source"] = "log"
            payload["status_detail"] = inferred.get("status_detail", "")
            if not payload.get("train_type"):
                payload["train_type"] = infer_train_type_from_log_path(log_path)
        elif payload.get("running"):
            payload["status_source"] = "process"
            payload["status_detail"] = "backend มี process train ที่กำลังทำงานอยู่"
        elif str(project.get("inspection_strategy") or "") == "normal_reference":
            payload.update(
                {
                    "status": "idle",
                    "status_source": "normal_reference",
                    "status_detail": (
                        "Normal Reference พร้อมใช้งาน"
                        if reference_runtime["ready"]
                        else reference_runtime["reason"]
                    ),
                    "historical_log": False,
                    "log_path": "",
                }
            )
            log_path = ""
        payload["project"] = project
        payload["inspection_reference"] = reference_runtime
        payload["log_tail"] = read_log_tail(log_path)
        payload["loss_history"] = parse_loss_history(log_path)
        progress = parse_training_progress(
            log_path,
            payload.get("started_at"),
            str(payload.get("status") or ""),
        )
        payload["progress"] = progress
        if progress:
            elapsed_seconds = float(progress.get("elapsed_seconds") or 0)
            remaining_seconds = float(progress.get("remaining_seconds") or 0)
            payload["time_estimate"] = {
                "elapsed_seconds": round(elapsed_seconds, 2),
                "remaining_seconds": round(remaining_seconds, 2),
                "total_estimated_seconds": round(elapsed_seconds + remaining_seconds, 2),
                "confidence": progress.get("confidence", "low"),
                "source": "log_progress",
                "speed_text": progress.get("speed_text", ""),
            }
        else:
            payload["time_estimate"] = {}
    return JSONResponse(payload)


# ===== HTTP routes: Camera stream/status =====
# browser poll JPEG ล่าสุดแทน MJPEG stream; start/stop ต้อง idempotent เพื่อรองรับการกดซ้ำ
# และต้องไม่ทิ้ง camera handle เมื่อผู้ใช้ปิด tab กะทันหัน
@app.get("/start_stream")
def start_stream(
    camera_index: int | None = Query(None, ge=0, le=10),
    primary_confidence: float | None = Query(None, ge=0.01, le=0.99),
    secondary_confidence: float | None = Query(None, ge=0.01, le=0.99),
    debug_detection: bool = Query(False),
    use_classifier: bool | None = Query(None),
) -> JSONResponse:
    """เริ่มเปิดกล้องและ detect ใน worker thread."""
    if training_is_running():
        return JSONResponse(
            {
                "ok": False,
                "error": "Camera detection cannot start while training is running. Stop training first to avoid unstable CPU/GPU and memory usage.",
            },
            status_code=409,
        )
    config = read_project_config()
    camera_index = int(config.get("camera_index", detector.DEFAULT_CAMERA_INDEX) if camera_index is None else camera_index)
    primary_confidence = float(config.get("primary_confidence", detector.AIRBAG_CONFIDENCE_THRESHOLD) if primary_confidence is None else primary_confidence)
    secondary_confidence = float(config.get("secondary_confidence", detector.SECONDARY_CONFIDENCE_THRESHOLD) if secondary_confidence is None else secondary_confidence)
    use_classifier = bool(config.get("use_classifier", True) if use_classifier is None else use_classifier)
    detector.SECONDARY_CONFIDENCE_THRESHOLD = secondary_confidence
    write_project_config(
        {
            "camera_index": camera_index,
            "primary_confidence": primary_confidence,
            "secondary_confidence": secondary_confidence,
            "use_classifier": use_classifier,
        }
    )
    try:
        start_camera_worker(camera_index, primary_confidence, debug_detection, use_classifier)
    except RuntimeError as exc:
        return JSONResponse(
            {"ok": False, "status": "not_ready", "error": str(exc)},
            status_code=409,
        )
    return JSONResponse(
        {
            "ok": True,
            "status": "starting",
            "camera_index": camera_index,
            "primary_confidence": primary_confidence,
            "secondary_confidence": secondary_confidence,
            "use_classifier": use_classifier,
        }
    )


@app.get("/stop_stream", include_in_schema=False)
@app.post("/stop_stream")
def stop_stream() -> JSONResponse:
    """หยุด worker และปล่อยกล้อง."""
    stopped = stop_camera_worker()
    return JSONResponse({"ok": stopped, "status": "stopped" if stopped else "stopping"},
                        status_code=200 if stopped else 202)


@app.get("/latest_frame.jpg")
def latest_frame() -> Response:
    """คืนภาพล่าสุดเป็น JPEG พร้อม no-cache."""
    with STATE_LOCK:
        jpeg = LATEST_JPEG
    if jpeg is None:
        jpeg = encode_frame(make_status_frame("No frame", "Press Start Stream"))
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/status")
def status() -> JSONResponse:
    """คืนสถานะ worker ปัจจุบันให้ frontend ใช้แสดงผล."""
    config = read_project_config()
    runtime = normal_reference_runtime_status(config)
    with STATE_LOCK:
        payload = dict(STATE)
        payload["inspection_strategy"] = runtime["strategy"]
        payload["inspection_ready"] = runtime["ready"]
        payload["inspection_model"] = runtime["model"]
        payload["model_loaded"] = runtime["ready"]
        payload["model_error"] = "" if runtime["ready"] else runtime["reason"]
        payload["worker_alive"] = bool(WORKER_THREAD and WORKER_THREAD.is_alive())
        # A ready inspection model does not imply that the camera worker opened
        # successfully. Preserve a hardware/worker failure instead of replacing
        # it with the misleading generic "ready" state.
        # model_error used to be produced by an obsolete Airbag/Mark preload.
        # In generic mode, an authoritative ready runtime supersedes that stale
        # startup state; real camera errors remain visible until the next start.
        stale_legacy_error = (
            runtime["strategy"] == "normal_reference"
            and runtime["ready"]
            and payload.get("status") == "model_error"
        )
        worker_failed = payload.get("status") == "error" or (
            bool(payload.get("error")) and not stale_legacy_error
        )
        if not payload["worker_alive"] and runtime["ready"] and not worker_failed:
            payload["running"] = False
            payload["status"] = "ready"
            payload["error"] = ""
        elif not payload["worker_alive"] and not runtime["ready"]:
            payload["running"] = False
            payload["status"] = "not_ready"
            payload["error"] = runtime["reason"]
    return JSONResponse(payload)


@app.get("/api/legacy/health")
def legacy_health() -> JSONResponse:
    """ตรวจสถานะพื้นฐานโดยไม่เปิดกล้อง."""
    project_config = read_project_config()
    runtime = normal_reference_runtime_status(project_config)
    configured_classifier = str(project_config.get("active_brand_model") or "").strip()
    brand_model_path = Path(configured_classifier) if configured_classifier and Path(configured_classifier).exists() else next(
        (path for path in detector.BRAND_MODEL_CANDIDATES if path.exists()),
        None,
    )
    locator_model_path = active_locator_model_path(project_config)
    locator_ready = bool(locator_model_path and locator_model_path.is_file())
    storage_ready = PROJECT_CONFIG_PATH.parent.is_dir() and os.access(PROJECT_CONFIG_PATH.parent, os.W_OK)
    healthy = bool(runtime["ready"] and locator_ready and storage_ready)
    return JSONResponse(
        {
            "ok": healthy,
            "device": detector.DEVICE,
            "inspection_strategy": runtime["strategy"],
            "inspection_ready": runtime["ready"],
            "inspection_model": runtime["model"] or None,
            "inspection_error": runtime["reason"],
            "detector_model": str(locator_model_path) if locator_model_path else None,
            "detector_ready": locator_ready,
            "detector_role": "part_locator",
            "classifier_model": str(brand_model_path) if brand_model_path else None,
            "storage": "filesystem_json",
            "storage_ready": storage_ready,
            "project": project_config,
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    """Web availability is separate from validated Production and camera readiness."""
    runtime = vision_service.runtime.snapshot()
    storage_ready = vision_service.store.root.is_dir() and os.access(vision_service.store.root, os.W_OK)
    active = None
    config_valid = False
    reason = "No validated Production configuration is active"
    try:
        active = vision_service.store.production()
        if active:
            saved = vision_service.store.recipe(active["recipe_id"])
            from inspection import ENGINE_VERSION
            config_valid = saved["hash"] == active["recipe_hash"] and active.get('engine_version')==ENGINE_VERSION
            reason = "; ".join(runtime["reasons"]) if config_valid else "Production configuration does not match"
    except (ValueError,KeyError,OSError) as exc:
        reason = str(exc)
    return JSONResponse({"ok":bool(storage_ready),"storage_ready":bool(storage_ready),
        "inspection_strategy":"area_recipe","production_config_valid":config_valid,
        "inspection_ready":bool(config_valid and runtime["running"] and runtime["state"]!="UNABLE_TO_INSPECT"),
        "inspection_error":reason,"production":active,"runtime":runtime,
        "legacy_health_url":"/api/legacy/health"})


# ===== HTTP routes: System audit และ cleanup =====
# cleanup แตะเฉพาะ staging ที่ระบบสร้างและไม่ถูกใช้งาน ห้ามขยายไปลบ dataset/model/run
# เพราะ endpoint นี้ไม่มีขั้นตอนสำรองข้อมูล
@app.get("/api/system/audit")
def api_system_audit() -> JSONResponse:
    config = read_project_config()
    runtime = normal_reference_runtime_status(config)
    classifier_text = str(config.get("active_brand_model") or "")
    classifier_path = Path(classifier_text) if classifier_text else next(
        (path for path in detector.BRAND_MODEL_CANDIDATES if path.exists()),
        None,
    )
    detector_validation = {
        "ok": runtime["ready"],
        "error": runtime["reason"],
        "classes": ["OK", "NG"] if runtime["ready"] else [],
    }
    classifier_validation = (
        inspect_classifier_model(classifier_path)
        if classifier_path and classifier_path.exists()
        else {"ok": True, "error": "", "classes": []}
    )
    disk = shutil.disk_usage(PROJECT_ROOT)
    staging_files = [path for path in UPLOAD_STAGING_DIR.rglob("*") if path.is_file()] if UPLOAD_STAGING_DIR.exists() else []
    annotation_manifests = list(ANNOTATION_SESSIONS_DIR.glob("*/manifest.json")) if ANNOTATION_SESSIONS_DIR.exists() else []
    corrupt_annotation_sessions = sum(1 for path in annotation_manifests if not json_file_is_valid(path))
    detection_stats = detection_train_stats()
    reference_dataset = None
    if runtime["source_dataset"] == "reviewed_known_good":
        reviewed_records = []
        for record in list_dataset_folder_records():
            meta = dict(record.get("meta") or {})
            if (
                meta.get("dataset_type") == "Known-good Reference"
                and record.get("review_status") == "reviewed"
                and not record.get("needs_info")
            ):
                reviewed_records.append(record)
        if reviewed_records:
            reference_dataset = {
                "name": "reviewed_known_good",
                "review_status": "reviewed",
                "images": sum(int(record.get("images") or 0) for record in reviewed_records),
                "dataset_count": len(reviewed_records),
            }
    elif runtime["source_dataset"]:
        try:
            reference_folder = dataset_folder_path(runtime["source_dataset"])
            if reference_folder.is_dir():
                reference_dataset = dataset_folder_record(reference_folder)
        except (ValueError, FileNotFoundError):
            reference_dataset = None
    detector_preflight = {
        "ok": runtime["ready"],
        "warnings": [],
        "errors": [] if runtime["ready"] else [runtime["reason"]],
    }
    try:
        brand_preflight = train_preflight("brand_classifier")
    except Exception as exc:
        brand_preflight = {"ok": False, "warnings": [], "errors": [str(exc)]}
    classifier_required = bool(config.get("use_classifier"))
    classifier_ready = bool(classifier_path and classifier_path.exists() and classifier_validation.get("ok"))
    locator_path = active_locator_model_path(config)
    cvat_config = cvat_connection_config()
    cvat_health = cvat_connection_health()
    checks = [
        {"label": "Inspection workspace", "status": "pass" if (WEB_APP_DIR / "templates" / "inspection.html").exists() else "fail", "detail": "unified inspection workspace"},
        {
            "label": "OK reference dataset",
            "status": "pass" if reference_dataset and reference_dataset.get("review_status") == "reviewed" else "fail",
            "detail": (
                (
                    f"{reference_dataset.get('dataset_count')} reviewed dataset(s) · "
                    f"{runtime.get('training_images') or reference_dataset.get('images', 0)} unique training image(s)"
                    if reference_dataset.get("dataset_count")
                    else f"{reference_dataset.get('name')} · {reference_dataset.get('images', 0)} reviewed image(s)"
                )
                if reference_dataset
                else "No reviewed Known-good Reference is active"
            ),
        },
        {
            "label": "Part locator",
            "status": "pass" if locator_path and locator_path.is_file() else "fail",
            "detail": str(locator_path) if locator_path else "No detector model is available",
        },
        {
            "label": "Normal reference",
            "status": "pass" if detector_validation.get("ok") else "fail",
            "detail": runtime["model"] or detector_validation.get("error", ""),
        },
        {
            "label": "Active classifier",
            "status": "pass" if classifier_ready else ("warn" if classifier_required else "pass"),
            "detail": (
                ", ".join(classifier_validation.get("classes") or [])
                if classifier_ready
                else ("enabled but no valid classifier is active" if classifier_required else "optional")
            ),
        },
        {
            "label": "Inspection readiness",
            "status": "fail" if detector_preflight.get("errors") else ("warn" if detector_preflight.get("warnings") else "pass"),
            "detail": (
                f"{len(detector_preflight.get('warnings') or [])} warning(s)"
                if not detector_preflight.get("errors")
                else f"{len(detector_preflight.get('errors') or [])} error(s)"
            ),
        },
        {
            "label": "Brand readiness",
            "status": "fail" if brand_preflight.get("errors") else ("warn" if brand_preflight.get("warnings") else "pass"),
            "detail": (
                "; ".join(brand_preflight.get("warnings") or []) or "ready"
                if not brand_preflight.get("errors")
                else "; ".join(brand_preflight.get("errors") or [])
            ),
        },
        {
            "label": "Project config backup",
            "status": "pass" if json_file_is_valid(PROJECT_CONFIG_PATH.with_suffix(".json.bak")) else "warn",
            "detail": "recoverable backup" if json_file_is_valid(PROJECT_CONFIG_PATH.with_suffix(".json.bak")) else "backup unavailable",
        },
        {
            "label": "Annotation sessions",
            "status": "fail" if corrupt_annotation_sessions else "pass",
            "detail": f"{len(annotation_manifests)} sessions, {corrupt_annotation_sessions} corrupt",
        },
        {
            "label": "CVAT direct sync",
            "status": "pass" if cvat_health["connected"] else "warn",
            "detail": cvat_config["base_url"] if cvat_health["connected"] else cvat_health["connection_error"],
        },
        {
            "label": "Data storage",
            "status": "warn",
            "detail": "filesystem JSON; suitable for one local user, not concurrent multi-user access",
        },
        {"label": "Disk free", "status": "pass" if disk.free >= 5 * 1024**3 else "warn", "detail": f"{disk.free / 1024**3:.1f} GB"},
        {"label": "Upload staging", "status": "warn" if staging_files else "pass", "detail": f"{len(staging_files)} files"},
    ]
    return JSONResponse(
        {
            "ok": not any(check["status"] == "fail" for check in checks),
            "checks": checks,
            "storage": {
                "type": "filesystem_json",
                "project_config": str(PROJECT_CONFIG_PATH),
                "staging_files": len(staging_files),
                "staging_bytes": sum(path.stat().st_size for path in staging_files),
                "deleted_datasets": len(deleted_dataset_records()),
                "disk_free_bytes": disk.free,
            },
            "runtime": {
                "training": bool(TRAIN_PROCESS),
                "camera": bool(WORKER_THREAD and WORKER_THREAD.is_alive()),
                "device": detector.DEVICE,
            },
            "detector_validation": detector_validation,
            "classifier_validation": classifier_validation,
            "detector_preflight": {
                "ok": detector_preflight.get("ok"),
                "warnings": detector_preflight.get("warnings") or [],
                "errors": detector_preflight.get("errors") or [],
            },
            "brand_preflight": {
                "ok": brand_preflight.get("ok"),
                "warnings": brand_preflight.get("warnings") or [],
                "errors": brand_preflight.get("errors") or [],
            },
            "detection_stats": detection_stats,
            "reference_dataset": reference_dataset,
            "checked_at": now_iso(),
        }
    )


@app.post("/api/system/cleanup-staging")
def api_cleanup_upload_staging() -> JSONResponse:
    blocked = dataset_mutation_blocked_response()
    if blocked:
        return blocked
    UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = UPLOAD_STAGING_DIR.resolve()
    deleted_files = 0
    deleted_bytes = 0
    deleted_entries = 0
    for child in list(staging_root.iterdir()):
        resolved = child.resolve()
        if resolved == staging_root or staging_root not in resolved.parents:
            continue
        if child.is_dir():
            files = [path for path in child.rglob("*") if path.is_file()]
            deleted_files += len(files)
            deleted_bytes += sum(path.stat().st_size for path in files)
            shutil.rmtree(child)
        else:
            deleted_files += 1
            deleted_bytes += child.stat().st_size
            child.unlink()
        deleted_entries += 1
    return JSONResponse(
        {
            "ok": True,
            "deleted_entries": deleted_entries,
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
        }
    )


def send_vision_reviews_to_cvat(selected):
    """Reuse the existing CVAT transaction/retry path without copying source images."""
    if not cvat_connection_config()["configured"]:
        raise ValueError("Configure the existing local CVAT URL and token first")
    ids = []
    for event, verdict in selected:
        key = event["id"]
        image_path = vision_service.store.path("images",event["image_id"],".png")
        manifest = INSPECTION_REVIEW_DIR / key / "review.json"
        previous = read_json(manifest,{})
        write_json(manifest,{**previous,"id":key,"status":"confirmed_ng" if verdict["status"]=="confirmed_ng" else "pending",
            "inspection_engine":"area_recipe",
            "human_verdict":verdict["status"],"decision":event["state"],"created_at":event["timestamp"],
            "source_image":str(image_path),"preview_image":str(image_path),"regions":[],
            "recipe_id":event["recipe_id"],"recipe_hash":event["recipe_hash"],"area_results":event["areas"],
            "note":verdict.get("note","")})
        ids.append(key)
    return sync_confirmed_reviews_to_cvat(ids,include_pending=True)


def pull_vision_annotations_from_cvat(event_ids):
    """Read completed CVAT tasks into native inspection annotations, without YOLO conversion."""
    tasks={};records=[]
    for key in event_ids:
        event=vision_service.store.read('evidence',key)
        review=read_json(INSPECTION_REVIEW_DIR/key/'review.json',{})
        sync=review.get('cvat_sync') or {}
        task_id=sync.get('task_id')
        if not task_id:raise ValueError('Send this evidence to CVAT before importing its annotations')
        if task_id not in tasks:
            task=cvat_request(f'/api/tasks/{int(task_id)}')
            if task.get('status')!='completed':raise ValueError(f'Finish and review CVAT task {task_id} before importing')
            labels=cvat_task_labels(task_id,task)
            tasks[task_id]=({int(i['id']):i['name'] for i in labels},cvat_request(f'/api/tasks/{int(task_id)}/annotations/'))
        labels,payload=tasks[task_id]
        if payload.get('tracks'):raise ValueError('Use image shape annotations in CVAT; tracked video interpolation is not supported')
        frame=int(sync['frame_index']);objects=[]
        for shape in payload.get('shapes',[]):
            if int(shape.get('frame',0))!=frame or shape.get('outside'):continue
            points=np.asarray(shape['points'],dtype=float)
            if shape['type']=='rectangle' and len(points)==4:
                a,b,c,d=points;bounds=[min(a,c),min(b,d),max(a,c),max(b,d)];polygons=[]
            elif shape['type']=='polygon' and len(points)>=6 and len(points)%2==0:
                polygon=points.reshape(-1,2);bounds=[*polygon.min(axis=0),*polygon.max(axis=0)];polygons=[polygon.tolist()]
            else:raise ValueError('Use rectangles or polygons for inspection labels in CVAT')
            objects.append({'label':labels[int(shape['label_id'])],'box':list(map(float,bounds)),'polygons':polygons})
        records.append({'image_id':event['image_id'],'objects':objects,'cvat_task_id':int(task_id),'cvat_frame':frame})
    return records


from inspection.api import InspectionService, make_router
vision_service = InspectionService(PROJECT_ROOT,
    legacy_busy=lambda:bool(STATE.get("running")) or training_is_running(),
    cvat_send=send_vision_reviews_to_cvat,cvat_pull=pull_vision_annotations_from_cvat,dataset_folder=dataset_folder_path)
app.include_router(make_router(vision_service))


@app.middleware("http")
async def protect_inspection_workflow(request, call_next):
    if request.url.path.startswith("/api/vision"):
        content_length=request.headers.get("content-length","")
        if content_length.isdigit() and int(content_length)>210*1024*1024:
            return JSONResponse({"detail":"Request exceeds the local upload size limit"},status_code=413)
    # The service binds to localhost. Reject cross-origin writes from unrelated websites.
    changes_state = request.method not in {"GET","HEAD","OPTIONS"} or request.url.path in {"/start_stream","/stop_stream"}
    if changes_state:
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site")=="cross-site" or (origin and origin != f"{request.url.scheme}://{request.url.netloc}"):
            return JSONResponse({"detail":"Cross-origin inspection changes are not allowed"},status_code=403)
    legacy_mutation = request.url.path in {"/start_stream","/api/train/start","/api/inspection/reference/train",
        "/api/models/activate","/api/project/config"} or (request.url.path.startswith("/api/jobs/") and request.url.path.endswith("/use"))
    if vision_service.busy() and legacy_mutation and (request.method=="POST" or request.url.path=="/start_stream"):
        return JSONResponse({"ok":False,"error":"Production inspection or candidate work is running"},status_code=409)
    return await call_next(request)


@app.get("/training", include_in_schema=False)
def training_redirect():
    return RedirectResponse("/inspection",status_code=307)


@app.get("/inspection",response_class=HTMLResponse)
def inspection_page(request:Request):
    return templates.TemplateResponse(request,"inspection.html",{})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        app_dir=str(WEB_APP_DIR),
    )
