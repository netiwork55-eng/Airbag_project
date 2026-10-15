r"""
ระบบตรวจจับ Airbag จากกล้องแบบ real-time ด้วย OpenCV และ Ultralytics YOLOv8

ลำดับงานหลัก:
    0. หากมี Production Recipe ที่ Activate แล้ว ใช้ inspection/ Core ร่วมกับ /inspection
       หากยังไม่มี คงโมเดลเดิมระหว่าง Migration; ไม่เลือก Candidate อัตโนมัติ
    1. เลือกและโหลด weight ของ Airbag detector จากโปรเจกต์
    2. เปิดกล้องและส่งแต่ละ frame เข้า detector
    3. ใช้ COCO guard และกฎ post-process ลด false positive
    4. crop กรอบ Airbag ที่ผ่านการกรองไปให้โมเดลจำแนกแบรนด์ (ถ้ามี)
    5. วาดกรอบ/ชื่อ/ความมั่นใจลงบนภาพแล้วแสดงผล

ไฟล์นี้เป็นแหล่งกลางของ logic inference ที่ทั้งหน้ากล้อง, หน้าเว็บ และระบบ verify
นำไปใช้ร่วมกัน หากแก้ threshold หรือกฎคัดกรอบต้องทดสอบทั้งสามส่วน ไม่ควรคัดลอก
logic ไปสร้างอีกชุด เพราะจะทำให้ผลหน้าเว็บกับผลตรวจสอบไม่ตรงกัน

ติดตั้ง: pip install ultralytics opencv-python
รัน: ..\venv\Scripts\python.exe 01_camera\airbag_webcam_detection.py
"""

import argparse
import json
from pathlib import Path
import sys
import time

# cv2 ใช้เปิดกล้อง webcam, อ่านภาพ, วาดกรอบ และแสดงหน้าต่างผลลัพธ์
import cv2
import numpy as np

# torch ใช้เช็กว่าจะรันบน GPU (cuda) หรือ CPU และเปิด half precision เมื่อใช้ GPU
import torch
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

# YOLO จาก ultralytics ใช้โหลดโมเดล Airbag, โมเดลกัน false positive, และโมเดลแยกแบรนด์
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = PROJECT_ROOT
PROJECT_CONFIG_PATH = PROJECT_ROOT / "07_web_app" / "project_config.json"

# ส่วนนี้รวม path ของ weight ที่ระบบจะลองโหลดตามลำดับ
# The best AI resources are expected here after negative-mining training:
#
# Important:
# A Roboflow-style YOLOv8 dataset folder usually contains data.yaml and image
# folders, but webcam inference needs a trained YOLO weight file such as best.pt.
# If best.pt does not exist, run train_airbag_model.py. It now trains with both
# Airbag.v1i.yolov8 and Airbag.v2i.yolov8 plus empty-label Background images.
COMBINED_NEGATIVE_MINING_DIR = Path.home() / "Downloads" / "Airbag_v1_v2_with_Background"
NEGATIVE_MINING_DIR = Path.home() / "Downloads" / "Airbag_v2i_with_Background"
LEGACY_DATASET_DIR = Path.home() / "Downloads" / "Airbag.v1i.yolov8"
BRAND_MODEL_CANDIDATES = [
    PROJECT_DIR / "06_brand_classification" / "04_trained_models" / "airbag_brand_best.pt",
]
MODEL_CANDIDATES = [
    # Use a future fully-combined model first when it exists.
    PROJECT_DIR / "models" / "Airbag_best.pt",
    # Current best live candidate. Local checks show this model detects the
    # Airbag.v1/v2/v3 exports more reliably than AirbagV2_best.pt.
    PROJECT_DIR / "models" / "best.pt",
    # Kept as fallback only. It was trained from an AirbagV2-named export and is
    # less reliable on the Airbag.v1/v2/v3 data currently in this project.
    PROJECT_DIR / "models" / "AirbagV2_best.pt",
    COMBINED_NEGATIVE_MINING_DIR / "weights" / "best.pt",
    COMBINED_NEGATIVE_MINING_DIR / "runs" / "detect" / "airbag_negative_mining" / "weights" / "best.pt",
    NEGATIVE_MINING_DIR / "weights" / "best.pt",
    NEGATIVE_MINING_DIR / "runs" / "detect" / "airbag_negative_mining" / "weights" / "best.pt",
    LEGACY_DATASET_DIR / "runs" / "detect" / "airbag_yolov8_clean" / "weights" / "best.pt",
    LEGACY_DATASET_DIR / "best.pt",
    LEGACY_DATASET_DIR / "runs" / "detect" / "airbag_yolov8" / "weights" / "best.pt",
    Path.cwd() / "runs" / "detect" / "airbag_yolov8" / "weights" / "best.pt",
]

WINDOW_NAME = "AI Airbag Detector (Negative Mining Trained)"
TARGET_CLASS_NAME = "Airbag"
AIRBAG_CLASS_ALIASES = {"airbag", "airbagv2", "airbag_v2", "airbag-v2"}
ACTIVE_PRIMARY_CLASS = TARGET_CLASS_NAME
ACTIVE_TARGET_CLASSES = [TARGET_CLASS_NAME, "Mark"]
ACTIVE_PRIMARY_ALIASES = set(AIRBAG_CLASS_ALIASES)
ACTIVE_CLASS_DISPLAY_NAMES = {}
ACTIVE_DETECTION_MODE = "inspection"
BRAND_CLASS_NAMES = {"toyota", "honda", "unknown"}

# ส่วนนี้คุมการใช้โมเดล COCO ทั่วไปเพื่อช่วยกัน laptop/person/keyboard ที่ไม่ใช่ Airbag
# This Roboflow dataset has only one class: Airbag. That means the custom model
# cannot detect unrelated objects by itself. To support the red "Not Airbag"
# requirement, we run a general COCO YOLOv8 model for other objects.
# Keep this off for the fastest and clearest Airbag detection.
# Turn it back to True only if you specifically need red COCO "Not Airbag" boxes.
USE_GENERAL_NOT_AIRBAG_DETECTOR = False
USE_COCO_NEGATIVE_GUARD = True
COCO_NEGATIVE_GUARD_CLASSES = {
    "person",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "tv",
}
UNKNOWN_BRAND_NEGATIVE_CLASSES = {
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "tv",
}
UNKNOWN_BRAND_NEGATIVE_CLASS_CONFIDENCE = {
    # COCO ชอบมอง Airbag จริงบางมุมเป็น mouse ได้ จึงต้องมั่นใจก่อนค่อยตัด
    "mouse": 0.65,
}
GENERAL_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "yolov8n.pt",
    PROJECT_ROOT.parent / "yolov8n.pt",
    Path.cwd() / "yolov8n.pt",
]
# ค่ากลุ่มนี้ควบคุมความไวของ Airbag และกฎกัน false positive หลังโมเดล
# Airbag threshold.
#
# This custom Roboflow model has only one class: Airbag. One-class detectors can
# become overconfident on unrelated camera textures if the threshold is too low.
# Presentation mode:
# - Run the model with a low raw threshold so a real Airbag does not disappear.
# - Draw only candidates that pass the stricter Airbag scoring filter below.
AIRBAG_CONFIDENCE_THRESHOLD = 0.30
RAW_AIRBAG_CONFIDENCE_THRESHOLD = 0.05
DEBUG_AIRBAG_CONFIDENCE_THRESHOLD = 0.01
GENERAL_CONFIDENCE_THRESHOLD = 0.35
COCO_NEGATIVE_GUARD_CONFIDENCE = 0.25
COCO_NEGATIVE_GUARD_IOU = 0.18
COCO_NEGATIVE_GUARD_CANDIDATE_OVERLAP = 0.22
OVERLAP_IOU_THRESHOLD = 0.35
AIRBAG_NMS_IOU_THRESHOLD = 0.45
MAX_AIRBAG_DETECTIONS = 10
AIRBAG_HOLD_FRAMES = 10
MAX_PRESENTED_AIRBAGS = 1
SECONDARY_CONFIDENCE_THRESHOLD = 0.40
SECONDARY_MIN_AIRBAG_OVERLAP_RATIO = 0.12
VISUAL_PRIMARY_FALLBACK_CONFIDENCE = 0.62
VISUAL_PRIMARY_MIN_AREA_RATIO = 0.06
VISUAL_PRIMARY_MAX_AREA_RATIO = 0.70
HIGH_CONFIDENCE_ACCEPT_THRESHOLD = 0.78
MIN_AIRBAG_PRESENTATION_SCORE = 0.56
MIN_SCORED_AIRBAG_CONFIDENCE = 0.20
WIDE_OBJECT_REJECT_CONFIDENCE = 0.80
MAX_WIDE_LOW_CONF_ASPECT_RATIO = 1.38
MAX_WIDE_LOW_CONF_AREA_RATIO = 0.22
EDGE_TOUCH_REJECT_CONFIDENCE = 0.86
HUGE_BOX_REJECT_CONFIDENCE = 0.88
NARROW_BOX_REJECT_CONFIDENCE = 0.85
NARROW_MOUSE_LIKE_ASPECT_RATIO = 0.72

# Speed tuning:
# - Airbag model still sees the full webcam frame. No crop / no ROI is used.
# - 640 gives better small/mid-frame detection than 512 on the live webcam.
# - The general COCO "Not Airbag" model is much heavier overall because it is a
#   second model, so it is refreshed every few frames and reused between refreshes.
AIRBAG_IMAGE_SIZE = 640
BRAND_IMAGE_SIZE = 224
BRAND_CONFIDENCE_THRESHOLD = 0.70
UNKNOWN_BRAND_REJECT_CONFIDENCE = 0.90
BRAND_CROP_PADDING_RATIO = 0.10
GENERAL_IMAGE_SIZE = 416
GENERAL_DETECT_EVERY_N_FRAMES = 3
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
DEFAULT_CAMERA_INDEX = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"
UNICODE_LABEL_FONT = None
UNICODE_LABEL_FONT_PATHS = [
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\LeelawUI.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


def get_xyxy(box):
    """Return an integer [x1, y1, x2, y2] list from a YOLO box."""
    return list(map(int, box.xyxy[0]))


def class_key(name: str) -> str:
    """Normalize class names for config/model comparisons."""
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def configure_detection_target(primary_class=None, target_classes=None, display_names=None, detection_mode=None):
    """Configure the runtime detector for the current project."""
    global TARGET_CLASS_NAME, ACTIVE_PRIMARY_CLASS, ACTIVE_TARGET_CLASSES
    global ACTIVE_PRIMARY_ALIASES, ACTIVE_CLASS_DISPLAY_NAMES, ACTIVE_DETECTION_MODE

    clean_targets = []
    seen = set()
    for raw_name in target_classes or []:
        name = str(raw_name or "").strip()
        if not name:
            continue
        key = class_key(name)
        if key in seen:
            continue
        seen.add(key)
        clean_targets.append(name)

    clean_primary = str(primary_class or "").strip()
    if not clean_primary:
        clean_primary = clean_targets[0] if clean_targets else TARGET_CLASS_NAME
    if clean_primary and class_key(clean_primary) not in {class_key(name) for name in clean_targets}:
        clean_targets.insert(0, clean_primary)

    ACTIVE_PRIMARY_CLASS = clean_primary or TARGET_CLASS_NAME
    TARGET_CLASS_NAME = ACTIVE_PRIMARY_CLASS
    ACTIVE_TARGET_CLASSES = clean_targets or [ACTIVE_PRIMARY_CLASS]
    ACTIVE_PRIMARY_ALIASES = {class_key(ACTIVE_PRIMARY_CLASS)}
    ACTIVE_CLASS_DISPLAY_NAMES = {
        class_key(name): str(display or "").strip()
        for name, display in (display_names or {}).items()
        if str(display or "").strip()
    }
    mode = str(detection_mode or "inspection").strip().lower()
    ACTIVE_DETECTION_MODE = mode if mode in {"inspection", "independent", "normal_reference"} else "inspection"
    if class_key(ACTIVE_PRIMARY_CLASS) in {class_key(name) for name in AIRBAG_CLASS_ALIASES}:
        ACTIVE_PRIMARY_ALIASES.update(class_key(name) for name in AIRBAG_CLASS_ALIASES)


def display_class_name(class_name):
    """Return a user-facing label without changing the model's internal class."""
    name = str(class_name or "").strip()
    return ACTIVE_CLASS_DISPLAY_NAMES.get(class_key(name), name)


def unicode_label_font():
    global UNICODE_LABEL_FONT
    if UNICODE_LABEL_FONT is not None:
        return UNICODE_LABEL_FONT
    if ImageFont is None:
        return None
    for font_path in UNICODE_LABEL_FONT_PATHS:
        if font_path.exists():
            UNICODE_LABEL_FONT = ImageFont.truetype(str(font_path), 22)
            return UNICODE_LABEL_FONT
    return None


def load_project_detection_config():
    """Load detector target setup from the web project config when available."""
    if not PROJECT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Project configuration missing: {PROJECT_CONFIG_PATH}. Refusing to switch to a different detector silently.")
    try:
        data = json.loads(PROJECT_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read project configuration: {PROJECT_CONFIG_PATH}") from exc
    if not isinstance(data, dict):
        raise ValueError("Project configuration must be a JSON object")
    target_classes = data.get("target_classes") or []
    if isinstance(target_classes, str):
        target_classes = [part.strip() for part in target_classes.split(",") if part.strip()]
    configure_detection_target(
        data.get("primary_class"),
        target_classes,
        data.get("class_display_names") or {},
        data.get("detection_mode"),
    )
    return data


def is_primary_class(class_id, class_name):
    """Return True when a model detection represents the active project's primary class."""
    normalized_name = class_key(class_name)
    if normalized_name in ACTIVE_PRIMARY_ALIASES:
        return True
    return class_id == 0 and normalized_name in {"", "0"} and len(ACTIVE_TARGET_CLASSES) == 1


def primary_supports_airbag_assists():
    """Airbag-only visual/brand assists are not safe for arbitrary future projects."""
    return class_key(ACTIVE_PRIMARY_CLASS) in {class_key(name) for name in AIRBAG_CLASS_ALIASES}


# ฟังก์ชันกลุ่มนี้ใช้ตรวจชื่อ class/model ว่าเป็น Airbag หรือ Brand classifier ที่ถูกต้อง
def is_airbag_class(class_id, class_name):
    """Backward-compatible wrapper for the active project's primary class."""
    return is_primary_class(class_id, class_name)


def model_names_list(model_names):
    if isinstance(model_names, dict):
        return [str(name).strip() for _, name in sorted(model_names.items())]
    return [str(name).strip() for name in model_names]


def model_names_are_airbag_only(model_names):
    """Return True only for a one-class primary-object custom model."""
    names = model_names_list(model_names)
    return len(names) == 1 and class_key(names[0]) in ACTIVE_PRIMARY_ALIASES


def model_names_are_inspection_detector(model_names):
    """Return True for a model that contains the active project's primary class."""
    names = model_names_list(model_names)
    return any(class_key(name) in ACTIVE_PRIMARY_ALIASES for name in names)


def model_names_are_brand_classifier(model_names):
    """Return True for a usable multi-class classifier label set."""
    names = model_names_list(model_names)
    return len(names) >= 2 and all(str(name).strip() for name in names)


def passes_airbag_false_positive_filter(xyxy, frame_shape, confidence):
    """Reject common low-confidence laptop/keyboard-shaped false positives."""
    metrics = get_airbag_box_metrics(xyxy, frame_shape)
    aspect_ratio = metrics["aspect_ratio"]
    area_ratio = metrics["area_ratio"]
    edge_touches = metrics["edge_touches"]

    if aspect_ratio < 0.45 or aspect_ratio > 2.10:
        return False

    if area_ratio < 0.015 or area_ratio > 0.72:
        return False

    if edge_touches >= 2 and confidence < EDGE_TOUCH_REJECT_CONFIDENCE:
        return False

    if area_ratio > 0.60 and confidence < HUGE_BOX_REJECT_CONFIDENCE:
        return False

    if aspect_ratio < NARROW_MOUSE_LIKE_ASPECT_RATIO and confidence < NARROW_BOX_REJECT_CONFIDENCE:
        return False

    # The live false positive on the laptop/keyboard is a large, wide black
    # rectangle around 0.6 confidence. A true Airbag is usually more square or
    # has much higher confidence after the new training.
    if (
        confidence < WIDE_OBJECT_REJECT_CONFIDENCE
        and aspect_ratio > MAX_WIDE_LOW_CONF_ASPECT_RATIO
        and area_ratio > MAX_WIDE_LOW_CONF_AREA_RATIO
    ):
        return False

    return True


def clamp01(value):
    """Clamp a numeric score to [0, 1]."""
    return max(0.0, min(1.0, value))


# ฟังก์ชันกลุ่มนี้ให้คะแนนและเลือกกรอบ Airbag ที่ควรวาดจริงบนหน้าจอ
def get_airbag_box_metrics(xyxy, frame_shape):
    """Return shape/size metrics for an Airbag candidate box."""
    x1, y1, x2, y2 = xyxy
    frame_height, frame_width = frame_shape[:2]
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    aspect_ratio = box_width / box_height
    area_ratio = (box_width * box_height) / max(1, frame_width * frame_height)
    edge_touches = sum(
        [
            x1 <= 2,
            y1 <= 2,
            x2 >= frame_width - 3,
            y2 >= frame_height - 3,
        ]
    )

    return {
        "aspect_ratio": aspect_ratio,
        "area_ratio": area_ratio,
        "edge_touches": edge_touches,
    }


def score_airbag_candidate(xyxy, frame_shape, confidence):
    """Score how presentation-worthy an Airbag candidate is."""
    metrics = get_airbag_box_metrics(xyxy, frame_shape)
    aspect_ratio = metrics["aspect_ratio"]
    area_ratio = metrics["area_ratio"]

    # Most real steering-wheel Airbag covers in this project are close to square
    # or mildly wide. Laptop/keyboard false positives are much wider.
    aspect_score = clamp01(1.0 - abs(aspect_ratio - 1.10) / 0.75)

    # Good Airbag boxes can be small or close-up large, but extremely tiny or
    # nearly full-frame boxes are unstable for presentation.
    if 0.05 <= area_ratio <= 0.55:
        size_score = 1.0
    elif area_ratio < 0.05:
        size_score = clamp01(area_ratio / 0.05)
    else:
        size_score = clamp01((0.72 - area_ratio) / 0.17)

    wide_low_conf_penalty = 0.0
    if (
        confidence < WIDE_OBJECT_REJECT_CONFIDENCE
        and aspect_ratio > MAX_WIDE_LOW_CONF_ASPECT_RATIO
        and area_ratio > MAX_WIDE_LOW_CONF_AREA_RATIO
    ):
        wide_low_conf_penalty = 0.30

    return (
        confidence * 0.72
        + aspect_score * 0.18
        + size_score * 0.10
        - wide_low_conf_penalty
    )


def select_airbag_detections(candidates, frame_shape, negative_detections=None):
    """Select only the best Airbag detections for a clean presentation view."""
    negative_detections = negative_detections or []
    if ACTIVE_DETECTION_MODE == "independent":
        return sorted(candidates, key=lambda item: item[1], reverse=True)[:MAX_AIRBAG_DETECTIONS]

    scored_candidates = []
    for xyxy, confidence in candidates:
        if primary_supports_airbag_assists():
            if not passes_airbag_false_positive_filter(xyxy, frame_shape, confidence):
                continue
            score = score_airbag_candidate(xyxy, frame_shape, confidence)
        else:
            score = confidence

        # ถ้าโมเดล Airbag มั่นใจสูง ให้เชื่อ Airbag ก่อน ไม่ให้ COCO guard
        # อย่าง person/laptop ตัดกรอบจริงทิ้ง โดยเฉพาะตอนถือ Airbag ใกล้กล้อง
        if (
            confidence < HIGH_CONFIDENCE_ACCEPT_THRESHOLD
            and overlaps_negative_guard(xyxy, negative_detections)
        ):
            continue

        if confidence >= HIGH_CONFIDENCE_ACCEPT_THRESHOLD:
            scored_candidates.append((score, xyxy, confidence))
            continue

        if (
            confidence >= MIN_SCORED_AIRBAG_CONFIDENCE
            and score >= MIN_AIRBAG_PRESENTATION_SCORE
        ):
            scored_candidates.append((score, xyxy, confidence))

    scored_candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
    return [
        (xyxy, confidence)
        for _, xyxy, confidence in scored_candidates[:MAX_PRESENTED_AIRBAGS]
    ]


def calculate_iou(box_a, box_b):
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_intersection_over_a(box_a, box_b):
    """Return how much of box_a is covered by box_b."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    return inter_area / area_a


def box_center_inside(inner_box, outer_box):
    """Return True when the center of inner_box is inside outer_box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    center_x = (ix1 + ix2) / 2
    center_y = (iy1 + iy2) / 2
    return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2


def secondary_matches_airbag(secondary_box, airbag_boxes):
    """Return True only when a defect/feature detection belongs to an Airbag."""
    for airbag_box in airbag_boxes:
        if box_center_inside(secondary_box, airbag_box):
            return True
        if calculate_intersection_over_a(secondary_box, airbag_box) >= SECONDARY_MIN_AIRBAG_OVERLAP_RATIO:
            return True
    return False


def filter_secondary_detections(secondary_detections, airbag_boxes, debug_detection=False):
    """Keep secondary classes only when a selected Airbag anchors them."""
    if ACTIVE_DETECTION_MODE == "independent":
        return [
            item
            for item in secondary_detections
            if item[2] >= SECONDARY_CONFIDENCE_THRESHOLD or debug_detection
        ]
    if not airbag_boxes:
        return []

    filtered = []
    for xyxy, class_name, confidence in secondary_detections:
        if confidence < SECONDARY_CONFIDENCE_THRESHOLD and not debug_detection:
            continue
        if secondary_matches_airbag(xyxy, airbag_boxes):
            filtered.append((xyxy, class_name, confidence))
    return filtered


def find_airbag_only_candidates(airbag_model, frame, airbag_confidence, debug_detection, image_size=AIRBAG_IMAGE_SIZE):
    """Run an Airbag-only detector and return primary Airbag candidates."""
    predict_confidence = (
        DEBUG_AIRBAG_CONFIDENCE_THRESHOLD
        if debug_detection
        else RAW_AIRBAG_CONFIDENCE_THRESHOLD
    )
    result = airbag_model.predict(
        frame,
        conf=predict_confidence,
        imgsz=image_size,
        iou=AIRBAG_NMS_IOU_THRESHOLD,
        max_det=MAX_AIRBAG_DETECTIONS,
        device=DEVICE,
        **({"half": True} if USE_HALF else {}),
        verbose=False,
    )[0]

    candidates = []
    best_raw_confidence = 0.0
    if result.boxes is None:
        return candidates, best_raw_confidence

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names.get(class_id, str(class_id))
        if not is_airbag_class(class_id, class_name):
            continue

        best_raw_confidence = max(best_raw_confidence, confidence)
        if confidence < airbag_confidence and not debug_detection:
            continue
        candidates.append((get_xyxy(box), confidence))

    return candidates, best_raw_confidence


def find_visual_airbag_candidates(frame, secondary_detections=None):
    """Find one large dark Airbag-like object when the detector misses the main box.

    ใช้เป็นตัวช่วยกู้กรอบ Airbag เฉพาะตอนมี secondary/defect detection อยู่แล้วเท่านั้น
    เพราะถ้าให้วัตถุสีเข้มขนาดใหญ่สร้างกรอบเองล้วน ๆ จะเกิด false positive ง่ายมาก
    เช่น เมาส์ คีย์บอร์ด หรือของสีดำบนโต๊ะที่หน้าตาใกล้กับ Airbag ในภาพกล้องจริง
    """
    if not secondary_detections:
        return []

    frame_height, frame_width = frame.shape[:2]
    frame_area = max(1, frame_width * frame_height)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    dark_mask = cv2.inRange(value, 0, 125)
    saturated_mask = cv2.inRange(saturation, 25, 255)
    mask = cv2.bitwise_and(dark_mask, saturated_mask)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    secondary_centers = []
    for xyxy, _, confidence in secondary_detections or []:
        if confidence < SECONDARY_CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = xyxy
        secondary_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    if not secondary_centers:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        area_ratio = (width * height) / frame_area
        if area_ratio < VISUAL_PRIMARY_MIN_AREA_RATIO or area_ratio > VISUAL_PRIMARY_MAX_AREA_RATIO:
            continue

        aspect_ratio = width / max(1, height)
        if aspect_ratio < 0.55 or aspect_ratio > 2.05:
            continue

        xyxy = [x, y, x + width, y + height]
        if secondary_centers and not any(x <= cx <= x + width and y <= cy <= y + height for cx, cy in secondary_centers):
            continue

        confidence = min(0.74, VISUAL_PRIMARY_FALLBACK_CONFIDENCE + (contour_area / frame_area) * 0.30)
        candidates.append((xyxy, confidence))

    candidates.sort(key=lambda item: (item[1], (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])), reverse=True)
    return candidates[:1]


def overlaps_airbag(candidate_box, airbag_boxes):
    """Return True when a general object overlaps a custom Airbag detection."""
    return any(
        calculate_iou(candidate_box, airbag_box) >= OVERLAP_IOU_THRESHOLD
        for airbag_box in airbag_boxes
    )


# ฟังก์ชันกลุ่มนี้ใช้ดูว่ากรอบ Airbag ไปซ้อนกับ object อื่นจนควรถูกกันออกหรือไม่
def is_negative_guard_detection(class_name, confidence):
    """Return True when a COCO detection should veto an Airbag candidate."""
    return (
        str(class_name).strip().lower() in COCO_NEGATIVE_GUARD_CLASSES
        and confidence >= COCO_NEGATIVE_GUARD_CONFIDENCE
    )


def overlaps_negative_guard(candidate_box, negative_detections):
    """Return True when an Airbag candidate overlaps a known non-Airbag object."""
    for negative_box, class_name, confidence in negative_detections:
        if not is_negative_guard_detection(class_name, confidence):
            continue

        iou = calculate_iou(candidate_box, negative_box)
        candidate_overlap = calculate_intersection_over_a(candidate_box, negative_box)
        if (
            iou >= COCO_NEGATIVE_GUARD_IOU
            or candidate_overlap >= COCO_NEGATIVE_GUARD_CANDIDATE_OVERLAP
        ):
            return True

    return False


def draw_detection(frame, xyxy, class_name, confidence):
    """Draw one YOLO detection on the frame."""
    x1, y1, x2, y2 = xyxy

    is_primary = is_primary_class(None, class_name)
    color = (0, 255, 0) if is_primary else (0, 0, 255)
    label_name = ACTIVE_PRIMARY_CLASS if is_primary else str(class_name)
    label_name = display_class_name(label_name)
    label = f"{label_name} Assist" if confidence is None else f"{label_name} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    use_unicode = any(ord(char) > 127 for char in label)
    font = unicode_label_font() if use_unicode else None
    if font is not None:
        left, top, right, bottom = font.getbbox(label)
        label_size = (max(1, right - left), max(1, bottom - top))
        baseline = 4
    else:
        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
    label_y = max(y1, label_size[1] + 10)

    box_left = max(0, x1)
    box_top = max(0, label_y - label_size[1] - baseline - 6)
    box_right = min(frame.shape[1], x1 + label_size[0] + 8)
    box_bottom = min(frame.shape[0], label_y + baseline - 2)
    cv2.rectangle(
        frame,
        (box_left, box_top),
        (box_right, box_bottom),
        color,
        thickness=-1,
    )
    if font is not None and box_right > box_left and box_bottom > box_top:
        roi = frame[box_top:box_bottom, box_left:box_right]
        image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        draw.text((3, -2), label, font=font, fill=(255, 255, 255))
        frame[box_top:box_bottom, box_left:box_right] = cv2.cvtColor(
            np.asarray(image),
            cv2.COLOR_RGB2BGR,
        )
    else:
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ฟังก์ชันกลุ่มนี้ใช้ crop กรอบ Airbag แล้วแยกแบรนด์ Toyota/Honda/Unknown
def draw_brand_label(frame, xyxy, brand_name, confidence):
    """Draw the Toyota/Honda brand label under the Airbag label."""
    x1, y1, _, _ = xyxy
    brand_text = f"Brand: {brand_name} {confidence:.2f}"
    color = (0, 180, 255) if brand_name.lower() != "unknown" else (0, 255, 255)

    label_size, baseline = cv2.getTextSize(
        brand_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        2,
    )
    label_y = max(y1 + 32, label_size[1] + 42)

    cv2.rectangle(
        frame,
        (x1, label_y - label_size[1] - baseline - 6),
        (x1 + label_size[0] + 6, label_y + baseline - 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        brand_text,
        (x1 + 3, label_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def crop_airbag_for_brand(frame, xyxy):
    """Crop the Airbag box with small padding for brand classification."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(box_width * BRAND_CROP_PADDING_RATIO)
    pad_y = int(box_height * BRAND_CROP_PADDING_RATIO)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(frame_width, x2 + pad_x)
    crop_y2 = min(frame_height, y2 + pad_y)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def classify_airbag_brand(brand_model, frame, xyxy):
    """Classify the detected Airbag crop as Toyota, Honda, or Unknown."""
    if brand_model is None:
        return None

    crop = crop_airbag_for_brand(frame, xyxy)
    if crop is None or crop.size == 0:
        return None

    result = brand_model.predict(
        crop,
        imgsz=BRAND_IMAGE_SIZE,
        device=DEVICE,
        **({"half": True} if USE_HALF else {}),
        verbose=False,
    )[0]
    top_index = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    brand_name = result.names.get(top_index, str(top_index))

    if confidence < BRAND_CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return brand_name, confidence


def overlaps_named_negative(candidate_box, negative_detections, class_names):
    """Return True when candidate overlaps one of the requested COCO classes."""
    class_names = {str(name).strip().lower() for name in class_names}
    for negative_box, class_name, confidence in negative_detections or []:
        if str(class_name).strip().lower() not in class_names:
            continue

        class_confidence = UNKNOWN_BRAND_NEGATIVE_CLASS_CONFIDENCE.get(
            str(class_name).strip().lower(),
            COCO_NEGATIVE_GUARD_CONFIDENCE,
        )
        if confidence < class_confidence:
            continue

        iou = calculate_iou(candidate_box, negative_box)
        candidate_overlap = calculate_intersection_over_a(candidate_box, negative_box)
        if (
            iou >= COCO_NEGATIVE_GUARD_IOU
            or candidate_overlap >= COCO_NEGATIVE_GUARD_CANDIDATE_OVERLAP
        ):
            return True
    return False


def should_reject_by_brand(brand_result, xyxy=None, negative_detections=None):
    """กัน false positive เมื่อ Brand Unknown สูงและ COCO เห็นว่าเป็นวัตถุอื่น."""
    if brand_result is None:
        return False

    brand_name, confidence = brand_result
    is_high_unknown = (
        str(brand_name).strip().lower() == "unknown"
        and confidence >= UNKNOWN_BRAND_REJECT_CONFIDENCE
    )
    if not is_high_unknown or xyxy is None:
        return False

    return overlaps_named_negative(
        xyxy,
        negative_detections,
        UNKNOWN_BRAND_NEGATIVE_CLASSES,
    )


def draw_fps(frame, fps):
    """Draw a small FPS indicator without covering detections."""
    text = f"FPS {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, text, color):
    """Draw a compact status line at the top-left of the frame."""
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args(argv=None):
    """Read optional runtime settings."""
    parser = argparse.ArgumentParser(
        description="Run real-time Airbag detection from a webcam.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=(
            f"OpenCV camera index to use. Default is {DEFAULT_CAMERA_INDEX}. "
            "Use 1 or 2 for another connected camera."
        ),
    )
    parser.add_argument(
        "--airbag-conf",
        type=float,
        default=AIRBAG_CONFIDENCE_THRESHOLD,
        help=(
            "Confidence threshold for drawing a green Airbag box. "
            "Lower values are more sensitive but can create more false positives."
        ),
    )
    parser.add_argument(
        "--debug-detection",
        action="store_true",
        help=(
            "Show the best raw Airbag confidence on screen by running prediction "
            "at a very low threshold. Use this to see whether the model is not "
            "detecting at all or only detecting below the normal threshold."
        ),
    )
    parser.add_argument(
        "--no-brand",
        action="store_true",
        help="Disable Toyota/Honda brand classification overlay.",
    )
    return parser.parse_args(argv)


def open_camera(camera_index):
    """Open the requested camera, preferring DirectShow on Windows."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    return cap


def camera_index_candidates(preferred_index, max_index=10):
    """Return a stable scan order with the configured camera first."""
    preferred = max(0, int(preferred_index))
    maximum = max(preferred, int(max_index))
    return [preferred, *(index for index in range(maximum + 1) if index != preferred)]


def frame_has_visual_signal(frame):
    """Reject blank placeholder frames returned by disconnected camera devices."""
    if frame is None or not getattr(frame, "size", 0):
        return False
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    # Some Windows camera drivers report success while returning a completely
    # black placeholder. A real, even very dark frame still normally contains
    # a small amount of luminance or sensor variation.
    return bool(float(grayscale.mean()) >= 3.0 or float(grayscale.std()) >= 1.0)


def open_available_camera(preferred_index, max_index=10):
    """Open the first camera that can deliver a real frame.

    ``isOpened`` alone is not sufficient for some virtual or disconnected
    Windows camera devices, so every candidate must also return a frame.
    """
    checked = []
    for camera_index in camera_index_candidates(preferred_index, max_index):
        checked.append(camera_index)
        cap = open_camera(camera_index)
        if not cap.isOpened():
            cap.release()
            continue

        readable = False
        for _ in range(12):
            ok, frame = cap.read()
            if ok and frame_has_visual_signal(frame):
                readable = True
                break
            time.sleep(0.05)
        if readable:
            return cap, camera_index
        cap.release()

    checked_text = ", ".join(str(index) for index in checked)
    raise RuntimeError(
        f"ไม่พบกล้องที่มีสัญญาณภาพจริง (ตรวจ index: {checked_text}) "
        "กรุณาต่อกล้อง เปิดฝาปิดเลนส์ หรือปิดโปรแกรมอื่นที่กำลังใช้กล้อง"
    )


def load_primary_object_locator(project_config):
    """Load a detector only for locating the inspected part, not defect typing."""
    configured = str(
        project_config.get("active_locator_model")
        or project_config.get("active_detector_model")
        or ""
    ).strip()
    dynamic = sorted(
        (PROJECT_DIR / "models").glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    # A configured recipe must use its chosen locator. Silently falling back to
    # another weight changes the ROI and invalidates the reference calibration.
    candidates = [Path(configured)] if configured else dynamic + MODEL_CANDIDATES
    seen = set()
    for path in candidates:
        path = Path(path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            model = YOLO(str(path))
            if getattr(model, "task", "") != "detect":
                continue
            names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
            # A locator may contain old defect classes, but they are deliberately
            # ignored. Prefer the first non-defect class as the inspected part.
            web_dir = str(PROJECT_ROOT / "07_web_app")
            if web_dir not in sys.path:
                sys.path.insert(0, web_dir)
            from part_locator import part_class_id
            primary_id = part_class_id(names)
            if primary_id is None:
                continue
            model.to(DEVICE)
            model.fuse()
            return model, primary_id
        except Exception as exc:
            print(f"Object locator skipped {path}: {exc}")
    return None, None


def analyze_primary_object(locator_model, class_id, frame):
    """Use the same ROI contract as reference training; defect classes cannot vote."""
    web_dir = str(PROJECT_ROOT / "07_web_app")
    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)
    from part_locator import analyze_part
    return analyze_part(locator_model, class_id, frame, device=DEVICE, half=USE_HALF)


def save_inspection_sample(frame, result, located, model_path):
    """Save unmodified evidence for offline diagnosis; never auto-label it good."""
    folder = PROJECT_ROOT / "work" / "live_inspection_samples"
    folder.mkdir(parents=True, exist_ok=True)
    stem = str(time.time_ns())
    image_path = folder / f"{stem}.png"
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("Could not encode inspection sample")
    encoded.tofile(str(image_path))
    record = {"image": image_path.name, "model": str(model_path),
              "object_box": list(located["box"]) if located else None,
              "expected_decision": None,
              "raw_result": result.to_dict() if result else None}
    (folder / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path


def run_normal_reference_camera(args, project_config, core_runtime=None):
    """Keep the existing capture/window loop; use the shared Core for activated recipes.

    The legacy inspector remains only while its configuration has no validated
    replacement. A broken active Core recipe must never fall back to that model.
    """
    web_app_dir = PROJECT_ROOT / "07_web_app"
    if str(web_app_dir) not in sys.path:
        sys.path.insert(0, str(web_app_dir))
    from inspection.runtime import evidence

    cap = None
    pump = None
    camera_released = False
    cycle = None
    reference = dict(project_config.get("normal_reference") or {})
    window = "Generic OK/NG Inspection"
    if core_runtime and str(project_config.get('inspection_strategy') or project_config.get('detection_mode') or '').lower()!='normal_reference':
        window = WINDOW_NAME
    try:
        if core_runtime:
            pipeline = core_runtime.production_pipeline()
            camera = pipeline.recipe.camera
            if camera.driver!='opencv' or camera.index!=args.camera_index:
                raise ValueError('Active recipe camera does not match the existing terminal camera; no camera/settings were changed')
            if camera.exposure_us is not None or camera.gain is not None:
                raise ValueError('Existing USB camera controls must be configured in its driver, not overridden by a recipe')
            if ((camera.width is not None and camera.width!=pipeline.reference.shape[1]) or
                    (camera.height is not None and camera.height!=pipeline.reference.shape[0])):
                raise ValueError('Recipe camera resolution must match its full-frame reference; no resizing or camera changes were applied')
            cycle = core_runtime.make_cycle(pipeline)
        else:
            from normal_reference import DecisionStabilizer, InspectionResult
            from deep_patch_reference import load_inspection_reference
            from inspection_contract import RecipeBoundInspector
            from inspection_cycles import PartCycle, save_cycle
            raw_model = str(reference.get("active_model") or "").strip()
            model_path = Path(raw_model) if raw_model else None
            if not model_path or not model_path.is_file():
                raise FileNotFoundError('Configured Normal Reference model not found; no other model was selected')
            inspector = RecipeBoundInspector(load_inspection_reference(model_path),model_path,reference)
            locator_model, locator_class_id = load_primary_object_locator(project_config)
            brand_model = None
            configured_brand = str(project_config.get("active_brand_model") or "").strip()
            if not args.no_brand and configured_brand and Path(configured_brand).is_file():
                candidate = YOLO(configured_brand)
                if getattr(candidate,"task","")=="classify" and model_names_are_brand_classifier(candidate.names):
                    brand_model = candidate
                    brand_model.to(DEVICE)
            stabilizer = DecisionStabilizer()
            cycle = PartCycle()
            width,height = CAMERA_WIDTH,CAMERA_HEIGHT

        cv2.setUseOptimized(True)
        if core_runtime:
            pump = core_runtime.open_capture(pipeline,terminal_fps=CAMERA_FPS)
            selected_camera_index = pipeline.recipe.camera.index
        else:
            cap, selected_camera_index = open_available_camera(args.camera_index)
            # Retained legacy branch: do not change the user's current camera settings.
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
            cap.set(cv2.CAP_PROP_FPS,CAMERA_FPS)
        print(f"Inspection Core recipe {pipeline.saved['id']}" if core_runtime else f"Legacy Normal Reference: {model_path}; no activated Core replacement")
        print(f"Camera index {selected_camera_index}. Press Esc/q to quit; Space/s to save an unmodified sample.")
        # Enlarge only the preview window; inspection and saved frames keep their pixels.
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 1280, 960)
        fps = 0.0
        last_time = time.perf_counter()
        sequence = 0
        while True:
            if core_runtime:
                sequence,frame,received,captured_at = pump.next(sequence)
            else:
                success, frame = cap.read()
                if not success or frame is None:
                    raise RuntimeError('Camera disconnected or failed to deliver a frame')
            measured_result = None
            if core_runtime:
                state,measured_result,maps = core_runtime.inspect_frame(pipeline,cycle,frame,received,captured_at,sequence)
                display = frame.copy()
                colors={'PASS':(0,190,0),'NG':(0,0,255),'REVIEW':(0,190,255),'UNABLE_TO_INSPECT':(220,0,200)}
                for area in measured_result.get('areas',[]):
                    if area.get('source_polygon'):
                        polygon=np.rint(area['source_polygon']).astype(np.int32)
                        color=colors.get(area['state'],colors['UNABLE_TO_INSPECT'])
                        cv2.polylines(display,[polygon],True,color,2)
                        x,y=polygon[0]
                        score=area.get('score')
                        text=f"{area['area_id']}: {area['state']}"+(f" {score:.3f}" if score is not None else '')
                        cv2.putText(display,text,(max(0,int(x)),max(45,int(y)-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                draw_status(display,state['state'],colors.get(state['state'],colors['UNABLE_TO_INSPECT']))
                cv2.putText(display,'; '.join(state.get('reasons',[]))[:100],(10,50),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
            else:
                located = analyze_primary_object(locator_model,locator_class_id,frame)
                if located is not None:
                    brand_name = brand_confidence = None
                    if brand_model is not None:
                        brand_result = classify_airbag_brand(brand_model,frame,located['box'])
                        if brand_result:
                            brand_name,brand_confidence = brand_result
                    result = inspector.inspect(frame,object_box=located['box'],brand=brand_name,brand_confidence=brand_confidence)
                    measured_result = result
                    result = stabilizer.update(result,object_box=located['box'])
                else:
                    stabilizer.reset()
                    result = None
                display = inspector.draw_result(frame,result or InspectionResult('NO_PART',0.0,0.0,inspector.threshold,[]),
                    object_box=located['box'] if located else None,object_label=located['label'] if located else 'Part',
                    object_confidence=located['confidence'] if located else None)
                closed = cycle.observe(located is not None,result,now=time.perf_counter(),frame=frame)
                save_cycle(PROJECT_ROOT,closed,reference,f'camera_{selected_camera_index}')
                cycle.draw(display)
            stamp = time.perf_counter()
            elapsed = stamp-last_time
            last_time = stamp
            if elapsed>0:
                fps=fps*.85+(1.0/elapsed)*.15
            draw_fps(display,fps)
            cv2.imshow(window,display)
            key=cv2.waitKey(1)&0xFF
            if key in (27,ord('q')):
                break
            if key in (32,ord('s')):
                try:
                    if core_runtime:
                        sample=evidence(core_runtime.store,measured_result,frame,maps,source='camera_sample',complete=False)
                        print(f"Saved unconfirmed camera sample: {sample['id']}")
                    else:
                        print(f"Saved inspection sample: {save_inspection_sample(frame,measured_result,located,model_path)}")
                except (OSError,RuntimeError,cv2.error) as exc:
                    print(f'Could not save inspection sample: {exc}')
    finally:
        try:
            if cycle is not None:
                if core_runtime:
                    core_runtime.finish_cycle(cycle)
                elif cap is not None:
                    save_cycle(PROJECT_ROOT,cycle.finish('camera_stopped'),reference,f'camera_{selected_camera_index}')
        finally:
            try:
                if pump is not None:
                    pump.close()
                elif cap is not None:
                    cap.release()
                camera_released = True
            finally:
                try:
                    if core_runtime:
                        core_runtime.anomaly.close()
                finally:
                    if cap is not None or pump is not None:
                        cv2.destroyAllWindows()
                        print('Camera released. Window closed.' if camera_released else 'Window closed. Camera release failed; stop this process before another run.')


def main(argv=None):
    """เปิดโมเดลและกล้อง แล้ววน inference จนผู้ใช้สั่งออก; ปิดกล้องในทุกเส้นทางจบงาน."""
    # โหลดโมเดล เปิดกล้อง แล้ววนอ่าน frame เพื่อ detect แบบ real-time
    args = parse_args() if argv is None else parse_args(argv)
    web_app_dir = str(PROJECT_ROOT / '07_web_app')
    if web_app_dir not in sys.path:
        sys.path.insert(0,web_app_dir)
    from inspection.ownership import project_lease
    with project_lease(PROJECT_ROOT/'inspection_data').hold('Terminal camera'):
        _run_camera(args)


def _run_camera(args):
    project_config = load_project_detection_config()
    # Same entry point and camera window. Only an explicitly activated Core recipe
    # replaces the configured legacy inspection; Candidates never auto-select.
    web_app_dir = str(PROJECT_ROOT / '07_web_app')
    if web_app_dir not in sys.path:
        sys.path.insert(0,web_app_dir)
    from inspection.store import Store
    from inspection.runtime import Runtime
    from inspection.worker_client import AnomalyClient
    store = Store(PROJECT_ROOT/'inspection_data')
    if store.production() is not None:
        runtime = Runtime(store,AnomalyClient(PROJECT_ROOT,store),source='production')
        run_normal_reference_camera(args,project_config,core_runtime=runtime)
        return
    if str(project_config.get("inspection_strategy") or project_config.get("detection_mode") or "").lower() == "normal_reference":
        run_normal_reference_camera(args, project_config)
        return
    model_path = None
    airbag_model = None
    skipped_models = []
    configured_model = str(project_config.get("active_detector_model") or "").strip()
    dynamic_candidates = sorted(
        (PROJECT_DIR / "models").glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidate_paths = []
    seen_paths = set()

    preferred_candidates = [Path(configured_model)] if configured_model else []
    for candidate_path in [*preferred_candidates, *dynamic_candidates, *MODEL_CANDIDATES]:
        resolved = candidate_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        candidate_paths.append(candidate_path)

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        candidate_model = YOLO(str(candidate_path))
        if model_names_are_inspection_detector(candidate_model.names):
            model_path = candidate_path
            airbag_model = candidate_model
            break
        skipped_models.append((candidate_path, candidate_model.names))

    if model_path is None:
        candidate_text = "\n".join(f"  - {path}" for path in candidate_paths)
        skipped_text = "\n".join(
            f"  - skipped non-Airbag model {path}: {names}"
            for path, names in skipped_models
        )
        raise FileNotFoundError(
            "YOLO model weight not found. Checked:\n"
            f"{candidate_text}\n\n"
            f"{skipped_text}\n\n"
            "Webcam inference requires a trained .pt file, for example best.pt.\n"
            "Train the combined Airbag v1+v2 dataset with Background negative examples first:\n"
            "  python train_airbag_model.py\n\n"
            "After training finishes, run this webcam script again."
        )

    # มาถึงตรงนี้แปลว่าเจอโมเดล Airbag แล้ว ช่วยให้ editor รู้ว่าไม่ใช่ None
    assert airbag_model is not None

    print(f"Loading YOLO model: {model_path}")
    airbag_model.to(DEVICE)
    airbag_model.fuse()

    primary_fallback_model = None
    for fallback_path in MODEL_CANDIDATES:
        if not fallback_path.exists() or fallback_path.resolve() == model_path.resolve():
            continue
        candidate_fallback_model = YOLO(str(fallback_path))
        if model_names_are_airbag_only(candidate_fallback_model.names):
            primary_fallback_model = candidate_fallback_model
            primary_fallback_model.to(DEVICE)
            primary_fallback_model.fuse()
            print(f"Loading Airbag primary fallback model: {fallback_path}")
            break

    brand_model = None
    if not args.no_brand:
        configured_brand = str(project_config.get("active_brand_model") or "").strip()
        dynamic_brand_candidates = sorted(
            (PROJECT_DIR / "06_brand_classification" / "04_trained_models").glob("*_best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        brand_candidates = ([Path(configured_brand)] if configured_brand else []) + dynamic_brand_candidates + BRAND_MODEL_CANDIDATES
        seen_brand_paths = set()
        for brand_model_path in brand_candidates:
            resolved = brand_model_path.resolve()
            if resolved in seen_brand_paths:
                continue
            seen_brand_paths.add(resolved)
            if not brand_model_path.exists():
                continue
            candidate_brand_model = YOLO(str(brand_model_path))
            if getattr(candidate_brand_model, "task", "") == "classify" and model_names_are_brand_classifier(candidate_brand_model.names):
                brand_model = candidate_brand_model
                brand_model.to(DEVICE)
                print(f"Loading Airbag brand classifier: {brand_model_path}")
                print(f"Brand classes: {brand_model.names}")
                break
            print(
                "Warning: skipped non-brand classifier model "
                f"{brand_model_path}: {candidate_brand_model.names}"
            )

        if brand_model is None:
            print(
                "Warning: Airbag brand classifier not found. "
                "Only Airbag detection will be shown."
            )

    general_model = None
    if USE_GENERAL_NOT_AIRBAG_DETECTOR or USE_COCO_NEGATIVE_GUARD:
        general_model_path = next(
            (path for path in GENERAL_MODEL_CANDIDATES if path.exists()),
            None,
        )
        if general_model_path is not None:
            print(f"Loading general object model: {general_model_path}")
            general_model = YOLO(str(general_model_path))
            general_model.to(DEVICE)
            general_model.fuse()
        else:
            print(
                "Warning: yolov8n.pt was not found in the YOLO folder. "
                "Only Airbag detections will be shown; COCO negative guard is disabled."
            )

    cv2.setUseOptimized(True)
    cap, selected_camera_index = open_available_camera(args.camera_index)

    # Small buffer keeps the displayed frame close to real time.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    print(
        f"AI Detection started on camera index {selected_camera_index} "
        f"using {DEVICE}. Press 'q' to quit."
    )
    print(f"Primary class: {ACTIVE_PRIMARY_CLASS}")
    print(f"Target classes: {', '.join(ACTIVE_TARGET_CLASSES)}")
    print(f"Primary draw threshold: {args.airbag_conf:.2f}")
    print("Detector scans the full frame. COCO negative guard filters laptop/mouse/person false positives.")

    frame_index = 0
    cached_general_detections = []
    last_airbag_detections = []
    missed_airbag_frames = AIRBAG_HOLD_FRAMES + 1
    fps = 0.0
    last_time = time.perf_counter()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 1280, 960)
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: failed to read a frame from the webcam.")
                break

            if general_model is not None and frame_index % GENERAL_DETECT_EVERY_N_FRAMES == 0:
                general_results = general_model.predict(
                    frame,
                    conf=min(GENERAL_CONFIDENCE_THRESHOLD, COCO_NEGATIVE_GUARD_CONFIDENCE),
                    imgsz=GENERAL_IMAGE_SIZE,
                    max_det=12,
                    device=DEVICE,
                    **({"half": True} if USE_HALF else {}),
                    verbose=False,
                )

                general_result = general_results[0]
                general_class_names = general_result.names
                cached_general_detections = []

                if general_result.boxes is not None:
                    for box in general_result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = general_class_names.get(class_id, str(class_id))
                        cached_general_detections.append((get_xyxy(box), class_name, confidence))

            # Custom Airbag inference. imgsz=640 is a good accuracy/speed balance.
            # Increase imgsz for more accuracy if your computer can handle it.
            predict_confidence = (
                DEBUG_AIRBAG_CONFIDENCE_THRESHOLD
                if args.debug_detection
                else RAW_AIRBAG_CONFIDENCE_THRESHOLD
            )
            airbag_results = airbag_model.predict(
                frame,
                conf=predict_confidence,
                imgsz=AIRBAG_IMAGE_SIZE,
                iou=AIRBAG_NMS_IOU_THRESHOLD,
                max_det=MAX_AIRBAG_DETECTIONS,
                device=DEVICE,
                **({"half": True} if USE_HALF else {}),
                verbose=False,
            )

            airbag_result = airbag_results[0]
            airbag_class_names = airbag_result.names
            raw_airbag_candidates = []
            secondary_detections = []
            airbag_boxes = []
            best_raw_airbag_confidence = 0.0

            if airbag_result.boxes is not None:
                for box in airbag_result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = airbag_class_names.get(class_id, str(class_id))
                    xyxy = get_xyxy(box)

                    if not is_airbag_class(class_id, class_name):
                        if confidence >= SECONDARY_CONFIDENCE_THRESHOLD or args.debug_detection:
                            secondary_detections.append((xyxy, class_name, confidence))
                        continue

                    best_raw_airbag_confidence = max(best_raw_airbag_confidence, confidence)
                    if confidence < args.airbag_conf and not args.debug_detection:
                        # Keep very low-confidence candidates out of presentation mode,
                        # while still allowing debug mode to show the raw confidence.
                        continue

                    raw_airbag_candidates.append((xyxy, confidence))

            current_airbag_detections = select_airbag_detections(
                raw_airbag_candidates,
                frame.shape,
                cached_general_detections if USE_COCO_NEGATIVE_GUARD else [],
            )
            used_visual_fallback = False
            if not current_airbag_detections and primary_fallback_model is not None:
                fallback_candidates, fallback_best_confidence = find_airbag_only_candidates(
                    primary_fallback_model,
                    frame,
                    args.airbag_conf,
                    args.debug_detection,
                )
                best_raw_airbag_confidence = max(
                    best_raw_airbag_confidence,
                    fallback_best_confidence,
                )
                current_airbag_detections = select_airbag_detections(
                    fallback_candidates,
                    frame.shape,
                    cached_general_detections if USE_COCO_NEGATIVE_GUARD else [],
                )
            if not current_airbag_detections and primary_supports_airbag_assists():
                visual_candidates = find_visual_airbag_candidates(frame, secondary_detections)
                current_airbag_detections = select_airbag_detections(
                    visual_candidates,
                    frame.shape,
                    cached_general_detections if USE_COCO_NEGATIVE_GUARD else [],
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
            elif missed_airbag_frames < AIRBAG_HOLD_FRAMES:
                missed_airbag_frames += 1
                detections_to_draw = last_airbag_detections
            else:
                missed_airbag_frames += 1
                detections_to_draw = []

            for xyxy, confidence in detections_to_draw:
                brand_result = classify_airbag_brand(brand_model, frame, xyxy)
                if should_reject_by_brand(brand_result, xyxy, cached_general_detections):
                    # ถ้า crop นี้เป็น Unknown แบบมั่นใจมาก และ COCO เห็นซ้อนว่าเป็น
                    # laptop / phone / mouse ให้ถือว่าเป็น false positive ไม่ใช่ Airbag
                    continue

                airbag_boxes.append(xyxy)
                draw_detection(frame, xyxy, ACTIVE_PRIMARY_CLASS, confidence)
                if brand_result is not None:
                    brand_name, brand_confidence = brand_result
                    draw_brand_label(frame, xyxy, brand_name, brand_confidence)

            if args.debug_detection:
                status_color = (
                    (0, 255, 0)
                    if best_raw_airbag_confidence >= args.airbag_conf
                    else (0, 255, 255)
                    if best_raw_airbag_confidence > 0
                    else (0, 0, 255)
                )
                draw_status(
                    frame,
                    f"Raw {ACTIVE_PRIMARY_CLASS} confidence: {best_raw_airbag_confidence:.2f}",
                    status_color,
                )

            # General COCO inference for red "Not Airbag" boxes. This is needed
            # because the custom Roboflow model only knows the Airbag class.
            if USE_GENERAL_NOT_AIRBAG_DETECTOR and general_model is not None:
                for xyxy, class_name, confidence in cached_general_detections:
                    if overlaps_airbag(xyxy, airbag_boxes):
                        continue

                    draw_detection(frame, xyxy, class_name, confidence)

            secondary_detections = filter_secondary_detections(
                secondary_detections,
                airbag_boxes,
                debug_detection=args.debug_detection,
            )
            for xyxy, class_name, confidence in secondary_detections:
                draw_detection(frame, xyxy, class_name, confidence)

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
            last_time = now
            draw_fps(frame, fps)

            cv2.imshow(WINDOW_NAME, frame)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Window closed.")


if __name__ == "__main__":
    main()
