r"""
Pipeline รวม dataset Airbag, ทำ negative mining และเปิดกล้องทดสอบ

ไฟล์นี้มีสองหน้าที่ที่ต้องแยกให้ออก:
    1. --prepare-dataset รวม positive export หลาย version กับรูป background/non-Airbag
       โดยสร้าง label ว่างให้ negative image แล้วเขียน data.yaml ของ dataset รวม
    2. --camera ส่งต่อไปใช้ inference logic กลางใน 01_camera/airbag_webcam_detection.py

การ refresh dataset จะล้างเฉพาะไฟล์ dataset รวมที่สร้างใหม่ ไม่ลบ weights และ runs
ที่ train ไว้แล้ว ห้ามใส่ label Airbag ปลอมให้ background เพราะ negative mining ต้องใช้
ไฟล์ label ว่างเพื่อสอนว่าในภาพนั้นไม่มี object เป้าหมาย

ติดตั้ง:
    pip install ultralytics opencv-python

เตรียม dataset รวม:
    ..\venv\Scripts\python.exe 03_dataset\airbag_negative_mining_pipeline.py --prepare-dataset

train ด้วย dataset ใหม่หลังเตรียมเสร็จ:
    yolo task=detect mode=train model=yolov8n.pt data="%USERPROFILE%/Downloads/Airbag_v1_v2_with_Background/data.yaml" epochs=80 imgsz=640 project="%USERPROFILE%/Downloads/Airbag_v1_v2_with_Background/runs/detect" name=airbag_negative_mining

หลัง train ให้วาง best weight ที่:
    ~/Downloads/Airbag_v1_v2_with_Background/weights/best.pt

เปิดกล้องทดสอบ:
    ..\venv\Scripts\python.exe 03_dataset\airbag_negative_mining_pipeline.py --camera
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = PROJECT_ROOT

# ---------------------------------------------------------------------------
# PART 1: Dataset Augmentation Utility
# ---------------------------------------------------------------------------

# path กลุ่มนี้คือแหล่งข้อมูลหลักจาก Downloads และ output dataset รวมที่ใช้ train Airbag
AIRBAG_V1_DATASET_PATH = Path.home() / "Downloads" / "Airbag.v1i.yolov8"
AIRBAG_V2_DATASET_PATH = Path.home() / "Downloads" / "Airbag.v2i.yolov8"
AIRBAG_V3_DATASET_PATH = Path.home() / "Downloads" / "Airbag.v3i.yolov8"
ORIGINAL_DATASET_PATH = AIRBAG_V2_DATASET_PATH
BACKGROUND_IMAGES_PATH = Path.home() / "Downloads" / "Background"
WEBCAM_POSITIVE_IMAGES_PATH = Path.home() / "Downloads" / "Airbag_Webcam_Captures"
OUTPUT_AUGMENTED_PATH = Path.home() / "Downloads" / "Airbag_v1_v2_with_Background"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["Airbag"]

# label นี้ใช้กับรูป webcam capture ดิบ เพื่อเสริมเคสจริงที่ถือ Airbag หน้ากล้อง
# Manual positive labels for the raw webcam capture set.
#
# These captures show the real deployment view: the Airbag is held on the left
# side of the camera frame. Roboflow v3 contains heavily augmented/cropped
# variants with noisy polygon labels, so these direct webcam positives anchor
# the model back to the real full-frame object location.
WEBCAM_AIRBAG_BOX = (0.23, 0.30, 0.46, 0.50)  # x_center, y_center, width, height
SYNTHETIC_AIRBAG_POSITIONS = [
    (0.25, 0.30, 0.46, 0.50),
    (0.50, 0.32, 0.46, 0.50),
    (0.75, 0.32, 0.46, 0.50),
    (0.50, 0.62, 0.38, 0.42),
    (0.74, 0.62, 0.38, 0.42),
]

# After running:
#     ..\venv\Scripts\python.exe 03_dataset\airbag_negative_mining_pipeline.py --prepare-dataset
#
# Retrain on the NEW negative-mining dataset with:
#     yolo task=detect mode=train model=yolov8n.pt data="%USERPROFILE%/Downloads/Airbag_v1_v2_with_Background/data.yaml" epochs=80 imgsz=640 project="%USERPROFILE%/Downloads/Airbag_v1_v2_with_Background/runs/detect" name=airbag_negative_mining
#
# Then copy:
#     ~/Downloads/Airbag_v1_v2_with_Background/runs/detect/airbag_negative_mining/weights/best.pt
# to:
#     ~/Downloads/Airbag_v1_v2_with_Background/weights/best.pt


def is_image_file(path: Path) -> bool:
    """Return True for supported image files."""
    # กรองเฉพาะไฟล์รูปที่ YOLO/OpenCV ใช้ได้
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def unique_destination_path(destination_dir: Path, filename: str) -> Path:
    """Return a non-conflicting destination path inside destination_dir."""
    # กันชื่อไฟล์ซ้ำเวลาคัดลอก background หลายชุดเข้าที่เดียวกัน
    candidate = destination_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1

    while True:
        candidate = destination_dir / f"{stem}_background_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def clamp01(value: float) -> float:
    """Clamp a normalized YOLO coordinate to the valid 0..1 range."""
    return max(0.0, min(1.0, value))


def xyxy_to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float) -> str | None:
    """Convert clamped xyxy coordinates to one YOLO detection label line."""
    x1 = clamp01(x1)
    y1 = clamp01(y1)
    x2 = clamp01(x2)
    y2 = clamp01(y2)

    width = x2 - x1
    height = y2 - y1
    if width < 0.001 or height < 0.001:
        return None

    x_center = x1 + width / 2
    y_center = y1 + height / 2

    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    )


def sanitize_yolo_label_text(label_text: str) -> str:
    """
    Convert mixed Roboflow labels into clean YOLO detection bbox labels.

    Some Roboflow exports contain segmentation polygon rows instead of simple
    5-value detection rows, and some augmented polygons slightly exceed the
    normalized 0..1 coordinate range. YOLOv8 detection training marks those
    labels as corrupt and ignores the matching images. Sanitizing here keeps
    the original dataset untouched while making the combined training dataset
    usable for Airbag detection.
    """
    # ทำ label จาก Roboflow ให้เป็น bbox ปกติที่ YOLO detect train รับได้
    clean_lines: list[str] = []

    for raw_line in label_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            values = [float(part) for part in raw_line.split()]
        except ValueError:
            continue

        if len(values) < 5:
            continue

        class_id = int(values[0])

        if len(values) == 5:
            x_center, y_center, width, height = values[1:5]
            if width <= 0 or height <= 0:
                continue

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            clean_line = xyxy_to_yolo_line(class_id, x1, y1, x2, y2)
        else:
            coords = values[1:]
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            if len(coords) < 4:
                continue

            xs = [clamp01(value) for value in coords[0::2]]
            ys = [clamp01(value) for value in coords[1::2]]
            clean_line = xyxy_to_yolo_line(class_id, min(xs), min(ys), max(xs), max(ys))

        if clean_line is not None:
            clean_lines.append(clean_line)

    return "\n".join(clean_lines) + ("\n" if clean_lines else "")


def copy_sanitized_label(source_label_path: Path, destination_label_path: Path) -> None:
    """Copy a label file after converting polygons/out-of-range coords to boxes."""
    # อ่าน label ต้นทาง แปลงให้สะอาด แล้วเขียนลง dataset รวม
    label_text = source_label_path.read_text(encoding="utf-8")
    destination_label_path.write_text(
        sanitize_yolo_label_text(label_text),
        encoding="utf-8",
    )


def refresh_dataset_files(original_dataset_path: Path, output_augmented_path: Path) -> None:
    """
    Refresh train/valid/test dataset files without deleting trained artifacts.

    Important MLOps guardrail:
    The output folder also stores weights, runs, and verification reports. A
    full rmtree would delete the trained best.pt and make the webcam detector
    silently fall back to an older model. Only dataset inputs are refreshed.
    """
    # refresh เฉพาะไฟล์ dataset ไม่ลบ weights/runs ที่ train ไว้แล้ว
    output_augmented_path.mkdir(parents=True, exist_ok=True)

    dataset_entries_to_refresh = [
        "train",
        "valid",
        "test",
        "data.yaml",
        "README.dataset.txt",
        "README.roboflow.txt",
    ]

    for entry_name in dataset_entries_to_refresh:
        destination = output_augmented_path / entry_name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()

        source = original_dataset_path / entry_name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.exists():
            shutil.copy2(source, destination)


def copy_yolo_split_with_prefix(
    source_dataset_path: Path,
    output_dataset_path: Path,
    split_name: str,
    filename_prefix: str,
) -> int:
    """Copy one YOLO split into the combined dataset with prefixed filenames."""
    # คัดลอก train/valid/test ของ dataset หนึ่งชุด และเติม prefix เพื่อไม่ให้ชื่อชนกัน
    source_images_dir = source_dataset_path / split_name / "images"
    source_labels_dir = source_dataset_path / split_name / "labels"
    output_images_dir = output_dataset_path / split_name / "images"
    output_labels_dir = output_dataset_path / split_name / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if not source_images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {source_images_dir}")

    copied_images = 0
    for source_image_path in sorted(source_images_dir.iterdir()):
        if not is_image_file(source_image_path):
            continue

        destination_image_name = f"{filename_prefix}_{source_image_path.name}"
        destination_image_path = output_images_dir / destination_image_name
        shutil.copy2(source_image_path, destination_image_path)

        source_label_path = source_labels_dir / f"{source_image_path.stem}.txt"
        destination_label_path = output_labels_dir / f"{Path(destination_image_name).stem}.txt"

        if source_label_path.exists():
            copy_sanitized_label(source_label_path, destination_label_path)
        else:
            destination_label_path.write_text("", encoding="utf-8")

        copied_images += 1

    return copied_images


def refresh_combined_dataset(
    positive_dataset_paths: list[tuple[Path, str]],
    output_augmented_path: Path,
) -> dict[str, int]:
    """
    Build a combined Airbag dataset from v1, v2, and v3 without deleting weights.

    The output dataset receives:
        v1 train/valid/test positives
        v2 train/valid/test positives
        v3 train/valid/test positives
        Background images as empty-label negatives in train only
    """
    # รวม positive dataset หลาย version เป็น dataset เดียว โดยยังไม่แตะ weights ที่มีอยู่
    output_augmented_path.mkdir(parents=True, exist_ok=True)

    for entry_name in ["train", "valid", "test", "data.yaml"]:
        destination = output_augmented_path / entry_name
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()

    counts: dict[str, int] = {}
    for dataset_path, prefix in positive_dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Positive dataset not found: {dataset_path}")

        for split_name in ["train", "valid", "test"]:
            copied = copy_yolo_split_with_prefix(
                dataset_path,
                output_augmented_path,
                split_name,
                prefix,
            )
            counts[f"{prefix}_{split_name}_images"] = copied

    return counts


def copy_webcam_positive_examples(
    webcam_images_path: Path,
    output_dataset_path: Path,
) -> int:
    """Add raw webcam Airbag captures with a fixed full-object Airbag box."""
    # เพิ่มรูป Airbag จากกล้องจริงเพื่อให้โมเดลจำเคสใช้งานจริง ไม่ใช่จำแต่รูป export/crop
    if not webcam_images_path.exists():
        return 0

    output_images_dir = output_dataset_path / "train" / "images"
    output_labels_dir = output_dataset_path / "train" / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    label_line = "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*WEBCAM_AIRBAG_BOX)

    copied_images = 0
    for source_image_path in sorted(webcam_images_path.iterdir()):
        if not is_image_file(source_image_path):
            continue

        destination_image_name = f"webcam_positive_{source_image_path.name}"
        destination_image_path = output_images_dir / destination_image_name
        shutil.copy2(source_image_path, destination_image_path)

        destination_label_path = output_labels_dir / f"{Path(destination_image_name).stem}.txt"
        destination_label_path.write_text(label_line, encoding="utf-8")
        copied_images += 1

    return copied_images


def yolo_box_to_pixels(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO xywh into clamped pixel xyxy coordinates."""
    # ใช้ตอน crop/paste synthetic image และตอนสร้าง label ใหม่ให้ตรง pixel
    x_center, y_center, width, height = box
    x1 = int(round((x_center - width / 2) * image_width))
    y1 = int(round((y_center - height / 2) * image_height))
    x2 = int(round((x_center + width / 2) * image_width))
    y2 = int(round((y_center + height / 2) * image_height))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))

    return x1, y1, x2, y2


def paste_airbag_crop(
    background,
    airbag_crop,
    target_box: tuple[float, float, float, float],
):
    """Paste an Airbag crop into a target normalized box on a background image."""
    # วาง crop Airbag ลงบน background ที่ตำแหน่งต่างๆ เพื่อแก้ bias ว่าต้องอยู่มุมเดิม
    image_height, image_width = background.shape[:2]
    x1, y1, x2, y2 = yolo_box_to_pixels(target_box, image_width, image_height)
    target_width = x2 - x1
    target_height = y2 - y1

    resized_crop = cv2.resize(
        airbag_crop,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    composed = background.copy()
    composed[y1:y2, x1:x2] = resized_crop
    return composed


def copy_synthetic_position_examples(
    webcam_images_path: Path,
    background_images_path: Path,
    output_dataset_path: Path,
) -> int:
    """
    Create extra webcam-style positives with Airbag spread across the frame.

    This fixes the real issue seen in the webcam: if every full-frame positive
    has the Airbag in the same corner, the model becomes position-biased. These
    synthetic examples crop the Airbag from the real webcam captures and paste
    it onto negative background frames at left, center, right, and lower-frame
    positions with matching YOLO labels.
    """
    if not webcam_images_path.exists() or not background_images_path.exists():
        return 0

    source_images = [
        path for path in sorted(webcam_images_path.iterdir()) if is_image_file(path)
    ]
    background_images = [
        path for path in sorted(background_images_path.iterdir()) if is_image_file(path)
    ]
    if not source_images or not background_images:
        return 0

    output_images_dir = output_dataset_path / "train" / "images"
    output_labels_dir = output_dataset_path / "train" / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    created_images = 0
    for image_index, source_image_path in enumerate(source_images):
        source_image = cv2.imread(str(source_image_path))
        if source_image is None:
            continue

        source_height, source_width = source_image.shape[:2]
        crop_x1, crop_y1, crop_x2, crop_y2 = yolo_box_to_pixels(
            WEBCAM_AIRBAG_BOX,
            source_width,
            source_height,
        )
        airbag_crop = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if airbag_crop.size == 0:
            continue

        for position_index, target_box in enumerate(SYNTHETIC_AIRBAG_POSITIONS):
            background_path = background_images[
                (image_index + position_index) % len(background_images)
            ]
            background = cv2.imread(str(background_path))
            if background is None:
                continue

            if background.shape[:2] != source_image.shape[:2]:
                background = cv2.resize(
                    background,
                    (source_width, source_height),
                    interpolation=cv2.INTER_AREA,
                )

            composed = paste_airbag_crop(background, airbag_crop, target_box)
            destination_stem = (
                f"synthetic_position_{source_image_path.stem}_{position_index}"
            )
            destination_image_path = output_images_dir / f"{destination_stem}.jpg"
            destination_label_path = output_labels_dir / f"{destination_stem}.txt"

            cv2.imwrite(str(destination_image_path), composed)
            destination_label_path.write_text(
                "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*target_box),
                encoding="utf-8",
            )
            created_images += 1

    return created_images


def write_data_yaml(dataset_path: Path) -> Path:
    """
    Create/update data.yaml for the augmented YOLOv8 dataset.

    YOLOv8 needs this file to know where train/val/test images are and what
    class IDs mean. Class 0 is Airbag.
    """
    data_yaml_path = dataset_path / "data.yaml"
    data_yaml_text = f"""train: {dataset_path / "train" / "images"}
val: {dataset_path / "valid" / "images"}
test: {dataset_path / "test" / "images"}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    data_yaml_path.write_text(data_yaml_text, encoding="utf-8")
    return data_yaml_path


def merge_background_examples(
    original_dataset_path: Path = ORIGINAL_DATASET_PATH,
    background_images_path: Path = BACKGROUND_IMAGES_PATH,
    output_augmented_path: Path = OUTPUT_AUGMENTED_PATH,
) -> dict[str, int]:
    """
    Copy a YOLOv8 dataset and add negative background examples to train/images.

    The crucial MLOps detail:
    For every background image, we create an EMPTY .txt label file in
    train/labels. An empty YOLO label file explicitly means:

        "This image contains zero objects of interest."

    This is called negative mining. It teaches the detector that common
    background textures, desks, rooms, walls, lighting patterns, and camera
    views are NOT Airbag. This directly reduces false positives like drawing
    Airbag boxes over a table or blank background.
    """
    # สร้าง dataset รวม + negative mining ให้พร้อม train ด้วยคำสั่ง YOLO
    if not background_images_path.exists():
        raise FileNotFoundError(f"Background folder not found: {background_images_path}")

    summary = refresh_combined_dataset(
        [
            (AIRBAG_V1_DATASET_PATH, "v1"),
            (AIRBAG_V2_DATASET_PATH, "v2"),
            (AIRBAG_V3_DATASET_PATH, "v3"),
        ],
        output_augmented_path,
    )

    train_images_dir = output_augmented_path / "train" / "images"
    train_labels_dir = output_augmented_path / "train" / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    webcam_positive_count = copy_webcam_positive_examples(
        WEBCAM_POSITIVE_IMAGES_PATH,
        output_augmented_path,
    )
    synthetic_position_count = copy_synthetic_position_examples(
        WEBCAM_POSITIVE_IMAGES_PATH,
        background_images_path,
        output_augmented_path,
    )

    background_images = [
        path for path in sorted(background_images_path.iterdir()) if is_image_file(path)
    ]

    copied_backgrounds = 0
    for source_image_path in background_images:
        destination_image_path = unique_destination_path(
            train_images_dir,
            source_image_path.name,
        )
        shutil.copy2(source_image_path, destination_image_path)

        empty_label_path = train_labels_dir / f"{destination_image_path.stem}.txt"

        # Empty file on purpose:
        # YOLO interprets this as a valid training image with no Airbag objects.
        # Do NOT write class IDs or dummy boxes into this file.
        empty_label_path.write_text("", encoding="utf-8")

        copied_backgrounds += 1

    data_yaml_path = write_data_yaml(output_augmented_path)

    summary.update({
        "webcam_positive_images_added": webcam_positive_count,
        "synthetic_position_images_added": synthetic_position_count,
        "background_images_added": copied_backgrounds,
        "data_yaml_written": int(data_yaml_path.exists()),
    })
    return summary


# ---------------------------------------------------------------------------
# PART 2: Real-time Camera Inference
# ---------------------------------------------------------------------------

# ส่วนนี้เป็น inference เก่าในไฟล์ pipeline; งานเปิดกล้องหลักปัจจุบันอยู่ที่ 01_camera/airbag_webcam_detection.py
TRAINED_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "best.pt",
    OUTPUT_AUGMENTED_PATH / "weights" / "best.pt",
]
FALLBACK_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "yolov8n.pt",
    PROJECT_ROOT.parent / "yolov8n.pt",
    Path.cwd() / "yolov8n.pt",
]
GENERAL_NOT_AIRBAG_MODEL_CANDIDATES = FALLBACK_MODEL_CANDIDATES

WINDOW_NAME = "AI Airbag Detector (Negative Mining Trained)"
TARGET_CLASS_NAME = "Airbag"
CONFIDENCE_THRESHOLD = 0.45
GENERAL_CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
OVERLAP_IOU_THRESHOLD = 0.35


def calculate_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""
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

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def overlaps_any_airbag(
    box: tuple[int, int, int, int],
    airbag_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """Avoid drawing a red general-object box on top of a green Airbag box."""
    return any(
        calculate_iou(box, airbag_box) >= OVERLAP_IOU_THRESHOLD
        for airbag_box in airbag_boxes
    )


def draw_labeled_box(
    frame,
    xyxy: tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: tuple[int, int, int],
) -> None:
    """Draw a readable bounding box and label on the current frame."""
    # วาดกรอบและป้ายชื่อให้อ่านง่ายบนหน้าต่างกล้อง
    x1, y1, x2, y2 = xyxy
    text = f"{label} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
    )
    text_width, text_height = text_size
    text_y = max(y1, text_height + 10)

    cv2.rectangle(
        frame,
        (x1, text_y - text_height - baseline - 6),
        (x1 + text_width + 6, text_y + baseline - 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 3, text_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def load_inference_model() -> tuple[YOLO, Path, bool]:
    """
    Load the retrained Airbag model if available.

    If the trained Airbag best.pt does not exist,
    this falls back to yolov8n.pt only so the script can still run. The fallback
    COCO model is NOT trained for the custom Airbag class, so retraining on the
    augmented dataset is required for the intended accuracy improvement.
    """
    # โหลด best.pt ที่ train แล้วก่อน ถ้าไม่มีจึง fallback เป็น yolov8n.pt เพื่อให้สคริปต์ยังรันได้
    trained_model_path = next(
        (path for path in TRAINED_MODEL_CANDIDATES if path.exists()),
        None,
    )
    if trained_model_path is not None:
        return YOLO(str(trained_model_path)), trained_model_path, True

    fallback_model_path = next(
        (path for path in FALLBACK_MODEL_CANDIDATES if path.exists()),
        None,
    )
    if fallback_model_path is None:
        fallback_model_path = Path("yolov8n.pt")

    print(
        "Warning: retrained Airbag model was not found:\n"
        f"  {TRAINED_MODEL_CANDIDATES}\n"
        "Falling back to yolov8n.pt. Retrain on the augmented dataset and place "
        "best.pt at the path above to get Airbag-specific accuracy."
    )

    return YOLO(str(fallback_model_path)), fallback_model_path, False


def run_camera_inference(camera_index: int = 0) -> None:
    """Run the canonical webcam detector so there is only one inference logic."""
    # ส่งต่อไปใช้ไฟล์เปิดกล้องหลัก เพื่อไม่ให้มี logic detect ซ้ำหลายชุดจนแก้ไม่ตรงกัน
    if camera_index != 0:
        print("The canonical webcam detector accepts camera selection via --camera-index.")

    sys.path.insert(0, str(PROJECT_ROOT / "01_camera"))
    from airbag_webcam_detection import main as run_airbag_webcam

    run_airbag_webcam()


def parse_args() -> argparse.Namespace:
    """เลือกว่าจะเตรียม dataset, เปิดกล้อง หรือทำทั้งสองงานต่อเนื่องกัน."""
    # เลือกว่าจะเตรียม dataset, เปิดกล้อง หรือทำทั้งสองอย่าง
    parser = argparse.ArgumentParser(
        description="Prepare an Airbag negative-mining dataset and run webcam inference.",
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Create ~/Downloads/Airbag_v2i_with_Background with empty-label background examples.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Run real-time webcam inference.",
    )
    return parser.parse_args()


def main() -> None:
    """เรียก workflow ตาม flag โดยไม่ผสม logic เตรียมข้อมูลกับ inference เข้าด้วยกัน."""
    # entry point ของไฟล์ pipeline: ทำตาม flag ที่ผู้ใช้ระบุ
    args = parse_args()

    if args.prepare_dataset:
        summary = merge_background_examples()
        print(f"Augmented dataset created: {OUTPUT_AUGMENTED_PATH}")
        print(f"Webcam positive images added: {summary['webcam_positive_images_added']}")
        print(f"Synthetic position images added: {summary['synthetic_position_images_added']}")
        print(f"Background images added: {summary['background_images_added']}")
        print(f"data.yaml written: {bool(summary['data_yaml_written'])}")
        print(
            "\nNext train command:\n"
            f'yolo task=detect mode=train model=yolov8n.pt data="{OUTPUT_AUGMENTED_PATH / "data.yaml"}" '
            f'epochs=80 imgsz=640 project="{OUTPUT_AUGMENTED_PATH / "runs" / "detect"}" '
            'name=airbag_negative_mining'
        )

    if args.camera:
        run_camera_inference()

    if not args.prepare_dataset and not args.camera:
        print("Nothing selected. Use --prepare-dataset and/or --camera.")


if __name__ == "__main__":
    main()
