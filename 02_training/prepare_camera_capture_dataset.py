"""
สร้าง YOLOv8 dataset ฉบับร่างจากรูปที่ถ่ายด้วยกรอบไกด์ของกล้อง

อ่านรูปจาก:
    05_train_new_object/04_camera_collection/01_airbag_images_to_label

ระบบอ่าน scenario จากชื่อไฟล์ แล้วหา object สีเข้มภายในกรอบไกด์เพื่อประมาณ bbox,
แบ่ง train/valid/test, เติม background label ว่าง และสร้าง review sheet

ข้อควรระวัง: label ที่ได้เป็น draft เท่านั้น ต้องตรวจ review sheet และแก้กรอบก่อน train
เสมอ เพราะกรอบผิดจะสอนโมเดลให้จำฉากหลังหรือส่วนที่ไม่ใช่ Airbag ได้ การสร้างใหม่จะล้าง
เฉพาะ output draft ของ workflow นี้ ไม่ลบรูปกล้องต้นฉบับ
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


# path กลุ่มนี้ชี้ไปยังรูปที่ถ่ายจากกล้องและ output dataset draft ที่จะสร้างจากรูปเหล่านั้น
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "05_train_new_object" / "04_camera_collection"
POSITIVE_DIR = COLLECTION_ROOT / "01_airbag_images_to_label"
BACKGROUND_DIR = COLLECTION_ROOT / "02_background_images_no_airbag"
OUTPUT_DIR = COLLECTION_ROOT / "03_yolov8_auto_labeled_draft"
PREVIEW_PATH = COLLECTION_ROOT / "auto_label_review.jpg"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = [("train", 0.80), ("valid", 0.15), ("test", 0.05)]

# ตารางนี้ผูกชื่อ scenario ในไฟล์รูปกับกรอบคร่าวๆ ที่เคยใช้ตอนถ่าย เพื่อช่วยเดา label เบื้องต้น
SCENARIO_BOXES = {
    "center_medium_normal": (0.30, 0.22, 0.70, 0.78),
    "left_medium_normal": (0.08, 0.22, 0.48, 0.78),
    "right_medium_normal": (0.52, 0.22, 0.92, 0.78),
    "top_medium_normal": (0.30, 0.06, 0.70, 0.56),
    "bottom_medium_normal": (0.30, 0.44, 0.70, 0.94),
    "close_large_center": (0.16, 0.10, 0.84, 0.90),
    "far_small_center": (0.38, 0.30, 0.62, 0.70),
    "angle_left_30": (0.26, 0.18, 0.74, 0.82),
    "angle_right_30": (0.26, 0.18, 0.74, 0.82),
    "tilt_up_down": (0.26, 0.18, 0.74, 0.82),
    "rotated_15_deg": (0.25, 0.18, 0.75, 0.82),
    "bright_light": (0.28, 0.18, 0.72, 0.82),
    "dim_light": (0.28, 0.18, 0.72, 0.82),
    "backlight": (0.28, 0.18, 0.72, 0.82),
    "shadow_half": (0.28, 0.18, 0.72, 0.82),
    "busy_background": (0.28, 0.18, 0.72, 0.82),
    "plain_background": (0.28, 0.18, 0.72, 0.82),
    "hand_partial_occlusion": (0.24, 0.16, 0.76, 0.84),
    "motion_slight_blur": (0.28, 0.18, 0.72, 0.82),
    "edge_partly_cut": (0.00, 0.24, 0.36, 0.80),
}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def scenario_from_name(image_path: Path) -> str | None:
    # อ่าน scenario จากชื่อไฟล์ เช่น airbag_center_medium_normal_...jpg
    name = image_path.stem
    for scenario in sorted(SCENARIO_BOXES, key=len, reverse=True):
        if f"_{scenario}_" in name:
            return scenario
    return None


def normalized_to_pixels(box, width: int, height: int) -> tuple[int, int, int, int]:
    # แปลงพิกัด normalized 0..1 ให้เป็น pixel จริงของภาพ
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1 * width)),
        max(0, int(y1 * height)),
        min(width - 1, int(x2 * width)),
        min(height - 1, int(y2 * height)),
    )


def expand_box(box, width: int, height: int, pad_ratio: float = 0.08):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    )


def yolo_label_from_box(box, width: int, height: int) -> str:
    # แปลงกรอบ pixel xyxy เป็น format label ของ YOLO: class x_center y_center width height
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"


def find_dark_object_box(image, guide_box) -> tuple[int, int, int, int] | None:
    # หา object สีเข้มในกรอบไกด์ เพื่อสร้าง label draft ของ Airbag จากรูปกล้องจริง
    height, width = image.shape[:2]
    gx1, gy1, gx2, gy2 = normalized_to_pixels(guide_box, width, height)
    roi = image[gy1:gy2, gx1:gx2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold around the lower brightness range. The Airbag is dark
    # in the current camera captures, while the desk/background is much lighter.
    threshold_value = min(120, max(45, int(np.percentile(blurred, 42))))
    mask = cv2.inRange(blurred, 0, threshold_value)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return normalized_to_pixels(guide_box, width, height)

    roi_area = roi.shape[0] * roi.shape[1]
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < roi_area * 0.03:
            continue
        if w < 30 or h < 30:
            continue
        aspect = w / max(h, 1)
        if not 0.45 <= aspect <= 2.4:
            continue
        candidates.append((area, x, y, w, h))

    if not candidates:
        return normalized_to_pixels(guide_box, width, height)

    _, x, y, w, h = max(candidates, key=lambda item: item[0])
    box = (gx1 + x, gy1 + y, gx1 + x + w, gy1 + y + h)
    return expand_box(box, width, height)


def split_images(images: list[Path]) -> dict[str, list[Path]]:
    # แบ่งรูปเป็น train/valid/test แบบคงที่ด้วย seed เดิม เพื่อให้ผลซ้ำได้
    random.seed(42)
    shuffled = images[:]
    random.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * SPLITS[0][1])
    valid_end = train_end + int(total * SPLITS[1][1])
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def prepare_output():
    # ล้าง output draft เดิม แล้วสร้างโครงสร้าง YOLOv8 ใหม่
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split, _ in SPLITS:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    # เขียน data.yaml ให้ YOLO รู้ path ของภาพและ class เดียวคือ Airbag
    data = {
        "train": str(OUTPUT_DIR / "train" / "images"),
        "val": str(OUTPUT_DIR / "valid" / "images"),
        "test": str(OUTPUT_DIR / "test" / "images"),
        "nc": 1,
        "names": ["Airbag"],
    }
    (OUTPUT_DIR / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def copy_backgrounds():
    # คัดลอกรูปที่ไม่มี Airbag พร้อม label ว่าง เพื่อสอนว่า background ไม่ต้อง detect
    if not BACKGROUND_DIR.exists():
        return 0
    count = 0
    for image_path in sorted(BACKGROUND_DIR.iterdir()):
        if not is_image(image_path):
            continue
        dest_image = OUTPUT_DIR / "train" / "images" / f"background_{image_path.name}"
        dest_label = OUTPUT_DIR / "train" / "labels" / f"background_{image_path.stem}.txt"
        shutil.copy2(image_path, dest_image)
        dest_label.write_text("", encoding="utf-8")
        count += 1
    return count


def create_review_sheet(review_items):
    # สร้างภาพรวมสำหรับตรวจเร็วๆ ว่ากรอบ draft ที่เดาไว้ครอบ Airbag ถูกหรือไม่
    thumbs = []
    for image_path, box in review_items[:32]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        thumb = cv2.resize(image, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (239, 24), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            image_path.stem.replace("airbag_", "")[:28],
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)

    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * 135, cols * 240, 3), 32, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        y = (index // cols) * 135
        x = (index % cols) * 240
        sheet[y : y + 135, x : x + 240] = thumb
    cv2.imwrite(str(PREVIEW_PATH), sheet)


def main() -> None:
    """สร้าง draft label, split, background และ review sheet จากรูปกล้องต้นฉบับ."""
    # pipeline หลัก: อ่านรูปกล้อง สร้าง label draft รวม background แล้วเขียน dataset YOLOv8
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dir", type=Path, default=POSITIVE_DIR)
    args = parser.parse_args()

    images = sorted(path for path in args.positive_dir.iterdir() if is_image(path))
    if not images:
        raise FileNotFoundError(f"No camera capture images found: {args.positive_dir}")

    prepare_output()
    split_map = split_images(images)
    review_items = []
    skipped = 0

    for split, split_images_list in split_map.items():
        for image_path in split_images_list:
            scenario = scenario_from_name(image_path)
            if scenario is None:
                skipped += 1
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue
            height, width = image.shape[:2]
            box = find_dark_object_box(image, SCENARIO_BOXES[scenario])
            if box is None:
                skipped += 1
                continue

            dest_image = OUTPUT_DIR / split / "images" / image_path.name
            dest_label = OUTPUT_DIR / split / "labels" / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dest_image)
            dest_label.write_text(yolo_label_from_box(box, width, height), encoding="utf-8")
            review_items.append((image_path, box))

    background_count = copy_backgrounds()
    write_data_yaml()
    create_review_sheet(review_items)

    print(f"Output dataset: {OUTPUT_DIR}")
    print(f"Positive images labeled: {len(review_items)}")
    print(f"Background images copied: {background_count}")
    print(f"Skipped images: {skipped}")
    print(f"Review sheet: {PREVIEW_PATH}")
    print("Review the green boxes before training. Draft boxes are not perfect labels.")


if __name__ == "__main__":
    main()
