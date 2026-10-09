r"""
Verify the current Airbag project after cleanup.

Run from D:\Users\is00001\Desktop\YOLO\Airbag_Project:
    ..\venv\Scripts\python.exe 04_verify\verify_airbag_model.py

ตรวจ 2 เรื่องแยกกัน:
    1. positive Airbag dataset ต้องยังถูกโมเดลเห็นเป็น Airbag
    2. background ที่ไม่มี Airbag ต้องไม่ถูกแสดงกรอบใน logic กล้องจริง
"""

from pathlib import Path
import sys

from ultralytics import YOLO

# ใช้ path นี้เพื่อ import ฟังก์ชัน filter เดียวกับไฟล์เปิดกล้องจริง
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_camera"))

from airbag_webcam_detection import (  # noqa: E402
    AIRBAG_CONFIDENCE_THRESHOLD,
    AIRBAG_IMAGE_SIZE,
    AIRBAG_NMS_IOU_THRESHOLD,
    COCO_NEGATIVE_GUARD_CONFIDENCE,
    GENERAL_CONFIDENCE_THRESHOLD,
    GENERAL_IMAGE_SIZE,
    MAX_AIRBAG_DETECTIONS,
    get_xyxy,
    select_airbag_detections,
)


PROJECT_DIR = PROJECT_ROOT

# โฟลเดอร์ที่ใช้ตรวจ: dataset รวม, export ต้นทาง, รูปจากกล้อง และ background negative
TRAIN_NEW_OBJECT_DIR = PROJECT_DIR / "05_train_new_object"
EXPORTS_DIR = TRAIN_NEW_OBJECT_DIR / "01_yolov8_exports"
COMBINED_DATASET_DIR = TRAIN_NEW_OBJECT_DIR / "03_output" / "Airbag_yolov8_combined"
CAMERA_COLLECTION_DIR = TRAIN_NEW_OBJECT_DIR / "04_camera_collection"
BACKGROUND_IMAGES_DIR = TRAIN_NEW_OBJECT_DIR / "02_background_images"
CAMERA_BACKGROUND_DIR = CAMERA_COLLECTION_DIR / "02_background_images_no_airbag"

MODEL_CANDIDATES = [
    # เลือก Airbag_best.pt ก่อน เพราะเป็นโมเดลที่ตั้งใจใช้กับกล้องจริงในโปรเจกต์นี้
    PROJECT_DIR / "models" / "Airbag_best.pt",
    PROJECT_DIR / "models" / "best.pt",
    PROJECT_DIR / "models" / "AirbagV2_best.pt",
]
GENERAL_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "yolov8n.pt",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path, recursive: bool = False) -> list[Path]:
    """คืนรายการรูปภาพในโฟลเดอร์ ถ้าไม่มีโฟลเดอร์ให้คืน list ว่าง."""
    if not folder.exists():
        return []
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def raw_airbag_boxes(model: YOLO, image_path: Path) -> list[dict]:
    """ตรวจว่าโมเดล Airbag เห็น object หรือไม่ โดยยังไม่ใช้ presentation filter."""
    result = model.predict(
        source=str(image_path),
        conf=AIRBAG_CONFIDENCE_THRESHOLD,
        imgsz=AIRBAG_IMAGE_SIZE,
        iou=AIRBAG_NMS_IOU_THRESHOLD,
        max_det=MAX_AIRBAG_DETECTIONS,
        classes=[0],
        verbose=False,
        save=False,
    )[0]

    return [
        {
            "confidence": round(float(box.conf[0]), 3),
            "xyxy": get_xyxy(box),
        }
        for box in result.boxes
    ]


def webcam_presented_airbag_boxes(
    airbag_model: YOLO,
    general_model: YOLO,
    image_path: Path,
) -> list[dict]:
    """ใช้ filter เดียวกับหน้ากล้องจริง เพื่อตรวจว่าจะมีกรอบเขียวแสดงไหม."""
    airbag_result = airbag_model.predict(
        source=str(image_path),
        conf=0.05,
        imgsz=AIRBAG_IMAGE_SIZE,
        iou=AIRBAG_NMS_IOU_THRESHOLD,
        max_det=MAX_AIRBAG_DETECTIONS,
        classes=[0],
        verbose=False,
        save=False,
    )[0]
    candidates = []
    for box in airbag_result.boxes:
        confidence = float(box.conf[0])
        if confidence >= AIRBAG_CONFIDENCE_THRESHOLD:
            candidates.append((get_xyxy(box), confidence))

    general_result = general_model.predict(
        source=str(image_path),
        conf=min(GENERAL_CONFIDENCE_THRESHOLD, COCO_NEGATIVE_GUARD_CONFIDENCE),
        imgsz=GENERAL_IMAGE_SIZE,
        max_det=12,
        verbose=False,
        save=False,
    )[0]
    negative_detections = []
    for box in general_result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = general_result.names.get(class_id, str(class_id))
        negative_detections.append((get_xyxy(box), class_name, confidence))

    selected = select_airbag_detections(
        candidates,
        airbag_result.orig_img.shape,
        negative_detections,
    )
    return [
        {
            "confidence": round(confidence, 3),
            "xyxy": xyxy,
        }
        for xyxy, confidence in selected
    ]


def verify_positive_folder(
    model: YOLO,
    image_paths: list[Path],
    label: str,
    min_detection_rate: float = 1.0,
) -> bool:
    """positive dataset ต้องมี raw Airbag detection ตามสัดส่วนที่กำหนด."""
    # ตรวจว่ารูปที่มี Airbag จริงยังถูกโมเดลเห็น ไม่ใช่ threshold/filter กดทิ้งหมด
    detected_images = 0
    total_boxes = 0
    print(f"\n{label}: {len(image_paths)} images")
    for image_path in image_paths:
        boxes = raw_airbag_boxes(model, image_path)
        if boxes:
            detected_images += 1
            total_boxes += len(boxes)
        print(f"{image_path.name}: boxes={len(boxes)}, kept={boxes}")

    detection_rate = detected_images / len(image_paths) if image_paths else 0.0
    passed = bool(image_paths) and detection_rate >= min_detection_rate
    print(f"{label} detected images: {detected_images}/{len(image_paths)}")
    print(f"{label} detection rate: {detection_rate:.3f}")
    print(f"{label} total boxes: {total_boxes}")
    print(f"{label} pass: {passed}")
    return passed


def verify_background_folder(
    airbag_model: YOLO,
    general_model: YOLO,
    image_paths: list[Path],
    label: str,
) -> bool:
    """background ต้องไม่เหลือกรอบหลังผ่าน filter กล้องจริง."""
    # ตรวจรูปที่ไม่มี Airbag ว่าหลังผ่าน filter แล้วไม่ควรวาดกรอบเขียว
    false_positive_images = 0
    total_boxes = 0
    print(f"\n{label}: {len(image_paths)} images")
    for image_path in image_paths:
        boxes = webcam_presented_airbag_boxes(airbag_model, general_model, image_path)
        if boxes:
            false_positive_images += 1
            total_boxes += len(boxes)
            print(f"FALSE POSITIVE {image_path.name}: boxes={len(boxes)}, kept={boxes}")

    passed = false_positive_images == 0
    print(f"{label} false positive images: {false_positive_images}/{len(image_paths)}")
    print(f"{label} total presented boxes: {total_boxes}")
    print(f"{label} pass: {passed}")
    return passed


def main() -> None:
    """โหลดโมเดลและตรวจชุด positive/negative ด้วย filter เดียวกับระบบกล้องจริง."""
    # โหลดโมเดล Airbag + COCO guard แล้วตรวจ positive/negative แยกกัน
    model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
    if model_path is None:
        checked = "\n".join(f"  - {path}" for path in MODEL_CANDIDATES)
        raise FileNotFoundError(f"No trained Airbag model found. Checked:\n{checked}")

    general_model_path = next((path for path in GENERAL_MODEL_CANDIDATES if path.exists()), None)
    if general_model_path is None:
        checked = "\n".join(f"  - {path}" for path in GENERAL_MODEL_CANDIDATES)
        raise FileNotFoundError(f"No general YOLO guard model found. Checked:\n{checked}")

    print(f"Airbag model: {model_path}")
    print(f"General guard model: {general_model_path}")
    print(f"Confidence threshold: {AIRBAG_CONFIDENCE_THRESHOLD}")

    airbag_model = YOLO(str(model_path))
    general_model = YOLO(str(general_model_path))

    checks: list[bool] = []
    checks.append(
        verify_positive_folder(
            airbag_model,
            image_files(COMBINED_DATASET_DIR / "test" / "images"),
            "Combined Airbag Test",
        )
    )

    for export_name in ["Airbag.v1i.yolov8", "Airbag.v2i.yolov8", "Airbag.v3i.yolov8"]:
        checks.append(
            verify_positive_folder(
                airbag_model,
                image_files(EXPORTS_DIR / export_name / "test" / "images"),
                f"{export_name} Test",
            )
        )

    camera_positive_images = image_files(
        CAMERA_COLLECTION_DIR / "01_airbag_images_to_label",
        recursive=False,
    )[:80]
    checks.append(
        verify_positive_folder(
            airbag_model,
            camera_positive_images,
            "Camera Airbag Sample",
            min_detection_rate=0.95,
        )
    )

    checks.append(
        verify_background_folder(
            airbag_model,
            general_model,
            image_files(BACKGROUND_IMAGES_DIR, recursive=True),
            "Project Background Negative",
        )
    )
    checks.append(
        verify_background_folder(
            airbag_model,
            general_model,
            image_files(CAMERA_BACKGROUND_DIR, recursive=True),
            "Camera Background Negative",
        )
    )

    print("\nOverall pass:", all(checks))
    print("Positive checks use raw Airbag model confidence.")
    print("Background checks use the same presentation filter and COCO guard as the webcam script.")


if __name__ == "__main__":
    main()
