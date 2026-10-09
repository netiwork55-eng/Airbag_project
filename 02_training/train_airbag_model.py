r"""
จุดเริ่ม train Airbag detector แบบรวม positive dataset และ negative mining

ข้อมูลที่ใช้:
    ~/Downloads/Airbag.v1i.yolov8      -> positive Airbag examples
    ~/Downloads/Airbag.v2i.yolov8      -> positive Airbag examples
    ~/Downloads/Airbag.v3i.yolov8      -> positive Airbag examples
    ~/Downloads/Background             -> negative / Not-Airbag examples

ลำดับงาน: สร้าง dataset รวมผ่าน 03_dataset/airbag_negative_mining_pipeline.py,
เลือก base model หรือ checkpoint ที่ resume ได้, สั่ง YOLO train และคัดลอก best.pt
กลับเข้า models เพื่อให้ระบบกล้อง/เว็บใช้โมเดลใหม่

ข้อควรระวัง: --resume ต้องใช้ Airbag checkpoint ของ dataset เดิมเท่านั้น ห้ามนำ
COCO checkpoint หรือ checkpoint คนละ class มาต่อ เพราะ class contract จะไม่ตรงกัน

รัน:
    ..\venv\Scripts\python.exe 02_training\train_airbag_model.py

Continue training from the latest Airbag checkpoint:
    ..\venv\Scripts\python.exe 02_training\train_airbag_model.py --resume

หลัง train สำเร็จ weight สำหรับ inference จะถูกคัดลอกไปที่:
    ~/Downloads/Airbag_v1_v2_with_Background/weights/best.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# ใช้ PROJECT_ROOT เพื่อให้รันจากในโฟลเดอร์โปรเจกต์หรือจาก path อื่นก็ยังหาไฟล์ถูก
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# เพิ่ม 03_dataset เข้า import path เพื่อเรียก pipeline รวม dataset v1/v2/v3 + Background
sys.path.insert(0, str(PROJECT_ROOT / "03_dataset"))

from airbag_negative_mining_pipeline import merge_background_examples


PROJECT_DIR = PROJECT_ROOT

# output นี้เป็น dataset หลักที่รวม positive และ negative mining แล้ว
DATASET_DIR = Path.home() / "Downloads" / "Airbag_v1_v2_with_Background"
PREVIOUS_NEGATIVE_MODEL = Path.home() / "Downloads" / "Airbag_v2i_with_Background" / "weights" / "best.pt"
BASE_MODEL_CANDIDATES = [
    PROJECT_DIR / "models" / "yolov8n.pt",
    PROJECT_ROOT.parent / "yolov8n.pt",
    Path.cwd() / "yolov8n.pt",
]
RUN_NAME = "airbag_negative_mining"
RUN_DIR = DATASET_DIR / "runs" / "detect" / RUN_NAME
RUN_LAST_PT = RUN_DIR / "weights" / "last.pt"
FINAL_WEIGHTS_DIR = DATASET_DIR / "weights"
FINAL_BEST_PT = FINAL_WEIGHTS_DIR / "best.pt"
LOCAL_BEST_PT = PROJECT_DIR / "models" / "best.pt"


def write_absolute_data_yaml(dataset_dir):
    """Create a YOLO data.yaml with absolute paths for reliable training."""
    # ใช้ absolute path เพื่อลดปัญหา YOLO หา train/valid/test ไม่เจอเวลารันจากคนละโฟลเดอร์
    yaml_path = dataset_dir / "airbag_absolute_data.yaml"
    yaml_text = f"""train: {dataset_dir / "train" / "images"}
val: {dataset_dir / "valid" / "images"}
test: {dataset_dir / "test" / "images"}

nc: 1
names: ["Airbag"]
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def copy_final_weights(results):
    """Copy the best trained model to the dataset and project model locations."""
    # copy best.pt กลับเข้า models เพื่อให้ไฟล์เปิดกล้องโหลดโมเดลใหม่ได้ทันที
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    FINAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, FINAL_BEST_PT)
    LOCAL_BEST_PT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, LOCAL_BEST_PT)

    print(f"Training complete. YOLO run best weight: {best_pt}")
    print(f"Copied final inference weight to: {FINAL_BEST_PT}")
    print(f"Copied local project weight to: {LOCAL_BEST_PT}")
    print("Next step: ..\\venv\\Scripts\\python.exe 01_camera\\airbag_webcam_detection.py")


def parse_args():
    """อ่านตัวเลือก train/resume โดยไม่เริ่มงานหรือแก้ไฟล์."""
    # รับ option สำหรับ train ใหม่หรือ resume ต่อจาก last.pt เดิม
    parser = argparse.ArgumentParser(
        description="Train or resume the Airbag YOLOv8 negative-mining model."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue fine-tuning from runs/detect/airbag_negative_mining/weights/last.pt "
            "while forcing the Airbag data.yaml. This avoids accidentally falling back "
            "to YOLO's default coco8.yaml if last.pt is not a resumable optimizer checkpoint."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs to run. Default: 50.",
    )
    return parser.parse_args()


def main():
    """เตรียม dataset รวมแล้ว train/resume พร้อมคัดลอก weight ที่ผ่านการตรวจ."""
    # จุดเริ่ม train: เตรียม dataset, เลือก checkpoint/base model, แล้วสั่ง YOLO train
    args = parse_args()

    if args.resume:
        # resume จะใช้ checkpoint Airbag เดิมเท่านั้น เพื่อไม่ให้เผลอกลับไปใช้ COCO/coco8.yaml
        checkpoint = RUN_LAST_PT if RUN_LAST_PT.exists() else FINAL_BEST_PT
        if not checkpoint.exists():
            raise FileNotFoundError(
                "Cannot continue training because no Airbag checkpoint was found:\n"
                f"  {RUN_LAST_PT}\n"
                f"  {FINAL_BEST_PT}\n\n"
                "Run python train_airbag_model.py first to create an Airbag checkpoint."
            )

        if not DATASET_DIR.exists():
            raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

        data_yaml = write_absolute_data_yaml(DATASET_DIR)
        print(f"Continuing Airbag training from checkpoint: {checkpoint}")
        print(f"Forced Airbag data yaml: {data_yaml}")
        model = YOLO(str(checkpoint))
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=640,
            batch=-1,
            patience=12,
            project=str(DATASET_DIR / "runs" / "detect"),
            name=RUN_NAME,
            exist_ok=True,
        )
        copy_final_weights(results)
        return

    print("Preparing negative-mining dataset from Airbag.v1i.yolov8 + Airbag.v2i.yolov8 + Airbag.v3i.yolov8 + Background...")
    # รวม dataset positive หลาย version กับ background label ว่างก่อนเริ่ม train
    summary = merge_background_examples(output_augmented_path=DATASET_DIR)
    print(f"Webcam positive images added: {summary['webcam_positive_images_added']}")
    print(f"Synthetic position images added: {summary['synthetic_position_images_added']}")
    print(f"Background images added: {summary['background_images_added']}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    train_images = DATASET_DIR / "train" / "images"
    valid_images = DATASET_DIR / "valid" / "images"

    if not train_images.exists() or not valid_images.exists():
        raise FileNotFoundError(
            "Expected train/images and valid/images folders inside "
            f"{DATASET_DIR}"
        )

    data_yaml = write_absolute_data_yaml(DATASET_DIR)
    if FINAL_BEST_PT.exists():
        # ถ้ามี Airbag checkpoint ที่ train แล้ว ให้ fine-tune ต่อจากตัวนั้นก่อน
        model_source = FINAL_BEST_PT
    elif PREVIOUS_NEGATIVE_MODEL.exists():
        model_source = PREVIOUS_NEGATIVE_MODEL
    else:
        base_model = next((path for path in BASE_MODEL_CANDIDATES if path.exists()), None)
        model_source = base_model if base_model is not None else "yolov8n.pt"

    print(f"Dataset: {DATASET_DIR}")
    print(f"Data yaml: {data_yaml}")
    print(f"Base model: {model_source}")
    print("Training started. This can take several minutes or longer.")

    model = YOLO(str(model_source))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=-1,
        patience=12,
        project=str(DATASET_DIR / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
    )

    copy_final_weights(results)


if __name__ == "__main__":
    main()
