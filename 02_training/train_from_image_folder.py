"""
สร้างและ train YOLOv8 detector แบบง่ายจากโฟลเดอร์รูปธรรมดา

ใช้ workflow นี้เมื่อต้องการทดลอง object ใหม่อย่างรวดเร็ว:
  1. Put positive object images in:
       05_train_new_object/99_raw_image_fallback/01_positive_images
  2. Put optional background / not-object images in:
       05_train_new_object/99_raw_image_fallback/02_background_images
  3. Run:
       python train_from_image_folder.py --class-name Airbag

ข้อควรระวังสำคัญ:
ระบบสร้าง bounding box เต็มภาพให้ positive image โดยค่าเริ่มต้น จึงเหมาะเฉพาะรูปที่
crop ชิด object หรือ object กินพื้นที่เกือบทั้งภาพ ไม่ควรใช้แทน label ที่คนตรวจแล้ว
สำหรับงานจริง ให้ใช้ Roboflow/LabelImg/CVAT หรือ Annotation Studio ของเว็บ

ผลลัพธ์คือ dataset YOLO ชั่วคราวและ best.pt; การรันอาจล้าง output dataset เดิมของ
workflow นี้ แต่ไม่ควรแตะ export ต้นฉบับ
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


# path กลุ่มนี้ใช้สำหรับ workflow แบบง่าย: เอารูป positive/background ธรรมดามาสร้าง dataset YOLO
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "05_train_new_object"
RAW_IMAGE_FALLBACK_DIR = WORKSPACE_DIR / "99_raw_image_fallback"
DEFAULT_POSITIVE_DIR = RAW_IMAGE_FALLBACK_DIR / "01_positive_images"
DEFAULT_BACKGROUND_DIR = RAW_IMAGE_FALLBACK_DIR / "02_background_images"
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / "03_output"
MODEL_DIR = PROJECT_ROOT / "models"
BASE_MODEL_CANDIDATES = [
    MODEL_DIR / "yolov8n.pt",
    PROJECT_ROOT.parent / "yolov8n.pt",
    Path.cwd() / "yolov8n.pt",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def clean_class_name(class_name: str) -> str:
    # ทำชื่อ class ให้ปลอดภัยสำหรับใช้เป็นชื่อโฟลเดอร์และชื่อไฟล์ output
    keep = []
    for char in class_name.strip():
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "custom_object"


def parse_box(box_text: str) -> tuple[float, float, float, float]:
    # อ่านกรอบ normalized จาก command line สำหรับกรณีใช้ full-image หรือ crop ง่ายๆ
    parts = [part.strip() for part in box_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must be x_center,y_center,width,height")

    values = tuple(float(part) for part in parts)
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("--box values must be normalized between 0 and 1")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("--box width and height must be greater than 0")
    return values


def image_files(folder: Path) -> list[Path]:
    # คืนเฉพาะไฟล์รูปที่รองรับ ถ้าโฟลเดอร์ยังไม่มีให้คืน list ว่าง
    if not folder.exists():
        return []
    return [path for path in sorted(folder.iterdir()) if is_image(path)]


def copy_image_and_label(
    image_path: Path,
    split_dir: Path,
    label_text: str,
    prefix: str,
) -> None:
    # copy รูปพร้อมสร้าง label; label ว่างหมายถึงรูป background ที่ไม่มี object เป้าหมาย
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    destination_name = f"{prefix}_{image_path.name}"
    destination_image = images_dir / destination_name
    destination_label = labels_dir / f"{Path(destination_name).stem}.txt"

    shutil.copy2(image_path, destination_image)
    destination_label.write_text(label_text, encoding="utf-8")


def split_train_valid(images: list[Path], valid_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    # แบ่ง train/valid แบบสุ่มซ้ำได้ด้วย seed เดิม
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) <= 2:
        return shuffled, shuffled

    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    valid_images = shuffled[:valid_count]
    train_images = shuffled[valid_count:]
    return train_images, valid_images


def write_data_yaml(dataset_dir: Path, class_name: str) -> Path:
    # สร้าง data.yaml สำหรับ dataset หนึ่ง class ที่สร้างจากรูปธรรมดา
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"""train: {dataset_dir / "train" / "images"}
val: {dataset_dir / "valid" / "images"}
test: {dataset_dir / "valid" / "images"}

nc: 1
names: ["{class_name}"]
""",
        encoding="utf-8",
    )
    return data_yaml


def build_dataset(
    class_name: str,
    positive_dir: Path,
    background_dir: Path,
    output_root: Path,
    box: tuple[float, float, float, float],
    valid_ratio: float,
    seed: int,
) -> Path:
    # สร้าง dataset YOLOv8 จาก positive images และ optional background images
    positives = image_files(positive_dir)
    backgrounds = image_files(background_dir)

    if not positives:
        raise FileNotFoundError(
            "No positive images found. Put object images in:\n"
            f"  {positive_dir}"
        )

    dataset_dir = output_root / f"{clean_class_name(class_name)}_dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_pos, valid_pos = split_train_valid(positives, valid_ratio, seed)
    train_bg, valid_bg = split_train_valid(backgrounds, valid_ratio, seed) if backgrounds else ([], [])

    label_line = "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*box)

    for image_path in train_pos:
        copy_image_and_label(image_path, dataset_dir / "train", label_line, "pos")
    for image_path in valid_pos:
        copy_image_and_label(image_path, dataset_dir / "valid", label_line, "pos")

    for image_path in train_bg:
        copy_image_and_label(image_path, dataset_dir / "train", "", "bg")
    for image_path in valid_bg:
        copy_image_and_label(image_path, dataset_dir / "valid", "", "bg")

    write_data_yaml(dataset_dir, class_name)
    print(f"Dataset created: {dataset_dir}")
    print(f"Positive images: train={len(train_pos)}, valid={len(valid_pos)}")
    print(f"Background images: train={len(train_bg)}, valid={len(valid_bg)}")
    return dataset_dir


def choose_base_model() -> str:
    # เลือก base model ในโปรเจกต์ก่อน ถ้าไม่มีให้ Ultralytics ดาวน์โหลด/ใช้ yolov8n.pt
    base_model = next((path for path in BASE_MODEL_CANDIDATES if path.exists()), None)
    return str(base_model) if base_model is not None else "yolov8n.pt"


def train_model(
    class_name: str,
    dataset_dir: Path,
    epochs: int,
    imgsz: int,
) -> Path:
    # train YOLO แล้ว copy best.pt เข้า models เพื่อเอาไปใช้ inference ต่อ
    data_yaml = dataset_dir / "data.yaml"
    run_name = f"{clean_class_name(class_name)}_train"
    model = YOLO(choose_base_model())
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
        patience=10,
        project=str(dataset_dir / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model = MODEL_DIR / f"{clean_class_name(class_name)}_best.pt"
    shutil.copy2(best_pt, final_model)
    print(f"Training complete: {best_pt}")
    print(f"Copied model to: {final_model}")
    return final_model


def parse_args() -> argparse.Namespace:
    """อ่าน path รูป, class, bbox เริ่มต้น และค่าฝึกจาก command line."""
    # รับค่า class, path รูป, epoch และกล่อง default จาก command line
    parser = argparse.ArgumentParser(
        description="Train a one-class detector from positive/background image folders.",
    )
    parser.add_argument("--class-name", required=True, help="Object name, for example Airbag.")
    parser.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--box",
        default="0.5,0.5,1.0,1.0",
        help=(
            "Normalized bbox used for every positive image: "
            "x_center,y_center,width,height. Default means full image."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the YOLO dataset; do not train.",
    )
    return parser.parse_args()


def main() -> None:
    """สร้าง dataset จากโฟลเดอร์รูป และ train เว้นแต่ผู้ใช้สั่ง prepare-only."""
    # pipeline หลัก: แปลงรูปเป็น dataset แล้ว train ถ้าไม่ได้สั่ง --prepare-only
    args = parse_args()
    box = parse_box(args.box)
    dataset_dir = build_dataset(
        class_name=args.class_name,
        positive_dir=args.positive_dir,
        background_dir=args.background_dir,
        output_root=args.output_root,
        box=box,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    if args.prepare_only:
        print("Prepared dataset only. Training skipped.")
        return

    train_model(args.class_name, dataset_dir, args.epochs, args.imgsz)


if __name__ == "__main__":
    main()
