"""
สร้าง dataset และ train โมเดลจำแนกแบรนด์/รุ่นของ Airbag

โมเดลนี้เป็น classification ขั้นที่สอง: ระบบต้อง detect และ crop Airbag ก่อน แล้วจึง
ส่ง crop มาจำแนก Toyota/Honda/Unknown จึงห้ามนำ weight นี้ไปใช้แทน detector

ข้อมูลเข้า:
    01_brand_exports/Toyota
    01_brand_exports/Honda
    02_unknown_background_images

รองรับทั้งโฟลเดอร์รูปธรรมดาและ exported dataset ระบบค้นหารูปแบบ recursive, อ่าน
dataset_meta.json เพื่อกำหนดชื่อ class, แบ่ง train/val, train YOLOv8-cls และเขียน
metadata คู่กับ weight เพื่อให้ระบบกล้องและหน้าเว็บตรวจ class contract ได้

ข้อควรระวัง: Unknown/background เป็น class จริงของ classifier ไม่ใช่ negative label ว่าง
แบบ detector และการเปลี่ยนชื่อโฟลเดอร์หรือ metadata อาจเปลี่ยนชื่อ class ตอน train
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# workspace นี้แยกจาก Airbag detector หลัก เพราะงานนี้คือ classify แบรนด์ Toyota/Honda/Unknown
WORKSPACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_DIR.parent
EXPORTS_DIR = WORKSPACE_DIR / "01_brand_exports"
WORKSPACE_EXPORTS_DIR = PROJECT_ROOT / "05_train_new_object" / "01_yolov8_exports"
UNKNOWN_DIR = WORKSPACE_DIR / "02_unknown_background_images"
OUTPUT_DATASET_DIR = WORKSPACE_DIR / "03_output_dataset" / "airbag_brand_cls"
TRAINED_MODELS_DIR = WORKSPACE_DIR / "04_trained_models"
BASE_MODEL_CANDIDATES = [
    WORKSPACE_DIR / "04_trained_models" / "airbag_brand_best.pt",
    PROJECT_ROOT / "models" / "yolov8n-cls.pt",
    Path.cwd() / "yolov8n-cls.pt",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def clean_name(text: str) -> str:
    # ทำชื่อไฟล์ปลายทางให้สะอาด กัน path ยาวหรืออักขระแปลกจาก export
    keep = []
    for char in text.strip():
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "image"


def clean_class_name(text: str) -> str:
    # ชื่อ class ของ YOLO classification ใช้ชื่อที่อ่านง่าย แต่ตัดอักขระที่เสี่ยงต่อ path ออก
    keep = []
    for char in text.strip():
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        elif char.isspace() or char in {"/", "\\"}:
            keep.append("_")
    return "_".join(part for part in "".join(keep).split("_") if part) or "Class"


def read_dataset_meta(folder_path: Path) -> dict:
    # อ่าน metadata ที่หน้า Training บันทึกไว้ เพื่อเอา brand/model มาใช้เป็น class จริง
    meta_path = folder_path / "dataset_meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def class_name_from_meta(meta: dict, fallback: str) -> str:
    # ถ้ามี brand/model ให้ใช้เป็นชื่อ class หลัก ถ้าไม่มีค่อย fallback ไป dataset_name/folder
    brand_model = " ".join(
        str(part or "").strip()
        for part in [meta.get("brand"), meta.get("model")]
        if str(part or "").strip()
    ).strip()
    return clean_class_name(brand_model or str(meta.get("dataset_name") or fallback))


def discover_workspace_classification_sources() -> dict[str, list[Path]]:
    """ค้นหา workspace ที่ metadata ระบุเป็น Classification แล้วจัดกลุ่มตามชื่อ class."""
    # ดึง folder จากหน้า Workspace ที่ user ตั้ง Dataset Type เป็น Classification
    sources: dict[str, list[Path]] = {}
    if not WORKSPACE_EXPORTS_DIR.exists():
        return sources
    for folder_path in sorted(path for path in WORKSPACE_EXPORTS_DIR.iterdir() if path.is_dir()):
        meta = read_dataset_meta(folder_path)
        if meta.get("dataset_type") != "Classification":
            continue
        class_name = class_name_from_meta(meta, folder_path.name)
        sources.setdefault(class_name, []).append(folder_path)
    return sources


def discover_brand_export_sources() -> dict[str, list[Path]]:
    # รองรับโครงสร้างเดิมใน 06_brand_classification/01_brand_exports เช่น Toyota/Honda
    sources: dict[str, list[Path]] = {}
    if not EXPORTS_DIR.exists():
        return sources
    for class_dir in sorted(path for path in EXPORTS_DIR.iterdir() if path.is_dir()):
        sources.setdefault(clean_class_name(class_dir.name), []).append(class_dir)
    return sources


def discover_class_sources() -> dict[str, list[Path]]:
    """รวม source จาก Workspace, brand exports และ Unknown โดยตัด path ซ้ำออก."""
    # รวม class จากระบบเดิมและจาก metadata ใหม่ โดย class ชื่อเดียวกันจะถูกรวมข้อมูลเข้าด้วยกัน
    sources = discover_brand_export_sources()
    for class_name, folders in discover_workspace_classification_sources().items():
        sources.setdefault(class_name, []).extend(folders)
    return sources


def collect_images(source_dir: Path) -> list[Path]:
    # ดึงรูปแบบ recursive เพื่อรองรับทั้งโฟลเดอร์รูปธรรมดาและ dataset export ที่มี subfolder
    if not source_dir.exists():
        return []
    return sorted(path for path in source_dir.rglob("*") if is_image(path))


def split_images(images: list[Path], seed: int) -> dict[str, list[Path]]:
    # แบ่ง train/val/test ของแต่ละแบรนด์ด้วย seed เดิม เพื่อให้ reproduce ได้
    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_class_images(class_name: str, images: list[Path], source_root: Path, seed: int) -> int:
    """แบ่งและคัดลอกรูปของ class ไป dataset ปลายทาง พร้อมตั้งชื่อกันไฟล์ชน."""
    # copy รูปเข้าโครงสร้าง classification dataset: train/Toyota, val/Honda, test/Unknown
    split_map = split_images(images, seed)
    copied = 0
    for split_name, split_images_list in split_map.items():
        output_dir = OUTPUT_DATASET_DIR / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(split_images_list, start=1):
            try:
                rel = image_path.relative_to(source_root)
            except ValueError:
                rel = image_path.name
            stem = clean_name(str(rel).replace("\\", "_").replace("/", "_"))
            destination = output_dir / f"{class_name}_{index:05d}_{stem}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination)
            copied += 1
    return copied


def prepare_dataset(seed: int, force: bool) -> dict[str, int]:
    """สร้าง classification dataset ใหม่หรือ reuse ของเดิม และคืนจำนวนรูปต่อ class."""
    # สร้าง dataset สำหรับ YOLO classification จาก Toyota/Honda และ Unknown background
    if OUTPUT_DATASET_DIR.exists():
        if not force:
            raise RuntimeError(
                "Output dataset already exists:\n"
                f"  {OUTPUT_DATASET_DIR}\n\n"
                "Use --force-rebuild to rebuild it, or train with the existing dataset."
            )
        shutil.rmtree(OUTPUT_DATASET_DIR)

    sources = discover_class_sources()
    if not sources:
        raise FileNotFoundError(
            "No Classification folders found. Set Dataset Type to Classification and fill Brand/Model, "
            f"or put class folders in:\n  {EXPORTS_DIR}"
        )
    sources["Unknown"] = [UNKNOWN_DIR]

    counts = {}
    for class_name, source_dirs in sources.items():
        class_copied = 0
        images: list[Path] = []
        for source_dir in source_dirs:
            images.extend(collect_images(source_dir))
        counts[class_name] = len(images)
        if class_name != "Unknown" and not images:
            raise FileNotFoundError(f"No images found for class {class_name}.")
        for source_dir in source_dirs:
            source_images = collect_images(source_dir)
            if source_images:
                class_copied += copy_class_images(class_name, source_images, source_dir, seed + class_copied)
        if class_name != "Unknown" and class_copied == 0:
            raise FileNotFoundError(f"No usable images copied for class {class_name}.")

    return counts


def choose_base_model() -> str:
    # ถ้ามี brand model เดิมจะใช้ fine-tune ต่อ ไม่งั้นเริ่มจาก yolov8n-cls.pt
    base_model = next((path for path in BASE_MODEL_CANDIDATES if path.exists()), None)
    return str(base_model) if base_model is not None else "yolov8n-cls.pt"


CLASSIFICATION_AUGMENTATION_LIMITS = {
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 20.0),
    "translate": (0.0, 0.25),
    "scale": (0.0, 0.5),
    "shear": (0.0, 10.0),
    "fliplr": (0.0, 1.0),
    "erasing": (0.0, 0.8),
}


def parse_classification_augmentation(raw_json: str) -> dict[str, float | None]:
    """Accept only bounded Ultralytics classification augmentation arguments."""
    try:
        incoming = json.loads(raw_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --augmentation-json: {exc}") from exc
    if not isinstance(incoming, dict):
        raise ValueError("--augmentation-json must contain a JSON object.")
    values: dict[str, float | None] = {}
    for key, (minimum, maximum) in CLASSIFICATION_AUGMENTATION_LIMITS.items():
        try:
            values[key] = max(minimum, min(maximum, float(incoming.get(key, 0.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid augmentation value for {key}.") from exc
    values["hsv_h"] = 0.0
    values["flipud"] = 0.0
    values["auto_augment"] = None
    return values


def train_classifier(
    epochs: int,
    imgsz: int,
    batch: int,
    augmentation: dict[str, float | None] | None = None,
) -> Path:
    """train YOLO classifier, ตรวจ best.pt แล้วคัดลอก weight/metadata สำหรับ inference."""
    # train classifier แล้ว copy best.pt ไปไว้ที่ 04_trained_models ให้ไฟล์กล้องโหลดได้
    model = YOLO(choose_base_model())
    run_name = f"classifier_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    augmentation_args = dict(augmentation or {})
    print(f"Augmentation settings: {json.dumps(augmentation_args, sort_keys=True)}")
    results = model.train(
        data=str(OUTPUT_DATASET_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(WORKSPACE_DIR / "03_output_dataset" / "runs" / "classify"),
        name=run_name,
        exist_ok=False,
        **augmentation_args,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    trained = YOLO(str(best_pt))
    names = trained.names if isinstance(trained.names, dict) else dict(enumerate(trained.names))
    if getattr(trained, "task", "") != "classify" or len(names) < 2:
        raise RuntimeError(f"Invalid classification model output: task={trained.task}, classes={names}")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model = TRAINED_MODELS_DIR / f"classifier_{version}_best.pt"
    shutil.copy2(best_pt, final_model)
    print(f"Training complete: {best_pt}")
    print(f"Copied trained model to: {final_model}")
    return final_model


def parse_args() -> argparse.Namespace:
    """อ่านแหล่งข้อมูล, split, base model และค่าฝึก classifier จาก command line."""
    # option หลักสำหรับ train brand classifier และ rebuild dataset เมื่อจำเป็น
    parser = argparse.ArgumentParser(description="Train Toyota/Honda Airbag brand classifier.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-profile",
        choices=("off", "recommended", "custom"),
        default="recommended",
    )
    parser.add_argument(
        "--augmentation-json",
        default="",
        help="Validated JSON values from the Training page. Omit to keep Ultralytics defaults.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    """ค้นหา class sources, สร้าง classification dataset แล้ว train และเขียน metadata."""
    # pipeline หลัก: เตรียม dataset แบรนด์ แล้ว train ถ้าไม่ได้สั่ง --prepare-only
    args = parse_args()
    counts = prepare_dataset(seed=args.seed, force=args.force_rebuild)
    print(f"Prepared classification dataset: {OUTPUT_DATASET_DIR}")
    for class_name in sorted(counts):
        print(f"{class_name}: source images={counts.get(class_name, 0)}")

    if args.prepare_only:
        print("Prepared dataset only. Training skipped.")
        return

    augmentation = parse_classification_augmentation(args.augmentation_json) if args.augmentation_json else None
    profile_label = args.augmentation_profile if augmentation is not None else "ultralytics-default"
    print(f"Augmentation profile: {profile_label}")
    train_classifier(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        augmentation=augmentation,
    )


if __name__ == "__main__":
    main()
