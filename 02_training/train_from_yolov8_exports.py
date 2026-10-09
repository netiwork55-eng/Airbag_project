r"""
รวม YOLOv8 export หลายชุดแล้ว train เป็น detector เดียว

ไฟล์นี้เป็น training workflow หลักของหน้าเว็บเมื่อผู้ใช้มี label ที่ตรวจแล้วจาก
Roboflow, LabelImg, CVAT หรือ Annotation Studio ระบบจะค้นหา data.yaml แบบ recursive,
อ่าน class ของแต่ละ export, map class เข้าสู่ target contract, ทำ label ให้สะอาด,
รวม train/valid/test และเพิ่ม background ที่มี label ว่าง

โครงสร้างข้อมูลที่รองรับ:
  05_train_new_object/01_yolov8_exports/
      DatasetA/
          data.yaml
          train/images, train/labels
          valid/images, valid/labels
          test/images, test/labels
      DatasetB/
          data.yaml
          train/images, train/labels
          ...

negative/background ที่เลือกใส่เพิ่มได้:
  05_train_new_object/02_background_images/
      images that do NOT contain the target object

ข้อควรระวัง:
    - ห้ามแก้ class mapping โดยดูเพียง export แรก เพราะแต่ละชุดอาจเรียง class ID ต่างกัน
    - ค่า coordinate ที่เกิน 0..1 จะถูก clamp และรายงาน warning ก่อน merge
    - resume ได้เฉพาะ combined dataset/source manifest ชุดเดิม
    - validate_trained_model ป้องกันโมเดล COCO หรือโมเดล class ผิดมาแทนโมเดลโปรเจกต์

รัน:
  ..\venv\Scripts\python.exe 02_training\train_from_yolov8_exports.py --class-name Airbag --epochs 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import yaml

from background_workflow import background_images, background_summary


# path หลักของ workflow ที่รวม export YOLOv8 หลายชุดแล้ว train เป็นโมเดลเดียว
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "05_train_new_object"
DEFAULT_EXPORTS_DIR = WORKSPACE_DIR / "01_yolov8_exports"
DEFAULT_BACKGROUND_DIR = WORKSPACE_DIR / "02_background_images"
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / "03_output"
MODEL_DIR = PROJECT_ROOT / "models"
BASE_MODEL_CANDIDATES = [
    MODEL_DIR / "yolov8n.pt",
    PROJECT_ROOT.parent / "yolov8n.pt",
    Path.cwd() / "yolov8n.pt",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "test"]
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}
REBALANCE_SPLIT_RATIOS = {"train": 0.86, "valid": 0.08, "test": 0.06}
NEAR_DUPLICATE_DHASH_DISTANCE = 4
FULL_FRAME_BOX_THRESHOLD = 0.98

DETECTION_AUGMENTATION_LIMITS = {
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


def parse_detection_augmentation(raw_json: str) -> dict[str, float]:
    """Accept only bounded Ultralytics detection augmentation arguments."""
    try:
        incoming = json.loads(raw_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --augmentation-json: {exc}") from exc
    if not isinstance(incoming, dict):
        raise ValueError("--augmentation-json must contain a JSON object.")
    values: dict[str, float] = {}
    for key, (minimum, maximum) in DETECTION_AUGMENTATION_LIMITS.items():
        try:
            values[key] = max(minimum, min(maximum, float(incoming.get(key, 0.0))))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid augmentation value for {key}.") from exc
    values["hsv_h"] = 0.0
    values["flipud"] = 0.0
    values["mixup"] = 0.0
    values["perspective"] = 0.0
    values["close_mosaic"] = 5 if values["mosaic"] > 0 else 0
    return values


def load_yolo_class():
    """โหลด Ultralytics เฉพาะตอนต้องใช้จริง เพื่อให้ --help และงานตรวจระบบไม่ช้า."""
    from ultralytics import YOLO

    return YOLO


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def clean_name(name: str) -> str:
    # ทำชื่อ dataset/class ให้ใช้เป็นชื่อโฟลเดอร์ได้ปลอดภัย
    keep = []
    for char in name.strip():
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    return "".join(keep).strip("_") or "custom_object"


def dataset_source_manifest(
    exports_dir: Path,
    background_dir: Path,
    target_names: list[str],
) -> dict:
    """Create a stable fingerprint of every source file used to build a dataset."""
    records = []
    roots = (("exports", exports_dir), ("background", background_dir))
    for group, root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if group == "background" and not is_image(path):
                continue
            if group == "exports" and path.suffix.lower() not in IMAGE_EXTENSIONS | {".txt", ".yaml", ".yml"}:
                continue
            stat = path.stat()
            records.append(
                {
                    "group": group,
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    encoded = json.dumps(
        {"target_names": target_names, "files": records},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "version": 1,
        "target_names": target_names,
        "file_count": len(records),
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "files": records,
    }


def source_manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "source_manifest.json"


def validate_resume_sources(
    dataset_dir: Path,
    exports_dir: Path,
    background_dir: Path,
    target_names: list[str],
) -> None:
    manifest_path = source_manifest_path(dataset_dir)
    if not manifest_path.exists():
        raise RuntimeError(
            "Cannot safely resume because source_manifest.json is missing. "
            "Start a new training run so the combined dataset is rebuilt."
        )
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = dataset_source_manifest(exports_dir, background_dir, target_names)
    if saved.get("fingerprint") != current["fingerprint"]:
        raise RuntimeError(
            "Cannot resume because the source dataset changed after this checkpoint was created. "
            "Turn Resume off and start a new training run so the new images and labels are included."
        )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def xyxy_to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float) -> str | None:
    # แปลงกรอบ xyxy normalized เป็น label YOLO 5 ค่า และตัดกรอบที่เล็ก/ผิดปกติทิ้ง
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
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def discover_yolov8_exports(exports_dir: Path) -> list[Path]:
    # หา export ที่มี data.yaml แบบ recursive เพื่อรองรับ Folder Explorer ที่จัด dataset เป็น workspace ย่อย
    if not exports_dir.exists():
        return []
    return sorted(
        dataset_dir
        for dataset_dir in {path.parent for path in exports_dir.rglob("data.yaml") if path.is_file()}
        if dataset_type_for_export(dataset_dir, exports_dir) == "Detection"
    )


def dataset_type_for_export(dataset_dir: Path, exports_dir: Path) -> str:
    # อ่าน dataset_meta.json ของ workspace ชั้นบนสุด ถ้า user ตั้งประเภทไว้ ให้ใช้ค่านั้นเป็นตัวคัดกรอง train
    try:
        relative = dataset_dir.resolve().relative_to(exports_dir.resolve())
    except ValueError:
        return "Detection"
    top_folder = exports_dir / relative.parts[0] if relative.parts else dataset_dir
    meta_path = top_folder / "dataset_meta.json"
    if not meta_path.exists():
        return "Detection"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return "Detection"
    dataset_type = str(meta.get("dataset_type") or "").strip()
    return dataset_type or "Detection"


def read_class_names(dataset_dir: Path) -> list[str]:
    # อ่านรายชื่อ class จาก data.yaml ของแต่ละ export
    data = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Unsupported names format in {dataset_dir / 'data.yaml'}")


BASE_CLASS_ALIASES = {
    "airbagv2": "airbag",
    "airbag_v2": "airbag",
    "airbag-v2": "airbag",
}


def raw_class_key(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_alias_lookup(value: object) -> dict[str, str]:
    lookup = {
        raw_class_key(alias): raw_class_key(target)
        for alias, target in BASE_CLASS_ALIASES.items()
    }
    if not isinstance(value, dict):
        return lookup
    for target, aliases in value.items():
        target_key = raw_class_key(target)
        if not target_key:
            continue
        items = aliases if isinstance(aliases, (list, tuple)) else str(aliases or "").replace("\n", ",").split(",")
        for alias in items:
            alias_key = raw_class_key(alias)
            if alias_key:
                lookup[alias_key] = target_key
    return lookup


def class_key(name: str, alias_lookup: dict[str, str] | None = None) -> str:
    key = raw_class_key(name)
    return (alias_lookup or normalize_alias_lookup({})).get(key, key)


def parse_class_names_text(class_names: str | None) -> list[str]:
    if not class_names:
        return []
    names = []
    seen = set()
    for raw_name in class_names.replace("\n", ",").split(","):
        name = raw_name.strip()
        if not name:
            continue
        key = class_key(name)
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def source_split_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path]:
    """Return image/label dirs for common YOLO export layouts."""
    aliases = SPLIT_ALIASES.get(split, (split,))
    candidates: list[tuple[Path, Path]] = []
    for alias in aliases:
        candidates.append((dataset_dir / alias / "images", dataset_dir / alias / "labels"))
        candidates.append((dataset_dir / "images" / alias, dataset_dir / "labels" / alias))

    for images_dir, labels_dir in candidates:
        if images_dir.exists() or labels_dir.exists():
            return images_dir, labels_dir

    alias = aliases[0]
    return dataset_dir / alias / "images", dataset_dir / alias / "labels"


def choose_target_names(
    datasets: list[Path],
    class_name: str | None,
    class_names: str | None = None,
    alias_lookup: dict[str, str] | None = None,
) -> list[str]:
    # ถ้าผู้ใช้ระบุ class list ให้ใช้ตามนั้น เพื่อรองรับ Airbag + defect classes.
    explicit_names = parse_class_names_text(class_names)
    if explicit_names:
        return explicit_names

    # ถ้าผู้ใช้ระบุ class-name เดี่ยว ให้คง behavior เดิมสำหรับงาน one-class.
    if class_name:
        return [class_name]

    # ถ้าไม่ระบุ ให้รวม class จากทุก export แทนการยึด export แรกอย่างเดียว.
    target_names = []
    seen = set()
    for dataset in datasets:
        for name in read_class_names(dataset):
            key = class_key(name, alias_lookup)
            if key in seen:
                continue
            seen.add(key)
            target_names.append(name)
    if not target_names:
        raise ValueError("No class names found in the YOLOv8 exports.")
    return target_names


def sanitize_label_text(
    label_text: str,
    source_names: list[str],
    target_names: list[str],
    alias_lookup: dict[str, str] | None = None,
) -> str:
    # ทำ label ให้สะอาด: map class ให้ตรง target และแปลง polygon เป็น bbox ถ้าจำเป็น
    target_lookup = {}
    for index, name in enumerate(target_names):
        target_lookup[str(name).strip().lower()] = index
        target_lookup[class_key(name, alias_lookup)] = index
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

        source_class_id = int(values[0])
        if source_class_id < 0 or source_class_id >= len(source_names):
            continue

        source_name = source_names[source_class_id]
        if len(target_names) == 1 and len(source_names) == 1:
            target_class_id = 0
        else:
            target_class_id = target_lookup.get(str(source_name).strip().lower())
            if target_class_id is None:
                target_class_id = target_lookup.get(class_key(source_name, alias_lookup))
            if target_class_id is None:
                continue

        if len(values) == 5:
            x_center, y_center, width, height = values[1:5]
            if width <= 0 or height <= 0:
                continue
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            clean_line = xyxy_to_yolo_line(target_class_id, x1, y1, x2, y2)
        else:
            coords = values[1:]
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            if len(coords) < 4:
                continue
            xs = [clamp01(value) for value in coords[0::2]]
            ys = [clamp01(value) for value in coords[1::2]]
            clean_line = xyxy_to_yolo_line(
                target_class_id,
                min(xs),
                min(ys),
                max(xs),
                max(ys),
            )

        if clean_line is not None:
            clean_lines.append(clean_line)

    return "\n".join(clean_lines) + ("\n" if clean_lines else "")


def copy_split(
    dataset_dir: Path,
    output_dataset_dir: Path,
    split: str,
    prefix: str,
    source_names: list[str],
    target_names: list[str],
    alias_lookup: dict[str, str] | None = None,
    rejected_full_frame: list[dict] | None = None,
) -> tuple[int, int, int]:
    # copy รูปและ label ของ split เดียว พร้อมเติม prefix กันชื่อไฟล์ชนกันระหว่าง export
    source_images_dir, source_labels_dir = source_split_dirs(dataset_dir, split)
    if not source_images_dir.exists():
        return 0, 0, 0

    output_images_dir = output_dataset_dir / split / "images"
    output_labels_dir = output_dataset_dir / split / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    object_lines = 0
    skipped_full_frame_images = 0
    for image_path in sorted(source_images_dir.iterdir()):
        if not is_image(image_path):
            continue

        destination_name = f"{prefix}_{image_path.name}"
        destination_image = output_images_dir / destination_name
        destination_label = output_labels_dir / f"{Path(destination_name).stem}.txt"

        label_path = source_labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            label_text = sanitize_label_text(
                label_path.read_text(encoding="utf-8"),
                source_names,
                target_names,
                alias_lookup,
            )
        else:
            label_text = ""

        full_frame_boxes = full_frame_box_count_from_text(label_text)
        if full_frame_boxes:
            # ภาพจาก offline augmentation บางชุดถูก Crop/Zoom จนกรอบเดิมถูก clamp เต็มภาพ
            # ต้องคัดออกทั้งภาพ เพราะลบเฉพาะกรอบจะทำให้วัตถุจริงกลายเป็น negative label
            skipped_full_frame_images += 1
            if rejected_full_frame is not None:
                rejected_full_frame.append(
                    {
                        "dataset": str(dataset_dir),
                        "split": split,
                        "image": str(image_path),
                        "label": str(label_path) if label_path.exists() else "",
                        "full_frame_boxes": full_frame_boxes,
                    }
                )
            continue

        shutil.copy2(image_path, destination_image)
        destination_label.write_text(label_text, encoding="utf-8")
        copied_images += 1
        object_lines += len([line for line in label_text.splitlines() if line.strip()])

    return copied_images, object_lines, skipped_full_frame_images


def copy_background_images(background_dir: Path, output_dataset_dir: Path) -> int:
    # ใช้เฉพาะ Background split=train; eval_valid/eval_test สงวนไว้ประเมินหลัง Train
    # จึงไม่ผ่าน Ultralytics augmentation และไม่รั่วเข้า weight ของโมเดล
    if not background_dir.exists():
        return 0

    output_images_dir = output_dataset_dir / "train" / "images"
    output_labels_dir = output_dataset_dir / "train" / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in background_images(background_dir, "train"):
        relative_stem = str(image_path.relative_to(background_dir).with_suffix("")).replace("\\", "_").replace("/", "_")
        destination_name = f"background_{clean_name(relative_stem)}{image_path.suffix.lower()}"
        shutil.copy2(image_path, output_images_dir / destination_name)
        (output_labels_dir / f"{Path(destination_name).stem}.txt").write_text(
            "",
            encoding="utf-8",
        )
        copied += 1
    return copied


def label_class_ids(label_path: Path) -> set[int]:
    return set(label_class_counts(label_path))


def label_class_counts(label_path: Path) -> dict[int, int]:
    class_counts: dict[int, int] = {}
    if not label_path.exists():
        return class_counts
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    return class_counts


def full_frame_box_count_from_text(label_text: str) -> int:
    full_frame_count = 0
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _, x_center, y_center, width, height = [float(part) for part in parts]
        except ValueError:
            continue
        if width >= FULL_FRAME_BOX_THRESHOLD and height >= FULL_FRAME_BOX_THRESHOLD:
            full_frame_count += 1
    return full_frame_count


def label_full_frame_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return full_frame_box_count_from_text(label_path.read_text(encoding="utf-8", errors="ignore"))


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


def rebalance_combined_splits(dataset_dir: Path, target_names: list[str]) -> dict:
    """จัด train/valid/test ใหม่หลัง merge เพื่อกันภาพซ้ำ/คล้ายรั่วข้าม split.

    Export จากหลายแหล่งมักมีเฟรมใกล้กันกระจายอยู่คนละ split อยู่แล้ว ถ้าคัดลอกตาม
    split ต้นทางตรง ๆ metric จะดูดีเกินจริง ฟังก์ชันนี้จึงรวมรายการที่สร้างแล้วทั้งหมด,
    จับกลุ่ม near-duplicate ด้วย dhash และย้ายทั้งกลุ่มไปอยู่ split เดียวกัน พร้อมเลือก
    valid/test ให้มี class สำคัญครบเท่าที่ข้อมูลอนุญาต
    """
    records: list[dict] = []
    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for image_path in sorted(path for path in images_dir.iterdir() if is_image(path)):
            label_path = labels_dir / f"{image_path.stem}.txt"
            class_counts = label_class_counts(label_path)
            records.append(
                {
                    "image": image_path,
                    "label": label_path,
                    "image_name": image_path.name,
                    "label_name": f"{image_path.stem}.txt",
                    "classes": set(class_counts),
                    "class_counts": class_counts,
                    "full_frame_boxes": label_full_frame_count(label_path),
                    "hash": image_dhash(image_path),
                    "source_split": split,
                }
            )

    if not records:
        return {"enabled": True, "images": 0, "groups": 0, "splits": {}}

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    hashed = [(index, record["hash"]) for index, record in enumerate(records) if record["hash"] is not None]
    for left_pos, (left_index, left_hash) in enumerate(hashed):
        for right_index, right_hash in hashed[left_pos + 1 :]:
            if int((left_hash ^ right_hash).bit_count()) <= NEAR_DUPLICATE_DHASH_DISTANCE:
                union(left_index, right_index)

    groups_by_root: dict[int, list[dict]] = {}
    for index, record in enumerate(records):
        groups_by_root.setdefault(find(index), []).append(record)

    groups = []
    for members in groups_by_root.values():
        classes: set[int] = set()
        class_counts: dict[int, int] = {}
        full_frame_boxes = 0
        source_splits = set()
        for member in members:
            classes.update(member["classes"])
            for class_id, count in member["class_counts"].items():
                class_counts[class_id] = class_counts.get(class_id, 0) + count
            full_frame_boxes += int(member.get("full_frame_boxes") or 0)
            source_splits.add(member["source_split"])
        groups.append(
            {
                "members": members,
                "classes": classes,
                "class_counts": class_counts,
                "full_frame_boxes": full_frame_boxes,
                "size": len(members),
                "source_splits": source_splits,
            }
        )

    total_images = len(records)
    targets = {
        "valid": max(len(target_names), round(total_images * REBALANCE_SPLIT_RATIOS["valid"])),
        "test": max(len(target_names), round(total_images * REBALANCE_SPLIT_RATIOS["test"])),
    }
    assignments = {"train": [], "valid": [], "test": []}
    split_class_counts = {split: {index: 0 for index in range(len(target_names))} for split in SPLITS}
    split_sizes = {split: 0 for split in SPLITS}
    total_class_counts = {index: 0 for index in range(len(target_names))}
    for group in groups:
        for class_id, count in group["class_counts"].items():
            if class_id in total_class_counts:
                total_class_counts[class_id] += count
    split_class_targets = {
        split: {
            class_id: max(1, round(total * REBALANCE_SPLIT_RATIOS[split]))
            for class_id, total in total_class_counts.items()
            if total > 0
        }
        for split in ("valid", "test")
    }

    positive_groups = [group for group in groups if group["classes"]]
    background_groups = [group for group in groups if not group["classes"]]
    positive_groups.sort(key=lambda group: (len(group["classes"]), -group["size"]), reverse=True)
    background_groups.sort(key=lambda group: group["size"], reverse=True)

    def assign(group: dict, split: str) -> None:
        assignments[split].append(group)
        split_sizes[split] += group["size"]
        for class_id, count in group["class_counts"].items():
            if class_id in split_class_counts[split]:
                split_class_counts[split][class_id] += count

    def unassigned_positive() -> list[dict]:
        assigned_ids = {id(group) for split_groups in assignments.values() for group in split_groups}
        return [group for group in positive_groups if id(group) not in assigned_ids]

    # เลือกกลุ่มที่มี class ขาดอยู่ลง test/valid ก่อน เพื่อไม่ให้ test ไม่มี Mark อีก
    for split in ("test", "valid"):
        for class_id in range(len(target_names)):
            if split_class_counts[split][class_id] > 0:
                continue
            candidates = [group for group in unassigned_positive() if class_id in group["classes"]]
            if not candidates:
                continue
            candidates.sort(
                key=lambda group: (
                    group["size"],
                    group["full_frame_boxes"] / max(1, sum(group["class_counts"].values())),
                    -len(group["classes"]),
                )
            )
            assign(candidates[0], split)

    # เติมขนาด valid/test ให้ใกล้ ratio โดยยังให้กลุ่มซ้ำอยู่ split เดียวกันเสมอ
    for split in ("test", "valid"):
        while split_sizes[split] < targets[split]:
            candidates = unassigned_positive()
            if not candidates:
                break
            class_targets = split_class_targets[split]

            def candidate_score(group: dict) -> tuple[int, int, int, float, int]:
                size_overflow = max(0, split_sizes[split] + group["size"] - targets[split])
                overage = 0
                useful_gain = 0
                for class_id, target in class_targets.items():
                    current = split_class_counts[split].get(class_id, 0)
                    group_count = group["class_counts"].get(class_id, 0)
                    if group_count <= 0:
                        continue
                    useful_gain += min(group_count, max(0, target - current))
                    overage += max(0, current + group_count - target)
                full_frame_ratio = group["full_frame_boxes"] / max(1, sum(group["class_counts"].values()))
                return (size_overflow, overage, -useful_gain, full_frame_ratio, group["size"])

            candidates.sort(key=candidate_score)
            assign(candidates[0], split)

    for group in unassigned_positive():
        assign(group, "train")
    for group in background_groups:
        assign(group, "train")

    temp_dir = dataset_dir / "_rebalance_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    temp_images_dir = temp_dir / "images"
    temp_labels_dir = temp_dir / "labels"
    temp_images_dir.mkdir()
    temp_labels_dir.mkdir()
    for index, record in enumerate(records):
        image_path = record["image"]
        label_path = record["label"]
        temp_image = temp_images_dir / f"{index}_{record['image_name']}"
        temp_label = temp_labels_dir / f"{index}_{record['label_name']}"
        if image_path.exists():
            shutil.move(str(image_path), str(temp_image))
        if label_path.exists():
            shutil.move(str(label_path), str(temp_label))
        else:
            temp_label.write_text("", encoding="utf-8")
        record["image"] = temp_image
        record["label"] = temp_label

    for split in SPLITS:
        for kind in ("images", "labels"):
            source_dir = dataset_dir / split / kind
            if source_dir.exists():
                shutil.rmtree(source_dir)
            source_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for split, split_groups in assignments.items():
        output_images_dir = dataset_dir / split / "images"
        output_labels_dir = dataset_dir / split / "labels"
        for group in split_groups:
            for member in group["members"]:
                image_path = member["image"]
                label_path = member["label"]
                destination_image = output_images_dir / member["image_name"]
                destination_label = output_labels_dir / member["label_name"]
                if image_path.exists():
                    shutil.move(str(image_path), str(destination_image))
                if label_path.exists():
                    shutil.move(str(label_path), str(destination_label))
                else:
                    destination_label.write_text("", encoding="utf-8")
                moved += 1
    shutil.rmtree(temp_dir, ignore_errors=True)

    split_report = {}
    for split in SPLITS:
        split_report[split] = {
            "images": split_sizes[split],
            "groups": len(assignments[split]),
            "class_counts": {
                target_names[class_id]: count
                for class_id, count in split_class_counts[split].items()
            },
        }
    return {
        "enabled": True,
        "images": total_images,
        "moved": moved,
        "groups": len(groups),
        "near_duplicate_threshold": NEAR_DUPLICATE_DHASH_DISTANCE,
        "splits": split_report,
    }


def write_data_yaml(dataset_dir: Path, target_names: list[str]) -> Path:
    # เขียน data.yaml ของ dataset รวม ให้ YOLO รู้ path และ class ที่ต้อง train
    data_yaml = dataset_dir / "data.yaml"
    names_text = "[" + ", ".join(f'"{name}"' for name in target_names) + "]"
    data_yaml.write_text(
        f"""train: {dataset_dir / "train" / "images"}
val: {dataset_dir / "valid" / "images"}
test: {dataset_dir / "test" / "images"}

nc: {len(target_names)}
names: {names_text}
""",
        encoding="utf-8",
    )
    return data_yaml


def validate_dataset_labels(dataset_dir: Path) -> tuple[int, int, int]:
    # ตรวจ label หลังรวม dataset เพื่อจับบรรทัดเสียก่อนเริ่ม train
    label_files = list(dataset_dir.glob("*/*/*.txt"))
    empty_files = 0
    bad_lines = 0

    for label_file in label_files:
        text = label_file.read_text(encoding="utf-8").strip()
        if not text:
            empty_files += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            ok = len(parts) == 5
            if ok:
                try:
                    values = [float(part) for part in parts]
                except ValueError:
                    ok = False
            if ok:
                ok = (
                    values[0].is_integer()
                    and all(0.0 <= value <= 1.0 for value in values[1:])
                    and values[3] > 0
                    and values[4] > 0
                )
            if not ok:
                bad_lines += 1

    return len(label_files), empty_files, bad_lines


def dataset_label_relationship_stats(dataset_dir: Path) -> dict:
    """Return class co-occurrence stats for the combined YOLO dataset."""
    stats = {
        "label_files": 0,
        "empty_files": 0,
        "single_class_files": 0,
        "multi_class_files": 0,
        "class_file_counts": {},
    }
    class_file_counts: dict[int, int] = {}
    for label_file in dataset_dir.glob("*/*/*.txt"):
        stats["label_files"] += 1
        class_ids = set()
        text = label_file.read_text(encoding="utf-8").strip()
        if not text:
            stats["empty_files"] += 1
            continue
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
        for class_id in class_ids:
            class_file_counts[class_id] = class_file_counts.get(class_id, 0) + 1
    stats["class_file_counts"] = class_file_counts
    return stats


def build_combined_dataset(
    exports_dir: Path,
    background_dir: Path,
    output_root: Path,
    class_name: str | None,
    class_names: str | None = None,
    class_aliases: dict | None = None,
    force_rebuild: bool = False,
    include_folders: set[str] | None = None,
) -> tuple[Path, list[str]]:
    # รวมหลาย YOLOv8 export + background ให้เป็น dataset เดียวสำหรับ train
    if include_folders:
        allowed = {str(name).strip().lower() for name in include_folders if str(name).strip()}
        # The generic Part locator is intentionally trained from user-reviewed
        # Known-good workspaces. Those folders are not typed as "Detection",
        # so the normal discovery filter would incorrectly discard all of them.
        datasets = sorted({
            yaml_path.parent
            for yaml_path in exports_dir.rglob("data.yaml")
            if yaml_path.is_file()
            and yaml_path.parent.relative_to(exports_dir).parts
            and yaml_path.parent.relative_to(exports_dir).parts[0].lower() in allowed
        })
    else:
        datasets = discover_yolov8_exports(exports_dir)
    if not datasets:
        raise FileNotFoundError(
            "No YOLOv8 export folders found. Put exported dataset folders in:\n"
            f"  {exports_dir}\n\n"
            "Each export folder should contain data.yaml and train/images, train/labels."
        )

    alias_lookup = normalize_alias_lookup(class_aliases or {})
    target_names = choose_target_names(datasets, class_name, class_names, alias_lookup)
    dataset_name = clean_name("_".join(target_names))
    output_dataset_dir = output_root / f"{dataset_name}_yolov8_combined"
    if output_dataset_dir.exists():
        last_checkpoint = find_last_checkpoint(output_dataset_dir, target_names)
        if last_checkpoint is not None and not force_rebuild:
            raise RuntimeError(
                "A previous training checkpoint already exists:\n"
                f"  {last_checkpoint}\n\n"
                "Use --resume to continue training from this checkpoint.\n"
                "Use --force-rebuild only if you intentionally want to delete the "
                "combined dataset/run output and start over."
            )
        # Rebuild only generated dataset inputs. Keep previous runs/checkpoints
        # so model history and rollback remain available.
        for entry_name in (
            "train",
            "valid",
            "test",
            "data.yaml",
            "source_manifest.json",
            "full_frame_rejections.json",
            "split_rebalance_report.json",
        ):
            entry = output_dataset_dir / entry_name
            if entry.is_dir():
                shutil.rmtree(entry)
            elif entry.exists():
                entry.unlink()
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_objects = 0
    rejected_full_frame: list[dict] = []
    for index, dataset_dir in enumerate(datasets, start=1):
        source_names = read_class_names(dataset_dir)
        prefix = f"d{index}_{clean_name(dataset_dir.name)}"
        print(f"Merging YOLOv8 export: {dataset_dir}")
        print(f"  source classes: {source_names}")
        for split in SPLITS:
            copied_images, object_lines, skipped_full_frame_images = copy_split(
                dataset_dir,
                output_dataset_dir,
                split,
                prefix,
                source_names,
                target_names,
                alias_lookup,
                rejected_full_frame,
            )
            total_images += copied_images
            total_objects += object_lines
            if copied_images:
                print(f"  {split}: images={copied_images}, objects={object_lines}")
            if skipped_full_frame_images:
                print(f"  {split}: rejected full-frame augmentation images={skipped_full_frame_images}")

    (output_dataset_dir / "full_frame_rejections.json").write_text(
        json.dumps(
            {
                "threshold": FULL_FRAME_BOX_THRESHOLD,
                "rejected_images": len(rejected_full_frame),
                "reason": (
                    "Images containing boxes that cover nearly the entire frame are excluded as unsafe "
                    "offline Crop/Zoom augmentation artifacts."
                ),
                "items": rejected_full_frame,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    background_split = background_summary(background_dir)
    background_count = copy_background_images(background_dir, output_dataset_dir)
    rebalance_report = rebalance_combined_splits(output_dataset_dir, target_names)
    (output_dataset_dir / "split_rebalance_report.json").write_text(
        json.dumps(rebalance_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_data_yaml(output_dataset_dir, target_names)
    source_manifest_path(output_dataset_dir).write_text(
        json.dumps(
            dataset_source_manifest(exports_dir, background_dir, target_names),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    label_files, empty_files, bad_lines = validate_dataset_labels(output_dataset_dir)
    relationship_stats = dataset_label_relationship_stats(output_dataset_dir)
    print(f"Combined dataset: {output_dataset_dir}")
    print(f"Target classes: {target_names}")
    print(f"Copied labeled images: {total_images}")
    print(f"Copied object labels: {total_objects}")
    print(f"Background negative images: {background_count}")
    print(
        "Background evaluation images: "
        f"valid={background_split['eval_valid']}, test={background_split['eval_test']}"
    )
    print(f"Rebalanced splits: {json.dumps(rebalance_report['splits'], ensure_ascii=False)}")
    print(f"Label files: {label_files}, empty labels: {empty_files}, bad lines: {bad_lines}")
    print(f"Multi-class label files: {relationship_stats['multi_class_files']}")
    if len(target_names) > 1 and relationship_stats["multi_class_files"] == 0:
        print(
            "WARNING: No label file contains more than one target class. "
            "For product+defect detection, add images labeled with both the "
            "main object and defect classes in the same image before training."
        )
    if bad_lines:
        raise ValueError("Combined dataset still contains bad label lines.")

    return output_dataset_dir, target_names


def model_names_list(model_names) -> list[str]:
    if isinstance(model_names, dict):
        return [str(model_names[index]) for index in sorted(model_names)]
    return [str(name) for name in model_names]


def choose_base_model(target_names: list[str]) -> str:
    """Fine-tune only from a detector with the exact same class contract."""
    expected = [name.lower() for name in target_names]
    YOLO = load_yolo_class()
    project_models = sorted(
        MODEL_DIR.glob("*_best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in project_models:
        try:
            model = YOLO(str(path))
            if getattr(model, "task", "") != "detect":
                continue
            if [name.lower() for name in model_names_list(model.names)] == expected:
                print(f"Using compatible project model as training base: {path}")
                return str(path)
        except Exception as exc:
            print(f"Skipping unreadable base model {path}: {exc}")
    base_model = next((path for path in BASE_MODEL_CANDIDATES if path.exists()), None)
    fallback = str(base_model) if base_model is not None else "yolov8n.pt"
    print(f"No exact class-compatible project model found; using generic base: {fallback}")
    return fallback


def validate_trained_model(best_pt: Path, target_names: list[str]) -> None:
    # กันไม่ให้โมเดล class ผิด เช่น COCO model มา overwrite โมเดล Airbag
    YOLO = load_yolo_class()
    names = model_names_list(YOLO(str(best_pt)).names)
    expected = [name.lower() for name in target_names]
    actual = [name.lower() for name in names]
    if actual != expected:
        raise RuntimeError(
            "Refusing to copy a model whose classes do not match the target.\n"
            f"  model: {best_pt}\n"
            f"  expected classes: {target_names}\n"
            f"  actual classes: {names}\n\n"
            "This prevents a generic COCO model from overwriting the Airbag model."
        )


def combined_dataset_path(output_root: Path, target_names: list[str]) -> Path:
    return output_root / f"{clean_name('_'.join(target_names))}_yolov8_combined"


def find_existing_combined_dataset(
    output_root: Path,
    class_name: str | None,
    class_names: str | None = None,
) -> Path | None:
    target_names = parse_class_names_text(class_names) or ([class_name] if class_name else [])
    if target_names:
        candidate = combined_dataset_path(output_root, target_names)
        if (candidate / "data.yaml").exists():
            return candidate

    candidates = [
        path
        for path in sorted(output_root.glob("*_yolov8_combined"))
        if (path / "data.yaml").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


def find_last_checkpoint(dataset_dir: Path, target_names: list[str]) -> Path | None:
    # หา last.pt สำหรับ resume training ต่อจากรอบเดิม
    run_root = dataset_dir / "runs" / "detect"
    checkpoints = [
        path
        for path in run_root.glob("*/weights/last.pt")
        if path.is_file()
    ]
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


def train_model(
    dataset_dir: Path,
    target_names: list[str],
    epochs: int,
    imgsz: int,
    resume: bool = False,
    augmentation: dict[str, float] | None = None,
    base_model: str | Path | None = None,
    progress_callback: Callable[[float, int, int, str], None] | None = None,
) -> Path:
    # train หรือ resume YOLOv8 แล้ว copy best.pt เข้าโฟลเดอร์ models ของโปรเจกต์
    YOLO = load_yolo_class()
    run_prefix = f"{clean_name('_'.join(target_names))}_yolov8_train"
    run_name = run_prefix if resume else f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if resume:
        last_checkpoint = find_last_checkpoint(dataset_dir, target_names)
        if last_checkpoint is None:
            raise FileNotFoundError(
                "Cannot resume because last.pt was not found.\n"
                f"Searched under:\n  {dataset_dir / 'runs' / 'detect'}\n\n"
                "Run training normally first, or check that the combined dataset "
                "folder has not been deleted."
            )
        print(f"Resuming training from: {last_checkpoint}")
        model = YOLO(str(last_checkpoint))
        results = model.train(resume=True)
    else:
        selected_base = str(base_model) if base_model else choose_base_model(target_names)
        print(f"Training base model: {selected_base}")
        model = YOLO(selected_base)
        if progress_callback:
            last_emit = {"time": 0.0}

            def emit_progress(trainer, *, force: bool = False) -> None:
                now = time.monotonic()
                if not force and now - last_emit["time"] < 1.0:
                    return
                last_emit["time"] = now
                total_epochs = max(1, int(getattr(trainer, "epochs", epochs) or epochs))
                epoch_index = max(0, int(getattr(trainer, "epoch", 0) or 0))
                train_loader = getattr(trainer, "train_loader", None)
                batches = max(1, len(train_loader)) if train_loader is not None else 1
                batch_index = max(0, int(getattr(trainer, "batch_i", 0) or 0))
                fraction = min(1.0, (epoch_index + min(1.0, (batch_index + 1) / batches)) / total_epochs)
                progress_callback(fraction, epoch_index + 1, total_epochs, f"batch {min(batch_index + 1, batches)}/{batches}")

            model.add_callback("on_train_start", lambda trainer: emit_progress(trainer, force=True))
            model.add_callback("on_train_batch_end", lambda trainer: emit_progress(trainer))
            model.add_callback("on_train_epoch_end", lambda trainer: emit_progress(trainer, force=True))
        augmentation_args = dict(augmentation or {})
        print(f"Augmentation settings: {json.dumps(augmentation_args, sort_keys=True)}")
        results = model.train(
            data=str(dataset_dir / "data.yaml"),
            epochs=epochs,
            imgsz=imgsz,
            batch=-1,
            patience=12,
            project=str(dataset_dir / "runs" / "detect"),
            name=run_name,
            exist_ok=True,
            **augmentation_args,
        )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    validate_trained_model(best_pt, target_names)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model = MODEL_DIR / f"{clean_name('_'.join(target_names))}_{version}_best.pt"
    shutil.copy2(best_pt, final_model)
    print(f"Training complete: {best_pt}")
    print(f"Copied trained model to: {final_model}")
    return final_model


def parse_args() -> argparse.Namespace:
    """อ่าน source/class mapping/ค่าฝึกและโหมด prepare/resume จาก command line."""
    # command line สำหรับรวม export, train, prepare-only, resume และ force rebuild
    parser = argparse.ArgumentParser(
        description="Merge YOLOv8 export folders and train a detector.",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=DEFAULT_EXPORTS_DIR,
        help="Folder containing one or more YOLOv8 exported dataset folders.",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=DEFAULT_BACKGROUND_DIR,
        help="Optional folder of negative/background images.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--class-name",
        default=None,
        help=(
            "Single target class to train. Kept for one-class compatibility."
        ),
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help=(
            "Comma-separated detection classes to preserve/map, for example "
            "'Airbag, Mark, Scratch'. If omitted, class names from all YOLOv8 "
            "exports are preserved."
        ),
    )
    parser.add_argument(
        "--class-aliases-json",
        default="{}",
        help="JSON object mapping canonical class names to accepted dataset aliases.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the existing runs/detect/*/weights/last.pt checkpoint. "
            "This does not rebuild or delete the combined dataset."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Delete and rebuild the combined dataset even if a previous training "
            "checkpoint exists. Do not use this when you want to resume."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """ตรวจ source, สร้างหรือ reuse dataset รวม แล้ว train/resume ตาม contract."""
    # จุดเริ่ม workflow: เตรียม/หา dataset รวม แล้ว train หรือ resume ตาม option ที่สั่ง
    args = parse_args()
    if args.resume and args.prepare_only:
        raise ValueError("--resume cannot be used together with --prepare-only.")

    if args.resume:
        dataset_dir = find_existing_combined_dataset(args.output_root, args.class_name, args.class_names)
        if dataset_dir is None:
            raise FileNotFoundError(
                "Cannot resume because no existing combined dataset was found in:\n"
                f"  {args.output_root}\n\n"
                "Run training once without --resume first."
            )
        target_names = read_class_names(dataset_dir)
        validate_resume_sources(
            dataset_dir,
            args.exports_dir,
            args.background_dir,
            target_names,
        )
        print(f"Using existing combined dataset for resume: {dataset_dir}")
        print(f"Target classes: {target_names}")
    else:
        try:
            class_aliases = json.loads(args.class_aliases_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --class-aliases-json: {exc}") from exc
        dataset_dir, target_names = build_combined_dataset(
            exports_dir=args.exports_dir,
            background_dir=args.background_dir,
            output_root=args.output_root,
            class_name=args.class_name,
            class_names=args.class_names,
            class_aliases=class_aliases,
            force_rebuild=args.force_rebuild,
        )

    if args.prepare_only:
        print("Prepared dataset only. Training skipped.")
        return
    augmentation = (
        parse_detection_augmentation(args.augmentation_json)
        if not args.resume and args.augmentation_json
        else None
    )
    if args.resume:
        print("Augmentation profile: checkpoint (using the original run settings)")
    else:
        profile_label = args.augmentation_profile if augmentation is not None else "ultralytics-default"
        print(f"Augmentation profile: {profile_label}")
    train_model(
        dataset_dir,
        target_names,
        args.epochs,
        args.imgsz,
        resume=args.resume,
        augmentation=augmentation,
    )


if __name__ == "__main__":
    main()
