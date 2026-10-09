from __future__ import annotations

import json
import os
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_APP_DIR = PROJECT_ROOT / "07_web_app"
PROJECT_CONFIG_PATH = WEB_APP_DIR / "project_config.json"
COMBINED_OUTPUT_DIR = PROJECT_ROOT / "05_train_new_object" / "03_output"


def read_json(path: Path) -> dict[str, Any]:
    """อ่าน JSON แบบปลอดภัย ถ้าไฟล์เสียจะไม่เขียนทับทันที"""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = path.with_suffix(path.suffix + f".broken_{datetime.now():%Y%m%d_%H%M%S}")
        backup_path.write_text(path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return {}


def write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    """เขียนไฟล์เฉพาะเมื่อเนื้อหาเปลี่ยนจริง เพื่อลดการแก้ timestamp โดยไม่จำเป็น"""
    next_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    current_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if current_text == next_text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(next_text, encoding="utf-8")
    os.replace(temporary, path)
    return True


def newest_file(folder: Path, patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in folder.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def repair_path_by_filename(old_value: object, folder: Path, fallback_patterns: list[str]) -> str:
    """Relocate the same named artifact only; never replace it with a newer model."""
    old_name = Path(str(old_value or "")).name
    if old_name:
        candidate = folder / old_name
        if candidate.exists():
            return str(candidate.resolve())
    return str(old_value or "")


def repair_nested_model_path(payload: object, replacement_path: str) -> bool:
    """แก้ path ใน validation metadata โดยไม่ลบค่า metric เดิม"""
    if not isinstance(payload, dict) or not replacement_path:
        return False
    if payload.get("path") == replacement_path:
        return False
    payload["path"] = replacement_path
    return True


def repair_project_config() -> bool:
    config = read_json(PROJECT_CONFIG_PATH)
    if not config:
        print(f"Skip project config: not found or unreadable: {PROJECT_CONFIG_PATH}")
        return False

    models_dir = PROJECT_ROOT / "models"
    brand_model_dir = PROJECT_ROOT / "06_brand_classification" / "04_trained_models"
    train_logs_dir = WEB_APP_DIR / "logs" / "train_jobs"

    active_detector = repair_path_by_filename(
        config.get("active_detector_model"),
        models_dir,
        ["Airbag_Mark_*_best.pt", "Airbag_Mark_best.pt", "Airbag_best.pt", "*.pt"],
    )
    previous_detector = repair_path_by_filename(
        config.get("previous_detector_model"),
        models_dir,
        ["Airbag_Mark_*_best.pt", "Airbag_Mark_best.pt", "Airbag_best.pt", "*.pt"],
    )
    active_brand = repair_path_by_filename(
        config.get("active_brand_model"),
        brand_model_dir,
        ["airbag_brand_best.pt", "*.pt"],
    )
    last_train_log = repair_path_by_filename(config.get("last_train_log"), train_logs_dir, ["*.log"])

    changed = False
    replacements = {
        "active_detector_model": active_detector,
        "previous_detector_model": previous_detector,
        "active_brand_model": active_brand,
        "last_train_log": last_train_log,
    }
    # Reference and locator are one recipe. Relocate exact filenames only;
    # choosing a newer weight would invalidate the saved anomaly memory.
    locator = repair_path_by_filename(config.get("active_locator_model"), models_dir, [])
    if locator:
        replacements["active_locator_model"] = locator
    for key, value in replacements.items():
        if value and config.get(key) != value:
            config[key] = value
            changed = True

    validation = config.get("last_model_validation")
    if repair_nested_model_path(validation, active_detector):
        changed = True

    reference = config.get("normal_reference")
    if isinstance(reference, dict):
        for key, directory in (("active_model", models_dir / "normal_reference"),
                               ("backbone_model", models_dir)):
            relocated = repair_path_by_filename(reference.get(key), directory, [])
            if relocated and reference.get(key) != relocated:
                reference[key] = relocated
                changed = True

    if changed:
        config["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return write_json_if_changed(PROJECT_CONFIG_PATH, config) if changed else False


def repair_yolo_data_yaml(dataset_dir: Path) -> bool:
    """แก้ data.yaml ของ combined dataset ให้ชี้ path ของเครื่องปัจจุบัน"""
    data_yaml = dataset_dir / "data.yaml"
    train_images = dataset_dir / "train" / "images"
    valid_images = dataset_dir / "valid" / "images"
    test_images = dataset_dir / "test" / "images"
    if not data_yaml.exists() or not train_images.exists() or not valid_images.exists():
        return False

    old_text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    data = yaml.safe_load(old_text)
    if not isinstance(data, dict) or not data.get("names"):
        raise ValueError(f"Cannot repair dataset without its class contract: {data_yaml}")
    repaired = dict(data)
    repaired.update({"path": str(dataset_dir.resolve()),
                     "train": str(train_images.resolve()), "val": str(valid_images.resolve())})
    if test_images.exists():
        repaired["test"] = str(test_images.resolve())
    if repaired == data:
        return False
    new_text = yaml.safe_dump(repaired, allow_unicode=True, sort_keys=False)
    temporary = data_yaml.with_name(f".{data_yaml.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(new_text, encoding="utf-8")
    os.replace(temporary, data_yaml)
    return True


def repair_combined_datasets() -> int:
    if not COMBINED_OUTPUT_DIR.exists():
        return 0
    changed = 0
    for data_yaml in COMBINED_OUTPUT_DIR.rglob("data.yaml"):
        if repair_yolo_data_yaml(data_yaml.parent):
            changed += 1
    return changed


def main() -> int:
    print(f"Airbag Project repair root: {PROJECT_ROOT}")
    config_changed = repair_project_config()
    yaml_changed = repair_combined_datasets()
    print(f"project_config.json: {'updated' if config_changed else 'already ok'}")
    print(f"YOLO data.yaml files updated: {yaml_changed}")
    print("Repair complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
