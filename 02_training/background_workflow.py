"""จัดการ Background/Hard-negative ของงาน Detection แบบไม่ปะปนกับ Dataset ที่มีกรอบ.

ผู้ใช้ Upload เพียงครั้งเดียว โมดูลนี้จะจัดกลุ่มภาพซ้ำ/คล้ายกันก่อนแบ่งเป็น
train, eval_valid และ eval_test แบบ deterministic เพื่อไม่ให้เฟรมใกล้กันรั่วข้ามชุด
ไฟล์แบบเดิมที่วางตรง 02_background_images จะยังถูกนับเป็น train เพื่อรองรับงานเก่า
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUND_SPLITS = ("train", "eval_valid", "eval_test")
BACKGROUND_MANIFEST = "background_split_manifest.json"
BACKGROUND_SEED = 42
MIN_EVAL_IMAGES = 5
DHASH_DISTANCE = 5


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path) -> int:
    """คืน difference hash 64 บิตสำหรับจับเฟรมที่หน้าตาใกล้กัน."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.tobytes())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def hash_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def background_images(root: Path, split: str) -> list[Path]:
    """อ่านภาพตาม split โดยถือโครงสร้างเก่าทั้งหมดเป็น train."""
    root = Path(root)
    explicit_root = root / split
    explicit = [path for path in explicit_root.rglob("*") if is_image(path)] if explicit_root.exists() else []
    if split != "train" or not root.exists():
        return sorted(explicit)

    reserved = {name.lower() for name in BACKGROUND_SPLITS}
    legacy = []
    for path in root.rglob("*"):
        if not is_image(path):
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0].lower() in reserved:
            continue
        legacy.append(path)
    return sorted({*legacy, *explicit})


def background_summary(root: Path) -> dict:
    counts = {split: len(background_images(root, split)) for split in BACKGROUND_SPLITS}
    return {
        "counts": counts,
        "train": counts["train"],
        "eval_valid": counts["eval_valid"],
        "eval_test": counts["eval_test"],
        "total": sum(counts.values()),
        "ready": counts["train"] > 0 and counts["eval_valid"] >= MIN_EVAL_IMAGES and counts["eval_test"] >= MIN_EVAL_IMAGES,
        "minimum_eval_images": MIN_EVAL_IMAGES,
        "manifest": str(Path(root) / BACKGROUND_MANIFEST),
    }


def background_integrity(root: Path) -> dict:
    """ตรวจภาพซ้ำและภาพใกล้เคียงข้าม split ก่อนอนุญาตให้ Train."""
    records = []
    for split in BACKGROUND_SPLITS:
        for path in background_images(root, split):
            records.append({"split": split, "path": str(path), "sha256": image_digest(path), "dhash": image_dhash(path)})
    exact_pairs = 0
    near_pairs = 0
    examples = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            first, second = records[left], records[right]
            if first["split"] == second["split"]:
                continue
            exact = first["sha256"] == second["sha256"]
            near = not exact and hash_distance(first["dhash"], second["dhash"]) <= DHASH_DISTANCE
            if exact:
                exact_pairs += 1
            elif near:
                near_pairs += 1
            if (exact or near) and len(examples) < 10:
                examples.append({"left": first["path"], "right": second["path"], "kind": "exact" if exact else "near"})
    return {"ok": exact_pairs == 0 and near_pairs == 0, "exact_pairs": exact_pairs, "near_pairs": near_pairs, "examples": examples}


def _safe_parts(relative: Path) -> tuple[str, ...]:
    parts = []
    for part in relative.parts:
        clean = "".join(character if character.isalnum() or character in "._-" else "_" for character in part)
        parts.append(clean or "image")
    return tuple(parts)


def _group_similar_images(images: list[Path]) -> list[list[dict]]:
    records = []
    for path in images:
        records.append({"path": path, "sha256": image_digest(path), "dhash": image_dhash(path)})

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left]["sha256"] == records[right]["sha256"] or hash_distance(records[left]["dhash"], records[right]["dhash"]) <= DHASH_DISTANCE:
                union(left, right)

    grouped: dict[int, list[dict]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def _target_counts(total: int) -> dict[str, int]:
    if total < MIN_EVAL_IMAGES * 2 + 1:
        return {"train": total, "eval_valid": 0, "eval_test": 0}
    valid = max(MIN_EVAL_IMAGES, round(total * 0.15))
    test = max(MIN_EVAL_IMAGES, round(total * 0.15))
    if valid + test >= total:
        valid = MIN_EVAL_IMAGES
        test = MIN_EVAL_IMAGES
    return {"train": total - valid - test, "eval_valid": valid, "eval_test": test}


def split_background_groups(images: list[Path], locked_hashes: dict[str, list[int]] | None = None) -> dict[str, list[dict]]:
    """แบ่งทั้งกลุ่มพร้อมกันเพื่อกันภาพซ้ำ/เฟรมใกล้กันข้าม split."""
    groups = _group_similar_images(images)
    targets = _target_counts(len(images))
    result = {split: [] for split in BACKGROUND_SPLITS}
    ordered = sorted(
        groups,
        key=lambda group: (
            -len(group),
            hashlib.sha256(f"{BACKGROUND_SEED}:{group[0]['sha256']}".encode("utf-8")).hexdigest(),
        ),
    )
    for group in ordered:
        locked_roles = {
            split
            for split, hashes in (locked_hashes or {}).items()
            if any(hash_distance(record["dhash"], known_hash) <= DHASH_DISTANCE for record in group for known_hash in hashes)
        }
        if locked_roles:
            # ถ้าคล้ายข้อมูลเดิม ต้องอยู่ split เดิม; กรณีชนหลายชุดให้เข้าฝั่ง Train แบบปลอดภัย
            split = "train" if "train" in locked_roles or len(locked_roles) > 1 else next(iter(locked_roles))
            result[split].extend(group)
            continue
        available = [split for split in BACKGROUND_SPLITS if targets[split] > 0]
        if not available:
            available = ["train"]
        split = max(
            available,
            key=lambda name: (
                (targets[name] - len(result[name])) / max(1, targets[name]),
                name == "train",
                name,
            ),
        )
        result[split].extend(group)
    return result


def _existing_hashes(root: Path) -> set[str]:
    return {image_digest(path) for split in BACKGROUND_SPLITS for path in background_images(root, split)}


def install_background_upload(staging_dir: Path, destination_root: Path, batch_name: str | None = None) -> dict:
    """ติดตั้ง Upload หนึ่งชุด พร้อมแบ่งและบันทึกหลักฐานการแบ่ง."""
    staging_dir, destination_root = Path(staging_dir), Path(destination_root)
    images = sorted(path for path in staging_dir.rglob("*") if is_image(path))
    if not images:
        raise ValueError("No background images were found in the upload.")

    existing = _existing_hashes(destination_root)
    seen_upload = set(existing)
    unique = []
    for path in images:
        digest = image_digest(path)
        if digest in seen_upload:
            continue
        seen_upload.add(digest)
        unique.append(path)
    skipped_duplicates = len(images) - len(unique)
    if not unique:
        return {"installed": 0, "installed_paths": [], "skipped_duplicates": skipped_duplicates, **background_summary(destination_root)}

    locked_hashes = {
        split: [image_dhash(path) for path in background_images(destination_root, split)]
        for split in BACKGROUND_SPLITS
    }
    split_map = split_background_groups(unique, locked_hashes)
    batch = batch_name or datetime.now().strftime("%Y%m%d_%H%M%S_background")
    entries = []
    for split, records in split_map.items():
        for record in records:
            relative = record["path"].relative_to(staging_dir)
            target = destination_root.joinpath(split, batch, *_safe_parts(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["path"], target)
            entries.append(
                {
                    "path": str(target.relative_to(destination_root)).replace("\\", "/"),
                    "split": split,
                    "sha256": record["sha256"],
                    "dhash": f"{record['dhash']:016x}",
                    "batch": batch,
                }
            )

    manifest_path = destination_root / BACKGROUND_MANIFEST
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    payload = {
        "version": 1,
        "seed": BACKGROUND_SEED,
        "near_duplicate_hamming_distance": DHASH_DISTANCE,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "entries": [*(previous.get("entries") or []), *entries],
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    installed_counts = {split: sum(1 for entry in entries if entry["split"] == split) for split in BACKGROUND_SPLITS}
    return {
        "installed": len(entries),
        "installed_counts": installed_counts,
        "installed_paths": [str(destination_root / split / batch) for split in BACKGROUND_SPLITS if installed_counts[split]],
        "skipped_duplicates": skipped_duplicates,
        "batch": batch,
        **background_summary(destination_root),
    }
