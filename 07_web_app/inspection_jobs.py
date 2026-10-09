"""Saved inspection jobs. Training creates a job; selecting it is a separate step."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import uuid

from inspection_contract import build_inspection_contract


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_path(folder, job_id):
    if not re.fullmatch(r"job_[0-9a-f]{32}", str(job_id)):
        raise ValueError("Invalid job ID")
    return Path(folder) / f"{job_id}.json"


def make_job(root, name, reference, locator_path, *, backbone_path, brand_path="", check_brand=False):
    """Capture model settings and file hashes, without changing the active job."""
    name = str(name).strip()
    if not name or len(name) > 120:
        raise ValueError("Job name must contain 1 to 120 characters")
    if not isinstance(check_brand, bool):
        raise ValueError("Brand check must be on or off")
    paths = {"model": reference.get("active_model"), "locator": locator_path, "backbone": backbone_path}
    if check_brand:
        paths["brand"] = brand_path
    files = {}
    root = Path(root).resolve()
    for role, raw in paths.items():
        path = Path(str(raw or "")).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Missing or invalid {role} file")
        files[role] = {"path": str(path), "sha256": file_hash(path)}
    job_id = "job_" + uuid.uuid4().hex
    saved = deepcopy(reference)
    saved.update({"job_id": job_id, "recipe_id": job_id, "recipe_name": name})
    saved["inspection_contract"] = build_inspection_contract(
        paths["model"], saved.get("recipe_datasets", []), check_brand=check_brand)
    if saved["inspection_contract"].get("error"):
        raise ValueError("Brand check needs one verified brand in the job data")
    return {"version": 1, "id": job_id, "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "use": "test", "production_ready": False, "files": files,
            "settings": {"normal_reference": saved, "active_locator_model": files["locator"]["path"],
                         "active_brand_model": files.get("brand", {}).get("path", ""),
                         "use_classifier": check_brand,
                         "inspection_strategy": "normal_reference", "detection_mode": "normal_reference"}}


def checked_settings(root, job):
    """Reject missing/replaced files before selecting any saved job."""
    root = Path(root).resolve()
    if job.get("version") != 1 or job.get("use") != "test":
        raise ValueError("Unsupported job version")
    job_path(root, job.get("id"))
    settings = job["settings"]
    files = job["files"]
    for role in ("model", "locator", "backbone"):
        if role not in files:
            raise ValueError(f"Job has no {role} file")
    for role, item in files.items():
        path = Path(item["path"]).resolve()
        if root not in path.parents or not path.is_file() or file_hash(path) != item["sha256"]:
            raise ValueError(f"The {role} file has changed or is missing")
    reference = settings["normal_reference"]
    if (Path(reference["active_model"]).resolve() != Path(files["model"]["path"])
            or Path(settings["active_locator_model"]).resolve() != Path(files["locator"]["path"])
            or reference.get("job_id") != job["id"]):
        raise ValueError("Job settings do not match its files")
    if settings.get("use_classifier") and (
            "brand" not in files or settings.get("active_brand_model") != files["brand"]["path"]):
        raise ValueError("Job brand model is missing")
    return deepcopy(settings)
