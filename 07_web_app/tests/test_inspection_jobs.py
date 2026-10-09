import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from inspection_jobs import make_job, checked_settings, job_path
from inspection_contract import RecipeBoundInspector
from deep_patch_reference import DeepPatchReference


@pytest.fixture
def saved(tmp_path):
    model, locator = tmp_path / "model.npz", tmp_path / "part.pt"
    model.write_bytes(b"reference")
    locator.write_bytes(b"locator")
    reference = {"active_model": str(model), "quality_metrics": {"ng_recall": None}}
    return tmp_path, make_job(tmp_path, "Plastic cover / front", reference, locator, backbone_path=locator)


def test_generic_job_needs_no_brand(saved):
    root, job = saved
    settings = checked_settings(root, job)
    assert not settings["use_classifier"]
    inspector = Mock(threshold=1.)
    wrapped = RecipeBoundInspector(inspector, job["files"]["model"]["path"], settings["normal_reference"])
    assert wrapped.inspect(None) is inspector.inspect.return_value
    assert job["production_ready"] is False


@pytest.mark.parametrize("role", ["model", "locator", "backbone"])
def test_changed_dependencies_block_job(saved, role):
    root, job = saved
    Path(job["files"][role]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checked_settings(root, job)


@pytest.mark.parametrize("job_id", ["../outside", "", "job_abc", "C:/file"])
def test_job_ids_cannot_escape_folder(tmp_path, job_id):
    with pytest.raises(ValueError):
        job_path(tmp_path, job_id)


def test_select_job_and_switch_back(saved, monkeypatch):
    root, first = saved
    second = make_job(root, "Metal plate", first["settings"]["normal_reference"],
                      first["files"]["locator"]["path"], backbone_path=first["files"]["backbone"]["path"])
    monkeypatch.setattr(app, "PROJECT_ROOT", root)
    monkeypatch.setattr(app, "JOB_DIR", root / "jobs")
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", None)
    monkeypatch.setattr(app, "load_inspection_reference", lambda _: SimpleNamespace(
        backbone_path=first["files"]["backbone"]["path"], threshold=1.))
    writer = Mock(side_effect=lambda settings: settings)
    monkeypatch.setattr(app, "write_project_config", writer)
    monkeypatch.setattr(app, "clear_shared_model_cache", lambda: None)
    for job in (first, second):
        app.write_json(job_path(app.JOB_DIR, job["id"]), job)
    for job in (first, second, first):
        response = app.api_use_job(job["id"])
        assert response.status_code == 200
        assert json.loads(response.body)["production_ready"] is False
        assert writer.call_args.args[0]["normal_reference"]["job_id"] == job["id"]


def test_train_and_camera_block_job_selection(monkeypatch):
    monkeypatch.setattr(app, "training_is_running", lambda: True)
    assert app.api_use_job("bad").status_code == 409
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", Mock(is_alive=lambda: True))
    assert app.api_use_job("bad").status_code == 409


def test_model_checks_never_auto_activate():
    checks = app.compare_reference_candidate({"independent_validation": True, "validation_images": 100,
        "strict_ok_rate": 1., "normal_acceptance_rate": 1., "confirmed_ng_images": 100,
        "ng_recall": 1., "reviewed_ok_images": 100, "reviewed_ok_acceptance_rate": 1.})
    assert checks["checks_passed"]
    assert not checks["promote"]


def test_no_box_when_all_regions_fail_area_check():
    features = np.zeros((32, 32, 1), np.float32)
    features[16, 16, 0] = 30.
    model = DeepPatchReference(np.zeros((1024, 1, 1)), 3., 2., 10, "unused",
                               input_size=320, grid_size=32, min_region_ratio=.02)
    model._encoder = SimpleNamespace(encode=lambda image, box: (features, box), last_alignment_warp=None)
    result = model.inspect(np.zeros((324, 324, 3), np.uint8), object_box=(2, 2, 322, 322))
    assert result.status == "NG"
    assert result.regions == []
    assert result.to_dict()["location"] == "not_found"


def test_training_saves_job_without_changing_active_model(tmp_path, monkeypatch):
    import threading
    locator = tmp_path / "part.pt"
    locator.write_bytes(b"locator")
    monkeypatch.setattr(app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(app, "NORMAL_REFERENCE_MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(app, "REFERENCE_TRAIN_STOP_EVENT", threading.Event())
    monkeypatch.setattr(app, "REFERENCE_TRAIN_STATE", {})
    monkeypatch.setattr(app, "read_project_config", lambda: {"active_locator_model": str(locator)})
    monkeypatch.setattr(app, "YOLO", lambda _: SimpleNamespace(names={0: "Part"}))
    def save(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new-model")
        return path
    model = SimpleNamespace(save=save, input_size=320, threshold=1., training_count=20,
                            training_manifest={"validation": [{}]})
    fit = Mock(return_value=(model, {}))
    monkeypatch.setattr(app.DeepPatchReference, "fit_with_evaluation", fit)
    monkeypatch.setattr(app, "evaluate_reference_on_reviewed_cases", lambda *args: {"confirmed_ng_images": 0})
    monkeypatch.setattr(app, "evaluate_reference_holdout", lambda *a: {
        "independent_validation": True, "validation_images": 5, "strict_ok_rate": 1., "normal_acceptance_rate": 1.})
    monkeypatch.setattr(app, "normal_reference_training_fingerprint", lambda *a, **kw: "test")
    writer = Mock()
    monkeypatch.setattr(app, "write_project_config", writer)
    app.run_normal_reference_training(["plastic"], [], {"inspection_size": 320,
        "min_good_images": 5, "min_region_ratio": .001, "inspection_area": [0., 0., .5, 1.],
        "recipe_datasets": [{"id": "plastic", "label": "Plastic cover", "recipe": {"model": "front"}}]}, "test", 20, 0)
    assert app.REFERENCE_TRAIN_STATE["status"] == "finished", app.REFERENCE_TRAIN_STATE
    result = app.REFERENCE_TRAIN_STATE["result"]
    assert not result["activated"]
    saved_job = json.loads(job_path(app.JOB_DIR, result["job_id"]).read_text(encoding="utf-8"))
    assert saved_job["name"] == "Plastic cover"
    assert fit.call_args.kwargs["inspection_area"] == [0., 0., .5, 1.]
    assert saved_job["settings"]["normal_reference"]["inspection_area"] == [0., 0., .5, 1.]
    assert Path(saved_job["files"]["model"]["path"]).name.startswith("Plastic_cover_")
    assert checked_settings(tmp_path, saved_job)["use_classifier"] is False
    writer.assert_not_called()


def test_image_test_does_not_activate_or_save_production_review(saved, monkeypatch):
    root, job = saved
    monkeypatch.setattr(app, "PROJECT_ROOT", root)
    monkeypatch.setattr(app, "JOB_DIR", root / "jobs")
    app.write_json(job_path(app.JOB_DIR, job["id"]), job)
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", None)
    model = Mock(backbone_path=job["files"]["backbone"]["path"], threshold=1.)
    model.inspect.return_value = app.InspectionResult("NG", .8, 2., 1., [])
    model.draw_result.side_effect = lambda image, *args, **kw: image
    monkeypatch.setattr(app, "load_inspection_reference", lambda _: model)
    monkeypatch.setattr(app.detector, "load_primary_object_locator", lambda settings: (None, 0))
    monkeypatch.setattr(app.detector, "analyze_primary_object", lambda *args: {"box": (2, 2, 30, 30)})
    write, review = Mock(), Mock()
    monkeypatch.setattr(app, "write_project_config", write)
    monkeypatch.setattr(app, "save_inspection_review", review)
    response = app.test_job_image(job["id"], np.zeros((32, 32, 3), np.uint8))
    assert response.status_code == 200
    assert json.loads(response.body)["decision"] == "NG"
    write.assert_not_called()
    review.assert_not_called()
