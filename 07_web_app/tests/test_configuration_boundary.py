"""Public settings must not bypass model activation or mutate running jobs."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


@pytest.mark.parametrize("payload", [
    {"normal_reference": {"active_model": "other.npz"}},
    {"active_locator_model": "other.pt"}, {"use_classifier": False}, {"camera_index": 1},
])
def test_training_locks_all_runtime_configuration(monkeypatch, payload):
    monkeypatch.setattr(app, "training_is_running", lambda: True)
    writer = Mock()
    monkeypatch.setattr(app, "write_project_config", writer)
    assert app.api_update_project_config(payload).status_code == 409
    writer.assert_not_called()


def test_live_camera_configuration_is_immutable(monkeypatch):
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", Mock(is_alive=lambda: True))
    writer = Mock()
    monkeypatch.setattr(app, "write_project_config", writer)
    assert app.api_update_project_config({"use_classifier": False}).status_code == 409
    writer.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"active_locator_model": "other.pt"}, {"active_detector_model": "other.pt"},
    {"active_brand_model": "other.pt"},
    {"normal_reference": {"active_model": "other.npz"}},
    {"normal_reference": {"quality_metrics": {"ng_recall": 1.}}},
    {"normal_reference": {"inspection_contract": {}}},
    {"last_model_validation": {"ok": True}},
])
def test_settings_cannot_replace_models_or_certification(monkeypatch, payload):
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", None)
    writer = Mock()
    monkeypatch.setattr(app, "write_project_config", writer)
    assert app.api_update_project_config(payload).status_code == 400
    writer.assert_not_called()


def test_reference_settings_preserve_active_artifact_and_evidence(monkeypatch):
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", None)
    existing = {"active_model": "original.npz", "quality_metrics": {"ng_recall": None},
                "inspection_size": 320}
    monkeypatch.setattr(app, "read_project_config", lambda: {"normal_reference": existing})
    writer = Mock(return_value={})
    monkeypatch.setattr(app, "write_project_config", writer)
    assert app.api_update_project_config({"normal_reference": {"inspection_size": 512}}).status_code == 200
    assert writer.call_args.args[0]["normal_reference"] == {**existing, "inspection_size": 512}


def test_existing_page_project_payload_is_supported(monkeypatch):
    monkeypatch.setattr(app, "training_is_running", lambda: False)
    monkeypatch.setattr(app, "WORKER_THREAD", None)
    writer = Mock(return_value={})
    monkeypatch.setattr(app, "write_project_config", writer)
    payload = {"project_name": "Inspection", "brand": "", "model": "1", "target_classes": ["OK", "NG"],
        "target_classes_text": "OK, NG", "primary_class": "OK", "detection_mode": "normal_reference",
        "inspection_strategy": "normal_reference", "class_aliases": {}, "class_display_names": {},
        "augmentation_profile": {}}
    assert app.api_update_project_config(payload).status_code == 200
    writer.assert_called_once_with(payload)
