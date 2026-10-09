"""Exercise the actual camera worker across missing frames and recovery."""
import itertools
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


@pytest.fixture(autouse=True)
def isolated_cycle_storage(monkeypatch):
    monkeypatch.setattr(app, "save_cycle", lambda root, closed, reference, source: closed[0] if closed else None)


@pytest.mark.parametrize("recover", [False, True])
def test_missing_frame_invalidates_result_and_recovery_reinspects(tmp_path, monkeypatch, recover):
    model = tmp_path / "reference.npz"
    model.write_bytes(b"test model")
    frame = np.zeros((40, 40, 3), np.uint8)
    event = threading.Event()
    monkeypatch.setattr(app, "STOP_EVENT", event)
    monkeypatch.setattr(app, "STATE", {"status": "starting", "frame_count": 0, "raw_frame_count": 0})
    monkeypatch.setattr(app, "CURRENT_CAP", None)
    monkeypatch.setattr(app, "read_project_config", lambda: {"normal_reference": {"active_model": str(model)}})
    inspector = Mock(threshold=1.0)
    inspector.inspect.return_value = app.InspectionResult("OK", .9, .3, 1., [])
    inspector.draw_result.side_effect = lambda image, *args, **kwargs: image
    monkeypatch.setattr(app, "load_inspection_reference", lambda _: inspector)
    monkeypatch.setattr(app, "RecipeBoundInspector", lambda value, *_: value)
    monkeypatch.setattr(app, "get_shared_locator", lambda _: (None, 0))
    monkeypatch.setattr(app.detector, "analyze_primary_object", lambda *_: {
        "box": (2, 2, 38, 38), "label": "Part", "confidence": .9})
    monkeypatch.setattr(app.detector, "draw_fps", lambda *_: None)
    # Recovery happens within the AI throttle interval, so reusing the cached
    # annotated frame would skip inference without an explicit reset.
    ticks = itertools.count(1., .001)
    monkeypatch.setattr(app.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(app, "WEB_AI_MIN_INTERVAL_SECONDS", 1.)
    outputs, waits, before_reads = [], [], []
    monkeypatch.setattr(app, "set_latest_frame", lambda image: outputs.append(image))
    monkeypatch.setattr(app, "make_status_frame", lambda title, detail="": title)
    monkeypatch.setattr(app.time, "sleep", lambda _: waits.append(dict(app.STATE)))
    cap = Mock()
    steps = iter([(True, frame), (False, None), (False, None)] + ([(True, frame + 1)] if recover else []))
    def read():
        before_reads.append(dict(app.STATE))
        try:
            return next(steps)
        except StopIteration:
            event.set()
            return False, None
    cap.read.side_effect = read
    monkeypatch.setattr(app, "open_camera_like_detector", lambda _: (cap, 0))
    app.normal_reference_camera_worker(0, False)
    assert before_reads[1]["decision"] == "REVIEW"
    assert waits and all(item.get("decision") == "NOT_INSPECTED" for item in waits)
    assert all(not item.get("regions") and item.get("confidence") == 0 for item in waits)
    assert any(isinstance(item, str) and item == "Camera frame unavailable" for item in outputs)
    assert inspector.inspect.call_count == (2 if recover else 1)
    assert app.STATE["decision"] == "NOT_INSPECTED"
    cap.release.assert_called_once()


@pytest.mark.parametrize("status", ["starting", "stopping", "error", "waiting_frame"])
def test_non_inspecting_status_clears_previous_decision(monkeypatch, status):
    monkeypatch.setattr(app, "STATE", {"decision": "OK", "confidence": .99,
        "anomaly_score": .2, "regions": [{"x": 1}], "brand": "Toyota", "brand_confidence": 1.})
    app.set_status(status)
    assert app.STATE["decision"] == "NOT_INSPECTED"
    assert app.STATE["confidence"] == 0
    assert app.STATE["regions"] == []
    assert app.STATE["brand"] is None


@pytest.mark.parametrize("boundary", ["no_part", "moved", "input_issue", "latched_ng"])
def test_review_evidence_is_separated_at_inspection_boundary(tmp_path, monkeypatch, boundary):
    model = tmp_path / "model.npz"
    model.write_bytes(b"test")
    event = threading.Event()
    monkeypatch.setattr(app, "STOP_EVENT", event)
    monkeypatch.setattr(app, "STATE", {"status": "starting", "frame_count": 0, "raw_frame_count": 0})
    monkeypatch.setattr(app, "read_project_config", lambda: {"normal_reference": {"active_model": str(model)}})
    frame = np.zeros((100, 100, 3), np.uint8)
    counter = [-1]
    states = []
    def read():
        states.append(dict(app.STATE))
        counter[0] += 1
        if counter[0] == 8:
            event.set()
            return False, None
        return True, frame.copy()
    cap = Mock()
    cap.read.side_effect = read
    monkeypatch.setattr(app, "open_camera_like_detector", lambda _: (cap, 0))
    monkeypatch.setattr(app, "get_shared_locator", lambda _: (None, 0))
    def locate(*_):
        if boundary == "no_part" and counter[0] == 4:
            return None
        box = (60, 60, 90, 90) if boundary == "moved" and counter[0] >= 4 else (5, 5, 35, 35)
        return {"box": box, "label": "Part", "confidence": .9}
    monkeypatch.setattr(app.detector, "analyze_primary_object", locate)
    inspector = Mock(threshold=1.)
    def inspect(*_, **kwargs):
        if boundary == "latched_ng" and counter[0] >= 4:
            return app.InspectionResult("OK", .9, .2, 1., [])
        if boundary == "input_issue" and counter[0] == 4:
            return app.InspectionResult("REVIEW", 0., 0., 1., [], review_required=True, input_issue="alignment_failed")
        return app.InspectionResult("NG", .9, 2., 1., [])
    inspector.inspect.side_effect = inspect
    inspector.draw_result.side_effect = lambda image, *args, **kwargs: image
    monkeypatch.setattr(app, "load_inspection_reference", lambda _: inspector)
    monkeypatch.setattr(app, "RecipeBoundInspector", lambda value, *_: value)
    monkeypatch.setattr(app.detector, "draw_fps", lambda *_: None)
    monkeypatch.setattr(app, "set_latest_frame", lambda *_: None)
    monkeypatch.setattr(app, "make_status_frame", lambda *_: frame)
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)
    ticks = itertools.count(1., .01)
    monkeypatch.setattr(app.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(app, "WEB_AI_MIN_INTERVAL_SECONDS", 0.)
    monkeypatch.setattr(app, "NG_CAPTURE_INTERVAL_SECONDS", 0.)
    saved = []
    def save(*_, **kwargs):
        old = kwargs["existing_review_id"]
        saved.append((counter[0], old))
        return {"id": old or f"review-{counter[0]}"}
    monkeypatch.setattr(app, "save_inspection_review", save)
    app.normal_reference_camera_worker(0, False)
    assert saved[0] == (2, "")
    assert saved[1] == (3, "review-2")
    if boundary == "latched_ng":
        assert states[-1]["frame_decision"] == "OK"
        assert states[-1]["decision"] == "NG"
    else:
        next_event = next(item for item in saved if item[0] >= 4)
        assert next_event[1] == "", saved
    cap.release.assert_called_once()
