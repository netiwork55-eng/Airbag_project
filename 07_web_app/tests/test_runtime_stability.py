"""Regression tests for worker ownership, evidence consistency and portable data."""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def test_registry_marks_actual_locator_instead_of_legacy_detector(tmp_path, monkeypatch):
    models = tmp_path / 'models'
    models.mkdir()
    locator = models / 'locator_best.pt'
    legacy = models / 'legacy_best.pt'
    locator.write_bytes(b'locator')
    legacy.write_bytes(b'legacy')
    config = {'active_locator_model': str(locator), 'active_detector_model': str(legacy)}
    monkeypatch.setattr(app, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(app, 'BRAND_CODE_DIR', tmp_path / 'brand')
    monkeypatch.setattr(app, 'read_project_config', lambda: config)
    records = app.model_registry()
    assert [r['name'] for r in records if r['active']] == ['locator_best.pt']
    config['active_locator_model'] = str(models / 'missing.pt')
    assert not any(r['active'] for r in app.model_registry())




def test_review_keeps_metadata_of_selected_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'INSPECTION_REVIEW_DIR', tmp_path)
    source = np.full((40, 40, 3), 80, np.uint8)
    first_result = app.InspectionResult('NG', .9, 2., 1., [])
    second_result = app.InspectionResult('NG', .6, 1.1, 1., [])
    with mock.patch.object(app, 'review_evidence_quality', side_effect=[.9, .1]):
        first = app.save_inspection_review(source, source, first_result,
            object_box=(2, 2, 30, 30), object_label='Part', object_confidence=.9)
        saved = (tmp_path / first['id'] / 'source.jpg').read_bytes()
        second = app.save_inspection_review(source + 20, source + 20, second_result,
            object_box=(10, 10, 38, 38), object_label='Part', object_confidence=.5,
            existing_review_id=first['id'])
    assert (tmp_path / first['id'] / 'source.jpg').read_bytes() == saved
    assert second['object'] == first['object']
    assert second['anomaly_score'] == first['anomaly_score']
    assert second['event_frame_count'] == 2


def test_cvat_upload_names_are_unique_and_ordered(tmp_path):
    images = []
    for name in ['z_event', 'a_event']:
        path = tmp_path / name / 'source.jpg'
        path.parent.mkdir()
        path.write_bytes(b'jpeg')
        images.append(path)
    body, _ = app.cvat_multipart(images)
    assert b'filename="000000_source.jpg"' in body
    assert b'filename="000001_source.jpg"' in body
    assert b'name="client_files[0]"' in body
    assert b'name="client_files[1]"' in body


def test_cvat_current_label_endpoint_is_paginated(monkeypatch):
    pages=[]
    def request(path):
        pages.append(path)
        return {'results':[{'id':len(pages),'name':'NG'}],'next':'more' if len(pages)==1 else None}
    monkeypatch.setattr(app,'cvat_request',request)
    labels=app.cvat_task_labels(42,{'labels':{'url':'unused','count':2}})
    assert len(labels)==2 and 'page=2' in pages[-1]


def test_reference_progress_stays_below_completion_until_commit(tmp_path, monkeypatch):
    locator = tmp_path / 'part.pt'
    locator.write_bytes(b'weight')
    monkeypatch.setattr(app, 'REFERENCE_TRAIN_STATE', dict(app.REFERENCE_TRAIN_STATE))
    monkeypatch.setattr(app, 'read_project_config', lambda: {'active_locator_model': str(locator)})
    monkeypatch.setattr(app, 'YOLO', lambda _: mock.Mock(names={0: 'Part'}))
    observed = []
    def fit(*args, **kwargs):
        for value in [0., .2, .8, .99]:
            kwargs['progress_callback'](value, 'extracting_multiscale_features')
            observed.append(app.REFERENCE_TRAIN_STATE['progress'])
        raise ValueError('end of isolated progress test')
    monkeypatch.setattr(app.DeepPatchReference, 'fit_with_evaluation', fit)
    app.run_normal_reference_training([], [], {'inspection_size': 320,
        'min_good_images': 5, 'min_region_ratio': .00035}, 'test', 0, 0)
    assert len(observed) == 4
    assert 1 <= observed[0] < 5
    assert observed == sorted(observed)
    assert observed[-1] < 100


def test_reference_stop_is_cooperative(monkeypatch):
    monkeypatch.setattr(app, 'REFERENCE_TRAIN_STATE', {
        **app.REFERENCE_TRAIN_STATE, 'running': True, 'status': 'training'})
    monkeypatch.setattr(app, 'REFERENCE_TRAIN_STOP_EVENT', threading.Event(), raising=False)
    response = app.api_stop_train()
    assert response.status_code == 200
    assert app.REFERENCE_TRAIN_STOP_EVENT.is_set()
    assert app.REFERENCE_TRAIN_STATE['status'] == 'stopping'
    assert app.REFERENCE_TRAIN_STATE['running'] is True


def test_migration_preserves_yaml_classes_and_extra_fields(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'tools/maintenance/repair_after_move.py'
    spec = importlib.util.spec_from_file_location('airbag_repair_test', script)
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    (tmp_path / 'train/images').mkdir(parents=True)
    (tmp_path / 'valid/images').mkdir(parents=True)
    path = tmp_path / 'data.yaml'
    original = {'path': 'C:/old/data', 'train': 'train/images', 'val': 'valid/images',
                'names': {0: 'Part', 1: 'Mark'}, 'nc': 2, 'custom': {'version': 3}}
    path.write_text(yaml.safe_dump(original), encoding='utf-8')
    repair.repair_yolo_data_yaml(tmp_path)
    repaired = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert repaired['names'] == original['names']
    assert repaired['custom'] == original['custom']
    assert Path(repaired['path']) == tmp_path.resolve()


def test_reference_cancel_exits_without_activating_model(tmp_path, monkeypatch):
    locator = tmp_path / 'part.pt'
    locator.write_bytes(b'weight')
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(app, 'REFERENCE_TRAIN_STOP_EVENT', stop)
    monkeypatch.setattr(app, 'REFERENCE_TRAIN_STATE', {**app.REFERENCE_TRAIN_STATE, 'running': True})
    monkeypatch.setattr(app, 'read_project_config', lambda: {'active_locator_model': str(locator)})
    monkeypatch.setattr(app, 'YOLO', lambda _: mock.Mock(names={0: 'Part'}))
    def fit(*args, **kwargs):
        kwargs['progress_callback'](.1, 'extracting_multiscale_features')
        raise AssertionError('Cancellation was ignored')
    monkeypatch.setattr(app.DeepPatchReference, 'fit_with_evaluation', fit)
    with mock.patch.object(app, 'write_project_config') as activate:
        app.run_normal_reference_training([], [], {'inspection_size': 320,
            'min_good_images': 5, 'min_region_ratio': .00035}, 'test', 0, 0)
    assert app.REFERENCE_TRAIN_STATE['status'] == 'stopped'
    assert app.REFERENCE_TRAIN_STATE['running'] is False
    activate.assert_not_called()


def test_reference_train_refuses_live_camera_before_reading_dataset(monkeypatch):
    monkeypatch.setattr(app.vision_service, 'busy', lambda: True)
    with mock.patch.object(app, 'reviewed_known_good_corpus') as corpus:
        response = app.api_train_normal_reference({})
    assert response.status_code == 409
    corpus.assert_not_called()


def test_cvat_ambiguous_upload_keeps_task_and_does_not_create_again(tmp_path, monkeypatch):
    review = tmp_path / 'ng_event'
    review.mkdir()
    source = review / 'source.jpg'
    source.write_bytes(b'jpeg')
    app.write_json(review / 'review.json', {
        'id': 'ng_event', 'status': 'confirmed_ng', 'source_image': str(source)})
    monkeypatch.setattr(app, 'INSPECTION_REVIEW_DIR', tmp_path)
    monkeypatch.setattr(app, 'cvat_connection_config', lambda: {
        'configured': True, 'base_url': 'http://cvat', 'project_id': None})
    monkeypatch.setattr(app, 'read_project_config', lambda: {})
    calls = []
    def request(path, **kwargs):
        calls.append(path)
        if path == '/api/tasks':
            return {'id': 42}
        raise RuntimeError('Network disconnected after upload POST')
    monkeypatch.setattr(app, 'cvat_request', request)
    with pytest.raises(RuntimeError, match='disconnected'):
        app.sync_confirmed_reviews_to_cvat()
    saved = app.read_json(review / 'review.json')['cvat_sync']
    assert saved['task_id'] == 42
    assert saved['status'] == 'needs_attention'
    app.sync_confirmed_reviews_to_cvat()
    assert calls.count('/api/tasks') == 1


def test_cvat_processing_retry_resumes_same_task(tmp_path, monkeypatch):
    review = tmp_path / 'ng_event'
    review.mkdir()
    source = review / 'source.jpg'
    source.write_bytes(b'jpeg')
    app.write_json(review / 'review.json', {
        'id': 'ng_event', 'status': 'confirmed_ng', 'source_image': str(source),
        'cvat_sync': {'status': 'processing', 'task_id': 42, 'frame_index': 0,
                      'request_id': 'job42', 'task_url': 'http://cvat/tasks/42'}})
    monkeypatch.setattr(app, 'INSPECTION_REVIEW_DIR', tmp_path)
    monkeypatch.setattr(app, 'cvat_connection_config', lambda: {
        'configured': True, 'base_url': 'http://cvat', 'project_id': None})
    with mock.patch.object(app, 'cvat_request') as remote, \
         mock.patch.object(app, 'wait_for_cvat_request') as wait, \
         mock.patch.object(app, 'add_cvat_draft_annotations', return_value=1):
        result = app.sync_confirmed_reviews_to_cvat()
    wait.assert_called_once_with('job42')
    remote.assert_not_called()
    assert result['task_id'] == 42
    assert app.read_json(review / 'review.json')['cvat_sync']['status'] == 'synced'


def test_reference_backbone_relocates_exact_file_after_move(tmp_path):
    import deep_patch_reference as deep
    models = tmp_path / 'models'
    references = models / 'normal_reference'
    references.mkdir(parents=True)
    backbone = models / 'part.pt'
    backbone.write_bytes(b'weight')
    reference = deep.DeepPatchReference(np.zeros((4, 1, 3)), 1., .5, 5,
        str(tmp_path / 'old_machine' / 'part.pt'), grid_size=2)
    saved = reference.save(references / 'reference.npz')
    loaded = deep.load_inspection_reference(saved)
    assert Path(loaded.backbone_path) == backbone


def test_uploaded_inference_does_not_run_on_event_loop(tmp_path, monkeypatch):
    import asyncio
    import io
    import cv2
    from starlette.datastructures import UploadFile
    model = tmp_path / 'reference.npz'
    model.write_bytes(b'reference')
    monkeypatch.setattr(app, 'read_project_config', lambda: {
        'normal_reference': {'active_model': str(model)}})
    event_loop_thread = threading.get_ident()
    def inspect(*args):
        assert threading.get_ident() != event_loop_thread
        return app.JSONResponse({'ok': True, 'decision': 'OK'})
    monkeypatch.setattr(app, 'inspect_uploaded_image', inspect)
    _, encoded = cv2.imencode('.jpg', np.zeros((20, 20, 3), np.uint8))
    file = UploadFile(filename='test.jpg', file=io.BytesIO(encoded.tobytes()))
    response = asyncio.run(app.api_inspect_against_normal(file))
    assert json.loads(response.body)['decision'] == 'OK'


def test_reference_fingerprint_changes_when_roi_or_backbone_changes(tmp_path):
    image = tmp_path / 'images' / 'part.jpg'
    label = tmp_path / 'labels' / 'part.txt'
    image.parent.mkdir()
    label.parent.mkdir()
    image.write_bytes(b'image')
    label.write_text('0 .5 .5 .8 .8', encoding='utf-8')
    backbone = tmp_path / 'part.pt'
    backbone.write_bytes(b'weight1')
    settings = {'inspection_size': 320, 'min_good_images': 5, 'min_region_ratio': .00035}
    def fingerprint():
        return app.normal_reference_training_fingerprint([image], settings, backbone_path=str(backbone))
    first = fingerprint()
    label.write_text('0 .4 .5 .6 .7', encoding='utf-8')
    second = fingerprint()
    backbone.write_bytes(b'weight2')
    assert first != second != fingerprint()


def test_runtime_status_uses_real_size_and_detects_missing_backbone(tmp_path, monkeypatch):
    import deep_patch_reference as deep
    backbone = tmp_path / 'part.pt'
    backbone.write_bytes(b'weight')
    reference = deep.DeepPatchReference(np.zeros((4, 1, 3)), 1., .5, 5,
        str(backbone), input_size=320, grid_size=2)
    path = reference.save(tmp_path / 'reference.npz')
    config = {'normal_reference': {'active_model': str(path), 'inspection_size': 512}}
    monkeypatch.setattr(app, 'ANALYSIS_CACHE', {})
    with mock.patch.object(app, 'load_inspection_reference', wraps=app.load_inspection_reference) as loader:
        first = app.normal_reference_runtime_status(config)
        second = app.normal_reference_runtime_status(config)
        assert loader.call_count == 1
    assert first['ready'] and second['inspection_size'] == 320
    backbone.unlink()
    assert app.normal_reference_runtime_status(config)['ready'] is False


def test_locator_cache_reloads_when_selected_locator_changes(tmp_path, monkeypatch):
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    monkeypatch.setattr(app, 'SHARED_LOCATOR', None)
    monkeypatch.setattr(app, 'SHARED_LOCATOR_KEY', '')
    models = [(object(), 0), (object(), 0)]
    with mock.patch.object(app.detector, 'load_primary_object_locator', side_effect=models) as loader:
        a = app.get_shared_locator({'active_locator_model': str(first)})
        cached = app.get_shared_locator({'active_locator_model': str(first)})
        b = app.get_shared_locator({'active_locator_model': str(second)})
    assert a is cached
    assert a is not b
    assert loader.call_count == 2


def test_configured_missing_locator_never_substitutes_another_model(tmp_path, monkeypatch):
    with mock.patch.object(app.detector, 'YOLO') as loader:
        model, class_id = app.detector.load_primary_object_locator({
            'active_locator_model': str(tmp_path / 'missing.pt')})
    assert model is None and class_id is None
    loader.assert_not_called()


def test_atomic_json_retries_windows_sharing_error(tmp_path, monkeypatch):
    target = tmp_path / "comparison.json"
    target.write_text('{"old": true}')
    original = app.os.replace
    calls = []
    def replace(src, dst):
        calls.append(1)
        if len(calls) < 3:
            error = PermissionError("temporarily locked")
            error.winerror = 5
            raise error
        return original(src, dst)
    monkeypatch.setattr(app.os, "replace", replace)
    monkeypatch.setattr(app.time, "sleep", lambda _: None)
    app.write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"new": True}
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_permanent_failure_keeps_original(tmp_path, monkeypatch):
    target = tmp_path / "comparison.json"
    target.write_text('{"old": true}')
    error = PermissionError("access denied")
    error.winerror = 5
    replace = mock.Mock(side_effect=error)
    monkeypatch.setattr(app.os, "replace", replace)
    monkeypatch.setattr(app.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        app.write_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert replace.call_count == 6
    assert not list(tmp_path.glob("*.tmp"))


def test_cvat_health_does_not_pass_for_settings_only(monkeypatch):
    monkeypatch.setattr(app, "CVAT_HEALTH_CACHE", {"key": None, "expires": 0, "result": {}})
    monkeypatch.setattr(app, "cvat_connection_config", lambda: {"configured": True, "base_url": "http://test", "token": "test"})
    request = mock.Mock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(app, "cvat_request", request)
    assert app.cvat_connection_health()["connected"] is False
    assert app.cvat_connection_health()["connection_status"] == "unavailable"
    assert request.call_count == 1


