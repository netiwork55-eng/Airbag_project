import sys
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def ref(name):
    return {"recipe_datasets": [{"id": name, "recipe": {"brand": "same", "model": "front"}}]}


def test_same_brand_different_data_and_unknown_records_are_excluded(tmp_path, monkeypatch):
    path = tmp_path / 'image.png'
    cv2.imwrite(str(path), np.zeros((40, 40, 3), np.uint8))
    record = {"source_image": str(path), "object": {"box": [3, 3, 35, 35]}}
    records = [{**record, "evaluation_scope": app.reference_review_scope(ref('a'))},
               {**record, "evaluation_scope": app.reference_review_scope(ref('b'))}, record]
    monkeypatch.setattr(app, 'inspection_review_records', lambda status: records if status == 'confirmed_ng' else [])
    inspector = Mock()
    inspector.inspect.return_value = app.InspectionResult('NG', .8, 2., 1., [])
    metrics = app.evaluate_reference_on_reviewed_cases(inspector, ref('a'))
    assert metrics['confirmed_ng_images'] == 1
    assert metrics['excluded_review_scope_images'] == 2
    assert inspector.inspect.call_count == 1
    unknown = app.evaluate_reference_on_reviewed_cases(inspector)
    assert unknown['ng_recall'] is None
    assert unknown['excluded_review_scope_images'] == 3


def test_scope_order_independent_but_recipe_changes_matter():
    a, b = ref('a')['recipe_datasets'][0], ref('b')['recipe_datasets'][0]
    assert app.reference_review_scope({'recipe_datasets': [a, b]}) == app.reference_review_scope({'recipe_datasets': [b, a]})
    changed = ref('a'); changed['recipe_datasets'][0]['recipe']['model'] = 'back'
    assert app.reference_review_scope(changed) != app.reference_review_scope(ref('a'))


def test_event_does_not_merge_across_work_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'INSPECTION_REVIEW_DIR', tmp_path)
    frame = np.zeros((40, 40, 3), np.uint8)
    result = app.InspectionResult('NG', .8, 2., 1., [])
    def save(reference, old=''):
        return app.save_inspection_review(frame, frame, result, object_box=(3,3,35,35),
            object_label='Part', object_confidence=.9, reference=reference, existing_review_id=old)
    first = save(ref('a'))
    same = save(ref('a'), first['id'])
    assert same['id'] == first['id']
    other = save(ref('b'), first['id'])
    assert other['id'] != first['id']
    changed = ref('a'); changed['inspection_contract'] = {'model_sha256': 'new-model'}
    assert save(changed, first['id'])['id'] != first['id']
