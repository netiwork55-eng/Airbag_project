import json
import sys
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


@pytest.fixture
def saved_preview(tmp_path, monkeypatch):
    image = tmp_path / 'frame.png'
    cv2.imwrite(str(image), np.zeros((40,40,3),np.uint8))
    model = tmp_path / 'model.npz'; model.write_bytes(b'model')
    manifest = {'validation':[{'path':str(image), 'sha256':app.file_sha256(image), 'box':[4,5,30,31]}]}
    app.write_json(model.with_suffix('.training.json'), manifest)
    monkeypatch.setattr(app, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(app, 'read_project_config', lambda: {})
    monkeypatch.setattr(app, 'normal_reference_runtime_status', lambda _: {'ready':True,'model':str(model)})
    monkeypatch.setattr(app, 'reviewed_known_good_corpus', Mock(side_effect=AssertionError('must not resplit current dataset')))
    inspector=Mock()
    inspector.inspect.return_value=app.InspectionResult('OK',.8,.4,1.,[])
    inspector.draw_result.side_effect=lambda image,*args,**kwargs:image
    monkeypatch.setattr(app,'load_inspection_reference',lambda _:inspector)
    return image,model,inspector


def test_preview_uses_saved_image_and_exact_roi(saved_preview):
    image,model,inspector=saved_preview
    response=app.api_normal_reference_validation_samples(limit=12)
    data=json.loads(response.body)
    assert data['validation_images']==1
    assert len(data['samples'])==1
    assert inspector.inspect.call_args.kwargs['object_box']==(4,5,30,31)
    assert data['evaluation_scope']=='saved_model_validation_replay'


@pytest.mark.parametrize('failure',['changed','missing_manifest'])
def test_preview_never_replaces_missing_or_changed_evidence(saved_preview,failure):
    image,model,inspector=saved_preview
    if failure=='changed': image.write_bytes(b'changed')
    else: model.with_suffix('.training.json').unlink()
    response=app.api_normal_reference_validation_samples(limit=12)
    assert response.status_code==409
    inspector.inspect.assert_not_called()
