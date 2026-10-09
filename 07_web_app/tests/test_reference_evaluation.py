from pathlib import Path
from unittest.mock import Mock,patch
import sys
import numpy as np
import cv2
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app

def result(status="OK",review=False,issue=None):
    return app.InspectionResult(status,.6,1.,1.2,[],review_required=review,input_issue=issue)

def test_review_is_not_false_ok_or_accepted_good(tmp_path):
    image=tmp_path/"frame.png"
    cv2.imwrite(str(image),np.zeros((40,40,3),np.uint8))
    reference={"recipe_datasets":[{"id":"part-a","recipe":{"model":"front"}}]}
    record={"source_image":str(image),"object":{"box":[3,3,35,35]},"evaluation_scope":app.reference_review_scope(reference)}
    inspector=Mock()
    inspector.inspect.side_effect=[result("NG"),result(),result(review=True),result("REVIEW",True,"part_clipped"),result(review=True)]
    with patch.object(app,"inspection_review_records",side_effect=lambda kind:[record]*4 if kind=="confirmed_ng" else [record]):
        metrics=app.evaluate_reference_on_reviewed_cases(inspector, reference)
    assert metrics["ng_recall"]==.25
    assert metrics["false_ok_rate"]==.25
    assert metrics["confirmed_ng_review_images"]==1
    assert metrics["confirmed_ng_uninspectable_images"]==1
    assert metrics["reviewed_ok_acceptance_rate"]==0

def test_paired_holdout_counts_review_separately_and_verifies_input(tmp_path):
    image=tmp_path/"frame.png"
    cv2.imwrite(str(image),np.zeros((40,40,3),np.uint8))
    manifest={"independent_validation":True,"validation":[{"path":str(image),"sha256":app.file_sha256(image),"box":[3,3,35,35]}]}
    inspector=Mock();inspector.inspect.return_value=result(review=True)
    metrics=app.evaluate_reference_holdout(inspector,manifest)
    assert metrics["normal_acceptance_rate"]==0
    assert metrics["strict_ok_rate"]==0
    assert metrics["review_rate"]==1
    assert not app.compare_reference_candidate(metrics)["promote"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed during training"):
        app.evaluate_reference_holdout(inspector,manifest)

def test_different_holdouts_cannot_authorize_promotion():
    candidate={"independent_validation":True,"validation_images":20,"normal_acceptance_rate":1.,"strict_ok_rate":1.,"comparison_set_sha256":"set-a"}
    comparison=app.compare_reference_candidate(candidate,{"normal_acceptance_rate":.9,"comparison_set_sha256":"set-b"})
    assert not comparison["promote"]
    assert "same_comparison_images_and_roi" in comparison["failed_checks"]


def test_corpus_has_no_160_image_limit_and_preview_counts_exact_duplicates(tmp_path):
    dataset=tmp_path/"dataset";dataset.mkdir()
    for i in range(170):
        (dataset/f"image{i}.png").write_bytes(str(i).encode())
    (dataset/"duplicate.png").write_bytes(b"0")
    records=[{"id":"large","label":"Large","meta":{"dataset_type":"Known-good Reference","model":"front"},"review_status":"reviewed","needs_info":False}]
    with patch.object(app,"list_dataset_folder_records",return_value=records),patch.object(app,"dataset_folder_path",return_value=dataset):
        response=app.api_reference_corpus_preview({"dataset_ids":["large"]})
    import json
    data=json.loads(response.body)
    assert data["source_images"]==171
    assert data["unique_images"]==170
    assert data["duplicates_removed"]==1
    assert data["selection_valid"]

def test_preview_does_not_substitute_all_datasets_for_empty_selection():
    with patch.object(app,"list_dataset_folder_records",return_value=[]):
        response=app.api_reference_corpus_preview({"dataset_ids":[]})
    import json
    assert json.loads(response.body)["unique_images"]==0
