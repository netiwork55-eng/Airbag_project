from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


WEB_APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB_APP))
from normal_reference import DecisionStabilizer, InspectionResult, NormalReferenceInspector  # noqa: E402
import deep_patch_reference  # noqa: E402


def _write(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    assert ok
    data.tofile(str(path))


def test_deep_patch_invalid_roi_falls_back_to_full_image():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    crop, box = deep_patch_reference._crop(image, (500, 500, 510, 510))
    assert crop.shape == image.shape
    assert box == (0, 0, 60, 40)


def test_known_good_is_ok_and_local_change_is_ng(tmp_path):
    paths = []
    rng = np.random.default_rng(7)
    for index in range(8):
        image = np.full((220, 260, 3), (72, 96, 118), np.uint8)
        cv2.rectangle(image, (45, 55), (215, 175), (125, 145, 165), -1)
        noise = rng.normal(0, 1.2, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = tmp_path / f"good_{index}.png"
        _write(path, image)
        paths.append(path)

    inspector = NormalReferenceInspector.fit(paths, inspection_size=256, min_images=5)
    good = cv2.imread(str(paths[0]))
    assert inspector.inspect(good).status == "OK"

    abnormal = good.copy()
    cv2.rectangle(abnormal, (110, 90), (170, 145), (0, 0, 255), -1)
    result = inspector.inspect(abnormal)
    assert result.status == "NG"
    assert result.regions
    assert result.confidence >= 0.5


def test_model_round_trip(tmp_path):
    paths = []
    for index in range(5):
        image = np.full((100, 100, 3), 100 + index, np.uint8)
        # Serialization fixture still needs a registrable part, not a flat image.
        cv2.rectangle(image,(20,25),(75,80),(160+index,150,130),-1)
        cv2.circle(image,(40,45),8,(70,80,90),-1)
        path = tmp_path / f"normal_{index}.png"
        _write(path, image)
        paths.append(path)
    original = NormalReferenceInspector.fit(paths, inspection_size=160)
    model_path = original.save(tmp_path / "reference.npz")
    loaded = NormalReferenceInspector.load(model_path)
    assert loaded.training_count == 5
    assert loaded.size == (160, 160)


def test_normal_only_evaluation_uses_holdout_and_reports_truthful_metrics(tmp_path):
    paths = []
    rng = np.random.default_rng(21)
    for index in range(10):
        image = np.full((160, 180, 3), (48, 72, 96), np.uint8)
        cv2.ellipse(image, (90, 80), (52, 42), 0, 0, 360, (116, 138, 158), -1)
        noise = rng.normal(0, 0.8, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path = tmp_path / f"known_good_{index:02d}.png"
        _write(path, image)
        paths.append(path)

    inspector, metrics = NormalReferenceInspector.fit_with_evaluation(
        paths, inspection_size=192, min_images=5
    )

    assert inspector.training_count == 10
    assert metrics["metric_scope"] == "known_good_holdout"
    assert metrics["independent_validation"] is True
    assert metrics["validation_images"] == 2
    assert 0.0 <= metrics["normal_acceptance_rate"] <= 1.0
    assert 0.0 <= metrics["false_ng_rate"] <= 1.0
    assert metrics["defect_metrics_available"] is False
    assert metrics["production_threshold"] > 0


def test_normal_reference_reports_monotonic_real_progress(tmp_path):
    paths = []
    for index in range(8):
        image = np.full((96, 112, 3), (55, 78, 102), np.uint8)
        cv2.ellipse(image, (56, 48), (30, 25), 0, 0, 360, (112 + index, 132, 150), -1)
        path = tmp_path / f"progress_good_{index:02d}.png"
        _write(path, image)
        paths.append(path)

    updates = []
    NormalReferenceInspector.fit_with_evaluation(
        paths,
        inspection_size=128,
        min_images=5,
        progress_callback=lambda value, phase: updates.append((value, phase)),
    )

    values = [value for value, _phase in updates]
    assert len(values) > len(paths)
    assert values == sorted(values)
    assert values[0] > 0
    assert values[-1] >= 0.99
    assert any(phase == "validating_holdout" for _value, phase in updates)


def test_deep_patch_uses_separate_calibration_and_validation_sets(tmp_path, monkeypatch):
    class FakeEncoder:
        def __init__(self, _model_path, input_size=256, grid_size=16):
            self.grid_size = 4

        def locate(self, image):
            return (3,3,image.shape[1]-3,image.shape[0]-3)

        def encode(self, image, box=None):
            value = float(image.mean()) / 255.0
            feature = np.zeros((4, 4, 3), dtype=np.float32)
            feature[..., 0] = value
            feature[..., 1] = 1.0 - value
            feature[..., 2] = np.linspace(0.0, 1.0, 16).reshape(4, 4)
            height, width = image.shape[:2]
            return feature, (0, 0, width, height)

    monkeypatch.setattr(deep_patch_reference, "YoloPatchEncoder", FakeEncoder)
    paths = []
    for index in range(30):
        path = tmp_path / f"good_{index:02d}.png"
        _write(path, np.full((48, 64, 3), 80 + index, np.uint8))
        paths.append(path)

    model, metrics = deep_patch_reference.DeepPatchReference.fit_with_evaluation(
        paths, backbone_path=tmp_path / "locator.pt", min_images=5
    )
    assert model.training_count == 30
    audit = model.training_manifest
    assert len(audit["fit"]) == 19
    assert len(audit["validation"]) == 6
    assert not audit["excluded"]
    assert all(len(item["sha256"]) == 64 for item in audit["fit"])

    assert metrics["independent_validation"] is True
    assert metrics["calibration_images"] == 5
    assert metrics["validation_images"] == 6
    assert metrics["metric_scope"] == "independent_known_good_holdout"
    assert 0.0 <= metrics["normal_acceptance_rate"] <= 1.0
    saved = model.save(tmp_path / "deep_reference.npz")
    loaded = deep_patch_reference.load_inspection_reference(saved)
    assert loaded.ENGINE == "deep_patch"
    assert loaded.training_count == 30
    # The saved model must reproduce the holdout scores that authorized it.
    assert loaded.memory.shape[1] == 19  # 30 - 5 calibration - 6 validation
    assert metrics["production_memory_scope"] == "fit_only_frozen"
    ordered = [paths[int(i)] for i in np.random.default_rng(20260901).permutation(30)]
    validation = ordered[-6:]
    replay = [loaded.inspect(cv2.imread(str(path))).anomaly_score for path in validation]
    assert abs(float(np.mean(replay)) - metrics["anomaly_score_mean"]) < 0.0002
    assert sum(score <= loaded.threshold for score in replay) / 6 == metrics["normal_acceptance_rate"]



def test_position_aware_memory_does_not_match_a_patch_from_the_wrong_location():
    # The old global bank considered both queries normal because each feature
    # existed somewhere in a good image. V4 must preserve spatial identity.
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    positional_memory = np.array([
        [[0.0, 1.0]],
        [[1.0, 0.0]],
    ], dtype=np.float32)
    distances = deep_patch_reference.DeepPatchReference._distances(query, positional_memory)
    assert np.all(distances > 1.0)


def test_deep_result_serializes_review_state_and_heatmap():
    result = InspectionResult(
        "OK", 0.55, 0.9, 1.0, [], review_required=True,
        anomaly_map=[[0.1, 0.9], [0.2, 0.3]],
    )
    payload = result.to_dict()
    assert payload["review_required"] is True
    assert payload["anomaly_map"][0][1] == 0.9


def test_background_change_outside_part_does_not_create_ng(tmp_path):
    paths = []
    object_box = (60, 45, 200, 175)
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(6):
        image = np.full((220, 260, 3), (20 + index * 3, 35, 50), np.uint8)
        cv2.ellipse(image, (130, 110), (62, 50), 0, 0, 360, (105, 125, 145), -1)
        path = image_dir / f"good_{index}.png"
        _write(path, image)
        (label_dir / f"good_{index}.txt").write_text(
            "0 0.5 0.5 0.5384615 0.5909091\n", encoding="utf-8"
        )
        paths.append(path)

    inspector = NormalReferenceInspector.fit(paths, inspection_size=256, min_images=5)
    query = cv2.imread(str(paths[0]))
    changed_background = np.full_like(query, (245, 30, 180))
    x1, y1, x2, y2 = object_box
    # Preserve the small locator safety margin; everything beyond the actual
    # inspection ROI is unrelated background and must not affect the decision.
    pad_x = int(round((x2 - x1) * 0.045))
    pad_y = int(round((y2 - y1) * 0.045))
    px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    px2, py2 = min(query.shape[1], x2 + pad_x), min(query.shape[0], y2 + pad_y)
    changed_background[py1:py2, px1:px2] = query[py1:py2, px1:px2]

    assert inspector.inspect(query, object_box=object_box).status == "OK"
    assert inspector.inspect(changed_background, object_box=object_box).status == "OK"


def test_draw_result_keeps_object_box_for_ok_and_ng():
    from normal_reference import InspectionRegion, InspectionResult

    image = np.zeros((120, 180, 3), dtype=np.uint8)
    ok = InspectionResult("OK", 0.91, 0.1, 0.5, [])
    rendered_ok = NormalReferenceInspector.draw_result(
        image, ok, object_box=(30, 25, 150, 105)
    )
    assert rendered_ok[25, 30, 1] > rendered_ok[25, 30, 2]

    ng = InspectionResult(
        "NG", 0.88, 1.2, 0.5, [InspectionRegion(70, 50, 25, 20, 1.2)]
    )
    rendered_ng = NormalReferenceInspector.draw_result(
        image, ng, object_box=(30, 25, 150, 105)
    )
    assert rendered_ng[25, 30, 2] > rendered_ng[25, 30, 1]
    # A measured region must be visible inside the overall part box.
    assert rendered_ng[50, 70, 2] > rendered_ng[50, 70, 1]


def test_live_decision_requires_persistent_change():
    stabilizer = DecisionStabilizer(window=5, ng_votes=3, ok_votes=4)
    ok = InspectionResult("OK", 0.9, 0.1, 0.5, [])
    ng = InspectionResult("NG", 0.9, 2.0, 0.5, [])
    assert stabilizer.update(ok).input_issue == "waiting_for_frames"
    assert stabilizer.update(ng).input_issue == "waiting_for_frames"
    assert stabilizer.update(ng).input_issue == "waiting_for_frames"
    assert stabilizer.update(ng).status == "NG"
    assert stabilizer.update(ok).review_required
    assert stabilizer.update(ok).review_required
    assert stabilizer.update(ok).review_required
    assert stabilizer.update(ok).status == "OK"


class NormalReferenceSmokeTests(unittest.TestCase):
    def test_reference_engine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_known_good_is_ok_and_local_change_is_ng(Path(temp_dir))

    def test_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_model_round_trip(Path(temp_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_pending_ng_is_review_not_false_green():
    stabilizer = DecisionStabilizer()
    stabilizer.update(InspectionResult("OK", .9, .2, 1., []))
    result = stabilizer.update(InspectionResult("NG", .6, 1.1, 1., []))
    assert result.review_required
    assert result.anomaly_score == 1.1


def test_initial_ok_requires_four_clean_frames_and_restarts_after_gap():
    stabilizer = DecisionStabilizer()
    ok = InspectionResult("OK", .9, .2, 1., [])
    for _ in range(3):
        result = stabilizer.update(ok)
        assert result.input_issue == "waiting_for_frames"
        assert result.to_dict()["location"] == "not_checked"
    assert stabilizer.update(ok) is ok
    stabilizer.update(InspectionResult("NO_PART", 0., 0., 1., []))
    assert stabilizer.update(ok).input_issue == "waiting_for_frames"


def test_temporal_hold_does_not_replay_previous_frame():
    stabilizer = DecisionStabilizer()
    stabilizer.update(InspectionResult("NG", .9, 2., 1., [], "Honda", .95, anomaly_map=[[2.]]))
    current = InspectionResult("OK", .8, .7, 1., [], "Honda", .8, anomaly_map=[[.7]])
    held = stabilizer.update(current)
    assert held.review_required
    assert held.anomaly_map == current.anomaly_map
    assert held.anomaly_score == current.anomaly_score
    assert held.brand_confidence == current.brand_confidence


def test_changed_part_resets_votes_without_missing_frame():
    for different_brand in (True, False):
        stabilizer = DecisionStabilizer()
        stabilizer.update(InspectionResult("NG", .9, 2., 1., [], "Honda"), object_box=(0,0,100,100))
        result = stabilizer.update(InspectionResult("OK", .9, .2, 1., [], "Toyota" if different_brand else "Honda"),
                                   object_box=(0,0,100,100) if different_brand else (200,200,300,300))
        assert result.status == "REVIEW" and result.input_issue == "waiting_for_frames"


def test_review_frames_cannot_clear_ng_latch():
    stabilizer = DecisionStabilizer()
    stabilizer.update(InspectionResult("NG", .9, 2., 1., []))
    for _ in range(6):
        assert stabilizer.update(InspectionResult("OK", .5, .95, 1., [], review_required=True)).review_required
    # Still awaiting four clean frames; one clean frame must remain REVIEW.
    assert stabilizer.update(InspectionResult("OK", .9, .2, 1., [])).review_required


def test_heatmap_returns_to_camera_coordinates():
    heatmap = np.zeros((32,32), np.float32)
    heatmap[10,10] = 3.
    # +20 px in a 320px ROI must move the peak +2 grid cells.
    warp = np.array([[1.,0.,20.],[0.,1.,30.]],np.float32)
    mapped = deep_patch_reference._camera_heatmap(heatmap, warp, 320)
    assert np.unravel_index(mapped.argmax(),mapped.shape) == (13,12)
    np.testing.assert_allclose(deep_patch_reference._camera_heatmap(heatmap, np.eye(2,3,dtype=np.float32),320),heatmap)


def test_truncated_part_abstains_without_running_defect_encoder():
    inspector = deep_patch_reference.DeepPatchReference(np.zeros((1,1),np.float32),1.2,1.,1,"unused")
    image = np.zeros((100,120,3),np.uint8)
    for box in [(0,10,80,90),(10,0,80,90),(10,10,120,90),(10,10,80,100)]:
        result = inspector.inspect(image,object_box=box)
        assert result.status == "REVIEW"
        assert result.review_required and result.input_issue == "part_clipped_by_frame"
        assert inspector._encoder is None
        assert not result.regions


def test_incomplete_part_resets_previous_ng_votes():
    stabilizer=DecisionStabilizer()
    stabilizer.update(InspectionResult("NG",.9,2.,1.,[]))
    issue=InspectionResult("REVIEW",0.,0.,1.,[],review_required=True,input_issue="part_clipped_by_frame")
    assert stabilizer.update(issue) is issue
    result=stabilizer.update(InspectionResult("OK",.9,.2,1.,[]))
    assert result.status == "REVIEW" and result.input_issue == "waiting_for_frames"


def test_reference_split_keeps_export_variants_together():
    paths=[Path(f"export_{v}/source_{i:02d}.rf.hash{v}.jpg") for i in range(30) for v in range(3)]
    fit,cal,val,independent=deep_patch_reference._split_reference_paths(paths,5)
    assert independent
    groups=lambda items:{p.stem.split(".rf.")[0] for p in items}
    assert not groups(fit)&groups(cal)
    assert not groups(fit)&groups(val)
    assert not groups(cal)&groups(val)
    assert len(fit)+len(cal)+len(val)==90


def test_many_copies_do_not_fake_independent_holdout():
    paths=[Path(f"source.rf.hash{i}.jpg") for i in range(50)]
    assert deep_patch_reference._split_reference_paths(paths,5)[3] is False


def test_train_excludes_clipped_parts_before_fitting(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self,*args): pass
        def locate(self,image): return (0,3,image.shape[1]-3,image.shape[0]-3)
        def fit_alignment(self,*args): raise AssertionError("Partial parts must not be trained")
    monkeypatch.setattr(deep_patch_reference,"YoloPatchEncoder",Encoder)
    paths=[]
    for i in range(5):
        path=tmp_path/f"frame{i}.png"
        _write(path,np.zeros((48,64,3),np.uint8));paths.append(path)
    import pytest
    with pytest.raises(ValueError,match="Only 0 images"):
        deep_patch_reference.DeepPatchReference.fit_with_evaluation(paths,backbone_path="unused.pt")


def test_position_normalization_excludes_same_source_variants():
    maps=[np.full((2,2,1),value,np.float32) for value in (0.,0.,2.)]
    paths=[Path("a.rf.1.jpg"),Path("a.rf.2.jpg"),Path("b.rf.1.jpg")]
    bank,indices=deep_patch_reference.DeepPatchReference._memory(maps,return_indices=True)
    scale=deep_patch_reference.DeepPatchReference._fit_position_scale(maps,paths,bank,indices,np.ones((2,2),bool))
    np.testing.assert_allclose(scale,2.*1.03,rtol=1e-6)
