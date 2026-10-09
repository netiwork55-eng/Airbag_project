import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_area import checked_area, area_grid
from deep_patch_reference import DeepPatchReference
import app


@pytest.mark.parametrize('value',[[0,0,0,1],[-1,0,1,1],[0,0,2,1],[0,0,float('nan'),1],[False,0,1,1],[], 'full'])
def test_invalid_area_rejected(value):
    with pytest.raises(ValueError): checked_area(value)


def test_area_is_persisted_and_controls_scoring(tmp_path):
    area=[0.,0.,.5,1.]
    mask=area_grid(area,(4,4))
    model=DeepPatchReference(np.zeros((16,1,1)),1.,1.,10,'unused.pt',grid_size=4,
                            active_mask=mask,inspection_area=area)
    path=model.save(tmp_path/'model.npz')
    loaded=DeepPatchReference.load(path)
    assert loaded.inspection_area==area
    assert np.array_equal(loaded.active_mask,mask)
    features=np.zeros((4,4,1),np.float32);features[:,2:]=10
    loaded._encoder=SimpleNamespace(encode=lambda *args:(features,(2,2,38,38)))
    image=np.zeros((40,40,3),np.uint8)
    assert loaded.inspect(image).status=='OK'
    features[:,:2]=10
    assert loaded.inspect(image).status=='NG'


def test_changed_area_changes_review_scope_and_training_fingerprint(tmp_path):
    first={'recipe_datasets':[{'id':'part'}], 'inspection_size':320,'min_good_images':5,'min_region_ratio':.001}
    second={**first,'inspection_area':[0.,0.,.5,1.]}
    assert app.reference_review_scope(first)!=app.reference_review_scope(second)
    image=tmp_path/'image.png';image.write_bytes(b'source')
    assert app.normal_reference_training_fingerprint([image],first)!=app.normal_reference_training_fingerprint([image],second)


def test_minimum_defect_area_is_not_silently_shrunk():
    assert app.reference_region_ratio(.002) == .002
    for value in (float('nan'), 0, .3, True, 'bad'):
        with pytest.raises(ValueError): app.reference_region_ratio(value)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_features_cannot_be_reported_as_ok(invalid):
    from types import SimpleNamespace
    features = np.full((32,32,1), invalid, dtype=np.float32)
    model = DeepPatchReference(np.zeros((1024,1,1),np.float32),1.,.5,1,"unused",320)
    model._encoder = SimpleNamespace(encode=lambda *_: (features,(0,0,40,40)))
    result = model.inspect(np.zeros((40,40,3),np.uint8))
    assert result.status == "REVIEW"
    assert result.input_issue == "invalid_model_output"
    assert result.to_dict()["location"] == "not_checked"


def test_transition_status_is_review_without_consumers_having_to_reinterpret_it():
    from normal_reference import DecisionStabilizer, InspectionResult
    stabilizer = DecisionStabilizer()
    good = InspectionResult("OK",.9,.2,1.,[])
    for _ in range(4):
        stabilizer.update(good)
    result = stabilizer.update(InspectionResult("NG",.9,2.,1.,[]))
    assert result.status == "REVIEW" and result.review_required


def test_disjoint_areas_round_trip_and_leave_gap_uninspected(tmp_path):
    from inspection_area import checked_areas, areas_grid
    areas=[[0.,0.,.25,1.],[.75,0.,1.,1.]]
    mask=areas_grid(areas,(4,4))
    assert mask[:,0].all() and mask[:,3].all() and not mask[:,1:3].any()
    model=DeepPatchReference(np.zeros((16,1,1)),1.,1.,10,'unused.pt',grid_size=4,active_mask=mask,inspection_areas=areas)
    loaded=DeepPatchReference.load(model.save(tmp_path/'multi.npz'))
    assert loaded.inspection_areas==areas
    features=np.zeros((4,4,1),np.float32);features[:,1:3]=10
    loaded._encoder=SimpleNamespace(encode=lambda *args:(features,(2,2,38,38)))
    assert loaded.inspect(np.zeros((40,40,3),np.uint8)).status=='OK'
    features[:,0]=10
    assert loaded.inspect(np.zeros((40,40,3),np.uint8)).status=='NG'


def test_multi_area_review_scope_differs_from_single_area():
    one={'recipe_datasets':[{'id':'part','recipe':{}}],'inspection_area':[0.,0.,.25,1.]}
    two={**one,'inspection_areas':[[0.,0.,.25,1.],[.75,0.,1.,1.]]}
    assert app.reference_review_scope(one)!=app.reference_review_scope(two)
