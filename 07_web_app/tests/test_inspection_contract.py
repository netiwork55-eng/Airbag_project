from pathlib import Path
from unittest.mock import Mock
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_contract import RecipeBoundInspector, build_inspection_contract
from normal_reference import InspectionResult, DecisionStabilizer
import app

@pytest.fixture
def bound(tmp_path):
    model=tmp_path/"model.npz";model.write_bytes(b"model-a")
    raw=Mock(threshold=1.2)
    raw.inspect.return_value=InspectionResult("OK",.9,.3,1.2,[])
    ref={"inspection_contract":build_inspection_contract(model,[{"id":"toyota","recipe":{"brand":"Toyota"}}],check_brand=True)}
    return RecipeBoundInspector(raw,model,ref),raw,model,ref

@pytest.mark.parametrize("brand,confidence,issue",[("Honda",1.,"recipe_brand_mismatch"),(None,None,"recipe_brand_unconfirmed"),("unknown",1.,"recipe_brand_unconfirmed"),("Toyota",.2,"recipe_brand_unconfirmed"),("Toyota",float("nan"),"recipe_brand_unconfirmed")])
def test_wrong_or_unconfirmed_brand_never_reaches_defect_model(bound,brand,confidence,issue):
    wrapped,raw,_,_=bound
    result=wrapped.inspect(None,brand=brand,brand_confidence=confidence)
    assert result.status=="REVIEW" and result.input_issue==issue and not result.regions
    raw.inspect.assert_not_called()
    assert DecisionStabilizer().update(result).input_issue==issue

def test_matching_brand_preserves_underlying_result(bound):
    wrapped,raw,_,_=bound
    assert wrapped.inspect(None,brand=" toyota ",brand_confidence=1.) is raw.inspect.return_value

def test_changed_model_invalidates_contract(bound):
    _,raw,path,ref=bound;path.write_bytes(b"different")
    result=RecipeBoundInspector(raw,path,ref).inspect(None,brand="Toyota",brand_confidence=1.)
    assert result.input_issue=="recipe_model_mismatch"
    raw.inspect.assert_not_called()

def test_missing_contract_cannot_pass_as_ok(bound):
    _,raw,path,_=bound
    assert RecipeBoundInspector(raw,path,{}).inspect(None).input_issue=="recipe_unverified"

def test_mixed_brand_corpus_cannot_build_verified_contract(bound):
    _,_,path,_=bound
    contract=build_inspection_contract(path,[{"recipe":{"brand":"Toyota"}},{"recipe":{"brand":"Honda"}}],check_brand=True)
    assert contract.get("error")=="recipe_brand_unverified"

@pytest.mark.parametrize("size",[256,320,512,640,1024])
def test_training_size_is_not_silently_reduced(size):
    assert app.reference_training_size(size)==size

@pytest.mark.parametrize("size",[0,160,513,512.5,2048,None,"bad",float("nan")])
def test_invalid_training_size_is_rejected(size):
    with pytest.raises(ValueError):app.reference_training_size(size)


@pytest.mark.parametrize("area", [[0.,0.,.5,1.], [0.,0.,0.,1.]])
def test_changed_area_blocks_inspection(bound, area):
    _, raw, path, reference = bound
    raw.inspection_area = [0.,0.,1.,1.]
    reference["inspection_area"] = area
    guard = RecipeBoundInspector(raw, path, reference)
    assert guard.inspect(None).input_issue == "recipe_area_mismatch"
    raw.inspect.assert_not_called()


def test_saved_float32_area_matches_config(bound):
    _, raw, path, reference = bound
    raw.inspection_area = [.10000000149,.10000000149,.899999976,.899999976]
    reference["inspection_area"] = [.1,.1,.9,.9]
    assert RecipeBoundInspector(raw, path, reference).contract_issue is None
