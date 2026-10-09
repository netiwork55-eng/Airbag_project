from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from part_locator import analyze_part,part_class_id

def box(cls,confidence,xyxy):
    return SimpleNamespace(cls=[cls],conf=[confidence],xyxy=[xyxy])

def test_defect_subclasses_cannot_be_selected_as_part():
    assert part_class_id({0:"Stain-1",1:"Mark_2",2:"Part"})==2
    assert part_class_id({0:"Stain-1",1:"NG"}) is None

def test_locator_ignores_defects_and_implausible_boxes():
    model=Mock()
    model.predict.return_value=[SimpleNamespace(boxes=[box(0,.99,(0,0,200,200)),box(1,.99,(15,15,185,185)),box(0,.8,(30,30,160,160))])]
    result=analyze_part(model,0,np.zeros((200,200,3),np.uint8))
    assert result["box"]==(30,30,160,160)
    assert result["defect_confidence"] is None
    assert model.predict.call_args.kwargs["imgsz"]==640
    assert "half" not in model.predict.call_args.kwargs
