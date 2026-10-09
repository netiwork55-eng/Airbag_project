import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_cycles import PartCycle, save_cycle
from normal_reference import InspectionResult

def result(status, issue=None):
    return InspectionResult(status,.9,2. if status=="NG" else .2,1.,[],review_required=status=="REVIEW",input_issue=issue)

def test_ng_is_held_until_confirmed_removal_and_next_part_starts_clean():
    cycle=PartCycle()
    image=np.zeros((30,30,3),np.uint8)
    cycle.observe(True,result("NG"),now=0,frame=image)
    cycle.observe(True,result("OK"),now=.1,frame=image+1)
    assert cycle.current["decision"]=="NG"
    assert cycle.observe(False,None,now=.2) is None
    assert cycle.observe(False,None,now=.3) is None
    closed=cycle.observe(False,None,now=.6)
    assert closed[0]["decision"]=="NG" and closed[0]["complete"]
    assert not closed[1].any()
    assert cycle.observe(False,None,now=.8) is None
    cycle.observe(True,result("OK"),now=1.,frame=image)
    assert cycle.current["decision"]=="OK"

def test_brief_locator_loss_does_not_create_second_part():
    cycle=PartCycle()
    cycle.observe(True,result("OK"),now=0)
    original=cycle.current["id"]
    cycle.observe(False,None,now=.1)
    cycle.observe(True,result("OK"),now=.2)
    assert cycle.current["id"]==original

def test_stop_during_ok_part_requires_review(tmp_path):
    cycle=PartCycle()
    cycle.observe(True,result("OK"),frame=np.zeros((30,30,3),np.uint8))
    record=save_cycle(tmp_path,cycle.finish("camera_stopped"),{"active_model":"model"},"test")
    assert record["decision"]=="REVIEW" and not record["complete"]
    assert (tmp_path/'03_dataset/inspection_cycles'/record['image']).is_file()
    assert cycle.finish("camera_stopped") is None

def test_waiting_is_not_a_defect_but_input_failure_prevents_ok():
    cycle=PartCycle()
    cycle.observe(True,result("REVIEW","waiting_for_frames"))
    cycle.observe(True,result("OK"))
    assert cycle.current["decision"]=="OK"
    cycle.observe(True,result("REVIEW","alignment_failed"))
    cycle.observe(True,result("OK"))
    assert cycle.current["decision"]=="REVIEW"
