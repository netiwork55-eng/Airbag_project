"""Contract and failure-path tests. Synthetic fixtures do not measure defect recall."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection.schema import Recipe, decision
from inspection.store import Store, atomic_json
from inspection.pipeline import Pipeline
from inspection.runtime import Cycle, FramePump
from inspection.validation import validate


@pytest.fixture
def setup(tmp_path):
    store=Store(tmp_path)
    frame=np.random.default_rng(8).integers(60,170,(120,160,3),dtype=np.uint8)
    def add(image,label="GOOD",group=""):
        return store.put_image(cv2.imencode('.png',image)[1].tobytes(),label=label,group=group or "fixture-"+hashlib.sha256(image.tobytes()).hexdigest())
    reference=add(frame)
    config={"product":"Part A","reference_id":reference['id'],
        "areas":[{"id":"surface","name":"Surface","region":{"x":.1,"y":.1,"width":.8,"height":.8},
                  "method":{"kind":"brightness","low":80,"high":180,"review_margin":2}}],
        "quality":{"min_brightness":5,"max_brightness":250},"locator":{"fixture_confirmed":True},
        "validation":{"min_good":1,"min_ng":1,"min_defect_recall":1}}
    return store,frame,config,add


@pytest.mark.parametrize('states,expected',[
    ([],"UNABLE_TO_INSPECT"),(["PASS"],"PASS"),(["PASS","NG"],"NG"),
    (["PASS","REVIEW"],"REVIEW"),(["NG","REVIEW"],"NG"),
    (["NG","UNABLE_TO_INSPECT"],"UNABLE_TO_INSPECT"),(["unexpected"],"UNABLE_TO_INSPECT")])
def test_decision(states,expected):
    assert decision([{"area_id":str(i),"state":s} for i,s in enumerate(states)])[0]==expected


def test_optional_cannot_certify():
    assert decision([{"area_id":"x","state":"PASS","required":False}])[0]=="UNABLE_TO_INSPECT"


def test_independent_areas_and_invalid_image(setup):
    store,frame,c,_=setup
    c['areas'].append({"id":"second","name":"Second","region":{"x":.2,"y":.2,"width":.3,"height":.3},
        "method":{"kind":"presence","threshold":.8}})
    p=Pipeline(store,store.save_recipe(c))
    result,_=p.run(frame)
    assert result['state']=='PASS' and len(result['areas'])==2
    assert p.run(None)[0]['state']=='UNABLE_TO_INSPECT'
    assert p.run(np.zeros_like(frame))[0]['state']=='UNABLE_TO_INSPECT'
    assert p.run(frame[:50])[0]['state']=='UNABLE_TO_INSPECT'


def test_bad_reference_never_passes_constant_template(setup):
    store,frame,c,add=setup
    c['reference_id']=add(np.full_like(frame,110))['id']
    c['areas'][0]['method']={'kind':'presence'}
    assert Pipeline(store,store.save_recipe(c)).run(frame)[0]['state']=='UNABLE_TO_INSPECT'


def test_template_alignment_and_clipped_roi(setup):
    store,frame,c,_=setup
    c['locator']={'method':'template','region':{'x':.3,'y':.3,'width':.4,'height':.4},'min_score':.9,'max_shift':.3}
    c['areas'][0]['method']={'kind':'presence','threshold':.9,'review_margin':.01}
    p=Pipeline(store,store.save_recipe(c))
    shifted=cv2.warpAffine(frame,np.float32([[1,0,8],[0,1,5]]),(160,120),borderValue=(100,100,100))
    assert p.run(shifted)[0]['state']=='PASS'
    c['areas'][0]['region']={'x':0,'y':0,'width':1,'height':1}
    result=Pipeline(store,store.save_recipe(c)).run(shifted)[0]
    assert result['state']=='UNABLE_TO_INSPECT' and 'visible' in result['reasons'][0]


def test_validation_needs_explicit_part_recall_policy(setup):
    store,frame,c,add=setup
    c['locator']={'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}}
    c['validation'].pop('min_defect_recall')
    ids=[add(frame+1)['id'],add(frame//2,'NG')['id']]
    saved=store.save_recipe(c)
    report=validate(store,Pipeline(store,saved),ids)
    assert report['metrics']['defect_recall']==1
    assert not report['eligible'] and any('Set the required part-level' in b for b in report['blockers'])
    assert report['limits']['min_defect_recall'] is None
    with pytest.raises(ValueError,match='passing independent'):
        store.activate(saved['id'],report['id'])
    assert store.production() is None


def test_many_good_images_cannot_hide_reviewed_ng(setup):
    store,frame,c,add=setup
    c['locator']={'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}}
    ids=[add(frame+i)['id'] for i in range(1,11)]
    # Deliberately exercise the decision band; this is not a defect-accuracy dataset.
    borderline=np.clip(frame.astype(float)*.71,0,255).astype(np.uint8)
    ids.append(add(borderline,'NG')['id'])
    saved=store.save_recipe(c)
    report=validate(store,Pipeline(store,saved),ids)
    assert report['rows'][-1]['result']['state']=='REVIEW'
    assert report['metrics']['review_rate']<c.get('validation',{}).get('max_review',.1)
    assert report['metrics']['false_accept_rate']==0
    assert report['metrics']['defect_recall']==0
    assert report['metrics']['ng_review_count']==1
    assert not report['eligible']
    assert any('defect_recall does not meet' in b for b in report['blockers'])
    with pytest.raises(ValueError):store.activate(saved['id'],report['id'])


@pytest.mark.parametrize('value',[0,-.1,1.01,float('nan'),float('inf')])
def test_invalid_part_recall_policy_is_rejected(setup,value):
    store,_,c,_=setup
    c['validation']['min_defect_recall']=value
    with pytest.raises(ValueError):store.save_recipe(c)


def test_anomaly_not_trained_is_unable(setup):
    store,frame,c,_=setup
    c['areas'][0]['method']={'kind':'anomaly'}
    r=Pipeline(store,store.save_recipe(c)).run(frame)[0]
    assert r['state']=='UNABLE_TO_INSPECT' and 'trained' in r['reasons'][0]


@pytest.mark.parametrize('change',[lambda c:c['areas'][0]['region'].update(width=2),
    lambda c:c['areas'][0]['method'].update(threshold=float('nan')),
    lambda c:c['areas'].append(copy.deepcopy(c['areas'][0]))])
def test_bad_recipe_rejected(setup,change):
    store,_,c,_=setup
    change(c)
    with pytest.raises(ValueError):store.save_recipe(c)


def test_pixel_duplicate_and_conflicting_labels(setup):
    store,frame,c,add=setup
    assert add(frame)['id']==c['reference_id']
    with pytest.raises(ValueError):add(frame,'NG')


def test_reference_and_same_group_excluded_from_validation(setup):
    store,frame,c,add=setup
    first=add(frame,'GOOD')
    other=add(np.clip(frame.astype(int)+1,0,255).astype(np.uint8),'GOOD',first['group'])
    r=validate(store,Pipeline(store,store.save_recipe(c)),[first['id'],other['id']])
    assert not r['eligible'] and r['metrics']['good_count']==0
    assert r['metrics']['false_accept_rate'] is None


def test_legacy_label_and_automatic_hash_group_cannot_validate(setup):
    store,frame,c,add=setup
    old=add(frame+1)
    old.pop('confirmed_at')
    atomic_json(store.path('images',old['id']),old)
    ungrouped=store.put_image(cv2.imencode('.png',frame//2)[1].tobytes(),label='NG')
    r=validate(store,Pipeline(store,store.save_recipe(c)),[old['id'],ungrouped['id']])
    assert r['metrics']['good_count']==r['metrics']['ng_count']==0
    assert r['metrics']['defect_recall'] is None and not r['eligible']
    assert 'human-confirmed' in r['rows'][0]['excluded']
    assert 'capture group' in r['rows'][1]['excluded']


def test_source_group_change_after_validation_blocks_activation(setup):
    store,frame,c,add=setup
    c['locator']={'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}}
    good,ng=add(frame+1),add(frame//2,'NG')
    saved=store.save_recipe(c)
    r=validate(store,Pipeline(store,saved),[good['id'],ng['id']])
    assert r['eligible'] and r['metrics']['defect_recall']==1
    ref=store.read('images',c['reference_id'])
    ref['group']=good['group']
    atomic_json(store.path('images',ref['id']),ref)
    with pytest.raises(ValueError,match='capture group changed'):
        store.activate(saved['id'],r['id'])
    assert store.production() is None
    retry=validate(store,Pipeline(store,saved),[good['id'],ng['id']])
    assert retry['rows'][0]['excluded'] and not retry['eligible']


def test_duplicate_unknown_can_receive_actual_group_without_label_confirmation(setup):
    store,frame,c,add=setup
    raw=cv2.imencode('.png',frame+2)[1].tobytes()
    old=store.put_image(raw)
    confirmed=store.put_image(raw,group='physical-part-22')
    assert confirmed['id']==old['id'] and confirmed['group_confirmed']
    assert confirmed['label']=='UNKNOWN' and 'confirmed_at' not in confirmed
    with pytest.raises(ValueError,match='different capture group'):
        store.put_image(raw,group='another-part')


def test_activation_requires_exact_validated_recipe_and_rollback(setup):
    store,frame,c,add=setup
    c['locator']={'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}}
    good=add(frame+1)['id']
    ng=add(frame//2,'NG')['id']
    saved=store.save_recipe(c)
    r=validate(store,Pipeline(store,saved),[good,ng])
    assert r['eligible']
    store.activate(saved['id'],r['id'])
    changed=copy.deepcopy(c);changed['areas'][0]['method']['high']=185
    next_recipe=store.save_recipe(changed,parent=saved['id'])
    with pytest.raises(ValueError):store.activate(next_recipe['id'],r['id'])
    r2=validate(store,Pipeline(store,next_recipe),[good,ng])
    store.activate(next_recipe['id'],r2['id'])
    assert store.rollback()['recipe_id']==saved['id']
    tampered=store.read('recipes',saved['id']);tampered['config']['product']='changed'
    atomic_json(store.path('recipes',saved['id']),tampered)
    with pytest.raises(ValueError):store.recipe(saved['id'])


def test_cycle_holds_ng_and_keeps_matching_evidence():
    c=Cycle(stable_frames=2)
    im=np.zeros((20,20,3),np.uint8)
    for i,state in enumerate(['PASS','NG','NG','PASS','PASS']):
        c.observe({'state':state,'reasons':[state]},im+i,{},now_mono=i*.1)
    assert c.current['result']['state']=='NG'
    assert c.observe({},im,{},absent=True,now_mono=1) is None
    assert c.observe({},im,{},absent=True,now_mono=1.2) is None
    closed=c.observe({},im,{},absent=True,now_mono=1.5)
    assert closed['complete'] and closed['result']['state']=='NG' and closed['image'][0,0,0]==2
    assert c.observe({},im,{},absent=True,now_mono=2) is None


def test_failed_or_incomplete_cycle_never_passes():
    c=Cycle();im=np.zeros((20,20,3),np.uint8)
    c.observe({'state':'PASS'},im,{})
    assert c.finish(False)['result']['state']=='UNABLE_TO_INSPECT'
    for state in ['PASS','UNABLE_TO_INSPECT','UNABLE_TO_INSPECT','UNABLE_TO_INSPECT','PASS']:
        c.observe({'state':state},im,{})
    assert c.finish(True)['result']['state']=='UNABLE_TO_INSPECT'


def test_cycle_settles_before_latching_and_uncertain_tail_is_not_pass():
    c=Cycle();im=np.zeros((20,20,3),np.uint8)
    for state in ['UNABLE_TO_INSPECT','PASS','PASS','PASS']:
        c.observe({'state':state},im,{})
    assert c.finish(True)['result']['state']=='PASS'
    for state in ['PASS','PASS','PASS','NG']:
        c.observe({'state':state},im,{})
    assert c.finish(True)['result']['state']=='REVIEW'


def test_ids_cannot_escape_store(setup):
    store,*_=setup
    for key in ['../foo','image_1','C:/data']:
        with pytest.raises(ValueError):store.path('images',key)


def test_frame_pump_discards_backlog_and_releases_camera():
    import time
    class Camera:
        closed=False
        count=0
        def read(self):
            time.sleep(.005)
            self.count+=1
            return np.full((16,16,3),self.count%255,np.uint8)
        def close(self):self.closed=True
    camera=Camera();pump=FramePump(camera)
    first=pump.next(0)
    time.sleep(.06)
    latest=pump.next(first[0])
    assert latest[0]>first[0]+1
    assert latest[1][0,0,0]==latest[0]%255
    pump.close()
    assert camera.closed and not pump.thread.is_alive()


def test_frame_pump_error_never_returns_old_image():
    class Camera:
        closed=False
        def read(self):raise RuntimeError('camera disconnected')
        def close(self):self.closed=True
    camera=Camera();pump=FramePump(camera)
    with pytest.raises(RuntimeError,match='disconnected'):pump.next(0)
    pump.close()
    assert camera.closed


def test_camera_release_failure_is_reported():
    class Camera:
        def read(self):raise RuntimeError('lost camera')
        def close(self):raise RuntimeError('driver release error')
    pump=FramePump(Camera())
    with pytest.raises(RuntimeError,match='release failed'):pump.close()


def test_modified_report_cannot_activate(setup):
    store,frame,c,add=setup
    good=add(np.full_like(frame,120))['id']
    saved=store.save_recipe(c)
    report=validate(store,Pipeline(store,saved),[good])
    report['eligible']=True
    atomic_json(store.path('tests',report['id']),report)
    with pytest.raises(ValueError,match='report has changed'):store.activate(saved['id'],report['id'])


def test_old_software_validation_cannot_activate(setup):
    from inspection.store import digest
    store,frame,c,add=setup
    c['locator']={'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}}
    saved=store.save_recipe(c)
    report=validate(store,Pipeline(store,saved),[add(frame+1)['id'],add(frame//2,'NG')['id']])
    assert report['eligible']
    report['engine_version']='old-inspection-software'
    report['report_hash']=digest({k:v for k,v in report.items() if k!='report_hash'})
    atomic_json(store.path('tests',report['id']),report)
    with pytest.raises(ValueError,match='software changed'):store.activate(saved['id'],report['id'])
