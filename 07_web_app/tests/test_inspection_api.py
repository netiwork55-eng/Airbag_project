"""Exercise the actual new HTTP workflow with isolated local data."""
from pathlib import Path
import sys
import hashlib
import cv2
import numpy as np
import pytest
from fastapi import FastAPI
pytest.importorskip("httpx2", reason="Run API tests with .venv-inspection, which contains the HTTP test client")
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection.api import InspectionService, make_router


@pytest.fixture
def client(tmp_path):
    service=InspectionService(tmp_path)
    app=FastAPI()
    app.include_router(make_router(service))
    with TestClient(app) as client:
        yield client,service
    service.close()


def upload(client,frame,label='GOOD'):
    raw=cv2.imencode('.png',frame)[1].tobytes()
    r=client.post('/api/vision/images',files=[('files',('image.png',raw,'image/png'))],data={'label':label,'group':'fixture-'+hashlib.sha256(raw).hexdigest()})
    assert r.status_code==200,r.text
    return r.json()['images'][0]


def test_cv_workflow_needs_no_training_and_review_does_not_send(client):
    c,s=client
    ref=upload(c,np.full((100,100,3),130,np.uint8))
    config={'product':'API test','reference_id':ref['id'],
        'areas':[{'id':'a','name':'Brightness','region':{'x':0,'y':0,'width':1,'height':1},
                  'method':{'kind':'brightness','low':90,'high':180}}],
        'locator':{'fixture_confirmed':True},'validation':{'min_good':1,'min_ng':1}}
    saved=c.post('/api/vision/recipes',json={'config':config}).json()
    good=upload(c,np.full((100,100,3),120,np.uint8))
    ng=upload(c,np.full((100,100,3),50,np.uint8),'NG')
    report=c.post('/api/vision/validate',json={'recipe_id':saved['id'],'image_ids':[good['id'],ng['id']]}).json()
    assert report['metrics']['good_accepted']==1 and report['metrics']['defects_rejected']==1
    assert not report['eligible'] and any('locator' in b for b in report['blockers'])
    assert c.post('/api/vision/activate',json={'recipe_id':saved['id'],'report_id':report['id']}).status_code==400
    assert c.get('/api/vision/status').json()['production'] is None
    event=c.post('/api/vision/test',json={'recipe_id':saved['id'],'image_id':ng['id']}).json()
    assert event['state']=='NG' and event['image_id']==ng['id']
    calls=[]
    s.cvat_send=lambda selected:calls.append(selected) or {'synced':len(selected)}
    assert c.post('/api/vision/review/'+event['id'],json={'status':'confirmed_ng'}).status_code==200
    assert calls==[]
    assert c.post('/api/vision/cvat/send',json={'event_ids':[]}).status_code==400
    s.work_lock.acquire()  # CVAT does not acquire the inspection lock.
    s.legacy_busy=lambda:True
    assert c.post('/api/vision/cvat/send',json={'event_ids':[event['id']]}).json()['synced']==1
    s.work_lock.release()
    s.legacy_busy=lambda:False
    assert len(calls)==1
    # A fixture image test cannot start continuous place/remove production without a locator.
    assert c.post('/api/vision/production/start',json={}).status_code==400


def test_busy_tasks_and_invalid_images_fail_cleanly(client):
    c,s=client
    s.work_lock.acquire()
    try:
        assert c.post('/api/vision/recipes',json={}).status_code==409
    finally:s.work_lock.release()
    s.legacy_busy=lambda:True
    assert c.post('/api/vision/test',json={}).status_code==409
    r=c.post('/api/vision/images',files=[('files',('not.png',b'bad','image/png'))])
    assert r.status_code==400


def test_reference_must_be_confirmed(client):
    c,s=client
    ref=upload(c,np.full((100,100,3),130,np.uint8),'UNKNOWN')
    config={'product':'Reference confirmation','reference_id':ref['id'],
        'areas':[{'id':'a','name':'Area','region':{'x':0,'y':0,'width':1,'height':1}}]}
    assert c.post('/api/vision/recipes',json={'config':config}).status_code==400
    assert c.post('/api/vision/images/'+ref['id']+'/confirm',json={'label':'GOOD'}).status_code==200
    assert c.post('/api/vision/recipes',json={'config':config}).status_code==200


def test_no_new_annotation_studio():
    import app
    assert not any(getattr(route,'path','').startswith('/api/annotation/') for route in app.app.routes)


def test_old_good_is_not_confirmed_and_can_be_corrected_by_human(client):
    from inspection.store import atomic_json
    c,s=client
    item=upload(c,np.full((40,40,3),100,np.uint8))
    item.pop('confirmed_at')
    item.pop('group_confirmed')
    atomic_json(s.store.path('images',item['id']),item)
    corrected=c.post('/api/vision/images/'+item['id']+'/confirm',json={'label':'NG','group':'physical-part-1'})
    assert corrected.status_code==200
    assert corrected.json()['label']=='NG' and corrected.json()['group_confirmed']
    assert c.post('/api/vision/images/'+item['id']+'/confirm',json={'label':'GOOD'}).status_code==400


def test_capture_keeps_full_pixels_provenance_and_releases_on_error(client,monkeypatch):
    import inspection.api as routes
    c,s=client
    frame=np.full((120,160,3),120,np.uint8)
    closed=[]
    class Camera:
        def __init__(self,config):pass
        def read(self):return frame
        def metadata(self):return {'driver':'fake','width':160,'height':120}
        def close(self):closed.append(True)
    monkeypatch.setattr(routes,'OpenCVCamera',Camera)
    r=c.post('/api/vision/camera/capture',json={'index':1,'capture_group':'part-1'})
    assert r.status_code==200,r.text
    item=r.json()
    assert item['label']=='UNKNOWN' and 'confirmed_at' not in item
    assert item['capture']['width']==160 and item['capture']['height']==120
    assert item['group']=='part-1' and item['group_confirmed']
    np.testing.assert_array_equal(s.store.image(item['id']),frame)
    assert closed==[True]
    def disconnect(self):raise RuntimeError('Camera disconnected')
    monkeypatch.setattr(Camera,'read',disconnect)
    r=c.post('/api/vision/camera/capture',json={'index':1})
    assert r.status_code>=400 and 'disconnected' in r.text
    assert closed==[True,True]


def test_stored_image_replay_links_previous_evidence(client):
    c,s=client
    ref=upload(c,np.full((60,60,3),130,np.uint8))
    config={'product':'Replay fixture','reference_id':ref['id'],
            'locator':{'fixture_confirmed':True},
            'areas':[{'id':'a','name':'Area','region':{'x':0,'y':0,'width':1,'height':1},
                      'method':{'kind':'brightness','low':80,'high':180}}]}
    saved=c.post('/api/vision/recipes',json={'config':config}).json()
    payload={'recipe_id':saved['id'],'image_id':ref['id']}
    first=c.post('/api/vision/test',json=payload).json()
    second=c.post('/api/vision/test',json=payload).json()
    assert second['repeat_check']['previous_event_id']==first['id']
    assert second['repeat_check']['same_decision']
    assert second['repeat_check']['score_deltas']=={'a':0}


def test_health_does_not_report_legacy_model_as_production_ready(client,monkeypatch):
    import json
    import app
    _,service=client
    monkeypatch.setattr(app,'vision_service',service)
    response=json.loads(app.health().body)
    assert response['ok'] and response['inspection_strategy']=='area_recipe'
    assert not response['inspection_ready'] and not response['production_config_valid']


def test_missing_evidence_and_stop_failure_are_actionable(client):
    c,s=client
    assert c.get('/api/vision/evidence/invalid/maps').status_code==404
    assert c.post('/api/vision/review/event_'+'0'*32,json={'status':'needs_review'}).status_code==404
    original=s.runtime.stop
    def fail():raise RuntimeError('Camera remains locked')
    s.runtime.stop=fail
    try:
        response=c.post('/api/vision/production/stop',json={})
        assert response.status_code==409 and 'locked' in response.json()['detail']
    finally:s.runtime.stop=original


def test_shutdown_cancels_workers_even_when_camera_stop_fails(client):
    _,s=client
    called=[]
    class Worker:
        def cancel(self):called.append('training')
        def close(self):called.append('inference')
    s.training={'status':'running'}
    s.training_client=s.anomaly=Worker()
    original=s.runtime.stop
    def fail():raise RuntimeError('Camera remains locked')
    s.runtime.stop=fail
    try:
        with pytest.raises(RuntimeError):s.close()
        assert called==['training','inference'] and s.training['cancel_requested']
    finally:s.runtime.stop=original
