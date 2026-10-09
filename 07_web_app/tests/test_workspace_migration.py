"""Phase 2 integration exercises existing endpoints with temporary data only."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app
from inspection.store import Store

@pytest.fixture
def workspace(tmp_path,monkeypatch):
    for name,path in [('AIRBAG_EXPORTS_DIR','exports'),('DELETED_DATASETS_DIR','trash'),('UPLOAD_STAGING_DIR','staging'),('INSPECTION_REVIEW_DIR','reviews'),('JOB_DIR','jobs')]:
        directory=tmp_path/path;directory.mkdir();monkeypatch.setattr(app,name,directory)
    monkeypatch.setattr(app,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(app,'WEB_APP_DIR',tmp_path/'web')
    monkeypatch.setattr(app,'read_project_config',lambda:{})
    monkeypatch.setattr(app,'training_is_running',lambda:False)
    monkeypatch.setattr(app,'clear_analysis_cache',lambda:None)
    monkeypatch.setattr(app.vision_service,'store',Store(tmp_path/'inspection_data'))
    monkeypatch.setattr(app.vision_service,'busy',lambda:False)
    return TestClient(app.app,base_url='http://127.0.0.1'),tmp_path

def image_bytes():return cv2.imencode('.png',np.full((32,48,3),120,np.uint8))[1].tobytes()
def upload(c,name='part'):
    assert c.post('/api/dataset-folders',json={'name':name}).status_code==200
    response=c.post('/api/dataset-folders/upload',data={'folder':name},files=[('files',('sample.png',image_bytes(),'image/png'))])
    assert response.status_code==200,response.text
    return response.json()

def test_upload_metadata_preview_undo_restore(workspace):
    c,root=workspace;record=upload(c)
    preview=c.get('/api/dataset-folders/part/preview').json()['preview']
    assert preview['images']==1
    assert c.get(preview['samples'][0]['url']).content==image_bytes()
    assert preview['folder']['meta']['review_status']=='needs_info'
    assert not (root/'exports/part/dataset_meta.json').exists(), 'GET must not write metadata'
    assert c.post('/api/dataset-folders/part/metadata',json={'dataset_name':'Part','dataset_type':'Known-good Reference','notes':'description only'}).status_code==200
    meta=json.loads((root/'exports/part/dataset_meta.json').read_text())
    assert 'confirmed_at' not in meta and 'label' not in meta
    before=(root/'exports/part/dataset_meta.json').read_bytes()
    latest=c.get('/api/dataset-folders/part/latest-upload').json()
    assert latest['can_undo'] and latest['upload_id']==record['latest_upload']['upload_id']
    result=c.post('/api/dataset-folders/undo-upload',json={'folder':'part','upload_id':latest['upload_id']})
    assert result.status_code==200,result.text
    assert not list((root/'exports/part').rglob('*.png'))
    assert (root/'exports/part/dataset_meta.json').read_bytes()==before
    trash=c.get('/api/deleted-dataset-folders').json()['folders']
    assert len(trash)==1
    restored=c.post('/api/deleted-dataset-folders/'+trash[0]['id']+'/restore',json={})
    assert restored.status_code==200,restored.text
    restored_path=Path(restored.json()['restored_path'])
    assert next(restored_path.rglob('*.png')).read_bytes()==image_bytes()
    assert c.delete('/api/dataset-folders/part').status_code==200
    trash=c.get('/api/deleted-dataset-folders').json()['folders']
    assert c.post('/api/deleted-dataset-folders/'+trash[0]['id']+'/restore',json={}).status_code==200
    assert (root/'exports/part/dataset_meta.json').read_bytes()==before

@pytest.mark.parametrize('tamper',['token','file','client_paths','old_manifest'])
def test_undo_rejects_stale_or_changed_batch_without_data_loss(workspace,tamper):
    c,root=workspace;record=upload(c);folder=root/'exports/part'
    token=record['latest_upload']['upload_id'];picture=next(folder.rglob('*.png'))
    payload={'folder':'part','upload_id':token}
    if tamper=='token':payload['upload_id']='stale'
    elif tamper=='file':picture.write_bytes(b'edited after upload')
    elif tamper=='old_manifest':(folder/app.LATEST_UPLOAD_MANIFEST).write_text('{}')
    else:payload={'folder':'part','installed_paths':[str(folder)],'remove_folder':True}
    before={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    assert c.post('/api/dataset-folders/undo-upload',json=payload).status_code==400
    assert before=={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}

def test_read_metadata_normalizes_in_memory_only(workspace):
    c,root=workspace;folder=root/'exports/old';folder.mkdir()
    meta=folder/'dataset_meta.json';meta.write_text(json.dumps({'dataset_type':'Detection','review_status':'reviewed','auto_filled':False}))
    before=meta.read_bytes()
    assert c.get('/api/dataset-folders').json()['folders'][0]['meta']['dataset_type']=='Known-good Reference'
    assert meta.read_bytes()==before

def test_legacy_machine_ng_and_human_review_remain_separate(workspace):
    c,root=workspace;folder=root/'reviews/ng_old';folder.mkdir()
    p=folder/'review.json';p.write_text(json.dumps({'id':'ng_old','decision':'NG','status':'pending','anomaly_score':2,'threshold':1}))
    before=p.read_bytes()
    row=c.get('/api/inspection/reviews').json()['reviews'][0]
    assert row['decision']=='NG' and row['status']=='pending' and 'reviewed_at' not in row
    assert p.read_bytes()==before
    result=c.post('/api/inspection/reviews/ng_old',json={'status':'false_alarm'}).json()['review']
    assert result['decision']=='NG' and result['status']=='false_alarm' and result['reviewed_at']

def test_read_old_jobs_models_and_test_reports(workspace,monkeypatch):
    c,root=workspace
    (root/'jobs/job_broken.json').write_text('{broken')
    (root/'jobs/job_old.json').write_text(json.dumps({'id':'job_old','name':'Old job','created_at':'2026','settings':{'normal_reference':{}}}))
    folder=root/'web/logs/model_tests';folder.mkdir(parents=True)
    p=folder/('test_'+'a'*32+'.json');p.write_text(json.dumps({'id':p.stem,'production_ready':False,'labels_source':'user','models':[{'rows':[{'decision':'NG','expected':'UNKNOWN'}]}]}))
    monkeypatch.setattr(app,'model_registry',lambda:[{'name':'old.pt','path':'models/old.pt','active':True}])
    model={'id':'model_'+'b'*32,'threshold':1,'algorithm':'padim'}
    app.vision_service.store.path('models',model['id']).write_text(json.dumps(model))
    response=c.get('/api/jobs').json()
    assert any(j.get('error') for j in response['jobs']) and any(j.get('name')=='Old job' for j in response['jobs'])
    assert response['test_reports']==[{'id':p.stem}]
    assert c.get('/api/jobs/test-report/'+p.stem).json()['models'][0]['rows'][0]['expected']=='UNKNOWN'
    registry=c.get('/api/models/registry').json()
    assert registry['inspection_models']==[model] and registry['models'][0]['name']=='old.pt'

def test_workspace_html_and_retired_routes(workspace):
    c,root=workspace
    response=c.get('/inspection');assert response.status_code==200
    from html.parser import HTMLParser
    class Ids(HTMLParser):
        def __init__(self):super().__init__();self.ids=[]
        def handle_starttag(self,tag,attrs):
            self.ids += [v for k,v in attrs if k=='id']
    parser=Ids();parser.feed(response.text)
    assert len(parser.ids)==len(set(parser.ids))
    for id in ['data-upload','data-undo','data-trash-panel','job-library','legacy-history','legacy-job-test']:
        assert id in parser.ids
    assert 'href="/engineering"' not in response.text
    for url in ['/engineering','/api/datasets/summary','/api/train/recommended-settings']:
        assert c.get(url).status_code==404
    assert c.post('/api/datasets/upload').status_code==404
    assert c.get('/jobs',follow_redirects=False).headers['location']=='/inspection#test'

def test_uploads_in_same_second_do_not_overwrite_and_undo_preserves_prior_batch(workspace,monkeypatch):
    c,root=workspace;monkeypatch.setattr(app,'timestamp_name',lambda:'same_second')
    first=upload(c)
    original=next((root/'exports/part').rglob('*.png'))
    newer=cv2.imencode('.png',np.full((32,48,3),200,np.uint8))[1].tobytes()
    r=c.post('/api/dataset-folders/upload',data={'folder':'part'},files=[('files',('sample.png',newer,'image/png'))])
    assert r.status_code==200,r.text
    assert original.read_bytes()==image_bytes()
    assert len(list((root/'exports/part').rglob('*.png')))==2
    result=c.post('/api/dataset-folders/undo-upload',json={'folder':'part','upload_id':r.json()['latest_upload']['upload_id']})
    assert result.status_code==200,result.text
    assert original.read_bytes()==image_bytes() and len(list((root/'exports/part').rglob('*.png')))==1


def test_undo_checks_entire_batch_before_moving_labels(workspace):
    c,root=workspace;folder=root/'exports/labelled';folder.mkdir();batch=folder/'batch';batch.mkdir()
    (batch/'sample.png').write_bytes(image_bytes());label=batch/'sample.txt';label.write_text('0 .5 .5 .1 .1')
    manifest=app.record_latest_dataset_upload(folder,[str(batch)])
    label.write_text('changed annotation')
    response=c.post('/api/dataset-folders/undo-upload',json={'folder':'labelled','upload_id':manifest['upload_id']})
    assert response.status_code==400 and (batch/'sample.png').exists() and label.read_text()=='changed annotation'
    assert not list((root/'trash').iterdir())
