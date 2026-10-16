"""Real OS-process contention tests, without cameras, AI inference or project data."""
from pathlib import Path
import json
import subprocess
import sys
import threading
from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

WEB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WEB))
from inspection.ownership import ProjectLease, InspectionBusy
from inspection.api import InspectionService,make_router


@pytest.fixture
def owned_project(tmp_path):
    script='''import sys,os
sys.path.insert(0,sys.argv[1])
from inspection.ownership import ProjectLease
with ProjectLease(sys.argv[2]).hold('Terminal fixture'):
    print(os.getpid(),flush=True)
    sys.stdin.read()
'''
    root=tmp_path/'inspection_data'
    proc=subprocess.Popen([sys.executable,'-B','-u','-c',script,str(WEB),str(root)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        owner_pid=int(proc.stdout.readline().strip())
        yield tmp_path,proc,owner_pid
    finally:
        proc.communicate('',timeout=15)
        assert proc.returncode==0


def test_other_process_blocks_api_mutation_but_keeps_status_readable(owned_project):
    root,proc,owner_pid=owned_project
    training=root/'inspection_data/training/train_fixture.json'
    training.parent.mkdir()
    training.write_text(json.dumps({'id':'train_fixture','status':'running'}))
    service=InspectionService(root)
    assert json.loads(training.read_text())['status']=='running'
    app=FastAPI();app.include_router(make_router(service))
    with TestClient(app) as client:
        state=client.get('/api/vision/status').json()
        assert state['busy'] and state['external_runtime']['pid']==owner_pid
        assert client.post('/api/vision/recipes',json={}).status_code==409
        assert client.post('/api/vision/camera/capture',json={}).status_code==409
        assert client.post('/api/vision/activate',json={}).status_code==409
    assert service.store.records('recipes')==[] and service.store.production() is None
    service.close()


def test_terminal_refuses_before_loading_models_or_opening_camera(owned_project,monkeypatch):
    root,_,_=owned_project
    sys.path.insert(0,str(WEB.parent/'01_camera'))
    import airbag_webcam_detection as camera
    monkeypatch.setattr(camera,'PROJECT_ROOT',root)
    run=Mock()
    monkeypatch.setattr(camera,'_run_camera',run)
    with pytest.raises(InspectionBusy):camera.main(['--camera-index','1'])
    run.assert_not_called()


def test_existing_capture_utility_uses_same_ownership(owned_project,monkeypatch):
    root,_,_=owned_project
    sys.path.insert(0,str(WEB.parent/'01_camera'))
    import capture_training_images as capture
    monkeypatch.setattr(capture,'__file__',str(root/'01_camera/capture_training_images.py'))
    run=Mock()
    monkeypatch.setattr(capture,'_capture',run)
    with pytest.raises(InspectionBusy):capture.main()
    run.assert_not_called()


def test_worker_retains_lease_after_request_and_releases_on_error(tmp_path):
    lease=ProjectLease(tmp_path)
    entered=threading.Event();finish=threading.Event()
    def worker():
        entered.set();finish.wait(5)
    with lease.hold('request'):
        thread=lease.start_thread(worker)
    assert entered.wait(5)
    with pytest.raises(InspectionBusy):ProjectLease(tmp_path).acquire('other')
    finish.set();thread.join(5)
    assert not thread.is_alive()
    with ProjectLease(tmp_path).hold('next'):pass


def test_os_releases_crashed_owner_and_stale_metadata_is_not_busy(tmp_path):
    script="import sys,os;sys.path.insert(0,sys.argv[1]);from inspection.ownership import ProjectLease;x=ProjectLease(sys.argv[2]);x.acquire('crash fixture');os._exit(0)"
    subprocess.run([sys.executable,'-B','-c',script,str(WEB),str(tmp_path)],check=True,timeout=15)
    assert (tmp_path/'.runtime-owner.json').exists()
    lease=ProjectLease(tmp_path)
    assert lease.external_owner() is None
    with lease.hold('recovered'):assert lease.lock.is_locked
    assert not (tmp_path/'.runtime-owner.json').exists()


def test_worker_start_failure_releases_retained_lease(tmp_path,monkeypatch):
    lease=ProjectLease(tmp_path)
    monkeypatch.setattr(threading.Thread,'start',Mock(side_effect=RuntimeError('cannot start')))
    with pytest.raises(RuntimeError):lease.start_thread(lambda:None)
    with ProjectLease(tmp_path).hold('next'):pass


@pytest.mark.parametrize('url', [
    '/api/project/config','/api/models/activate','/api/jobs/job_missing/use',
    '/api/inspection/reference/train','/api/train/start',
])
def test_terminal_owner_blocks_existing_web_management(owned_project,monkeypatch,url):
    import app as web_app
    root,proc,pid=owned_project
    service=InspectionService(root)
    monkeypatch.setattr(web_app,'vision_service',service)
    writer=Mock(side_effect=AssertionError('Busy operation must not write configuration'))
    monkeypatch.setattr(web_app,'write_project_config',writer)
    try:
        client=TestClient(web_app.app,base_url='http://127.0.0.1')
        assert client.post(url,json={}).status_code==409
        assert service.runtime.ownership.external_owner()['pid']==pid
        writer.assert_not_called()
    finally:
        service.close()
