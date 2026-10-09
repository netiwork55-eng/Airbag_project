"""Live capture lifecycle and software-only parity; no physical camera is opened."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import time
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection import runtime as shared
from inspection.ownership import ProjectLease,InspectionBusy
from inspection.schema import Camera,DecisionSettings
from inspection.store import Store


def test_blocked_capture_keeps_os_lease_after_close_timeout(tmp_path,monkeypatch):
    lease=ProjectLease(tmp_path);entered=threading.Event();finish=threading.Event()
    camera=Mock()
    def read():
        entered.set();assert finish.wait(5)
        return np.zeros((16,16,3),np.uint8)
    camera.read.side_effect=read
    pump=None
    try:
        with lease.hold('caller'):
            pump=shared.FramePump(camera,ownership=lease)
            assert entered.wait(5)
        join=pump.thread.join
        monkeypatch.setattr(pump.thread,'join',lambda timeout:None)
        with pytest.raises(RuntimeError,match='did not stop'):pump.close()
        assert pump.thread.is_alive()
        with pytest.raises(InspectionBusy):ProjectLease(tmp_path).acquire('another process')
    finally:
        finish.set()
        if pump:join(5)
    assert not pump.thread.is_alive()
    camera.close.assert_called_once()
    with ProjectLease(tmp_path).hold('next camera'):pass


def test_capture_thread_start_failure_releases_camera_and_lease(tmp_path,monkeypatch):
    camera=Mock();lease=ProjectLease(tmp_path)
    monkeypatch.setattr(threading.Thread,'start',Mock(side_effect=RuntimeError('thread failed')))
    with pytest.raises(RuntimeError,match='thread failed'):shared.FramePump(camera,ownership=lease)
    camera.close.assert_called_once()
    with ProjectLease(tmp_path).hold('next camera'):pass


def test_capture_setup_refuses_external_owner_before_opening_device(tmp_path,monkeypatch):
    runtime=shared.Runtime(Store(tmp_path),Mock())
    opener=Mock();monkeypatch.setattr(shared,'OpenCVCamera',opener)
    with ProjectLease(tmp_path).hold('external camera'):
        with pytest.raises(InspectionBusy):runtime.open_capture(SimpleNamespace(recipe=SimpleNamespace(camera=Camera())))
    opener.assert_not_called()


def test_metadata_failure_closes_camera_without_starting_reader(tmp_path,monkeypatch):
    runtime=shared.Runtime(Store(tmp_path),Mock());camera=Mock();camera.metadata.side_effect=RuntimeError('metadata failed')
    monkeypatch.setattr(shared,'OpenCVCamera',lambda *a,**k:camera)
    reader=Mock();monkeypatch.setattr(shared,'FramePump',reader)
    with pytest.raises(RuntimeError,match='metadata failed'):runtime.open_capture(SimpleNamespace(recipe=SimpleNamespace(camera=Camera())))
    reader.assert_not_called();camera.close.assert_called_once()
    with ProjectLease(tmp_path).hold('next camera'):pass


def test_inspection_worker_start_failure_closes_started_capture(tmp_path,monkeypatch):
    runtime=shared.Runtime(Store(tmp_path),Mock());pump=Mock();pump.thread.is_alive.return_value=False
    monkeypatch.setattr(runtime,'production_pipeline',lambda:SimpleNamespace(saved={'id':'recipe','hash':'hash'}))
    monkeypatch.setattr(runtime,'open_capture',lambda p:pump)
    monkeypatch.setattr(runtime.ownership,'start_thread',Mock(side_effect=RuntimeError('worker failed')))
    with pytest.raises(RuntimeError,match='worker failed'):runtime.start()
    pump.close.assert_called_once()
    assert runtime.state['state']=='UNABLE_TO_INSPECT' and not runtime.state['running']
    with ProjectLease(tmp_path).hold('next camera'):pass


@pytest.mark.parametrize('action',['stop','reopen'])
def test_release_failure_cannot_be_hidden_by_stop_or_reopen(tmp_path,monkeypatch,action):
    runtime=shared.Runtime(Store(tmp_path),Mock())
    camera=Mock()
    camera.read.side_effect=RuntimeError('disconnected')
    camera.close.side_effect=RuntimeError('driver release failed')
    pump=shared.FramePump(camera,ownership=runtime.ownership)
    pump.thread.join(5)
    assert not pump.thread.is_alive()
    runtime.capture_pump=pump
    pipeline=SimpleNamespace(recipe=SimpleNamespace(camera=Camera(),decision=DecisionSettings()))
    runtime.loop(pump,pipeline)
    assert 'Camera release failed:' in runtime.snapshot()['reasons'][0]
    opener=Mock(side_effect=AssertionError('Must not reopen after release failure'))
    monkeypatch.setattr(shared,'OpenCVCamera',opener)
    with pytest.raises(RuntimeError,match='Camera release failed:'):
        if action=='stop':runtime.stop()
        else:runtime.open_capture(pipeline)
    assert runtime.snapshot()['state']=='UNABLE_TO_INSPECT'
    assert 'Camera release failed:' in runtime.snapshot()['reasons'][0]
    opener.assert_not_called()


@pytest.mark.parametrize('source',['production','candidate'])
def test_shared_live_stale_frame_never_passes_and_evidence_retains_capture(tmp_path,source):
    runtime=shared.Runtime(Store(tmp_path),Mock(),source=source)
    runtime.capture_metadata={'actual':{'driver':'opencv','index':1,'width':40,'height':30},'timing_scope':'Host read completion'}
    pipeline=SimpleNamespace(recipe=SimpleNamespace(camera=Camera(max_frame_age_ms=100),decision=DecisionSettings()),
        run=lambda frame:({'state':'PASS','reasons':[],'areas':[],'quality':{'brightness':120}},{}))
    cycle=runtime.make_cycle(pipeline);frame=np.full((30,40,3),120,np.uint8)
    for seq in range(3):
        state,result,_=runtime.inspect_frame(pipeline,cycle,frame,time.monotonic()-1,'fixture',seq)
        assert result['state']=='UNABLE_TO_INSPECT' and state['state']=='UNABLE_TO_INSPECT'
    runtime.finish_cycle(cycle)
    record=runtime.store.records('evidence')[0]
    assert record['source']==source and record['state']=='UNABLE_TO_INSPECT' and not record['complete']
    assert record['capture']['actual']['index']==1 and record['frame_age_ms']>=1000
    assert runtime.store.read('images',record['image_id'])['label']=='UNKNOWN'
