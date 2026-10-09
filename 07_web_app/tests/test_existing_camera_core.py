"""Existing terminal entry point + shared Core; synthetic contract tests, no camera/accuracy claims."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app
from inspection.store import Store,atomic_json
from inspection.pipeline import Pipeline
from inspection.runtime import Runtime
from inspection.validation import validate

detector=app.detector


@pytest.fixture
def camera_core(tmp_path,monkeypatch):
    store=Store(tmp_path/'inspection_data')
    frame=np.random.default_rng(8).integers(60,170,(120,160,3),dtype=np.uint8)
    def image(frame,label,group):
        return store.put_image(cv2.imencode('.png',frame)[1].tobytes(),label=label,group=group)
    reference=image(frame,'GOOD','reference')
    good=image(frame+1,'GOOD','independent-good')
    ng=image(frame//2,'NG','independent-ng')
    config={'product':'Synthetic integration fixture','reference_id':reference['id'],
        'camera':{'index':1,'width':160,'height':120},
        'locator':{'method':'template','region':{'x':.25,'y':.25,'width':.5,'height':.5}},
        'areas':[{'id':'a','name':'Surface','region':{'x':.1,'y':.1,'width':.8,'height':.8},
                  'method':{'kind':'brightness','low':80,'high':180}}],
        'decision':{'stable_frames':2,'release_frames':2,'release_ms':100},
        'validation':{'min_good':1,'min_ng':1}}
    saved=store.save_recipe(config)
    report=validate(store,Pipeline(store,saved),[good['id'],ng['id']])
    assert report['eligible']
    store.activate(saved['id'],report['id'])  # Temp test store only, never project Production.
    worker=Mock()
    runtime=Runtime(store,worker)
    monkeypatch.setattr(detector,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(detector,'CAMERA_WIDTH',160)
    monkeypatch.setattr(detector,'CAMERA_HEIGHT',120)
    monkeypatch.setattr(detector,'YOLO',Mock(side_effect=AssertionError('Legacy model must not load')))
    monkeypatch.setattr(detector.cv2,'imshow',Mock())
    monkeypatch.setattr(detector.cv2,'destroyAllWindows',Mock())
    args=SimpleNamespace(camera_index=1,no_brand=True)
    return store,frame,saved,report,runtime,args


def capture(frames):
    cap=Mock()
    cap.read.side_effect=[(True,f.copy()) for f in frames]+[(False,None)]
    return cap


def test_existing_window_uses_shared_decision_and_full_frame_evidence(camera_core,monkeypatch):
    store,frame,saved,report,runtime,args=camera_core
    blank=np.full_like(frame,120)
    cap=capture([frame+1,frame+1,blank,blank])
    opener=Mock(return_value=(cap,1))
    monkeypatch.setattr(detector,'open_available_camera',opener)
    monkeypatch.setattr(detector.cv2,'waitKey',Mock(side_effect=[-1,-1,-1,ord('q')]))
    clock=[10.0]
    def tick():
        clock[0]+=.1
        return clock[0]
    monkeypatch.setattr(detector.time,'monotonic',tick)
    observe=Mock(wraps=runtime.inspect_frame)
    monkeypatch.setattr(runtime,'inspect_frame',observe)
    detector.run_normal_reference_camera(args,{'inspection_strategy':'normal_reference'},runtime)
    assert observe.call_count==4 and opener.call_count==1
    rows=store.records('evidence')
    assert len(rows)==1 and rows[0]['complete'] and rows[0]['state']=='PASS'
    assert rows[0]['recipe_hash']==saved['hash'] and rows[0]['source']=='production'
    assert rows[0]['areas'][0]['source_polygon']
    np.testing.assert_array_equal(store.image(rows[0]['image_id']),frame+1)
    assert store.read('images',rows[0]['image_id'])['label']=='GOOD'  # Same existing fixture pixels; not relabelled.
    cap.release.assert_called_once()
    runtime.anomaly.close.assert_called_once()
    assert detector.cv2.imshow.call_count==4
    assert all(c.args[0]=='Generic OK/NG Inspection' for c in detector.cv2.imshow.call_args_list)
    assert [c.args for c in cap.set.call_args_list]==[(cv2.CAP_PROP_BUFFERSIZE,1),(cv2.CAP_PROP_FRAME_WIDTH,160),(cv2.CAP_PROP_FRAME_HEIGHT,120),(cv2.CAP_PROP_FPS,detector.CAMERA_FPS)]


def test_disconnect_records_incomplete_unable_and_releases(camera_core,monkeypatch):
    store,frame,_,_,runtime,args=camera_core
    cap=capture([frame+2,frame+2])
    monkeypatch.setattr(detector,'open_available_camera',lambda index:(cap,1))
    monkeypatch.setattr(detector.cv2,'waitKey',lambda delay:-1)
    with pytest.raises(RuntimeError,match='disconnected'):
        detector.run_normal_reference_camera(args,{},runtime)
    row=store.records('evidence')[0]
    assert row['state']=='UNABLE_TO_INSPECT' and not row['complete']
    assert store.read('images',row['image_id'])['label']=='UNKNOWN'
    np.testing.assert_array_equal(store.image(row['image_id']),frame+2)
    cap.release.assert_called_once()
    runtime.anomaly.close.assert_called_once()


@pytest.mark.parametrize('failure',['report','source','index','resolution','fallback'])
def test_invalid_core_configuration_never_falls_back_or_changes_settings(camera_core,monkeypatch,failure):
    store,frame,saved,report,runtime,args=camera_core
    cap=capture([frame])
    opener=Mock(return_value=(cap,2 if failure=='fallback' else 1))
    monkeypatch.setattr(detector,'open_available_camera',opener)
    if failure=='report':
        report['metrics']['good_count']=999
        atomic_json(store.path('tests',report['id']),report)
    elif failure=='source':
        ref=store.read('images',saved['config']['reference_id'])
        ref['group']='changed-after-activation'
        atomic_json(store.path('images',ref['id']),ref)
    elif failure=='index':args.camera_index=0
    elif failure=='resolution':monkeypatch.setattr(detector,'CAMERA_WIDTH',640)
    with pytest.raises(ValueError):
        detector.run_normal_reference_camera(args,{},runtime)
    if failure=='fallback':
        cap.release.assert_called_once()
    else:opener.assert_not_called()
    cap.set.assert_not_called()
    cap.read.assert_not_called()
    runtime.anomaly.close.assert_called_once()


def test_no_activated_core_recipe_keeps_original_entry_point(monkeypatch,tmp_path):
    args=SimpleNamespace(camera_index=1,no_brand=True)
    config={'inspection_strategy':'normal_reference'}
    monkeypatch.setattr(detector,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(detector,'parse_args',lambda:args)
    monkeypatch.setattr(detector,'load_project_detection_config',lambda:config)
    original=Mock()
    monkeypatch.setattr(detector,'run_normal_reference_camera',original)
    detector.main()
    original.assert_called_once_with(args,config)


def test_main_selects_only_explicit_production_without_writing_config(camera_core,monkeypatch):
    store,frame,saved,_,_,args=camera_core
    monkeypatch.setattr(detector,'parse_args',lambda:args)
    config={'inspection_strategy':'normal_reference','active_brand_model':'keep-this-setting'}
    monkeypatch.setattr(detector,'load_project_detection_config',lambda:config)
    selected=Mock()
    monkeypatch.setattr(detector,'run_normal_reference_camera',selected)
    before=(store.root/'production.json').read_bytes()
    detector.main()
    runtime=selected.call_args.kwargs['core_runtime']
    assert runtime.store.production()['recipe_id']==saved['id']
    assert (store.root/'production.json').read_bytes()==before
    assert config=={'inspection_strategy':'normal_reference','active_brand_model':'keep-this-setting'}
    assert runtime.thread is None and runtime.capture_pump is None


def test_existing_legacy_window_and_sample_keys_remain_until_replacement(camera_core,monkeypatch,tmp_path):
    import json
    import deep_patch_reference
    import inspection_contract
    _,frame,_,_,_,args=camera_core
    model=tmp_path/'legacy.npz'
    model.write_bytes(b'test-only-placeholder')
    config={'inspection_strategy':'normal_reference','normal_reference':{'active_model':str(model)}}
    inspector=Mock(threshold=.5)
    inspector.draw_result.side_effect=lambda frame,*args,**kwargs:frame.copy()
    loader=Mock(return_value=object())
    monkeypatch.setattr(deep_patch_reference,'load_inspection_reference',loader)
    monkeypatch.setattr(inspection_contract,'RecipeBoundInspector',lambda *args:inspector)
    monkeypatch.setattr(detector,'load_primary_object_locator',lambda config:(None,None))
    monkeypatch.setattr(detector,'analyze_primary_object',lambda *args:None)
    cap=capture([frame,frame])
    monkeypatch.setattr(detector,'open_available_camera',lambda index:(cap,1))
    monkeypatch.setattr(detector.cv2,'waitKey',Mock(side_effect=[ord('s'),ord('q')]))
    detector.run_normal_reference_camera(args,config)
    loader.assert_called_once_with(model)
    assert detector.cv2.imshow.call_count==2
    assert detector.cv2.imshow.call_args.args[0]=='Generic OK/NG Inspection'
    samples=list((tmp_path/'work'/'live_inspection_samples').glob('*.json'))
    assert len(samples)==1
    sample=json.loads(samples[0].read_text(encoding='utf-8'))
    assert sample['expected_decision'] is None and sample['model']==str(model)
    np.testing.assert_array_equal(cv2.imread(str(samples[0].with_suffix('.png'))),frame)
    cap.release.assert_called_once()


def test_web_runtime_keeps_same_shared_part_cycle_and_raw_evidence(camera_core,monkeypatch):
    import inspection.runtime as shared
    store,frame,saved,_,runtime,_=camera_core
    frames=iter([frame+1,frame+1,np.full_like(frame,120),np.full_like(frame,120)])
    clock=[10.0]
    def tick():
        clock[0]+=.1
        return clock[0]
    monkeypatch.setattr(shared.time,'monotonic',tick)
    camera=Mock()
    class Pump:
        def __init__(self,camera):self.camera=camera
        def next(self,sequence):
            if sequence==3:runtime.stop_event.set()
            return sequence+1,next(frames),tick(),'fixture-time'
        def close(self):self.camera.close()
    monkeypatch.setattr(shared,'FramePump',Pump)
    runtime.loop(camera,runtime.production_pipeline())
    rows=store.records('evidence')
    assert len(rows)==1 and rows[0]['complete'] and rows[0]['state']=='PASS'
    assert rows[0]['recipe_hash']==saved['hash']
    np.testing.assert_array_equal(store.image(rows[0]['image_id']),frame+1)
    assert runtime.state['last_event']==rows[0]['id'] and runtime.frame is None
    camera.close.assert_called_once()
