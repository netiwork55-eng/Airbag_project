"""Synthetic geometry/failure regressions only; these do not validate defect accuracy."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspection.store import Store, atomic_json, file_hash
from inspection.pipeline import Pipeline, input_signature
from inspection.schema import Recipe
from inspection.runtime import Cycle, evidence


@pytest.fixture
def scene(tmp_path):
    store = Store(tmp_path)
    image = np.random.default_rng(42).integers(50, 200, (240,320,3), dtype=np.uint8)
    ref = store.put_image(cv2.imencode('.png',image)[1].tobytes(),label='GOOD')
    config = {'product':'Geometry fixture', 'reference_id':ref['id'],
              'locator':{'method':'template','region':{'x':.3,'y':.3,'width':.3,'height':.3}},
              'areas':[{'id':'surface','name':'Surface','region':{'x':.2,'y':.2,'width':.6,'height':.6},
                        'method':{'kind':'presence','threshold':.9,'review_margin':.02}}]}
    return store,image,config


def test_full_frame_shift_roi_cycle_and_evidence(scene):
    store,image,config = scene
    saved = store.save_recipe(config)
    shifted = cv2.warpAffine(image,np.float32([[1,0,12],[0,1,7]]),(320,240),borderValue=(120,120,120))
    result,maps = Pipeline(store,saved).run(shifted)
    assert result['state']=='PASS'
    area = result['areas'][0]
    np.testing.assert_allclose(area['source_polygon'],[[76,55],[268,55],[268,199],[76,199]],atol=.01)
    assert area['settings']['threshold']==.9 and area['settings']['review_margin']==.02
    cycle = Cycle(stable_frames=2,release_frames=2,release_seconds=.1)
    cycle.observe(result,shifted,maps)
    cycle.observe(result,shifted,maps)
    assert cycle.observe({},shifted,{},absent=True,now_mono=1) is None
    closed = cycle.observe({},shifted,{},absent=True,now_mono=1.2)
    event = evidence(store,closed['result'],closed['image'],closed['maps'],source='test')
    recorded = store.read('evidence',event['id'])
    np.testing.assert_array_equal(store.image(recorded['image_id']),shifted)
    assert recorded['recipe_hash']==saved['hash']
    assert recorded['areas'][0]['source_polygon']==area['source_polygon']


@pytest.mark.parametrize('mode',['dark','blur','dimensions','missing','clipped'])
def test_failed_gate_retains_measurements_and_cannot_pass(scene,mode):
    store,image,config = scene
    frame = image.copy()
    if mode=='dark': frame[:]=0
    elif mode=='blur':
        config['quality']={'min_blur':100}
        frame=cv2.GaussianBlur(frame,(31,31),5)
    elif mode=='dimensions': frame=frame[:120]
    elif mode=='missing': frame[:]=120
    else:
        config['areas'][0]['region']={'x':0,'y':0,'width':1,'height':1}
        frame=cv2.warpAffine(frame,np.float32([[1,0,12],[0,1,7]]),(320,240))
    result,_ = Pipeline(store,store.save_recipe(config)).run(frame)
    assert result['state']=='UNABLE_TO_INSPECT'
    assert result['quality']['brightness']==pytest.approx(float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).mean()))
    assert 'sharpness' in result['quality'] and 'limits' in result['quality']
    assert result['areas'][0]['state']=='UNABLE_TO_INSPECT'
    assert result['areas'][0]['settings']['threshold']==.9
    assert result['failed_stage']==('image_quality' if mode in ('dark','blur') else 'inspection_areas' if mode=='clipped' else 'alignment')


def test_existing_feature_locator_rotation_maps_roi_to_full_source(scene):
    store,image,config = scene
    config['locator']={'method':'features','min_score':.7}
    transform=cv2.getRotationMatrix2D((160,120),4,1)
    transform[:,2]+=[5,3]
    moved=cv2.warpAffine(image,transform,(320,240),borderValue=(120,120,120))
    config['areas'][0]['method']={'kind':'brightness','low':50,'high':200}
    result,_=Pipeline(store,store.save_recipe(config)).run(moved)
    assert result['state']=='PASS'
    expected=cv2.transform(np.float32([[[64,48],[256,48],[256,192],[64,192]]]),transform)[0]
    np.testing.assert_allclose(result['areas'][0]['source_polygon'],expected,atol=2)


def test_anomaly_receives_aligned_roi_and_setup_mismatch_never_passes(scene):
    store,image,config=scene
    key='model_'+'a'*32
    path=store.path('models',key,'.pt');path.write_bytes(b'isolated test double, not a real AI model')
    config['areas'][0]['method']={'kind':'anomaly','model_id':key,'threshold':2,'review_margin':.1}
    # Workers calculate signatures from the persisted, normalized recipe.
    recipe=Recipe.model_validate(Recipe.model_validate(config).model_dump())
    metadata={'id':key,'sha256':file_hash(path),'framework':'anomalib',
              'input_signature':input_signature(recipe,recipe.areas[0])}
    atomic_json(store.path('models',key),metadata)
    class Predictor:
        calls=0
        def predict(self,model_id,crop):
            self.calls+=1
            assert model_id==key
            np.testing.assert_array_equal(crop,image[48:192,64:256])
            return {'score':1.0,'map':np.zeros(crop.shape[:2],np.float32)}
    worker=Predictor()
    shifted=cv2.warpAffine(image,np.float32([[1,0,12],[0,1,7]]),(320,240))
    result,_=Pipeline(store,store.save_recipe(config),worker).run(shifted)
    assert result['state']=='PASS' and worker.calls==1, result['reasons']
    config['areas'][0]['region']['width']=.5
    result,_=Pipeline(store,store.save_recipe(config),worker).run(shifted)
    assert result['state']=='UNABLE_TO_INSPECT' and worker.calls==1
    assert 'different image setup or area' in result['reasons'][0]


def test_capture_keeps_full_dimensions_group_and_preview(scene, tmp_path, monkeypatch):
    pytest.importorskip('httpx2')
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from inspection.api import InspectionService, make_router
    import inspection.api as api
    _,image,_=scene
    closed=[]
    class Camera:
        def __init__(self,config):pass
        def read(self):return image
        def close(self):closed.append(True)
    monkeypatch.setattr(api,'OpenCVCamera',Camera)
    service=InspectionService(tmp_path/'http')
    app=FastAPI();app.include_router(make_router(service))
    try:
        with TestClient(app) as client:
            response=client.post('/api/vision/camera/capture',json={'capture_group':'same-physical-part'})
            assert response.status_code==200
            item=response.json()
            assert (item['width'],item['height'],item['group'])==(320,240,'same-physical-part')
            assert closed==[True] and item['label']=='UNKNOWN'
            client.post('/api/vision/images/'+item['id']+'/confirm',json={'label':'GOOD'})
            config={'product':'Full frame HTTP test','reference_id':item['id'],
                    'areas':[{'id':'surface','name':'Surface','region':{'x':.2,'y':.2,'width':.6,'height':.6}}]}
            saved=client.post('/api/vision/recipes',json={'config':config}).json()
            event=client.post('/api/vision/test',json={'recipe_id':saved['id'],'image_id':item['id']}).json()
            preview=client.get('/api/vision/evidence/'+event['id']+'/preview')
            assert preview.status_code==200
            rendered=cv2.imdecode(np.frombuffer(preview.content,np.uint8),cv2.IMREAD_COLOR)
            assert rendered.shape==image.shape
            assert not np.array_equal(rendered,image)
            assert client.post('/api/vision/production/start',json={}).status_code==400
    finally:
        service.close()
