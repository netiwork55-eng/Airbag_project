"""Contracts and failure paths for labelled inspection, annotation import and metrics."""
import copy
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection.annotations import import_coco,read_dataset,area_sample
from inspection.schema import Recipe,Method
from inspection.pipeline import detection_result
from inspection.store import Store,atomic_json
from inspection.detection_metrics import matched_counts


@pytest.fixture
def sample(tmp_path):
    store=Store(tmp_path);image=np.full((80,100,3),90,np.uint8)
    image[20:40,25:55]=200
    item=store.put_image(cv2.imencode('.png',image)[1].tobytes(),'part.png',label='GOOD',group='part-a')
    coco={'images':[{'id':1,'file_name':'part.png','width':100,'height':80}],
        'categories':[{'id':3,'name':'clip'}],
        'annotations':[{'id':1,'image_id':1,'category_id':3,'bbox':[25,20,30,20],
                        'segmentation':[[25,20,55,20,55,40,25,40]]}]}
    return store,image,item,coco


def test_cvat_import_requires_review_preserves_polygons_and_integrity(sample):
    store,image,item,coco=sample
    with pytest.raises(ValueError,match='Confirm'):import_coco(store,coco)
    d=import_coco(store,coco,reviewed=True)
    assert d['labels']==['clip'] and d['records'][0]['image_id']==item['id']
    assert read_dataset(store,d['id'])['records'][0]['objects'][0]['polygons']
    d['records'][0]['objects']=[];atomic_json(store.path('datasets',d['id']),d)
    with pytest.raises(ValueError,match='changed'):read_dataset(store,d['id'])


def test_invalid_coco_bounds_and_rle_are_not_silently_ignored(sample):
    store,_,_,coco=sample
    bad=copy.deepcopy(coco);bad['annotations'][0]['bbox']=[90,20,30,20]
    with pytest.raises(ValueError,match='outside'):import_coco(store,bad,reviewed=True)
    bad=copy.deepcopy(coco);bad['annotations'][0]['segmentation']={'counts':'unsupported'}
    with pytest.raises(ValueError,match='RLE'):import_coco(store,bad,reviewed=True)


def test_segmentation_alignment_and_area_clipping(sample):
    store,image,item,coco=sample;d=import_coco(store,coco,reviewed=True)
    recipe=Recipe(product='Clip',reference_id=item['id'],areas=[{'id':'a','name':'Clip',
        'region':{'x':.1,'y':.1,'width':.7,'height':.7},
        'method':{'kind':'segmentation','threshold':.5,'review_margin':.1,'target_labels':['clip'],'min_count':1,'max_count':1}}])
    roi,objects=area_sample(store,recipe,recipe.areas[0],d['records'][0],image)
    assert objects[0]['box']==[15,12,45,32] and objects[0]['mask'].any()
    recipe.areas[0].region.width=.2
    with pytest.raises(ValueError,match='crosses'):area_sample(store,recipe,recipe.areas[0],d['records'][0],image)


def test_missing_component_and_uncertain_confidence_are_distinct():
    image=np.zeros((80,100,3),np.uint8)
    method=Method(kind='detection',threshold=.5,review_margin=.1,target_labels=['clip'],min_count=1,max_count=1)
    def check(scores):return detection_result({'detections':[{'label':'clip','score':s,'box':[10,10,20,20]} for s in scores]},method,image)[0]
    assert check([])=='NG' and check([.9])=='PASS' and check([.5])=='REVIEW'
    assert check([.9,.5])=='REVIEW' and check([.9,.95])=='NG'
    method.min_count=method.max_count=0
    assert check([])=='PASS' and check([.5])=='REVIEW' and check([.9])=='NG'
    with pytest.raises(ValueError):check([float('nan')])


def test_metrics_match_labels_and_do_not_double_count():
    p={'label':'defect','score':.9,'box':[10,10,30,30]};t={'label':'defect','box':[10,10,30,30]}
    assert matched_counts([p,p],[t],(40,40))==(1,1,0)
    assert matched_counts([{**p,'label':'other'}],[t],(40,40))==(0,1,1)
    assert matched_counts([],[t],(40,40))==(0,0,1)


def test_cvat_ordered_filename_resolves_original_image(sample):
    store,_,item,coco=sample
    coco['images'][0]['file_name']='000003_'+item['id']+'.png'
    assert import_coco(store,coco,reviewed=True)['records'][0]['image_id']==item['id']


def test_mask_metrics_preserve_holes():
    import base64
    mask=np.ones((30,30),np.uint8);mask[3:27,3:27]=0
    encoded=base64.b64encode(cv2.imencode('.png',mask)[1]).decode()
    p={'label':'clip','score':.9,'box':[0,0,30,30],'mask_png':encoded,
       'polygons':[[[0,0],[29,0],[29,29],[0,29]]]}
    truth={'label':'clip','box':[0,0,30,30],'mask':mask}
    assert matched_counts([p],[truth],(30,30),True)==(1,0,0)


def test_two_clips_cannot_substitute_for_a_missing_screw():
    with pytest.raises(ValueError,match='one label'):
        Method(kind='detection',threshold=.5,review_margin=.1,target_labels=['clip','screw'],min_count=2,max_count=2)


def test_evidence_pagination_reaches_old_records_without_duplicates(tmp_path):
    store=Store(tmp_path)
    import uuid
    for i in range(503):
        key='event_'+uuid.uuid4().hex
        # Synthetic pagination records need no image/model artifacts.
        store.path('evidence',key).write_text(__import__('json').dumps({'id':key,'state':'NG','index':i}))
    seen=[];cursor=None
    while True:
        page=store.evidence_page(100,cursor)
        seen.extend(e['id'] for e in page['events']);cursor=page['next_cursor']
        if not cursor:break
    assert len(seen)==503 and len(set(seen))==503
    with pytest.raises(ValueError):store.evidence_page(100,'../outside')
