"""Explicit object precision/recall at IoU 0.5, not generic image accuracy or COCO mAP."""
import cv2
import base64
import numpy as np
from .annotations import area_sample,read_dataset


def overlap(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    size=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(size.prod());union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)
    return intersection/union if union>0 else 0.


def matched_counts(predictions,truth,shape,segmentation=False):
    used=set();tp=0
    for p in sorted(predictions,key=lambda p:p['score'],reverse=True):
        candidates=[]
        for index,t in enumerate(truth):
            if index in used or p['label']!=t['label']:continue
            if segmentation:
                if p.get('mask_png'):
                    mask=cv2.imdecode(np.frombuffer(base64.b64decode(p['mask_png'],validate=True),np.uint8),cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape!=shape:raise ValueError('Segmentation mask dimensions do not match the area')
                else:
                    mask=np.zeros(shape,np.uint8)
                    polygons=[np.rint(polygon).astype(np.int32) for polygon in p.get('polygons',[]) if len(polygon)>=3]
                    if polygons:cv2.fillPoly(mask,polygons,1)
                union=np.logical_or(mask,t['mask']).sum()
                score=float(np.logical_and(mask,t['mask']).sum()/union) if union else 0.
            else:score=overlap(p['box'],t['box'])
            candidates.append((score,index))
        if candidates:
            score,index=max(candidates)
            if score>=.5:tp+=1;used.add(index)
    return tp,len(predictions)-tp,len(truth)-tp


def evaluate(store,pipeline,rows,dataset_id):
    areas=[a for a in pipeline.recipe.areas if a.method.kind in ('detection','segmentation')]
    if not areas:return {},[]
    if not dataset_id:
        return {},['Select independent reviewed annotations to validate detection/segmentation'] if any(a.required for a in areas) else []
    data=read_dataset(store,dataset_id);records={r['image_id']:r for r in data['records']}
    output={};blockers=[]
    for area in areas:
        tp=fp=fn=0;count=0;area_blockers=[]
        for row in rows:
            if row['excluded']:continue
            if row['image_id'] not in records:
                area_blockers.append(f'{area.name}: missing test annotations for {row["name"]}');continue
            try:
                image,truth=area_sample(store,pipeline.recipe,area,records[row['image_id']],pipeline.reference)
            except ValueError as exc:
                area_blockers.append(f'{area.name}: annotation alignment failed: {exc}')
                continue
            result=next((r for r in row['result']['areas'] if r['area_id']==area.id),{})
            predictions=[p for p in result.get('detections',[]) if p['score']>=area.method.threshold+area.method.review_margin]
            a,b,c=matched_counts(predictions,truth,image.shape[:2],area.method.kind=='segmentation')
            tp+=a;fp+=b;fn+=c;count+=1
        precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
        output[area.id]={'name':area.name,'images':count,'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'precision':precision,'recall':recall,'iou_threshold':.5,
            'scope':'mask instances' if area.method.kind=='segmentation' else 'boxes'}
        limits=pipeline.recipe.validation
        if precision is None or precision<limits.min_precision:area_blockers.append(f'{area.name}: object precision does not meet {limits.min_precision:g}')
        if recall is None or recall<limits.min_recall:area_blockers.append(f'{area.name}: object recall does not meet {limits.min_recall:g}')
        output[area.id].update(required=area.required,warnings=area_blockers)
        if area.required:blockers.extend(area_blockers)
    return {'dataset_id':data['id'],'dataset_hash':data['hash'],'areas':output},blockers
