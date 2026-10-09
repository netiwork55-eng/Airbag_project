"""Import reviewed annotations; drawing/editing remains in the existing CVAT."""
import re
from pathlib import PurePosixPath
import uuid
import cv2
import numpy as np
from .store import atomic_json, digest, now
from .pipeline import prepare, box


def save_dataset(store, records, *, source, name):
    if not records or len(records)>50000:
        raise ValueError("Choose 1–50000 annotated images")
    if len({r['image_id'] for r in records})!=len(records):
        raise ValueError("Duplicate images in annotation set")
    labels=set()
    for row in records:
        meta=store.read('images',row['image_id'])
        store.image(row['image_id'])
        row.update(image_sha256=meta['sha256'],group=meta['group'])
        for obj in row['objects']:
            label=str(obj['label']).strip()
            if not label or len(label)>100:
                raise ValueError("Invalid annotation label")
            obj['label']=label;labels.add(label)
            bounds=np.asarray(obj['box'],dtype=float)
            if bounds.shape!=(4,) or not np.isfinite(bounds).all():
                raise ValueError("Invalid annotation coordinates")
            x1,y1,x2,y2=bounds
            if not 0<=x1<x2<=meta['width'] or not 0<=y1<y2<=meta['height']:
                raise ValueError("Annotation is outside its source image")
            for polygon in obj.get('polygons',[]):
                p=np.asarray(polygon,dtype=float)
                if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
                    raise ValueError("Invalid segmentation polygon")
                if (p<0).any() or (p[:,0]>meta['width']).any() or (p[:,1]>meta['height']).any():
                    raise ValueError("Segmentation is outside the image")
    item={'id':'data_'+uuid.uuid4().hex,'name':str(name)[:160],'source':source,
          'created_at':now(),'labels':sorted(labels),'records':records}
    item['hash']=digest(item)
    atomic_json(store.path('datasets',item['id']),item)
    return item


def read_dataset(store,key):
    item=store.read('datasets',key)
    if item['hash']!=digest({k:v for k,v in item.items() if k!='hash'}):
        raise ValueError("Annotation set changed after import")
    for row in item['records']:
        meta=store.read('images',row['image_id'])
        if meta['sha256']!=row['image_sha256'] or meta['group']!=row['group']:
            raise ValueError("Annotated source image/group changed")
    return item


def import_coco(store, payload, *, reviewed=False, name='CVAT annotations'):
    if reviewed is not True:
        raise ValueError("Confirm that all annotations, including empty images, were reviewed in CVAT")
    categories={c['id']:str(c['name']) for c in payload['categories']}
    if len(categories)!=len(payload['categories']) or len(set(categories.values()))!=len(categories):
        raise ValueError("Duplicate annotation categories")
    assets=store.records('images');by_id={i['id']:i for i in assets};by_name={}
    for asset in assets:by_name.setdefault(asset['name'],[]).append(asset)
    rows={}
    for item in payload['images']:
        name_part=PurePosixPath(str(item['file_name']).replace('\\','/')).name
        key=PurePosixPath(name_part).stem
        # CVAT upload prefixes preserve frame order; retain the original content ID.
        match=re.fullmatch(r"\d{6}_(image_[0-9a-f]{64})",key)
        if match:key=match.group(1)
        matches=[by_id[key]] if key in by_id else by_name.get(name_part,[])
        if len(matches)!=1:
            raise ValueError(f"Import the source image first; name must match uniquely: {name_part}")
        image=matches[0]
        if image['width']!=item['width'] or image['height']!=item['height']:
            raise ValueError("Annotation dimensions do not match the source image")
        if item['id'] in rows:raise ValueError("Duplicate COCO image ID")
        rows[item['id']]={'image_id':image['id'],'objects':[]}
    for a in payload['annotations']:
        if a.get('iscrowd',0):raise ValueError("Crowd annotations are not supported for individual part checks")
        x,y,w,h=map(float,a['bbox'])
        segmentation=a.get('segmentation') or []
        if not isinstance(segmentation,list):
            raise ValueError("Export polygon annotations from CVAT; compressed RLE masks are not supported")
        polygons=[]
        for polygon in segmentation:
            p=np.asarray(polygon,dtype=float)
            if p.ndim!=1 or len(p)<6 or len(p)%2:raise ValueError("Invalid COCO polygon")
            polygons.append(p.reshape(-1,2).tolist())
        rows[a['image_id']]['objects'].append({'label':categories[a['category_id']],
            'box':[x,y,x+w,y+h],'polygons':polygons})
    return save_dataset(store,list(rows.values()),source='cvat_coco',name=name)


def area_sample(store,recipe,area,row,reference):
    """Apply the same locator transform to pixels, boxes and annotation masks."""
    source=store.image(row['image_id'])
    aligned,visible,_,location=prepare(source,reference,recipe)
    x1,y1,x2,y2=box(area.region,aligned)
    if np.any(visible[y1:y2,x1:x2]!=255):raise ValueError("Annotated area is not fully visible")
    matrix=np.asarray(location['transform'],np.float32)
    objects=[];height,width=aligned.shape[:2]
    for obj in row['objects']:
        if obj['label'] not in area.method.target_labels:continue
        a,b,c,d=obj['box']
        corners=cv2.transform(np.float32([[[a,b],[c,b],[c,d],[a,d]]]),matrix)[0]
        lo=corners.min(axis=0);hi=corners.max(axis=0)
        if hi[0]<=x1 or hi[1]<=y1 or lo[0]>=x2 or lo[1]>=y2:continue
        if lo[0]<x1-.01 or lo[1]<y1-.01 or hi[0]>x2+.01 or hi[1]>y2+.01:
            raise ValueError("A labelled object crosses the inspection area; adjust the area or annotation")
        item={'label':obj['label'],'box':[float(lo[0]-x1),float(lo[1]-y1),float(hi[0]-x1),float(hi[1]-y1)]}
        if area.method.kind=='segmentation':
            if not obj.get('polygons'):raise ValueError("Segmentation training requires polygons, not only rectangles")
            mask=np.zeros(source.shape[:2],np.uint8)
            cv2.fillPoly(mask,[np.rint(p).astype(np.int32) for p in obj['polygons']],1)
            mask=cv2.warpAffine(mask,matrix,(width,height),flags=cv2.INTER_NEAREST)[y1:y2,x1:x2]
            if not mask.any():raise ValueError("Segmentation has no visible pixels")
            item['mask']=mask
        objects.append(item)
    return aligned[y1:y2,x1:x2],objects
