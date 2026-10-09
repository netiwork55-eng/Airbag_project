"""Inspection routes mounted into the existing FastAPI application."""
import importlib.util
from pathlib import Path
import threading
import uuid
from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, Response
from .schema import Recipe, Camera
from .store import Store, atomic_json, now, label_confirmed, group_confirmed
from .pipeline import Pipeline
from .runtime import Runtime, evidence, OpenCVCamera, BaslerCamera
from .validation import validate
from .worker_client import AnomalyClient
from .annotations import import_coco,read_dataset,save_dataset
from . import ENGINE_VERSION


class InspectionService:
    def __init__(self, root, *, legacy_busy=lambda:False, cvat_send=None, cvat_pull=None, dataset_folder=None):
        self.root = Path(root)
        self.store = Store(self.root/"inspection_data")
        self.anomaly = AnomalyClient(root,self.store)
        self.runtime = Runtime(self.store,self.anomaly)
        self.legacy_busy,self.cvat_send,self.dataset_folder = legacy_busy,cvat_send,dataset_folder
        self.cvat_pull=cvat_pull
        self.work_lock = threading.Lock()
        self.training = None
        self.training_client = None
        self.cvat_lock = threading.Lock()
        # An interrupted training process must not keep a stale "running" state after restart.
        for item in self.store.records("training"):
            if item["status"] == "running":
                item.update(status="failed",error="Application restarted before training completed")
                atomic_json(self.store.path("training",item["id"]),item)

    def busy(self):
        return self.runtime.running or self.work_lock.locked()

    def idle(self):
        if self.runtime.running or self.legacy_busy():
            raise ValueError("Stop the running inspection/training before changing or testing a configuration")

    def close(self):
        # Release both workers even when a camera driver cannot stop cleanly.
        if self.training and self.training.get("status") == "running":
            self.training["cancel_requested"] = True
        try:
            self.runtime.stop()
        finally:
            try:
                if self.training_client:
                    self.training_client.cancel()
            finally:
                self.anomaly.close()

    def begin_train(self, payload):
        self.idle()
        if not self.work_lock.acquire(blocking=False):
            raise ValueError("Another test or training task is running")
        try:
            recipe = self.store.recipe(payload["recipe_id"])
            area = next((a for a in recipe["config"]["areas"] if a["id"]==payload["area_id"]),None)
            if not area or area['method']['kind'] not in ('anomaly','detection','segmentation'):
                raise ValueError("This inspection method does not need AI training")
            request={'recipe_id':recipe['id'],'area_id':area['id']}
            if area['method']['kind']=='anomaly':
                reference = self.store.read('images',recipe['config']['reference_id'])
                if reference['label']!='GOOD' or not label_confirmed(reference) or not group_confirmed(reference):
                    raise ValueError('Confirm the GOOD reference and its actual capture group before training')
                algorithm = payload.get("algorithm","padim")
                if algorithm not in ("padim","patchcore"):
                    raise ValueError("Choose an evaluated Anomalib method: padim or patchcore")
                ids = payload["image_ids"]
                if not 20<=len(ids) or len(set(ids))!=len(ids):
                    raise ValueError("Choose at least 20 distinct confirmed GOOD images")
                for key in ids:
                    image = self.store.read("images",key)
                    if image['label']!='GOOD' or not label_confirmed(image) or not group_confirmed(image):
                        raise ValueError("Training needs human-confirmed GOOD images with explicit capture groups")
                request.update(action='train',image_ids=ids,algorithm=algorithm)
            else:
                dataset=read_dataset(self.store,payload['dataset_id'])
                algorithm=area['method']['kind']
                epochs=int(payload.get('epochs',10));size=int(payload.get('input_size',512))
                if not 1<=epochs<=100 or size not in (320,512,640):raise ValueError('Invalid training settings')
                request.update(action='train_detection',dataset_id=dataset['id'],epochs=epochs,input_size=size)
            item = {"id":"train_"+uuid.uuid4().hex,"status":"running","created_at":now(),
                    "recipe_id":recipe["id"],"area_id":area["id"],"algorithm":algorithm}
            atomic_json(self.store.path("training",item["id"]),item)
            self.training = item
            model_id = "model_"+uuid.uuid4().hex
            def work():
                client = AnomalyClient(self.root,self.store)
                self.training_client = client
                try:
                    if item.get("cancel_requested"):
                        raise RuntimeError("Candidate training was cancelled")
                    model = client.call({**request,'model_id':model_id},timeout=7200)
                    if item.get("cancel_requested"):
                        raise RuntimeError("Candidate training was cancelled")
                    config = recipe["config"]
                    target = next(a for a in config["areas"] if a["id"]==area["id"])
                    target["method"].update(model_id=model["id"],threshold=model["threshold"],review_margin=model["review_margin"])
                    candidate = self.store.save_recipe(config,parent=recipe["id"])
                    item.update(status="complete",model_id=model["id"],candidate_id=candidate["id"])
                except Exception as exc:
                    item.update(status="cancelled" if item.get("cancel_requested") else "failed",error=str(exc))
                finally:
                    try:
                        client.close()
                        if item["status"] in ("cancelled","failed"):
                            # Remove only artifacts owned by this failed task; never a saved/active model.
                            for suffix in (".pt",".onnx",".json"):
                                self.store.path("models",model_id,suffix).unlink(missing_ok=True)
                        item["ended_at"] = now()
                        atomic_json(self.store.path("training",item["id"]),item)
                    except Exception as exc:
                        item.update(status="failed",error="Training cleanup/storage failed: "+str(exc))
                    finally:
                        self.training_client = None
                        self.work_lock.release()
            threading.Thread(target=work,daemon=True,name="candidate-anomaly-training").start()
            return item
        except Exception:
            self.work_lock.release()
            raise


def make_router(service):
    router = APIRouter(prefix="/api/vision",tags=["Inspection"])
    store = service.store

    def operation(fn):
        try:
            service.idle()
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        if not service.work_lock.acquire(blocking=False):
            raise HTTPException(409,"Another test or training task is running")
        try:
            return fn()
        except (ValueError,KeyError,StopIteration,FileNotFoundError,RuntimeError) as exc:
            raise HTTPException(400,str(exc)) from exc
        finally:
            service.work_lock.release()

    @router.get("/status")
    def status():
        return {"engine_version":ENGINE_VERSION,"production":store.production(),"runtime":service.runtime.snapshot(),
                "busy":service.busy(),"training":service.training,
                "capabilities":{"opencv":True,"anomalib":(service.root/".venv-inspection/Scripts/python.exe").is_file(),
                    "basler_sdk":importlib.util.find_spec("pypylon") is not None,
                    "detection":"TorchVision detection and segmentation; reviewed annotations and independent validation required",
                    "legacy":"Existing models remain available in Engineering; never automatically promoted"}}

    @router.get("/recipes")
    def recipes():
        return {"engine_version":ENGINE_VERSION,"recipes":store.records("recipes"),"production":store.production()}

    @router.post("/recipes")
    def save_recipe(payload:dict=Body(...)):
        return operation(lambda:store.save_recipe(payload["config"],payload.get("parent")))

    @router.get('/datasets')
    def datasets():
        return {'datasets':[{k:v for k,v in d.items() if k!='records'}|{'images':len(d['records'])}
                            for d in store.records('datasets')]}

    @router.post('/datasets/import-coco')
    def coco(payload:dict=Body(...)):
        return operation(lambda:import_coco(store,payload['coco'],reviewed=payload.get('reviewed'),name=payload.get('name','CVAT annotations')))

    @router.post('/datasets/cvat')
    def pull_annotations(payload:dict=Body(...)):
        if service.cvat_pull is None:raise HTTPException(400,'Existing CVAT integration is unavailable')
        if payload.get('reviewed') is not True:raise HTTPException(400,'Confirm annotation review in CVAT first')
        ids=payload.get('event_ids',[])
        if not 1<=len(ids)<=100:raise HTTPException(400,'Select 1–100 evidence images')
        if not service.cvat_lock.acquire(blocking=False):raise HTTPException(409,'Another CVAT transfer is running')
        try:
            records=service.cvat_pull(list(dict.fromkeys(ids)))
            return save_dataset(store,records,source='cvat_api',name=payload.get('name','Reviewed CVAT images'))
        except (ValueError,KeyError,OSError,RuntimeError) as exc:
            raise HTTPException(400,str(exc)) from exc
        finally:service.cvat_lock.release()

    @router.post('/review/use-images')
    def reviewed_images(payload:dict=Body(...)):
        def run():
            ids=payload.get('event_ids',[])
            if not 1<=len(ids)<=500:raise ValueError('Choose 1–500 reviewed images')
            rows=[]
            for key in dict.fromkeys(ids):
                event=store.read('evidence',key);review=store.read('reviews',key)
                verdict=review['status']
                if verdict not in ('confirmed_ng','false_alarm'):raise ValueError('Confirm NG or False Alarm before using an image')
                item=store.read('images',event['image_id']);label='NG' if verdict=='confirmed_ng' else 'GOOD'
                if label_confirmed(item) and item['label']!=label:raise ValueError('Review conflicts with an existing confirmed label')
                rows.append((item,label,key))
            for item,label,key in rows:
                item.update(label=label,confirmed_at=now(),review_event=key)
                atomic_json(store.path('images',item['id']),item)
            return {'images':[r[0] for r in rows]}
        return operation(run)

    @router.get("/images")
    def images():
        return {"images":store.records("images")}

    @router.get("/images/{key}")
    def image(key:str):
        try:
            path = store.path("images",key,".png")
            if not path.exists():
                raise ValueError("Image not found")
            return FileResponse(path,media_type="image/png")
        except ValueError as exc:
            raise HTTPException(404,str(exc)) from exc

    @router.post("/images/{key}/confirm")
    def confirm(key:str,payload:dict=Body(...)):
        def run():
            item = store.read("images",key)
            verdict = payload.get("label")
            if verdict not in ("GOOD","NG") or (label_confirmed(item) and item['label']!=verdict):
                raise ValueError("Conflicting or invalid image label")
            item.update(label=verdict,confirmed_at=now())
            if 'group' in payload:
                group=payload['group']
                if not isinstance(group,str) or not group.strip() or len(group)>120:
                    raise ValueError('Enter the actual part/capture group, up to 120 characters')
                item.update(group=group.strip(),group_confirmed=True)
            atomic_json(store.path("images",key),item)
            return item
        return operation(run)

    @router.post("/images")
    async def upload(files:list[UploadFile]=File(...),label:str=Form("UNKNOWN"),group:str=Form("")):
        # Files are local project data, never transmitted to a remote image service.
        if len(files)>1000:
            raise HTTPException(400,"Upload at most 1000 files per request; additional batches are supported")
        result,total = [],0
        for file in files:
            raw = await file.read(20*1024*1024+1)
            total += len(raw)
            if total>200*1024*1024:
                raise HTTPException(400,"Upload batch exceeds 200 MB; use another batch")
            try:
                result.append(store.put_image(raw,file.filename or "image",label=label,group=group))
            except ValueError as exc:
                raise HTTPException(400,str(exc)) from exc
            finally:
                await file.close()
        return {"images":result}

    @router.post("/images/import")
    def import_existing(payload:dict=Body(...)):
        def run():
            if service.dataset_folder is None:
                raise ValueError("Existing dataset integration is unavailable")
            folder = service.dataset_folder(str(payload["folder"]))
            paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"})
            if not paths:
                raise ValueError("No images found in this existing dataset")
            result = []
            for path in paths:
                if folder.resolve() not in path.resolve().parents:
                    raise ValueError("Dataset contains an image outside its folder")
                # Imported images are unlabelled by default. Dataset names are not quality labels.
                result.append(store.put_image(path.read_bytes(),path.name,label=payload.get("label","UNKNOWN"),
                    group=payload.get("group", "")))
            return {"images":result}
        return operation(run)

    @router.post("/test")
    def test(payload:dict=Body(...)):
        def run():
            saved = store.recipe(payload["recipe_id"])
            frame = store.image(payload["image_id"])
            result,maps = Pipeline(store,saved,service.anomaly).run(frame)
            image_meta=store.read('images',payload['image_id'])
            result['source'] = 'test'
            result['capture'] = image_meta.get('capture')
            result['capture_group'] = image_meta['group']
            previous=next((e for e in store.records('evidence',limit=100)
                           if e.get('source')=='test' and e.get('image_id')==payload['image_id']
                           and e.get('recipe_hash')==saved['hash'] and e.get('engine_version')==ENGINE_VERSION),None)
            if previous:
                old_scores={a['area_id']:a['score'] for a in previous['areas'] if 'score' in a}
                result['repeat_check']={'previous_event_id':previous['id'],
                    'same_decision':previous['state']==result['state'],
                    'score_deltas':{a['area_id']:a['score']-old_scores[a['area_id']] for a in result['areas']
                                    if 'score' in a and a['area_id'] in old_scores},
                    'scope':'Same stored image/recipe/engine replay; not camera repeatability or accuracy'}
            return evidence(store,result,frame,maps,source="test")
        return operation(run)

    @router.post("/camera/capture")
    def capture(payload:dict=Body(...)):
        def run():
            import cv2
            settings = dict(payload)
            group = settings.pop('capture_group', '')
            if not isinstance(group,str) or len(group)>120:
                raise ValueError('Capture group must be text up to 120 characters')
            config = Camera.model_validate(settings)
            camera = (BaslerCamera if config.driver=="basler" else OpenCVCamera)(config)
            try:
                frame = camera.read()
                ok,png = cv2.imencode(".png",frame)
                if not ok:
                    raise ValueError("Camera image could not be encoded")
                item=store.put_image(png.tobytes(),"Camera capture "+now(),label="UNKNOWN",group=group)
                if 'capture' not in item:
                    item['capture']={'captured_at':now(),'requested':config.model_dump(),
                                     'actual':camera.metadata() if hasattr(camera,'metadata') else None,
                                     'width':frame.shape[1],'height':frame.shape[0]}
                    atomic_json(store.path('images',item['id']),item)
                return item
            finally:
                camera.close()
        return operation(run)

    @router.post("/validate")
    def validation(payload:dict=Body(...)):
        return operation(lambda:validate(store,Pipeline(store,store.recipe(payload["recipe_id"]),service.anomaly),payload["image_ids"],payload.get('dataset_id')))

    @router.get("/reports")
    def reports():
        return {"reports":store.records("tests")}

    @router.post("/activate")
    def activate(payload:dict=Body(...)):
        return operation(lambda:store.activate(payload["recipe_id"],payload["report_id"]))

    @router.post("/rollback")
    def rollback():
        return operation(store.rollback)

    @router.post("/train")
    def train(payload:dict=Body(...)):
        try:
            return service.begin_train(payload)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @router.get("/training")
    def training():
        return {"training":store.records("training")}

    @router.post("/training/stop")
    def stop_training():
        if service.training and service.training["status"]=="running":
            service.training["cancel_requested"] = True
            if service.training_client:
                service.training_client.cancel()
            return {"stopping":True}
        return {"stopping":False}

    @router.post("/production/start")
    def start():
        return operation(service.runtime.start)

    @router.post("/production/stop")
    def stop():
        try:
            service.runtime.stop()
        except RuntimeError as exc:
            raise HTTPException(409,str(exc)) from exc
        return service.runtime.snapshot()

    @router.get("/production/frame")
    def frame():
        with service.runtime.lock:
            data = service.runtime.frame
        if data is None:
            raise HTTPException(404,"No live camera image")
        return Response(data,media_type="image/jpeg",headers={"Cache-Control":"no-store"})

    @router.get("/evidence")
    def events(limit:int=Query(default=100,ge=1,le=250),before:str|None=None):
        try:return store.evidence_page(limit,before)
        except ValueError as exc:raise HTTPException(400,str(exc)) from exc

    @router.get("/evidence/{key}/maps")
    def maps(key:str):
        try:
            path = store.path("evidence",key,".npz")
        except ValueError as exc:
            raise HTTPException(404,str(exc)) from exc
        if not path.exists():
            raise HTTPException(404,"This event has no anomaly maps")
        return FileResponse(path,filename=key+".npz",media_type="application/octet-stream")

    @router.get("/evidence/{key}/preview")
    def map_preview(key:str):
        import cv2
        import numpy as np
        from .pipeline import box
        from .schema import Rect
        try:
            event = store.read("evidence",key)
            source = store.image(event["image_id"])
            path = store.path("evidence",key,".npz")
            has_locations=any(a.get('detections') or a.get('source_polygon') for a in event['areas'])
            if not path.exists() and not has_locations:
                raise ValueError("This result has no location evidence")
            overlay = np.zeros_like(source)
            mask = np.zeros(source.shape[:2],np.uint8)
            from contextlib import nullcontext
            with (np.load(path,allow_pickle=False) if path.exists() else nullcontext({})) as maps:
                for area in event["areas"]:
                    if area["area_id"] not in maps:
                        continue
                    amap = maps[area["area_id"]]
                    x1,y1,x2,y2 = box(Rect.model_validate(area["region"]),source)
                    lo,hi = float(amap.min()),float(amap.max())
                    intensity = np.clip((amap-lo)/max(hi-lo,1e-12)*255,0,255).astype(np.uint8)
                    color = cv2.applyColorMap(cv2.resize(intensity,(x2-x1,y2-y1)),cv2.COLORMAP_TURBO)
                    overlay[y1:y2,x1:x2] = color
                    mask[y1:y2,x1:x2] = 255
            matrix = np.float32(event.get("locator",{}).get("transform",[[1,0,0],[0,1,0]]))
            inverse = cv2.invertAffineTransform(matrix)
            h,w = source.shape[:2]
            overlay = cv2.warpAffine(overlay,inverse,(w,h))
            mask = cv2.warpAffine(mask,inverse,(w,h),flags=cv2.INTER_NEAREST)
            mixed = cv2.addWeighted(source,.55,overlay,.45,0)
            source[mask>0] = mixed[mask>0]
            for area in event['areas']:
                if area.get('source_polygon'):
                    polygon=np.rint(area['source_polygon']).astype(np.int32)
                    color=(30,170,30) if area['state']=='PASS' else (30,60,240)
                    cv2.polylines(source,[polygon],True,color,2)
                    point=tuple(np.maximum(polygon[0],[0,14]))
                    cv2.putText(source,area['area_id']+' '+area['state'],point,cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                for item in area.get('detections',[]):
                    polygon=np.rint(item['source_polygon']).astype(np.int32)
                    cv2.polylines(source,[polygon],True,(30,60,240),2)
                    for p in item.get('source_masks',[]):
                        cv2.polylines(source,[np.rint(p).astype(np.int32)],True,(0,170,255),1)
                    point=tuple(np.maximum(polygon[0],[0,14]))
                    cv2.putText(source,f"{item['label']} {item['score']:.2f}",point,cv2.FONT_HERSHEY_SIMPLEX,.45,(30,60,240),1)
            ok,png = cv2.imencode(".png",source)
            if not ok:
                raise ValueError("Preview encoding failed")
            return Response(png.tobytes(),media_type="image/png")
        except (ValueError,FileNotFoundError,KeyError) as exc:
            raise HTTPException(404,str(exc)) from exc

    @router.post("/review/{key}")
    def review(key:str,payload:dict=Body(...)):
        try:
            store.read("evidence",key)
        except (ValueError,FileNotFoundError) as exc:
            raise HTTPException(404,"Inspection evidence not found") from exc
        verdict = payload.get("status")
        if verdict not in ("confirmed_ng","false_alarm","needs_review"):
            raise HTTPException(400,"Choose Confirmed NG, False Alarm or Needs Review")
        record = {"id":key,"status":verdict,"note":str(payload.get("note",""))[:1000],"reviewed_at":now()}
        atomic_json(store.path("reviews",key),record)
        return record

    @router.post("/cvat/send")
    def cvat(payload:dict=Body(...)):
        def run():
            ids = payload.get("event_ids",[])
            if not 1<=len(ids)<=100:
                raise ValueError("Select 1–100 review images to send to the existing CVAT")
            if service.cvat_send is None:
                raise ValueError("Existing CVAT integration is unavailable")
            selected = []
            for key in dict.fromkeys(ids):
                event = store.read("evidence",key)
                verdict = store.read("reviews",key)
                if verdict["status"] not in ("confirmed_ng","needs_review"):
                    raise ValueError("Select Confirmed NG or Needs Review images for annotation")
                selected.append((event,verdict))
            return service.cvat_send(selected)
        # Annotation transfer owns neither the camera nor the inference/training lock.
        # Sending selected evidence must remain possible during Production.
        if not service.cvat_lock.acquire(blocking=False):
            raise HTTPException(409,"Another CVAT transfer is running")
        try:
            return run()
        except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:
            raise HTTPException(400,str(exc)) from exc
        finally:
            service.cvat_lock.release()

    return router
