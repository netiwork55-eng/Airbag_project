"""One capture-to-decision pipeline shared by tests and Production."""
import math
import time
import cv2
import numpy as np
from .schema import Recipe, State, decision
from . import ENGINE_VERSION


class CannotInspect(ValueError):
    pass


def box(region, image):
    h, w = image.shape[:2]
    values = (round(region.x*w), round(region.y*h), round((region.x+region.width)*w), round((region.y+region.height)*h))
    x1,y1,x2,y2 = values
    if x2-x1 < 8 or y2-y1 < 8 or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise CannotInspect("Inspection area is missing or smaller than 8 pixels")
    return values


def crop(image, region):
    x1,y1,x2,y2 = box(region, image)
    return image[y1:y2, x1:x2]


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def match(template, image):
    if np.std(template) < 2:
        raise CannotInspect("Reference has too little detail for template matching")
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise CannotInspect("Reference region is outside the captured image")
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, point = cv2.minMaxLoc(scores)
    if not math.isfinite(score):
        raise CannotInspect("Template matching returned an invalid score")
    return float(score), point


def align(frame, reference, locator, detail=None):
    detail = detail if detail is not None else {}
    detail.update(method=locator.method, score=None, part_present=None)
    if frame.shape != reference.shape:
        raise CannotInspect("Image dimensions differ from the reference; configure the camera first")
    h,w = reference.shape[:2]
    matrix = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
    if locator.method == "template":
        x,y,x2,y2 = box(locator.region, reference)
        score, point = match(gray(reference)[y:y2, x:x2], gray(frame))
        detail["score"] = score
        if score < locator.min_score:
            detail["part_present"] = False
            raise CannotInspect("Part not found: template score below the locator limit")
        dx,dy = point[0]-x, point[1]-y
        if abs(dx)/w > locator.max_shift or abs(dy)/h > locator.max_shift:
            raise CannotInspect("Part position exceeds the configured tolerance")
        matrix[:,2] = [-dx,-dy]
    elif locator.method == "features":
        mask = np.zeros((h,w), np.uint8)
        x,y,x2,y2 = box(locator.region, reference)
        mask[y:y2,x:x2] = 255
        orb = cv2.ORB_create(nfeatures=1800)
        kr,dr = orb.detectAndCompute(gray(reference), mask)
        kf,df = orb.detectAndCompute(gray(frame), None)
        if dr is None or df is None:
            detail["part_present"] = False
            raise CannotInspect("Part not found: insufficient reference features")
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(df, dr, k=2)
        good = [p[0] for p in pairs if len(p)==2 and p[0].distance < .75*p[1].distance]
        detail["matches"] = len(good)
        if len(good) < locator.min_matches:
            detail["part_present"] = False
            raise CannotInspect("Part not found: insufficient matching features")
        src = np.float32([kf[m.queryIdx].pt for m in good])
        dst = np.float32([kr[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
        score = float(inliers.mean()) if inliers is not None else 0
        detail["score"] = score
        if matrix is None or not np.isfinite(matrix).all() or score < locator.min_score or int(inliers.sum()) < locator.min_matches:
            raise CannotInspect("Part alignment failed: unreliable feature match")
        scale = math.hypot(matrix[0,0],matrix[1,0])
        rotation = abs(math.degrees(math.atan2(matrix[1,0],matrix[0,0])))
        if (abs(scale-1)>locator.scale_tolerance or rotation>locator.max_rotation
                or abs(matrix[0,2])/w>locator.max_shift or abs(matrix[1,2])/h>locator.max_shift):
            raise CannotInspect("Part alignment exceeds position, rotation or scale tolerance")
        detail.update(score=score, rotation=rotation, scale=scale)
    aligned = cv2.warpAffine(frame, matrix, (w,h), flags=cv2.INTER_LINEAR)
    visible = cv2.warpAffine(np.full((h,w),255,np.uint8), matrix, (w,h), flags=cv2.INTER_NEAREST)
    detail["transform"] = matrix.tolist()
    detail["part_present"] = True
    return aligned, visible, detail


def prepare(frame, reference, recipe, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(quality={}, locator={}, stage="image_quality")
    if frame is None or not isinstance(frame,np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
        raise CannotInspect("No valid camera image received")
    g = gray(frame)
    mean, blur = float(g.mean()), float(cv2.Laplacian(g,cv2.CV_64F).var())
    q = recipe.quality
    diagnostics["image_size"] = {"width":frame.shape[1], "height":frame.shape[0]}
    diagnostics["quality"] = {"brightness":mean, "sharpness":blur,
                              "limits":q.model_dump(), "blur_enabled":bool(q.min_blur)}
    if not q.min_brightness <= mean <= q.max_brightness:
        raise CannotInspect(f"Image brightness {mean:.1f} is outside {q.min_brightness:g}–{q.max_brightness:g}")
    if q.min_blur and blur < q.min_blur:
        raise CannotInspect(f"Image is too blurry: {blur:.1f} < {q.min_blur:g}")
    diagnostics["stage"] = "alignment"
    aligned,visible,location = align(frame,reference,recipe.locator,diagnostics["locator"])
    diagnostics["stage"] = "inspection_areas"
    for area in recipe.areas:
        if area.required and np.any(crop(visible,area.region) != 255):
            raise CannotInspect(f"Required area is not fully visible: {area.name}")
    return aligned, visible, diagnostics["quality"],location


def detection_result(output, method, image):
    """Count only configured labels; boundary detections cannot quietly pass."""
    detections=output.get('detections')
    if not isinstance(detections,list):raise CannotInspect('Detector returned no valid result')
    selected=[];strong=0;uncertain=0;h,w=image.shape[:2]
    for item in detections:
        score=float(item['score']);bounds=np.asarray(item['box'],float)
        if not math.isfinite(score) or not 0<=score<=1 or bounds.shape!=(4,) or not np.isfinite(bounds).all():
            raise CannotInspect('Detector returned invalid confidence or coordinates')
        x1,y1,x2,y2=bounds
        if not 0<=x1<x2<=w+.01 or not 0<=y1<y2<=h+.01:
            raise CannotInspect('Detection is outside its inspection area')
        if item['label'] not in method.target_labels or score<method.threshold-method.review_margin:continue
        selected.append(item)
        if score>=method.threshold+method.review_margin:strong+=1
        else:uncertain+=1
    if strong>method.max_count or strong+uncertain<method.min_count:state=State.NG
    elif strong<method.min_count or strong+uncertain>method.max_count:state=State.REVIEW
    else:state=State.PASS
    return state,selected,strong,uncertain


class Pipeline:
    def __init__(self, store, saved, anomaly=None):
        self.store, self.saved, self.anomaly = store,saved,anomaly
        self.recipe = Recipe.model_validate(saved["config"])
        self.reference = store.image(self.recipe.reference_id)

    def run(self, frame):
        start = time.perf_counter()
        result = {"engine_version":ENGINE_VERSION,"recipe_id":self.saved["id"],"recipe_hash":self.saved["hash"],"areas":[],"quality":{},"locator":{}}
        failures = []
        maps = {}
        try:
            aligned,visible,_,_ = prepare(frame,self.reference,self.recipe,result)
            for area in self.recipe.areas:
                row = {"area_id":area.id,"name":area.name,"required":area.required,
                       "method":area.method.kind,"region":area.region.model_dump(),
                       "settings":area.method.model_dump()}
                try:
                    x1,y1,x2,y2 = box(area.region,aligned)
                    inverse = cv2.invertAffineTransform(np.float32(result['locator']['transform']))
                    row['source_polygon'] = cv2.transform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]),inverse)[0].tolist()
                    if np.any(crop(visible,area.region) != 255):
                        raise CannotInspect("Area is outside the captured image")
                    image,ref = crop(aligned,area.region),crop(self.reference,area.region)
                    m = area.method
                    if m.kind == "presence":
                        score,_ = match(gray(ref),gray(image))
                        state = State.PASS if score >= m.threshold+m.review_margin else State.NG if score < m.threshold-m.review_margin else State.REVIEW
                        row.update(score=score,threshold=m.threshold,reason=f"{area.name}: reference match {score:.3f}")
                    elif m.kind == "brightness":
                        score = float(gray(image).mean())
                        state = State.NG if score < m.low or score > m.high else State.REVIEW if min(score-m.low,m.high-score) <= m.review_margin else State.PASS
                        row.update(score=score,limits=[m.low,m.high],reason=f"{area.name}: brightness {score:.1f}")
                    else:
                        if not m.model_id or self.anomaly is None:
                            raise CannotInspect("This area needs a trained model")
                        metadata = self.store.model(m.model_id)
                        from .store import digest
                        if (m.model_id in self.saved.get("model_metadata",{}) and
                                digest(metadata)!=self.saved["model_metadata"][m.model_id]):
                            raise CannotInspect("Model configuration changed after recipe validation")
                        # Model input must match its trained image setup and area, not just image dimensions.
                        if metadata["input_signature"] != input_signature(self.recipe,area):
                            raise CannotInspect("Model was trained with a different image setup or area")
                        expected='torchvision' if m.kind in ('detection','segmentation') else 'anomalib'
                        if metadata.get('framework')!=expected or (expected=='torchvision' and metadata.get('kind')!=m.kind):
                            raise CannotInspect('Model type does not match this inspection method')
                        output = self.anomaly.predict(m.model_id,image)
                        if expected=='torchvision':
                            if not set(m.target_labels)<=set(metadata['labels']):raise CannotInspect('Selected labels are not in this model')
                            state,items,count,uncertain=detection_result(output,m,image)
                            x,y,_,_=box(area.region,aligned)
                            inverse=cv2.invertAffineTransform(np.float32(result['locator']['transform']))
                            for item in items:
                                a,b,c,d=item['box']
                                item['source_polygon']=cv2.transform(np.float32([[[a+x,b+y],[c+x,b+y],[c+x,d+y],[a+x,d+y]]]),inverse)[0].tolist()
                                item['source_masks']=[cv2.transform(np.float32([np.asarray(p)+[x,y]]),inverse)[0].tolist() for p in item.get('polygons',[])]
                            row.update(state=state.value,detections=items,count=count,uncertain_count=uncertain,
                                threshold=m.threshold,review_margin=m.review_margin,limits=[m.min_count,m.max_count],
                                model_id=m.model_id,model_sha256=metadata['sha256'],
                                reason=f'{area.name}: {count} confirmed, {uncertain} uncertain; required {m.min_count}–{m.max_count}')
                            result['areas'].append(row)
                            continue
                        score = float(output["score"])
                        if not math.isfinite(score):
                            raise CannotInspect("Model returned a non-finite anomaly score")
                        state = State.NG if score > m.threshold+m.review_margin else State.PASS if score < m.threshold-m.review_margin else State.REVIEW
                        row.update(score=score,threshold=m.threshold,model_id=m.model_id,
                                   model_sha256=metadata["sha256"],reason=f"{area.name}: anomaly score {score:.4f}")
                        if output.get("map") is not None:
                            amap = np.asarray(output["map"],np.float32)
                            if not np.isfinite(amap).all():
                                raise CannotInspect("Model returned an invalid anomaly map")
                            maps[area.id] = amap
                    row["state"] = state.value
                except Exception as exc:
                    row.update(state=State.UNABLE.value,reason=f"{area.name}: {exc}")
                result["areas"].append(row)
        except Exception as exc:
            failures.append(str(exc))
        if failures:
            result["failed_stage"] = result.get("stage", "image_quality")
            # Keep every requested area visible in evidence even when an upstream gate failed.
            result["areas"] = [{"area_id":a.id,"name":a.name,"required":a.required,
                                "method":a.method.kind,"region":a.region.model_dump(),
                                "settings":a.method.model_dump(),"state":State.UNABLE.value,
                                "reason":"Not inspected: "+"; ".join(failures)} for a in self.recipe.areas]
        result["stage"] = "decision"
        result["state"],result["reasons"] = decision(result["areas"],failures)
        result["elapsed_ms"] = round((time.perf_counter()-start)*1000,3)
        return result,maps


def input_signature(recipe, area):
    from .store import digest
    camera = recipe.camera.model_dump()
    for key in ("width","height"):
        if camera[key] is None:
            camera.pop(key)
    camera.pop("max_frame_age_ms")  # Timing limit does not change model input pixels.
    return digest({"reference_id":recipe.reference_id,"camera":camera,
                   "locator":recipe.locator.model_dump(),"region":area.region.model_dump(),
                   "quality":recipe.quality.model_dump()})
