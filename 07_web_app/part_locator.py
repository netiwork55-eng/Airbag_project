"""One ROI contract for reference training, image inspection and live cameras."""

def part_class_id(names):
    names = names if isinstance(names, dict) else dict(enumerate(names))
    excluded = ("mark", "stain", "defect", "scratch", "dent", "ng", "background")
    for index, name in names.items():
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if not key.startswith(excluded):
            return int(index)
    return None


def analyze_part(model, class_id, frame, *, device="cpu", half=False):
    if model is None or class_id is None:
        return None
    prediction = model.predict(frame, conf=0.12, imgsz=640, iou=0.45,
                               max_det=10, device=device, verbose=False,
                               **({"half": True} if half else {}))[0]
    if prediction.boxes is None:
        return None
    area = max(1, frame.shape[0] * frame.shape[1])
    best, best_score = None, -1.0
    for box in prediction.boxes:
        if int(box.cls[0]) != int(class_id):
            continue
        x1,y1,x2,y2 = [int(float(v)) for v in box.xyxy[0]]
        ratio = max(0,x2-x1)*max(0,y2-y1)/area
        if not 0.015 <= ratio <= 0.92:
            continue
        confidence = float(box.conf[0])
        score = confidence * ratio ** 0.35
        if score > best_score:
            best_score = score
            best = {"box": (x1,y1,x2,y2), "confidence": confidence,
                    "label": "Part", "defect_confidence": None}
    return best
