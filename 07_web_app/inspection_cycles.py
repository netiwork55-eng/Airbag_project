"""One result for each place-and-remove camera inspection cycle."""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2


class PartCycle:
    def __init__(self, release_frames=3, release_seconds=.3):
        self.release_frames = release_frames
        self.release_seconds = release_seconds
        self.current = None
        self.missing = 0
        self.missing_since = None
        self.evidence = None

    def observe(self, present, result, *, now=None, frame=None):
        now = time.monotonic() if now is None else now
        if not present:
            if self.current is None:
                return None
            if self.missing_since is None:
                self.missing_since = now
            self.missing += 1
            if self.missing >= self.release_frames and now - self.missing_since >= self.release_seconds:
                return self.finish("removed")
            return None
        self.missing = 0
        self.missing_since = None
        if self.current is None:
            self.current = {"id": "part_" + uuid.uuid4().hex,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "frames": 0, "decision": "REVIEW", "confirmed_ng": False,
                "confirmed_ok": False, "needs_review": False, "input_issues": []}
        item = self.current
        item["frames"] += 1
        if result is None:
            return None
        issue = result.input_issue
        if issue and issue != "waiting_for_frames":
            if issue not in item["input_issues"]:
                item["input_issues"].append(issue)
            item["needs_review"] = True
        elif not issue:
            if result.status == "NG":
                item["confirmed_ng"] = True
            elif result.status == "REVIEW" or result.review_required:
                item["needs_review"] = True
            elif result.status == "OK":
                item["confirmed_ok"] = True
        decision = "NG" if item["confirmed_ng"] else "REVIEW" if item["needs_review"] or not item["confirmed_ok"] else "OK"
        # Keep a representative frame for the held decision, never place an old heatmap on a new frame.
        if frame is not None and (self.evidence is None or decision != item["decision"]):
            self.evidence = frame.copy()
            item["evidence_result"] = result.to_dict()
            item["evidence_result"].pop("anomaly_map", None)
        item["decision"] = decision
        return None

    def finish(self, reason):
        if self.current is None:
            return None
        item = self.current
        item["ended_at"] = datetime.now(timezone.utc).isoformat()
        item["end_reason"] = reason
        item["complete"] = reason == "removed"
        if not item["complete"] and item["decision"] != "NG":
            item["decision"] = "REVIEW"
        image = self.evidence
        self.current = self.evidence = None
        self.missing = 0
        self.missing_since = None
        return item, image

    def draw(self, frame):
        if self.current:
            decision = self.current["decision"]
            color = {"NG": (50,50,240), "OK": (70,200,70), "REVIEW": (40,190,240)}[decision]
            cv2.rectangle(frame, (0,0), (frame.shape[1],30), (20,24,30), -1)
            cv2.putText(frame, f"Part result: {decision} - lift part to finish", (8,21), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        return frame


def save_cycle(root, closed, reference, source):
    if closed is None:
        return None
    item, image = closed
    folder = Path(root) / "03_dataset" / "inspection_cycles"
    folder.mkdir(parents=True, exist_ok=True)
    item = {**item, "source": source, "job_id": reference.get("job_id"),
            "model_sha256": (reference.get("inspection_contract") or {}).get("model_sha256"),
            "model": reference.get("active_model"), "method": "place_and_remove"}
    if image is not None:
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise OSError("Could not save part evidence")
        encoded.tofile(str(folder / (item["id"] + ".jpg")))
        item["image"] = item["id"] + ".jpg"
    path = folder / (item["id"] + ".json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
    return item
