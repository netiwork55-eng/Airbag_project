"""Generic normal-reference inspection for reusable visual inspection projects.

The model deliberately does not classify defect types.  It learns the appearance
and normal variation of known-good images, then returns OK/NG plus localized
regions that differ significantly from that reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from collections import deque
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class AlignmentError(ValueError):
    """The image cannot be compared in the model's reference coordinates."""


@dataclass(frozen=True)
class InspectionRegion:
    x: int
    y: int
    width: int
    height: int
    score: float


@dataclass(frozen=True)
class InspectionResult:
    status: str
    confidence: float
    anomaly_score: float
    threshold: float
    regions: list[InspectionRegion]
    brand: str | None = None
    brand_confidence: float | None = None
    review_required: bool = False
    anomaly_map: list[list[float]] | None = None
    input_issue: str | None = None
    inspection_area: list[float] | None = None
    inspection_areas: list[list[float]] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["regions"] = [asdict(region) for region in self.regions]
        payload["location"] = ("not_checked" if self.input_issue else "found" if self.regions
                               else "not_found" if self.status == "NG" or self.review_required else "not_needed")
        return payload


class DecisionStabilizer:
    """Debounce live OK/NG decisions without hiding persistent defects."""

    def __init__(self, window: int = 5, ng_votes: int = 3, ok_votes: int = 4) -> None:
        self.window = max(3, int(window))
        self.ng_votes = max(2, min(int(ng_votes), self.window))
        self.ok_votes = max(2, min(int(ok_votes), self.window))
        self._history: deque[InspectionResult] = deque(maxlen=self.window)
        self._stable_status: str | None = None
        self._last_box = None
        self._last_brand = None
        self.segment = 0

    def reset(self) -> None:
        self.segment += 1
        self._history.clear()
        self._stable_status = None
        self._last_box = None
        self._last_brand = None

    def update(self, result: InspectionResult, *, object_box=None) -> InspectionResult:
        if result.input_issue:
            self.reset()
            return result
        # A different part or a discontinuous ROI must not inherit votes/evidence.
        changed = bool(self._last_brand and result.brand and self._last_brand != result.brand)
        if object_box is not None and self._last_box is not None:
            ax, ay, bx, by = self._last_box
            cx, cy, dx, dy = object_box
            intersection = max(0, min(bx, dx) - max(ax, cx)) * max(0, min(by, dy) - max(ay, cy))
            union = (bx - ax) * (by - ay) + (dx - cx) * (dy - cy) - intersection
            changed = changed or union <= 0 or intersection / union < 0.5
        if changed or result.status == "NO_PART":
            self.reset()
        self._last_box = tuple(object_box) if object_box is not None else None
        self._last_brand = result.brand
        if result.status == "NO_PART":
            return result
        self._history.append(result)
        ng_count = sum(item.status == "NG" for item in self._history)
        # REVIEW is not evidence of a clean part and cannot clear an NG latch.
        ok_count = sum(item.status == "OK" and not item.review_required for item in self._history)
        if self._stable_status is None:
            if ng_count >= self.ng_votes:
                self._stable_status = "NG"
            elif ok_count >= self.ok_votes:
                self._stable_status = "OK"
            else:
                return replace(result, status="REVIEW", review_required=True, input_issue="waiting_for_frames")
        if self._stable_status == "OK" and ng_count >= self.ng_votes:
            self._stable_status = "NG"
        elif self._stable_status == "NG" and ok_count >= self.ok_votes:
            self._stable_status = "OK"
        if self._stable_status == result.status:
            return result
        # Preserve this frame's score, brand and geometry. Pending transitions
        # are REVIEW, never a stale NG heatmap or a false green OK.
        return replace(result, status="REVIEW", review_required=True)


def iter_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    if not root.exists():
        return []
    return sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def _read_bgr(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or unreadable image: {path}")
    return image


def _clip_box(box: Sequence[int | float], image: np.ndarray, margin: float = 0.045) -> tuple[int, int, int, int]:
    """Return a safe, slightly padded object box in source-image coordinates."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = max(2.0, (x2 - x1) * margin)
    pad_y = max(2.0, (y2 - y1) * margin)
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def yolo_primary_box(image_path: str | Path, image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Read the largest class-0 YOLO box paired with an image, when available."""
    path = Path(image_path)
    parts = list(path.parts)
    try:
        image_index = next(index for index, value in enumerate(parts) if value.lower() == "images")
    except StopIteration:
        return None
    label_path = Path(*parts[:image_index], "labels", *parts[image_index + 1 :]).with_suffix(".txt")
    if not label_path.is_file():
        return None
    height, width = image.shape[:2]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = raw_line.split()
        if len(fields) < 5:
            continue
        try:
            class_id, cx, cy, box_width, box_height = map(float, fields[:5])
        except ValueError:
            continue
        if int(class_id) != 0 or box_width <= 0 or box_height <= 0:
            continue
        box = (
            int(round((cx - box_width / 2) * width)),
            int(round((cy - box_height / 2) * height)),
            int(round((cx + box_width / 2) * width)),
            int(round((cy + box_height / 2) * height)),
        )
        area = box_width * box_height
        if best is None or area > best[0]:
            best = (area, box)
    return best[1] if best else None


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize without stretching and return a mask excluding padded background."""
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / max(1, width), target_height / max(1, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    valid = np.zeros((target_height, target_width), dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    valid[top : top + resized_height, left : left + resized_width] = 255
    return canvas, valid


def _foreground_mask(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remove background inside the detector ROI using a conservative GrabCut mask."""
    original_size = (valid.shape[1], valid.shape[0])
    work_image = image
    work_valid = valid
    max_side = max(original_size)
    if max_side > 256:
        scale = 256.0 / max_side
        work_size = (max(1, int(round(original_size[0] * scale))), max(1, int(round(original_size[1] * scale))))
        work_image = cv2.resize(image, work_size, interpolation=cv2.INTER_AREA)
        work_valid = cv2.resize(valid, work_size, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(work_valid > 0)
    if not len(xs):
        return valid
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    inset = max(2, int(round(min(x2 - x1 + 1, y2 - y1 + 1) * 0.025)))
    rect = (x1 + inset, y1 + inset, max(2, x2 - x1 - inset * 2), max(2, y2 - y1 - inset * 2))
    grab_mask = np.full(work_valid.shape, cv2.GC_BGD, dtype=np.uint8)
    grab_mask[work_valid > 0] = cv2.GC_PR_BGD
    rx, ry, rw, rh = rect
    grab_mask[ry : ry + rh, rx : rx + rw] = cv2.GC_PR_FGD
    # A central seed prevents a uniform part from being discarded as background.
    seed_x1, seed_x2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
    seed_y1, seed_y2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
    grab_mask[seed_y1:seed_y2, seed_x1:seed_x2] = cv2.GC_FGD
    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(work_image, grab_mask, None, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
        foreground = np.where(
            (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        foreground = cv2.bitwise_and(foreground, work_valid)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean = np.zeros_like(foreground)
            cv2.drawContours(clean, [largest], -1, 255, -1)
            # Reject implausibly small masks instead of silently inspecting background.
            if cv2.countNonZero(clean) >= cv2.countNonZero(work_valid) * 0.22:
                return cv2.resize(clean, original_size, interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        pass
    # Safe fallback: exclude the detector-box boundary where background dominates.
    fallback = cv2.erode(work_valid, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.resize(fallback, original_size, interpolation=cv2.INTER_NEAREST)


def _prepare(
    image: np.ndarray,
    size: tuple[int, int],
    object_box: Sequence[int | float] | None = None,
    *,
    estimate_foreground: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    source_box = _clip_box(object_box, image) if object_box is not None else (0, 0, image.shape[1], image.shape[0])
    x1, y1, x2, y2 = source_box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("The inspected-part ROI is empty.")
    normalized, valid = _letterbox(crop, size)
    # GrabCut is intentionally used while fitting, where the consensus mask is
    # learned once.  Re-running it for every camera frame is both expensive and
    # unstable.  Live inspection intersects the cheap letterbox-valid mask with
    # that learned foreground mask below.
    foreground = _foreground_mask(normalized, valid) if estimate_foreground else valid
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    return lab, foreground, source_box


def _align(image: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Affine alignment absorbs small translation, rotation and scale changes."""
    image_gray = image[:, :, 0]
    reference_gray = reference[:, :, 0]
    alignment_scale = min(1.0, 192.0 / max(reference_gray.shape))
    if alignment_scale < 1.0:
        alignment_size = (
            max(1, int(round(reference_gray.shape[1] * alignment_scale))),
            max(1, int(round(reference_gray.shape[0] * alignment_scale))),
        )
        image_gray = cv2.resize(image_gray, alignment_size, interpolation=cv2.INTER_AREA)
        reference_gray = cv2.resize(reference_gray, alignment_size, interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(
            reference_gray,
            image_gray,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 2e-4),
            None,
            3,
        )
        if not np.isfinite(correlation) or not np.isfinite(warp).all() or np.linalg.det(warp[:, :2]) <= 0:
            raise AlignmentError('Image alignment failed: invalid transform')
        if alignment_scale < 1.0:
            warp[0, 2] /= alignment_scale
            warp[1, 2] /= alignment_scale
        aligned = cv2.warpAffine(
            image,
            warp,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        aligned_mask = None if mask is None else cv2.warpAffine(
            mask, warp, (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return aligned, aligned_mask
    except cv2.error as exc:
        raise AlignmentError('Image alignment failed: check image and reference') from exc


class NormalReferenceInspector:
    """Robust reference model using median appearance and median deviation."""

    VERSION = 2

    def __init__(
        self,
        reference: np.ndarray,
        deviation: np.ndarray,
        threshold: float,
        region_threshold: float,
        min_region_ratio: float,
        training_count: int,
        object_mask: np.ndarray | None = None,
        model_version: int = VERSION,
    ) -> None:
        self.reference = reference.astype(np.float32)
        self.deviation = np.maximum(deviation.astype(np.float32), 3.0)
        self.threshold = float(threshold)
        self.region_threshold = float(region_threshold)
        self.min_region_ratio = float(min_region_ratio)
        self.training_count = int(training_count)
        self.object_mask = (
            (object_mask > 0).astype(np.uint8) * 255
            if object_mask is not None
            else np.full(self.reference.shape[:2], 255, dtype=np.uint8)
        )
        self.model_version = int(model_version)

    @property
    def size(self) -> tuple[int, int]:
        return self.reference.shape[1], self.reference.shape[0]

    @classmethod
    def fit(
        cls,
        image_paths: Iterable[str | Path],
        *,
        inspection_size: int = 512,
        min_images: int = 5,
        threshold_percentile: float = 99.0,
        min_region_ratio: float = 0.002,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> "NormalReferenceInspector":
        paths = [Path(path) for path in image_paths]
        if len(paths) < min_images:
            raise ValueError(f"Need at least {min_images} known-good images; received {len(paths)}")
        size = (int(inspection_size), int(inspection_size))
        prepared: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for index, path in enumerate(paths):
            source = _read_bgr(path)
            box = yolo_primary_box(path, source)
            normalized, mask, _ = _prepare(source, size, box)
            prepared.append(normalized)
            masks.append(mask)
            if progress_callback:
                progress_callback(0.55 * ((index + 1) / len(paths)), "preparing_images")
        initial = np.median(np.stack(prepared), axis=0).astype(np.float32)
        if progress_callback:
            progress_callback(0.58, "building_initial_reference")
        aligned_pairs = []
        for index, (image, mask) in enumerate(zip(prepared, masks)):
            aligned_pairs.append(_align(image, initial, mask))
            if progress_callback:
                progress_callback(0.58 + 0.24 * ((index + 1) / len(prepared)), "aligning_images")
        aligned = [pair[0] for pair in aligned_pairs]
        aligned_masks = [pair[1] for pair in aligned_pairs if pair[1] is not None]
        stack = np.stack(aligned)
        reference = np.median(stack, axis=0)
        deviation = np.median(np.abs(stack - reference), axis=0) * 1.4826
        object_mask = (np.mean(np.stack(aligned_masks) > 0, axis=0) >= 0.60).astype(np.uint8) * 255
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        if progress_callback:
            progress_callback(0.86, "calculating_normal_variation")
        training_scores = []
        for index, image in enumerate(aligned):
            residual = np.mean(np.abs(image - reference) / (deviation + 6.0), axis=2)
            values = residual[object_mask > 0]
            training_scores.append(float(np.percentile(values, threshold_percentile)))
            if progress_callback:
                progress_callback(0.86 + 0.13 * ((index + 1) / len(aligned)), "calibrating_threshold")
        learned = float(np.percentile(training_scores, 99.0))
        region_threshold = max(2.5, learned * 1.35)
        decision_threshold = max(0.35, float(np.percentile(training_scores, 99.0)) * 1.15)
        if progress_callback:
            progress_callback(1.0, "reference_ready")
        return cls(reference, deviation, decision_threshold, region_threshold, min_region_ratio, len(paths), object_mask)

    @classmethod
    def fit_with_evaluation(
        cls,
        image_paths: Iterable[str | Path],
        *,
        inspection_size: int = 512,
        min_images: int = 5,
        threshold_percentile: float = 99.0,
        min_region_ratio: float = 0.002,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple["NormalReferenceInspector", dict]:
        """Fit a production reference and report truthful normal-only metrics.

        Without confirmed NG validation images, mAP/precision/recall/F1 would be
        misleading.  We instead measure a deterministic known-good holdout and
        report acceptance, false-NG, and anomaly-score calibration.  The final
        production reference is fitted again using every reviewed good image.
        """
        paths = sorted((Path(path) for path in image_paths), key=lambda item: str(item).lower())
        if len(paths) < min_images:
            raise ValueError(f"Need at least {min_images} known-good images; received {len(paths)}")

        holdout_count = 0
        if len(paths) >= max(min_images + 2, 8):
            holdout_count = max(2, min(len(paths) // 5, len(paths) - min_images))
        validation_paths = paths[-holdout_count:] if holdout_count else paths
        fit_paths = paths[:-holdout_count] if holdout_count else paths
        if progress_callback:
            progress_callback(0.01, "preparing_calibration")
        calibration_model = cls.fit(
            fit_paths,
            inspection_size=inspection_size,
            min_images=min_images,
            threshold_percentile=threshold_percentile,
            min_region_ratio=min_region_ratio,
            progress_callback=(
                (lambda value, phase: progress_callback(0.02 + value * 0.43, f"calibration_{phase}"))
                if progress_callback else None
            ),
        )

        scores: list[float] = []
        accepted = 0
        for index, path in enumerate(validation_paths):
            image = _read_bgr(path)
            result = calibration_model.inspect(image, object_box=yolo_primary_box(path, image))
            scores.append(float(result.anomaly_score))
            accepted += int(result.status == "OK")
            if progress_callback:
                progress_callback(0.45 + 0.15 * ((index + 1) / len(validation_paths)), "validating_holdout")
        count = len(scores)
        metrics = {
            "metric_scope": "known_good_holdout" if holdout_count else "training_calibration",
            "independent_validation": bool(holdout_count),
            "validation_images": count,
            "normal_acceptance_rate": round(accepted / max(1, count), 4),
            "false_ng_rate": round((count - accepted) / max(1, count), 4),
            "anomaly_score_mean": round(float(np.mean(scores)), 4) if scores else None,
            "anomaly_score_p95": round(float(np.percentile(scores, 95)), 4) if scores else None,
            "anomaly_score_max": round(float(np.max(scores)), 4) if scores else None,
            "calibration_threshold": round(float(calibration_model.threshold), 4),
            "defect_metrics_available": False,
            "defect_metrics_reason": "Confirmed NG validation images are not available yet.",
        }
        if progress_callback:
            progress_callback(0.61, "building_production_reference")
        production_model = cls.fit(
            paths,
            inspection_size=inspection_size,
            min_images=min_images,
            threshold_percentile=threshold_percentile,
            min_region_ratio=min_region_ratio,
            progress_callback=(
                (lambda value, phase: progress_callback(0.61 + value * 0.37, f"production_{phase}"))
                if progress_callback else None
            ),
        )
        metrics["production_threshold"] = round(float(production_model.threshold), 4)
        if progress_callback:
            progress_callback(0.99, "finalizing_reference")
        return production_model, metrics

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            version=np.array([self.VERSION], dtype=np.int16),
            reference=self.reference.astype(np.float16),
            deviation=self.deviation.astype(np.float16),
            threshold=np.array([self.threshold], dtype=np.float32),
            region_threshold=np.array([self.region_threshold], dtype=np.float32),
            min_region_ratio=np.array([self.min_region_ratio], dtype=np.float32),
            training_count=np.array([self.training_count], dtype=np.int32),
            object_mask=self.object_mask.astype(np.uint8),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "NormalReferenceInspector":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["version"][0]) if "version" in data else 1
            return cls(
                data["reference"], data["deviation"], float(data["threshold"][0]),
                float(data["region_threshold"][0]), float(data["min_region_ratio"][0]),
                int(data["training_count"][0]),
                data["object_mask"] if "object_mask" in data else None,
                version,
            )

    def inspect(
        self,
        image: np.ndarray,
        *,
        object_box: Sequence[int | float] | None = None,
        brand: str | None = None,
        brand_confidence: float | None = None,
    ) -> InspectionResult:
        prepared, frame_mask, source_box = _prepare(
            image, self.size, object_box, estimate_foreground=False
        )
        try:
            prepared, frame_mask = _align(prepared, self.reference, frame_mask)
        except AlignmentError:
            return InspectionResult('REVIEW',0.0,0.0,self.threshold,[],brand,brand_confidence,
                                    review_required=True,input_issue='alignment_failed')
        active_mask = cv2.bitwise_and(self.object_mask, frame_mask) if frame_mask is not None else self.object_mask
        if not np.any(active_mask):
            return InspectionResult('REVIEW',0.0,0.0,self.threshold,[],brand,brand_confidence,
                                    review_required=True,input_issue='inspection_area_not_visible')
        residual = np.mean(np.abs(prepared - self.reference) / (self.deviation + 6.0), axis=2)
        if not np.isfinite(residual).all() or not all(np.isfinite(v) and v>0 for v in (self.threshold,self.region_threshold)):
            return InspectionResult('REVIEW',0.0,0.0,0.0,[],brand,brand_confidence,
                                    review_required=True,input_issue='invalid_model_output')
        residual = cv2.GaussianBlur(residual, (7, 7), 0)
        mask = cv2.bitwise_and((residual >= self.region_threshold).astype(np.uint8) * 255, active_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        min_area = self.reference.shape[0] * self.reference.shape[1] * self.min_region_ratio
        source_x1, source_y1, source_x2, source_y2 = source_box
        source_width, source_height = source_x2 - source_x1, source_y2 - source_y1
        scale = min(self.reference.shape[1] / max(1, source_width), self.reference.shape[0] / max(1, source_height))
        rendered_width, rendered_height = source_width * scale, source_height * scale
        pad_x = (self.reference.shape[1] - rendered_width) / 2
        pad_y = (self.reference.shape[0] - rendered_height) / 2
        regions: list[InspectionRegion] = []
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            local_score = float(np.percentile(residual[y : y + height, x : x + width], 95))
            mapped_x = max(source_x1, int(round(source_x1 + (x - pad_x) / scale)))
            mapped_y = max(source_y1, int(round(source_y1 + (y - pad_y) / scale)))
            mapped_width = min(source_x2 - mapped_x, max(1, int(round(width / scale))))
            mapped_height = min(source_y2 - mapped_y, max(1, int(round(height / scale))))
            regions.append(InspectionRegion(
                mapped_x, mapped_y, mapped_width, mapped_height, round(local_score, 4),
            ))

        valid_values = residual[active_mask > 0]
        fallback_score = float(np.percentile(valid_values, 99.5)) if valid_values.size else 0.0
        anomaly_score = max((region.score for region in regions), default=fallback_score)
        is_ng = bool(regions) and anomaly_score >= self.threshold
        margin = abs(anomaly_score - self.threshold) / max(self.threshold, 1e-6)
        confidence = float(np.clip(0.5 + margin * 0.35, 0.5, 0.995))
        return InspectionResult(
            "NG" if is_ng else "OK", round(confidence, 4), round(anomaly_score, 4),
            round(self.threshold, 4), regions if is_ng else [], brand, brand_confidence,
        )

    def inspect_path(self, path: str | Path, **kwargs) -> InspectionResult:
        return self.inspect(_read_bgr(path), **kwargs)

    @staticmethod
    def draw_result(
        image: np.ndarray,
        result: InspectionResult,
        *,
        object_box: tuple[int, int, int, int] | None = None,
        object_label: str = "Part",
        object_confidence: float | None = None,
    ) -> np.ndarray:
        output = image.copy()
        visible_status = ("REPOSITION" if result.input_issue == "part_clipped_by_frame" else "NOT INSPECTED") if result.input_issue else ("REVIEW" if result.review_required and result.status != "NG" else result.status)
        color = (0, 0, 255) if result.status == "NG" else ((0, 190, 255) if result.review_required else (40, 210, 80))
        if object_box is not None:
            x1, y1, x2, y2 = object_box
            x1 = max(0, min(int(x1), output.shape[1] - 1))
            y1 = max(0, min(int(y1), output.shape[0] - 1))
            x2 = max(x1 + 1, min(int(x2), output.shape[1] - 1))
            y2 = max(y1 + 1, min(int(y2), output.shape[0] - 1))
            if result.anomaly_map and (result.status == "NG" or result.review_required):
                raw_heatmap = np.asarray(result.anomaly_map, dtype=np.float32)
                if raw_heatmap.ndim == 2 and raw_heatmap.size:
                    roi_width, roi_height = x2 - x1, y2 - y1
                    heatmap = cv2.resize(raw_heatmap, (roi_width, roi_height), interpolation=cv2.INTER_CUBIC)
                    baseline = (
                        float(result.threshold)
                        if result.status == "NG"
                        else max(0.0, float(result.anomaly_score) * 0.72)
                    )
                    denominator = max(1e-6, float(heatmap.max()) - baseline)
                    activation = np.clip((heatmap - baseline) / denominator, 0.0, 1.0)
                    red_layer = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
                    red_layer[..., 2] = np.uint8(80 + activation * 175)
                    red_layer[..., 1] = np.uint8(activation * 42)
                    alpha = (activation * 0.48)[..., None]
                    roi_view = output[y1:y2, x1:x2].astype(np.float32)
                    output[y1:y2, x1:x2] = np.uint8(roi_view * (1.0 - alpha) + red_layer.astype(np.float32) * alpha)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            detector_text = (
                f" {float(object_confidence):.0%}"
                if object_confidence is not None
                else ""
            )
            label = f"{object_label}{detector_text} | {visible_status}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            label_top = max(0, y1 - label_height - 12)
            label_bottom = label_top + label_height + 12
            cv2.rectangle(
                output,
                (x1, label_top),
                (min(output.shape[1] - 1, x1 + label_width + 14), label_bottom),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 7, label_bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (10, 18, 22),
                2,
                cv2.LINE_AA,
            )
            if result.brand:
                brand_label = f"Brand: {result.brand} {result.brand_confidence or 0:.0%}"
                (brand_width, brand_height), _ = cv2.getTextSize(
                    brand_label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
                )
                brand_top = min(output.shape[0] - brand_height - 12, max(y1 + 5, label_bottom + 4))
                cv2.rectangle(
                    output,
                    (x1, brand_top),
                    (min(output.shape[1] - 1, x1 + brand_width + 14), brand_top + brand_height + 12),
                    (0, 170, 255),
                    -1,
                )
                cv2.putText(
                    output,
                    brand_label,
                    (x1 + 7, brand_top + brand_height + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (10, 18, 22),
                    2,
                    cv2.LINE_AA,
                )
        else:
            # Never report an OK decision when the inspected part is not located.
            cv2.putText(
                output,
                "NO PART",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 180, 255),
                3,
                cv2.LINE_AA,
            )
        if object_box is not None:
            if not result.input_issue and (result.status == "NG" or result.review_required):
                for region in result.regions:
                    left = max(0, min(output.shape[1] - 1, int(region.x)))
                    top = max(0, min(output.shape[0] - 1, int(region.y)))
                    right = max(left, min(output.shape[1] - 1, int(region.x + region.width)))
                    bottom = max(top, min(output.shape[0] - 1, int(region.y + region.height)))
                    cv2.rectangle(output, (left, top), (right, bottom), color, 2)
            issue_messages = {
                "waiting_for_frames": "Checking - waiting for stable frames",
                "alignment_failed": "Image alignment failed - not checked",
                "invalid_model_output": "Invalid model output - not checked",
                "recipe_area_mismatch": "Inspection area does not match model",
                "part_clipped_by_frame": "Full part required - move away from frame edge",
                "recipe_unverified": "Job model has not been checked",
                "recipe_model_mismatch": "Model does not match this job",
                "recipe_brand_unconfirmed": "Brand not found - not checked",
                "recipe_brand_mismatch": "Wrong brand for this job - not checked",
            }
            score_text = issue_messages.get(result.input_issue, "Image not ready for inspection") if result.input_issue else f"Score {result.anomaly_score:.3f} / NG > {result.threshold:.3f}"
            if not result.input_issue and not result.regions and (result.status == "NG" or result.review_required):
                score_text += " | Location not found"
            if result.inspection_areas and result.inspection_areas != [[0., 0., 1., 1.]] or result.inspection_area and result.inspection_area != [0., 0., 1., 1.]:
                score_text += " | Selected area only"
            cv2.rectangle(output, (0, output.shape[0] - 54), (output.shape[1] - 1, output.shape[0] - 30), (20, 24, 30), -1)
            cv2.putText(output, score_text, (10, output.shape[0] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return output
