"""Good-only, aligned and position-aware visual anomaly inspection."""
from __future__ import annotations

from pathlib import Path
import re
import hashlib
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from part_locator import analyze_part, part_class_id
from inspection_area import checked_area, area_grid, checked_areas, areas_grid
from normal_reference import AlignmentError, InspectionRegion, InspectionResult, NormalReferenceInspector, _foreground_mask, yolo_primary_box


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _crop(image: np.ndarray, box: Sequence[int | float] | None):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Empty image")
    source = (0, 0, width, height)
    if box is not None:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        candidate = (max(0, min(width, x1)), max(0, min(height, y1)),
                     max(0, min(width, x2)), max(0, min(height, y2)))
        if candidate[2] > candidate[0] and candidate[3] > candidate[1]:
            source = candidate
    x1, y1, x2, y2 = source
    return image[y1:y2, x1:x2], source


def _align_bgr(image: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None, *, return_warp=False):
    """Affine alignment absorbs normal camera translation, rotation and scale."""
    moving = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    fixed = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    scale = min(1.0, 192.0 / max(fixed.shape))
    if scale < 1.0:
        size = (max(8, int(reference.shape[1] * scale)), max(8, int(reference.shape[0] * scale)))
        moving = cv2.resize(moving, size, interpolation=cv2.INTER_AREA)
        fixed = cv2.resize(fixed, size, interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(fixed, moving, warp, cv2.MOTION_AFFINE,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 28, 1e-4), None, 3)
        if not np.isfinite(correlation) or not np.isfinite(warp).all() or np.linalg.det(warp[:, :2]) <= 0:
            raise AlignmentError("Image alignment failed: invalid transform")
        if scale < 1.0:
            warp[:, 2] /= scale
        aligned = cv2.warpAffine(image, warp, (reference.shape[1], reference.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT_101)
        aligned_mask = None if mask is None else cv2.warpAffine(
            mask, warp, (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (aligned, aligned_mask, warp) if return_warp else (aligned, aligned_mask)
    except cv2.error as exc:
        raise AlignmentError("Image alignment failed: check image and reference") from exc


def _camera_heatmap(heatmap, alignment_warp, input_size):
    """ECC warp maps template coordinates to the original, unaligned ROI."""
    warp = np.asarray(alignment_warp, dtype=np.float32).copy()
    height, width = heatmap.shape
    scale = np.diag([width / input_size, height / input_size, 1.0])
    transform = np.eye(3)
    transform[:2] = warp
    mapped = (scale @ transform @ np.linalg.inv(scale))[:2].astype(np.float32)
    return cv2.warpAffine(heatmap, mapped, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class YoloPatchEncoder:
    """Frozen multi-scale YOLO backbone; detection classes never decide OK/NG."""

    def __init__(self, model_path: str | Path, input_size: int = 256, grid_size: int = 32):
        self.model_path = str(Path(model_path).resolve())
        self.input_size, self.grid_size = int(input_size), int(grid_size)
        self.wrapper = YOLO(self.model_path)
        self.model = self.wrapper.model.eval().cpu()
        self.outputs: dict[int, torch.Tensor] = {}
        self.layer_indices = sorted(set(min(index, len(self.model.model) - 2) for index in (2, 4)))
        self.hooks = [self.model.model[index].register_forward_hook(self._capture(index)) for index in self.layer_indices]
        self.alignment_template: np.ndarray | None = None
        self.alignment_reference_path: Path | None = None
        self.object_mask: np.ndarray | None = None

    def _capture(self, index: int):
        def capture(_module, _inputs, output) -> None:
            if isinstance(output, torch.Tensor):
                self.outputs[index] = output
            elif isinstance(output, (list, tuple)):
                tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
                if tensor is not None:
                    self.outputs[index] = tensor
        return capture

    def locate(self, image: np.ndarray):
        with torch.inference_mode():
            result = analyze_part(self.wrapper, part_class_id(self.wrapper.names), image)
        return result["box"] if result else None

    def _roi(self, image: np.ndarray, box=None):
        height, width = image.shape[:2]
        if box is not None:
            box = tuple(max(0, min(width if i % 2 == 0 else height, int(round(float(v))))) for i, v in enumerate(box))
            if box[2] <= box[0] or box[3] <= box[1]:
                box = None
        roi, source_box = _crop(image, box or self.locate(image))
        resized = cv2.resize(roi, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        self.last_alignment_warp = None
        if self.alignment_template is not None:
            resized, _, self.last_alignment_warp = _align_bgr(resized, self.alignment_template, return_warp=True)
        return resized, source_box

    def fit_alignment(self, paths: list[Path], boxes: dict[str, tuple[int, int, int, int]] | None = None, reference_path=None) -> None:
        if not paths:
            return
        indices = np.linspace(0, len(paths) - 1, min(48, len(paths)), dtype=int)
        images, masks, accepted_paths = [], [], []
        valid = np.full((self.input_size, self.input_size), 255, np.uint8)
        for index in indices:
            path = paths[int(index)]
            source = _read_bgr(path)
            box = (boxes or {}).get(str(path)) or yolo_primary_box(path, source) or self.locate(source)
            if box is None:
                continue
            roi, _ = _crop(source, box)
            resized = cv2.resize(roi, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            images.append(resized)
            accepted_paths.append(path)
            masks.append(_foreground_mask(resized, valid))
        if not images:
            raise ValueError("No usable Part ROI was found in Known-good images.")
        # Establish coordinates with a real fit image. A pixel median of moving
        # parts erases texture/logo detail before ECC has a chance to align it.
        # Keep the anchor unchanged after alignment, including an explicit ROI anchor.
        self.alignment_reference_path = accepted_paths[0]
        initial = images[0].copy()
        if reference_path is not None:
            self.alignment_reference_path = Path(reference_path)
            source = _read_bgr(self.alignment_reference_path)
            roi, _ = _crop(source, boxes[str(reference_path)])
            initial = cv2.resize(roi, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        self.alignment_template = initial
        pairs = [_align_bgr(image, self.alignment_template, mask) for image, mask in zip(images, masks)]
        mask = (np.mean(np.stack([item[1] > 0 for item in pairs]), axis=0) >= 0.55).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        if cv2.countNonZero(mask) < mask.size * 0.22:
            mask[:] = 255
            mask[:4] = mask[-4:] = mask[:, :4] = mask[:, -4:] = 0
        self.object_mask = mask

    def encode(self, image: np.ndarray, box=None):
        resized, source_box = self._roi(image, box)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        self.outputs.clear()
        with torch.inference_mode():
            self.model(tensor)
            pooled = [F.normalize(F.adaptive_avg_pool2d(self.outputs[index].float(), (self.grid_size, self.grid_size)), p=2, dim=1)
                      for index in self.layer_indices if index in self.outputs]
            if not pooled:
                raise RuntimeError("YOLO backbone feature hooks did not produce output")
            # Deep semantics alone can intentionally ignore a tiny stain. Add a
            # normalized appearance/texture branch so small local colour and
            # edge changes remain measurable without teaching defect classes.
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
            appearance = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0)
            mean = F.adaptive_avg_pool2d(appearance, (self.grid_size, self.grid_size))
            mean_square = F.adaptive_avg_pool2d(appearance.square(), (self.grid_size, self.grid_size))
            deviation = torch.sqrt(torch.clamp(mean_square - mean.square(), min=0.0))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            edge = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            edge_tensor = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0)
            edge_pooled = F.adaptive_avg_pool2d(edge_tensor, (self.grid_size, self.grid_size))
            appearance_features = F.normalize(torch.cat((mean, deviation, edge_pooled), dim=1), p=2, dim=1)
            features = F.normalize(torch.cat((*pooled, appearance_features), dim=1), p=2, dim=1)
        return features[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32), source_box

    def active_grid(self):
        if self.object_mask is None:
            return np.ones((self.grid_size, self.grid_size), bool)
        active = cv2.resize(self.object_mask, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA) >= 96
        return active if active.sum() >= 8 else np.ones_like(active, bool)


def _split_reference_paths(paths, min_images):
    # Roboflow exports of the same source frame must not span train/holdout.
    groups = {}
    for path in paths:
        key = re.split(r"\.rf\.", path.stem.lower(), maxsplit=1)[0]
        groups.setdefault(key, []).append(path)
    keys = sorted(groups)
    rng = np.random.default_rng(20260901)
    keys = [keys[int(i)] for i in rng.permutation(len(keys))]
    independent = len(keys) >= max(15, min_images + 5)
    if not independent:
        return paths, paths, paths, False
    validation_count, calibration_count = max(2,len(keys)//5), max(2,len(keys)//6)
    flatten = lambda selected: [p for key in selected for p in groups[key]]
    return (flatten(keys[:-(validation_count+calibration_count)]),
            flatten(keys[-(validation_count+calibration_count):-validation_count]),
            flatten(keys[-validation_count:]), True)


class DeepPatchReference:
    VERSION, ENGINE = 7, "deep_patch"

    def __init__(self, memory, threshold, region_threshold, training_count, backbone_path,
                 input_size=256, grid_size=32, min_region_ratio=0.00035,
                 review_threshold=None, active_mask=None, alignment_template=None,
                 object_mask=None, position_scale=None, model_version=VERSION, inspection_area=None, inspection_areas=None):
        self.memory = np.asarray(memory, np.float32)
        self.threshold, self.region_threshold = float(threshold), float(region_threshold)
        self.review_threshold = float(review_threshold if review_threshold is not None else threshold * 0.93)
        self.training_count, self.backbone_path = int(training_count), str(backbone_path)
        self.input_size, self.grid_size = int(input_size), int(grid_size)
        self.min_region_ratio = float(min_region_ratio)
        self.active_mask = np.asarray(active_mask).astype(bool) if active_mask is not None else np.ones((grid_size, grid_size), bool)
        self.alignment_template = None if alignment_template is None else np.asarray(alignment_template, np.uint8)
        self.object_mask = None if object_mask is None else np.asarray(object_mask, np.uint8)
        self.position_scale = None if position_scale is None else np.asarray(position_scale, np.float32).reshape(-1)
        self.model_version, self._encoder = int(model_version), None
        self.inspection_area = checked_area(inspection_area)
        self.inspection_areas = checked_areas(inspection_areas, self.inspection_area)

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = YoloPatchEncoder(self.backbone_path, self.input_size, self.grid_size)
            self._encoder.alignment_template, self._encoder.object_mask = self.alignment_template, self.object_mask
        return self._encoder

    @staticmethod
    def _distances(patches, memory):
        query, bank = torch.from_numpy(np.asarray(patches, np.float32)), torch.from_numpy(np.asarray(memory, np.float32))
        with torch.inference_mode():
            if bank.ndim == 3:
                if query.shape[0] != bank.shape[0]:
                    raise ValueError("Reference position grid does not match the current encoder.")
                squared = ((query[:, None, :] - bank) ** 2).sum(dim=2)
                return torch.sqrt(torch.clamp(squared, min=0)).min(dim=1).values.numpy()
            best = torch.full((query.shape[0],), float("inf"))
            for start in range(0, bank.shape[0], 2048):
                best = torch.minimum(best, torch.cdist(query, bank[start:start + 2048]).min(dim=1).values)
            return best.numpy()

    @staticmethod
    def _memory(maps, limit=64, *, return_indices=False):
        indices = np.linspace(0, len(maps) - 1, min(limit,len(maps)), dtype=int)
        memory = np.stack([maps[int(i)].reshape(-1, maps[int(i)].shape[-1]) for i in indices], axis=1).astype(np.float16)
        return (memory, indices) if return_indices else memory

    @classmethod
    def _fit_position_scale(cls, fit_maps, fit_paths, memory, memory_indices, active):
        # Learn normalization only from fit data, excluding the query's source
        # group. Calibration images then set a threshold on unseen scores.
        groups = [re.split(r"\.rf\.", path.stem.lower(), maxsplit=1)[0] for path in fit_paths]
        bank_groups = np.array([groups[int(i)] for i in memory_indices])
        distances = []
        for item, group in zip(fit_maps, groups):
            keep = bank_groups != group
            if not np.any(keep):
                continue
            distances.append(cls._distances(item.reshape(-1,item.shape[-1]),memory[:,keep,:]))
        if not distances:
            raise ValueError("Need multiple independent source images to normalize the Reference; copies are not independent evidence.")
        raw_stack = np.stack(distances)
        floor = max(1e-4, float(np.percentile(raw_stack[:,active.reshape(-1)],10)))
        return np.maximum(np.percentile(raw_stack,99.0,axis=0)*1.03,floor).astype(np.float32)

    @staticmethod
    def _image_score(distances, active):
        values = np.asarray(distances)[np.asarray(active).reshape(-1)]
        if not values.size:
            return 0.0
        count = max(1, int(np.ceil(values.size * 0.005)))
        return float(np.mean(np.partition(values, -count)[-count:]))

    @classmethod
    def fit_with_evaluation(cls, image_paths: Iterable[str | Path], *, backbone_path: str | Path,
                            min_images=5, inspection_size=256, min_region_ratio=0.00035,
                            progress_callback: Callable[[float, str], None] | None = None, inspection_area=None, inspection_areas=None, alignment_reference=None):
        inspection_area = checked_area(inspection_area)
        inspection_areas = checked_areas(inspection_areas, inspection_area)
        source_paths = sorted((Path(path) for path in image_paths), key=lambda item: str(item).lower())
        if len(source_paths) < min_images:
            raise ValueError(f"Need at least {min_images} Known-good images; received {len(source_paths)}")
        encoder = YoloPatchEncoder(backbone_path, inspection_size, 32)
        boxes: dict[str, tuple[int, int, int, int]] = {}
        paths: list[Path] = []
        excluded = []
        for index, path in enumerate(source_paths):
            image = _read_bgr(path)
            box = encoder.locate(image)
            if box is None:
                excluded.append({"path": str(path.resolve()), "reason": "no_part"})
            elif box[0] <= 1 or box[1] <= 1 or box[2] >= image.shape[1] - 1 or box[3] >= image.shape[0] - 1:
                # Match the inference REPOSITION condition before learning or
                # scoring. A partial part is not valid Known-good evidence.
                excluded.append({"path": str(path.resolve()), "reason": "part_clipped_by_frame", "box": list(box)})
            else:
                boxes[str(path)] = box
                paths.append(path)
            if progress_callback:
                progress_callback(0.01 + 0.09 * ((index + 1) / len(source_paths)), "screening_part_roi")
        if len(paths) < min_images:
            raise ValueError(
                f"Only {len(paths)} images contain a usable Part ROI; need at least {min_images}. "
                "Review the Part labels or retrain the Part Locator."
            )
        fit_paths, calibration_paths, validation_paths, independent = _split_reference_paths(paths, min_images)
        anchor_path = None
        if alignment_reference is not None:
            anchor_path = Path(alignment_reference["path"])
            if anchor_path not in paths or hashlib.sha256(anchor_path.read_bytes()).hexdigest() != alignment_reference["sha256"]:
                raise ValueError("Reference image changed or is not a usable training image; load the area preview again")
            if "box" in alignment_reference:
                preview_box = np.asarray(alignment_reference["box"], dtype=float)
                if preview_box.shape != (4,) or not np.isfinite(preview_box).all() or not np.allclose(preview_box, boxes[str(anchor_path)], atol=1):
                    raise ValueError("Part location changed since area preview; load the preview again")
            # Reserve the entire source group for fit, never tune alignment on validation.
            key = re.split(r"\.rf\.", anchor_path.stem.lower(), maxsplit=1)[0]
            group = [p for p in paths if re.split(r"\.rf\.", p.stem.lower(), maxsplit=1)[0] == key]
            if independent:
                fit_paths = sorted(set(fit_paths + group))
                calibration_paths = [p for p in calibration_paths if p not in group]
                validation_paths = [p for p in validation_paths if p not in group]
                if not calibration_paths or not validation_paths:
                    raise ValueError("Not enough independent images after reserving the reference image")
        if hasattr(encoder, "fit_alignment"):
            if progress_callback:
                progress_callback(0.105, "learning_alignment_and_foreground")
            if anchor_path is None:
                encoder.fit_alignment(fit_paths, boxes)
            else:
                encoder.fit_alignment(fit_paths, boxes, reference_path=anchor_path)
            selected_anchor = getattr(encoder, "alignment_reference_path", None)
            if alignment_reference is None and selected_anchor is not None:
                alignment_reference = {"path": str(selected_anchor.resolve()),
                    "sha256": hashlib.sha256(selected_anchor.read_bytes()).hexdigest(),
                    "box": list(boxes[str(selected_anchor)]), "selection": "first_usable_fit_image"}
        active = encoder.active_grid() if hasattr(encoder, "active_grid") else np.ones((encoder.grid_size, encoder.grid_size), bool)
        active = active & areas_grid(inspection_areas, active.shape)
        if active.sum() < 8:
            raise ValueError("Inspection area covers fewer than 8 usable feature cells; enlarge the area")

        def encode(items, start, width, phase):
            maps = []
            for index, path in enumerate(items):
                image = _read_bgr(path)
                features = encoder.encode(image, boxes[str(path)])[0]
                if not np.isfinite(features).all():
                    raise ValueError(f"Invalid model features for image: {path.name}")
                maps.append(features)
                if progress_callback:
                    progress_callback(start + width * ((index + 1) / max(1, len(items))), phase)
            return maps

        fit_maps = encode(fit_paths, 0.12, 0.34, "extracting_multiscale_features")
        calibration_memory, memory_indices = cls._memory(fit_maps, return_indices=True)
        position_scale = cls._fit_position_scale(fit_maps, fit_paths, calibration_memory, memory_indices, active)
        calibration_maps = encode(calibration_paths, 0.47, 0.14, "calibrating_threshold")
        calibration_raw = [cls._distances(item.reshape(-1, item.shape[-1]), calibration_memory) for item in calibration_maps]
        calibration_distances = [item / position_scale for item in calibration_raw]
        if not all(np.isfinite(item).all() for item in calibration_distances):
            raise ValueError("Invalid calibration distances; model was not saved")
        calibration_scores = [cls._image_score(item, active) for item in calibration_distances]
        threshold = max(0.01, max(calibration_scores) * 1.06 if calibration_scores else 0.25)
        review_threshold = max(0.005, threshold * 0.93)
        patch_scores = np.concatenate([item[active.reshape(-1)] for item in calibration_distances]) if calibration_distances else np.array([threshold])
        region_threshold = max(float(np.percentile(patch_scores, 99.5)) * 1.02, threshold * 0.65)
        validation_maps = encode(validation_paths, 0.62, 0.15, "validating_holdout")
        validation_scores = [cls._image_score(cls._distances(item.reshape(-1, item.shape[-1]), calibration_memory) / position_scale, active) for item in validation_maps]
        if not all(np.isfinite(score) for score in validation_scores):
            raise ValueError("Invalid validation scores; model was not saved")
        accepted = sum(score < review_threshold for score in validation_scores)
        rejected = sum(score > threshold for score in validation_scores)
        pending_review = sum(review_threshold <= score <= threshold for score in validation_scores)
        if progress_callback:
            progress_callback(0.78, "building_production_memory")
        # Freeze the exact bank used for calibration and independent validation.
        # Rebuilding a capped bank here evicts previously calibrated exemplars
        # and leaks validation images into the deployed model.
        memory = calibration_memory
        model = cls(memory, threshold, region_threshold, len(paths), str(backbone_path), inspection_size,
                    encoder.grid_size, min_region_ratio, review_threshold, active,
                    getattr(encoder, "alignment_template", None), getattr(encoder, "object_mask", None), position_scale,
                    inspection_area=inspection_area, inspection_areas=inspection_areas)
        acceptance = accepted / max(1, len(validation_scores))
        metrics = {
            "engine": cls.ENGINE, "pipeline_version": cls.VERSION,
            "feature_mode": "aligned_masked_multiscale_position_aware",
            "metric_scope": "independent_known_good_holdout" if independent else "training_calibration",
            "independent_validation": independent, "calibration_images": len(calibration_paths),
            "fit_images": len(fit_paths),
            "production_memory_scope": "fit_only_frozen",
            "roi_policy": "shared_runtime_locator_complete_part",
            "split_policy": "source_group_disjoint",
            "validation_in_production_memory": False if independent else True,
            "source_images_considered": len(source_paths), "usable_training_images": len(paths),
            "images_excluded_no_part": sum(item["reason"] == "no_part" for item in excluded),
            "images_excluded_clipped_part": sum(item["reason"] == "part_clipped_by_frame" for item in excluded),
            "images_excluded_total": len(excluded),
            "validation_images": len(validation_paths), "normal_acceptance_rate": round(acceptance, 4),
            "strict_ok_rate": round(acceptance, 4),
            "false_ng_rate": round(rejected / max(1, len(validation_scores)), 4),
            "review_rate": round(pending_review / max(1, len(validation_scores)), 4),
            "anomaly_score_mean": round(float(np.mean(validation_scores)), 4) if validation_scores else None,
            "anomaly_score_p95": round(float(np.percentile(validation_scores, 95)), 4) if validation_scores else None,
            "anomaly_score_max": round(float(np.max(validation_scores)), 4) if validation_scores else None,
            "production_threshold": round(threshold, 4), "review_threshold": round(review_threshold, 4),
            "region_threshold": round(region_threshold, 4), "memory_positions": int(memory.shape[0]),
            "memory_samples_per_position": int(memory.shape[1]), "foreground_coverage": round(float(active.mean()), 4),
            "threshold_calibration": "fit_group_excluded_scale_then_calibration_threshold",
            "defect_metrics_available": False,
            "inspection_area": inspection_area, "inspection_areas": inspection_areas, "alignment_reference": alignment_reference,
            "inspection_area_coordinates": "aligned_part_fraction",
            "defect_metrics_reason": "Confirmed NG validation images are not available yet.",
        }
        def manifest_items(items):
            return [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "source_group": re.split(r"\.rf\.", path.stem.lower(), maxsplit=1)[0],
                     "box": list(boxes[str(path)])} for path in items]
        model.training_manifest = {
            "pipeline_version": cls.VERSION, "independent_validation": independent,
            "fit": manifest_items(fit_paths), "calibration": manifest_items(calibration_paths),
            "validation": manifest_items(validation_paths), "excluded": excluded,
            "validation_scores": [float(score) for score in validation_scores],
            "threshold": float(threshold), "roi_policy": "shared_runtime_locator_complete_part",
            "inspection_area": inspection_area, "inspection_areas": inspection_areas, "alignment_reference": alignment_reference,
        }
        if progress_callback:
            progress_callback(0.99, "deep_patch_ready")
        return model, metrics

    def inspect(self, image, *, object_box=None, brand=None, brand_confidence=None):
        if object_box is not None:
            x1, y1, x2, y2 = object_box
            height, width = image.shape[:2]
            if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
                # A truncated part cannot be compared to a complete reference.
                # Abstain explicitly: neither a clean OK nor a product defect.
                return InspectionResult("REVIEW", 0.0, 0.0, round(self.threshold, 4), [],
                                        brand, brand_confidence, review_required=True,
                                        input_issue="part_clipped_by_frame")
        try:
            feature_map, source_box = self.encoder.encode(image, object_box)
        except AlignmentError:
            return InspectionResult("REVIEW", 0.0, 0.0, round(self.threshold, 4), [],
                                    brand, brand_confidence, review_required=True,
                                    input_issue="alignment_failed")
        distances = self._distances(feature_map.reshape(-1, feature_map.shape[-1]), self.memory)
        if self.position_scale is not None and self.position_scale.size == distances.size:
            distances = distances / np.maximum(self.position_scale, 1e-6)
        active = self.active_mask.reshape(-1)
        distances[~active] = 0
        heatmap = distances.reshape(feature_map.shape[:2])
        score = self._image_score(distances, self.active_mask)
        if (not np.isfinite(distances).all() or not np.isfinite(score)
                or not all(np.isfinite(value) and value > 0 for value in
                           (self.threshold, self.review_threshold, self.region_threshold))):
            return InspectionResult("REVIEW", 0.0, 0.0, 0.0, [], brand, brand_confidence,
                                    review_required=True, input_issue="invalid_model_output")
        is_ng, review = score > self.threshold, False
        review = not is_ng and score >= self.review_threshold
        # Scores are calibrated in template coordinates. Only presentation is
        # mapped back into the current camera ROI; decision thresholds stay fixed.
        warp = getattr(self.encoder, "last_alignment_warp", None)
        visible_mask = self.active_mask.astype(np.float32)
        if warp is not None:
            heatmap = _camera_heatmap(heatmap, warp, self.input_size)
            visible_mask = _camera_heatmap(visible_mask, warp, self.input_size)
        regions = []
        if is_ng or review:
            x1, y1, x2, y2 = source_box
            width, height = max(1, x2 - x1), max(1, y2 - y1)
            binary = cv2.morphologyEx((heatmap >= self.region_threshold).astype(np.uint8) * 255,
                                      cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            binary[visible_mask < .5] = 0
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
            rendered_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            minimum = max(4.0, width * height * self.min_region_ratio)
            for contour in cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(contour) >= minimum:
                    rx, ry, rw, rh = cv2.boundingRect(contour)
                    regions.append(InspectionRegion(x1 + rx, y1 + ry, rw, rh,
                                   round(float(rendered_heatmap[ry:ry + rh, rx:rx + rw].max()), 4)))
            # An image-level anomaly does not prove a valid defect region.
            # Keep regions empty when every region fails the area test.
        boundary = self.threshold if is_ng else self.review_threshold
        confidence = float(np.clip(0.5 + abs(score - boundary) / max(boundary, 1e-6) * 0.35, 0.5, 0.995))
        return InspectionResult("NG" if is_ng else "OK", round(confidence, 4), round(score, 4),
                                round(self.threshold, 4), regions, brand, brand_confidence,
                                review, heatmap.round(6).tolist(), inspection_area=self.inspection_area, inspection_areas=self.inspection_areas)

    draw_result = staticmethod(NormalReferenceInspector.draw_result)

    def save(self, path: str | Path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": np.array([self.ENGINE]), "version": np.array([self.model_version], np.int16),
            "memory": self.memory.astype(np.float16), "threshold": np.array([self.threshold], np.float32),
            "review_threshold": np.array([self.review_threshold], np.float32),
            "region_threshold": np.array([self.region_threshold], np.float32),
            "training_count": np.array([self.training_count], np.int32), "backbone_path": np.array([self.backbone_path]),
            "input_size": np.array([self.input_size], np.int16), "grid_size": np.array([self.grid_size], np.int16),
            "min_region_ratio": np.array([self.min_region_ratio], np.float32), "active_mask": self.active_mask.astype(np.uint8),
            "inspection_area": np.array(self.inspection_area, np.float32),
            "inspection_areas": np.array(self.inspection_areas, np.float32),
        }
        if self.alignment_template is not None:
            payload["alignment_template"] = self.alignment_template
        if self.object_mask is not None:
            payload["object_mask"] = self.object_mask
        if self.position_scale is not None:
            payload["position_scale"] = self.position_scale.astype(np.float32)
        np.savez_compressed(output, **payload)
        return output

    @classmethod
    def load(cls, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as data:
            grid = int(data["grid_size"][0])
            backbone = Path(str(data["backbone_path"][0]))
            if not backbone.is_file():
                # Relocate the same backbone after moving the project. Never
                # substitute a different trained model for this feature bank.
                relocated = Path(path).resolve().parent.parent / backbone.name
                if relocated.is_file():
                    backbone = relocated
            return cls(data["memory"], float(data["threshold"][0]), float(data["region_threshold"][0]),
                       int(data["training_count"][0]), str(backbone), int(data["input_size"][0]), grid,
                       float(data["min_region_ratio"][0]),
                       float(data["review_threshold"][0]) if "review_threshold" in data else float(data["threshold"][0]) * 0.93,
                       data["active_mask"] if "active_mask" in data else np.ones((grid, grid), np.uint8),
                       data["alignment_template"] if "alignment_template" in data else None,
                       data["object_mask"] if "object_mask" in data else None,
                       data["position_scale"] if "position_scale" in data else None,
                       int(data["version"][0]) if "version" in data else 2,
                       data["inspection_area"].astype(float).tolist() if "inspection_area" in data else None,
                       data["inspection_areas"].astype(float).tolist() if "inspection_areas" in data else None)


def load_inspection_reference(path: str | Path):
    model_path = Path(path)
    with np.load(model_path, allow_pickle=False) as data:
        engine = str(data["engine"][0]) if "engine" in data else "statistical_reference"
    return DeepPatchReference.load(model_path) if engine == DeepPatchReference.ENGINE else NormalReferenceInspector.load(model_path)
