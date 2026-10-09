"""Bind runtime decisions to the trained artifact and its product brand.

This is a product-identity check, not a defect-accuracy qualification.
Offline model evaluation intentionally uses the unwrapped inspector.
"""
from pathlib import Path
import hashlib
import math
from normal_reference import InspectionResult
from inspection_area import checked_area, checked_areas


def build_inspection_contract(model_path, datasets, min_brand_confidence=0.8, *, check_brand=False):
    brands = [str((item.get("recipe") or {}).get("brand") or "").strip() for item in datasets]
    verified_brand = bool(brands) and all(brands) and len({b.casefold() for b in brands}) == 1
    if check_brand and not verified_brand:
        return {"schema_version": 2, "error": "recipe_brand_unverified"}
    return {"schema_version": 2, "check_brand": check_brand,
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "brand": brands[0] if verified_brand else "", "min_brand_confidence": min_brand_confidence,
            "datasets": datasets}


class RecipeBoundInspector:
    def __init__(self, inspector, model_path, reference):
        self.inspector = inspector
        self.contract = dict(reference.get("inspection_contract") or {})
        self.contract_issue = None
        version = self.contract.get("schema_version")
        # Preserve old saved jobs; new jobs choose the identity check explicitly.
        self.check_brand = True if version == 1 else self.contract.get("check_brand")
        if version not in (1, 2) or not isinstance(self.check_brand, bool) or (self.check_brand and not self.contract.get("brand")):
            self.contract_issue = "recipe_unverified"
        elif hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != self.contract.get("model_sha256"):
            self.contract_issue = "recipe_model_mismatch"
        if "inspection_area" in reference:
            try:
                requested = checked_area(reference["inspection_area"])
                stored = checked_area(getattr(inspector, "inspection_area", None))
                if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(requested, stored)):
                    self.contract_issue = "recipe_area_mismatch"
            except (TypeError, ValueError):
                self.contract_issue = "recipe_area_mismatch"
        if "inspection_areas" in reference:
            try:
                requested = checked_areas(reference["inspection_areas"], reference.get("inspection_area"))
                stored = checked_areas(getattr(inspector, "inspection_areas", None), getattr(inspector, "inspection_area", None))
                if len(requested) != len(stored) or any(not math.isclose(a,b,abs_tol=1e-6) for left,right in zip(requested,stored) for a,b in zip(left,right)):
                    self.contract_issue = "recipe_area_mismatch"
            except (TypeError, ValueError):
                self.contract_issue = "recipe_area_mismatch"
        try:
            self.minimum = float(self.contract.get("min_brand_confidence", 0.8))
            if not math.isfinite(self.minimum) or not 0 <= self.minimum <= 1:
                self.contract_issue = "recipe_unverified"
        except (TypeError, ValueError):
            self.minimum = 0.8
            self.contract_issue = "recipe_unverified"

    def __getattr__(self, name):
        return getattr(self.inspector, name)

    def inspect(self, image, *, object_box=None, brand=None, brand_confidence=None):
        issue = self.contract_issue
        if not issue and self.check_brand:
            try:
                confidence = float(brand_confidence)
            except (TypeError, ValueError):
                confidence = float("nan")
            if not brand or str(brand).strip().casefold() == "unknown" or not math.isfinite(confidence) or not self.minimum <= confidence <= 1:
                issue = "recipe_brand_unconfirmed"
            elif str(brand).strip().casefold() != self.contract["brand"].strip().casefold():
                issue = "recipe_brand_mismatch"
        if issue:
            return InspectionResult("REVIEW", 0.0, 0.0, self.inspector.threshold, [],
                                    brand=brand, brand_confidence=brand_confidence,
                                    review_required=True, input_issue=issue)
        return self.inspector.inspect(image, object_box=object_box, brand=brand,
                                      brand_confidence=brand_confidence)
