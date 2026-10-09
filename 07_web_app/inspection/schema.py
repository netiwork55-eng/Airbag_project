"""Validated, reproducible inspection configuration and one decision contract."""
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class State(str, Enum):
    PASS = "PASS"
    NG = "NG"
    REVIEW = "REVIEW"
    UNABLE = "UNABLE_TO_INSPECT"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Rect(Config):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def inside(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Area must be fully inside the reference image")
        return self


class Quality(Config):
    min_brightness: float = Field(default=10, ge=0, le=255)
    max_brightness: float = Field(default=245, ge=0, le=255)
    min_blur: float = Field(default=0, ge=0)  # disabled until calibrated for this camera/job

    @model_validator(mode="after")
    def bounds(self):
        if self.max_brightness <= self.min_brightness:
            raise ValueError("Maximum brightness must exceed minimum")
        return self


class Camera(Config):
    driver: Literal["opencv", "basler"] = "opencv"
    index: int = Field(default=0, ge=0, le=32)
    serial: str = Field(default="", max_length=100)
    exposure_us: float | None = Field(default=None, gt=0)
    gain: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=16, le=8192)
    height: int | None = Field(default=None, ge=16, le=8192)
    max_frame_age_ms: int = Field(default=1000, ge=100, le=10000)


class Locator(Config):
    method: Literal["fixed", "template", "features"] = "fixed"
    region: Rect = Field(default_factory=lambda: Rect(x=0, y=0, width=1, height=1))
    min_score: float = Field(default=.8, gt=0, le=1)
    min_matches: int = Field(default=12, ge=6, le=500)
    max_shift: float = Field(default=.2, ge=0, le=1)
    max_rotation: float = Field(default=15, ge=0, le=180)
    scale_tolerance: float = Field(default=.15, ge=0, le=.8)
    fixture_confirmed: bool = False


class Method(Config):
    kind: Literal["presence", "brightness", "anomaly", "detection", "segmentation"] = "presence"
    threshold: float = Field(default=.8, ge=0)
    review_margin: float = Field(default=.03, ge=0)
    low: float = Field(default=30, ge=0, le=255)
    high: float = Field(default=220, ge=0, le=255)
    model_id: str | None = Field(default=None, pattern=r"^model_[0-9a-f]{32}$")
    target_labels: list[str] = Field(default_factory=list, max_length=100)
    min_count: int = Field(default=0, ge=0, le=99)
    max_count: int = Field(default=0, ge=0, le=99)

    @model_validator(mode="after")
    def bounds(self):
        if self.kind == "presence" and self.threshold > 1:
            raise ValueError("Presence threshold must be between 0 and 1")
        if self.kind == "brightness" and self.low >= self.high:
            raise ValueError("Brightness lower limit must be below upper limit")
        if self.kind in ("detection","segmentation"):
            self.target_labels=list(dict.fromkeys(s.strip() for s in self.target_labels))
            if not self.target_labels or any(not s.strip() for s in self.target_labels):
                raise ValueError("Choose the labels this area needs to check")
            if self.min_count and len(self.target_labels)!=1:
                raise ValueError("Use one label for required-component counts; create a separate area for each component type")
            if self.min_count>self.max_count:
                raise ValueError("Minimum count cannot exceed maximum count")
            if not 0 <= self.threshold-self.review_margin < self.threshold+self.review_margin <= 1:
                raise ValueError("Confidence and review band must stay between 0 and 1")
        return self


class Area(Config):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    name: str = Field(min_length=1, max_length=100)
    region: Rect
    method: Method = Field(default_factory=Method)
    required: bool = True


class ValidationLimits(Config):
    # Part-level rejection policy is distinct from detector object recall.
    # Leave unset until the inspection owner defines the acceptance requirement.
    min_defect_recall: float | None = Field(default=None, gt=0, le=1)
    min_precision: float = Field(default=.9,ge=0,le=1)
    min_recall: float = Field(default=.9,ge=0,le=1)
    min_good: int = Field(default=10, ge=1, le=100000)
    min_ng: int = Field(default=10, ge=1, le=100000)
    max_false_accept: float = Field(default=0, ge=0, le=1)
    max_false_reject: float = Field(default=.05, ge=0, le=1)
    max_review: float = Field(default=.1, ge=0, le=1)
    max_unable: float = Field(default=0, ge=0, le=1)


class DecisionSettings(Config):
    stable_frames: int = Field(default=3, ge=2, le=30)
    release_frames: int = Field(default=3, ge=2, le=30)
    release_ms: int = Field(default=400, ge=100, le=5000)


class Recipe(Config):
    schema_version: Literal[1] = 1
    product: str = Field(min_length=1, max_length=120)
    reference_id: str = Field(pattern=r"^image_[0-9a-f]{64}$")
    camera: Camera = Field(default_factory=Camera)
    quality: Quality = Field(default_factory=Quality)
    locator: Locator = Field(default_factory=Locator)
    areas: list[Area] = Field(min_length=1, max_length=16)
    validation: ValidationLimits = Field(default_factory=ValidationLimits)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)

    @model_validator(mode="after")
    def unique_areas(self):
        if len({a.id for a in self.areas}) != len(self.areas):
            raise ValueError("Area IDs must be unique")
        if not any(a.required for a in self.areas):
            raise ValueError("At least one inspection area must be required")
        return self


def decision(results: list[dict], failures: list[str] = ()) -> tuple[str, list[str]]:
    """An incomplete inspection cannot certify a part, even if another area found NG."""
    required = [r for r in results if r.get("required", True)]
    reasons = list(failures)
    reasons.extend(r.get("reason", "Inspection failed") for r in required
                   if r["state"] == State.UNABLE)
    if reasons or not required:
        return State.UNABLE.value, reasons or ["No required inspection completed"]
    for state in (State.NG, State.REVIEW):
        matches = [r for r in required if r["state"] == state]
        if matches:
            return state.value, [r.get("reason", r["area_id"]) for r in matches]
    if any(r["state"] != State.PASS for r in required):
        return State.UNABLE.value, ["Unknown inspection result"]
    return State.PASS.value, ["All required checks passed"]
