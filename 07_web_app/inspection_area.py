"""Inspection area in normalized, aligned part coordinates."""
import math
import numpy as np

FULL_AREA = [0.0, 0.0, 1.0, 1.0]


def checked_area(value=None):
    if value is None:
        return FULL_AREA.copy()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Inspection area needs left, top, right, bottom")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Inspection area must contain finite numbers")
    left, top, right, bottom = map(float, value)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("Inspection area must be inside the part, with positive width and height")
    return [left, top, right, bottom]


def area_grid(area, shape):
    left, top, right, bottom = checked_area(area)
    height, width = shape
    # Include cells whose centers are inside the selected area.
    x = (np.arange(width) + .5) / width
    y = (np.arange(height) + .5) / height
    return ((y[:, None] >= top) & (y[:, None] < bottom)
            & (x[None, :] >= left) & (x[None, :] < right))


def checked_areas(value=None, fallback=None):
    if value is None:
        return [checked_area(fallback)]
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 16:
        raise ValueError("Choose 1 to 16 inspection areas")
    return [list(item) for item in sorted(set(tuple(checked_area(x)) for x in value))]


def areas_grid(areas, shape):
    return np.logical_or.reduce([area_grid(area, shape) for area in checked_areas(areas)])
