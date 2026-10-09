import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_patch_reference import AlignmentError, DeepPatchReference, _align_bgr


def test_blank_image_cannot_silently_use_identity():
    image = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(AlignmentError):
        _align_bgr(image, image, return_warp=True)


@pytest.mark.parametrize('invalid', ['nan', 'singular', 'reflection'])
def test_invalid_ecc_output_is_rejected(monkeypatch, invalid):
    warp = np.eye(2, 3, dtype=np.float32)
    if invalid == 'singular':
        warp[0, 0] = 0
    if invalid == 'reflection':
        warp[0, 0] = -1
    monkeypatch.setattr(cv2, 'findTransformECC', lambda *args: (float('nan') if invalid == 'nan' else .9, warp))
    with pytest.raises(AlignmentError):
        _align_bgr(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))


def test_failed_alignment_does_not_score_part():
    inspector = object.__new__(DeepPatchReference)
    inspector.threshold = 3.
    inspector._encoder = SimpleNamespace(encode=Mock(side_effect=AlignmentError('failed')))
    inspector._distances = Mock(side_effect=AssertionError('must not score'))
    result = inspector.inspect(np.zeros((100, 100, 3), np.uint8), object_box=(10, 10, 90, 90))
    assert result.status == 'REVIEW'
    assert result.input_issue == 'alignment_failed'
    assert result.to_dict()['location'] == 'not_checked'
    assert not result.regions


def test_matching_textured_image_still_aligns():
    image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    aligned, _, warp = _align_bgr(image, image, return_warp=True)
    assert np.allclose(warp, np.eye(2, 3), atol=.01)
    assert np.max(np.abs(aligned.astype(float) - image)) < 2
