"""Split contracts only: no model construction, fitting or accuracy evaluation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspection.anomaly_worker import split_training_groups


@pytest.mark.parametrize('reference_group', ['part_a', 'part_b', 'part_c', 'part_d', 'part_e'])
def test_reference_cannot_empty_calibration_when_independent_groups_exist(reference_group):
    images = [{'id': f'{g}-{i}', 'group': g} for g in
              ['part_a', 'part_b', 'part_c', 'part_d', 'part_e'] for i in range(4)]
    groups, fit, calibration = split_training_groups(images, reference_group)
    fit_groups = {i['group'] for i in fit}
    cal_groups = {i['group'] for i in calibration}
    assert reference_group in fit_groups and reference_group not in cal_groups
    assert fit_groups.isdisjoint(cal_groups)
    assert len(fit) == 16 and len(calibration) == 4
    assert {i['id'] for i in fit + calibration} == {i['id'] for i in images}
    assert set(groups) == fit_groups | cal_groups
    _, reversed_fit, reversed_cal = split_training_groups(list(reversed(images)), reference_group)
    assert {i['id'] for i in reversed_fit} == {i['id'] for i in fit}
    assert {i['id'] for i in reversed_cal} == {i['id'] for i in calibration}


def test_many_frames_of_one_part_are_not_independent_capture_groups():
    with pytest.raises(ValueError, match='five independent capture groups'):
        split_training_groups([{'id': str(i), 'group': 'one-physical-part'} for i in range(268)],
                              'one-physical-part')


def test_insufficient_group_samples_still_block_training():
    images = [{'id': f'{g}-{i}', 'group': g} for g, n in
              [('part_a', 12), ('part_b', 2), ('part_c', 2), ('part_d', 2), ('part_e', 2)]
              for i in range(n)]
    with pytest.raises(ValueError, match='10 fit and 4 calibration'):
        split_training_groups(images, 'part_a')
