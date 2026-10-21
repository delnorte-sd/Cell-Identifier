from __future__ import annotations

import numpy as np
import pytest

from cell_identifier import match_labels


def _labels() -> np.ndarray:
    truth = np.zeros((50, 50), dtype=np.int32)
    truth[5:15, 5:15] = 1
    truth[5:15, 30:40] = 2
    truth[30:40, 5:15] = 3
    return truth


def test_perfect_match() -> None:
    truth = _labels()
    m = match_labels(truth, truth)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (3, 0, 0)
    assert m.precision == m.recall == m.f1 == m.mean_iou == 1.0
    assert m.count_error == 0


def test_label_values_do_not_matter() -> None:
    truth = _labels()
    renumbered = np.where(truth > 0, 10 - truth, 0)
    assert match_labels(renumbered, truth).f1 == 1.0


def test_missed_extra_and_merged_objects() -> None:
    truth = _labels()
    pred = truth.copy()
    pred[pred == 3] = 0  # missed
    pred[40:48, 40:48] = 7  # spurious
    m = match_labels(pred, truth)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 1, 1)
    assert m.precision == pytest.approx(2 / 3, abs=1e-4)
    assert m.recall == pytest.approx(2 / 3, abs=1e-4)

    merged = truth.copy()
    merged[5:15, 5:40] = 1  # objects 1 and 2 merged into one blob
    m = match_labels(merged, truth)
    # The merged blob matches neither original (IoU < 0.5); object 3 still matches.
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 2)
    assert m.count_error == -1


def test_iou_threshold() -> None:
    truth = np.zeros((20, 20), dtype=np.int32)
    truth[0:10, 0:10] = 1
    shifted = np.zeros_like(truth)
    shifted[0:10, 3:13] = 1  # IoU = 70 / 130 ≈ 0.54
    assert match_labels(shifted, truth, 0.5).true_positives == 1
    assert match_labels(shifted, truth, 0.6).true_positives == 0


def test_empty_cases_and_validation() -> None:
    empty = np.zeros((10, 10), dtype=np.int32)
    assert match_labels(empty, empty).f1 == 1.0
    some = empty.copy()
    some[2:5, 2:5] = 1
    assert match_labels(empty, some).recall == 0.0
    assert match_labels(some, empty).precision == 0.0
    with pytest.raises(ValueError, match="shape"):
        match_labels(empty, np.zeros((5, 5), dtype=np.int32))
    with pytest.raises(ValueError, match="iou_threshold"):
        match_labels(empty, empty, 0.3)
