"""Object-level comparison of a segmentation against ground truth.

A predicted object and a ground-truth object are a *match* when their
intersection-over-union (IoU) exceeds a threshold (0.5 by default). Because
labels within one image never overlap, any IoU > 0.5 pairing is necessarily
one-to-one, so a greedy assignment is exact. From the matches we report the
standard detection metrics:

* precision = TP / (TP + FP)
* recall    = TP / (TP + FN)
* F1        = harmonic mean of precision and recall
* mean IoU of matched pairs, and the signed count error (predicted - true).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["MatchResult", "match_labels"]


@dataclass(frozen=True)
class MatchResult:
    """Detection metrics for one image."""

    true_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    count_error: int
    iou_threshold: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


def match_labels(
    predicted: np.ndarray, truth: np.ndarray, iou_threshold: float = 0.5
) -> MatchResult:
    """Match predicted objects to ground-truth objects by IoU.

    Args:
        predicted: Predicted label image (0 = background).
        truth: Ground-truth label image of the same shape.
        iou_threshold: Minimum IoU for a match; must be >= 0.5 for the matching
            to be guaranteed one-to-one.
    """
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    if not 0.5 <= iou_threshold < 1.0:
        raise ValueError("iou_threshold must be in [0.5, 1)")
    pred = predicted.astype(np.int64).ravel()
    true = truth.astype(np.int64).ravel()
    pred_ids = np.unique(pred[pred > 0])
    true_ids = np.unique(true[true > 0])
    n_pred, n_true = len(pred_ids), len(true_ids)

    matched_ious: list[float] = []
    if n_pred and n_true:
        pred_area = np.bincount(pred)
        true_area = np.bincount(true)
        both = (pred > 0) & (true > 0)
        base = int(np.max(true)) + 1
        pairs, counts = np.unique(pred[both] * base + true[both], return_counts=True)
        p_lab = pairs // base
        t_lab = pairs % base
        ious = counts / (pred_area[p_lab] + true_area[t_lab] - counts)
        matched_ious = [float(v) for v in ious[ious > iou_threshold]]

    tp = len(matched_ious)
    fp, fn = n_pred - tp, n_true - tp
    precision = tp / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    recall = tp / n_true if n_true else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return MatchResult(
        true_count=n_true,
        predicted_count=n_pred,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_iou=round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        count_error=n_pred - n_true,
        iou_threshold=iou_threshold,
    )
