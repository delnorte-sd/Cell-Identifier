"""Brightness classification of measured cells."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .config import ClassificationConfig
from .measurement import CellMeasurement

__all__ = ["BRIGHTNESS_CLASSES", "classify_brightness"]

#: Class names in increasing brightness, plus the label used when disabled.
BRIGHTNESS_CLASSES: tuple[str, str, str] = ("Low", "Medium", "High")


def _bin(value: float, low: float, high: float) -> str:
    if value < low:
        return "Low"
    if value < high:
        return "Medium"
    return "High"


def classify_brightness(
    cells: Sequence[CellMeasurement], config: ClassificationConfig, full_scale: float
) -> list[CellMeasurement]:
    """Return copies of ``cells`` with their ``brightness`` field set.

    Args:
        cells: Measured cells.
        config: Classification rules.
        full_scale: Full-scale intensity of the source image (255, 65535 or 1.0),
            used to convert the fractional cut-offs of ``"absolute"`` mode into
            native units.

    In ``"absolute"`` mode a cell is "Low" if its raw mean intensity is below
    ``low_cutoff``, "High" if at or above ``high_cutoff`` and "Medium" otherwise.
    In ``"quantile"`` mode the cut-offs are the 1/3 and 2/3 quantiles of this
    image's cell intensities, so classes are relative to the image only.
    """
    if config.method == "none" or not cells:
        return [replace(c, brightness="Unclassified") for c in cells]
    if config.method == "absolute":
        low = config.low_cutoff * full_scale
        high = config.high_cutoff * full_scale
    else:
        means = np.array([c.mean_intensity for c in cells], dtype=np.float64)
        low, high = (float(q) for q in np.quantile(means, [1 / 3, 2 / 3]))
    return [replace(c, brightness=_bin(c.mean_intensity, low, high)) for c in cells]
