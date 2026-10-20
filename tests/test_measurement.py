from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from cell_identifier.classification import classify_brightness
from cell_identifier.config import ClassificationConfig, FilterConfig
from cell_identifier.measurement import CellMeasurement, measure_cells


def _measure(labels: np.ndarray, intensity: np.ndarray | None = None, **filters: object):
    if intensity is None:
        intensity = (labels > 0).astype(np.float32) * 100 + 10
    return measure_cells(labels, intensity, FilterConfig(**filters))  # type: ignore[arg-type]


def test_disc_geometry() -> None:
    labels = np.zeros((101, 101), dtype=np.int32)
    cv2.circle(labels, (50, 50), 20, 1, -1)
    _, cells, _ = _measure(labels)
    (cell,) = cells
    assert cell.area_px == pytest.approx(math.pi * 20**2, rel=0.02)
    assert (cell.centroid_x, cell.centroid_y) == pytest.approx((50, 50), abs=1e-6)
    assert cell.equivalent_diameter_px == pytest.approx(40, rel=0.02)
    assert cell.circularity > 0.9
    assert cell.eccentricity < 0.1
    assert cell.solidity > 0.95  # skimage hulls include pixel corners, so a digital disc is ~0.96
    assert cell.aspect_ratio == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("angle", [0, 30, 90, 135])
def test_rod_axes_and_orientation(angle: float) -> None:
    labels = np.zeros((200, 200), dtype=np.int32)
    box = cv2.boxPoints(((100, 100), (60, 10), angle)).astype(np.int32)
    cv2.fillPoly(labels, [box], 1)
    (cell,) = _measure(labels)[1]
    assert cell.major_axis_px / cell.minor_axis_px == pytest.approx(6, rel=0.15)
    diff = abs((cell.orientation_deg - angle + 90) % 180 - 90)
    assert diff < 2, f"orientation {cell.orientation_deg} vs drawn {angle}"
    assert cell.circularity < 0.6


def test_intensity_metrics_are_exact_on_raw_values() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:20, 10:20] = 1
    intensity = np.full((40, 40), 20.0, dtype=np.float32)
    intensity[10:20, 10:20] = 120.0
    intensity[15, 15] = 220.0
    _, (cell,), background = _measure(labels, intensity)
    assert background == 20.0
    assert cell.area_px == 100
    assert cell.max_intensity == 220.0
    assert cell.integrated_intensity == pytest.approx(99 * 120 + 220)
    assert cell.mean_intensity == pytest.approx((99 * 120 + 220) / 100)
    assert cell.mean_intensity_corrected == pytest.approx(cell.mean_intensity - 20)


def test_area_and_border_filters_and_stable_ids() -> None:
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[60:70, 10:20] = 5  # lower-left, 100 px
    labels[10:20, 70:80] = 9  # upper-right, 100 px
    labels[10:20, 10:20] = 2  # upper-left, 100 px
    labels[40:42, 40:42] = 7  # tiny, 4 px
    labels[0:10, 45:55] = 3  # touches the top border
    new, cells, _ = _measure(labels, min_area=10)
    assert [c.cell_id for c in cells] == [1, 2, 3, 4]
    # IDs follow centroid raster order: top-border, upper-left, upper-right, lower-left.
    assert [(round(c.centroid_y), round(c.centroid_x)) for c in cells] == [
        (4, 50),
        (14, 14),
        (14, 74),
        (64, 14),
    ]
    assert cells[0].touches_border
    assert not cells[1].touches_border
    assert set(np.unique(new)) == {0, 1, 2, 3, 4}
    assert new[15, 15] == 2
    assert new[41, 41] == 0

    _, without_border, _ = _measure(labels, min_area=10, exclude_border=True)
    assert len(without_border) == 3
    assert not any(c.touches_border for c in without_border)
    _, capped, _ = _measure(labels, min_area=10, max_area=99)
    assert capped == []


def test_ids_independent_of_input_label_values() -> None:
    a = np.zeros((50, 50), dtype=np.int32)
    a[5:10, 5:10], a[30:35, 30:35] = 1, 2
    b = np.where(a == 1, 40, np.where(a == 2, 3, 0)).astype(np.int32)
    la, ca, _ = _measure(a, min_area=1)
    lb, cb, _ = _measure(b, min_area=1)
    np.testing.assert_array_equal(la, lb)
    assert ca == cb


def test_physical_units_and_shape_mismatch() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:15, 5:15] = 1
    _, (cell,), _ = measure_cells(
        labels, np.ones((30, 30), np.float32), FilterConfig(min_area=1), pixel_size_um=0.5
    )
    assert cell.area_um2 == pytest.approx(25.0)
    assert cell.major_axis_um == pytest.approx(cell.major_axis_px * 0.5, rel=1e-4)
    with pytest.raises(ValueError, match="shape"):
        measure_cells(labels, np.ones((10, 10), np.float32), FilterConfig())


def test_empty_label_image() -> None:
    new, cells, background = _measure(np.zeros((20, 20), dtype=np.int32))
    assert cells == []
    assert new.max() == 0
    assert background == 10.0


def _cell(cell_id: int, mean: float) -> CellMeasurement:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:10, 5:10] = 1
    (base,) = _measure(labels, min_area=1)[1]
    from dataclasses import replace

    return replace(base, cell_id=cell_id, mean_intensity=mean)


def test_absolute_brightness_classification() -> None:
    cells = [_cell(1, 50), _cell(2, 100), _cell(3, 200)]
    config = ClassificationConfig(method="absolute", low_cutoff=0.3, high_cutoff=0.6)
    assert [c.brightness for c in classify_brightness(cells, config, 255)] == [
        "Low",
        "Medium",
        "High",
    ]
    # Same fractional cut-offs scale with bit depth.
    deep = [_cell(1, 50 * 257), _cell(2, 100 * 257), _cell(3, 200 * 257)]
    assert [c.brightness for c in classify_brightness(deep, config, 65535)] == [
        "Low",
        "Medium",
        "High",
    ]


def test_quantile_and_disabled_classification() -> None:
    cells = [_cell(i, float(v)) for i, v in enumerate([10, 20, 30, 40, 50, 60], start=1)]
    labels = [
        c.brightness
        for c in classify_brightness(cells, ClassificationConfig(method="quantile"), 255)
    ]
    assert labels.count("Low") == 2
    assert labels.count("Medium") == 2
    assert labels.count("High") == 2
    off = classify_brightness(cells, ClassificationConfig(method="none"), 255)
    assert {c.brightness for c in off} == {"Unclassified"}
    assert classify_brightness([], ClassificationConfig(), 255) == []
