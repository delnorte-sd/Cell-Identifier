from __future__ import annotations

import cv2
import numpy as np
import pytest

from cell_identifier.config import SegmentationConfig
from cell_identifier.segmentation import (
    compute_threshold,
    estimate_noise_sigma,
    h_maxima,
    noise_floor,
    segment,
    split_touching,
)


def _bimodal() -> np.ndarray:
    img = np.full((64, 64), 0.1, dtype=np.float32)
    img[20:40, 20:40] = 0.8
    return img


@pytest.mark.parametrize("method", ["otsu", "triangle"])
def test_automatic_threshold_separates_bimodal_image(method: str) -> None:
    t = compute_threshold(_bimodal(), SegmentationConfig(threshold_method=method))  # type: ignore[arg-type]
    assert 0.1 < float(t) < 0.8


def test_fixed_and_adaptive_thresholds() -> None:
    img = _bimodal()
    assert (
        compute_threshold(img, SegmentationConfig(threshold_method="fixed", fixed_threshold=0.3))
        == 0.3
    )
    local = compute_threshold(
        img, SegmentationConfig(threshold_method="adaptive", adaptive_block_size=15)
    )
    assert isinstance(local, np.ndarray)
    assert local.shape == img.shape


def test_constant_image_has_no_foreground() -> None:
    for value in (0.0, 0.5):
        img = np.full((32, 32), value, dtype=np.float32)
        result = segment(img, SegmentationConfig())
        assert result.labels.max() == 0
        assert not result.mask.any()


def test_noise_floor_rejects_pure_noise() -> None:
    rng = np.random.default_rng(1)
    noise = np.clip(0.1 + rng.normal(0, 0.02, (256, 256)), 0, 1).astype(np.float32)
    guarded = segment(noise, SegmentationConfig(split_method="none"))
    unguarded = segment(noise, SegmentationConfig(split_method="none", min_snr=0))
    # A 3-sigma floor passes ~0.13 % of pixels; the opening removes those isolated pixels.
    assert guarded.mask.sum() < 0.0005 * noise.size, "noise floor should suppress noise"
    assert unguarded.mask.sum() > 0.1 * noise.size, "plain Otsu splits noise in half"
    assert guarded.noise_floor == pytest.approx(guarded.threshold)


@pytest.mark.parametrize("slope", [0.0, 0.5])
def test_noise_sigma_estimate_ignores_gradients_and_cells(slope: float) -> None:
    rng = np.random.default_rng(2)
    ramp = np.tile(np.linspace(0.1, 0.1 + slope, 256, dtype=np.float32), (256, 1))
    img = ramp + rng.normal(0, 0.02, ramp.shape).astype(np.float32)
    assert estimate_noise_sigma(img) == pytest.approx(0.02, rel=0.05)
    for x in range(20, 256, 40):
        cv2.circle(img, (x, 128), 6, 0.9, -1)
    # Bright cells inflate the estimate slightly, making the floor conservative.
    assert 0.019 <= estimate_noise_sigma(img) <= 0.026


def test_noise_floor_formula() -> None:
    rng = np.random.default_rng(4)
    img = (0.2 + rng.normal(0, 0.01, (128, 128))).astype(np.float32)
    expected = float(np.median(img)) + 3 * estimate_noise_sigma(img)
    assert noise_floor(img, 3.0) == pytest.approx(expected)
    assert noise_floor(img, 0.0) == float("-inf")


def test_h_maxima_ignores_shallow_bumps() -> None:
    profile = np.array([[0, 5, 4.5, 5, 0, 0, 9, 0]], dtype=float)
    peaks = h_maxima(profile, 1.0)
    # The two 5-peaks are separated by a 0.5 dip (< h) and merge into one seed.
    assert peaks[0, 6]
    assert peaks[0, 1]
    assert peaks[0, 3]
    assert peaks[0, 2]


def _two_discs(offset: int) -> np.ndarray:
    mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 12, 1, -1)
    cv2.circle(mask, (40 + offset, 40), 12, 1, -1)
    return mask.astype(bool)


@pytest.mark.parametrize("offset", [23, 18])
def test_distance_watershed_splits_touching_discs_evenly(offset: int) -> None:
    mask = _two_discs(offset)
    labels = split_touching(mask, 1.0)
    assert labels.max() == 2
    areas = np.bincount(labels.ravel())[1:]
    assert abs(int(areas[0]) - int(areas[1])) <= 0.1 * areas.mean()
    assert int((labels > 0).sum()) == int(mask.sum()), "no foreground pixel may be lost"
    assert not (labels[~mask] > 0).any()


@pytest.mark.parametrize("width", [3, 5, 8])
def test_distance_watershed_keeps_rods_whole(width: int) -> None:
    rod = np.zeros((80, 120), dtype=np.uint8)
    cv2.line(rod, (20, 30), (90, 55), 1, width)
    assert split_touching(rod.astype(bool), 1.0).max() == 1


def test_intensity_watershed_splits_side_by_side_rods() -> None:
    # Two parallel rods touching along their length: no geometric neck, but a dim seam.
    img = np.zeros((60, 60), dtype=np.float32)
    img[10:50, 20:28] = 0.8
    img[10:50, 28:36] = 0.8
    img[10:50, 27:29] = 0.5
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    mask = img > 0.3
    assert split_touching(mask, 1.0, method="distance").max() == 1
    assert split_touching(mask, 0.05, method="intensity", intensity=img).max() == 2
    with pytest.raises(ValueError, match="intensity image is required"):
        split_touching(mask, 0.05, method="intensity")
    with pytest.raises(ValueError, match="unknown split"):
        split_touching(mask, 1.0, method="voronoi")  # type: ignore[arg-type]


def test_split_none_and_empty_mask() -> None:
    mask = _two_discs(18).astype(np.float32) * 0.9
    labels = segment(mask, SegmentationConfig(split_method="none")).labels
    assert labels.max() == 1
    assert split_touching(np.zeros((10, 10), dtype=bool), 1.0).max() == 0


def test_morphology_options() -> None:
    img = np.zeros((40, 40), dtype=np.float32)
    img[10:30, 10:30] = 0.9
    img[15:25, 15:25] = 0.0  # hole
    img[2, 2] = 0.9  # single-pixel speckle
    filled = segment(img, SegmentationConfig(threshold_method="fixed", fixed_threshold=0.5))
    assert filled.mask[20, 20], "hole should be filled"
    assert not filled.mask[2, 2], "speckle should be opened away"
    raw = segment(
        img,
        SegmentationConfig(
            threshold_method="fixed",
            fixed_threshold=0.5,
            fill_holes=False,
            opening_radius=0,
            closing_radius=1,
        ),
    )
    assert not raw.mask[20, 20]
    assert raw.mask[2, 2]


def test_adaptive_threshold_handles_illumination_gradient() -> None:
    ramp = np.tile(np.linspace(0.05, 0.6, 200, dtype=np.float32), (100, 1))
    img = ramp.copy()
    for x in (30, 100, 170):
        cv2.circle(img, (x, 50), 6, float(ramp[50, x] + 0.25), -1)
    result = segment(img, SegmentationConfig(threshold_method="adaptive", adaptive_block_size=41))
    assert result.labels.max() == 3
