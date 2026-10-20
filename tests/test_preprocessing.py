from __future__ import annotations

import numpy as np
import pytest

from cell_identifier.config import PreprocessConfig
from cell_identifier.preprocessing import (
    correct_background,
    denoise,
    extract_channel,
    normalize,
    preprocess,
)


@pytest.fixture
def rgb() -> np.ndarray:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 100, 200, 50
    return img


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("red", 100),
        ("green", 200),
        ("blue", 50),
        ("max", 200),
        ("gray", 0.299 * 100 + 0.587 * 200 + 0.114 * 50),
    ],
)
def test_extract_channel(rgb: np.ndarray, mode: str, expected: float) -> None:
    out = extract_channel(rgb, mode)  # type: ignore[arg-type]
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_extract_channel_greyscale_passthrough_and_errors(rgb: np.ndarray) -> None:
    grey = np.full((8, 8), 7, dtype=np.uint16)
    out = extract_channel(grey, "green")
    assert out.dtype == np.float32
    assert float(out.max()) == 7.0
    with pytest.raises(ValueError, match="unknown channel"):
        extract_channel(rgb, "uv")  # type: ignore[arg-type]


def test_denoise_reduces_noise_and_preserves_mean() -> None:
    rng = np.random.default_rng(0)
    noisy = (0.5 + rng.normal(0, 0.1, (128, 128))).astype(np.float32)
    for method in ("gaussian", "median"):
        out = denoise(noisy, PreprocessConfig(denoise=method))  # type: ignore[arg-type]
        assert out.std() < 0.6 * noisy.std()
        assert abs(out.mean() - noisy.mean()) < 0.01
    untouched = denoise(noisy, PreprocessConfig(denoise="none"))
    np.testing.assert_array_equal(untouched, noisy)
    assert untouched is not noisy


@pytest.mark.parametrize("method", ["tophat", "gaussian"])
def test_background_correction_flattens_illumination(method: str) -> None:
    ramp = np.tile(np.linspace(0.1, 0.5, 200, dtype=np.float32), (200, 1))
    image = ramp.copy()
    image[95:105, 95:105] += 0.4  # small bright object on a sloped background
    corrected, background = correct_background(image, method, 15)  # type: ignore[arg-type]
    assert background is not None
    left, right = corrected[10:40, 10:40].mean(), corrected[10:40, 160:190].mean()
    assert abs(left - right) < 0.02, "background slope should be removed"
    assert corrected[98:102, 98:102].mean() > 0.3, "object signal should survive"
    assert corrected.min() >= 0


def test_background_none_and_unknown() -> None:
    img = np.ones((10, 10), dtype=np.float32)
    corrected, background = correct_background(img, "none", 5)
    assert background is None
    np.testing.assert_array_equal(corrected, img)
    with pytest.raises(ValueError, match="unknown background"):
        correct_background(img, "wavelet", 5)  # type: ignore[arg-type]


def test_normalize() -> None:
    img = np.linspace(0, 10, 101, dtype=np.float32).reshape(1, -1)
    out = normalize(img, 10, 90)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert np.all(normalize(np.full((5, 5), 3.0, dtype=np.float32), 1, 99) == 0)


def test_preprocess_scales_to_fraction_of_full_scale() -> None:
    raw = np.full((32, 32), 51.0, dtype=np.float32)
    result = preprocess(raw, 255.0, PreprocessConfig(denoise="none"))
    np.testing.assert_allclose(result.processed, 0.2, rtol=1e-6)
    assert result.background is None
    stretched = preprocess(
        np.tile(np.arange(32, dtype=np.float32), (32, 1)),
        255.0,
        PreprocessConfig(
            denoise="none", normalize="percentile", percentile_low=0, percentile_high=100
        ),
    )
    assert stretched.processed.max() == pytest.approx(1.0)
