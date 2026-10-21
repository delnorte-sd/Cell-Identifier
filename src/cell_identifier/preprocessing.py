"""Preprocessing: channel extraction, denoising, background correction, normalisation.

Two intensity images come out of this stage and they must not be confused:

* the **raw intensity** (native units, e.g. 0-255), used for every intensity
  *measurement*, and
* the **processed image** (fraction of full scale, denoised and optionally
  background-corrected), used only to *find* cells.

Measuring on a filtered image would bias intensities, so the pipeline keeps
both.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from .config import BackgroundMethod, ChannelMode, PreprocessConfig

__all__ = [
    "PreprocessResult",
    "correct_background",
    "denoise",
    "disk",
    "extract_channel",
    "normalize",
    "preprocess",
]

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)
_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


@dataclass(frozen=True)
class PreprocessResult:
    """Output of :func:`preprocess`.

    Attributes:
        processed: Float32 image used for segmentation (fraction of full scale).
        background: Estimated background that was subtracted (same units), or
            ``None`` when background correction is disabled.
    """

    processed: np.ndarray
    background: np.ndarray | None


def disk(radius: int) -> np.ndarray:
    """Return a ``uint8`` disk-shaped structuring element of the given radius."""
    if radius < 1:
        return np.ones((1, 1), dtype=np.uint8)
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def extract_channel(image: np.ndarray, mode: ChannelMode = "gray") -> np.ndarray:
    """Reduce an image to one float32 intensity channel in its native units.

    Args:
        image: ``(H, W)`` or RGB ``(H, W, 3)`` array.
        mode: ``"gray"`` (luminance weights 0.299/0.587/0.114, identical to
            OpenCV), ``"red"``, ``"green"``, ``"blue"`` or ``"max"``.

    Single-channel inputs are returned unchanged (as float32) for every mode.
    """
    if image.ndim == 2:
        return image.astype(np.float32, copy=True)
    pixels = image.astype(np.float64, copy=False)
    if mode == "gray":
        out = pixels @ _LUMA
    elif mode == "max":
        out = pixels.max(axis=2)
    elif mode in _CHANNEL_INDEX:
        out = pixels[:, :, _CHANNEL_INDEX[mode]]
    else:
        raise ValueError(f"unknown channel mode {mode!r}")
    return out.astype(np.float32)


def denoise(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply the configured noise filter (Gaussian or median) to a float image."""
    if config.denoise == "gaussian":
        return cv2.GaussianBlur(
            image, (0, 0), sigmaX=config.denoise_sigma, borderType=cv2.BORDER_REFLECT
        )
    if config.denoise == "median":
        return ndimage.median_filter(image, size=config.median_size, mode="reflect")
    return image.copy()


def correct_background(
    image: np.ndarray, method: BackgroundMethod, radius: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Remove slowly varying background / uneven illumination.

    Args:
        image: Float image.
        method: ``"tophat"`` subtracts a grey-level opening with a disk of
            ``radius`` (a rolling-ball approximation that removes everything
            narrower than the disk *except* bright objects smaller than it);
            ``"gaussian"`` subtracts a Gaussian blur with sigma ``radius``.
        radius: Disk radius or blur sigma in pixels; must exceed the cell radius.

    Returns:
        ``(corrected, background)``; the corrected image is clipped at zero.
    """
    if method == "none":
        return image.copy(), None
    if method == "tophat":
        background = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, disk(round(radius)), borderType=cv2.BORDER_REFLECT
        )
    elif method == "gaussian":
        background = cv2.GaussianBlur(
            image, (0, 0), sigmaX=float(radius), borderType=cv2.BORDER_REFLECT
        )
    else:
        raise ValueError(f"unknown background method {method!r}")
    corrected = np.clip(image - background, 0.0, None).astype(np.float32)
    return corrected, background.astype(np.float32)


def normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Linearly map the ``low``/``high`` percentiles to 0/1 and clip.

    A constant image maps to all zeros (there is no contrast to stretch).
    """
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess(intensity: np.ndarray, scale: float, config: PreprocessConfig) -> PreprocessResult:
    """Produce the segmentation image from a raw intensity channel.

    Order of operations: scale to fraction of full scale → denoise →
    background correction → optional percentile normalisation.

    Args:
        intensity: Raw single-channel intensity in native units.
        scale: Full-scale value of the source dtype (255, 65535 or 1.0).
        config: Preprocessing settings.
    """
    image = (intensity / np.float32(scale)).astype(np.float32)
    image = denoise(image, config)
    image, background = correct_background(image, config.background, config.background_radius)
    if config.normalize == "percentile":
        image = normalize(image, config.percentile_low, config.percentile_high)
    return PreprocessResult(processed=image, background=background)
