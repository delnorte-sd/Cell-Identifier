"""Foreground detection, morphological clean-up and splitting of touching cells.

Method summary
--------------
1. **Threshold** the processed image (Otsu, triangle, fixed or local-adaptive).
2. Raise the threshold to a **noise floor** ``median + k·σ``. σ is a robust
   estimate of pixel noise: 1.4826 × the median absolute deviation of the image
   *after subtracting its local mean*, so slow illumination gradients do not
   inflate it. Automatic thresholds always split a histogram in two, even one
   that contains only noise; the floor makes "no cells" a possible answer. For
   the adaptive method the floor is local: a pixel must exceed its local mean
   by at least ``k·σ``.
3. **Morphology**: opening removes speckle, optional closing bridges gaps,
   holes are filled.
4. **Watershed** (optional): the Euclidean distance transform of the mask is
   high in cell centres and low at the "necks" where two cells touch. Seeds are
   the *h-maxima* of the distance transform (local maxima at least ``h`` pixels
   more prominent than their surroundings), which suppresses the spurious
   maxima that pixelated rod-shaped cells produce along their mid-line.
   A marker-controlled watershed (scikit-image) then assigns every foreground
   pixel to one seed, placing boundaries along the distance-transform saddles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.segmentation import watershed

from .config import SegmentationConfig
from .preprocessing import disk

__all__ = [
    "SegmentationResult",
    "compute_threshold",
    "estimate_noise_sigma",
    "h_maxima",
    "noise_floor",
    "segment",
    "split_touching",
]

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class SegmentationResult:
    """Output of :func:`segment`.

    Attributes:
        labels: ``int32`` label image; 0 is background, objects are 1..N.
        mask: Boolean foreground mask after morphology (before size filtering).
        threshold: Threshold actually applied (fraction of full scale). For the
            adaptive method this is the median of the local threshold map.
        noise_floor: Lower bound imposed by ``min_snr`` (fraction of full scale).
            For the adaptive method this is the minimum margin above the local mean.
    """

    labels: np.ndarray
    mask: np.ndarray
    threshold: float
    noise_floor: float


def estimate_noise_sigma(image: np.ndarray, window: int = 51) -> float:
    """Robust standard deviation of pixel noise.

    Computes 1.4826 × MAD of the residual ``image - box_mean(image, window)``.
    Removing the local mean makes the estimate insensitive to illumination
    gradients; the median makes it robust to cells, provided they cover less
    than half of each window. Densely packed bright cells still bias the
    estimate slightly upwards (a conservative, i.e. higher, noise floor).
    """
    residual = image - cv2.blur(image, (window, window), borderType=cv2.BORDER_REPLICATE)
    centre = float(np.median(residual))
    return _MAD_TO_SIGMA * float(np.median(np.abs(residual - centre)))


def noise_floor(image: np.ndarray, min_snr: float, window: int = 51) -> float:
    """Return the global noise floor ``median(image) + min_snr * sigma``.

    Returns ``-inf`` when ``min_snr`` is 0 (floor disabled).
    """
    if min_snr <= 0:
        return float("-inf")
    return float(np.median(image)) + min_snr * estimate_noise_sigma(image, window)


def _histogram_threshold(image: np.ndarray, method: str) -> float:
    """Run a histogram-based automatic threshold on a float image."""
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi - lo <= 1e-12:
        return hi  # constant image: nothing lies strictly above the threshold
    if method == "otsu":
        return float(filters.threshold_otsu(image, nbins=1024))
    return float(filters.threshold_triangle(image, nbins=1024))


def compute_threshold(image: np.ndarray, config: SegmentationConfig) -> np.ndarray | float:
    """Compute the raw threshold (scalar, or a per-pixel map for ``"adaptive"``)."""
    method = config.threshold_method
    if method == "otsu":
        return _histogram_threshold(image, "otsu")
    if method == "triangle":
        return _histogram_threshold(image, "triangle")
    if method == "fixed":
        return config.fixed_threshold
    if method == "adaptive":
        size = config.adaptive_block_size
        local_mean = cv2.blur(image, (size, size), borderType=cv2.BORDER_REPLICATE)
        return local_mean + np.float32(config.adaptive_offset)
    raise ValueError(f"unknown threshold method {method!r}")


def h_maxima(image: np.ndarray, h: float) -> np.ndarray:
    """Boolean mask of the regional maxima whose dynamic (prominence) is at least ``h``.

    Uses the textbook definition, the regional maxima of the morphological
    reconstruction of ``image - h`` under ``image``. (``skimage.morphology.h_maxima``
    is avoided because its float handling fragments the flat ridge of rod-shaped
    cells into many seeds.)
    """
    image = np.asarray(image, dtype=np.float64)
    capped = morphology.reconstruction(
        image - h, image, method="dilation", footprint=_EIGHT_CONNECTED
    )
    return morphology.local_maxima(capped, connectivity=2, allow_borders=True).astype(bool)


def split_touching(
    mask: np.ndarray,
    h: float,
    *,
    method: Literal["distance", "intensity"] = "distance",
    intensity: np.ndarray | None = None,
) -> np.ndarray:
    """Label ``mask`` and split touching objects with a marker-controlled watershed.

    Args:
        mask: Boolean foreground mask.
        h: Minimum seed prominence: pixels of distance-transform height for
            ``"distance"``, fraction of full scale for ``"intensity"``.
        method: ``"distance"`` splits at geometric necks (best for round or
            isolated rod cells); ``"intensity"`` splits along dark valleys between
            cells (best for cells packed side by side, where there is no neck).
        intensity: Processed intensity image, required for ``"intensity"``.

    Returns:
        ``int32`` labels in which every foreground pixel belongs to exactly one
        object and no background pixel is labelled.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    if method == "distance":
        relief = ndimage.distance_transform_edt(mask)
    elif method == "intensity":
        if intensity is None:
            raise ValueError("intensity image is required for intensity-based splitting")
        relief = np.where(mask, intensity, 0.0).astype(np.float64)
    else:
        raise ValueError(f"unknown split method {method!r}")
    markers, _ = ndimage.label(h_maxima(relief, h) & mask, structure=_EIGHT_CONNECTED)
    labels = watershed(-relief, markers, mask=mask, connectivity=2)
    # Every connected component contains at least one h-maximum (its highest point),
    # but keep a safety net so no foreground pixel can ever be dropped.
    orphans = mask & (labels == 0)
    if orphans.any():  # pragma: no cover - defensive
        extra, _ = ndimage.label(orphans, structure=_EIGHT_CONNECTED)
        labels[orphans] = extra[orphans] + int(labels.max())
    return labels.astype(np.int32)


def segment(image: np.ndarray, config: SegmentationConfig) -> SegmentationResult:
    """Segment a processed (fraction-of-full-scale) image into labelled objects."""
    window = config.adaptive_block_size
    threshold: np.ndarray | float
    if config.threshold_method == "adaptive":
        sigma = estimate_noise_sigma(image, window) if config.min_snr > 0 else 0.0
        margin = max(config.adaptive_offset, config.min_snr * sigma)
        local_mean = cv2.blur(image, (window, window), borderType=cv2.BORDER_REPLICATE)
        threshold = local_mean + np.float32(margin)
        floor = config.min_snr * sigma
    else:
        floor = noise_floor(image, config.min_snr, window)
        threshold = max(float(compute_threshold(image, config)), floor)
    mask = image > threshold
    if config.opening_radius > 0:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, disk(config.opening_radius)
        ).astype(bool)
    if config.closing_radius > 0:
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(config.closing_radius)
        ).astype(bool)
    if config.fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if config.split_method == "distance":
        labels = split_touching(mask, config.watershed_h, method="distance")
    elif config.split_method == "intensity":
        labels = split_touching(mask, config.intensity_h, method="intensity", intensity=image)
    else:
        labels, _ = ndimage.label(mask, structure=_EIGHT_CONNECTED)
        labels = labels.astype(np.int32)
    reported = float(np.median(threshold)) if np.ndim(threshold) else float(threshold)
    return SegmentationResult(
        labels=labels,
        mask=mask,
        threshold=reported,
        noise_floor=max(float(floor), 0.0) if np.isfinite(floor) else 0.0,
    )
