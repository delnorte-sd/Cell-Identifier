"""Learned segmentation engine: a trained U-Net run through OpenCV's DNN module.

The network (see ``training/model.py``) predicts, per pixel, a foreground
probability, a cell-boundary probability and a per-cell normalised distance
map. Instances are recovered by a seeded watershed:

1. foreground = ``p_fg > prob_threshold``;
2. seeds = connected regions where the normalised distance exceeds
   ``seed_threshold`` (cell mid-lines, which stay separate even when cells
   touch along their whole length);
3. a watershed on the inverted distance map grows the seeds to fill the
   foreground;
4. objects the network is unsure about (low mean probability) or that are not
   clearly brighter than the background (contrast below ``min_snr`` robust
   noise standard deviations) are discarded. Because inputs are contrast-
   stretched before inference, this physical noise test is what lets an empty
   or noise-only image correctly yield no cells.

Each model is trained with cells rescaled to a fixed working width (stored in
the model's JSON metadata as ``target_cell_width``; :data:`TARGET_CELL_WIDTH`
if absent). Images are processed in tiles of the *original* image: each tile
is resized by ``target_width / cell_width``, run through the network, and the
probability maps are resized back, so memory stays bounded however much an
image is up-scaled and labels are produced on the original pixel grid. The
cell width is configured or estimated automatically.

Inference needs only OpenCV, which is already a core dependency; PyTorch is
needed for training only.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries, watershed

from .config import LearnedConfig

__all__ = [
    "DEFAULT_MODEL",
    "TARGET_CELL_WIDTH",
    "LearnedModel",
    "estimate_cell_width",
    "load_model",
    "make_targets",
    "normalize_input",
    "postprocess",
    "reject_low_contrast",
    "segment_learned",
]

#: Default working cell width (px) for models whose metadata does not specify one.
TARGET_CELL_WIDTH = 10.0
#: File name of the bundled model inside ``cell_identifier/models``.
DEFAULT_MODEL = "cell_unet.onnx"

_EIGHT = np.ones((3, 3), dtype=bool)
#: Context margin (px of the original image) added around each tile.
_TILE_MARGIN = 32


def normalize_input(intensity: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    """Map the ``low``/``high`` intensity percentiles to 0/1 (float32, clipped).

    Percentile normalisation makes the network independent of bit depth,
    exposure and offset. A constant image maps to zeros.
    """
    lo, hi = np.percentile(intensity, [low, high])
    if hi <= lo:
        return np.zeros(intensity.shape, dtype=np.float32)
    scaled = (intensity.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def make_targets(labels: np.ndarray) -> np.ndarray:
    """Training targets ``(3, H, W)`` float32 from an instance label image.

    Channels: foreground, inner boundary (including seams between touching
    cells), and each cell's distance transform divided by its own maximum.
    """
    labels = labels.astype(np.int32)
    fg = (labels > 0).astype(np.float32)
    boundary = find_boundaries(labels, connectivity=1, mode="inner").astype(np.float32)
    dist = np.zeros(labels.shape, dtype=np.float32)
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        padded = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, n))
            for s, n in zip(box, labels.shape, strict=True)
        )
        cell = labels[padded] == index
        edt = ndimage.distance_transform_edt(cell)
        peak = float(edt.max())
        if peak > 0:
            dist[padded][cell] = (edt[cell] / peak).astype(np.float32)
    return np.stack([fg, boundary, dist])


def postprocess(
    probs: np.ndarray,
    prob_threshold: float = 0.5,
    seed_threshold: float = 0.5,
    min_seed_area: int = 4,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """Turn ``(3, H, W)`` network outputs into an ``int32`` instance label image.

    ``min_confidence`` drops objects whose mean foreground probability is
    below it (e.g. faint out-of-focus haze the network is unsure about).
    """
    p_fg, p_boundary, dist = probs
    mask = p_fg > prob_threshold
    labels = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return labels
    seeds = (dist > seed_threshold) & mask & (p_boundary < 0.5)
    # Flooding cannot cross between separate foreground regions, so each region is
    # processed on its own bounding box: identical result, memory bounded by the
    # largest clump rather than the whole image.
    regions, _ = ndimage.label(mask, structure=_EIGHT)
    next_id = 1
    for index, box in enumerate(ndimage.find_objects(regions), start=1):
        if box is None:
            continue
        region = regions[box] == index
        markers, count = ndimage.label(seeds[box] & region, structure=_EIGHT)
        if count:
            sizes = np.bincount(markers.ravel())
            small = sizes < min_seed_area
            small[0] = False
            markers[small[markers]] = 0
            markers, count = ndimage.label(markers > 0, structure=_EIGHT)
        if count == 0:  # no usable seed: the whole region is one cell
            labels[box][region] = next_id
            next_id += 1
            continue
        relief = -np.asarray(dist[box], dtype=np.float32)
        flooded = watershed(relief, markers, mask=region, connectivity=2)
        labels[box][region] = flooded[region] + (next_id - 1)
        next_id += count
    if min_confidence > 0 and labels.max() > 0:
        counts = np.bincount(labels.ravel())
        sums = np.bincount(labels.ravel(), weights=p_fg.ravel())
        mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        weak = mean < min_confidence
        weak[0] = False
        labels = np.where(weak[labels], 0, labels)
    return _relabel(labels)


def reject_low_contrast(labels: np.ndarray, intensity: np.ndarray, min_snr: float) -> np.ndarray:
    """Drop objects whose mean raw intensity is < background + ``min_snr`` x noise sigma.

    The background level is the median of unlabelled pixels; sigma is the robust,
    gradient-insensitive pixel-noise estimate used by the classical engine.
    """
    if min_snr <= 0 or labels.max() == 0 or not (labels == 0).any():
        return labels
    from .segmentation import estimate_noise_sigma

    image = intensity.astype(np.float32)
    background = float(np.median(image[labels == 0]))
    sigma = estimate_noise_sigma(image)
    counts = np.bincount(labels.ravel())
    sums = np.bincount(labels.ravel(), weights=image.ravel())
    mean = np.divide(sums, counts, out=np.full_like(sums, background), where=counts > 0)
    faint = mean < background + min_snr * max(sigma, 1e-12)
    faint[0] = False
    return _relabel(np.where(faint[labels], 0, labels))


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Renumber labels consecutively (1..N)."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(np.max(labels)) + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels]


class LearnedModel:
    """A U-Net loaded with ``cv2.dnn``. Thread-safe; inference is serialised."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self._lock = threading.Lock()
        meta_path = path.with_suffix(".json")
        self.metadata: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
        #: Working cell width (px) the model was trained at.
        self.target_width = float(self.metadata.get("target_cell_width", TARGET_CELL_WIDTH))

    def _forward(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        ph, pw = (-h) % 16, (-w) % 16
        padded = np.pad(image, ((0, ph), (0, pw)), mode="reflect") if ph or pw else image
        blob = np.ascontiguousarray(padded[np.newaxis, np.newaxis], dtype=np.float32)
        with self._lock:
            self.net.setInput(blob)
            out = self.net.forward()
        return np.asarray(out[0, :, :h, :w], dtype=np.float32)

    def predict(self, image: np.ndarray, tile: int = 512, overlap: int = 64) -> np.ndarray:
        """Return ``(3, H, W)`` probability maps for a normalised float image.

        Images larger than ``tile`` are processed in overlapping tiles and
        stitched from the tile centres, which keeps memory bounded.
        """
        h, w = image.shape
        if h <= tile and w <= tile:
            return self._forward(image)
        out = np.zeros((3, h, w), dtype=np.float32)
        step = tile - 2 * overlap
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                ys, xs = max(y0 - overlap, 0), max(x0 - overlap, 0)
                ye, xe = min(y0 + step + overlap, h), min(x0 + step + overlap, w)
                pred = self._forward(image[ys:ye, xs:xe])
                cy, cx = min(y0 + step, h), min(x0 + step, w)
                out[:, y0:cy, x0:cx] = pred[:, y0 - ys : cy - ys, x0 - xs : cx - xs]
        return out


def _resolve(model: str) -> Path:
    if model:
        return Path(model)
    return Path(str(resources.files("cell_identifier").joinpath("models", DEFAULT_MODEL)))


@lru_cache(maxsize=4)
def _load(path: str) -> LearnedModel:
    return LearnedModel(Path(path))


def load_model(model: str = "") -> LearnedModel:
    """Load (and cache) a model; ``""`` selects the bundled default."""
    path = _resolve(model)
    if not path.is_file():
        raise FileNotFoundError(f"learned segmentation model not found: {path}")
    return _load(str(path.resolve()))


#: Candidate cell widths (px) tried by :func:`estimate_cell_width`; neighbours differ by √2,
#: so the true width is always within ±19 % of a candidate.
WIDTH_CANDIDATES = (5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0)


def _resize(image: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) <= 0.05:
        return image
    h, w = image.shape[:2]
    size = (max(16, round(w * factor)), max(16, round(h * factor)))
    interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interp)


def _signal_crop(image: np.ndarray, size: int) -> np.ndarray:
    """The ``size`` x ``size`` window (on a coarse grid) with the most bright pixels."""
    h, w = image.shape
    if h <= size and w <= size:
        return image
    bright = image > np.percentile(image, 90)
    best, best_score = (0, 0), -1.0
    for y in range(0, max(h - size, 0) + 1, max(size // 2, 1)):
        for x in range(0, max(w - size, 0) + 1, max(size // 2, 1)):
            score = float(bright[y : y + size, x : x + size].mean())
            if score > best_score:
                best, best_score = (y, x), score
    y, x = best
    return image[y : y + size, x : x + size]


def _median_predicted_width(model: LearnedModel, image: np.ndarray) -> tuple[float, int]:
    from skimage import measure

    labels = postprocess(model.predict(image))
    widths = [r.axis_minor_length for r in measure.regionprops(labels) if r.area >= 12]
    return (float(np.median(widths)) if widths else 0.0), len(widths)


def _search_cell_width(model: LearnedModel, normalized: np.ndarray) -> float | None:
    """Exhaustive fallback: try every candidate width, keep the most self-consistent."""
    best: tuple[float, float] | None = None  # (log error, refined width)
    for candidate in WIDTH_CANDIDATES:
        factor = model.target_width / candidate
        crop = _signal_crop(normalized, max(64, round(256 / factor)))
        measured, count = _median_predicted_width(model, _resize(crop, factor))
        if count < 3 or measured <= 0:
            continue
        error = abs(np.log(measured / model.target_width))
        refined = candidate * measured / model.target_width
        if best is None or error < best[0]:
            best = (error, refined)
    return None if best is None else best[1]


def estimate_cell_width(model: LearnedModel, normalized: np.ndarray, max_passes: int = 4) -> float:
    """Estimate the typical cell width (px) of an image using the model itself.

    The model segments a signal-rich crop at the current guess of the scale
    (starting at the model's working width), the median width of the detected
    cells is measured, and the guess is corrected by that ratio. At the right
    scale detected cells are as wide as the working width, so this converges in one to four
    passes on a small crop - cheap enough for a modest web server. Only if no
    cells are found at the default scale is every candidate width tried.
    """
    width, found = model.target_width, False
    for _ in range(max_passes):
        factor = model.target_width / width
        crop = _signal_crop(normalized, max(64, round(256 / factor)))
        measured, count = _median_predicted_width(model, _resize(crop, factor))
        if count < 3 or measured <= 0:
            break
        found = True
        ratio = measured / model.target_width
        width = float(np.clip(width * ratio, 3.0, 120.0))
        if abs(np.log(ratio)) < 0.15:
            break
    if found:
        return width
    # No cells at the default scale: they may be far smaller or larger than expected.
    searched = _search_cell_width(model, normalized)
    return float(np.clip(searched, 3.0, 120.0)) if searched else model.target_width


def segment_learned(intensity: np.ndarray, config: LearnedConfig) -> tuple[np.ndarray, float]:
    """Segment a raw intensity image with the learned engine.

    Returns ``(labels, cell_width_px)`` where ``cell_width_px`` is the width
    used for rescaling (configured or estimated).
    """
    model = load_model(config.model)
    normalized = normalize_input(intensity)
    width = config.cell_width_px or estimate_cell_width(model, normalized)
    factor = model.target_width / width
    h, w = normalized.shape
    probs = np.zeros((3, h, w), dtype=np.float16)  # half precision halves peak memory
    # Tile in original coordinates so memory stays bounded even when up-scaling.
    core = max(64, int(config.tile_size / max(factor, 1e-6)) - 2 * _TILE_MARGIN)
    for y0 in range(0, h, core):
        for x0 in range(0, w, core):
            ys, xs = max(y0 - _TILE_MARGIN, 0), max(x0 - _TILE_MARGIN, 0)
            ye, xe = min(y0 + core + _TILE_MARGIN, h), min(x0 + core + _TILE_MARGIN, w)
            tile = normalized[ys:ye, xs:xe]
            pred = model.predict(_resize(tile, factor), tile=config.tile_size * 2)
            if pred.shape[1:] != tile.shape:
                size = (tile.shape[1], tile.shape[0])
                pred = np.stack([cv2.resize(c, size, interpolation=cv2.INTER_LINEAR) for c in pred])
            cy, cx = min(y0 + core, h), min(x0 + core, w)
            probs[:, y0:cy, x0:cx] = pred[:, y0 - ys : cy - ys, x0 - xs : cx - xs]
    labels = postprocess(
        probs,
        config.prob_threshold,
        config.seed_threshold,
        config.min_seed_area,
        config.min_confidence,
    )
    labels = reject_low_contrast(labels, intensity, config.min_snr)
    return labels, float(width)
