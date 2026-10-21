"""The end-to-end analysis pipeline: load → preprocess → segment → measure → classify.

The pipeline is a pure function of (image, configuration): it uses no global
state and no randomness, so identical inputs always give identical results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .classification import BRIGHTNESS_CLASSES, classify_brightness
from .config import AnalysisConfig
from .image_io import ImageMetadata, LoadedImage, load_image
from .learned import segment_learned
from .measurement import CellMeasurement, measure_cells
from .preprocessing import extract_channel, preprocess
from .segmentation import segment

__all__ = ["AnalysisResult", "analyze"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Everything produced by one analysis run.

    Attributes:
        cells: Accepted cells, ordered by ``cell_id`` (1..N).
        labels: ``int32`` label image whose values match ``cell_id``.
        image: The validated input image (native dtype, RGB or greyscale).
        intensity: Raw analysed intensity channel (float32, native units).
        processed: Segmentation image (float32, fraction of full scale).
        mask: Foreground mask before size/border filtering.
        threshold: Applied threshold (fraction of full scale).
        noise_floor: Noise floor from ``min_snr`` (fraction of full scale).
        background_level: Median background intensity (native units).
        full_scale: Full-scale intensity of the input dtype.
        metadata: Input provenance.
        config: Exact configuration used.
        engine_info: Which segmentation engine ran and, for the learned engine,
            the cell width used for rescaling.
        software_version: Package version that produced the result.
        created_at: UTC timestamp (ISO 8601).
    """

    cells: list[CellMeasurement]
    labels: np.ndarray
    image: np.ndarray
    intensity: np.ndarray
    processed: np.ndarray
    mask: np.ndarray
    threshold: float
    noise_floor: float
    background_level: float
    full_scale: float
    metadata: ImageMetadata
    config: AnalysisConfig
    engine_info: dict[str, Any] = field(default_factory=lambda: {"engine": "classical"})
    software_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def cell_count(self) -> int:
        """Number of accepted cells."""
        return len(self.cells)

    def summary(self) -> dict[str, Any]:
        """Population statistics for the image.

        Includes count, coverage (fraction of the field occupied by accepted
        cells), density, and mean / standard deviation / coefficient of
        variation of area and intensity. Standard deviations are sample
        standard deviations (``ddof=1``) and are ``None`` with fewer than two cells.
        """
        n = self.cell_count
        height, width = self.labels.shape
        field_px = float(height * width)

        def stats(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "std": None, "median": None, "cv": None}
            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else None
            return {
                "mean": round(mean, 4),
                "std": round(std, 4) if std is not None else None,
                "median": round(float(np.median(arr)), 4),
                "cv": round(std / mean, 4) if std is not None and mean > 0 else None,
            }

        brightness = dict.fromkeys(BRIGHTNESS_CLASSES, 0)
        for cell in self.cells:
            if cell.brightness in brightness:
                brightness[cell.brightness] += 1
        summary: dict[str, Any] = {
            "cell_count": n,
            "coverage_fraction": round(float(np.count_nonzero(self.labels)) / field_px, 6),
            "cells_per_megapixel": round(n / field_px * 1e6, 3),
            "area_px": stats([c.area_px for c in self.cells]),
            "circularity": stats([c.circularity for c in self.cells]),
            "aspect_ratio": stats(
                [c.aspect_ratio for c in self.cells if c.aspect_ratio is not None]
            ),
            "mean_intensity": stats([c.mean_intensity for c in self.cells]),
            "mean_intensity_corrected": stats([c.mean_intensity_corrected for c in self.cells]),
            "integrated_intensity": stats([c.integrated_intensity for c in self.cells]),
            "brightness_counts": brightness,
            **self.engine_info,
            "threshold_fraction": round(self.threshold, 6),
            "noise_floor_fraction": round(self.noise_floor, 6),
            "background_level": round(self.background_level, 4),
            "border_cells": sum(c.touches_border for c in self.cells),
        }
        if self.config.pixel_size_um:
            field_mm2 = field_px * (self.config.pixel_size_um / 1000.0) ** 2
            summary["cells_per_mm2"] = round(n / field_mm2, 3)
            summary["area_um2"] = stats([c.area_um2 for c in self.cells if c.area_um2 is not None])
        return summary


def analyze(
    source: str | Path | bytes | np.ndarray | LoadedImage,
    config: AnalysisConfig | None = None,
    *,
    name: str | None = None,
) -> AnalysisResult:
    """Detect, segment and measure cells in one image.

    Args:
        source: Image path, encoded bytes, RGB/greyscale array, or a
            :class:`~cell_identifier.image_io.LoadedImage`.
        config: Analysis settings; defaults to :class:`AnalysisConfig()`.
        name: Optional label recorded in the result metadata.

    Returns:
        An :class:`AnalysisResult`.

    Example:
        >>> from cell_identifier import analyze, generate
        >>> synthetic = generate(seed=1)
        >>> result = analyze(synthetic.image)
        >>> result.cell_count == synthetic.cell_count
        True
    """
    config = config or AnalysisConfig()
    loaded = source if isinstance(source, LoadedImage) else load_image(source, name=name)
    scale = loaded.full_scale
    intensity = extract_channel(loaded.data, config.preprocess.channel)
    pre = preprocess(intensity, scale, config.preprocess)
    if config.segmentation.engine == "learned":
        raw_labels, width = segment_learned(intensity, config.learned)
        mask = raw_labels > 0
        threshold = noise_floor = float("nan")  # not applicable: no intensity threshold
        engine_info: dict[str, Any] = {"engine": "learned", "cell_width_px": round(width, 2)}
    else:
        seg = segment(pre.processed, config.segmentation)
        raw_labels, mask = seg.labels, seg.mask
        threshold, noise_floor = seg.threshold, seg.noise_floor
        engine_info = {"engine": "classical"}
    labels, cells, background = measure_cells(
        raw_labels,
        intensity,
        config.filter,
        background_mask=~mask,
        pixel_size_um=config.pixel_size_um,
    )
    cells = classify_brightness(cells, config.classification, scale)
    logger.info(
        "%s: %d cells (%s engine)", loaded.metadata.source, len(cells), engine_info["engine"]
    )
    return AnalysisResult(
        cells=cells,
        labels=labels,
        image=loaded.data,
        intensity=intensity,
        processed=pre.processed,
        mask=mask,
        threshold=threshold,
        noise_floor=noise_floor,
        background_level=background,
        full_scale=scale,
        metadata=loaded.metadata,
        config=config,
        engine_info=engine_info,
    )
