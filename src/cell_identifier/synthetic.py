"""Synthetic fluorescence images with exact ground truth, for validating the pipeline.

This module is deliberately isolated from the analysis code: nothing in the
analysis pipeline imports it, and synthetic images are only ever used to
measure how well the pipeline recovers a *known* answer.

Image formation model
---------------------
For each cell a binary footprint is drawn analytically (a capsule for rods, an
ellipse otherwise) with a smooth brightness profile that falls off towards the
cell edge. Fluorescence from overlapping cells adds. The noise-free field is

    field = (background + Σ cells) × illumination

which is then blurred by a Gaussian point-spread function and corrupted by
additive Gaussian detector noise (noise is added *after* blurring, as in a real
camera), clipped to [0, 1], tinted with the fluorophore colour and quantised to
8 bits.

Ground truth is an ``int32`` label image in which each pixel belongs to the
top-most cell covering it (later cells are drawn on top), plus a list of the
generating parameters for every cell.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .config import SyntheticConfig, apply_overrides
from .preprocessing import disk

__all__ = ["SyntheticCell", "SyntheticImage", "generate"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticCell:
    """Generating parameters of one synthetic cell (lengths in pixels).

    ``length``/``width`` are the end-to-end rod length and width for pills and
    the major/minor axis lengths for ellipses. ``angle_deg`` follows the same
    convention as :class:`~cell_identifier.measurement.CellMeasurement`:
    from +x towards +y, in [0, 180).
    """

    cell_id: int
    shape: str
    center_x: float
    center_y: float
    length: float
    width: float
    angle_deg: float
    intensity: float
    area_px: int
    visible_area_px: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SyntheticImage:
    """A generated image together with its ground truth.

    Attributes:
        image: ``uint8`` RGB image ``(H, W, 3)``.
        labels: ``int32`` ground-truth labels (0 = background, 1..N = cells).
        cells: Parameters of each placed cell, ordered by ``cell_id``.
        requested: Number of cells requested.
        config: Configuration used (including the seed, if any).
    """

    image: np.ndarray
    labels: np.ndarray
    cells: list[SyntheticCell]
    requested: int
    config: SyntheticConfig

    @property
    def cell_count(self) -> int:
        """Number of cells actually placed."""
        return len(self.cells)

    def ground_truth_dict(self) -> dict[str, Any]:
        """JSON-serialisable description of the ground truth and its configuration."""
        return {
            "requested_cells": self.requested,
            "placed_cells": self.cell_count,
            "config": self.config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
        }


def _footprint(
    shape: str,
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle_deg: float,
) -> np.ndarray:
    """Return the normalised radial coordinate (0 at the centre line, 1 at the edge)."""
    theta = math.radians(angle_deg)
    dx, dy = xs - cx, ys - cy
    u = dx * math.cos(theta) + dy * math.sin(theta)  # along the long axis
    v = -dx * math.sin(theta) + dy * math.cos(theta)  # across it
    if shape == "pill":
        half_width = width / 2.0
        half_core = max(length - width, 0.0) / 2.0
        dist = np.hypot(np.maximum(np.abs(u) - half_core, 0.0), v)
        return dist / half_width
    semi_major, semi_minor = length / 2.0, width / 2.0
    return np.sqrt((u / semi_major) ** 2 + (v / semi_minor) ** 2)


def generate(config: SyntheticConfig | None = None, **overrides: Any) -> SyntheticImage:
    """Generate a synthetic fluorescence image with exact ground truth.

    Args:
        config: Generator settings. Defaults to :class:`SyntheticConfig()`.
        **overrides: Convenience keyword overrides, e.g. ``generate(seed=1,
            num_cells=40, shape="ellipse")``.

    Returns:
        A :class:`SyntheticImage`. When the requested density cannot be reached
        under the overlap rules, fewer cells are placed and a warning is logged.
    """
    config = config or SyntheticConfig()
    if overrides:
        config = apply_overrides(
            config, [f"{k}={json.dumps(_listify(v))}" for k, v in overrides.items()]
        )
    rng = np.random.default_rng(config.seed)
    height, width = config.height, config.width

    labels = np.zeros((height, width), dtype=np.int32)
    signal = np.zeros((height, width), dtype=np.float64)
    placed: list[dict[str, Any]] = []
    separation_kernel = disk(config.min_separation) if config.min_separation > 0 else None

    max_attempts = config.max_attempts_per_cell * max(config.num_cells, 1)
    attempts = 0
    while len(placed) < config.num_cells and attempts < max_attempts:
        attempts += 1
        cx = rng.uniform(config.margin, width - 1 - config.margin)
        cy = rng.uniform(config.margin, height - 1 - config.margin)
        if config.shape == "pill":
            length = float(rng.integers(config.pill_length[0], config.pill_length[1] + 1))
            cell_width = float(rng.integers(config.pill_width[0], config.pill_width[1] + 1))
            length = max(length, cell_width)
        else:
            length = float(rng.integers(config.ellipse_major[0], config.ellipse_major[1] + 1))
            cell_width = max(1.0, length * rng.uniform(*config.ellipse_minor_ratio))
        angle = float(rng.uniform(0.0, 180.0))
        peak = float(rng.uniform(*config.intensity))

        # Render inside a local window only, for speed.
        reach = math.ceil(length / 2.0 + cell_width) + config.min_separation + 2
        x0, x1 = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        radial = _footprint(config.shape, xs, ys, cx, cy, length, cell_width, angle)
        mask = radial <= 1.0
        area = int(mask.sum())
        if area == 0:
            continue
        occupied = labels[y0:y1, x0:x1] > 0
        if config.allow_overlap:
            if (mask & occupied).sum() / area > config.max_overlap_fraction:
                continue
        else:
            grown = mask
            if separation_kernel is not None:
                grown = cv2.dilate(mask.astype(np.uint8), separation_kernel).astype(bool)
            if (grown & occupied).any():
                continue

        cell_id = len(placed) + 1
        profile = np.where(mask, peak * (1.0 - config.edge_falloff * np.clip(radial, 0, 1) ** 2), 0)
        signal[y0:y1, x0:x1] += profile
        labels[y0:y1, x0:x1][mask] = cell_id
        placed.append(
            {
                "cell_id": cell_id,
                "shape": config.shape,
                "center_x": round(cx, 3),
                "center_y": round(cy, 3),
                "length": round(length, 3),
                "width": round(cell_width, 3),
                "angle_deg": round(angle, 3),
                "intensity": round(peak, 4),
                "area_px": area,
            }
        )

    if len(placed) < config.num_cells:
        logger.warning(
            "placed %d of %d requested cells; the field is too crowded for the overlap rules",
            len(placed),
            config.num_cells,
        )

    visible = np.bincount(labels.ravel(), minlength=len(placed) + 1)
    cells = [SyntheticCell(**p, visible_area_px=int(visible[p["cell_id"]])) for p in placed]

    field: np.ndarray = config.background_level + signal
    if config.illumination_gradient > 0:
        ramp = 1.0 - config.illumination_gradient * np.linspace(0.0, 1.0, width)
        field = field * ramp[np.newaxis, :]
    if config.blur_sigma > 0:
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=config.blur_sigma)
    if config.noise_sigma > 0:
        field = field + rng.normal(0.0, config.noise_sigma, size=field.shape)
    field = np.clip(field, 0.0, 1.0)
    color = np.asarray(config.color, dtype=np.float64)
    image = np.rint(field[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]).astype(np.uint8)
    return SyntheticImage(
        image=image, labels=labels, cells=cells, requested=config.num_cells, config=config
    )


def _listify(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value
