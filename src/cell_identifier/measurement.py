"""Per-cell measurements, object filtering and stable ID assignment.

Shape descriptors
-----------------
* **area** - pixel count (exact for the segmentation).
* **perimeter** - scikit-image's 4-connected, corner-weighted boundary length
  estimate, which is far less biased than counting boundary pixels.
* **equivalent diameter** - diameter of the disk with the same area.
* **major / minor axis** - axes of the ellipse with the same second central
  moments as the object.
* **orientation** - angle of the major axis in degrees, ``[0, 180)``, measured
  from the +x (column) axis towards +y (rows). Image rows run downwards, so on
  screen this is a *clockwise* angle.
* **eccentricity** - of the moment ellipse (0 = circle, →1 = line).
* **aspect ratio** - major / minor axis (``None`` for one-pixel-wide objects).
* **circularity** - 4π·area / perimeter², 1 for a perfect disk. Clipped at 1
  because perimeter estimates on tiny objects can push it slightly above.
* **solidity** - area / convex-hull area; low values flag clumps or irregular
  shapes.

Intensity descriptors are always measured on the **raw** intensity channel in
native units: mean, max, integrated (sum) and a background-corrected mean that
subtracts the median intensity of all background pixels.

Stable IDs
----------
Objects are numbered 1..N in raster order of their centroids (top-to-bottom,
then left-to-right), independent of the internal labelling algorithm, so the
same image and configuration always produce the same IDs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage import measure

from .config import FilterConfig

__all__ = ["CellMeasurement", "measure_cells", "relabel_sequential_by_position"]


@dataclass(frozen=True)
class CellMeasurement:
    """Measurements of one detected cell. Lengths are in pixels unless suffixed ``_um``."""

    cell_id: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    area_px: int
    perimeter_px: float
    equivalent_diameter_px: float
    major_axis_px: float
    minor_axis_px: float
    orientation_deg: float
    eccentricity: float
    aspect_ratio: float | None
    circularity: float
    solidity: float
    mean_intensity: float
    max_intensity: float
    integrated_intensity: float
    mean_intensity_corrected: float
    touches_border: bool
    brightness: str = "Unclassified"
    area_um2: float | None = None
    equivalent_diameter_um: float | None = None
    major_axis_um: float | None = None
    minor_axis_um: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a flat, JSON-serialisable dictionary."""
        return asdict(self)


def _orientation_deg(coords: np.ndarray) -> float:
    """Major-axis angle in degrees [0, 180) from +x towards +y, from pixel coordinates."""
    if len(coords) < 2:
        return 0.0
    ys = coords[:, 0].astype(np.float64)
    xs = coords[:, 1].astype(np.float64)
    cov = np.cov(np.vstack([xs, ys]))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    vx, vy = eigenvectors[:, int(np.argmax(eigenvalues))]
    return math.degrees(math.atan2(vy, vx)) % 180.0


def relabel_sequential_by_position(
    labels: np.ndarray, keep: list[int], centroids: dict[int, tuple[float, float]]
) -> tuple[np.ndarray, dict[int, int]]:
    """Renumber the kept labels 1..N in centroid raster order; drop all others.

    Args:
        labels: Input label image.
        keep: Labels to keep.
        centroids: ``label -> (row, col)`` centroid for each kept label.

    Returns:
        ``(new_labels, mapping)`` where ``mapping[old_label] = new_id``.
    """
    order = sorted(keep, key=lambda lab: (round(centroids[lab][0], 6), centroids[lab][1]))
    mapping = {old: new for new, old in enumerate(order, start=1)}
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for old, new in mapping.items():
        lut[old] = new
    return lut[labels], mapping


def measure_cells(
    labels: np.ndarray,
    intensity: np.ndarray,
    filters: FilterConfig,
    *,
    background_mask: np.ndarray | None = None,
    pixel_size_um: float | None = None,
) -> tuple[np.ndarray, list[CellMeasurement], float]:
    """Measure every labelled object, apply the acceptance filters and assign stable IDs.

    Args:
        labels: ``int`` label image from segmentation (0 = background).
        intensity: Raw intensity channel in native units, same shape as ``labels``.
        filters: Size and border rules.
        background_mask: Pixels used to estimate the background level. Defaults
            to all unlabelled pixels.
        pixel_size_um: Optional physical pixel size for µm / µm² outputs.

    Returns:
        ``(labels, cells, background_level)``: the relabelled image containing only
        accepted cells with IDs 1..N, the measurements in ID order, and the median
        background intensity (native units).
    """
    if labels.shape != intensity.shape:
        raise ValueError(f"labels {labels.shape} and intensity {intensity.shape} differ in shape")
    if background_mask is None:
        background_mask = labels == 0
    background_level = (
        float(np.median(intensity[background_mask])) if background_mask.any() else 0.0
    )

    height, width = labels.shape
    props = measure.regionprops(labels.astype(np.int32), intensity_image=intensity)
    accepted: dict[int, Any] = {}
    for region in props:
        if not filters.min_area <= region.area <= filters.max_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        on_border = min_row == 0 or min_col == 0 or max_row == height or max_col == width
        if filters.exclude_border and on_border:
            continue
        accepted[region.label] = region

    centroids = {lab: (float(r.centroid[0]), float(r.centroid[1])) for lab, r in accepted.items()}
    new_labels, mapping = relabel_sequential_by_position(labels, list(accepted), centroids)

    scale = pixel_size_um
    cells: list[CellMeasurement] = []
    for old_label, new_id in sorted(mapping.items(), key=lambda item: item[1]):
        region = accepted[old_label]
        min_row, min_col, max_row, max_col = region.bbox
        area = int(region.area)
        perimeter = float(region.perimeter)
        major = float(region.axis_major_length)
        minor = float(region.axis_minor_length)
        values = intensity[region.coords[:, 0], region.coords[:, 1]].astype(np.float64)
        mean_intensity = float(values.mean())
        circularity = min(1.0, 4.0 * math.pi * area / perimeter**2) if perimeter > 0 else 1.0
        equivalent_diameter = float(region.equivalent_diameter_area)
        cells.append(
            CellMeasurement(
                cell_id=new_id,
                centroid_x=round(float(region.centroid[1]), 3),
                centroid_y=round(float(region.centroid[0]), 3),
                bbox_x=int(min_col),
                bbox_y=int(min_row),
                bbox_width=int(max_col - min_col),
                bbox_height=int(max_row - min_row),
                area_px=area,
                perimeter_px=round(perimeter, 3),
                equivalent_diameter_px=round(equivalent_diameter, 3),
                major_axis_px=round(major, 3),
                minor_axis_px=round(minor, 3),
                orientation_deg=round(_orientation_deg(region.coords), 2),
                eccentricity=round(float(region.eccentricity), 4),
                aspect_ratio=round(major / minor, 4) if minor > 0 else None,
                circularity=round(circularity, 4),
                solidity=round(float(region.solidity), 4),
                mean_intensity=round(mean_intensity, 4),
                max_intensity=round(float(values.max()), 4),
                integrated_intensity=round(float(values.sum()), 4),
                mean_intensity_corrected=round(mean_intensity - background_level, 4),
                touches_border=bool(
                    min_row == 0 or min_col == 0 or max_row == height or max_col == width
                ),
                area_um2=round(area * scale**2, 6) if scale else None,
                equivalent_diameter_um=round(equivalent_diameter * scale, 6) if scale else None,
                major_axis_um=round(major * scale, 6) if scale else None,
                minor_axis_um=round(minor * scale, 6) if scale else None,
            )
        )
    return new_labels, cells, background_level
