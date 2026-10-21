"""Non-destructive visualisation of analysis results.

All renderers return **new** ``uint8`` RGB arrays and never modify their
inputs. Colours come from the Okabe-Ito palette, which stays distinguishable
under the common forms of colour-vision deficiency, and outlines are drawn on
exact pixel boundaries of the label image (no contour approximation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from .image_io import to_display_uint8
from .measurement import CellMeasurement

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

__all__ = [
    "BRIGHTNESS_COLORS",
    "OKABE_ITO",
    "comparison_panel",
    "encode_label_ids",
    "render_label_layer",
    "render_labels",
    "render_mask",
    "render_outline_layer",
    "render_overlay",
    "render_processed",
    "result_figure",
    "to_rgb",
]

#: Okabe & Ito (2008) colour-blind-safe palette, RGB.
OKABE_ITO: dict[str, tuple[int, int, int]] = {
    "orange": (230, 159, 0),
    "sky_blue": (86, 180, 233),
    "bluish_green": (0, 158, 115),
    "yellow": (240, 228, 66),
    "blue": (0, 114, 178),
    "vermillion": (213, 94, 0),
    "reddish_purple": (204, 121, 167),
    "black": (0, 0, 0),
}

#: Outline colour per brightness class; chosen to contrast with green fluorescence.
BRIGHTNESS_COLORS: dict[str, tuple[int, int, int]] = {
    "Low": OKABE_ITO["sky_blue"],
    "Medium": OKABE_ITO["yellow"],
    "High": OKABE_ITO["vermillion"],
    "Unclassified": OKABE_ITO["orange"],
}

_LABEL_CYCLE = [c for name, c in OKABE_ITO.items() if name != "black"]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PANEL_BG = (17, 17, 17)
_TEXT = (240, 240, 240)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit RGB copy of any greyscale or RGB image."""
    display = to_display_uint8(image)
    if display.ndim == 2:
        display = np.repeat(display[:, :, np.newaxis], 3, axis=2)
    return np.ascontiguousarray(display)


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float) -> None:
    """Draw legible text: a dark halo under light glyphs."""
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, _TEXT, 1, cv2.LINE_AA)


def render_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    cells: Sequence[CellMeasurement],
    *,
    color_by: Literal["brightness", "uniform"] = "brightness",
    show_ids: bool = True,
    show_centroids: bool = False,
    outline_px: int = 1,
) -> np.ndarray:
    """Draw cell outlines (and optionally IDs / centroids) over a copy of ``image``.

    Args:
        image: Original image (any supported dtype, greyscale or RGB).
        labels: Label image whose values are the ``cell_id`` of ``cells``.
        cells: Measurements, used for colours, IDs and centroids.
        color_by: Colour outlines by brightness class, or use one colour.
        show_ids: Write each cell's ID next to it.
        show_centroids: Mark centroids with a small dot.
        outline_px: Outline thickness in pixels (drawn inwards from the boundary).
    """
    canvas = to_rgb(image)
    if labels.shape != canvas.shape[:2]:
        raise ValueError("labels and image must have the same height and width")
    if not cells:
        return canvas
    boundaries = find_boundaries(labels, connectivity=1, mode="inner")
    if outline_px > 1:
        boundaries = cv2.dilate(
            boundaries.astype(np.uint8), np.ones((outline_px, outline_px), np.uint8)
        ).astype(bool) & (labels > 0)
    lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
    for cell in cells:
        key = cell.brightness if color_by == "brightness" else "Unclassified"
        lut[cell.cell_id] = BRIGHTNESS_COLORS.get(key, BRIGHTNESS_COLORS["Unclassified"])
    canvas[boundaries] = lut[labels[boundaries]]
    font_scale = max(0.3, min(0.6, canvas.shape[1] / 1600))
    for cell in cells:
        x, y = round(cell.centroid_x), round(cell.centroid_y)
        if show_centroids:
            cv2.circle(canvas, (x, y), 1, OKABE_ITO["reddish_purple"], -1, cv2.LINE_AA)
        if show_ids:
            _put_text(
                canvas, str(cell.cell_id), (cell.bbox_x + cell.bbox_width + 1, y + 4), font_scale
            )
    return canvas


def render_labels(labels: np.ndarray) -> np.ndarray:
    """Colour each labelled object with a cycling colour-blind-safe palette on black."""
    palette = np.array([(0, 0, 0), *_LABEL_CYCLE], dtype=np.uint8)
    index = np.where(labels > 0, (labels - 1) % len(_LABEL_CYCLE) + 1, 0)
    return palette[index]


def render_outline_layer(
    labels: np.ndarray,
    cells: Sequence[CellMeasurement],
    *,
    color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Transparent RGBA layer containing only the cell outlines.

    Outline pixels are opaque and coloured by brightness class (or ``color`` when
    given, e.g. for ground truth); every other pixel is fully transparent, so the
    layer can be composited over any base image at any opacity.
    """
    layer = np.zeros((*labels.shape, 4), dtype=np.uint8)
    if not cells and color is None:
        return layer
    boundaries = find_boundaries(labels, connectivity=1, mode="inner")
    if color is not None:
        layer[boundaries, :3] = color
    else:
        lut = np.zeros((int(labels.max()) + 1, 3), dtype=np.uint8)
        for cell in cells:
            lut[cell.cell_id] = BRIGHTNESS_COLORS.get(
                cell.brightness, BRIGHTNESS_COLORS["Unclassified"]
            )
        layer[boundaries, :3] = lut[labels[boundaries]]
    layer[boundaries, 3] = 255
    return layer


def render_label_layer(labels: np.ndarray) -> np.ndarray:
    """Transparent RGBA layer: each cell filled with a palette colour, background clear."""
    layer = np.zeros((*labels.shape, 4), dtype=np.uint8)
    layer[..., :3] = render_labels(labels)
    layer[labels > 0, 3] = 255
    return layer


def encode_label_ids(labels: np.ndarray) -> np.ndarray:
    """Pack label IDs into 24-bit RGB (R = low byte) so a browser can read exact IDs.

    Supports up to 16 777 215 labels; decode with ``r + 256 * g + 65536 * b``.
    """
    ids = labels.astype(np.uint32)
    if ids.size and int(np.max(ids)) > 0xFFFFFF:
        raise ValueError("too many labels to encode in 24 bits")
    return np.stack([ids & 0xFF, (ids >> 8) & 0xFF, (ids >> 16) & 0xFF], axis=-1).astype(np.uint8)


def render_processed(processed: np.ndarray) -> np.ndarray:
    """8-bit greyscale view of the segmentation image, linearly stretched min → max.

    This is a display stretch only; measurements never use this image.
    """
    lo, hi = float(np.min(processed)), float(np.max(processed))
    if hi <= lo:
        return np.zeros(processed.shape, dtype=np.uint8)
    return np.rint((processed - lo) / (hi - lo) * 255).astype(np.uint8)


def render_mask(mask: np.ndarray) -> np.ndarray:
    """Render a boolean mask as white-on-black RGB."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = 255
    return out


def _bar(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), _PANEL_BG, dtype=np.uint8)


def _legend(width: int, items: Sequence[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    bar = _bar(width, 26)
    x = 8
    for text, colour in items:
        cv2.rectangle(bar, (x, 8), (x + 12, 20), colour, -1)
        cv2.putText(bar, text, (x + 18, 19), _FONT, 0.45, _TEXT, 1, cv2.LINE_AA)
        x += 30 + cv2.getTextSize(text, _FONT, 0.45, 1)[0][0]
    return bar


def comparison_panel(
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    legend: Sequence[tuple[str, tuple[int, int, int]]] | None = None,
    gap: int = 6,
) -> np.ndarray:
    """Place titled RGB panels side by side on a dark background.

    Args:
        panels: ``(title, image)`` pairs; images must share their height.
        legend: Optional ``(label, colour)`` swatches drawn below the panels.
        gap: Spacing between panels in pixels.
    """
    if not panels:
        raise ValueError("at least one panel is required")
    heights = {img.shape[0] for _, img in panels}
    if len(heights) != 1:
        raise ValueError("all panels must have the same height")
    columns: list[np.ndarray] = []
    for title, img in panels:
        rgb = to_rgb(img)
        header = _bar(rgb.shape[1], 30)
        cv2.putText(header, title, (8, 21), _FONT, 0.55, _TEXT, 1, cv2.LINE_AA)
        columns.append(np.vstack([header, rgb]))
        columns.append(_bar(gap, header.shape[0] + rgb.shape[0]))
    figure = np.hstack(columns[:-1])
    if legend:
        figure = np.vstack([figure, _legend(figure.shape[1], legend)])
    return figure


def result_figure(result: AnalysisResult) -> np.ndarray:
    """Original | overlay | label-mask comparison figure for an analysis result."""
    overlay = render_overlay(result.image, result.labels, result.cells)
    legend = None
    if result.config.classification.method != "none":
        legend = [(name, BRIGHTNESS_COLORS[name]) for name in ("Low", "Medium", "High")]
    return comparison_panel(
        [
            ("Original", result.image),
            (f"Detected cells (n={result.cell_count})", overlay),
            ("Segmentation labels", render_labels(result.labels)),
        ],
        legend=legend,
    )
