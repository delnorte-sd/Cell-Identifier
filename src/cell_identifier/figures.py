"""Publication-style distribution plots (optional; requires ``matplotlib``).

Install with ``pip install 'cell-identifier[plot]'``. The core pipeline never
imports this module, so matplotlib stays an optional dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .visualization import BRIGHTNESS_COLORS, OKABE_ITO

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

__all__ = ["save_summary_figure", "summary_figure"]


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def summary_figure(result: AnalysisResult) -> Any:
    """Return a matplotlib Figure with area and intensity distributions.

    Panels: (a) cell area histogram, (b) mean intensity histogram with the
    brightness cut-offs when absolute classification is used, (c) area vs.
    intensity scatter coloured by brightness class.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = np.array([c.area_px for c in result.cells], dtype=float)
    means = np.array([c.mean_intensity for c in result.cells], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    bins = "auto" if len(result.cells) > 1 else 1

    axes[0].hist(areas, bins=bins, color=_hex(OKABE_ITO["blue"]), edgecolor="white")
    axes[0].set(xlabel="Area (px²)", ylabel="Cells", title="(a) Cell area")

    axes[1].hist(means, bins=bins, color=_hex(OKABE_ITO["bluish_green"]), edgecolor="white")
    cls = result.config.classification
    if cls.method == "absolute":
        for cut in (cls.low_cutoff, cls.high_cutoff):
            axes[1].axvline(cut * result.full_scale, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Mean intensity (raw units)", ylabel="Cells", title="(b) Intensity")

    for name, colour in BRIGHTNESS_COLORS.items():
        sel = [c.brightness == name for c in result.cells]
        if any(sel):
            axes[2].scatter(
                areas[sel], means[sel], s=14, color=_hex(colour), label=name, edgecolors="none"
            )
    axes[2].set(xlabel="Area (px²)", ylabel="Mean intensity", title="(c) Area vs. intensity")
    if result.cells:
        axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{Path(result.metadata.source).name} - n = {result.cell_count}", fontsize=10)
    return fig


def save_summary_figure(result: AnalysisResult, path: str | Path, dpi: int = 200) -> Path:
    """Render :func:`summary_figure` to ``path`` (PNG, PDF or SVG by extension)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = summary_figure(result)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
