"""Structured export of analysis results (JSON, CSV) and output-folder writing.

JSON reports are self-describing: they carry a schema version, the software
version, the input file hash, the complete configuration and a timestamp, so
any number in them can be traced back to exactly how it was produced.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .image_io import save_image
from .measurement import CellMeasurement
from .pipeline import AnalysisResult
from .visualization import render_labels, render_overlay, result_figure

__all__ = [
    "CELL_COLUMNS",
    "SCHEMA_VERSION",
    "result_to_dict",
    "write_batch_summary",
    "write_cells_csv",
    "write_json",
    "write_outputs",
]

#: Version of the JSON report layout; bump on breaking changes.
SCHEMA_VERSION = "1.0"
#: Column order of per-cell CSV files.
CELL_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(CellMeasurement))


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats into strict-JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Serialise a result (minus its image arrays) to a strict-JSON-compatible dict."""
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "software": {"name": "cell-identifier", "version": result.software_version},
            "created_at": result.created_at,
            "input": result.metadata.to_dict(),
            "config": result.config.to_dict(),
            "summary": result.summary(),
            "cells": [c.to_dict() for c in result.cells],
        }
    )


def write_json(result: AnalysisResult, path: str | Path) -> Path:
    """Write the full JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result_to_dict(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def write_cells_csv(cells: Sequence[CellMeasurement], path: str | Path) -> Path:
    """Write one row per cell with the columns in :data:`CELL_COLUMNS`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_COLUMNS)
        writer.writeheader()
        for cell in cells:
            writer.writerow({k: "" if v is None else v for k, v in cell.to_dict().items()})
    return path


_SUMMARY_COLUMNS = (
    "image",
    "sha256",
    "cell_count",
    "coverage_fraction",
    "cells_per_megapixel",
    "mean_area_px",
    "std_area_px",
    "mean_intensity",
    "std_intensity",
    "mean_intensity_corrected",
    "mean_circularity",
    "low",
    "medium",
    "high",
    "threshold_fraction",
    "status",
    "error",
)


def write_batch_summary(rows: Iterable[AnalysisResult | tuple[str, str]], path: str | Path) -> Path:
    """Write one row per image: results for successes, ``(name, error)`` for failures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            if isinstance(row, tuple):
                writer.writerow({"image": row[0], "status": "failed", "error": row[1]})
                continue
            s = row.summary()
            writer.writerow(
                {
                    "image": row.metadata.source,
                    "sha256": row.metadata.sha256,
                    "cell_count": s["cell_count"],
                    "coverage_fraction": s["coverage_fraction"],
                    "cells_per_megapixel": s["cells_per_megapixel"],
                    "mean_area_px": s["area_px"]["mean"],
                    "std_area_px": s["area_px"]["std"],
                    "mean_intensity": s["mean_intensity"]["mean"],
                    "std_intensity": s["mean_intensity"]["std"],
                    "mean_intensity_corrected": s["mean_intensity_corrected"]["mean"],
                    "mean_circularity": s["circularity"]["mean"],
                    "low": s["brightness_counts"]["Low"],
                    "medium": s["brightness_counts"]["Medium"],
                    "high": s["brightness_counts"]["High"],
                    "threshold_fraction": s["threshold_fraction"],
                    "status": "ok",
                    "error": "",
                }
            )
    return path


def write_outputs(
    result: AnalysisResult, directory: str | Path, *, images: bool = True
) -> dict[str, Path]:
    """Write the standard output bundle for one image into ``directory``.

    Files: ``results.json``, ``cells.csv`` and, when ``images`` is true,
    ``overlay.png``, ``labels.png`` (colour-coded), ``labels_16bit.png``
    (raw IDs for downstream tools) and ``figure.png`` (side-by-side panel).
    """
    directory = Path(directory)
    written = {
        "json": write_json(result, directory / "results.json"),
        "csv": write_cells_csv(result.cells, directory / "cells.csv"),
    }
    if images:
        written["overlay"] = save_image(
            directory / "overlay.png", render_overlay(result.image, result.labels, result.cells)
        )
        written["labels"] = save_image(directory / "labels.png", render_labels(result.labels))
        if result.labels.max() <= np.iinfo(np.uint16).max:
            written["labels_16bit"] = save_image(
                directory / "labels_16bit.png", result.labels.astype(np.uint16)
            )
        written["figure"] = save_image(directory / "figure.png", result_figure(result))
    return written
