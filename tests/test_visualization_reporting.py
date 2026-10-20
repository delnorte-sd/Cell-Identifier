from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cell_identifier import AnalysisConfig, analyze, load_image
from cell_identifier.config import apply_overrides
from cell_identifier.reporting import (
    CELL_COLUMNS,
    SCHEMA_VERSION,
    result_to_dict,
    write_batch_summary,
    write_cells_csv,
    write_json,
    write_outputs,
)
from cell_identifier.visualization import (
    BRIGHTNESS_COLORS,
    OKABE_ITO,
    comparison_panel,
    render_labels,
    render_mask,
    render_overlay,
    result_figure,
    to_rgb,
)


@pytest.fixture(scope="module")
def result(pill_image):
    return analyze(pill_image.image, AnalysisConfig(pixel_size_um=0.1))


# ---------------------------------------------------------------- visualisation


def test_overlay_is_non_destructive(result) -> None:
    original = result.image.copy()
    overlay = render_overlay(result.image, result.labels, result.cells, show_centroids=True)
    np.testing.assert_array_equal(result.image, original)
    assert overlay.shape == result.image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, result.image)
    # Outline pixels carry the brightness colours; background far from cells is untouched.
    colours = {tuple(int(v) for v in px) for px in overlay[result.labels > 0]}
    assert set(BRIGHTNESS_COLORS.values()) & colours
    assert np.array_equal(overlay[result.labels == 0][:100], result.image[result.labels == 0][:100])


def test_overlay_options_and_validation(result) -> None:
    uniform = render_overlay(
        result.image, result.labels, result.cells, color_by="uniform", show_ids=False, outline_px=2
    )
    outline = uniform[result.labels > 0]
    assert (outline == BRIGHTNESS_COLORS["Unclassified"]).all(axis=1).any()
    assert np.array_equal(render_overlay(result.image, result.labels, []), to_rgb(result.image))
    with pytest.raises(ValueError, match="same height"):
        render_overlay(result.image, result.labels[:10], result.cells)


def test_label_and_mask_rendering() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3], labels[5:7, 5:7] = 1, 2
    rgb = render_labels(labels)
    assert rgb.shape == (10, 10, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[1, 1]) != tuple(rgb[5, 5])
    assert tuple(rgb[1, 1]) in OKABE_ITO.values()
    mask = render_mask(labels > 0)
    assert mask[1, 1].tolist() == [255, 255, 255]
    assert mask[0, 0].tolist() == [0, 0, 0]


def test_comparison_panel_layout(result) -> None:
    fig = comparison_panel(
        [("a", result.image), ("b", result.image)], legend=[("x", (1, 2, 3))], gap=4
    )
    h, w = result.image.shape[:2]
    assert fig.shape[1] == 2 * w + 4
    assert fig.shape[0] == h + 30 + 26
    with pytest.raises(ValueError, match="at least one"):
        comparison_panel([])
    with pytest.raises(ValueError, match="same height"):
        comparison_panel([("a", result.image), ("b", result.image[:10])])
    assert result_figure(result).shape[1] > 3 * w


def test_okabe_ito_palette_is_the_published_one() -> None:
    assert OKABE_ITO["orange"] == (230, 159, 0)
    assert OKABE_ITO["sky_blue"] == (86, 180, 233)
    assert OKABE_ITO["vermillion"] == (213, 94, 0)
    assert len(set(BRIGHTNESS_COLORS.values())) == len(BRIGHTNESS_COLORS)


# -------------------------------------------------------------------- reporting


def test_result_dict_schema(result) -> None:
    report = result_to_dict(result)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["software"]["name"] == "cell-identifier"
    assert report["input"]["sha256"] == result.metadata.sha256
    assert report["config"]["pixel_size_um"] == 0.1
    assert report["summary"]["cell_count"] == len(report["cells"]) == result.cell_count
    assert set(report["cells"][0]) == set(CELL_COLUMNS)
    json.dumps(report, allow_nan=False)  # strict JSON: no numpy types, NaN or inf


def test_write_json_and_csv(result, tmp_path: Path) -> None:
    data = json.loads(write_json(result, tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["summary"]["cell_count"] == result.cell_count
    with write_cells_csv(result.cells, tmp_path / "c.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.cell_count
    assert tuple(rows[0]) == CELL_COLUMNS
    assert int(rows[0]["cell_id"]) == 1
    assert float(rows[0]["area_px"]) == result.cells[0].area_px


def test_csv_writes_missing_values_as_empty(tmp_path: Path, draw_discs) -> None:
    res = analyze(draw_discs((64, 64, 10)))  # no pixel size -> *_um fields are None
    with write_cells_csv(res.cells, tmp_path / "c.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["area_um2"] == ""


def test_write_outputs_bundle(result, tmp_path: Path) -> None:
    written = write_outputs(result, tmp_path / "out")
    assert set(written) == {"json", "csv", "overlay", "labels", "labels_16bit", "figure"}
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    raw_labels = load_image(written["labels_16bit"]).data
    assert raw_labels.dtype == np.uint16
    np.testing.assert_array_equal(raw_labels, result.labels)
    only_reports = write_outputs(result, tmp_path / "lean", images=False)
    assert set(only_reports) == {"json", "csv"}


def test_batch_summary(result, tmp_path: Path) -> None:
    path = write_batch_summary([result, ("broken.png", "could not decode")], tmp_path / "s.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "ok"
    assert int(rows[0]["cell_count"]) == result.cell_count
    assert rows[1]["status"] == "failed"
    assert rows[1]["error"] == "could not decode"


def test_quantile_mode_reports_counts(pill_image) -> None:
    config = apply_overrides(AnalysisConfig(), ["classification.method=quantile"])
    counts = analyze(pill_image.image, config).summary()["brightness_counts"]
    assert max(counts.values()) - min(counts.values()) <= 2


def test_summary_figure(result, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from cell_identifier.figures import save_summary_figure

    path = save_summary_figure(result, tmp_path / "dist.png", dpi=60)
    assert path.is_file()
    assert path.stat().st_size > 1000
