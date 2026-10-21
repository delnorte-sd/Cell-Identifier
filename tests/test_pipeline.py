"""End-to-end tests of analyze() against known ground truth."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from cell_identifier import (
    AnalysisConfig,
    SyntheticImage,
    analyze,
    generate,
    load_image,
    match_labels,
)
from cell_identifier.config import apply_overrides
from cell_identifier.reporting import result_to_dict

CLASSICAL = apply_overrides(AnalysisConfig(), ["segmentation.engine=classical"])


@pytest.mark.parametrize("fixture", ["pill_image", "ellipse_image"])
def test_classical_recovers_exact_count_on_separated_cells(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    synthetic: SyntheticImage = request.getfixturevalue(fixture)
    assert synthetic.cell_count == 40
    result = analyze(synthetic.image, CLASSICAL)
    metrics = match_labels(result.labels, synthetic.labels)
    assert result.cell_count == synthetic.cell_count
    assert metrics.f1 == 1.0
    assert metrics.mean_iou > 0.85


@pytest.mark.parametrize("fixture", ["pill_image", "ellipse_image"])
def test_default_learned_engine_detects_separated_cells(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    synthetic: SyntheticImage = request.getfixturevalue(fixture)
    result = analyze(synthetic.image)
    assert result.summary()["engine"] == "learned"
    # The model draws outlines the way human annotators do (slightly wider than the
    # generator's nominal footprint), so detection is scored, not pixel overlap.
    assert match_labels(result.labels, synthetic.labels).f1 >= 0.95


def test_default_learned_engine_with_overlapping_cells() -> None:
    scores = []
    for seed in range(3):
        synthetic = generate(seed=seed, shape="pill", allow_overlap=True, num_cells=60)
        scores.append(match_labels(analyze(synthetic.image).labels, synthetic.labels).f1)
    # Measured ~0.87. The synthetic convention (the top cell owns every shared pixel of
    # up to 30 % overlap) differs from real annotations; on expert-annotated real images
    # the learned engine scores F1 0.93 vs 0.50 for the classical one (docs/TRAINING.md).
    assert np.mean(scores) >= 0.8, scores


@pytest.mark.parametrize("split", ["distance", "intensity"])
@pytest.mark.parametrize("shape", ["pill", "ellipse"])
def test_high_f1_with_overlapping_cells(shape: str, split: str) -> None:
    config = apply_overrides(CLASSICAL, [f"segmentation.split_method={split}"])
    scores = []
    for seed in range(3):
        synthetic = generate(seed=seed, shape=shape, allow_overlap=True, num_cells=60)
        scores.append(match_labels(analyze(synthetic.image, config).labels, synthetic.labels).f1)
    assert np.mean(scores) >= 0.93, scores


def test_robust_to_noise_and_uneven_illumination() -> None:
    synthetic = generate(seed=5, num_cells=40, noise_sigma=0.05, illumination_gradient=0.5)
    config = apply_overrides(
        CLASSICAL, ["preprocess.background=tophat", "preprocess.background_radius=20"]
    )
    assert match_labels(analyze(synthetic.image, config).labels, synthetic.labels).f1 >= 0.95


@pytest.mark.parametrize("value", [0, 128, 255])
def test_empty_images_yield_no_cells(value: int) -> None:
    result = analyze(np.full((64, 64), value, dtype=np.uint8))
    assert result.cell_count == 0
    summary = result.summary()
    assert summary["cell_count"] == 0
    assert summary["area_px"] == {"mean": None, "std": None, "median": None, "cv": None}
    assert summary["coverage_fraction"] == 0


@pytest.mark.parametrize("engine", ["learned", "classical"])
def test_pure_noise_yields_no_cells(engine: str) -> None:
    rng = np.random.default_rng(3)
    noise = np.clip(rng.normal(20, 4, (256, 256)), 0, 255).astype(np.uint8)
    config = apply_overrides(AnalysisConfig(), [f"segmentation.engine={engine}"])
    assert analyze(noise, config).cell_count == 0


def test_cells_at_the_boundary(draw_discs) -> None:
    img = draw_discs((0, 64, 10), (64, 64, 10), (127, 20, 10))
    counted = analyze(img)
    assert counted.cell_count == 3
    assert sum(c.touches_border for c in counted.cells) == 2
    excluded = analyze(img, apply_overrides(AnalysisConfig(), ["filter.exclude_border=true"]))
    assert excluded.cell_count == 1
    assert excluded.cells[0].centroid_x == pytest.approx(64, abs=0.5)


def _dome_discs(*centres: tuple[int, int], radius: int = 14) -> np.ndarray:
    """Discs with a dome-shaped brightness profile, like real fluorescent cells."""
    ys, xs = np.mgrid[0:128, 0:128]
    img = np.full((128, 128), 10.0)
    for cx, cy in centres:
        r2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / radius**2
        img = np.maximum(img, np.where(r2 <= 1, 10 + 190 * (1 - 0.6 * r2), 0))
    return img.astype(np.uint8)


def test_overlapping_cells_are_split() -> None:
    img = _dome_discs((50, 64), (74, 64))
    assert analyze(img, CLASSICAL).cell_count == 2
    merged = analyze(img, apply_overrides(CLASSICAL, ["segmentation.split_method=none"]))
    assert merged.cell_count == 1


def test_flat_touching_cells_need_distance_splitting(draw_discs) -> None:
    # Uniformly bright cells have no intensity valley between them: only the shape
    # (distance-transform) watershed can separate them. This documents the trade-off.
    img = draw_discs((50, 64, 14), (74, 64, 14))
    assert analyze(img, CLASSICAL).cell_count == 1
    distance = apply_overrides(CLASSICAL, ["segmentation.split_method=distance"])
    assert analyze(img, distance).cell_count == 2


def test_results_are_deterministic(pill_image: SyntheticImage) -> None:
    first, second = analyze(pill_image.image), analyze(pill_image.image)
    np.testing.assert_array_equal(first.labels, second.labels)
    a, b = result_to_dict(first), result_to_dict(second)
    a.pop("created_at"), b.pop("created_at")
    assert a == b


def test_bit_depth_invariance(pill_image: SyntheticImage) -> None:
    eight = analyze(pill_image.image)
    sixteen = analyze(pill_image.image.astype(np.uint16) * 257)
    assert sixteen.full_scale == 65535
    assert sixteen.cell_count == eight.cell_count
    assert [c.brightness for c in sixteen.cells] == [c.brightness for c in eight.cells]
    np.testing.assert_allclose(
        [c.mean_intensity for c in sixteen.cells],
        [c.mean_intensity * 257 for c in eight.cells],
        rtol=1e-4,
    )


def test_greyscale_and_float_inputs(pill_image: SyntheticImage) -> None:
    grey = cv2.cvtColor(pill_image.image, cv2.COLOR_RGB2GRAY)
    assert analyze(grey).cell_count == pill_image.cell_count
    assert analyze(grey.astype(np.float32) / 255).cell_count == pill_image.cell_count


def test_labels_match_cell_ids(pill_image: SyntheticImage) -> None:
    result = analyze(pill_image.image)
    assert set(np.unique(result.labels)) - {0} == {c.cell_id for c in result.cells}
    for cell in result.cells[:5]:
        assert int((result.labels == cell.cell_id).sum()) == cell.area_px


def test_summary_statistics(pill_image: SyntheticImage) -> None:
    result = analyze(pill_image.image, AnalysisConfig(pixel_size_um=0.1))
    s = result.summary()
    areas = np.array([c.area_px for c in result.cells])
    assert s["area_px"]["mean"] == pytest.approx(areas.mean(), abs=1e-3)
    assert s["area_px"]["std"] == pytest.approx(areas.std(ddof=1), abs=1e-3)
    assert sum(s["brightness_counts"].values()) == result.cell_count
    assert s["cells_per_mm2"] == pytest.approx(result.cell_count / (512 * 512 * 1e-8), rel=1e-3)
    assert s["area_um2"]["mean"] == pytest.approx(areas.mean() * 0.01, rel=1e-3)


def test_single_cell_has_no_std(draw_discs) -> None:
    s = analyze(draw_discs((64, 64, 10)), CLASSICAL).summary()
    assert s["cell_count"] == 1
    assert s["area_px"]["std"] is None
    assert s["area_px"]["cv"] is None


def test_accepts_loaded_image_and_bytes(pill_image: SyntheticImage, write_png) -> None:
    path = write_png("cells.png", pill_image.image)
    from_path = analyze(path)
    from_loaded = analyze(load_image(path))
    from_bytes = analyze(path.read_bytes(), name="cells.png")
    assert from_path.cell_count == from_loaded.cell_count == from_bytes.cell_count
    assert from_bytes.metadata.source == "cells.png"


def test_real_sample_image(sample_path) -> None:
    result = analyze(sample_path)
    # The sample shows ~150-170 E. coli once packed clumps are separated; guard
    # against gross regressions only.
    assert 130 <= result.cell_count <= 200
    counts = result.summary()["brightness_counts"]
    assert all(v > 0 for v in counts.values()), "default cut-offs are calibrated on this image"
    areas = [c.area_px for c in result.cells]
    assert 100 < float(np.median(areas)) < 600


def test_config_is_recorded_in_result(pill_image: SyntheticImage) -> None:
    config = replace(AnalysisConfig(), pixel_size_um=0.2)
    assert analyze(pill_image.image, config).config is config
