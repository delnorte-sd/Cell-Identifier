"""Tests for the learned (U-Net) segmentation engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cell_identifier import AnalysisConfig, analyze, generate, match_labels
from cell_identifier.config import ConfigError, LearnedConfig, apply_overrides
from cell_identifier.learned import (
    TARGET_CELL_WIDTH,
    estimate_cell_width,
    load_model,
    make_targets,
    normalize_input,
    postprocess,
)

try:
    MODEL = load_model()
except FileNotFoundError:  # pragma: no cover - the model ships with the package
    MODEL = None
needs_model = pytest.mark.skipif(MODEL is None, reason="bundled model not available")
LEARNED = apply_overrides(AnalysisConfig(), ["segmentation.engine=learned"])


def _touching_rods() -> np.ndarray:
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10:50, 20:28] = 1
    labels[10:50, 28:36] = 2  # touches rod 1 along its whole length
    cv2.ellipse(labels, (48, 12), (8, 4), 30, 0, 360, 3, -1)
    return labels


def test_normalize_input() -> None:
    img = np.arange(1000, dtype=np.uint16).reshape(10, 100)
    out = normalize_input(img)
    assert out.dtype == np.float32
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert not normalize_input(np.full((8, 8), 7, dtype=np.uint16)).any()


def test_make_targets_channels() -> None:
    labels = _touching_rods()
    fg, boundary, dist = make_targets(labels)
    np.testing.assert_array_equal(fg, labels > 0)
    assert boundary[30, 27] == 1, "seam between touching cells"
    assert boundary[30, 28] == 1, "seam between touching cells"
    assert boundary[30, 24] == 0, "cell interior"
    for cell in (1, 2, 3):
        assert dist[labels == cell].max() == pytest.approx(1.0)
    assert not dist[labels == 0].any()


def test_postprocess_inverts_targets_exactly() -> None:
    labels = _touching_rods()
    recovered = postprocess(make_targets(labels))
    assert match_labels(recovered, labels).f1 == 1.0
    assert match_labels(recovered, labels).mean_iou > 0.95


def test_postprocess_empty_and_seedless() -> None:
    assert postprocess(np.zeros((3, 16, 16), dtype=np.float32)).max() == 0
    probs = np.zeros((3, 16, 16), dtype=np.float32)
    probs[0, 4:7, 4:7] = 0.9  # foreground but no seed: still becomes one object
    assert postprocess(probs).max() == 1


def test_learned_config_validation() -> None:
    assert LearnedConfig().cell_width_px == 0
    with pytest.raises(ConfigError, match="cell_width_px"):
        LearnedConfig(cell_width_px=1)
    with pytest.raises(ConfigError, match="tile_size"):
        LearnedConfig(tile_size=100)
    with pytest.raises(ConfigError, match="engine"):
        apply_overrides(AnalysisConfig(), ["segmentation.engine=magic"])


def test_missing_model_raises() -> None:
    config = apply_overrides(LEARNED, ['learned.model="does/not/exist.onnx"'])
    with pytest.raises(FileNotFoundError, match="not found"):
        analyze(np.zeros((32, 32), dtype=np.uint8), config)


@needs_model
def test_model_output_shape_and_range() -> None:
    probs = MODEL.predict(np.random.default_rng(0).random((100, 140), dtype=np.float32))
    assert probs.shape == (3, 100, 140)
    assert probs.min() >= 0
    assert probs.max() <= 1


@needs_model
def test_tiled_prediction_matches_whole_image() -> None:
    synthetic = generate(seed=2, num_cells=40, width=600, height=600)
    x = normalize_input(cv2.cvtColor(synthetic.image, cv2.COLOR_RGB2GRAY))
    whole = MODEL.predict(x, tile=1024)
    tiled = MODEL.predict(x, tile=256, overlap=48)
    assert np.corrcoef(whole[0].ravel(), tiled[0].ravel())[0, 1] > 0.98


@needs_model
@pytest.mark.parametrize("shape", ["pill", "ellipse"])
def test_learned_engine_on_synthetic_ground_truth(shape: str) -> None:
    synthetic = generate(seed=4, num_cells=50, shape=shape)
    result = analyze(synthetic.image, LEARNED)
    metrics = match_labels(result.labels, synthetic.labels)
    assert metrics.f1 >= 0.9, metrics
    assert result.summary()["engine"] == "learned"


@needs_model
def test_width_estimate_tracks_cell_size() -> None:
    # Absolute accuracy is validated on real annotated images (docs/TRAINING.md); here we
    # check that the estimate follows the image scale (optical blur makes synthetic cells
    # look wider than their nominal width, so absolute values are not compared).
    synthetic = generate(seed=5, num_cells=60)
    grey = cv2.cvtColor(synthetic.image, cv2.COLOR_RGB2GRAY)
    estimates = {}
    for scale in (1.0, 1.6, 2.5):
        img = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        estimates[scale] = estimate_cell_width(MODEL, normalize_input(img))
    assert estimates[1.6] / estimates[1.0] == pytest.approx(1.6, rel=0.25), estimates
    assert estimates[2.5] / estimates[1.0] == pytest.approx(2.5, rel=0.25), estimates


@needs_model
def test_learned_engine_handles_empty_and_16bit_images(sample_path) -> None:
    assert analyze(np.zeros((64, 64), dtype=np.uint8), LEARNED).cell_count == 0
    eight = analyze(sample_path, LEARNED)
    grey = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2GRAY)
    sixteen = analyze(grey.astype(np.uint16) * 257, LEARNED)
    assert abs(sixteen.cell_count - eight.cell_count) <= 0.05 * eight.cell_count
    assert TARGET_CELL_WIDTH == 10.0
