from __future__ import annotations

import logging

import numpy as np
import pytest

from cell_identifier import SyntheticConfig, generate
from cell_identifier.config import ConfigError


def test_seed_makes_generation_reproducible() -> None:
    a, b = generate(seed=42, num_cells=30), generate(seed=42, num_cells=30)
    np.testing.assert_array_equal(a.image, b.image)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.cells == b.cells
    c = generate(seed=43, num_cells=30)
    assert not np.array_equal(a.image, c.image)


def test_output_layout(pill_image) -> None:
    assert pill_image.image.shape == (512, 512, 3)
    assert pill_image.image.dtype == np.uint8
    assert pill_image.labels.dtype == np.int32
    assert pill_image.cell_count == pill_image.requested == 40
    assert [c.cell_id for c in pill_image.cells] == list(range(1, 41))
    # GFP-like colour: green dominates in every cell.
    for cell in pill_image.cells[:5]:
        r, g, b = pill_image.image[round(cell.center_y), round(cell.center_x)]
        assert g > r
        assert g > b


@pytest.mark.parametrize("shape", ["pill", "ellipse"])
def test_forbidden_overlap_gives_disjoint_separated_cells(shape: str) -> None:
    synthetic = generate(seed=1, num_cells=50, shape=shape, min_separation=2)
    for cell in synthetic.cells:
        assert cell.visible_area_px == cell.area_px, "no cell may be covered by another"
    assert int((synthetic.labels > 0).sum()) == sum(c.area_px for c in synthetic.cells)


def test_allowed_overlap_respects_max_fraction() -> None:
    synthetic = generate(seed=2, num_cells=80, allow_overlap=True, max_overlap_fraction=0.3)
    assert any(c.visible_area_px < c.area_px for c in synthetic.cells), "expected some overlap"
    for cell in synthetic.cells:
        assert cell.visible_area_px >= 0


def test_cells_respect_margin_and_size_ranges() -> None:
    config = SyntheticConfig(
        seed=3, num_cells=40, margin=30, pill_length=(12, 20), pill_width=(5, 7)
    )
    synthetic = generate(config)
    for cell in synthetic.cells:
        assert 30 <= cell.center_x <= 512 - 1 - 30
        assert 30 <= cell.center_y <= 512 - 1 - 30
        assert 12 <= cell.length <= 20
        assert 5 <= cell.width <= 7
        assert 0 <= cell.angle_deg < 180


def test_footprint_area_matches_geometry() -> None:
    ellipse = generate(seed=4, num_cells=20, shape="ellipse", blur_sigma=0, noise_sigma=0)
    for cell in ellipse.cells:
        expected = np.pi * cell.length * cell.width / 4
        assert cell.area_px == pytest.approx(expected, rel=0.2)


def test_crowded_field_places_fewer_cells_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        synthetic = generate(
            SyntheticConfig(
                seed=0, width=64, height=64, num_cells=200, margin=5, max_attempts_per_cell=5
            )
        )
    assert synthetic.cell_count < synthetic.requested == 200
    assert "placed" in caplog.text


def test_zero_cells_gives_background_only() -> None:
    synthetic = generate(seed=0, num_cells=0, noise_sigma=0)
    assert synthetic.cell_count == 0
    assert synthetic.labels.max() == 0
    assert len(np.unique(synthetic.image[..., 1])) == 1


def test_illumination_gradient_darkens_right_side() -> None:
    synthetic = generate(
        seed=0, num_cells=0, noise_sigma=0, background_level=0.5, illumination_gradient=0.5
    )
    green = synthetic.image[..., 1].astype(float)
    assert green[:, :10].mean() > 1.8 * green[:, -10:].mean()


def test_keyword_overrides_and_validation() -> None:
    synthetic = generate(seed=0, num_cells=5, width=128, height=96, pill_length=(10, 12))
    assert synthetic.image.shape == (96, 128, 3)
    assert synthetic.config.pill_length == (10, 12)
    with pytest.raises(ConfigError):
        generate(num_cells=-1)
    with pytest.raises(ConfigError, match="unknown setting"):
        generate(colour=(1, 2, 3))


def test_ground_truth_dict_is_serialisable() -> None:
    import json

    gt = generate(seed=0, num_cells=3).ground_truth_dict()
    data = json.loads(json.dumps(gt))
    assert data["placed_cells"] == 3
    assert len(data["cells"]) == 3
    assert data["config"]["seed"] == 0
