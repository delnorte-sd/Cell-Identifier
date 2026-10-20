from __future__ import annotations

import json
from pathlib import Path

import pytest

from cell_identifier.config import (
    AnalysisConfig,
    ConfigError,
    FilterConfig,
    PreprocessConfig,
    SegmentationConfig,
    SyntheticConfig,
    apply_overrides,
    load_config_file,
    parse_override,
)


def test_defaults_are_valid_and_round_trip() -> None:
    config = AnalysisConfig()
    assert AnalysisConfig.from_dict(config.to_dict()) == config
    synthetic = SyntheticConfig()
    assert SyntheticConfig.from_dict(synthetic.to_dict()) == synthetic


def test_configs_are_immutable() -> None:
    config = AnalysisConfig()
    with pytest.raises(AttributeError):
        config.pixel_size_um = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PreprocessConfig(channel="purple"), "channel"),  # type: ignore[arg-type]
        (lambda: PreprocessConfig(denoise_sigma=0), "denoise_sigma"),
        (lambda: PreprocessConfig(median_size=4), "median_size"),
        (lambda: PreprocessConfig(percentile_low=50, percentile_high=10), "percentiles"),
        (lambda: SegmentationConfig(fixed_threshold=1.5), "fixed_threshold"),
        (lambda: SegmentationConfig(adaptive_block_size=10), "adaptive_block_size"),
        (lambda: SegmentationConfig(min_snr=-1), "min_snr"),
        (lambda: SegmentationConfig(split_method="magic"), "split_method"),  # type: ignore[arg-type]
        (lambda: FilterConfig(min_area=0), "min_area"),
        (lambda: FilterConfig(min_area=100, max_area=10), "max_area"),
        (lambda: AnalysisConfig(pixel_size_um=0), "pixel_size_um"),
        (lambda: SyntheticConfig(width=4), "16x16"),
        (lambda: SyntheticConfig(pill_length=(10, 5)), "pill_length"),
        (lambda: SyntheticConfig(margin=300), "margin"),
        (lambda: SyntheticConfig(color=(0, 0, 300)), "color"),
    ],
)
def test_invalid_values_raise_config_error(factory: object, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        factory()  # type: ignore[operator]


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigError, match="unknown setting"):
        AnalysisConfig.from_dict({"segmentation": {"treshold_method": "otsu"}})
    with pytest.raises(ConfigError, match="unknown setting"):
        AnalysisConfig.from_dict({"colour": "green"})


def test_from_dict_coerces_and_rejects_types() -> None:
    config = AnalysisConfig.from_dict(
        {"filter": {"min_area": "20", "exclude_border": "true"}, "pixel_size_um": 1}
    )
    assert config.filter.min_area == 20
    assert config.filter.exclude_border is True
    assert config.pixel_size_um == 1.0
    with pytest.raises(ConfigError, match="integer"):
        AnalysisConfig.from_dict({"filter": {"min_area": "lots"}})
    with pytest.raises(ConfigError, match="boolean"):
        AnalysisConfig.from_dict({"filter": {"exclude_border": "maybe"}})
    with pytest.raises(ConfigError, match="mapping"):
        AnalysisConfig.from_dict({"filter": 3})


def test_parse_override() -> None:
    assert parse_override("segmentation.min_snr=2.5") == (["segmentation", "min_snr"], 2.5)
    assert parse_override("preprocess.channel=green") == (["preprocess", "channel"], "green")
    assert parse_override("seed=null") == (["seed"], None)
    with pytest.raises(ConfigError):
        parse_override("no-equals-sign")


def test_apply_overrides() -> None:
    config = apply_overrides(
        AnalysisConfig(),
        ["segmentation.threshold_method=triangle", "filter.min_area=30", "pixel_size_um=0.1"],
    )
    assert config.segmentation.threshold_method == "triangle"
    assert config.filter.min_area == 30
    assert config.pixel_size_um == 0.1
    with pytest.raises(ConfigError, match="unknown setting"):
        apply_overrides(AnalysisConfig(), ["filter.nope=1"])
    with pytest.raises(ConfigError, match="unknown config section"):
        apply_overrides(AnalysisConfig(), ["nope.x=1"])
    synthetic = apply_overrides(SyntheticConfig(), ["pill_length=[5, 9]", "seed=3"])
    assert synthetic.pill_length == (5, 9)
    assert synthetic.seed == 3


def test_load_toml_and_json(tmp_path: Path) -> None:
    toml_file = tmp_path / "analysis.toml"
    toml_file.write_text(
        'pixel_size_um = 0.065\n[segmentation]\nthreshold_method = "adaptive"\n', encoding="utf-8"
    )
    config = load_config_file(toml_file, AnalysisConfig)
    assert config.pixel_size_um == 0.065
    assert config.segmentation.threshold_method == "adaptive"

    json_file = tmp_path / "synthetic.json"
    json_file.write_text(json.dumps({"num_cells": 12, "shape": "ellipse"}), encoding="utf-8")
    synthetic = load_config_file(json_file, SyntheticConfig)
    assert (synthetic.num_cells, synthetic.shape) == (12, "ellipse")


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config_file(tmp_path / "missing.toml", AnalysisConfig)
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config_file(bad, AnalysisConfig)
    yaml = tmp_path / "config.yaml"
    yaml.write_text("a: 1", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"\.toml or \.json"):
        load_config_file(yaml, AnalysisConfig)


def test_example_config_file_loads() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "analysis.toml"
    config = load_config_file(example, AnalysisConfig)
    assert config.segmentation.split_method == "intensity"
    assert config.classification.low_cutoff == pytest.approx(142 / 255, abs=1e-4)
