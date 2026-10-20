"""Typed, validated configuration for the analysis pipeline and the synthetic generator.

Every tunable number used by the pipeline lives here, with its unit and meaning
documented, instead of being scattered through the code. Configurations are
plain frozen dataclasses so they are hashable, cheap to copy and trivially
serialisable into result files for provenance.

Intensity conventions
---------------------
Thresholds and cut-offs that refer to image intensity are expressed as a
*fraction of the image's full scale* (``0.0``-``1.0``): 255 for 8-bit images,
65535 for 16-bit images and 1.0 for floating-point images. This keeps a single
configuration valid across bit depths. ``0.2`` therefore means "51 grey levels"
in an 8-bit image and "13107" in a 16-bit image.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args

__all__ = [
    "AnalysisConfig",
    "ClassificationConfig",
    "ConfigError",
    "FilterConfig",
    "PreprocessConfig",
    "SegmentationConfig",
    "SyntheticConfig",
    "apply_overrides",
    "load_config_file",
    "parse_override",
]

ChannelMode = Literal["gray", "red", "green", "blue", "max"]
DenoiseMethod = Literal["none", "gaussian", "median"]
BackgroundMethod = Literal["none", "tophat", "gaussian"]
NormalizeMethod = Literal["none", "percentile"]
ThresholdMethod = Literal["otsu", "triangle", "fixed", "adaptive"]
SplitMethod = Literal["none", "distance", "intensity"]
ClassificationMethod = Literal["none", "absolute", "quantile"]
CellShape = Literal["pill", "ellipse"]


class ConfigError(ValueError):
    """Raised when a configuration value is missing, malformed or out of range."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_choice(name: str, value: object, literal: object) -> None:
    choices = get_args(literal)
    _check(value in choices, f"{name} must be one of {list(choices)}, got {value!r}")


def _check_fraction(name: str, value: float) -> None:
    _check(0.0 <= value <= 1.0, f"{name} must be within [0, 1], got {value}")


@dataclass(frozen=True)
class PreprocessConfig:
    """How raw pixels are turned into a single, clean intensity image for segmentation.

    Attributes:
        channel: Which signal to analyse. ``"gray"`` is the luminance-weighted mix
            (0.299 R + 0.587 G + 0.114 B); ``"green"`` isolates GFP-like signal;
            ``"max"`` takes the per-pixel maximum over channels. Ignored for
            single-channel images.
        denoise: Noise filter applied before thresholding.
        denoise_sigma: Gaussian standard deviation in pixels (``denoise="gaussian"``).
        median_size: Median window width in pixels, odd (``denoise="median"``).
        background: Illumination/background correction. ``"tophat"`` subtracts a
            morphological opening (rolling-ball-like); ``"gaussian"`` subtracts a
            heavily blurred copy. Correction only affects *segmentation*;
            intensity measurements always use the raw signal.
        background_radius: Structuring-element radius / blur sigma in pixels. Must
            exceed the radius of the largest cell or cells will be eroded.
        normalize: Optional contrast stretch applied after background correction.
        percentile_low: Lower percentile mapped to 0 when ``normalize="percentile"``.
        percentile_high: Upper percentile mapped to 1 when ``normalize="percentile"``.
    """

    channel: ChannelMode = "gray"
    denoise: DenoiseMethod = "gaussian"
    denoise_sigma: float = 1.0
    median_size: int = 3
    background: BackgroundMethod = "none"
    background_radius: float = 25.0
    normalize: NormalizeMethod = "none"
    percentile_low: float = 1.0
    percentile_high: float = 99.8

    def __post_init__(self) -> None:
        _check_choice("preprocess.channel", self.channel, ChannelMode)
        _check_choice("preprocess.denoise", self.denoise, DenoiseMethod)
        _check_choice("preprocess.background", self.background, BackgroundMethod)
        _check_choice("preprocess.normalize", self.normalize, NormalizeMethod)
        _check(self.denoise_sigma > 0, "preprocess.denoise_sigma must be > 0")
        _check(
            self.median_size >= 3 and self.median_size % 2 == 1,
            "preprocess.median_size must be an odd integer >= 3",
        )
        _check(self.background_radius >= 1, "preprocess.background_radius must be >= 1 px")
        _check(
            0.0 <= self.percentile_low < self.percentile_high <= 100.0,
            "preprocess percentiles must satisfy 0 <= low < high <= 100",
        )


@dataclass(frozen=True)
class SegmentationConfig:
    """Foreground detection and splitting of touching objects.

    Attributes:
        threshold_method: ``"otsu"`` and ``"triangle"`` pick a global threshold
            automatically from the histogram (triangle suits sparse, dim cells);
            ``"fixed"`` uses ``fixed_threshold``; ``"adaptive"`` compares each pixel
            with its local mean, which tolerates uneven illumination.
        fixed_threshold: Global threshold as a fraction of full scale.
        adaptive_block_size: Local window width in pixels (odd) for ``"adaptive"``.
        adaptive_offset: Added to the local mean, as a fraction of full scale.
        min_snr: Noise floor. Whatever the method, the threshold is never allowed
            below ``median + min_snr * sigma`` of the image, where sigma is a robust
            (MAD-based) noise estimate. Prevents an automatic threshold from
            "finding" cells in pure noise. ``0`` disables the floor. Note that the
            floor assumes cells cover less than half of the field of view.
        opening_radius: Radius in pixels of the morphological opening that removes
            single-pixel speckle. ``0`` disables. Keep this smaller than half of the
            narrowest cell width, or thin cells disappear.
        closing_radius: Radius in pixels of a morphological closing that bridges
            small gaps inside cells. ``0`` disables.
        fill_holes: Fill enclosed holes in objects (e.g. dim cell centres).
        split_method: How touching cells are separated by the marker-controlled
            watershed. ``"distance"`` cuts at geometric necks of the mask (ideal
            for round or end-to-end touching cells); ``"intensity"`` cuts along
            dark valleys in the signal (needed for rods lying side by side, which
            have no neck); ``"none"`` keeps connected components whole.
        watershed_h: ``"distance"`` mode: minimum prominence, in pixels of
            distance-transform height, for a local maximum to seed its own cell
            (h-maxima transform). Larger values merge more.
        intensity_h: ``"intensity"`` mode: minimum prominence of an intensity
            peak (fraction of full scale) to seed its own cell. Raise it if
            uniformly bright cells are cut in pieces; lower it to split more.
    """

    threshold_method: ThresholdMethod = "otsu"
    fixed_threshold: float = 0.2
    adaptive_block_size: int = 51
    adaptive_offset: float = 0.02
    min_snr: float = 3.0
    opening_radius: int = 1
    closing_radius: int = 0
    fill_holes: bool = True
    split_method: SplitMethod = "intensity"
    watershed_h: float = 1.0
    intensity_h: float = 0.05

    def __post_init__(self) -> None:
        _check_choice("segmentation.threshold_method", self.threshold_method, ThresholdMethod)
        _check_fraction("segmentation.fixed_threshold", self.fixed_threshold)
        _check(
            self.adaptive_block_size >= 3 and self.adaptive_block_size % 2 == 1,
            "segmentation.adaptive_block_size must be an odd integer >= 3",
        )
        _check(
            -1.0 <= self.adaptive_offset <= 1.0,
            "segmentation.adaptive_offset must be within [-1, 1]",
        )
        _check(self.min_snr >= 0, "segmentation.min_snr must be >= 0")
        _check(self.opening_radius >= 0, "segmentation.opening_radius must be >= 0")
        _check(self.closing_radius >= 0, "segmentation.closing_radius must be >= 0")
        _check_choice("segmentation.split_method", self.split_method, SplitMethod)
        _check(self.watershed_h > 0, "segmentation.watershed_h must be > 0")
        _check(0 < self.intensity_h <= 1, "segmentation.intensity_h must be within (0, 1]")


@dataclass(frozen=True)
class FilterConfig:
    """Rules for accepting a segmented object as a cell.

    Attributes:
        min_area: Smallest accepted object, in pixels (inclusive).
        max_area: Largest accepted object, in pixels (inclusive). Larger objects
            are usually clumps or debris.
        exclude_border: Drop objects touching the image edge. Recommended for
            size/shape statistics because cut-off cells bias them low; leave off
            when you need a total count.
    """

    min_area: int = 15
    max_area: int = 5000
    exclude_border: bool = False

    def __post_init__(self) -> None:
        _check(self.min_area >= 1, "filter.min_area must be >= 1 px")
        _check(
            self.max_area >= self.min_area,
            f"filter.max_area ({self.max_area}) must be >= filter.min_area ({self.min_area})",
        )


@dataclass(frozen=True)
class ClassificationConfig:
    """Brightness classification of cells by their raw mean intensity.

    Attributes:
        method: ``"absolute"`` uses fixed cut-offs and is comparable across images
            acquired with identical settings; ``"quantile"`` splits each image's
            cells into tertiles and is only meaningful within one image;
            ``"none"`` skips classification.
        low_cutoff: Mean intensity (fraction of full scale) below which a cell is
            "Low". The default (142/255) is the lower tertile of cell intensities
            in the bundled E. coli GFP sample analysed with the default settings;
            it must be re-calibrated for other strains, exposures or microscopes.
        high_cutoff: Mean intensity at or above which a cell is "High"; values in
            between are "Medium". Default 156/255 (upper tertile of the sample).
    """

    method: ClassificationMethod = "absolute"
    low_cutoff: float = 142 / 255
    high_cutoff: float = 156 / 255

    def __post_init__(self) -> None:
        _check_choice("classification.method", self.method, ClassificationMethod)
        _check_fraction("classification.low_cutoff", self.low_cutoff)
        _check_fraction("classification.high_cutoff", self.high_cutoff)
        _check(
            self.low_cutoff <= self.high_cutoff,
            "classification.low_cutoff must be <= classification.high_cutoff",
        )


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete configuration of the analysis pipeline.

    Attributes:
        preprocess: Channel selection, denoising and background correction.
        segmentation: Thresholding, morphology and watershed splitting.
        filter: Object acceptance rules.
        classification: Brightness classes.
        pixel_size_um: Physical pixel size in micrometres. When set, areas and
            lengths are also reported in µm / µm² and densities per mm².
    """

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    pixel_size_um: float | None = None

    def __post_init__(self) -> None:
        _check(
            self.pixel_size_um is None or self.pixel_size_um > 0,
            "pixel_size_um must be > 0 when given",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all settings."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalysisConfig:
        """Build a configuration from a (possibly partial) nested mapping.

        Missing keys keep their defaults; unknown keys raise :class:`ConfigError`
        so typos never silently fall back to a default.
        """
        return _from_mapping(cls, data, "")


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic fluorescence image generator.

    Synthetic images exist to validate the analysis pipeline against a known
    ground truth; they are never mixed with experimental data.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        num_cells: Number of cells to place. Fewer may be placed if the field is
            too crowded for the overlap rules; the result reports both numbers.
        shape: ``"pill"`` (rod-shaped, E. coli-like) or ``"ellipse"``.
        pill_length: Inclusive (min, max) end-to-end rod length in pixels.
        pill_width: Inclusive (min, max) rod width in pixels.
        ellipse_major: Inclusive (min, max) major-axis length in pixels.
        ellipse_minor_ratio: (min, max) minor/major axis ratio.
        intensity: (min, max) peak cell brightness as a fraction of full scale.
        edge_falloff: Fractional brightness drop from the cell centre line to its
            edge, emulating the thinner optical path at the cell periphery.
        background_level: Constant background fluorescence (fraction of full scale).
        illumination_gradient: Linear left-to-right illumination fall-off,
            fraction lost at the right edge (vignetting / uneven field).
        blur_sigma: Optical point-spread function, Gaussian sigma in pixels.
        noise_sigma: Additive Gaussian detector noise, fraction of full scale.
        allow_overlap: Whether cells may overlap.
        max_overlap_fraction: When overlap is allowed, the largest fraction of a
            new cell's area that may cover existing cells.
        min_separation: When overlap is forbidden, minimum gap in pixels between
            cell boundaries.
        margin: Minimum distance in pixels from cell centre to the image edge.
        max_attempts_per_cell: Placement attempts per requested cell.
        color: RGB colour of the fluorophore, 0-255 per channel (GFP-like default).
        seed: Random seed. ``None`` draws fresh entropy; set it for reproducibility.
    """

    width: int = 512
    height: int = 512
    num_cells: int = 60
    shape: CellShape = "pill"
    pill_length: tuple[int, int] = (12, 24)
    pill_width: tuple[int, int] = (5, 8)
    ellipse_major: tuple[int, int] = (10, 25)
    ellipse_minor_ratio: tuple[float, float] = (0.5, 0.8)
    intensity: tuple[float, float] = (0.6, 1.0)
    edge_falloff: float = 0.3
    background_level: float = 0.05
    illumination_gradient: float = 0.0
    blur_sigma: float = 1.0
    noise_sigma: float = 0.02
    allow_overlap: bool = False
    max_overlap_fraction: float = 0.3
    min_separation: int = 2
    margin: int = 15
    max_attempts_per_cell: int = 50
    color: tuple[int, int, int] = (26, 251, 115)
    seed: int | None = None

    def __post_init__(self) -> None:
        _check(self.width >= 16 and self.height >= 16, "synthetic image must be >= 16x16 px")
        _check(self.width * self.height <= 64_000_000, "synthetic image is too large")
        _check(0 <= self.num_cells <= 10_000, "synthetic.num_cells must be in [0, 10000]")
        _check_choice("synthetic.shape", self.shape, CellShape)
        for name in ("pill_length", "pill_width", "ellipse_major"):
            lo, hi = getattr(self, name)
            _check(1 <= lo <= hi, f"synthetic.{name} must satisfy 1 <= min <= max")
        lo_r, hi_r = self.ellipse_minor_ratio
        _check(0 < lo_r <= hi_r <= 1, "synthetic.ellipse_minor_ratio must be in (0, 1]")
        lo_i, hi_i = self.intensity
        _check(0 < lo_i <= hi_i <= 1, "synthetic.intensity must satisfy 0 < min <= max <= 1")
        _check_fraction("synthetic.edge_falloff", self.edge_falloff)
        _check_fraction("synthetic.background_level", self.background_level)
        _check_fraction("synthetic.illumination_gradient", self.illumination_gradient)
        _check(self.blur_sigma >= 0, "synthetic.blur_sigma must be >= 0")
        _check(self.noise_sigma >= 0, "synthetic.noise_sigma must be >= 0")
        _check_fraction("synthetic.max_overlap_fraction", self.max_overlap_fraction)
        _check(self.min_separation >= 0, "synthetic.min_separation must be >= 0")
        _check(
            0 <= self.margin < min(self.width, self.height) // 2,
            "synthetic.margin must be >= 0 and less than half the smaller image side",
        )
        _check(self.max_attempts_per_cell >= 1, "synthetic.max_attempts_per_cell must be >= 1")
        _check(
            len(self.color) == 3 and all(0 <= c <= 255 for c in self.color),
            "synthetic.color must be three integers in [0, 255]",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all settings."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SyntheticConfig:
        """Build a configuration from a (possibly partial) mapping; see AnalysisConfig."""
        return _from_mapping(cls, data, "")


_T = TypeVar("_T")


def _coerce(value: Any, default: Any, path: str) -> Any:
    """Coerce a JSON/TOML scalar to the type of the field's default value."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false", "1", "0", "yes", "no"}:
            return value.lower() in {"true", "1", "yes"}
        raise ConfigError(f"{path} must be a boolean, got {value!r}")
    if isinstance(default, tuple):
        if not isinstance(value, list | tuple) or len(value) != len(default):
            raise ConfigError(f"{path} must be a list of {len(default)} numbers, got {value!r}")
        return tuple(_coerce(v, d, path) for v, d in zip(value, default, strict=True))
    if isinstance(default, int) and not isinstance(value, bool):
        if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigError(f"{path} must be an integer, got {value!r}")
    if isinstance(default, float):
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigError(f"{path} must be a number, got {value!r}")
    return value


def _from_mapping(cls: type[_T], data: Mapping[str, Any], prefix: str) -> _T:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{prefix or 'config'} must be a mapping, got {type(data).__name__}")
    defaults = cls()
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(known))
    if unknown:
        where = prefix.rstrip(".") or cls.__name__
        raise ConfigError(f"unknown setting(s) in {where}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = getattr(defaults, name)
        path = f"{prefix}{name}"
        if is_dataclass(default):
            kwargs[name] = _from_mapping(type(default), value, f"{path}.")
        elif name == "seed" or name == "pixel_size_um":
            if value is None:
                kwargs[name] = None
            else:
                kwargs[name] = _coerce(value, 0 if name == "seed" else 0.0, path)
        else:
            kwargs[name] = _coerce(value, default, path)
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - guarded by the unknown-key check
        raise ConfigError(str(exc)) from exc


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse a ``section.key=value`` override string.

    The value is decoded as JSON when possible (so ``true``, ``3``, ``[1, 2]`` and
    ``null`` work), otherwise kept as a plain string.
    """
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"override must look like 'section.key=value', got {text!r}")
    raw = raw.strip()
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.split("."), value


def apply_overrides(config: _T, overrides: Iterable[str]) -> _T:
    """Return a copy of ``config`` with ``section.key=value`` overrides applied."""
    data = asdict(config)  # type: ignore[call-overload]
    for text in overrides:
        path, value = parse_override(text)
        node = data
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                raise ConfigError(f"unknown config section {'.'.join(path[:-1])!r}")
            node = node[part]
        if path[-1] not in node:
            raise ConfigError(f"unknown setting {'.'.join(path)!r}")
        node[path[-1]] = value
    return _from_mapping(type(config), data, "")


def load_config_file(path: str | Path, cls: type[_T]) -> _T:
    """Load a configuration from a ``.toml`` or ``.json`` file.

    For :class:`AnalysisConfig` a TOML file looks like::

        pixel_size_um = 0.1

        [segmentation]
        threshold_method = "triangle"

        [filter]
        min_area = 20
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".toml":
            data: Any = tomllib.loads(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ConfigError(f"config file must be .toml or .json, got {path.suffix!r}")
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    return _from_mapping(cls, data, "")
