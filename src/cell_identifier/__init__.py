"""Cell Identifier: reproducible cell detection and measurement for fluorescence microscopy.

Developed by iGEM Team DelNorte-SD.

Quick start::

    from cell_identifier import AnalysisConfig, analyze

    result = analyze("cells.png", AnalysisConfig())
    print(result.cell_count, result.summary()["area_px"])

Subpackages and modules:

* :mod:`~cell_identifier.config` - validated configuration dataclasses
* :mod:`~cell_identifier.image_io` - loading, validation and saving
* :mod:`~cell_identifier.preprocessing` - channel, denoise, background correction
* :mod:`~cell_identifier.segmentation` - thresholding and watershed splitting
* :mod:`~cell_identifier.measurement` - per-cell shape and intensity metrics
* :mod:`~cell_identifier.classification` - brightness classes
* :mod:`~cell_identifier.pipeline` - :func:`analyze`, the end-to-end entry point
* :mod:`~cell_identifier.visualization` - overlays and comparison figures
* :mod:`~cell_identifier.reporting` - JSON / CSV export
* :mod:`~cell_identifier.synthetic` - ground-truth synthetic images (validation only)
* :mod:`~cell_identifier.evaluation` - precision / recall / F1 against ground truth
"""

__version__ = "1.0.0"

from .config import (
    AnalysisConfig,
    ClassificationConfig,
    ConfigError,
    FilterConfig,
    PreprocessConfig,
    SegmentationConfig,
    SyntheticConfig,
)
from .evaluation import MatchResult, match_labels
from .image_io import ImageValidationError, load_image
from .measurement import CellMeasurement
from .pipeline import AnalysisResult, analyze
from .synthetic import SyntheticImage, generate

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "CellMeasurement",
    "ClassificationConfig",
    "ConfigError",
    "FilterConfig",
    "ImageValidationError",
    "MatchResult",
    "PreprocessConfig",
    "SegmentationConfig",
    "SyntheticConfig",
    "SyntheticImage",
    "__version__",
    "analyze",
    "generate",
    "load_image",
    "match_labels",
]
