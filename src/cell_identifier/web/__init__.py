"""Flask web interface for interactive analysis and synthetic image generation.

The web layer is a thin adapter over :func:`cell_identifier.analyze` and
:func:`cell_identifier.generate`:

* uploads are decoded **in memory** - nothing is written to disk, so there is no
  upload folder to clean up and no file-name handling to exploit;
* every request builds its own immutable configuration - there is no shared
  mutable analyser state between concurrent requests;
* invalid input returns HTTP 400 with a readable message; unexpected failures
  return HTTP 500 with a generic message and are logged server-side.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
from collections.abc import Mapping
from importlib import resources
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .. import __version__
from ..config import AnalysisConfig, ConfigError, SyntheticConfig
from ..evaluation import match_labels
from ..image_io import ImageValidationError, encode_png
from ..pipeline import AnalysisResult, analyze
from ..reporting import CELL_COLUMNS, result_to_dict
from ..synthetic import generate
from ..visualization import render_overlay, result_figure

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_SYNTHETIC_CELLS = 500
SAMPLE_IMAGE = "sample.jpeg"


class InvalidRequestError(ValueError):
    """Client error with a message that is safe to show to the user."""


def _data_url(image: Any) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(image)).decode("ascii")


def _get(params: Mapping[str, Any], key: str, cast: type, default: Any = None) -> Any:
    value = params.get(key, default)
    if value is None or value == "":
        return default
    try:
        if cast is bool:
            return (
                value
                if isinstance(value, bool)
                else str(value).lower() in {"1", "true", "on", "yes"}
            )
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(f"{key} must be a {cast.__name__}, got {value!r}") from exc


def analysis_config_from_params(params: Mapping[str, Any]) -> AnalysisConfig:
    """Translate web-form parameters into a validated :class:`AnalysisConfig`.

    Accepted keys (all optional): ``threshold_method``, ``manual_threshold``
    (0-255 scale), ``min_cell_area``, ``max_cell_area``, ``smoothing_sigma``
    (0 disables smoothing), ``split_method``, ``exclude_border``, ``channel``,
    ``classification``. The legacy keys ``intensity_threshold`` and
    ``blur_kernel`` of the original interface are still understood.
    """
    data: dict[str, dict[str, Any]] = {
        "preprocess": {},
        "segmentation": {},
        "filter": {},
        "classification": {},
    }
    method = _get(params, "threshold_method", str)
    manual = _get(params, "manual_threshold", float)
    legacy_threshold = _get(params, "intensity_threshold", float)
    if method is None and legacy_threshold is not None:
        method, manual = "fixed", legacy_threshold
    if method is not None:
        data["segmentation"]["threshold_method"] = method
    if manual is not None:
        data["segmentation"]["fixed_threshold"] = manual / 255.0
    sigma = _get(params, "smoothing_sigma", float)
    legacy_kernel = _get(params, "blur_kernel", int)
    if sigma is None and legacy_kernel is not None:
        sigma = 0.3 * ((legacy_kernel - 1) * 0.5 - 1) + 0.8  # OpenCV's ksize -> sigma rule
    if sigma is not None:
        if sigma <= 0:
            data["preprocess"]["denoise"] = "none"
        else:
            data["preprocess"]["denoise_sigma"] = sigma
    if (channel := _get(params, "channel", str)) is not None:
        data["preprocess"]["channel"] = channel
    if (split := _get(params, "split_method", str)) is not None:
        data["segmentation"]["split_method"] = split
    if (min_area := _get(params, "min_cell_area", int)) is not None:
        data["filter"]["min_area"] = min_area
    if (max_area := _get(params, "max_cell_area", int)) is not None:
        data["filter"]["max_area"] = max_area
    if (border := _get(params, "exclude_border", bool)) is not None:
        data["filter"]["exclude_border"] = border
    if (classification := _get(params, "classification", str)) is not None:
        data["classification"]["method"] = classification
    return AnalysisConfig.from_dict(data)


def _cells_csv(result: AnalysisResult) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CELL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for cell in result.cells:
        writer.writerow({k: "" if v is None else v for k, v in cell.to_dict().items()})
    return buffer.getvalue()


def _analysis_payload(result: AnalysisResult) -> dict[str, Any]:
    report = result_to_dict(result)
    return {
        "success": True,
        "result_image": _data_url(render_overlay(result.image, result.labels, result.cells)),
        "figure_image": _data_url(result_figure(result)),
        "cell_count": result.cell_count,
        "summary": report["summary"],
        "cells": report["cells"],
        "report": report,
        "csv": _cells_csv(result),
    }


def create_app(config: Mapping[str, Any] | None = None) -> Flask:
    """Application factory.

    Args:
        config: Optional Flask configuration overrides (e.g. ``{"TESTING": True}``).
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["JSON_SORT_KEYS"] = False
    if config:
        app.config.update(config)

    @app.errorhandler(InvalidRequestError)
    @app.errorhandler(ConfigError)
    @app.errorhandler(ImageValidationError)
    def _bad_request(exc: Exception) -> Any:
        return jsonify({"success": False, "error": str(exc)}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_: Exception) -> Any:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        return jsonify({"success": False, "error": f"file exceeds the {limit} MB limit"}), 413

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception) -> Any:
        if isinstance(exc, HTTPException):
            return exc
        logger.exception("unhandled error: %s", exc)
        return jsonify(
            {"success": False, "error": "internal error while processing the request"}
        ), 500

    @app.get("/")
    def index() -> str:
        return render_template("index.html", version=__version__)

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok", "version": __version__})

    @app.post("/api/analyze")
    @app.post("/analyze_image")  # legacy route used by the original interface
    def api_analyze() -> Any:
        if request.is_json:
            params = request.get_json(silent=True) or {}
            if not params.get("sample"):
                raise InvalidRequestError(
                    "JSON requests must set 'sample': true; upload files as multipart"
                )
            raw = (
                resources.files("cell_identifier.web").joinpath("static", SAMPLE_IMAGE).read_bytes()
            )
            name = SAMPLE_IMAGE
        else:
            params = request.form
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise InvalidRequestError("no file uploaded")
            raw = upload.read()
            name = upload.filename
        config = analysis_config_from_params(params)
        result = analyze(raw, config, name=name)
        return jsonify(_analysis_payload(result))

    @app.post("/api/generate")
    @app.post("/generate_synthetic")  # legacy route used by the original interface
    def api_generate() -> Any:
        params = request.get_json(silent=True)
        if not isinstance(params, dict):
            raise InvalidRequestError("expected a JSON body")
        num_cells = _get(params, "num_cells", int, 60)
        if not 1 <= num_cells <= MAX_SYNTHETIC_CELLS:
            raise InvalidRequestError(f"num_cells must be between 1 and {MAX_SYNTHETIC_CELLS}")
        settings: dict[str, Any] = {
            "num_cells": num_cells,
            "shape": _get(params, "cell_shape", str, "pill"),
            "allow_overlap": _get(params, "allow_overlap", bool, False),
            "noise_sigma": _get(params, "noise_level", float, 0.02),
            "seed": _get(params, "seed", int),
        }
        synthetic = generate(SyntheticConfig.from_dict(settings))
        payload: dict[str, Any] = {
            "success": True,
            "image": _data_url(synthetic.image),
            "cell_count": synthetic.cell_count,
            "requested": synthetic.requested,
            "cell_shape": synthetic.config.shape,
            "cell_data": [c.to_dict() for c in synthetic.cells],
            "seed": synthetic.config.seed,
        }
        if _get(params, "evaluate", bool, False):
            result = analyze(synthetic.image, AnalysisConfig(), name="synthetic")
            payload["evaluation"] = match_labels(result.labels, synthetic.labels).to_dict()
            payload["result_image"] = _data_url(
                render_overlay(result.image, result.labels, result.cells)
            )
        return jsonify(payload)

    return app
