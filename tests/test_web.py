from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest

pytest.importorskip("flask")

from cell_identifier import generate, load_image
from cell_identifier.image_io import encode_png
from cell_identifier.web import analysis_config_from_params, create_app


@pytest.fixture
def client():
    return create_app({"TESTING": True}).test_client()


def test_index_and_health(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    html = page.data.decode()
    for landmark in ('id="canvas"', 'id="settings"', 'id="cellTable"', 'aria-live="polite"'):
        assert landmark in html
    assert "font-awesome" not in html, "no third-party CDN dependencies"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/healthz").get_json()["status"] == "ok"
    assert client.get("/missing").status_code == 404


def test_analyze_sample(client) -> None:
    response = client.post("/api/analyze", json={"sample": True})
    data = response.get_json()
    assert response.status_code == 200
    assert data["success"]
    assert data["cell_count"] == len(data["cells"]) == data["summary"]["cell_count"]
    assert data["result_image"].startswith("data:image/png;base64,")
    assert data["csv"].splitlines()[0].startswith("cell_id,")
    assert data["report"]["input"]["source"] == "sample.jpeg"
    assert data["image_info"] == {
        "name": "sample.jpeg",
        "width": 600,
        "height": 456,
        "channels": 3,
        "dtype": "uint8",
    }


def _decode(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return load_image(raw).data


def test_analysis_layers_are_separable_and_exact(client) -> None:
    synthetic = generate(seed=3, num_cells=25)
    form = {"file": (io.BytesIO(encode_png(synthetic.image)), "cells.png")}
    data = client.post("/api/analyze", data=form, content_type="multipart/form-data").get_json()
    layers = data["layers"]
    assert set(layers) == {"original", "processed", "mask", "outlines", "ids"}
    original = _decode(layers["original"])
    np.testing.assert_array_equal(original, synthetic.image)
    assert _decode(layers["processed"]).shape == (512, 512)
    # Overlays are transparent RGBA: opaque only on cells / outlines.
    mask = _decode_rgba(layers["mask"])
    outlines = _decode_rgba(layers["outlines"])
    ids = _decode(layers["ids"]).astype(np.int64)
    labels = ids[..., 0] + 256 * ids[..., 1] + 65536 * ids[..., 2]
    assert labels.max() == data["cell_count"] == 25
    np.testing.assert_array_equal(mask[..., 3] > 0, labels > 0)
    assert (outlines[..., 3] > 0).sum() < (labels > 0).sum()
    assert not (outlines[..., 3][labels == 0]).any()
    cell = data["cells"][0]
    assert (labels == cell["cell_id"]).sum() == cell["area_px"]


def _decode_rgba(data_url: str) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), dtype=np.uint8)
    bgra = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    assert bgra.shape[2] == 4
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


def test_analyze_upload_with_parameters(client) -> None:
    synthetic = generate(seed=9, num_cells=20)
    form = {
        "file": (io.BytesIO(encode_png(synthetic.image)), "cells.png"),
        "threshold_method": "otsu",
        "split_method": "distance",
        "min_cell_area": "10",
        "exclude_border": "false",
    }
    data = client.post("/api/analyze", data=form, content_type="multipart/form-data").get_json()
    assert data["success"]
    assert data["cell_count"] == 20
    assert data["report"]["config"]["segmentation"]["split_method"] == "distance"


def test_legacy_route_and_parameters(client) -> None:
    data = client.post(
        "/analyze_image", json={"sample": True, "intensity_threshold": 60, "blur_kernel": 5}
    ).get_json()
    assert data["success"]
    seg = data["report"]["config"]["segmentation"]
    assert seg["threshold_method"] == "fixed"
    assert seg["fixed_threshold"] == pytest.approx(60 / 255)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"data": {}, "content_type": "multipart/form-data"}, "no file"),
        (
            {
                "data": {"file": (io.BytesIO(b"junk"), "../../etc/passwd.png")},
                "content_type": "multipart/form-data",
            },
            "could not decode",
        ),
        ({"json": {"sample": False}}, "multipart"),
        ({"json": {"sample": True, "min_cell_area": 50, "max_cell_area": 5}}, "max_area"),
        ({"json": {"sample": True, "threshold_method": "guess"}}, "threshold_method"),
        ({"json": {"sample": True, "min_cell_area": "many"}}, "must be a int"),
    ],
)
def test_analyze_rejects_bad_requests(client, kwargs, message: str) -> None:
    response = client.post("/api/analyze", **kwargs)
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert message in body["error"]


def test_uploads_are_never_written_to_disk(client, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    form = {"file": (io.BytesIO(encode_png(generate(seed=1, num_cells=3).image)), "../escape.png")}
    assert (
        client.post("/api/analyze", data=form, content_type="multipart/form-data").status_code
        == 200
    )
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path.parent / "escape.png").exists()


def test_upload_size_limit() -> None:
    app = create_app({"TESTING": True, "MAX_CONTENT_LENGTH": 1024})
    form = {"file": (io.BytesIO(b"0" * 4096), "big.png")}
    response = app.test_client().post("/api/analyze", data=form, content_type="multipart/form-data")
    assert response.status_code == 413


def test_generate_with_evaluation(client) -> None:
    data = client.post(
        "/api/generate", json={"num_cells": 30, "seed": 4, "evaluate": True}
    ).get_json()
    assert data["success"]
    assert data["cell_count"] == 30
    assert data["seed"] == 4
    assert data["evaluation"]["f1"] == 1.0
    assert "truth" in data["layers"]
    assert data["image_info"]["name"] == "synthetic_seed4.png"
    again = client.post("/api/generate", json={"num_cells": 30, "seed": 4}).get_json()
    assert again["image"] == data["image"], "seeded generation must be reproducible"
    assert "evaluation" not in again


def test_generate_legacy_route_and_validation(client) -> None:
    ok = client.post(
        "/generate_synthetic",
        json={"num_cells": 10, "noise_level": 0.1, "cell_shape": "ellipse", "allow_overlap": True},
    ).get_json()
    assert ok["success"]
    assert ok["cell_shape"] == "ellipse"
    for body, message in [
        ({"num_cells": 0}, "between"),
        ({"num_cells": 10, "cell_shape": "star"}, "shape"),
        ("not json", "JSON"),
    ]:
        response = (
            client.post("/api/generate", data=body, content_type="text/plain")
            if isinstance(body, str)
            else client.post("/api/generate", json=body)
        )
        assert response.status_code == 400
        assert message in response.get_json()["error"]


def test_unexpected_errors_are_hidden(client, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("cell_identifier.web.analyze", boom)
    response = client.post("/api/analyze", json={"sample": True})
    assert response.status_code == 500
    assert "secret" not in response.get_json()["error"]


def test_param_translation() -> None:
    config = analysis_config_from_params(
        {"smoothing_sigma": "0", "channel": "green", "classification": "quantile"}
    )
    assert config.preprocess.denoise == "none"
    assert config.preprocess.channel == "green"
    assert config.classification.method == "quantile"
    assert analysis_config_from_params({"smoothing_sigma": 2}).preprocess.denoise_sigma == 2.0


def test_generate_applies_analysis_settings(client) -> None:
    body = {"num_cells": 20, "seed": 1, "evaluate": True, "split_method": "distance", "min_snr": 2}
    data = client.post("/api/generate", json=body).get_json()
    seg = data["report"]["config"]["segmentation"]
    assert seg["split_method"] == "distance"
    assert seg["min_snr"] == 2.0


def test_extended_param_translation() -> None:
    config = analysis_config_from_params(
        {
            "background": "tophat",
            "background_radius": "30",
            "min_snr": "2.5",
            "intensity_h": "0.08",
            "watershed_h": "2",
            "pixel_size_um": "0.065",
        }
    )
    assert config.preprocess.background == "tophat"
    assert config.preprocess.background_radius == 30.0
    assert config.segmentation.min_snr == 2.5
    assert config.segmentation.intensity_h == 0.08
    assert config.segmentation.watershed_h == 2.0
    assert config.pixel_size_um == 0.065
    assert analysis_config_from_params({"pixel_size_um": ""}).pixel_size_um is None
    assert analysis_config_from_params({"pixel_size_um": "0"}).pixel_size_um is None
