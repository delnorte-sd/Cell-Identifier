from __future__ import annotations

import io

import pytest

pytest.importorskip("flask")

from cell_identifier import generate
from cell_identifier.image_io import encode_png
from cell_identifier.web import analysis_config_from_params, create_app


@pytest.fixture
def client():
    return create_app({"TESTING": True}).test_client()


def test_index_and_health(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert b"Cell Image Analysis" in page.data
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
