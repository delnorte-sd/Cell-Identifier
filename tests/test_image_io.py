from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cell_identifier.image_io import (
    MAX_PIXELS,
    ImageValidationError,
    encode_png,
    full_scale,
    load_image,
    save_image,
    to_display_uint8,
    validate_array,
)


def test_load_png_path_returns_rgb_and_metadata(tmp_path: Path) -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # pure red
    path = save_image(tmp_path / "red.png", rgb)
    loaded = load_image(path)
    assert loaded.data.shape == (20, 30, 3)
    assert loaded.data[0, 0].tolist() == [200, 0, 0], "channels must be RGB, not BGR"
    meta = loaded.metadata
    assert (meta.width, meta.height, meta.channels, meta.dtype) == (30, 20, 3, "uint8")
    assert len(meta.sha256) == 64
    assert meta.source == str(path)


def test_bytes_and_path_give_identical_pixels_and_hash(tmp_path: Path) -> None:
    img = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    path = save_image(tmp_path / "ramp.png", img)
    from_path = load_image(path)
    from_bytes = load_image(path.read_bytes(), name="upload.png")
    np.testing.assert_array_equal(from_path.data, from_bytes.data)
    assert from_path.metadata.sha256 == from_bytes.metadata.sha256
    assert from_bytes.metadata.source == "upload.png"


def test_16bit_tiff_is_not_truncated(tmp_path: Path) -> None:
    img = np.full((16, 16), 40000, dtype=np.uint16)
    loaded = load_image(save_image(tmp_path / "deep.tif", img))
    assert loaded.data.dtype == np.uint16
    assert int(loaded.data.max()) == 40000
    assert loaded.full_scale == 65535.0


def test_rgba_alpha_is_dropped(tmp_path: Path) -> None:
    bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    bgra[..., 1] = 255
    bgra[..., 3] = 128
    path = tmp_path / "alpha.png"
    path.write_bytes(cv2.imencode(".png", bgra)[1].tobytes())
    assert load_image(path).data.shape == (10, 10, 3)


def test_non_ascii_path(tmp_path: Path) -> None:
    path = save_image(tmp_path / "células_µm.png", np.zeros((12, 12), dtype=np.uint8))
    assert load_image(path).data.shape == (12, 12)


def test_missing_file_and_bad_extension(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageValidationError, match="unsupported file type"):
        load_image(txt)


@pytest.mark.parametrize("payload", [b"", b"not an image at all"])
def test_undecodable_bytes(payload: bytes) -> None:
    with pytest.raises(ImageValidationError):
        load_image(payload)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((4, 100), dtype=np.uint8), "at least"),
        (np.zeros((10, 10, 2), dtype=np.uint8), "shape"),
        (np.zeros((10, 10, 3, 1), dtype=np.uint8), "shape"),
        (np.zeros((10, 10), dtype=np.int64), "unsupported pixel type"),
        (np.full((10, 10), np.nan, dtype=np.float32), "NaN"),
        (np.full((10, 10), -1.0, dtype=np.float32), "non-negative"),
    ],
)
def test_validate_array_rejects_bad_input(array: np.ndarray, message: str) -> None:
    with pytest.raises(ImageValidationError, match=message):
        validate_array(array)


def test_validate_array_rejects_non_arrays_and_huge_images(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ImageValidationError, match="numpy array"):
        validate_array([[1, 2], [3, 4]])  # type: ignore[arg-type]
    monkeypatch.setattr("cell_identifier.image_io.MAX_PIXELS", 99)
    with pytest.raises(ImageValidationError, match="limit"):
        validate_array(np.zeros((10, 10), dtype=np.uint8))
    assert MAX_PIXELS > 99


def test_validate_array_normalises_layout_without_mutating() -> None:
    single = np.ones((10, 10, 1), dtype=np.uint8)
    assert validate_array(single).shape == (10, 10)
    rgba = np.ones((10, 10, 4), dtype=np.uint8)
    assert validate_array(rgba).shape == (10, 10, 3)
    assert rgba.shape == (10, 10, 4)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = True
    assert validate_array(mask).max() == 255


def test_full_scale_and_display_conversion() -> None:
    assert full_scale(np.zeros(4, dtype=np.uint8)) == 255.0
    assert full_scale(np.zeros(4, dtype=np.uint16)) == 65535.0
    assert full_scale(np.array([0.0, 0.5], dtype=np.float32)) == 1.0
    assert full_scale(np.array([0.0, 4.0], dtype=np.float32)) == 4.0
    deep = np.array([[0, 65535]], dtype=np.uint16)
    assert to_display_uint8(deep).tolist() == [[0, 255]]


def test_encode_png_round_trip() -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (9, 9, 3), dtype=np.uint8)
    np.testing.assert_array_equal(load_image(encode_png(rgb)).data, rgb)


def test_save_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ImageValidationError):
        save_image(tmp_path / "x.gif", np.zeros((10, 10), dtype=np.uint8))
