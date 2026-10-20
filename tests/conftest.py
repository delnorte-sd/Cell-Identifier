"""Shared, deterministic fixtures. Every random input is seeded."""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
import pytest

from cell_identifier import SyntheticImage, generate
from cell_identifier.image_io import save_image


@pytest.fixture(scope="session")
def sample_path() -> Path:
    """The bundled real E. coli GFP micrograph."""
    path = Path(str(resources.files("cell_identifier.web").joinpath("static", "sample.jpeg")))
    assert path.is_file()
    return path


@pytest.fixture(scope="session")
def pill_image() -> SyntheticImage:
    """Well-separated rod-shaped cells with exact ground truth (seed 7)."""
    return generate(seed=7, num_cells=40, shape="pill")


@pytest.fixture(scope="session")
def ellipse_image() -> SyntheticImage:
    """Well-separated elliptical cells with exact ground truth (seed 11)."""
    return generate(seed=11, num_cells=40, shape="ellipse")


DrawFn = Callable[..., np.ndarray]


@pytest.fixture
def draw_discs() -> DrawFn:
    """Draw bright discs on a dark 8-bit greyscale canvas.

    ``draw_discs((cx, cy, r), ..., size=(h, w), value=200, background=10)``
    """

    def _draw(
        *discs: tuple[int, int, int],
        size: tuple[int, int] = (128, 128),
        value: int = 200,
        background: int = 10,
    ) -> np.ndarray:
        img = np.full(size, background, dtype=np.uint8)
        for cx, cy, r in discs:
            cv2.circle(img, (cx, cy), r, value, -1)
        return img

    return _draw


@pytest.fixture
def write_png(tmp_path: Path) -> Callable[[str, np.ndarray], Path]:
    """Save an array as a PNG inside the test's temporary directory."""

    def _write(name: str, image: np.ndarray) -> Path:
        return save_image(tmp_path / name, image)

    return _write
