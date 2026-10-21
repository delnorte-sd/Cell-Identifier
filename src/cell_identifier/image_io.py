"""Image loading, validation and saving.

All images inside the package use **RGB** channel order (OpenCV's BGR is
converted at the I/O boundary) and keep their native dtype, so 16-bit
microscopy data is never silently truncated to 8 bits.

Files are read as bytes and decoded in memory, which avoids OpenCV's problems
with non-ASCII paths on Windows and lets the web app analyse uploads without
writing them to disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "MAX_PIXELS",
    "MIN_DIMENSION",
    "SUPPORTED_EXTENSIONS",
    "ImageMetadata",
    "ImageValidationError",
    "LoadedImage",
    "encode_png",
    "full_scale",
    "load_image",
    "save_image",
    "to_display_uint8",
    "validate_array",
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
#: Smallest accepted image side, in pixels.
MIN_DIMENSION = 8
#: Largest accepted image, in pixels (guards memory use; ~100 megapixels).
MAX_PIXELS = 100_000_000

_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)


class ImageValidationError(ValueError):
    """Raised when an input cannot be decoded or is not a usable microscopy image."""


@dataclass(frozen=True)
class ImageMetadata:
    """Provenance information recorded alongside every analysis.

    Attributes:
        source: File path or caller-supplied name, or ``"<array>"``.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: 1 for greyscale, 3 for RGB (alpha channels are dropped).
        dtype: Native pixel type, e.g. ``"uint8"`` or ``"uint16"``.
        sha256: Hash of the encoded file bytes, or of the raw array buffer.
    """

    source: str
    width: int
    height: int
    channels: int
    dtype: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class LoadedImage:
    """A validated image plus its metadata.

    Attributes:
        data: ``(H, W)`` greyscale or ``(H, W, 3)`` RGB array in its native dtype.
        metadata: Provenance information.
    """

    data: np.ndarray
    metadata: ImageMetadata

    @property
    def full_scale(self) -> float:
        """Intensity value that corresponds to "fully saturated" for this dtype."""
        return full_scale(self.data)


def full_scale(image: np.ndarray) -> float:
    """Return the nominal maximum intensity for an image's dtype.

    255 for ``uint8``, 65535 for ``uint16``; floating-point images are assumed to
    be normalised to ``[0, 1]`` unless their maximum exceeds 1, in which case the
    maximum itself is used.
    """
    if image.dtype == np.uint8:
        return 255.0
    if image.dtype == np.uint16:
        return 65535.0
    peak = float(np.max(image)) if image.size else 1.0
    return peak if peak > 1.0 else 1.0


def validate_array(image: np.ndarray) -> np.ndarray:
    """Check that ``image`` is a usable 2-D greyscale or RGB array and normalise its layout.

    Returns the array with any singleton channel axis squeezed away and any alpha
    channel removed. Never modifies the input.

    Raises:
        ImageValidationError: on wrong dimensionality, channel count, dtype or size,
            or on non-finite pixel values.
    """
    if not isinstance(image, np.ndarray):
        raise ImageValidationError(f"expected a numpy array, got {type(image).__name__}")
    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255
    if image.dtype not in _SUPPORTED_DTYPES:
        raise ImageValidationError(
            f"unsupported pixel type {image.dtype}; use uint8, uint16 or float images"
        )
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ImageValidationError(
            f"expected a greyscale (H, W) or colour (H, W, 3) image, got shape {image.shape}"
        )
    height, width = image.shape[:2]
    if min(height, width) < MIN_DIMENSION:
        raise ImageValidationError(
            f"image is {width}x{height} px; both sides must be at least {MIN_DIMENSION} px"
        )
    if height * width > MAX_PIXELS:
        raise ImageValidationError(
            f"image has {height * width:,} pixels; the limit is {MAX_PIXELS:,}"
        )
    if image.dtype.kind == "f":
        if not np.all(np.isfinite(image)):
            raise ImageValidationError("image contains NaN or infinite values")
        if float(np.min(image)) < 0:
            raise ImageValidationError("floating-point images must be non-negative")
    return image


def _decode(data: bytes, source: str) -> np.ndarray:
    if not data:
        raise ImageValidationError(f"{source}: file is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ImageValidationError(
            f"{source}: could not decode image (supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )
    if decoded.ndim == 3 and decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(decoded)


def load_image(
    source: str | Path | bytes | np.ndarray,
    *,
    name: str | None = None,
) -> LoadedImage:
    """Load and validate an image from a path, encoded bytes, or an array.

    Args:
        source: File path, encoded image bytes (PNG/JPEG/TIFF/BMP), or an array
            already in RGB / greyscale layout.
        name: Label stored in the metadata (defaults to the path, ``"<bytes>"``
            or ``"<array>"``).

    Raises:
        FileNotFoundError: if a path does not exist.
        ImageValidationError: if the input is not a decodable, sensible image.
    """
    if isinstance(source, np.ndarray):
        array = validate_array(source)
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        label = name or "<array>"
    else:
        if isinstance(source, bytes | bytearray | memoryview):
            raw = bytes(source)
            label = name or "<bytes>"
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"image not found: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ImageValidationError(
                    f"{path.name}: unsupported file type {path.suffix!r} (supported: "
                    f"{', '.join(sorted(SUPPORTED_EXTENSIONS))})"
                )
            raw = path.read_bytes()
            label = name or str(path)
        array = validate_array(_decode(raw, label))
        digest = hashlib.sha256(raw).hexdigest()
    metadata = ImageMetadata(
        source=label,
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        channels=1 if array.ndim == 2 else 3,
        dtype=str(array.dtype),
        sha256=digest,
    )
    return LoadedImage(data=array, metadata=metadata)


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any supported image to 8-bit for display, scaling by its full scale."""
    if image.dtype == np.uint8:
        return image.copy()
    scaled = image.astype(np.float64) / full_scale(image) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB(A) / greyscale ``uint8`` or ``uint16`` array as PNG bytes."""
    if image.dtype not in (np.uint8, np.uint16):
        image = to_display_uint8(image)
    ok, buffer = cv2.imencode(".png", _to_bgr(image))
    if not ok:  # pragma: no cover - OpenCV only fails on invalid arrays
        raise ImageValidationError("PNG encoding failed")
    return bytes(buffer.tobytes())


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Write an RGB / greyscale image; the format follows the file extension.

    ``uint16`` arrays (e.g. label images) are preserved in PNG and TIFF.
    Parent directories are created as needed.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(f"cannot save to {path.name}: unsupported extension")
    lossy_16bit = image.dtype == np.uint16 and ext not in {".png", ".tif", ".tiff"}
    if lossy_16bit or image.dtype not in (np.uint8, np.uint16):
        image = to_display_uint8(image)
    ok, buffer = cv2.imencode(ext, _to_bgr(image))
    if not ok:  # pragma: no cover
        raise ImageValidationError(f"could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())
    return path
