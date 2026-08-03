"""
Helpers for working with camera frames without importing OpenCV.

Frames in this project are standard BGR uint8 numpy arrays from the camera
pipeline. These helpers convert them to RGB and JPEG-encode them for vision APIs
without loading cv2, which avoids SDL conflicts with pygame on macOS.
"""

import base64
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Return a contiguous RGB view of a BGR frame."""
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected HxWx3 frame, got shape={frame.shape!r}")
    return np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])


def encode_jpeg_bytes(
    frame: np.ndarray, quality: int = 85, max_dim: Optional[int] = None
) -> Optional[bytes]:
    """JPEG-encode a BGR frame and return bytes, or None on failure.

    max_dim: when set and the frame's longest side exceeds it, downscale
    (aspect-preserving, LANCZOS) before encoding. A full 1920x1080 frame is
    ~300-600KB of JPEG that the vision API tiles at high token cost; room-level
    queries lose nothing at 1024px and upload ~4x less.
    """
    try:
        rgb = bgr_to_rgb(frame).astype(np.uint8, copy=False)
        image = Image.fromarray(rgb)
        if max_dim and max(image.size) > int(max_dim):
            image.thumbnail((int(max_dim), int(max_dim)), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()
    except Exception as exc:
        _log.error("encode_jpeg_bytes: failed to encode frame: %s", exc)
        return None


def encode_jpeg_base64(
    frame: np.ndarray, quality: int = 85, max_dim: Optional[int] = None
) -> Optional[str]:
    """JPEG-encode a BGR frame and return a base64 string, or None on failure."""
    encoded = encode_jpeg_bytes(frame, quality=quality, max_dim=max_dim)
    if encoded is None:
        return None
    return base64.b64encode(encoded).decode("ascii")
