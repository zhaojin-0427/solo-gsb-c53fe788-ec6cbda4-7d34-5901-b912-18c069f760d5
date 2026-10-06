"""Image decoding / encoding helpers.

Inputs are 16-bit linear RGB PNGs.  A sample is *invalid* (saturated /
black-clipped) when any channel is exactly 0 or 65535.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from .errors import MergeError

UINT16_MAX = 65535
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def decode_frame_png(b64_data: str) -> np.ndarray:
    """Decode a Base64 16-bit RGB PNG to an ``uint16`` HxWx3 array."""
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except Exception as exc:  # noqa: BLE001 - normalize every decode failure
        raise MergeError("frame is not valid Base64") from exc

    if not raw.startswith(PNG_SIGNATURE):
        raise MergeError("frame is not a PNG image")

    # IMREAD_UNCHANGED keeps 16-bit depth; convert to 3-channel RGB afterwards.
    decoded = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if decoded is None:
        raise MergeError("PNG could not be decoded")

    if decoded.dtype != np.uint16:
        raise MergeError(
            f"PNG must be 16-bit per channel, got bit depth '{decoded.dtype}'"
        )

    if decoded.ndim == 2:
        img = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        img = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        raise MergeError(f"unsupported channel count {decoded.shape}")

    return np.ascontiguousarray(img, dtype=np.uint16)


def valid_mask(digital: np.ndarray) -> np.ndarray:
    """True where no channel is clipped to 0 or 65535 (HxW, bool)."""
    return np.all((digital > 0) & (digital < UINT16_MAX), axis=2)


def frame_radiance(digital: np.ndarray, exposure_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(radiance float32 HxWx3, valid bool HxW)`` for one frame.

    Valid samples are divided by 65535 and by the exposure time (seconds).
    """
    mask = valid_mask(digital)
    radiance = digital.astype(np.float32) / np.float32(UINT16_MAX * exposure_s)
    radiance[~mask] = 0.0
    return radiance, mask


def encode_radiance_tiff(radiance: np.ndarray) -> bytes:
    """Encode an ``HxWx3 float32`` radiance image as a TIFF (unknown=NaN).

    32-bit float TIFFs have no BGR/RGB photometric ambiguity for 3 channels;
    OpenCV writes them as sample format IEEEFP with photometric RGB.
    """
    assert radiance.dtype == np.float32 and radiance.ndim == 3 and radiance.shape[2] == 3
    success, buf = cv2.imencode(
        ".tif", radiance, (cv2.IMWRITE_TIFF_COMPRESSION, 8)
    )  # 8 = Deflate
    if not success:
        # Some builds lack deflate support for float TIFFs - fall back to none.
        success, buf = cv2.imencode(".tif", radiance)
        if not success:
            raise MergeError("failed to encode float32 TIFF")
    return buf.tobytes()


def encode_mask_png(mask: np.ndarray) -> bytes:
    """Encode a boolean / uint8 single-channel mask as an 8-bit PNG."""
    m = (mask.astype(bool) * 255).astype(np.uint8)
    success, buf = cv2.imencode(".png", m)
    if not success:
        raise MergeError("failed to encode mask PNG")
    return buf.tobytes()


def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
