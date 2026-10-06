"""Synthetic 16-bit bracketed stacks for tests and README examples.

Scene: textured static background (gradient + checkerboard + faint noise),
a bright moving rectangle, and a clipped highlight.  Frames differ by an
integer translation and by the rectangle's position.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

H, W = 48, 64
EXPOSURES = [0.25, 1.0, 2.0]  # seconds
# Shift (dy, dx) mapping each frame's samples onto the reference (frame 1):
# frame pixel (y, x) corresponds to reference pixel (y + dy, x + dx).
SHIFTS = [(3, -5), (0, 0), (-4, 6)]
# Moving bright rectangle (top, left, bottom, right) in each frame's own coords.
RECTS = [(8, 10, 18, 22), (8, 10, 18, 22), (26, 36, 36, 48)]
REF_INDEX = 1


def background_at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Static background as a function of *world* pixel coordinates.

    Uses only a constant linear ramp plus a periodic checkerboard, so values
    computed for shifted viewports agree exactly (no window-dependent
    interpolation) while remaining richly textured for registration.
    """
    yy = np.asarray(yy, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    rad = np.empty(yy.shape + (3,), dtype=np.float64)
    rad[..., 0] = 0.30 + 0.003 * yy
    rad[..., 1] = 0.25 + 0.003 * xx
    rad[..., 2] = 0.35 + 0.001 * (yy + xx)
    checker = (((yy // 8).astype(int) + (xx // 8).astype(int)) % 2)
    rad += checker[..., None] * 0.10
    return np.clip(rad, 0.0, 1.0)


def background() -> np.ndarray:
    """Reference-viewport background radiance (world coords = viewport)."""
    yy, xx = np.indices((H, W))
    return background_at(yy, xx)


def scene(rect: tuple[int, int, int, int]) -> np.ndarray:
    rad = background()
    t, l, b, r = rect
    rad[t:b, l:r, :] = 0.12  # moving object (unsheared even at 2.0 s)
    rad[2:7, 44:51, :] = 1.0  # clipped highlight: saturates on long exposure
    return rad


def render(rad: np.ndarray, exposure: float) -> np.ndarray:
    """Convert scene radiance to 16-bit digital values (with clipping)."""
    digital = rad * exposure * 65535.0
    return np.clip(np.rint(digital), 0, 65535).astype(np.uint16)


def _padded_scene(
    rect: tuple[int, int, int, int], shift: tuple[int, int]
) -> np.ndarray:
    """Scene on an (H+2M) x (W+2M) canvas, padded with background so a
    shifted camera always sees real scene content (black padding would be
    indistinguishable from clipping). The moving rectangle is placed at this
    frame's viewport coordinates *rect* (it moves between frames); the
    highlight is a fixed world-space feature."""
    m = 8
    dy, dx = shift
    wyy, wxx = np.indices((H + 2 * m, W + 2 * m))
    wyy -= m
    wxx -= m
    big = background_at(wyy, wxx)
    t, l, b, r = rect  # viewport coords -> world coords via view offset
    big[m + dy + t : m + dy + b, m + dx + l : m + dx + r, :] = 0.12
    big[m + 2 : m + 7, m + 44 : m + 51, :] = 1.0  # fixed clipped highlight
    return big


def make_frames() -> list[np.ndarray]:
    """Return 3 shifted digital frames.

    Frame i shows the padded scene shifted by ``-SHIFTS[i]`` so that warping
    by ``SHIFTS[i]`` (source pixel -> reference coords s + shift) aligns its
    static background with the reference viewport.
    """
    frames = []
    m = 8
    for (dy, dx), rect in zip(SHIFTS, RECTS):
        big = _padded_scene(rect, (dy, dx))
        # Frame pixel s = reference pixel t - shift, so the frame's viewport
        # window into the padded scene starts at margin + shift.
        view = big[m + dy : m + dy + H, m + dx : m + dx + W]
        frames.append(render(view, EXPOSURES[len(frames)]))
    return frames


def encode_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def request_payload(
    frames: list[np.ndarray] | None = None,
    exposures: list[float] | None = None,
    reference_index: int = REF_INDEX,
    tolerance: float = 0.05,
) -> dict:
    frames = frames if frames is not None else make_frames()
    exposures = exposures if exposures is not None else list(EXPOSURES)
    return {
        "frames": [{"image_base64": encode_png_b64(f)} for f in frames],
        "exposure_seconds": exposures,
        "reference_index": reference_index,
        "tolerance": tolerance,
    }
