"""Exposure-stack merging: registration, motion masking, radiance fusion."""
from __future__ import annotations

import numpy as np

from .errors import MergeError
from .imaging import frame_radiance
from .registration import register_frame

# Request-level guards
MAX_FRAMES = 9
MIN_FRAMES = 3
# Largest accepted side. With 9 float32 RGB frames this keeps peak memory
# (reference frame + current frame + per-pixel statistics) around 1 GB.
MAX_SIDE = 2048


def merge_stack(
    frames: list[np.ndarray],
    exposures: list[float],
    reference_index: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Merge a bracketed stack.

    Returns ``(radiance float32 HxWx3, motion_mask, unknown_mask, shifts)``
    where ``shifts[i] = (dy, dx)`` maps frame i onto the reference frame and
    masks are boolean HxW.
    """
    if not (MIN_FRAMES <= len(frames) <= MAX_FRAMES):
        raise MergeError(
            f"expected between {MIN_FRAMES} and {MAX_FRAMES} frames, "
            f"got {len(frames)}"
        )
    n = len(frames)
    if not (0 <= reference_index < n):
        raise MergeError(
            f"reference_index must be in [0, {n - 1}], got {reference_index}"
        )
    if len(exposures) != n:
        raise MergeError("number of exposures must equal number of frames")
    if any(not np.isfinite(e) or e <= 0.0 for e in exposures):
        raise MergeError("exposure seconds must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise MergeError("tolerance must be a finite non-negative number")

    h, w = frames[0].shape[:2]
    if h < 8 or w < 8 or h > MAX_SIDE or w > MAX_SIDE:
        raise MergeError(
            f"frame dimensions must be between 8 and {MAX_SIDE} px per side"
        )
    for i, f in enumerate(frames):
        if f.shape[:2] != (h, w):
            raise MergeError(
                f"frame {i} size {f.shape[:2]} differs from reference size "
                f"{(h, w)}; all frames must be the same size"
            )

    radiance = []
    valid = []
    for i, (frame, exp) in enumerate(zip(frames, exposures)):
        r, m = frame_radiance(frame, float(exp))
        radiance.append(r)
        valid.append(m)

    ref_rad = radiance[reference_index]
    ref_valid = valid[reference_index]

    # ---- register every frame against the unsheared static background ----
    shifts: list[tuple[int, int]] = [(0, 0)] * n
    for i in range(n):
        if i == reference_index:
            continue
        shifts[i] = register_frame(
            ref_rad,
            ref_valid,
            radiance[i],
            valid[i],
            float(tolerance),
            float(exposures[reference_index]),
            float(exposures[i]),
        )

    # ---- gather per-pixel statistics over valid observations ----
    # Frames are warped one at a time into reusable buffers so peak memory
    # stays at ~(reference + one frame + accumulators) instead of n frames.
    count = np.zeros((h, w), dtype=np.int32)
    mins = np.full((h, w, 3), np.inf, dtype=np.float32)
    maxs = np.zeros((h, w, 3), dtype=np.float32)
    sums = np.zeros((h, w, 3), dtype=np.float32)

    warp_rad = np.empty_like(ref_rad)
    warp_valid = np.empty((h, w), dtype=bool)
    for i in range(n):
        if i == reference_index:
            r, m = ref_rad, ref_valid
        else:
            dy, dx = shifts[i]
            warp_rad.fill(0.0)
            warp_valid.fill(False)
            ty0, ty1 = max(0, dy), min(h, h + dy)
            tx0, tx1 = max(0, dx), min(w, w + dx)
            warp_rad[ty0:ty1, tx0:tx1] = radiance[i][
                ty0 - dy : ty1 - dy, tx0 - dx : tx1 - dx
            ]
            warp_valid[ty0:ty1, tx0:tx1] = valid[i][
                ty0 - dy : ty1 - dy, tx0 - dx : tx1 - dx
            ]
            r, m = warp_rad, warp_valid

        count += m
        sums += r * m[:, :, None]
        mins = np.where(m[:, :, None], np.minimum(mins, r), mins)
        maxs = np.where(m[:, :, None], np.maximum(maxs, r), maxs)

    at_least_two = count >= 2
    spread = (maxs - mins).max(axis=2)
    motion = at_least_two & (spread > np.float32(tolerance))

    # ---- fuse ----
    result = np.full((h, w, 3), np.nan, dtype=np.float32)

    static = ~motion
    has_static_obs = count > 0
    static_fuse = static & has_static_obs
    result[static_fuse] = sums[static_fuse] / count[static_fuse][:, None]

    # Motion regions: keep the reference exposure only (no ghosting).
    motion_fuse = motion & ref_valid
    result[motion_fuse] = ref_rad[motion_fuse]

    # Motion region with invalid reference observation -> unknown.
    unknown = (static & ~has_static_obs) | (motion & ~ref_valid)

    return result, motion, unknown, shifts
