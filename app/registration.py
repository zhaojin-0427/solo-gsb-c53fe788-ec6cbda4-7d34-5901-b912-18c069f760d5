"""Robust integer-translation registration against the static background.

The scene may contain a moving high-contrast foreground, so ordinary
least-squares / correlation registration can be dragged onto the moving
object.  Instead we exhaustively test every integer shift in +-8 px and, for
each shift, score a fixed set of background-sized tiles by the *fraction of
unsheared samples whose exposure-normalized radiance agrees within the
tolerance*.  The per-tile fractions are combined by their median: a minority of
tiles covered by the moving object (or by clipped highlights) cannot pull the
result, because the majority static-background tiles dominate the median.

Clipped samples (any channel 0 / 65535) are masked out before scoring, and
flat tiles (gradient at rounding-noise level) are ignored, so saturation or
uniform regions never anchor the shift.
"""
from __future__ import annotations

import numpy as np

from .errors import MergeError

MAX_SHIFT = 8

# Floating-point rounding allowance (radiance units) added to the tolerance.
# A value below 1/1000 of one normalized LSB per second of exposure; it only
# absorbs float32 rounding of the same physical radiance divided by different
# exposure times.
ROUND_EPS = 1e-6

# A candidate shift must reach this median per-tile inlier fraction.
MIN_MEDIAN_INLIER = 0.80
# ... and this fraction of inliers over all textured background tiles.
MIN_MEAN_INLIER = 0.50
# Best vs second-best margin for a unique answer.
UNIQUE_TIGHT_MARGIN = 0.03   # fraction of well-fitted background samples
UNIQUE_MEDIAN_MARGIN = 0.02
UNIQUE_REL_MARGIN = 0.05
UNIQUE_ABS_MARGIN = 0.02
# A tile needs at least this fraction of comparable (unclipped, in-frame)
# samples before it counts for a given shift.
MIN_TILE_COVERAGE = 0.25
# Texture floor in radiance units (mean abs gradient); keeps flat tiles out.
# Independent of the tolerance: tiles with *any* structure above rounding
# noise can be tested, while genuinely uniform tiles (which agree for every
# shift and are uninformative) are discarded.
TEXTURE_FLOOR = 2e-3

_TILE_SIZES = (16, 8, 4, 2, 1)
_STRIDES = (1, 2, 4, 8, 16)
_SAMPLE_BUDGET = 400_000  # sampled pixels used while scoring one shift


def _pick_layout(h: int, w: int) -> tuple[int, int]:
    """Return ``(tile, stride)`` so at least a 4x4 tile grid survives the
    worst 8 px displacement on every side, and the per-shift sampled pixel
    count stays near :data:`_SAMPLE_BUDGET`."""
    tile = 1
    # After a shift of up to MAX_SHIFT px the usable grid is the interior
    # (nh*tile - 2*MAX_SHIFT) by (nw*tile - 2*MAX_SHIFT); keep >= 4x4 there.
    for s in _TILE_SIZES:
        nh, nw = h // s, w // s
        if nh * s - 2 * MAX_SHIFT >= 4 * s and nw * s - 2 * MAX_SHIFT >= 4 * s:
            tile = s
            break
    stride = tile
    for st in _STRIDES:
        if st <= tile and (h * w) // (st * st) <= _SAMPLE_BUDGET:
            stride = st
            break
    return tile, stride


def _tile_texture(
    radiance: np.ndarray, valid: np.ndarray, ids: np.ndarray, n_tiles: int
) -> np.ndarray:
    """Mean magnitude of the reference gradient per tile (radiance units)."""
    h, w = valid.shape
    # Horizontal gradients, compare sampled positions consistently via ids.
    gx = np.abs(radiance[:, 1:, :] - radiance[:, :-1, :]).mean(axis=2)
    gxv = valid[:, 1:] & valid[:, :-1]
    gy = np.abs(radiance[1:, :, :] - radiance[:-1, :, :]).mean(axis=2)
    gyv = valid[1:, :] & valid[:-1, :]

    mag_sum = np.zeros(n_tiles, dtype=np.float64)
    cnt = np.zeros(n_tiles, dtype=np.int64)
    np.add.at(mag_sum, ids[:, 1:][gxv], gx[gxv])
    np.add.at(cnt, ids[:, 1:][gxv], 1)
    np.add.at(mag_sum, ids[1:, :][gyv], gy[gyv])
    np.add.at(cnt, ids[1:, :][gyv], 1)
    texture = np.zeros(n_tiles, dtype=np.float64)
    ok = cnt > 0
    texture[ok] = mag_sum[ok] / cnt[ok]
    return texture


def register_frame(
    ref_rad: np.ndarray,
    ref_valid: np.ndarray,
    rad: np.ndarray,
    valid: np.ndarray,
    tolerance: float,
    ref_exposure: float,
    exposure: float,
) -> tuple[int, int]:
    """Estimate ``(dy, dx)`` that maps *rad* onto the reference frame.

    A frame pixel at ``(y, x)`` was photographed from the same scene point as
    reference pixel ``(y + dy, x + dx)``.  Raises :class:`MergeError` when the
    static background cannot be registered reliably.
    """
    h, w = ref_valid.shape
    tile, stride = _pick_layout(h, w)
    nh, nw = h // tile, w // tile
    n_tiles = nh * nw

    base = np.arange(n_tiles, dtype=np.int32).reshape(nh, nw)
    ids_full = np.repeat(np.repeat(base, tile, axis=0), tile, axis=1)
    if ids_full.shape[0] < h:
        pad_h = h - ids_full.shape[0]
        ids_full = np.vstack(
            [ids_full, np.full((pad_h, ids_full.shape[1]), -1, np.int32)]
        )
    if ids_full.shape[1] < w:
        pad_w = w - ids_full.shape[1]
        ids_full = np.hstack(
            [ids_full, np.full((ids_full.shape[0], pad_w), -1, np.int32)]
        )

    texture = _tile_texture(ref_rad, ref_valid, ids_full, n_tiles)

    # Samples expected inside one fully-covered, sampled tile.
    per_tile = (tile // stride) ** 2
    min_pairs = max(3, int(np.ceil(MIN_TILE_COVERAGE * per_tile)))
    # 16-bit quantization + float32 rounding allowance (radiance units) for
    # the two shortest exposures involved. Every agreement test gets this even
    # when the user tolerance is 0: it is below one full 16-bit level at the
    # shortest exposure, so it absorbs rounding without accepting misalignment.
    quant = 0.5 / 65535.0 * (1.0 / ref_exposure + 1.0 / exposure)
    quant_allowance = np.float32(quant + ROUND_EPS)
    threshold = np.float32(tolerance) + quant_allowance
    # "Tight" agreement band for localization: neighboring integer shifts over
    # textured background differ by far more than quantization, so it pins the
    # peak precisely even when the user tolerance is generous.
    tight_threshold = np.maximum(quant_allowance, np.float32(3e-5))

    def score_shift(dy: int, dx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Frame pixel (y, x) maps to reference (y + dy, x + dx); hence for a
        # reference target t the corresponding frame sample is t - (dy, dx).
        ty0, ty1 = max(0, dy), min(h, h + dy)
        tx0, tx1 = max(0, dx), min(w, w + dx)
        sy0, sy1 = ty0 - dy, ty1 - dy
        sx0, sx1 = tx0 - dx, tx1 - dx

        a = ref_rad[ty0:ty1:stride, tx0:tx1:stride]
        b = rad[sy0:sy1:stride, sx0:sx1:stride]
        va = ref_valid[ty0:ty1:stride, tx0:tx1:stride]
        vb = valid[sy0:sy1:stride, sx0:sx1:stride]
        tids = ids_full[ty0:ty1:stride, tx0:tx1:stride]

        both = va & vb
        diff = np.abs(a - b).max(axis=2)
        agree = both & (diff <= threshold)
        tight_ok = both & (diff <= tight_threshold)

        vcount = np.bincount(
            tids[both], minlength=n_tiles
        ).astype(np.float64)
        icount = np.bincount(
            tids[agree], minlength=n_tiles
        ).astype(np.float64)
        tcount = np.bincount(
            tids[tight_ok], minlength=n_tiles
        ).astype(np.float64)
        return vcount, icount, tcount

    # Zero shift first: it defines reference coverage for tile eligibility and
    # is, of course, also a candidate.
    v0, i0, t0 = score_shift(0, 0)
    eligible = (texture > TEXTURE_FLOOR) & (v0 >= min_pairs)
    n_eligible = int(eligible.sum())
    if n_eligible < max(1, min(3, n_tiles // 9)):
        raise MergeError(
            "registration failed: static background lacks sufficient "
            "unsheared texture to anchor alignment"
        )

    def evaluate(vcount: np.ndarray, icount: np.ndarray, tcount: np.ndarray):
        used = eligible & (vcount >= min_pairs)
        if used.sum() < max(1, n_eligible // 2):
            return None
        ratios = icount[used] / vcount[used]
        median = float(np.median(ratios))
        wmean = float(icount[used].sum() / vcount[used].sum())
        gmean = float(icount.sum() / vcount.sum()) if vcount.sum() > 0 else 0.0
        tight = float(tcount[used].sum() / vcount[used].sum())
        return median, wmean, gmean, tight

    results: dict[tuple[int, int], tuple] = {}
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            if dy == 0 and dx == 0:
                scores = evaluate(v0, i0, t0)
            else:
                scores = evaluate(*score_shift(dy, dx))
            if scores is None:
                results[(dy, dx)] = (-1.0, -1.0, -1.0, -1.0)
            else:
                results[(dy, dx)] = scores

    # Primary: tolerance-band per-tile median (background/mover robustness);
    # then tight-band localization, coverage-weighted mean, global mean;
    # finally prefer the smaller shift on exact ties.
    ranked = sorted(
        results.items(),
        key=lambda kv: (
            kv[1][0],
            kv[1][3],
            kv[1][1],
            kv[1][2],
            -((kv[0][0]) ** 2 + (kv[0][1]) ** 2),
        ),
        reverse=True,
    )
    (best_dy, best_dx), (b_med, b_wmean, _, b_tight) = ranked[0]
    (_, _), (s_med, s_wmean, _, s_tight) = ranked[1]
    if b_med < MIN_MEDIAN_INLIER:
        raise MergeError(
            "registration failed: no shift aligns enough unsheared static "
            "background (moving occlusion or frame mismatch too large)"
        )
    if b_wmean < MIN_MEAN_INLIER:
        raise MergeError(
            "registration failed: overall static-background agreement too low"
        )
    if not (
        b_tight - s_tight >= UNIQUE_TIGHT_MARGIN
        or (
            b_med - s_med >= UNIQUE_MEDIAN_MARGIN
            and b_wmean >= (1.0 + UNIQUE_REL_MARGIN) * s_wmean
        )
        or b_wmean - s_wmean >= UNIQUE_ABS_MARGIN
    ):
        raise MergeError(
            "registration failed: alignment not unique (ambiguous background)"
        )

    return best_dy, best_dx
