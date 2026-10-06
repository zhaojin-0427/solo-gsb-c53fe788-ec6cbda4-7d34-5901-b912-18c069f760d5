"""End-to-end tests for the merge pipeline and the HTTP API."""
from __future__ import annotations

import base64
import json
import math

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.errors import MergeError
from app.imaging import decode_frame_png
from app.main import app
from app.merge import merge_stack

from . import synthetic as syn


def run_merge(**overrides):
    payload = syn.request_payload(**overrides)
    frames = [decode_frame_png(f["image_base64"]) for f in payload["frames"]]
    return merge_stack(
        frames,
        payload["exposure_seconds"],
        payload["reference_index"],
        payload["tolerance"],
    )


def test_registration_recovers_true_shifts_not_dragged_by_mover():
    result, motion, unknown, shifts = run_merge()
    assert shifts == [(3, -5), (0, 0), (-4, 6)]


def test_static_background_fuses_to_scene_radiance():
    result, motion, unknown, _ = run_merge()
    bg = syn.background()

    # Interior pixels far from mover (old ref loc 8:18,10:22; new loc
    # 22:36,42:53), highlight and borders should be static, known, and fused
    # close to true radiance across all three exposures.
    interior = np.zeros_like(motion)
    interior[20:44, 2:38] = True
    static_known = interior & ~motion & ~unknown
    assert static_known.sum() > 250

    err = np.abs(result - bg.astype(np.float32)).max(axis=2)
    # Error < 1% of full-scale radiance (normalization/fusion error only).
    assert np.median(err[static_known]) < 0.01
    assert np.percentile(err[static_known], 95) < 0.03


def test_motion_mask_marks_moving_object():
    _, motion, _, _ = run_merge()
    # Frame 2's mover in its own coords (26..35, 36..47) maps to reference
    # coords (22..31, 42..53) with shift (-4, +6); frame 1's mover stays at
    # (8..17, 10..21). Both the object's new and old location cores are motion.
    assert motion[23:30, 43:51].mean() > 0.9
    assert motion[9:16, 11:20].mean() > 0.9
    # A static patch well away from anything moving is not motion.
    assert motion[40:46, 4:12].mean() == 0.0


def test_motion_region_takes_reference_radiance_without_ghosting():
    result, motion, _, _ = run_merge()
    frames = syn.make_frames()
    ref_digital = frames[syn.REF_INDEX]
    ref_rad = ref_digital.astype(np.float32) / np.float32(65535.0 * 1.0)
    patch_motion = motion[23:30, 43:51]
    patch_result = result[23:30, 43:51]
    patch_ref = ref_rad[23:30, 43:51]
    m = patch_motion & np.any(ref_digital[23:30, 43:51] > 0, axis=2)
    assert np.allclose(patch_result[m], patch_ref[m], atol=2e-4)
    # Frame 2 sees the moving 0.12 object at this patch while the reference
    # sees background (~0.4): a naive average would land between the two,
    # which is exactly the ghosting the reference-only rule prevents.
    ref_bg = ref_digital[25, 46].astype(np.float32) / 65535.0
    assert abs(float(ref_bg.mean()) - 0.12) > 0.2


def test_saturated_highlight_fuses_from_other_valid_frames():
    result, motion, unknown, _ = run_merge()
    # Highlight at reference coords (2:7, 44:51). Reference exposure 1.0 with
    # radiance 1.0 -> 65535 saturated -> invalid there, frame 2 (2.0 s) is
    # saturated too, but frame 0 (0.25 s) is unsheared where it overlaps the
    # patch (rows >= 4 given its shift +3); the pixel is static, so it must
    # fuse from frame 0 and be known.
    ref = syn.make_frames()[1]
    assert np.all(ref[2:7, 44:51] == 65535)
    assert unknown[4:7, 44:51].sum() == 0
    assert np.isfinite(result[4:7, 44:51]).all()
    assert np.allclose(result[4:7, 44:51], 1.0, atol=0.02)


def test_all_invalid_static_pixel_is_unknown():
    payload = syn.request_payload()
    frames = [decode_frame_png(f["image_base64"]) for f in payload["frames"]]
    # Black-clip the same static pixel in every aligned frame:
    # target (5, 40) -> frame 0 source (2, 45), frame 2 source (9, 34).
    frames[0][2, 45, :] = 0
    frames[1][5, 40, :] = 0
    frames[2][9, 34, :] = 0
    result, motion, unknown, _ = merge_stack(
        frames, payload["exposure_seconds"], 1, payload["tolerance"]
    )
    assert unknown[5, 40]
    assert np.isnan(result[5, 40]).all()


def test_out_of_bounds_with_invalid_reference_is_unknown():
    payload = syn.request_payload()
    frames = [decode_frame_png(f["image_base64"]) for f in payload["frames"]]
    # Frame 0 (shift +3 rows, -5 cols) misses target rows 0..2; frame 2
    # (shift -4 rows, +6 cols) misses target cols 0..5. Black-clip the
    # reference on the top-left 3x6 strip: every observation there is invalid
    # / out of bounds => unknown.
    frames[1][:3, :6, :] = 0
    result, motion, unknown, _ = merge_stack(
        frames, payload["exposure_seconds"], 1, payload["tolerance"]
    )
    assert unknown[:3, :6].all()
    assert np.isnan(result[:3, :6]).all()


def test_zero_tolerance_strict_motion_detection():
    # With tolerance 0, registration still succeeds (quantization allowance),
    # and per-pixel motion detection is strict: quantization differences
    # between exposures then also count as motion, per the literal rule.
    result, motion, unknown, shifts = run_merge(tolerance=0.0)
    assert shifts == [(3, -5), (0, 0), (-4, 6)]
    # The mover's new location is definitely motion.
    assert motion[23:30, 43:51].mean() > 0.9


def test_registration_fails_on_unrelated_frame():
    frames = syn.make_frames()
    rng = np.random.default_rng(0)
    frames[0] = rng.integers(100, 60000, (syn.H, syn.W, 3), np.uint16)
    with pytest.raises(MergeError, match="registration failed"):
        run_merge(frames=frames)


def test_registration_fails_on_flat_frames():
    flat = [
        np.full((syn.H, syn.W, 3), 30000, np.uint16) for _ in range(3)
    ]
    with pytest.raises(MergeError, match="texture|registration"):
        run_merge(frames=flat)


def test_size_mismatch_rejected():
    frames = syn.make_frames()
    frames[2] = frames[2][:-2, :, :]
    with pytest.raises(MergeError, match="same size"):
        run_merge(frames=frames)


def test_bad_png_rejected():
    with pytest.raises(MergeError, match="PNG"):
        decode_frame_png(base64.b64encode(b"not a png").decode())


def test_8bit_png_rejected():
    img8 = (np.ones((syn.H, syn.W, 3), np.uint8) * 128)
    ok, buf = cv2.imencode(".png", img8)
    assert ok
    with pytest.raises(MergeError, match="16-bit"):
        decode_frame_png(base64.b64encode(buf).decode())


# ---------------- non-tile-multiple sizes (50x64 with 8 px tiles) ----------


def test_tile_layout_partial_edge_tiles():
    from app.registration import _pick_layout, _tile_capacities, _tile_ids

    tile, stride = _pick_layout(50, 64)
    assert tile == 8 and stride == 1
    ids, nh, nw = _tile_ids(50, 64, tile)
    # ceil grid: 7 rows of 8 px + one partial row of 2 px; 8 px columns.
    assert (nh, nw) == (7, 8)
    assert ids.shape == (50, 64)
    # No -1 padding: every in-frame pixel belongs to a real tile, ids are
    # contiguous and the last row tiles are exactly the partial ones.
    assert ids.min() == 0 and ids.max() == nh * nw - 1
    assert set(np.unique(ids).tolist()) == set(range(nh * nw))
    assert np.all(ids[48:50, :] == np.repeat(np.arange(48, 56), 8)[None, :])
    assert np.all(ids[:8, :8] == 0)
    # Capacities: 8x8 sampled interior tiles hold 64 stride-1 positions; the
    # 2x8 partial edge tiles hold only 16.
    cap = _tile_capacities(50, 64, nh, nw, tile, stride)
    assert cap.shape == (56,)
    assert np.all(cap[:48] == 64)
    assert np.all(cap[48:] == 16)
    # And capacity matches the actual stride-sampled positions of every tile.
    sampled = np.bincount(ids[::stride, ::stride].ravel(), minlength=nh * nw)
    assert np.array_equal(sampled, cap)


def test_nontile_multiple_registration_recovers_shifts():
    # The original bug: 50x64 with 8 px tiles leaves 2 rows padded with -1;
    # they wrapped into the last tile's texture stats and made np.bincount
    # raise an uncaught ValueError instead of registering or a 422 MergeError.
    shifts = [(3, -5), (0, 0), (-4, 6)]
    exposures = [0.25, 1.0, 2.0]
    frames = syn.make_frames_sized(50, 64, shifts, exposures, highlight=False)
    result, motion, unknown, got = merge_stack(
        frames, exposures, 1, 0.05
    )
    assert got == shifts
    assert motion.sum() == 0 and unknown.sum() == 0
    bg = syn.background_at_sized(50, 64, *np.indices((50, 64)))
    err = np.abs(result - bg.astype(np.float32)).max(axis=2)
    assert np.median(err) < 0.01


def test_nontile_multiple_with_mover_and_border_shifts():
    shifts = [(8, -8), (0, 0), (-8, 8)]
    exposures = [0.25, 1.0, 2.0]
    rects = [
        (8, 10, 18, 22),
        (8, 10, 18, 22),
        (26, 36, 36, 48),
    ]
    frames = syn.make_frames_sized(
        50, 63, shifts, exposures, rects=rects, highlight=False
    )
    _, motion, unknown, got = merge_stack(frames, exposures, 1, 0.05)
    assert got == shifts  # +/-8 boundary shifts on both non-multiple axes
    # Frame 2's mover in its own coords maps to ref (18..27, 44..55); the
    # clipped columns 55..62 miss frame 2, so only the in-frame core is motion.
    assert motion[20:26, 45:53].mean() > 0.9
    assert motion[9:16, 11:20].mean() > 0.9
    assert unknown.sum() == 0


def test_partial_edge_rows_anchor_registration():
    """Texture living only in the 2-row partial edge band must register.

    With the old -1 padding those rows were excluded (and corrupted the last
    full tiles); here they are the *only* textured rows, so registration can
    succeed only if the partial edge tiles participate on equal footing.
    """
    h, w = 50, 64
    m = 8
    # Positive shifts keep the bottom edge band visible in every frame's
    # viewport; a negative shift would geometrically miss it.
    shifts = [(6, 0), (4, 0), (0, 0)]
    exposures = [0.3, 0.4, 0.25]
    # One fixed world-space stripe shared by every frame (rng outside loop).
    stripe = 0.25 + 0.35 * np.random.default_rng(3).random(w)
    frames = []
    for dy, exp in zip([s[0] for s in shifts], exposures):
        big = np.full((h + 2 * m, w + 2 * m, 3), 0.30, np.float64)
        big[h - 2 + m : h + m, m : m + w, :] = stripe[None, :, None]
        view = big[m + dy : m + dy + h, m : m + w]
        frames.append(
            np.clip(np.rint(view * exp * 65535.0), 1, 65534).astype(np.uint16)
        )
    _, motion, unknown, got = merge_stack(frames, exposures, 2, 0.02)
    assert got == shifts
    assert motion.sum() == 0 and unknown.sum() == 0


def test_nontile_multiple_garbage_frame_returns_merge_error():
    # An unrelated frame must surface as MergeError (HTTP 422), never the raw
    # np.bincount ValueError the -1 padding used to trigger.
    shifts = [(3, -5), (0, 0), (-4, 6)]
    frames = syn.make_frames_sized(
        50, 64, shifts, [0.25, 1.0, 2.0], highlight=False
    )
    rng = np.random.default_rng(11)
    frames[0] = rng.integers(100, 60000, (50, 64, 3), np.uint16)
    with pytest.raises(MergeError, match="registration failed"):
        merge_stack(frames, [0.25, 1.0, 2.0], 1, 0.05)


def test_nontile_multiple_flat_frames_returns_merge_error():
    flat = [np.full((50, 64, 3), 30000, np.uint16) for _ in range(3)]
    with pytest.raises(MergeError, match="texture|registration"):
        merge_stack(flat, [0.25, 1.0, 2.0], 1, 0.05)


# ----------------------------- HTTP layer --------------------------------


client = TestClient(app)


def _post(payload):
    return client.post("/merges", json=payload)


def test_http_merge_ok_decodes_roundtrip():
    payload = syn.request_payload()
    resp = _post(payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == {
        "radiance_tiff_base64",
        "motion_mask_base64",
        "unknown_mask_base64",
        "shifts",
        "width",
        "height",
        "reference_index",
    }
    assert data["shifts"] == [[3, -5], [0, 0], [-4, 6]]
    assert data["width"] == syn.W and data["height"] == syn.H

    tiff = base64.b64decode(data["radiance_tiff_base64"])
    arr = cv2.imdecode(np.frombuffer(tiff, np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr is not None
    assert arr.dtype == np.float32 and arr.shape == (syn.H, syn.W, 3)

    motion_png = cv2.imdecode(
        np.frombuffer(base64.b64decode(data["motion_mask_base64"]), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert set(np.unique(motion_png)).issubset({0, 255})
    assert (motion_png > 0).sum() > 0


def test_http_nontile_multiple_merge_ok():
    # 50x64 PNGs (8 px tiles leave a 2-row partial edge band) must merge end
    # to end through the HTTP layer instead of 500-ing on np.bincount.
    payload = syn.request_payload_sized(
        50,
        64,
        shifts=[(3, -5), (0, 0), (-4, 6)],
        exposures=[0.25, 1.0, 2.0],
        reference_index=1,
        highlight=False,
    )
    resp = _post(payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["shifts"] == [[3, -5], [0, 0], [-4, 6]]
    assert data["width"] == 64 and data["height"] == 50

    tiff = base64.b64decode(data["radiance_tiff_base64"])
    arr = cv2.imdecode(np.frombuffer(tiff, np.uint8), cv2.IMREAD_UNCHANGED)
    assert arr is not None
    assert arr.dtype == np.float32 and arr.shape == (50, 64, 3)
    assert np.isfinite(arr).all()


def test_http_invalid_exposure_returns_422():
    payload = syn.request_payload()
    payload["exposure_seconds"][0] = 0.0
    resp = _post(payload)
    assert resp.status_code == 422
    assert "error" in resp.json()


def test_http_nan_tolerance_returns_422():
    payload = syn.request_payload()
    payload["tolerance"] = float("nan")
    # A real (lenient) client serializes NaN as the bare token NaN.
    body = json.dumps(payload, allow_nan=True)
    assert "NaN" in body
    resp = client.post(
        "/merges", content=body, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422


def test_http_too_few_frames_returns_422():
    payload = syn.request_payload()
    payload["frames"] = payload["frames"][:2]
    payload["exposure_seconds"] = payload["exposure_seconds"][:2]
    resp = _post(payload)
    assert resp.status_code == 422


def test_http_registration_failure_returns_422():
    payload = syn.request_payload()
    rng = np.random.default_rng(1)
    garbage = rng.integers(100, 60000, (syn.H, syn.W, 3), np.uint16)
    payload["frames"][0]["image_base64"] = syn.encode_png_b64(garbage)
    resp = _post(payload)
    assert resp.status_code == 422
    assert "registration failed" in resp.json()["error"]


def test_health():
    assert client.get("/health").status_code == 200
