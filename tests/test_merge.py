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
