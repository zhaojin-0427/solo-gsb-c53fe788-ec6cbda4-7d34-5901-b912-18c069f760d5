"""FastAPI application exposing POST /merges (exposure-bracketing merge)."""
from __future__ import annotations

import json
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .errors import MergeError
from .imaging import (
    b64_bytes,
    decode_frame_png,
    encode_mask_png,
    encode_radiance_tiff,
)
from .merge import merge_stack
from .schemas import MergeRequest, MergeResponse


def _json_safe(obj):
    """Make validation errors strict-JSON safe.

    NaN/Infinity can occur in a raw JSON body, and ``ctx`` may contain Python
    exception objects; replace anything non-representable by its string form.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)

app = FastAPI(
    title="包围曝光合成 API",
    description=(
        "接收 3~9 张同尺寸 16 位线性 RGB PNG（Base64），按未剪切静态背景配准，"
        "输出 float32 RGB 辐亮度 TIFF、运动/未知掩膜 PNG 和各帧位移。"
    ),
    version="1.0.0",
)


@app.middleware("http")
async def permissive_json_loader(request: Request, call_next):
    """Accept (then reject at validation) NaN/Infinity tokens in JSON bodies.

    Starlette's default parser is strict-RFC JSON and raises 403-style errors
    before Pydantic sees the value; we parse with ``parse_constant`` so the
    schema validators produce the intended 422 response.
    """
    if (
        request.method == "POST"
        and request.headers.get("content-type", "").split(";")[0].strip()
        == "application/json"
    ):

        async def _json_loose():
            if not hasattr(request, "_loose_json"):
                body = await request.body()
                try:
                    request._loose_json = json.loads(
                        body,
                        parse_constant=lambda c: {
                            "NaN": math.nan,
                            "Infinity": math.inf,
                            "-Infinity": -math.inf,
                        }[c],
                    )
                except (json.JSONDecodeError, KeyError):
                    raise StarletteHTTPException(
                        status_code=422, detail="请求体不是合法的 JSON"
                    )
            return request._loose_json

        request.json = _json_loose  # type: ignore[method-assign]
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail)},
    )


@app.exception_handler(MergeError)
async def merge_error_handler(_request: Request, exc: MergeError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "请求参数不合法",
            "detail": _json_safe(exc.errors()),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/merges", response_model=MergeResponse)
async def post_merge(req: MergeRequest) -> MergeResponse:
    frames = [decode_frame_png(f.image_base64) for f in req.frames]

    radiance, motion, unknown, shifts = merge_stack(
        frames=frames,
        exposures=list(req.exposure_seconds),
        reference_index=req.reference_index,
        tolerance=req.tolerance,
    )

    h, w = motion.shape
    return MergeResponse(
        radiance_tiff_base64=b64_bytes(encode_radiance_tiff(radiance)),
        motion_mask_base64=b64_bytes(encode_mask_png(motion)),
        unknown_mask_base64=b64_bytes(encode_mask_png(unknown)),
        shifts=[[int(dy), int(dx)] for dy, dx in shifts],
        width=w,
        height=h,
        reference_index=req.reference_index,
    )
