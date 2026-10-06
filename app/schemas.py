"""Pydantic request / response models for POST /merges."""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class FrameIn(BaseModel):
    image_base64: str = Field(..., description="Base64 编码的 16 位线性 RGB PNG")


class MergeRequest(BaseModel):
    frames: list[FrameIn] = Field(
        ...,
        min_length=3,
        max_length=9,
        description="3~9 张同尺寸 16 位线性 RGB PNG（Base64）",
    )
    exposure_seconds: list[float] = Field(
        ...,
        min_length=3,
        max_length=9,
        description="每张帧的曝光秒数（有限正数），与 frames 一一对应",
    )
    reference_index: int = Field(
        ..., ge=0, le=8, description="参考帧序号（从 0 开始）"
    )
    tolerance: float = Field(
        ..., description="辐亮度容差（有限非负数，单位与 exposure 归一化后的辐亮度相同）"
    )

    @field_validator("frames", "exposure_seconds")
    @classmethod
    def _not_empty(cls, v: list) -> list:
        if len(v) < 3 or len(v) > 9:
            raise ValueError("需要 3~9 张帧")
        return v

    @field_validator("exposure_seconds")
    @classmethod
    def _exposures_valid(cls, v: list[float]) -> list[float]:
        for i, e in enumerate(v):
            if not isinstance(e, (int, float)) or not math.isfinite(e) or e <= 0:
                raise ValueError(f"第 {i} 张帧的曝光秒数必须为有限正数")
        return v

    @field_validator("tolerance")
    @classmethod
    def _tolerance_valid(cls, v: float) -> float:
        # NaN compares False against every bound, so check finiteness first.
        if not math.isfinite(v):
            raise ValueError("辐亮度容差必须为有限非负数")
        if v < 0:
            raise ValueError("辐亮度容差必须为非负数")
        return v


class MergeResponse(BaseModel):
    radiance_tiff_base64: str = Field(
        ...,
        description=(
            "Base64 float32 RGB TIFF；未知像素为 NaN。"
            "像素值 = 有效数字值 / 65535 / 曝光秒数，静态区为有效观测均值，"
            "运动区取参考帧观测。"
        ),
    )
    motion_mask_base64: str = Field(
        ...,
        description="Base64 8 位灰度 PNG，运动像素=255，其余=0",
    )
    unknown_mask_base64: str = Field(
        ...,
        description="Base64 8 位灰度 PNG，未知像素=255，其余=0（未知可与运动重叠）",
    )
    shifts: list[list[int]] = Field(
        ...,
        description=(
            "每帧相对参考帧的整数平移 [dy, dx]："
            "帧 i 的源像素 (y, x) 对应参考坐标 (y + dy, x + dx)，"
            "参考帧为 [0, 0]；各轴绝对值不超过 8"
        ),
    )
    width: int
    height: int
    reference_index: int
