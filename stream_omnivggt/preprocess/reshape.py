"""Image, depth, and intrinsic reshaping for OmniVGGT shape buckets."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def resize_to_bucket(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    intrinsic: np.ndarray | None,
    target_width: int,
    target_size: int,
    patch_multiple: int = 14,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Resize width to a fixed bucket and crop height to the target size.

    Height is scaled proportionally, rounded to a multiple of ``patch_multiple``,
    and center-cropped if it exceeds ``target_size``. Intrinsics are scaled and
    shifted to match the resized/cropped image. Complexity is O(H * W).
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected rgb HxWx3, got {rgb.shape}")
    src_h, src_w = rgb.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Input image must have positive height and width.")

    scale = float(target_width) / float(src_w)
    scaled_h = max(patch_multiple, int(round((src_h * scale) / patch_multiple) * patch_multiple))
    scaled_w = int(target_width)
    resized_rgb = _resize_array(rgb, (scaled_w, scaled_h), is_depth=False)
    resized_depth = _resize_array(depth, (scaled_w, scaled_h), is_depth=True) if depth is not None else None

    crop_top = 0
    crop_bottom = scaled_h
    if scaled_h > target_size:
        crop_top = (scaled_h - target_size) // 2
        crop_bottom = crop_top + target_size
        resized_rgb = resized_rgb[crop_top:crop_bottom]
        if resized_depth is not None:
            resized_depth = resized_depth[crop_top:crop_bottom]

    new_intrinsic = None
    if intrinsic is not None:
        new_intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        new_intrinsic[0, :] *= scale
        new_intrinsic[1, :] *= scale
        new_intrinsic[1, 2] -= crop_top

    meta = {
        "source_hw": (src_h, src_w),
        "resized_hw": (scaled_h, scaled_w),
        "output_hw": resized_rgb.shape[:2],
        "scale": scale,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "patch_multiple": patch_multiple,
    }
    return resized_rgb, resized_depth, new_intrinsic, meta


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8/float RGB to a CxHxW float32 tensor in [0, 1]."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected rgb HxWx3, got {rgb.shape}")
    arr = rgb.astype(np.float32, copy=False)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()


def _resize_array(arr: np.ndarray | None, size_wh: tuple[int, int], is_depth: bool) -> np.ndarray:
    """Resize an image-like array with PIL using bilinear or nearest sampling."""

    if arr is None:
        raise ValueError("Cannot resize None array.")
    width, height = size_wh
    source = np.asarray(arr)
    if source.ndim == 2:
        pil_mode = "F" if np.issubdtype(source.dtype, np.floating) else None
        image = Image.fromarray(source.astype(np.float32) if pil_mode == "F" else source)
    else:
        image = Image.fromarray(source.astype(np.uint8) if source.dtype != np.uint8 else source)
    resample = Image.Resampling.NEAREST if is_depth else Image.Resampling.BILINEAR
    resized = np.asarray(image.resize((width, height), resample=resample))
    if is_depth:
        return resized.astype(np.float32, copy=False)
    if source.dtype == np.uint8:
        return resized.astype(np.uint8, copy=False)
    return resized.astype(np.float32, copy=False)

