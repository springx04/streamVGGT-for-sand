"""Change-mask computation for affected pixel and block detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from stream_omnivggt.detect.optical_flow import compute_dense_farneback_flow, compute_sparse_lk_flow
from stream_omnivggt.types import ChangeMaskResult


def compute_change_mask(
    curr_rgb: np.ndarray,
    prev_rgb: np.ndarray | None,
    curr_depth: np.ndarray | None,
    reproj_rgb: np.ndarray | None,
    reproj_depth: np.ndarray | None,
    flow_mode: str,
    conf_map: np.ndarray | None,
    thresholds: dict[str, Any],
) -> ChangeMaskResult:
    """Compute a dense score map from image, depth, flow, and confidence terms.

    Complexity:
        O(H * W) for image/depth/confidence residuals. Dense Farneback flow is
        also linear in pixel count up to a method-dependent constant.
    """

    curr = _rgb_float(curr_rgb)
    height, width = curr.shape[:2]
    reference_rgb = reproj_rgb if reproj_rgb is not None else prev_rgb
    if reference_rgb is None:
        mask = np.ones((height, width), dtype=bool)
        score_map = np.ones((height, width), dtype=np.float32)
        return ChangeMaskResult(mask, score_map, 1.0, int(mask.size), 1.0, 0.0, _mask_to_blocks(mask, thresholds))

    ref = _resize_like(_rgb_float(reference_rgb), (height, width), is_depth=False)
    image_res = np.mean(np.abs(curr - ref), axis=2).astype(np.float32)

    depth_res = np.zeros((height, width), dtype=np.float32)
    depth_error_mean = 0.0
    if curr_depth is not None and reproj_depth is not None:
        d_curr = _resize_like(np.asarray(curr_depth, dtype=np.float32), (height, width), is_depth=True)
        d_ref = _resize_like(np.asarray(reproj_depth, dtype=np.float32), (height, width), is_depth=True)
        valid = (d_curr > 1e-6) & (d_ref > 1e-6)
        depth_res[valid] = np.abs(d_curr[valid] - d_ref[valid]) / np.maximum(np.abs(d_ref[valid]), 1e-6)
        depth_error_mean = float(depth_res[valid].mean()) if valid.any() else 0.0

    flow_mag = np.zeros((height, width), dtype=np.float32)
    if prev_rgb is not None and flow_mode != "none":
        prev = _resize_like(_rgb_float(prev_rgb), (height, width), is_depth=False)
        prev_gray = prev.mean(axis=2)
        curr_gray = curr.mean(axis=2)
        if flow_mode == "farneback":
            flow = compute_dense_farneback_flow(prev_gray, curr_gray)
            flow_mag = np.linalg.norm(flow, axis=2).astype(np.float32)
        elif flow_mode == "lk":
            sparse = compute_sparse_lk_flow(prev_gray, curr_gray)
            mean_shift = 0.0
            if int(sparse["count"]) > 0:
                mean_shift = float(np.linalg.norm(np.asarray(sparse["curr_pts"]) - np.asarray(sparse["prev_pts"]), axis=1).mean())
            flow_mag.fill(mean_shift)

    conf_penalty = np.zeros((height, width), dtype=np.float32)
    if conf_map is not None:
        conf = _resize_like(np.asarray(conf_map, dtype=np.float32), (height, width), is_depth=True)
        conf_penalty = np.clip(1.0 - conf, 0.0, 1.0).astype(np.float32)

    score_map = (
        float(thresholds.get("lambda_image", 1.0)) * image_res
        + float(thresholds.get("lambda_depth", 1.0)) * depth_res
        + float(thresholds.get("lambda_flow", 0.25)) * (flow_mag / max(float(thresholds.get("flow_thr_px", 2.0)), 1e-6))
        + float(thresholds.get("lambda_conf", 0.25)) * conf_penalty
    ).astype(np.float32)

    mask = (
        (image_res > float(thresholds.get("image_l1_thr", 12.0 / 255.0)))
        | (depth_res > float(thresholds.get("depth_rel_thr", 0.03)))
        | (flow_mag > float(thresholds.get("flow_thr_px", 2.0)))
    )
    if conf_map is not None:
        conf = 1.0 - conf_penalty
        mask |= conf < float(thresholds.get("low_conf_thr", 0.2))
    mask = dilate_change_mask(mask, int(thresholds.get("dilate_ksize", 3)))
    changed_pixels = int(mask.sum())
    changed_ratio = float(changed_pixels / max(mask.size, 1))
    return ChangeMaskResult(
        mask=mask,
        score_map=score_map,
        changed_ratio=changed_ratio,
        changed_pixels=changed_pixels,
        reprojection_error_mean=float(image_res.mean()),
        depth_error_mean=depth_error_mean,
        blocks_hint=_mask_to_blocks(mask, thresholds),
    )


def dilate_change_mask(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Dilate a boolean mask with a square kernel using a small NumPy max filter."""

    src = np.asarray(mask, dtype=bool)
    if ksize <= 1 or not src.any():
        return src.copy()
    radius = ksize // 2
    padded = np.pad(src, radius, mode="constant", constant_values=False)
    out = np.zeros_like(src, dtype=bool)
    for dy in range(ksize):
        for dx in range(ksize):
            out |= padded[dy : dy + src.shape[0], dx : dx + src.shape[1]]
    return out


def _mask_to_blocks(mask: np.ndarray, thresholds: dict[str, Any]) -> set[tuple[int, int, int]]:
    """Project changed pixels to coarse image-space block hints."""

    block_px = max(int(thresholds.get("block_pixels", 16)), 1)
    ys, xs = np.nonzero(mask)
    return {(int(x // block_px), int(y // block_px), 0) for y, x in zip(ys, xs)}


def _rgb_float(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB input to HxWx3 float32 in [0, 1]."""

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3, got {arr.shape}")
    out = arr.astype(np.float32, copy=False)
    if out.max(initial=0.0) > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def _resize_like(arr: np.ndarray, target_hw: tuple[int, int], is_depth: bool) -> np.ndarray:
    """Resize an array to HxW using PIL via the preprocess helper."""

    if arr.shape[:2] == target_hw:
        return arr.astype(np.float32, copy=False)
    from stream_omnivggt.preprocess.reshape import _resize_array

    height, width = target_hw
    return _resize_array(arr, (width, height), is_depth=is_depth).astype(np.float32, copy=False)

