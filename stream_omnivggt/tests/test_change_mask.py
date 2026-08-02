"""Tests for dense change-mask behavior."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.detect import compute_change_mask


def _thresholds() -> dict[str, float | int]:
    """Return deterministic test thresholds."""

    return {
        "image_l1_thr": 12.0 / 255.0,
        "depth_rel_thr": 0.03,
        "flow_thr_px": 2.0,
        "low_conf_thr": 0.2,
        "lambda_image": 1.0,
        "lambda_depth": 1.0,
        "lambda_flow": 0.0,
        "lambda_conf": 0.0,
        "dilate_ksize": 3,
        "block_pixels": 16,
    }


def test_no_change_ratio_is_near_zero() -> None:
    """Unchanged images should produce almost no changed pixels."""

    rgb = np.full((56, 56, 3), 100, dtype=np.uint8)
    result = compute_change_mask(rgb, rgb.copy(), None, None, None, "none", None, _thresholds())
    assert result.changed_ratio < 0.01
    assert result.changed_pixels == 0


def test_small_patch_change_is_local() -> None:
    """A small image patch should not mark the whole frame as changed."""

    prev = np.full((56, 56, 3), 100, dtype=np.uint8)
    curr = prev.copy()
    curr[20:26, 20:26] = 240
    result = compute_change_mask(curr, prev, None, None, None, "none", None, _thresholds())
    assert 0.0 < result.changed_ratio < 0.1
    assert len(result.blocks_hint) <= 4


def test_degenerate_without_depth_or_reprojection_runs() -> None:
    """The detector should degrade to image residuals without geometry inputs."""

    prev = np.zeros((56, 56, 3), dtype=np.uint8)
    curr = prev.copy()
    curr[10:14, 10:14, 0] = 255
    result = compute_change_mask(curr, prev, None, None, None, "none", None, _thresholds())
    assert result.mask.shape == (56, 56)
    assert result.score_map.dtype == np.float32
    assert result.changed_ratio > 0.0

