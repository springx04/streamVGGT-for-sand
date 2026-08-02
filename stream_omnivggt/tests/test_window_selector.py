"""Tests for active-window selection policy."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.types import InputPacket
from stream_omnivggt.window import make_window_frame, select_active_window


def _packet(frame_id: int, timestamp: float, camera: bool = False, depth: bool = False) -> InputPacket:
    """Create a small packet for selector tests."""

    rgb = np.zeros((56, 56, 3), dtype=np.uint8)
    intrinsic = np.eye(3, dtype=np.float32) if camera else None
    extrinsic = np.eye(4, dtype=np.float32) if camera else None
    depth_arr = np.ones((56, 56), dtype=np.float32) if depth else None
    return InputPacket(frame_id, timestamp, rgb, depth_arr, intrinsic, extrinsic, {})


def _cfg() -> dict[str, object]:
    """Return selector test config."""

    return {
        "allowed_buckets": (3, 4, 6, 8),
        "default_window": 4,
        "target_window_len": 4,
        "recency_tau_sec": 4.0,
        "camera_bonus": 1.0,
        "depth_bonus": 0.4,
        "anchor_bonus": 2.0,
        "overlap_bonus": 1.0,
        "mean_conf_bonus": 0.5,
    }


def test_current_frame_must_be_in_window() -> None:
    """The newest current packet must be selected."""

    history = [make_window_frame(_packet(0, 0.0), is_anchor=True)]
    selected = select_active_window(_packet(1, 0.1), history, history, set(), _cfg())
    assert 1 in [frame.frame_id for frame in selected.frames]


def test_at_least_one_anchor() -> None:
    """The selected window must include an anchor frame."""

    selected = select_active_window(_packet(0, 0.0), [], [], set(), _cfg())
    assert any(frame.is_anchor for frame in selected.frames)


def test_camera_anchor_is_preferred_first() -> None:
    """A camera-bearing anchor should be placed first when available."""

    anchor_no_cam = make_window_frame(_packet(0, 0.0), is_anchor=True)
    anchor_cam = make_window_frame(_packet(1, 0.1, camera=True), is_anchor=True)
    history = [anchor_no_cam, anchor_cam, make_window_frame(_packet(2, 0.2), is_anchor=False)]
    selected = select_active_window(_packet(3, 0.3), history, [anchor_no_cam, anchor_cam], set(), _cfg())
    assert selected.frames[0].frame_id == 1
    assert selected.frames[0].has_camera


def test_bucket_length_is_allowed() -> None:
    """The selected window length must be one of the allowed bucket sizes."""

    history = [make_window_frame(_packet(i, float(i) * 0.1), is_anchor=(i == 0)) for i in range(6)]
    cfg = _cfg()
    cfg["target_window_len"] = 6
    selected = select_active_window(_packet(7, 0.7), history, history[:1], set(), cfg)
    assert len(selected.frames) in {3, 4, 6, 8}
    assert selected.bucket_key in {"S3", "S4", "S6", "S8"}

