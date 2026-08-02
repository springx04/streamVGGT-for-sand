"""Keyframe helper functions."""

from __future__ import annotations

from stream_omnivggt.types import InputPacket, WindowFrame


def make_window_frame(packet: InputPacket, is_anchor: bool, priority_score: float = 0.0) -> WindowFrame:
    """Create a WindowFrame from an input packet."""

    return WindowFrame(
        frame_id=packet.frame_id,
        is_anchor=is_anchor,
        has_camera=packet.intrinsic is not None and packet.extrinsic_c2w is not None,
        has_depth=packet.depth is not None,
        priority_score=priority_score,
        packet=packet,
        cached_prediction_ref=packet.meta.get("cached_prediction_ref"),
    )


def trim_keyframes(keyframes: list[WindowFrame], max_keyframes: int) -> list[WindowFrame]:
    """Keep the newest keyframes while always preserving camera anchors first."""

    if len(keyframes) <= max_keyframes:
        return keyframes
    ordered = sorted(keyframes, key=lambda f: (not f.has_camera, f.packet.timestamp), reverse=False)
    protected = [f for f in ordered if f.has_camera][-(max_keyframes // 2) :]
    protected_ids = {id(frame) for frame in protected}
    remaining = [f for f in keyframes if id(f) not in protected_ids]
    newest = sorted(remaining, key=lambda f: f.packet.timestamp)[-(max_keyframes - len(protected)) :]
    return sorted(protected + newest, key=lambda f: f.packet.timestamp)

