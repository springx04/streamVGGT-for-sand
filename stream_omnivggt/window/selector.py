"""Active local-window selection for OmniVGGT black-box inference."""

from __future__ import annotations

from typing import Any
import math

from stream_omnivggt.types import ChangeMaskResult, InputPacket, OmniPrediction, SelectedWindow, WindowFrame
from stream_omnivggt.window.keyframes import make_window_frame


def select_active_window(
    curr_packet: InputPacket,
    frame_history: list[WindowFrame],
    keyframes: list[WindowFrame],
    affected_blocks: set[tuple[int, int, int]],
    cfg: dict[str, Any],
) -> SelectedWindow:
    """Select a bucketed local context window with the current frame included.

    Complexity:
        O(N log N), where N is the number of retained history/keyframe entries.
        N is bounded by configuration and never includes all stream history.
    """

    allowed = tuple(cfg.get("allowed_buckets", (3, 4, 6, 8)))
    target_len = int(cfg.get("target_window_len", cfg.get("default_window", 4)))
    bucket_len = min((b for b in allowed if b >= target_len), default=max(allowed))
    bucket_len = bucket_len if bucket_len in allowed else min(allowed, key=lambda b: abs(b - target_len))

    has_anchor = any(f.is_anchor for f in keyframes) or any(f.is_anchor for f in frame_history)
    curr_is_anchor = bool(curr_packet.meta.get("force_anchor", False) or not has_anchor)
    curr_frame = make_window_frame(curr_packet, is_anchor=curr_is_anchor)

    candidates = _dedupe_frames([*keyframes, *frame_history])
    for frame in candidates:
        frame.priority_score = score_frame_for_window(frame, affected_blocks, curr_packet.timestamp, cfg)

    anchors = [f for f in candidates if f.is_anchor]
    camera_anchors = [f for f in anchors if f.has_camera]
    selected: list[WindowFrame] = []
    first_anchor = None
    if camera_anchors:
        first_anchor = max(camera_anchors, key=lambda f: f.priority_score)
    elif anchors:
        first_anchor = max(anchors, key=lambda f: f.priority_score)
    elif curr_frame.is_anchor:
        first_anchor = curr_frame

    if first_anchor is not None:
        selected.append(first_anchor)

    scored = sorted(candidates, key=lambda f: f.priority_score, reverse=True)
    for frame in scored:
        if len(selected) >= bucket_len - 1:
            break
        if frame.frame_id == curr_frame.frame_id or any(f.frame_id == frame.frame_id for f in selected):
            continue
        selected.append(frame)

    if not any(f.frame_id == curr_frame.frame_id for f in selected):
        selected.append(curr_frame)

    if not any(f.is_anchor for f in selected):
        curr_frame.is_anchor = True
    while len(selected) < bucket_len:
        selected.insert(0, first_anchor or curr_frame)
    if len(selected) > bucket_len:
        selected = selected[: bucket_len - 1] + [curr_frame]

    if any(f.has_camera for f in selected):
        camera_anchor_positions = [i for i, f in enumerate(selected) if f.is_anchor and f.has_camera]
        if camera_anchor_positions and camera_anchor_positions[0] != 0:
            anchor = selected.pop(camera_anchor_positions[0])
            selected.insert(0, anchor)

    camera_gt_index = [idx for idx, frame in enumerate(selected) if frame.has_camera]
    depth_gt_index = [idx for idx, frame in enumerate(selected) if frame.has_depth]
    reason = f"bucket={bucket_len};affected_blocks={len(affected_blocks)};current={curr_packet.frame_id}"
    return SelectedWindow(selected, camera_gt_index, depth_gt_index, f"S{bucket_len}", reason)


def score_frame_for_window(
    frame: WindowFrame,
    affected_blocks: set[tuple[int, int, int]],
    now_ts: float,
    cfg: dict[str, Any],
) -> float:
    """Score a history/keyframe candidate using recency, overlap, and geometry."""

    tau = max(float(cfg.get("recency_tau_sec", 4.0)), 1e-6)
    age = max(now_ts - frame.packet.timestamp, 0.0)
    recency = math.exp(-age / tau)
    frame_blocks = set(frame.packet.meta.get("block_keys", set()))
    overlap = len(frame_blocks & affected_blocks) / max(len(affected_blocks), 1) if affected_blocks else 0.0
    mean_conf = float(frame.packet.meta.get("mean_conf", 0.5))
    score = recency
    score += float(cfg.get("overlap_bonus", 1.0)) * overlap
    score += float(cfg.get("camera_bonus", 1.0)) if frame.has_camera else 0.0
    score += float(cfg.get("depth_bonus", 0.4)) if frame.has_depth else 0.0
    score += float(cfg.get("anchor_bonus", 2.0)) if frame.is_anchor else 0.0
    score += float(cfg.get("mean_conf_bonus", 0.5)) * mean_conf
    return float(score)


def maybe_promote_keyframe(
    packet: InputPacket,
    change: ChangeMaskResult,
    pred: OmniPrediction | None,
    cfg: dict[str, Any],
) -> bool:
    """Decide whether the current packet should become a protected keyframe."""

    if packet.frame_id == 0 or packet.meta.get("force_anchor", False):
        return True
    if change.changed_ratio >= float(cfg.get("scene_jump_ratio", 0.35)):
        return True
    if packet.intrinsic is not None and packet.extrinsic_c2w is not None and change.changed_ratio >= float(cfg.get("small_change_ratio", 0.02)):
        return True
    if pred is not None and packet.depth is not None and change.changed_ratio >= 0.5 * float(cfg.get("small_change_ratio", 0.02)):
        return True
    return False


def _dedupe_frames(frames: list[WindowFrame]) -> list[WindowFrame]:
    """Keep the newest instance for each frame_id while preserving order."""

    by_id: dict[int, WindowFrame] = {}
    for frame in frames:
        by_id[frame.frame_id] = frame
    return sorted(by_id.values(), key=lambda f: f.packet.timestamp)

