"""Fallback and recovery policies for stream robustness."""

from __future__ import annotations

from typing import Any
import logging

from stream_omnivggt.types import ChangeMaskResult, InputPacket, SelectedWindow

logger = logging.getLogger(__name__)


def should_trigger_full_refresh(change: ChangeMaskResult, cfg: dict[str, Any]) -> bool:
    """Return True when the change ratio indicates a scene jump."""

    return change.changed_ratio >= float(cfg.get("scene_jump_ratio", 0.35))


def should_start_new_segment(overlap_ratio: float, align_quality: float, cfg: dict[str, Any]) -> bool:
    """Return True when overlap or alignment quality is too low for continuity."""

    return overlap_ratio < float(cfg.get("min_overlap_ratio", 0.1)) or align_quality < float(cfg.get("min_align_quality", 0.2))


def recover_from_alignment_failure(packet: InputPacket, state: dict[str, Any], cfg: dict[str, Any]) -> SelectedWindow | None:
    """Recover from alignment failure by reusing the last selected window if available."""

    logger.warning("Recovering from alignment failure on frame %s.", packet.frame_id)
    state["local_alignment_disabled"] = True
    window = state.get("last_selected_window")
    return window if isinstance(window, SelectedWindow) else None


def handle_dropped_frame(packet: InputPacket, state: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Mark a dropped-frame protection interval and disable short-term flow."""

    logger.warning("Dropped-frame protection triggered at frame %s.", packet.frame_id)
    state["last_dropped_frame_id"] = packet.frame_id
    state["disable_flow_once"] = True

