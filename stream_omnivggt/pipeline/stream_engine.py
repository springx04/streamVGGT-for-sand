"""Main streaming engine for no-training OmniVGGT wrapping."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
import logging
import pickle

import numpy as np
import torch

from stream_omnivggt.align import run_height_align, run_rotation_align
from stream_omnivggt.backend.base import BaseOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.detect import compute_change_mask
from stream_omnivggt.map.hybrid_map import HybridMap
from stream_omnivggt.pipeline.fallback import (
    handle_dropped_frame,
    recover_from_alignment_failure,
    should_start_new_segment,
    should_trigger_full_refresh,
)
from stream_omnivggt.pipeline.scheduler import SerialScheduler, ThreadedScheduler
from stream_omnivggt.preprocess import convert_camera_c2w_to_w2c, default_intrinsic, normalize_rgb, resize_to_bucket
from stream_omnivggt.types import InputPacket, OmniPrediction, SelectedWindow, StreamMetrics, WindowFrame
from stream_omnivggt.window import make_window_frame, maybe_promote_keyframe, select_active_window, trim_keyframes

logger = logging.getLogger(__name__)


class StreamEngine:
    """Incremental stream engine that treats OmniVGGT as a local-window black box."""

    def __init__(self, backend: BaseOmniBackend, cfg: StreamConfig) -> None:
        """Create the stream engine and warm the backend bucket shapes."""

        self.backend = backend
        self.cfg = cfg
        self.state: dict[str, Any] = {"segments": [], "segment_id": 0}
        self.frame_history: list[WindowFrame] = []
        self.keyframes: list[WindowFrame] = []
        self.map = HybridMap(cfg)
        self.scheduler = ThreadedScheduler() if cfg.benchmark.async_render else SerialScheduler()
        self.backend.warmup(list(cfg.omni.warmup_buckets))

    def push(self, packet: InputPacket) -> StreamMetrics:
        """Ingest one frame and incrementally update the global map.

        Complexity:
            O(H * W + P_changed + B_touched) plus one black-box model call when
            the frame is not skipped. No full history is sent into the model.
        """

        total_t0 = perf_counter()
        metrics = StreamMetrics()
        fallback_reason: str | None = None

        t0 = perf_counter()
        dropped = self._is_dropped_frame(packet)
        if dropped:
            handle_dropped_frame(packet, self.state, asdict(self.cfg.fallback))
            fallback_reason = "dropped_frame"
        metrics.ingest_ms = _elapsed_ms(t0)

        t0 = perf_counter()
        aligned_packet, align_quality, align_failed = self._align_packet(packet)
        if align_failed:
            fallback_reason = "alignment_failure"
            recover_from_alignment_failure(packet, self.state, asdict(self.cfg.fallback))
        metrics.align_ms = _elapsed_ms(t0)

        t0 = perf_counter()
        proc_packet = self._preprocess_packet(aligned_packet)
        prev_rgb = None if dropped else (self.frame_history[-1].packet.rgb if self.frame_history else None)
        flow_mode = "none" if self.state.pop("disable_flow_once", False) else self.cfg.change.flow_mode
        change = compute_change_mask(
            curr_rgb=_rgb_np(proc_packet.rgb),
            prev_rgb=_rgb_np(prev_rgb) if prev_rgb is not None else None,
            curr_depth=_depth_np(proc_packet.depth),
            reproj_rgb=None,
            reproj_depth=None,
            flow_mode=flow_mode,
            conf_map=self.state.get("last_conf_map"),
            thresholds=self.cfg.flat_thresholds(),
        )
        strategy = self.cfg.benchmark.strategy
        if strategy == "full_rebuild":
            change.mask[:, :] = True
            change.changed_pixels = int(change.mask.size)
            change.changed_ratio = 1.0
            change.blocks_hint = set(change.blocks_hint) or {(0, 0, 0)}
            fallback_reason = fallback_reason or "full_rebuild_strategy"
        full_refresh = should_trigger_full_refresh(change, {**asdict(self.cfg.change), **asdict(self.cfg.fallback)})
        if full_refresh:
            fallback_reason = fallback_reason or "full_refresh"
        if should_start_new_segment(1.0 - change.changed_ratio, align_quality, asdict(self.cfg.fallback)) and self.frame_history:
            self._start_new_segment(proc_packet, fallback_reason or "scene_jump_or_low_quality")
            fallback_reason = fallback_reason or "new_segment"
        metrics.diff_ms = _elapsed_ms(t0)

        t0 = perf_counter()
        window_cfg = self._window_cfg(change.changed_ratio, full_refresh)
        selected = select_active_window(proc_packet, self.frame_history, self.keyframes, change.blocks_hint, window_cfg)
        self.state["last_selected_window"] = selected
        metrics.select_ms = _elapsed_ms(t0)

        skip_model = self._should_skip_model(change, full_refresh, strategy)
        pred: OmniPrediction | None = None
        updated_blocks: set[tuple[int, int, int]] = set()
        if skip_model:
            metrics.skipped_model = True
            fallback_reason = fallback_reason or "no_change_skip"
        else:
            t0 = perf_counter()
            batch = self._build_backend_batch(selected)
            pred = self.backend.run_window(batch)
            metrics.model_ms = _elapsed_ms(t0)

            t0 = perf_counter()
            updated_blocks = self.map.fuse_prediction(pred, selected, change, self._map_cfg(), proc_packet.timestamp)
            metrics.project_ms = 0.0
            metrics.fuse_ms = _elapsed_ms(t0)

            t0 = perf_counter()
            committed = self.map.commit_dirty()
            metrics.commit_ms = _elapsed_ms(t0)
            logger.debug("Committed %d dirty blocks.", committed)
            self._update_last_conf(pred, selected)

        self._update_frame_state(proc_packet, change, pred)
        metrics.updated_block_count = len(updated_blocks)
        metrics.updated_point_ratio = float(self.map.last_updated_point_ratio if pred is not None else 0.0)
        metrics.fallback_reason = fallback_reason
        metrics.total_ms = _elapsed_ms(total_t0)
        return metrics

    def flush(self) -> None:
        """Wait for background work and shut down the scheduler."""

        self.scheduler.shutdown(wait=True)

    def snapshot(self, path: str) -> None:
        """Persist engine state without exporting heavy visualizations."""

        self._write_snapshot(Path(path))

    def load_snapshot(self, path: str) -> None:
        """Restore engine state from a snapshot created by snapshot()."""

        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        self.state = state.get("state", {})
        self.frame_history = state.get("frame_history", [])
        self.keyframes = state.get("keyframes", [])
        self.map.load_state(state.get("map", {}))

    def _align_packet(self, packet: InputPacket) -> tuple[InputPacket, float, bool]:
        """Run rotation and height alignment around the packet pose."""

        rot = run_rotation_align(packet, self.state) if self.cfg.align.enable_rotation else None
        rot_packet = packet
        if rot is not None and rot.aligned_extrinsic_c2w is not None:
            rot_packet = InputPacket(packet.frame_id, packet.timestamp, packet.rgb, packet.depth, packet.intrinsic, rot.aligned_extrinsic_c2w, dict(packet.meta))
        height = run_height_align(rot_packet, self.state) if self.cfg.align.enable_height else None
        final_packet = rot_packet
        quality = 1.0
        failed = False
        if height is not None:
            quality = height.quality_score
            failed = bool(height.flags.get("failed", False))
            if height.aligned_extrinsic_c2w is not None:
                final_packet = InputPacket(packet.frame_id, packet.timestamp, packet.rgb, packet.depth, packet.intrinsic, height.aligned_extrinsic_c2w, dict(packet.meta))
        if rot is not None:
            quality = min(quality, rot.quality_score)
            failed = failed or bool(rot.flags.get("failed", False))
        return final_packet, quality, failed or quality < self.cfg.align.min_quality

    def _preprocess_packet(self, packet: InputPacket) -> InputPacket:
        """Resize packet image/depth and adjust intrinsics for bucketed inference."""

        rgb, depth, intrinsic, meta = resize_to_bucket(
            _rgb_np(packet.rgb),
            _depth_np(packet.depth),
            packet.intrinsic,
            self.cfg.omni.target_width,
            self.cfg.omni.target_size,
            self.cfg.omni.patch_multiple,
        )
        merged_meta = dict(packet.meta)
        merged_meta["preprocess"] = meta
        return InputPacket(packet.frame_id, packet.timestamp, rgb, depth, intrinsic, packet.extrinsic_c2w, merged_meta)

    def _build_backend_batch(self, selected: SelectedWindow) -> dict[str, Any]:
        """Build a static-shaped backend batch from a SelectedWindow."""

        images = torch.stack([normalize_rgb(_rgb_np(frame.packet.rgb)) for frame in selected.frames], dim=0)[None]
        _, s_count, _, height, width = images.shape
        depth_list = []
        mask_list = []
        intrinsics = []
        extrinsics = []
        for frame in selected.frames:
            depth = _depth_np(frame.packet.depth)
            if depth is None:
                depth_arr = np.zeros((height, width), dtype=np.float32)
                mask_arr = np.zeros((height, width), dtype=np.float32)
            else:
                depth_arr = np.asarray(depth, dtype=np.float32)
                if depth_arr.shape != (height, width):
                    from stream_omnivggt.preprocess.reshape import _resize_array

                    depth_arr = _resize_array(depth_arr, (width, height), is_depth=True)
                mask_arr = (depth_arr > 1e-6).astype(np.float32)
            depth_list.append(depth_arr[..., None])
            mask_list.append(mask_arr)
            intrinsics.append(frame.packet.intrinsic if frame.packet.intrinsic is not None else default_intrinsic(width, height))
            w2c = convert_camera_c2w_to_w2c(frame.packet.extrinsic_c2w)
            extrinsics.append(w2c if w2c is not None else np.eye(4, dtype=np.float32)[:3])
        return {
            "images": images.contiguous(),
            "depth": torch.from_numpy(np.stack(depth_list, axis=0)[None]).float(),
            "mask": torch.from_numpy(np.stack(mask_list, axis=0)[None]).float(),
            "intrinsics": torch.from_numpy(np.stack(intrinsics, axis=0)[None]).float(),
            "extrinsics": torch.from_numpy(np.stack(extrinsics, axis=0)[None]).float(),
            "camera_gt_index": selected.camera_gt_index,
            "depth_gt_index": selected.depth_gt_index,
            "selected_window": selected,
            "shape": (s_count, 3, height, width),
        }

    def _update_last_conf(self, pred: OmniPrediction, selected: SelectedWindow) -> None:
        """Cache current-frame confidence for the next change detector call."""

        conf = pred.world_points_conf
        arr = conf.detach().float().cpu().numpy() if isinstance(conf, torch.Tensor) else np.asarray(conf)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        idx = int(np.argmax([frame.packet.timestamp for frame in selected.frames]))
        if arr.ndim == 3 and idx < arr.shape[0]:
            self.state["last_conf_map"] = arr[idx].astype(np.float32)

    def _update_frame_state(self, packet: InputPacket, change: Any, pred: OmniPrediction | None) -> None:
        """Append history and promote keyframes when policy triggers."""

        promoted = maybe_promote_keyframe(packet, change, pred, {**asdict(self.cfg.change), **asdict(self.cfg.window)})
        frame = make_window_frame(packet, is_anchor=promoted)
        self.frame_history.append(frame)
        max_history = max(self.cfg.window.allowed_buckets) * 4
        self.frame_history = self.frame_history[-max_history:]
        if promoted:
            self.keyframes.append(make_window_frame(packet, is_anchor=True))
            self.keyframes = trim_keyframes(self.keyframes, self.cfg.window.max_keyframes)

    def _window_cfg(self, change_ratio: float, full_refresh: bool) -> dict[str, Any]:
        """Build a flat selector config for the current change level."""

        cfg = asdict(self.cfg.window)
        cfg.update(asdict(self.cfg.change))
        if full_refresh or self.cfg.benchmark.strategy == "full_rebuild":
            cfg["target_window_len"] = self.cfg.window.refresh_window
        elif change_ratio > self.cfg.change.small_change_ratio:
            cfg["target_window_len"] = self.cfg.window.medium_window
        else:
            cfg["target_window_len"] = self.cfg.window.default_window
        return cfg

    def _map_cfg(self) -> dict[str, Any]:
        """Build a flat map/fusion config dictionary."""

        cfg = asdict(self.cfg.fuse)
        cfg.update(asdict(self.cfg.block))
        cfg.update(asdict(self.cfg.change))
        return cfg

    def _should_skip_model(self, change: Any, full_refresh: bool, strategy: str) -> bool:
        """Return True for no-change incremental frames after initialization."""

        if strategy == "full_rebuild" or full_refresh or not self.frame_history:
            return False
        return change.changed_ratio <= self.cfg.change.no_change_ratio

    def _is_dropped_frame(self, packet: InputPacket) -> bool:
        """Detect a large timestamp gap without assuming fixed FPS."""

        if not self.frame_history:
            return False
        last_ts = self.frame_history[-1].packet.timestamp
        return (packet.timestamp - last_ts) > self.cfg.fallback.dropped_frame_dt

    def _start_new_segment(self, packet: InputPacket, reason: str) -> None:
        """Record a read-only old segment marker without clearing the map."""

        self.state.setdefault("segments", []).append(
            {
                "segment_id": self.state.get("segment_id", 0),
                "closed_at_frame": packet.frame_id,
                "closed_at_ts": packet.timestamp,
                "reason": reason,
                "block_count": len(self.map.meta),
            }
        )
        self.state["segment_id"] = int(self.state.get("segment_id", 0)) + 1

    def _write_snapshot(self, path: Path) -> None:
        """Write a pickle snapshot to disk."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "state": self.state,
                    "frame_history": self.frame_history,
                    "keyframes": self.keyframes,
                    "map": self.map.export_state(),
                },
                handle,
            )


def _elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since a perf_counter start."""

    return (perf_counter() - start) * 1000.0


def _rgb_np(value: Any) -> np.ndarray:
    """Convert RGB input to HxWx3 NumPy."""

    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr
    return np.asarray(value)


def _depth_np(value: Any) -> np.ndarray | None:
    """Convert optional depth input to HxW NumPy."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32, copy=False)

