"""Hybrid foreground surfel and background voxel map."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import logging

import numpy as np
import torch

from stream_omnivggt.config import StreamConfig
from stream_omnivggt.map.block_hash import points_to_block_keys
from stream_omnivggt.map.cold_store import ColdStore
from stream_omnivggt.map.surfel_map import SurfelFusionConfig, SurfelMap
from stream_omnivggt.map.tsdf_map import TsdfMap
from stream_omnivggt.types import BlockMeta, ChangeMaskResult, OmniPrediction, SelectedWindow

logger = logging.getLogger(__name__)


class HybridMap:
    """Persistent incremental map combining hot surfels, TSDF, and cold blocks."""

    def __init__(self, cfg: StreamConfig | None = None, cold_root: str | Path | None = None) -> None:
        """Create a hybrid map with NumPy fallback storage."""

        self.cfg = cfg or StreamConfig()
        self.surfels = SurfelMap(
            SurfelFusionConfig(
                w_max=self.cfg.fuse.w_max,
                merge_radius=self.cfg.fuse.merge_radius,
                lambda_age=self.cfg.fuse.lambda_age,
                occ_z_thr=self.cfg.fuse.occ_z_thr,
                replace_on_occlusion=self.cfg.fuse.replace_on_occlusion,
            )
        )
        self.tsdf = TsdfMap(w_max=self.cfg.fuse.w_max, voxel_size=self.cfg.block.voxel_size)
        root = cold_root or self.cfg.cache.cold_store_dir
        self.cold_store = (
            ColdStore(root, capacity=self.cfg.cache.memmap_capacity, max_points_per_block=self.cfg.cache.max_points_per_block)
            if self.cfg.cache.enable_memmap
            else None
        )
        self.meta: dict[tuple[int, int, int], BlockMeta] = {}
        self.last_updated_point_ratio: float = 0.0
        self.uses_open3d = self._try_open3d()

    def ensure_hot(self, block_keys: set[tuple[int, int, int]]) -> None:
        """Ensure block metadata and restore cold blocks into the hot map."""

        for key in block_keys:
            meta = self.meta.get(key)
            if meta is None and self.cold_store is not None:
                meta = self.cold_store.read_meta(key)
            if meta is None:
                meta = BlockMeta(key=key, last_update_ts=0.0, obs_count=0, mean_conf=0.0, is_anchor_block=False, is_hot=True, dirty=False)
            if not meta.is_hot and self.cold_store is not None:
                block = self.cold_store.read_block(key)
                if block is not None:
                    self.surfels.blocks[key] = block
            meta.is_hot = True
            self.meta[key] = meta

    def fuse_prediction(
        self,
        pred: OmniPrediction,
        selected_window: SelectedWindow,
        change_mask: ChangeMaskResult,
        cfg: dict[str, Any],
        timestamp: float,
    ) -> set[tuple[int, int, int]]:
        """Fuse only changed, confidence-gated points from the current frame.

        Complexity:
            O(P + B), where P is the selected frame's pixel count and B is the
            number of touched blocks.
        """

        current_idx = _current_frame_index(selected_window)
        images = np.stack([_packet_rgb_np(frame.packet.rgb) for frame in selected_window.frames], axis=0)
        extracted = extract_points_from_prediction(pred, str(cfg.get("point_mode", self.cfg.fuse.point_mode)), images)
        coords = extracted["image_coords"]
        current_mask = coords[:, 0] == current_idx if coords.shape[1] == 3 else np.ones((coords.shape[0],), dtype=bool)
        points = extracted["points_xyz"][current_mask]
        colors = extracted["colors"][current_mask]
        conf = extracted["conf"][current_mask]
        coords_current = coords[current_mask]
        if coords_current.shape[1] == 3:
            coords_current = coords_current[:, 1:3]
        changed = keep_only_changed_points(points, colors, conf, coords_current, change_mask.mask)
        conf_gate = gate_by_confidence(changed["conf"], float(cfg.get("min_conf", self.cfg.fuse.min_conf)))
        points = changed["points_xyz"][conf_gate]
        colors = changed["colors"][conf_gate]
        conf = changed["conf"][conf_gate]

        total_points = max(int(np.prod(change_mask.mask.shape)), 1)
        self.last_updated_point_ratio = float(points.shape[0] / total_points)
        if points.size == 0:
            return set()

        key_arr = points_to_block_keys(points, float(cfg.get("voxel_size", self.cfg.block.voxel_size)), int(cfg.get("block_resolution", self.cfg.block.block_resolution)))
        unique_keys = {tuple(map(int, row)) for row in np.unique(key_arr, axis=0)}
        self.ensure_hot(unique_keys)
        current_frame = selected_window.frames[current_idx]
        for key in unique_keys:
            block_mask = np.all(key_arr == np.asarray(key, dtype=np.int64), axis=1)
            self.surfels.fuse(key, points[block_mask], colors[block_mask], conf[block_mask], timestamp)
            self.tsdf.fuse_points(key, points[block_mask], conf[block_mask], timestamp)
            meta = self.meta[key]
            obs = int(block_mask.sum())
            old_count = meta.obs_count
            meta.last_update_ts = float(timestamp)
            meta.obs_count += obs
            meta.mean_conf = float((meta.mean_conf * old_count + float(conf[block_mask].mean()) * obs) / max(meta.obs_count, 1))
            meta.is_anchor_block = bool(meta.is_anchor_block or current_frame.is_anchor)
            meta.is_hot = True
            meta.dirty = True
            if current_frame.frame_id not in meta.frame_ids:
                meta.frame_ids.append(current_frame.frame_id)
            self.meta[key] = meta
        current_frame.packet.meta["block_keys"] = unique_keys
        current_frame.packet.meta["mean_conf"] = float(conf.mean()) if conf.size else 0.0
        self.evict_cold()
        return unique_keys

    def evict_cold(self) -> list[tuple[int, int, int]]:
        """Evict ordinary hot blocks by LRU while protecting anchor blocks."""

        hot = [meta for meta in self.meta.values() if meta.is_hot]
        over = len(hot) - int(self.cfg.cache.max_hot_blocks)
        if over <= 0:
            return []
        candidates = [meta for meta in hot if not meta.is_anchor_block]
        candidates.sort(key=lambda meta: (meta.last_update_ts, meta.mean_conf))
        evicted: list[tuple[int, int, int]] = []
        for meta in candidates[:over]:
            block = self.surfels.query_block(meta.key)
            if block is not None and self.cold_store is not None:
                self.cold_store.write_block(meta.key, block, meta)
            self.surfels.blocks.pop(meta.key, None)
            self.tsdf.blocks.pop(meta.key, None)
            meta.is_hot = False
            meta.dirty = False
            self.meta[meta.key] = meta
            evicted.append(meta.key)
        return evicted

    def commit_dirty(self) -> int:
        """Mark dirty hot blocks as committed and return the count."""

        count = 0
        for meta in self.meta.values():
            if meta.dirty:
                meta.dirty = False
                count += 1
        return count

    def export_state(self) -> dict[str, Any]:
        """Export lightweight map state for StreamEngine snapshots."""

        return {
            "meta": {str(key): asdict(meta) for key, meta in self.meta.items()},
            "surfels": {str(key): {name: value for name, value in block.items()} for key, block in self.surfels.blocks.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load lightweight map state from a StreamEngine snapshot."""

        self.meta.clear()
        self.surfels.blocks.clear()
        for key_str, raw_meta in state.get("meta", {}).items():
            key = _eval_key(key_str)
            raw = dict(raw_meta)
            raw["key"] = tuple(raw["key"])
            self.meta[key] = BlockMeta(**raw)
        for key_str, block in state.get("surfels", {}).items():
            key = _eval_key(key_str)
            self.surfels.blocks[key] = {name: np.asarray(value, dtype=np.float32) for name, value in block.items()}

    def _try_open3d(self) -> bool:
        """Detect Open3D availability without making it a hard runtime dependency."""

        if not self.cfg.block.enable_open3d:
            return False
        try:
            import open3d as _open3d  # noqa: F401
        except ImportError:
            logger.info("Open3D is not installed; using NumPy block-hash map.")
            return False
        logger.info("Open3D is available; current implementation keeps NumPy state for portability.")
        return True


def extract_points_from_prediction(
    pred: OmniPrediction,
    mode: str,
    images_rgb: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract flattened XYZ, color, confidence, and image coordinates.

    The default path uses ``world_points``. ``depth_backproject`` creates a
    simple normalized pinhole backprojection when world points are unavailable.
    Complexity is O(S * H * W).
    """

    world_points = _to_numpy(pred.world_points) if pred.world_points is not None else None
    if mode == "world_points" and world_points is not None and world_points.size:
        points = _normalize_points_shape(world_points)
    else:
        if pred.depth is None:
            raise ValueError("depth_backproject requires pred.depth when world_points are unavailable.")
        points = _backproject_depth(_to_numpy(pred.depth))
    s_count, height, width, _ = points.shape
    conf_source = pred.world_points_conf if pred.world_points_conf is not None else pred.depth_conf
    conf = _to_numpy(conf_source).reshape(s_count, height, width).astype(np.float32)
    coords = _image_coords(s_count, height, width)
    colors = _colors_for_points(images_rgb, s_count, height, width)
    return {
        "points_xyz": points.reshape(-1, 3).astype(np.float32, copy=False),
        "colors": colors.reshape(-1, 3).astype(np.float32, copy=False),
        "conf": conf.reshape(-1).astype(np.float32, copy=False),
        "image_coords": coords,
    }


def keep_only_changed_points(
    points_xyz: np.ndarray,
    colors: np.ndarray,
    conf: np.ndarray,
    image_coords: np.ndarray,
    change_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Filter point/color/conf arrays to pixels hit by the change mask."""

    coords = np.asarray(image_coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"Expected image_coords Nx2 or Nx3, got {coords.shape}")
    if coords.shape[1] == 3:
        yy = coords[:, 1]
        xx = coords[:, 2]
    else:
        yy = coords[:, 0]
        xx = coords[:, 1]
    mask = np.asarray(change_mask, dtype=bool)
    valid = (yy >= 0) & (yy < mask.shape[0]) & (xx >= 0) & (xx < mask.shape[1])
    keep = np.zeros((coords.shape[0],), dtype=bool)
    keep[valid] = mask[yy[valid], xx[valid]]
    return {
        "points_xyz": np.asarray(points_xyz, dtype=np.float32)[keep],
        "colors": np.asarray(colors, dtype=np.float32)[keep],
        "conf": np.asarray(conf, dtype=np.float32)[keep],
        "image_coords": coords[keep],
    }


def gate_by_confidence(conf: np.ndarray, min_conf: float) -> np.ndarray:
    """Return a boolean mask for confidence values above ``min_conf``."""

    return np.asarray(conf, dtype=np.float32) >= float(min_conf)


def _current_frame_index(selected_window: SelectedWindow) -> int:
    """Return the selected-window index with the newest timestamp."""

    timestamps = [frame.packet.timestamp for frame in selected_window.frames]
    return int(np.argmax(np.asarray(timestamps, dtype=np.float64)))


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to NumPy arrays."""

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _normalize_points_shape(points: np.ndarray) -> np.ndarray:
    """Normalize points to SxHxWx3."""

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected world_points SxHxWx3, got {arr.shape}")
    return arr


def _backproject_depth(depth: np.ndarray) -> np.ndarray:
    """Create a normalized-grid backprojection for depth-only predictions."""

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected depth SxHxW or SxHxWx1, got {arr.shape}")
    s_count, height, width = arr.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    points = np.empty((s_count, height, width, 3), dtype=np.float32)
    points[..., 0] = xx[None] * arr
    points[..., 1] = yy[None] * arr
    points[..., 2] = arr
    return points


def _image_coords(s_count: int, height: int, width: int) -> np.ndarray:
    """Return flattened S,Y,X integer coordinates."""

    ss, yy, xx = np.meshgrid(np.arange(s_count), np.arange(height), np.arange(width), indexing="ij")
    return np.stack([ss, yy, xx], axis=-1).reshape(-1, 3).astype(np.int32)


def _colors_for_points(images_rgb: np.ndarray | None, s_count: int, height: int, width: int) -> np.ndarray:
    """Return SxHxWx3 colors, resizing by nearest crop only when needed."""

    if images_rgb is None:
        return np.zeros((s_count, height, width, 3), dtype=np.float32)
    arr = np.asarray(images_rgb)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.shape[0] != s_count:
        arr = np.repeat(arr[:1], s_count, axis=0)
    if arr.shape[1:3] != (height, width):
        from stream_omnivggt.preprocess.reshape import _resize_array

        arr = np.stack([_resize_array(frame, (width, height), is_depth=False) for frame in arr], axis=0)
    arr = arr.astype(np.float32, copy=False)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _packet_rgb_np(value: Any) -> np.ndarray:
    """Convert packet RGB to HxWx3 NumPy."""

    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr
    return np.asarray(value)


def _eval_key(value: str) -> tuple[int, int, int]:
    """Parse tuple keys exported with str(tuple)."""

    stripped = value.strip().strip("()")
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Invalid exported key: {value}")
    return int(parts[0]), int(parts[1]), int(parts[2])

