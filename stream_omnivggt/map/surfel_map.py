"""Foreground surfel map with lightweight weighted fusion."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SurfelFusionConfig:
    """Fusion constants for SurfelMap."""

    w_max: float = 32.0
    merge_radius: float = 0.015
    lambda_age: float = 0.02
    occ_z_thr: float = 0.15
    replace_on_occlusion: bool = True


class SurfelMap:
    """Sparse block-indexed surfel map stored in float32 NumPy arrays."""

    def __init__(self, config: SurfelFusionConfig | None = None) -> None:
        """Create an empty surfel map."""

        self.config = config or SurfelFusionConfig()
        self.blocks: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}

    def fuse(
        self,
        block_key: tuple[int, int, int],
        points: np.ndarray,
        colors: np.ndarray,
        conf: np.ndarray,
        timestamp: float,
    ) -> None:
        """Fuse observations into one block using bounded weighted averages.

        Complexity:
            O(N + M), where N is the new observation count and M is the number
            of existing surfels in the target block.
        """

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return
        cols = _normalize_colors(colors, pts.shape[0])
        obs_conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        obs_conf = np.clip(obs_conf, 0.0, self.config.w_max)
        existing = self.blocks.get(block_key)
        if existing is None:
            weights = np.clip(obs_conf, 1e-6, self.config.w_max).astype(np.float32)
            normals = _default_normals(pts.shape[0])
            self.blocks[block_key] = {
                "points": pts.copy(),
                "colors": cols.copy(),
                "weights": weights,
                "normals": normals,
                "timestamps": np.full((pts.shape[0],), timestamp, dtype=np.float32),
            }
            return

        points_old = existing["points"]
        colors_old = existing["colors"]
        weights_old = existing["weights"]
        normals_old = existing["normals"]
        ts_old = existing["timestamps"]
        index = _quantized_index(points_old, self.config.merge_radius)

        append_points: list[np.ndarray] = []
        append_colors: list[np.ndarray] = []
        append_weights: list[float] = []
        append_normals: list[np.ndarray] = []
        append_ts: list[float] = []
        for point, color, confidence in zip(pts, cols, obs_conf):
            if confidence <= 0.0:
                continue
            qkey = _quant_key(point, self.config.merge_radius)
            match = index.get(qkey)
            if match is None:
                append_points.append(point)
                append_colors.append(color)
                append_weights.append(float(np.clip(confidence, 1e-6, self.config.w_max)))
                append_normals.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
                append_ts.append(float(timestamp))
                continue
            z_delta = abs(float(point[2] - points_old[match, 2]))
            if z_delta > self.config.occ_z_thr and self.config.replace_on_occlusion:
                points_old[match] = point
                colors_old[match] = color
                weights_old[match] = np.clip(confidence, 1e-6, self.config.w_max)
                normals_old[match] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                ts_old[match] = timestamp
                continue
            dt = max(float(timestamp - ts_old[match]), 0.0)
            w_obs = float(confidence) * float(np.exp(-self.config.lambda_age * dt))
            w_new = min(float(weights_old[match] + w_obs), self.config.w_max)
            if w_new <= 1e-8:
                continue
            points_old[match] = (points_old[match] * weights_old[match] + point * w_obs) / w_new
            colors_old[match] = (colors_old[match] * weights_old[match] + color * w_obs) / w_new
            normals_old[match] = _normalize_vector(normals_old[match] * weights_old[match] + np.array([0.0, 0.0, 1.0], dtype=np.float32) * w_obs)
            weights_old[match] = w_new
            ts_old[match] = timestamp

        if append_points:
            existing["points"] = np.concatenate([points_old, np.asarray(append_points, dtype=np.float32)], axis=0)
            existing["colors"] = np.concatenate([colors_old, np.asarray(append_colors, dtype=np.float32)], axis=0)
            existing["weights"] = np.concatenate([weights_old, np.asarray(append_weights, dtype=np.float32)], axis=0)
            existing["normals"] = np.concatenate([normals_old, np.asarray(append_normals, dtype=np.float32)], axis=0)
            existing["timestamps"] = np.concatenate([ts_old, np.asarray(append_ts, dtype=np.float32)], axis=0)

    def query_block(self, block_key: tuple[int, int, int]) -> dict[str, np.ndarray] | None:
        """Return one block's surfel arrays or None if absent."""

        block = self.blocks.get(block_key)
        if block is None:
            return None
        return {name: value.copy() for name, value in block.items()}


def _normalize_colors(colors: np.ndarray, count: int) -> np.ndarray:
    """Normalize color observations to Nx3 float32 in [0, 1]."""

    cols = np.asarray(colors)
    if cols.size == 0:
        return np.zeros((count, 3), dtype=np.float32)
    cols = cols.reshape(-1, 3).astype(np.float32, copy=False)
    if cols.shape[0] != count:
        raise ValueError(f"Expected {count} colors, got {cols.shape[0]}")
    if cols.max(initial=0.0) > 1.0:
        cols = cols / 255.0
    return np.clip(cols, 0.0, 1.0).astype(np.float32, copy=False)


def _default_normals(count: int) -> np.ndarray:
    """Return count unit z normals."""

    normals = np.zeros((count, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    return normals


def _quant_key(point: np.ndarray, radius: float) -> tuple[int, int, int]:
    """Quantize one point for approximate surfel merging."""

    r = max(float(radius), 1e-6)
    q = np.round(point / r).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2])


def _quantized_index(points: np.ndarray, radius: float) -> dict[tuple[int, int, int], int]:
    """Build a quantized lookup table for existing surfels."""

    index: dict[tuple[int, int, int], int] = {}
    for idx, point in enumerate(points):
        index.setdefault(_quant_key(point, radius), idx)
    return index


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    """Normalize a vector and fall back to +Z for degenerate input."""

    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (value / norm).astype(np.float32)

