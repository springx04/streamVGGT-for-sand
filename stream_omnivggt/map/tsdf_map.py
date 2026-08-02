"""Simplified background TSDF/voxel map."""

from __future__ import annotations

import numpy as np


class TsdfMap:
    """A minimal block-indexed TSDF-like accumulator."""

    def __init__(self, w_max: float = 32.0, voxel_size: float = 0.03) -> None:
        """Create an empty simplified TSDF map."""

        self.w_max = float(w_max)
        self.voxel_size = float(voxel_size)
        self.blocks: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}

    def fuse_points(self, block_key: tuple[int, int, int], points: np.ndarray, conf: np.ndarray, timestamp: float) -> None:
        """Fuse points into a simplified voxel block with weighted occupancy.

        Complexity is O(N + M), where N is incoming points and M existing
        voxels in the block.
        """

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return
        weights = np.clip(np.asarray(conf, dtype=np.float32).reshape(-1), 0.0, self.w_max)
        block = self.blocks.get(block_key)
        if block is None:
            self.blocks[block_key] = {
                "centers": pts.copy(),
                "tsdf": np.zeros((pts.shape[0],), dtype=np.float32),
                "weights": np.clip(weights, 1e-6, self.w_max).astype(np.float32),
                "timestamps": np.full((pts.shape[0],), timestamp, dtype=np.float32),
            }
            return
        index = {_voxel_key(point, self.voxel_size): idx for idx, point in enumerate(block["centers"])}
        append_centers: list[np.ndarray] = []
        append_weights: list[float] = []
        append_tsdf: list[float] = []
        append_ts: list[float] = []
        for point, weight in zip(pts, weights):
            if weight <= 0.0:
                continue
            qkey = _voxel_key(point, self.voxel_size)
            match = index.get(qkey)
            if match is None:
                append_centers.append(point)
                append_weights.append(float(np.clip(weight, 1e-6, self.w_max)))
                append_tsdf.append(0.0)
                append_ts.append(float(timestamp))
                continue
            old_w = float(block["weights"][match])
            new_w = min(old_w + float(weight), self.w_max)
            block["centers"][match] = (block["centers"][match] * old_w + point * float(weight)) / max(new_w, 1e-6)
            block["tsdf"][match] = (block["tsdf"][match] * old_w) / max(new_w, 1e-6)
            block["weights"][match] = new_w
            block["timestamps"][match] = timestamp
        if append_centers:
            block["centers"] = np.concatenate([block["centers"], np.asarray(append_centers, dtype=np.float32)], axis=0)
            block["tsdf"] = np.concatenate([block["tsdf"], np.asarray(append_tsdf, dtype=np.float32)], axis=0)
            block["weights"] = np.concatenate([block["weights"], np.asarray(append_weights, dtype=np.float32)], axis=0)
            block["timestamps"] = np.concatenate([block["timestamps"], np.asarray(append_ts, dtype=np.float32)], axis=0)

    def query_block(self, block_key: tuple[int, int, int]) -> dict[str, np.ndarray] | None:
        """Return a copy of one TSDF block or None."""

        block = self.blocks.get(block_key)
        if block is None:
            return None
        return {name: value.copy() for name, value in block.items()}


def _voxel_key(point: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    """Quantize one point to a voxel key."""

    q = np.floor(point / max(voxel_size, 1e-6)).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2])

