"""Spatial block hashing utilities."""

from __future__ import annotations

import numpy as np


def point_to_block_key(point_xyz: np.ndarray, voxel_size: float, block_resolution: int) -> tuple[int, int, int]:
    """Map one XYZ point to an integer block key.

    Complexity is O(1).
    """

    point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    block_size = float(voxel_size) * int(block_resolution)
    if block_size <= 0:
        raise ValueError("voxel_size * block_resolution must be positive.")
    key = np.floor(point / block_size).astype(np.int64)
    return int(key[0]), int(key[1]), int(key[2])


def points_to_block_keys(points_xyz: np.ndarray, voxel_size: float, block_resolution: int) -> np.ndarray:
    """Map Nx3 points to Nx3 integer block keys with vectorized floor division.

    Complexity is O(N).
    """

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points Nx3, got {points.shape}")
    block_size = float(voxel_size) * int(block_resolution)
    if block_size <= 0:
        raise ValueError("voxel_size * block_resolution must be positive.")
    return np.floor(points / block_size).astype(np.int64)


def block_key_to_str(key: tuple[int, int, int]) -> str:
    """Encode a block key for JSON dictionaries."""

    return f"{int(key[0])},{int(key[1])},{int(key[2])}"


def block_key_from_str(value: str) -> tuple[int, int, int]:
    """Decode a JSON block-key string."""

    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid block key string: {value}")
    return int(parts[0]), int(parts[1]), int(parts[2])

