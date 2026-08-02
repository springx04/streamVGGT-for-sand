"""Tests for block hashing and hybrid map persistence."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.config import StreamConfig
from stream_omnivggt.map import HybridMap, point_to_block_key, points_to_block_keys
from stream_omnivggt.map.surfel_map import SurfelFusionConfig, SurfelMap
from stream_omnivggt.types import BlockMeta


def test_points_map_to_blocks() -> None:
    """Small point batches should map to expected block keys."""

    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float32)
    keys = points_to_block_keys(points, voxel_size=0.1, block_resolution=4)
    assert keys.shape == (3, 3)
    assert point_to_block_key(points[0], 0.1, 4) == (0, 0, 0)
    assert tuple(keys[1]) == (1, 0, 0)
    assert tuple(keys[2]) == (-1, 0, 0)


def test_surfel_weights_increase_but_clamp() -> None:
    """Repeated fusion should raise weights without exceeding w_max."""

    surfels = SurfelMap(SurfelFusionConfig(w_max=2.0, merge_radius=0.1))
    key = (0, 0, 0)
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    conf = np.array([1.0], dtype=np.float32)
    surfels.fuse(key, points, colors, conf, 0.0)
    first = surfels.query_block(key)["weights"][0]  # type: ignore[index]
    surfels.fuse(key, points, colors, conf, 0.1)
    second = surfels.query_block(key)["weights"][0]  # type: ignore[index]
    surfels.fuse(key, points, colors, conf, 0.2)
    third = surfels.query_block(key)["weights"][0]  # type: ignore[index]
    assert second > first
    assert third <= 2.0


def test_cold_evict_and_restore_from_memmap(tmp_path) -> None:
    """Evicted ordinary blocks should be restorable from cold memmap storage."""

    cfg = StreamConfig()
    cfg.cache.max_hot_blocks = 1
    cfg.cache.cold_store_dir = str(tmp_path)
    hybrid = HybridMap(cfg, cold_root=tmp_path)
    for idx, key in enumerate([(0, 0, 0), (1, 0, 0)]):
        points = np.array([[idx, 0.0, 1.0]], dtype=np.float32)
        hybrid.surfels.blocks[key] = {
            "points": points,
            "colors": np.ones((1, 3), dtype=np.float32),
            "weights": np.ones((1,), dtype=np.float32),
            "normals": np.array([[0, 0, 1]], dtype=np.float32),
            "timestamps": np.zeros((1,), dtype=np.float32),
        }
        hybrid.meta[key] = BlockMeta(key, float(idx), 1, 1.0, False, True, True, [idx])
    evicted = hybrid.evict_cold()
    assert evicted
    restored_key = evicted[0]
    assert hybrid.surfels.query_block(restored_key) is None
    hybrid.ensure_hot({restored_key})
    assert hybrid.surfels.query_block(restored_key) is not None
    assert hybrid.meta[restored_key].is_hot

