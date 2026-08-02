"""Small-change stream engine tests."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.backend import MockOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.pipeline import StreamEngine
from stream_omnivggt.types import InputPacket


def _cfg() -> StreamConfig:
    """Return a low-resolution engine config for tests."""

    cfg = StreamConfig()
    cfg.omni.target_width = 56
    cfg.omni.target_size = 56
    cfg.omni.warmup_buckets = [(3, 3, 56, 56), (4, 3, 56, 56)]
    cfg.change.flow_mode = "none"
    cfg.fuse.min_conf = 0.0
    cfg.cache.max_hot_blocks = 128
    return cfg


def _packet(idx: int, patch: bool = False) -> InputPacket:
    """Create a deterministic stream packet."""

    rgb = np.full((56, 56, 3), 80, dtype=np.uint8)
    if patch:
        rgb[20:26, 20:26] = np.array([230, 30, 30], dtype=np.uint8)
    depth = np.ones((56, 56), dtype=np.float32)
    return InputPacket(idx, idx * 0.1, rgb, depth, None, None, {})


def test_small_patch_uses_incremental_update() -> None:
    """Only the patch frame should update substantially fewer blocks than refresh."""

    engine = StreamEngine(MockOmniBackend(), _cfg())
    metrics = [engine.push(_packet(idx, patch=(idx == 2))) for idx in range(5)]
    engine.flush()
    assert metrics[0].updated_block_count > 0
    assert metrics[2].updated_block_count < metrics[0].updated_block_count
    assert sum(1 for item in metrics[1:] if item.fallback_reason == "full_refresh") <= 1
    assert sum(1 for item in metrics[1:] if item.skipped_model) >= 2

