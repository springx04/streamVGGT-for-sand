"""Relative performance contract tests with the mock backend."""

from __future__ import annotations

import logging
import statistics

import numpy as np
import pytest
import torch

from stream_omnivggt.backend import MockOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.pipeline import StreamEngine
from stream_omnivggt.types import InputPacket, StreamMetrics

logger = logging.getLogger(__name__)


def _cfg(strategy: str) -> StreamConfig:
    """Return a deterministic benchmark config."""

    cfg = StreamConfig()
    cfg.benchmark.strategy = strategy
    cfg.omni.target_width = 56
    cfg.omni.target_size = 56
    cfg.omni.warmup_buckets = [(3, 3, 56, 56), (4, 3, 56, 56), (6, 3, 56, 56), (8, 3, 56, 56)]
    cfg.change.flow_mode = "none"
    cfg.fuse.min_conf = 0.0
    cfg.cache.max_hot_blocks = 256
    return cfg


def _packet(idx: int) -> InputPacket:
    """Create one synthetic benchmark packet."""

    rgb = np.full((56, 56, 3), 90, dtype=np.uint8)
    if idx % 5 == 2:
        rgb[18:26, 18:26] = np.array([240, 40, 40], dtype=np.uint8)
    depth = np.ones((56, 56), dtype=np.float32)
    return InputPacket(idx, idx * 0.1, rgb, depth, None, None, {})


def _run(strategy: str) -> list[StreamMetrics]:
    """Run one mock benchmark strategy."""

    engine = StreamEngine(MockOmniBackend(), _cfg(strategy))
    metrics = [engine.push(_packet(idx)) for idx in range(20)]
    engine.flush()
    return metrics


def _avg(values: list[float]) -> float:
    """Return arithmetic mean."""

    return float(statistics.fmean(values))


def _p90(values: list[float]) -> float:
    """Return simple p90."""

    ordered = sorted(values)
    return float(ordered[int(round((len(ordered) - 1) * 0.9))])


def test_incremental_is_cheaper_than_full_rebuild() -> None:
    """Incremental strategy should touch fewer points and run faster on average."""

    full = _run("full_rebuild")
    incr = _run("block_incremental")
    assert _avg([m.updated_point_ratio for m in incr]) < _avg([m.updated_point_ratio for m in full])
    assert _avg([m.total_ms for m in incr]) < _avg([m.total_ms for m in full])


def test_keyframe_hybrid_p90_within_contract() -> None:
    """Keyframe hybrid p90 should not exceed twice block incremental p90."""

    incr = _run("block_incremental")
    hybrid = _run("keyframe_hybrid")
    assert _p90([m.total_ms for m in hybrid]) <= 2.0 * _p90([m.total_ms for m in incr])


def test_cuda_pin_memory_log_if_available() -> None:
    """Log pinned and unpinned transfer timing when CUDA is present."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tensor = torch.zeros((4, 3, 56, 56), dtype=torch.float32)
    pinned = tensor.pin_memory()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = tensor.to("cuda", non_blocking=False)
    end.record()
    torch.cuda.synchronize()
    unpinned_ms = start.elapsed_time(end)
    start.record()
    _ = pinned.to("cuda", non_blocking=True)
    end.record()
    torch.cuda.synchronize()
    pinned_ms = start.elapsed_time(end)
    logger.info("CUDA transfer comparison: pinned=%.4fms unpinned=%.4fms", pinned_ms, unpinned_ms)

