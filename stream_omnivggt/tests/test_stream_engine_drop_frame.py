"""Dropped-frame protection tests."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.backend import MockOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.pipeline import StreamEngine
from stream_omnivggt.types import InputPacket


def test_dropped_frame_does_not_crash_and_sets_fallback() -> None:
    """A large timestamp gap should disable short-term flow and continue."""

    cfg = StreamConfig()
    cfg.omni.target_width = 56
    cfg.omni.target_size = 56
    cfg.omni.warmup_buckets = [(3, 3, 56, 56)]
    cfg.change.flow_mode = "farneback"
    cfg.fallback.dropped_frame_dt = 1.0
    engine = StreamEngine(MockOmniBackend(), cfg)
    rgb = np.zeros((56, 56, 3), dtype=np.uint8)
    engine.push(InputPacket(0, 0.0, rgb, None, None, None, {}))
    metrics = engine.push(InputPacket(1, 5.0, rgb.copy(), None, None, None, {}))
    engine.flush()
    assert metrics.fallback_reason == "dropped_frame"
    assert metrics.total_ms >= 0.0

