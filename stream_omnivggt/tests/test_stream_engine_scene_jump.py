"""Scene-jump fallback tests."""

from __future__ import annotations

import numpy as np

from stream_omnivggt.backend import MockOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.pipeline import StreamEngine
from stream_omnivggt.types import InputPacket


def test_scene_jump_triggers_refresh_or_new_segment() -> None:
    """A large synthetic change should trigger full refresh or segmentation."""

    cfg = StreamConfig()
    cfg.omni.target_width = 56
    cfg.omni.target_size = 56
    cfg.omni.warmup_buckets = [(3, 3, 56, 56)]
    cfg.change.flow_mode = "none"
    engine = StreamEngine(MockOmniBackend(), cfg)
    first = np.zeros((56, 56, 3), dtype=np.uint8)
    second = np.full((56, 56, 3), 255, dtype=np.uint8)
    engine.push(InputPacket(0, 0.0, first, np.ones((56, 56), dtype=np.float32), None, None, {}))
    metrics = engine.push(InputPacket(1, 0.1, second, np.ones((56, 56), dtype=np.float32), None, None, {}))
    engine.flush()
    assert metrics.fallback_reason in {"full_refresh", "new_segment"}

