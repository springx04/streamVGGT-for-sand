"""Tests for deterministic mock backend output."""

from __future__ import annotations

import numpy as np
import torch

from stream_omnivggt.backend import MockOmniBackend


def test_mock_backend_shapes() -> None:
    """Mock backend should emit point and confidence maps with expected shapes."""

    backend = MockOmniBackend()
    images = torch.zeros((1, 3, 3, 56, 56), dtype=torch.float32)
    pred = backend.run_window({"images": images})
    assert pred.world_points.shape == (3, 56, 56, 3)
    assert pred.world_points_conf.shape == (3, 56, 56)


def test_mock_backend_is_stable() -> None:
    """Identical inputs should produce identical mock outputs."""

    backend = MockOmniBackend()
    images = torch.rand((1, 2, 3, 28, 28), dtype=torch.float32)
    first = backend.run_window({"images": images})
    second = backend.run_window({"images": images})
    assert np.allclose(first.world_points, second.world_points)
    assert np.allclose(first.world_points_conf, second.world_points_conf)

