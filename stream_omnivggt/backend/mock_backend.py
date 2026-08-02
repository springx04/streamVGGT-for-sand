"""Deterministic mock backend used when OmniVGGT weights are unavailable."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
import torch

from stream_omnivggt.types import OmniPrediction

logger = logging.getLogger(__name__)


class MockOmniBackend:
    """A stable, zero-training backend that maps RGB/depth to pseudo geometry."""

    def __init__(self, noise: float = 0.0) -> None:
        """Create a deterministic mock backend.

        Args:
            noise: Optional deterministic sinusoidal perturbation amplitude.
        """

        self.noise = float(noise)
        self._warm_shapes: list[tuple[int, int, int, int]] = []

    def warmup(self, bucket_shapes: list[tuple[int, int, int, int]]) -> None:
        """Record bucket shapes; no heavyweight work is needed for the mock."""

        self._warm_shapes = list(bucket_shapes)
        logger.info("MockOmniBackend warmup buckets=%s", self._warm_shapes)

    def name(self) -> str:
        """Return the backend display name."""

        return "mock"

    def run_window(self, batch: dict[str, Any]) -> OmniPrediction:
        """Generate SxHxWx3 pseudo world points and SxHxW confidence maps.

        Complexity:
            O(S * H * W) time and memory. The function performs only vectorized
            NumPy operations and is deterministic for identical inputs.
        """

        images = _to_numpy(batch["images"])
        if images.ndim == 5:
            images = images[0]
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape SxCxHxW or BxSxCxHxW, got {images.shape}")

        images = images.astype(np.float32, copy=False)
        if images.max(initial=0.0) > 1.0:
            images = images / 255.0
        s_count, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3 image channels, got {channels}")

        depth = _optional_depth_to_numpy(batch.get("depth"), s_count, height, width)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, height, dtype=np.float32),
            np.linspace(-1.0, 1.0, width, dtype=np.float32),
            indexing="ij",
        )
        gray = images.mean(axis=1)
        world_points = np.empty((s_count, height, width, 3), dtype=np.float32)
        conf = np.empty((s_count, height, width), dtype=np.float32)
        for idx in range(s_count):
            z = depth[idx] if depth is not None else 1.0 + 0.5 * gray[idx]
            z = z.astype(np.float32, copy=False)
            if self.noise:
                z = z + self.noise * np.sin((idx + 1) * xx * np.pi).astype(np.float32)
            world_points[idx, ..., 0] = xx * z + 0.03 * idx
            world_points[idx, ..., 1] = yy * z
            world_points[idx, ..., 2] = z
            grad_x = np.abs(np.gradient(gray[idx], axis=1))
            grad_y = np.abs(np.gradient(gray[idx], axis=0))
            local_texture = np.clip(grad_x + grad_y, 0.0, 1.0)
            conf[idx] = np.clip(0.55 + 0.4 * (1.0 - local_texture) + 0.05 * gray[idx], 0.0, 1.0)

        return OmniPrediction(
            world_points=world_points,
            world_points_conf=conf.astype(np.float32, copy=False),
            depth=depth[..., None] if depth is not None else world_points[..., 2:3].copy(),
            depth_conf=conf.copy(),
            pose_enc=None,
            extra={"backend": self.name(), "warm_shapes": list(self._warm_shapes)},
        )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors and arrays to a CPU NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _optional_depth_to_numpy(value: Any, s_count: int, height: int, width: int) -> np.ndarray | None:
    """Normalize optional depth input to SxHxW float32 or return None."""

    if value is None:
        return None
    depth = _to_numpy(value).astype(np.float32, copy=False)
    if depth.ndim == 5:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim == 3 and depth.shape == (s_count, height, width):
        valid = depth > 1e-6
        if not valid.any():
            return None
        fallback = np.ones_like(depth, dtype=np.float32)
        return np.where(valid, depth, fallback).astype(np.float32, copy=False)
    logger.warning("Ignoring depth with incompatible shape %s", depth.shape)
    return None

