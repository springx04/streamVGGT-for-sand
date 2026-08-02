"""Rotation alignment adapter for external no-training preprocessors."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np

from stream_omnivggt.types import AlignmentResult, InputPacket

logger = logging.getLogger(__name__)


def run_rotation_align(packet: InputPacket, state: dict[str, Any], **kwargs: Any) -> AlignmentResult:
    """Run an external rotation aligner without touching model weights.

    The optional callable is read from ``state["rotation_aligner"]`` and may
    return either a 4x4 transform or a dict with ``transform`` and ``quality``.
    Complexity is delegated to the user-provided aligner; the built-in path is
    O(1).
    """

    transform = np.eye(4, dtype=np.float32)
    quality = 1.0
    flags = {"failed": False, "used_external": False}
    try:
        aligner = state.get("rotation_aligner")
        if callable(aligner):
            result = aligner(packet, state=state, **kwargs)
            flags["used_external"] = True
            if isinstance(result, dict):
                transform = np.asarray(result.get("transform", transform), dtype=np.float32)
                quality = float(result.get("quality", quality))
            else:
                transform = np.asarray(result, dtype=np.float32)
            if transform.shape != (4, 4):
                raise ValueError(f"Rotation aligner returned shape {transform.shape}, expected 4x4")
    except Exception as exc:  # noqa: BLE001 - warning and fail-open are required for stream robustness.
        logger.warning("Rotation alignment failed for frame %s: %s", packet.frame_id, exc)
        transform = np.eye(4, dtype=np.float32)
        quality = 0.0
        flags["failed"] = True

    aligned = _apply_transform_to_c2w(packet.extrinsic_c2w, transform)
    state["last_rotation_transform"] = transform
    state["last_rotation_quality"] = quality
    return AlignmentResult(
        packet=packet,
        rotation_transform=transform,
        height_transform=np.eye(4, dtype=np.float32),
        aligned_extrinsic_c2w=aligned,
        quality_score=quality,
        flags=flags,
    )


def _apply_transform_to_c2w(extrinsic_c2w: np.ndarray | None, transform: np.ndarray) -> np.ndarray | None:
    """Left-multiply a 4x4 world transform onto a camera-to-world pose."""

    if extrinsic_c2w is None:
        return None
    pose = np.asarray(extrinsic_c2w, dtype=np.float32)
    if pose.shape == (3, 4):
        pose4 = np.eye(4, dtype=np.float32)
        pose4[:3, :4] = pose
    elif pose.shape == (4, 4):
        pose4 = pose.copy()
    else:
        raise ValueError(f"Expected extrinsic_c2w shape 3x4 or 4x4, got {pose.shape}")
    return (transform @ pose4).astype(np.float32)

