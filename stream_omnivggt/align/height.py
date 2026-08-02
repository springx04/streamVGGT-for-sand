"""Height alignment adapter for external no-training preprocessors."""

from __future__ import annotations

from typing import Any
import logging

import numpy as np

from stream_omnivggt.align.rotation import _apply_transform_to_c2w
from stream_omnivggt.types import AlignmentResult, InputPacket

logger = logging.getLogger(__name__)


def run_height_align(packet: InputPacket, state: dict[str, Any], **kwargs: Any) -> AlignmentResult:
    """Run an external height aligner and return fail-open alignment metadata.

    The optional callable is read from ``state["height_aligner"]``. It is
    treated as a pure input/pose preprocessor and must not update model
    weights. The built-in path is O(1).
    """

    transform = np.eye(4, dtype=np.float32)
    quality = 1.0
    flags = {"failed": False, "used_external": False}
    try:
        aligner = state.get("height_aligner")
        if callable(aligner):
            result = aligner(packet, state=state, **kwargs)
            flags["used_external"] = True
            if isinstance(result, dict):
                transform = np.asarray(result.get("transform", transform), dtype=np.float32)
                quality = float(result.get("quality", quality))
            else:
                transform = np.asarray(result, dtype=np.float32)
            if transform.shape != (4, 4):
                raise ValueError(f"Height aligner returned shape {transform.shape}, expected 4x4")
    except Exception as exc:  # noqa: BLE001 - streaming fallback needs to catch preprocessor failures.
        logger.warning("Height alignment failed for frame %s: %s", packet.frame_id, exc)
        transform = np.eye(4, dtype=np.float32)
        quality = 0.0
        flags["failed"] = True

    rotation_transform = np.asarray(state.get("last_rotation_transform", np.eye(4)), dtype=np.float32)
    combined = transform @ rotation_transform
    aligned = _apply_transform_to_c2w(packet.extrinsic_c2w, combined)
    state["last_height_transform"] = transform
    state["last_height_quality"] = quality
    return AlignmentResult(
        packet=packet,
        rotation_transform=rotation_transform,
        height_transform=transform,
        aligned_extrinsic_c2w=aligned,
        quality_score=min(float(state.get("last_rotation_quality", 1.0)), quality),
        flags=flags,
    )

