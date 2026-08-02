"""Camera conversion helpers for OmniVGGT input preparation."""

from __future__ import annotations

import numpy as np


def convert_camera_c2w_to_w2c(extrinsic_c2w: np.ndarray | None) -> np.ndarray | None:
    """Convert a 3x4 or 4x4 camera-to-world matrix to 3x4 world-to-camera."""

    if extrinsic_c2w is None:
        return None
    pose = np.asarray(extrinsic_c2w, dtype=np.float32)
    if pose.shape == (3, 4):
        pose4 = np.eye(4, dtype=np.float32)
        pose4[:3, :4] = pose
    elif pose.shape == (4, 4):
        pose4 = pose
    else:
        raise ValueError(f"Expected c2w pose shape 3x4 or 4x4, got {pose.shape}")
    return np.linalg.inv(pose4)[:3, :4].astype(np.float32)


def default_intrinsic(width: int, height: int) -> np.ndarray:
    """Create a conservative pinhole intrinsic matrix for missing cameras."""

    focal = float(max(width, height))
    return np.array([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

