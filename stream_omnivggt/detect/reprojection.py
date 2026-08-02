"""Lightweight reprojection placeholders for external geometry hints."""

from __future__ import annotations

import numpy as np


def identity_reprojection(rgb: np.ndarray | None, depth: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the provided RGB/depth as a no-op reprojection baseline."""

    return rgb, depth


def estimate_overlap_from_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Estimate overlap ratio between two boolean validity masks."""

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = a | b
    if not union.any():
        return 0.0
    return float((a & b).sum() / union.sum())

