"""Optical-flow helpers with OpenCV and safe NumPy fallbacks."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only when OpenCV is missing.
    cv2 = None  # type: ignore


def compute_sparse_lk_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Compute sparse Lucas-Kanade flow or return an empty fallback result."""

    prev = _to_uint8_gray(prev_gray)
    curr = _to_uint8_gray(curr_gray)
    if cv2 is None:
        logger.warning("OpenCV is unavailable; sparse LK flow returns no tracks.")
        return {"prev_pts": np.empty((0, 2), np.float32), "curr_pts": np.empty((0, 2), np.float32), "count": 0, "mean_error": 0.0}
    features = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=5)
    if features is None:
        return {"prev_pts": np.empty((0, 2), np.float32), "curr_pts": np.empty((0, 2), np.float32), "count": 0, "mean_error": 0.0}
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev, curr, features, None)
    if curr_pts is None or status is None:
        return {"prev_pts": np.empty((0, 2), np.float32), "curr_pts": np.empty((0, 2), np.float32), "count": 0, "mean_error": 0.0}
    valid = status.reshape(-1) > 0
    return {
        "prev_pts": features.reshape(-1, 2)[valid].astype(np.float32),
        "curr_pts": curr_pts.reshape(-1, 2)[valid].astype(np.float32),
        "count": int(valid.sum()),
        "mean_error": float(np.mean(err.reshape(-1)[valid])) if err is not None and valid.any() else 0.0,
    }


def compute_dense_farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense Farneback optical flow as HxWx2 float32."""

    prev = _to_uint8_gray(prev_gray)
    curr = _to_uint8_gray(curr_gray)
    if cv2 is None:
        logger.warning("OpenCV is unavailable; dense flow returns zeros.")
        return np.zeros((*curr.shape, 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.astype(np.float32, copy=False)


def _to_uint8_gray(gray: np.ndarray) -> np.ndarray:
    """Normalize a grayscale array to uint8 for OpenCV flow."""

    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32, copy=False)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)

