"""Shared dataclasses for the streaming OmniVGGT wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(slots=True)
class InputPacket:
    """One stream input frame with optional geometry hints."""

    frame_id: int
    timestamp: float
    rgb: np.ndarray | torch.Tensor
    depth: np.ndarray | None = None
    intrinsic: np.ndarray | None = None
    extrinsic_c2w: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlignmentResult:
    """Result from an external, non-learning alignment preprocessor."""

    packet: InputPacket
    rotation_transform: np.ndarray
    height_transform: np.ndarray
    aligned_extrinsic_c2w: np.ndarray | None
    quality_score: float
    flags: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class ChangeMaskResult:
    """Dense image-space change mask and coarse affected block hints."""

    mask: np.ndarray
    score_map: np.ndarray
    changed_ratio: float
    changed_pixels: int
    reprojection_error_mean: float
    depth_error_mean: float
    blocks_hint: set[tuple[int, int, int]] = field(default_factory=set)


@dataclass(slots=True)
class WindowFrame:
    """Frame metadata used by the active-window selector."""

    frame_id: int
    is_anchor: bool
    has_camera: bool
    has_depth: bool
    priority_score: float
    packet: InputPacket
    cached_prediction_ref: str | None = None


@dataclass(slots=True)
class SelectedWindow:
    """A bucketed local inference window for OmniVGGT."""

    frames: list[WindowFrame]
    camera_gt_index: list[int]
    depth_gt_index: list[int]
    bucket_key: str
    reason: str


@dataclass(slots=True)
class OmniPrediction:
    """Backend prediction normalized to the fields used by the map layer."""

    world_points: np.ndarray | torch.Tensor
    world_points_conf: np.ndarray | torch.Tensor
    depth: np.ndarray | torch.Tensor | None = None
    depth_conf: np.ndarray | torch.Tensor | None = None
    pose_enc: np.ndarray | torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockMeta:
    """Persistent metadata for one spatial map block."""

    key: tuple[int, int, int]
    last_update_ts: float
    obs_count: int
    mean_conf: float
    is_anchor_block: bool
    is_hot: bool
    dirty: bool
    frame_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class StreamMetrics:
    """Per-push timing and update metrics for the streaming engine."""

    ingest_ms: float = 0.0
    align_ms: float = 0.0
    diff_ms: float = 0.0
    select_ms: float = 0.0
    model_ms: float = 0.0
    project_ms: float = 0.0
    fuse_ms: float = 0.0
    commit_ms: float = 0.0
    total_ms: float = 0.0
    updated_block_count: int = 0
    updated_point_ratio: float = 0.0
    skipped_model: bool = False
    fallback_reason: str | None = None

