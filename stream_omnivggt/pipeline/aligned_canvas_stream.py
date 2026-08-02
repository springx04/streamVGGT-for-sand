"""Fast anchor-canvas stream for uncalibrated image sequences.

This path is intended for data such as ``data2`` where frames have no reliable
camera/depth side inputs. It does not trust per-window 3D poses. Instead, it
uses a canonical 2D anchor canvas, aligns each new image by homography, runs
OmniVGGT for per-frame depth/confidence, affine-aligns depth to the anchor
height field, and fuses one sample per canonical XY cell.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Any
import logging

import numpy as np
import torch

from stream_omnivggt.backend.base import BaseOmniBackend
from stream_omnivggt.config import StreamConfig
from stream_omnivggt.preprocess import normalize_rgb, resize_to_bucket
from stream_omnivggt.types import InputPacket

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - OpenCV is declared in pyproject.
    cv2 = None  # type: ignore


@dataclass(slots=True)
class CanvasStreamMetrics:
    """Per-frame timings for aligned-canvas streaming."""

    frame_id: int
    read_ms: float = 0.0
    preprocess_ms: float = 0.0
    align2d_ms: float = 0.0
    diff_ms: float = 0.0
    model_ms: float = 0.0
    depth_align_ms: float = 0.0
    fuse_ms: float = 0.0
    total_ms: float = 0.0
    changed_ratio: float = 0.0
    photometric_changed_ratio: float = 0.0
    support_changed_ratio: float = 0.0
    roi_width: int = 0
    roi_height: int = 0
    fused_pixels: int = 0
    point_count: int = 0
    anchor_pixels: int = 0
    homography_inliers: int = 0
    homography_error_px: float | None = None
    skipped_model: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics dictionary."""

        return asdict(self)


class AlignedCanvasStream:
    """Low-latency stream for no-camera/no-depth sequences using a 2D anchor."""

    def __init__(self, backend: BaseOmniBackend, cfg: StreamConfig) -> None:
        """Create an empty aligned-canvas stream."""

        self.backend = backend
        self.cfg = cfg
        self.anchor_rgb: np.ndarray | None = None
        self.anchor_match_rgb: np.ndarray | None = None
        # Alignment benefits from detail, while the black-box model must receive
        # a bounded crop.  Keeping these budgets separate is what makes the
        # stream path cheaper than a full-frame reconstruction.
        self.match_width = max(700, int(cfg.omni.target_width))
        self.detect_width = int(cfg.omni.target_width)
        self.roi_width = _round_down_patch(int(cfg.omni.target_width), cfg.omni.patch_multiple)
        self.roi_height = _round_down_patch(int(cfg.omni.target_size), cfg.omni.patch_multiple)
        # The first data2 view touches the top and bottom of the image.  A
        # fixed canvas with the same extent as that frame permanently loses
        # any surface that a later view maps outside those borders.  Keep the
        # matching canvas padded, while the first model call still uses the
        # unpadded image so its effective resolution does not change.
        self.canvas_pad_left = max(32, int(round(self.match_width * 0.05)))
        self.canvas_pad_right = max(32, int(round(self.match_width * 0.05)))
        self.canvas_pad_top = max(128, int(round(self.match_width * 0.18)))
        self.canvas_pad_bottom = max(64, int(round(self.match_width * 0.09)))
        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.conf: np.ndarray | None = None
        self.weight: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        self.support: np.ndarray | None = None
        self.transforms: dict[int, np.ndarray] = {}
        self.last_debug: dict[str, Any] = {}

    def push(self, packet: InputPacket, read_ms: float = 0.0) -> CanvasStreamMetrics:
        """Process one frame and update the canonical canvas.

        Complexity:
            O(HW) for warping, differencing, depth alignment, and fusion. Model
            cost is one black-box OmniVGGT call when the changed ratio is not
            negligible.
        """

        start = perf_counter()
        metrics = CanvasStreamMetrics(frame_id=packet.frame_id, read_ms=read_ms)

        t0 = perf_counter()
        raw_rgb = _rgb_np(packet.rgb)
        raw_match_rgb = _resize_for_matching(raw_rgb, self.match_width)
        # All canonical alignment/state operations use the padded canvas.  The
        # unpadded image is retained only for the first full-frame inference.
        match_rgb = _pad_canvas(
            raw_match_rgb,
            self.canvas_pad_left,
            self.canvas_pad_top,
            self.canvas_pad_right,
            self.canvas_pad_bottom,
        )
        rgb_f = match_rgb
        metrics.preprocess_ms = _ms(t0)

        if self.anchor_rgb is None:
            self.anchor_rgb = rgb_f
            self.anchor_match_rgb = match_rgb
            transform = np.eye(3, dtype=np.float32)
            transform_match = np.eye(3, dtype=np.float32)
            align_info = {"inliers": 0, "median_error": None, "fallback": None}
        else:
            t0 = perf_counter()
            assert self.anchor_match_rgb is not None
            transform_match, align_info = estimate_homography_to_anchor(match_rgb, self.anchor_match_rgb)
            transform = transform_match.astype(np.float32)
            if self.rgb is not None and self.valid is not None:
                transform, refine_info = _refine_transform_to_canvas(match_rgb, transform, self.rgb, self.valid)
                transform_match = transform
                align_info.update(refine_info)
            metrics.align2d_ms = _ms(t0)
        self.transforms[packet.frame_id] = transform.astype(np.float32)
        metrics.homography_inliers = int(align_info.get("inliers", 0))
        metrics.homography_error_px = align_info.get("median_error")
        metrics.fallback_reason = align_info.get("fallback")

        t0 = perf_counter()
        change_mask, warped_rgb, valid_warp, debug = self._change_mask(rgb_f, transform)
        unreliable_alignment = self.depth is not None and align_info.get("fallback") is not None
        if unreliable_alignment:
            logger.warning("Skipping update for unreliable 2D alignment: frame_id=%s fallback=%s", packet.frame_id, align_info.get("fallback"))
            change_mask = np.zeros_like(change_mask, dtype=bool)
            debug["photometric_changed_ratio"] = 0.0
            debug["support_changed_ratio"] = 0.0
            debug["unreliable_alignment"] = True
        metrics.changed_ratio = float(change_mask.sum() / max(change_mask.size, 1))
        metrics.photometric_changed_ratio = float(debug.get("photometric_changed_ratio", 0.0))
        metrics.support_changed_ratio = float(debug.get("support_changed_ratio", 0.0))
        metrics.diff_ms = _ms(t0)
        self.last_debug = {
            **debug,
            "frame_id": packet.frame_id,
            "warped_rgb": warped_rgb,
            "change_mask": change_mask,
            "valid_warp": valid_warp,
            "reference_rgb": self.rgb.copy() if self.rgb is not None else None,
        }
        if self.support is None:
            self.support = valid_warp.copy()
        elif not unreliable_alignment:
            self.support |= valid_warp

        if self.depth is not None and metrics.changed_ratio <= self.cfg.change.no_change_ratio:
            metrics.skipped_model = True
            metrics.point_count = int(self.valid.sum()) if self.valid is not None else 0
            metrics.total_ms = _ms(start)
            return metrics

        t0 = perf_counter()
        fusion_mask = _seam_fusion_mask(change_mask, self.valid, valid_warp, radius=8)
        # A few isolated photometric pixels can otherwise stretch the bounding
        # box to the full canvas.  They do not provide enough geometry to merit
        # a model invocation, so remove them before cropping the model inputs.
        fusion_mask = _filter_components(fusion_mask, min_pixels=256)
        anchor_ring = _anchor_ring_mask(
            change_mask,
            self.valid,
            valid_warp,
            inner_radius=8,
            outer_radius=32,
        )
        self.last_debug["fusion_mask"] = fusion_mask
        self.last_debug["anchor_ring"] = anchor_ring
        metrics.anchor_pixels = int(anchor_ring.sum())
        if self.depth is not None and not fusion_mask.any():
            metrics.skipped_model = True
            metrics.fallback_reason = "filtered_small_change"
            metrics.point_count = int(self.valid.sum()) if self.valid is not None else 0
            metrics.total_ms = _ms(start)
            return metrics
        if self.depth is None:
            model_rgb, model_to_match = _resize_full_canvas_for_model(
                raw_match_rgb,
                self.roi_width,
                self.roi_height,
                self.cfg.omni.patch_multiple,
            )
            model_to_canvas = _translation_homography(
                self.canvas_pad_left,
                self.canvas_pad_top,
            ) @ model_to_match
            metrics.roi_height, metrics.roi_width = model_rgb.shape[:2]
            self.last_debug["roi_shape"] = [int(metrics.roi_height), int(metrics.roi_width)]
            pred = self.backend.run_window(_single_frame_batch(model_rgb))
            metrics.model_ms = _ms(t0)
            depth, conf = _depth_conf_from_prediction_frame(pred, 0)
            t0 = perf_counter()
            warped_depth = _warp(depth, model_to_canvas, rgb_f.shape[:2], is_mask=False)
            warped_conf = _warp(conf, model_to_canvas, rgb_f.shape[:2], is_mask=False)
            warped_roi_rgb = _warp(model_rgb, model_to_canvas, rgb_f.shape[:2], is_mask=False)
            warped_roi_valid = _warp(np.ones(model_rgb.shape[:2], dtype=np.float32), model_to_canvas, rgb_f.shape[:2], is_mask=True) > 0.5
            metrics.depth_align_ms = _ms(t0)
        else:
            assert self.anchor_match_rgb is not None
            anchor_roi, current_roi, roi_to_canvas = _crop_aligned_roi_pair(
                match_rgb,
                self.anchor_match_rgb,
                fusion_mask,
                transform_match,
                self.roi_width,
                self.roi_height,
                self.cfg.omni.patch_multiple,
            )
            metrics.roi_height, metrics.roi_width = current_roi.shape[:2]
            self.last_debug["roi_shape"] = [int(metrics.roi_height), int(metrics.roi_width)]
            pred = self.backend.run_window(_two_frame_batch(anchor_roi, current_roi))
            metrics.model_ms = _ms(t0)
            depth, conf = _depth_conf_from_prediction_frame(pred, 1)
            t0 = perf_counter()
            warped_depth = _warp(depth, roi_to_canvas, rgb_f.shape[:2], is_mask=False)
            warped_conf = _warp(conf, roi_to_canvas, rgb_f.shape[:2], is_mask=False)
            warped_roi_rgb = _warp(current_roi, roi_to_canvas, rgb_f.shape[:2], is_mask=False)
            # The model sees a resized crop.  The outermost pixels of that
            # crop are interpolation/extrapolation support, not reliable
            # geometry.  The crop itself has 32 px of context, so removing a
            # small model-space border cannot remove the actual changed core.
            model_support = _model_roi_support(current_roi.shape[:2], margin=8)
            warped_roi_valid = _warp(model_support.astype(np.float32), roi_to_canvas, rgb_f.shape[:2], is_mask=True) > 0.5
            metrics.depth_align_ms = _ms(t0)

        # Match the latest offline reconstruction path: confidence-percentile
        # gating removes low-quality model pixels before they reach the map,
        # and distance feathering suppresses the crop/support boundary.
        candidate_valid = (
            valid_warp
            & warped_roi_valid
            & np.isfinite(warped_depth)
            & (warped_depth > 0.0)
            & np.isfinite(warped_conf)
        )
        self.last_debug["candidate_valid"] = candidate_valid
        self.last_debug["depth_positive"] = np.isfinite(warped_depth) & (warped_depth > 0.0)
        self.last_debug["warped_confidence"] = warped_conf.copy()
        alignment_conf = warped_conf.copy()
        if self.depth is None:
            # The first frame establishes the canvas.  Keep all finite,
            # supported samples here; the exporter performs its own narrow
            # border trim and this avoids making the initial cloud sparse.
            confidence_threshold = float(self.cfg.fuse.min_conf)
            quality_valid = candidate_valid
            model_valid = candidate_valid
            feather = np.ones(candidate_valid.shape, dtype=np.float32)
        else:
            confidence_threshold = _model_confidence_threshold(
                warped_conf,
                candidate_valid,
                min_conf=self.cfg.fuse.min_conf,
                percentile=20.0,
            )
            # Keep the stricter set for depth calibration only.  Newly
            # exposed geometry is commonly lower-confidence than the stable
            # ring, so applying this gate to the write mask drops the very
            # boundary we are trying to reconstruct.
            quality_valid = candidate_valid & (warped_conf >= confidence_threshold)
            model_valid = candidate_valid & (warped_conf >= float(self.cfg.fuse.min_conf))
            feather = _feather_mask(model_valid)
            # Feathering is a weight, not a hard mask.  Hard-cutting at 0.08
            # removes thin newly exposed regions altogether.  The ROI border
            # margin above already excludes interpolation support.
            warped_conf = np.where(
                model_valid,
                np.maximum(warped_conf * feather, float(self.cfg.fuse.min_conf)),
                0.0,
            ).astype(np.float32)
        self.last_debug["model_valid"] = model_valid
        self.last_debug["quality_valid"] = quality_valid
        self.last_debug["model_feather"] = feather
        self.last_debug["model_confidence_threshold"] = float(confidence_threshold)

        t_align = perf_counter()
        if self.depth is None:
            aligned_depth = warped_depth.astype(np.float32)
        else:
            aligned_depth = self._align_depth(
                warped_depth,
                alignment_conf,
                model_valid,
                anchor_mask=anchor_ring & quality_valid,
            )
        metrics.depth_align_ms = _ms(t_align)

        # ``fusion_mask`` is deliberately larger than the changed region so
        # the model receives stable context.  It must not also be the commit
        # mask: writing that context back re-fuses old geometry along the ROI
        # boundary and creates the broken/overlapping strip visible in the
        # viewer.  The anchor ring is used for alignment only; the changed
        # core is the only new observation committed to the canvas.
        t_fuse = perf_counter()
        support_change_mask = self.last_debug.get("support_change_mask")
        photometric_change_mask = self.last_debug.get("photometric_change_mask")
        if support_change_mask is None:
            support_change_mask = np.zeros_like(change_mask, dtype=bool)
        if photometric_change_mask is None:
            photometric_change_mask = np.zeros_like(change_mask, dtype=bool)
        # A photometric-only change on an existing cell is not a new surface.
        # It is deliberately left in the model ROI for alignment context, but
        # it is not committed to the canvas.  This keeps exposure/vignetting
        # changes from becoming a second dark layer.
        photo_only_existing = photometric_change_mask & ~support_change_mask
        if self.valid is not None:
            photo_only_existing &= self.valid
        update_mask = model_valid & change_mask & ~photo_only_existing
        if self.valid is not None and anchor_ring.any():
            update_mask &= ~anchor_ring
        self.last_debug["write_mask"] = update_mask
        # The model/depth commit remains limited to the changed core.  For RGB
        # only, build a bridge through the old overlap up to (but never into)
        # the anchor ring.  This is the missing part between a newly exposed
        # bright source edge and a stale dark canvas edge: it lets the current
        # texture reach the seam while the untouched ring still anchors the
        # canvas.  No model depth is written by this bridge.
        color_bridge_mask = np.zeros_like(update_mask, dtype=bool)
        color_bridge_mix = np.zeros_like(update_mask, dtype=np.float32)
        color_bridge_old: np.ndarray | None = None
        color_apply_mask = update_mask
        if (
            cv2 is not None
            and self.rgb is not None
            and self.valid is not None
            and support_change_mask.any()
            and anchor_ring.any()
        ):
            bridge_radius = 32
            bridge_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (bridge_radius * 2 + 1, bridge_radius * 2 + 1),
            )
            expanded_support = cv2.dilate(
                support_change_mask.astype(np.uint8),
                bridge_kernel,
                iterations=1,
            ).astype(bool)
            color_bridge_mask = (
                expanded_support
                & self.valid
                & valid_warp
                & ~anchor_ring
                & ~update_mask
            )
            if color_bridge_mask.any():
                color_bridge_old = self.rgb[color_bridge_mask].copy()
                distance_to_ring = cv2.distanceTransform(
                    (~anchor_ring).astype(np.uint8),
                    cv2.DIST_L2,
                    3,
                )
                # Keep the current aligned source through the old overlap.
                # Only the last few pixels immediately before the untouched
                # anchor ring fade back to the old canvas.  Using a ratio of
                # distances to both masks made the old color dominate a wide
                # inner band when the support boundary was irregular.
                color_bridge_mix = np.clip(
                    distance_to_ring / 8.0,
                    0.0,
                    1.0,
                ).astype(np.float32)
                color_apply_mask = update_mask | color_bridge_mask
        color_update_mask = update_mask
        # The ROI is a model-space input and may contain resize/warp padding
        # at its crop boundary.  It must never be used as display texture.
        # Commit color from the full-resolution current image already aligned
        # to the canonical canvas; this removes the visible dark/duplicated
        # strip while keeping the model ROI responsible only for depth.
        fused_rgb = self.last_debug.get("warped_texture", warped_rgb).copy()
        if self.rgb is not None and self.valid is not None:
            fused_rgb, seam_mix = _anchor_texture_transfer(
                fused_rgb,
                self.rgb,
                self.valid,
                valid_warp,
                color_apply_mask,
                support_change_mask,
                anchor_ring,
            )
            if color_bridge_mask.any() and color_bridge_old is not None:
                current_bridge = fused_rgb[color_bridge_mask]
                alpha = color_bridge_mix[color_bridge_mask, None]
                fused_rgb[color_bridge_mask] = (
                    current_bridge * alpha + color_bridge_old * (1.0 - alpha)
                ).astype(np.float32)
            self.last_debug["anchor_texture_mix"] = seam_mix
            # Keep the anchor ring as a diagnostic mask, but do not apply a
            # global photometric gain to changed pixels.  In this sequence the
            # aligned RGB already contains the real texture; a global gain
            # estimated from a narrow ring can turn a local illumination or
            # exposure difference into a synthetic dark strip.
            color_anchor_mask = anchor_ring & self.valid & np.isfinite(fused_rgb).all(axis=2)
            self.last_debug["color_anchor_mask"] = color_anchor_mask
        self.last_debug["photo_only_existing"] = photo_only_existing
        self.last_debug["color_update_mask"] = color_update_mask
        self.last_debug["color_bridge_mask"] = color_bridge_mask
        self.last_debug["color_bridge_mix"] = color_bridge_mix
        self.last_debug["fused_rgb"] = fused_rgb
        self.last_debug["model_roi_rgb"] = warped_roi_rgb
        # The unchanged annulus is also a local surface constraint.  The
        # model is still authoritative in the changed core, but its first
        # pixels next to an old/new boundary are blended toward a smooth
        # continuation of the old depth field.  Without this step a valid
        # model ROI can introduce a second, parallel sheet at the seam.
        continuity_mask = update_mask & support_change_mask
        if self.depth is not None and self.valid is not None and anchor_ring.any():
            aligned_depth, continuity_delta = _anchor_depth_continuity(
                aligned_depth,
                self.depth,
                self.valid,
                continuity_mask,
                anchor_ring,
                model_confidence=warped_conf,
            )
            self.last_debug["depth_continuity_mask"] = continuity_mask
            self.last_debug["depth_continuity_delta"] = continuity_delta
        fused = self._fuse(fused_rgb, aligned_depth, warped_conf, update_mask, color_update_mask=color_update_mask)
        if color_bridge_mask.any() and self.rgb is not None:
            # ``_fuse`` does not see RGB-only bridge cells.  Commit their
            # already feathered source color after the depth/core update.
            self.rgb[color_bridge_mask] = fused_rgb[color_bridge_mask]
        metrics.fused_pixels = fused
        metrics.fuse_ms = _ms(t_fuse)
        metrics.point_count = int(self.valid.sum()) if self.valid is not None else 0
        metrics.total_ms = _ms(start)
        return metrics

    def save_last_debug(self, output_dir: str | Path, image_name: str | None = None) -> dict[str, str]:
        """Save the most recent change-mask diagnostics as PNG files.

        Complexity:
            O(HW) for image conversion and overlay generation.
        """

        if not self.last_debug:
            return {}
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_id = int(self.last_debug.get("frame_id", -1))
        stem = f"frame_{frame_id:03d}"
        if image_name:
            stem += "_" + Path(image_name).stem

        warped_rgb = self.last_debug.get("warped_rgb")
        warped_texture = self.last_debug.get("warped_texture")
        fused_rgb = self.last_debug.get("fused_rgb")
        model_roi_rgb = self.last_debug.get("model_roi_rgb")
        reference_rgb = self.last_debug.get("reference_rgb")
        change_mask = self.last_debug.get("change_mask")
        fusion_mask = self.last_debug.get("fusion_mask")
        anchor_ring = self.last_debug.get("anchor_ring")
        diff = self.last_debug.get("diff")
        current_support = self.last_debug.get("current_support")
        paths: dict[str, str] = {}

        if warped_rgb is not None:
            path = out_dir / f"{stem}_warped_current.png"
            _save_rgb(path, warped_rgb)
            paths["warped_current"] = str(path)
        if warped_texture is not None:
            path = out_dir / f"{stem}_warped_texture.png"
            _save_rgb(path, warped_texture)
            paths["warped_texture"] = str(path)
        if fused_rgb is not None:
            path = out_dir / f"{stem}_fused_rgb.png"
            _save_rgb(path, fused_rgb)
            paths["fused_rgb"] = str(path)
        if model_roi_rgb is not None:
            path = out_dir / f"{stem}_model_roi_rgb.png"
            _save_rgb(path, model_roi_rgb)
            paths["model_roi_rgb"] = str(path)
        if reference_rgb is not None:
            path = out_dir / f"{stem}_reference_canvas.png"
            _save_rgb(path, reference_rgb)
            paths["reference_canvas"] = str(path)
        if diff is not None:
            path = out_dir / f"{stem}_diff.png"
            _save_gray(path, diff)
            paths["diff"] = str(path)
        if current_support is not None:
            path = out_dir / f"{stem}_support.png"
            _save_mask(path, current_support)
            paths["support"] = str(path)
        if change_mask is not None:
            path = out_dir / f"{stem}_change_mask.png"
            _save_mask(path, change_mask)
            paths["change_mask"] = str(path)
        if fusion_mask is not None:
            path = out_dir / f"{stem}_fusion_mask.png"
            _save_mask(path, fusion_mask)
            paths["fusion_mask"] = str(path)
        if anchor_ring is not None:
            path = out_dir / f"{stem}_anchor_ring.png"
            _save_mask(path, anchor_ring)
            paths["anchor_ring"] = str(path)
        color_update_mask = self.last_debug.get("color_update_mask")
        if color_update_mask is not None:
            path = out_dir / f"{stem}_color_update_mask.png"
            _save_mask(path, color_update_mask)
            paths["color_update_mask"] = str(path)
        color_bridge_mask = self.last_debug.get("color_bridge_mask")
        if color_bridge_mask is not None:
            path = out_dir / f"{stem}_color_bridge_mask.png"
            _save_mask(path, color_bridge_mask)
            paths["color_bridge_mask"] = str(path)
        color_bridge_mix = self.last_debug.get("color_bridge_mix")
        if color_bridge_mix is not None:
            path = out_dir / f"{stem}_color_bridge_mix.png"
            _save_gray(path, color_bridge_mix)
            paths["color_bridge_mix"] = str(path)
        for debug_name in ("support_change_mask", "photometric_change_mask", "photo_only_existing"):
            debug_mask = self.last_debug.get(debug_name)
            if debug_mask is not None:
                path = out_dir / f"{stem}_{debug_name}.png"
                _save_mask(path, debug_mask)
                paths[debug_name] = str(path)
        texture_mix = self.last_debug.get("anchor_texture_mix")
        if texture_mix is not None:
            path = out_dir / f"{stem}_anchor_texture_mix.png"
            _save_gray(path, texture_mix)
            paths["anchor_texture_mix"] = str(path)
        continuity_mask = self.last_debug.get("depth_continuity_mask")
        if continuity_mask is not None:
            path = out_dir / f"{stem}_depth_continuity_mask.png"
            _save_mask(path, continuity_mask)
            paths["depth_continuity_mask"] = str(path)
        continuity_delta = self.last_debug.get("depth_continuity_delta")
        if continuity_delta is not None:
            path = out_dir / f"{stem}_depth_continuity_delta.png"
            _save_gray(path, continuity_delta)
            paths["depth_continuity_delta"] = str(path)
        model_valid = self.last_debug.get("model_valid")
        if model_valid is not None:
            path = out_dir / f"{stem}_model_valid.png"
            _save_mask(path, model_valid)
            paths["model_valid"] = str(path)
        quality_valid = self.last_debug.get("quality_valid")
        if quality_valid is not None:
            path = out_dir / f"{stem}_quality_valid.png"
            _save_mask(path, quality_valid)
            paths["quality_valid"] = str(path)
        candidate_valid = self.last_debug.get("candidate_valid")
        if candidate_valid is not None:
            path = out_dir / f"{stem}_candidate_valid.png"
            _save_mask(path, candidate_valid)
            paths["candidate_valid"] = str(path)
        depth_positive = self.last_debug.get("depth_positive")
        if depth_positive is not None:
            path = out_dir / f"{stem}_depth_positive.png"
            _save_mask(path, depth_positive)
            paths["depth_positive"] = str(path)
        warped_confidence = self.last_debug.get("warped_confidence")
        if warped_confidence is not None:
            path = out_dir / f"{stem}_warped_confidence.png"
            _save_gray(path, warped_confidence)
            paths["warped_confidence"] = str(path)
        write_mask = self.last_debug.get("write_mask")
        if write_mask is not None:
            path = out_dir / f"{stem}_write_mask.png"
            _save_mask(path, write_mask)
            paths["write_mask"] = str(path)
        if warped_rgb is not None and change_mask is not None:
            path = out_dir / f"{stem}_overlay.png"
            _save_overlay(path, warped_rgb, change_mask)
            paths["overlay"] = str(path)
        if warped_rgb is not None and fusion_mask is not None:
            path = out_dir / f"{stem}_fusion_overlay.png"
            _save_overlay(path, warped_rgb, fusion_mask)
            paths["fusion_overlay"] = str(path)
        return paths

    def export_pointcloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Export the current canvas as one point per valid XY cell."""

        if self.depth is None or self.rgb is None or self.valid is None:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        return export_canvas_pointcloud(self.depth, self.rgb, self.valid)

    def _change_mask(self, rgb_f: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Compare current aligned image against the fused canvas."""

        if self.rgb is None or self.valid is None:
            support = _foreground_mask(rgb_f)
            # The input image is the authoritative texture.  The support rim
            # may be darker because of the camera optics, but lifting it here
            # creates a color that does not exist in the source frame and can
            # become a visible halo after the first canvas write.
            texture = _rgb_float(rgb_f).astype(np.float32, copy=True)
            return support.copy(), rgb_f, support, {
                "diff": np.zeros(rgb_f.shape[:2], dtype=np.float32),
                "current_support": support,
                "warped_texture": texture,
                "support_changed_ratio": float(support.sum() / max(support.size, 1)),
                "photometric_changed_ratio": 0.0,
                "robust_l1_threshold": None,
            }
        warped_rgb = _warp(rgb_f, transform, rgb_f.shape[:2], is_mask=False)
        source_support = _foreground_mask(rgb_f)
        # Do not synthesize a corrected edge color before alignment.  The
        # source frame remains the only color observation; the anchor ring is
        # used later only to remove a measured seam residual.
        texture_rgb = _rgb_float(rgb_f).astype(np.float32, copy=True)
        warped_texture = _warp(texture_rgb, transform, rgb_f.shape[:2], is_mask=False)
        valid_warp = _warp(source_support.astype(np.float32), transform, rgb_f.shape[:2], is_mask=True) > 0.5
        reference_support = self.support if self.support is not None else self.valid
        overlap = valid_warp & reference_support & self.valid
        diff = np.zeros(rgb_f.shape[:2], dtype=np.float32)
        diff[overlap] = np.mean(np.abs(warped_rgb[overlap] - self.rgb[overlap]), axis=1)
        support_change = valid_warp & ~reference_support
        robust_thr: float | None = None
        photometric_change = np.zeros_like(valid_warp, dtype=bool)
        if int(overlap.sum()) > 0:
            vals = diff[overlap]
            center = float(np.median(vals))
            mad = max(float(np.median(np.abs(vals - center))), 1e-6)
            robust_thr = max(float(self.cfg.change.image_l1_thr) * 2.0, center + 3.0 * 1.4826 * mad)
            robust_thr = min(max(robust_thr, 0.08), 0.22)
            photometric_change = overlap & (diff > robust_thr)
            photometric_change = _filter_components(photometric_change, min_pixels=128)

        support_ratio = float(support_change.sum() / max(support_change.size, 1))
        photo_ratio = float(photometric_change.sum() / max(photometric_change.size, 1))
        if photo_ratio > self.cfg.change.scene_jump_ratio or (photo_ratio > 0.15 and support_ratio < 0.05):
            logger.warning(
                "Suppressing oversized photometric mask: photo_ratio=%.4f support_ratio=%.4f threshold=%s",
                photo_ratio,
                support_ratio,
                robust_thr,
            )
            photometric_change[:] = False
            photo_ratio = 0.0

        # Keep the two causes separate.  A support change is a genuinely new
        # canvas area and may contribute a new surface/texture.  A pure
        # photometric change is only a different exposure of an already valid
        # surface; replacing that area with the darker image is what creates
        # the apparent overlapping strip in the replay.
        change = support_change | photometric_change
        change = _dilate(change, self.cfg.change.dilate_ksize) & valid_warp
        support_commit = _dilate(support_change, self.cfg.change.dilate_ksize) & valid_warp
        photo_commit = _dilate(photometric_change, self.cfg.change.dilate_ksize) & overlap
        return change, warped_rgb, valid_warp, {
            "diff": diff,
            "current_support": valid_warp,
            "warped_texture": warped_texture,
            "support_changed_ratio": support_ratio,
            "photometric_changed_ratio": photo_ratio,
            "support_change_mask": support_commit,
            "photometric_change_mask": photo_commit,
            "robust_l1_threshold": robust_thr,
        }

    def _align_depth(
        self,
        warped_depth: np.ndarray,
        warped_conf: np.ndarray,
        valid_warp: np.ndarray,
        anchor_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Align new depth from a stable unchanged ring into the canvas.

        The changed region is excluded from the calibration statistics. Its
        surrounding unchanged pixels define the local scale/offset and the
        residual field used to make the transition into the new surface
        continuous.
        """

        if self.depth is None or self.valid is None:
            return warped_depth.astype(np.float32)
        calibration_mask = valid_warp if anchor_mask is None else (anchor_mask & valid_warp)
        overlap = calibration_mask & self.valid & np.isfinite(warped_depth) & np.isfinite(self.depth) & (warped_conf > self.cfg.fuse.min_conf)
        # A narrow ring can disappear when a change touches the image border.
        # Retain the broad-overlap fallback in that case instead of dropping
        # depth alignment altogether.
        if anchor_mask is not None and int(overlap.sum()) < 128:
            calibration_mask = valid_warp
            overlap = calibration_mask & self.valid & np.isfinite(warped_depth) & np.isfinite(self.depth) & (warped_conf > self.cfg.fuse.min_conf)
        self.last_debug["depth_alignment_anchor_pixels"] = int(overlap.sum())
        if int(overlap.sum()) < 128:
            return warped_depth.astype(np.float32)
        src = warped_depth[overlap].astype(np.float64)
        dst = self.depth[overlap].astype(np.float64)
        weights = np.maximum(warped_conf[overlap].astype(np.float64), 1e-4)
        lo, hi = np.percentile(src, [2, 98])
        keep = (src >= lo) & (src <= hi)
        src = src[keep]
        dst = dst[keep]
        weights = weights[keep]
        if src.size < 128:
            return warped_depth.astype(np.float32)

        # Keep the same affine depth calibration as the latest offline
        # reconstruction.  Spatial X/Y terms over a narrow ring overfit the
        # boundary and can turn a smooth plane into a raised strip.
        design = np.stack([src, np.ones_like(src)], axis=1)
        keep_fit = np.ones(src.shape, dtype=bool)
        coeff = np.array([1.0, float(np.median(dst - src))], dtype=np.float64)
        for _ in range(4):
            if int(keep_fit.sum()) < 128:
                break
            w = np.sqrt(weights[keep_fit])
            try:
                coeff = np.linalg.lstsq(design[keep_fit] * w[:, None], dst[keep_fit] * w, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            residual_fit = dst - (design @ coeff)
            center = float(np.median(residual_fit[keep_fit]))
            mad = max(float(np.median(np.abs(residual_fit[keep_fit] - center))), 1e-6)
            keep_fit = np.abs(residual_fit - center) <= 3.0 * 1.4826 * mad
        scale, bias = float(coeff[0]), float(coeff[1])
        if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0.0 or scale > 8.0:
            scale = 1.0
            bias = float(np.median(dst - src))
        scale = float(np.clip(scale, 0.25, 4.0))
        bias = float(np.clip(bias, -10.0, 10.0))
        aligned = warped_depth * scale + bias
        aligned = aligned.astype(np.float32)
        seam = calibration_mask & self.valid & np.isfinite(aligned) & np.isfinite(self.depth)
        if int(seam.sum()) >= 64:
            seam_residual = (self.depth - aligned).astype(np.float32)
            spatial_correction = _fit_spatial_seam_residual(
                residual=seam_residual,
                seam_mask=seam,
                target_mask=valid_warp & np.isfinite(aligned),
                max_abs=0.08,
            )
            propagated_correction = _propagate_seam_residual(
                residual=seam_residual,
                seam_mask=seam,
                target_mask=valid_warp & np.isfinite(aligned),
                sigma=24.0,
                max_abs=0.08,
            )
            # The fitted low-order field carries the depth-origin/tilt
            # correction into newly exposed pixels.  Keep a smaller fraction
            # of the local nearest-ring field for fine alignment without
            # letting one noisy anchor pixel bend the whole patch.
            correction = np.clip(
                0.78 * spatial_correction + 0.22 * propagated_correction,
                -0.08,
                0.08,
            ).astype(np.float32)
            aligned = (aligned + correction).astype(np.float32)
            self.last_debug["depth_alignment_spatial_correction"] = correction
        self.last_debug["depth_alignment_scale"] = float(scale)
        self.last_debug["depth_alignment_bias"] = float(bias)
        return aligned

    def _fuse(
        self,
        warped_rgb: np.ndarray,
        depth: np.ndarray,
        conf: np.ndarray,
        update_mask: np.ndarray,
        color_update_mask: np.ndarray | None = None,
    ) -> int:
        """Vectorized single-layer weighted depth/color fusion."""

        valid_obs = update_mask & np.isfinite(depth) & (conf >= self.cfg.fuse.min_conf)
        if self.depth is None:
            shape = depth.shape
            self.depth = np.zeros(shape, dtype=np.float32)
            self.conf = np.zeros(shape, dtype=np.float32)
            self.weight = np.zeros(shape, dtype=np.float32)
            self.valid = np.zeros(shape, dtype=bool)
            self.rgb = np.zeros((*shape, 3), dtype=np.float32)
        assert self.weight is not None and self.conf is not None and self.valid is not None and self.rgb is not None
        idx = valid_obs
        if not idx.any():
            return 0
        color_obs = valid_obs if color_update_mask is None else valid_obs & color_update_mask
        w_old = self.weight[idx]
        w_obs = np.clip(conf[idx], 1e-4, self.cfg.fuse.w_max)
        w_new = np.minimum(w_old + w_obs, self.cfg.fuse.w_max)
        # ``update_mask`` contains only the changed core (the surrounding
        # anchor ring has already been excluded).  Averaging the old canvas
        # depth with the new, anchor-calibrated surface here creates a third
        # artificial surface whenever the scene moved in depth: the viewer
        # then shows a narrow raised/duplicated band along the change edge.
        # Commit the calibrated observation directly for changed cells; the
        # anchor ring remains the continuity constraint and is never written.
        self.depth[idx] = depth[idx]
        self.conf[idx] = np.maximum(self.conf[idx], conf[idx])
        self.weight[idx] = w_new
        self.valid[idx] = True
        if color_obs.any():
            # Do not average a changed surface with stale canvas texture.  A
            # direct write preserves the true RGB and prevents a one-frame
            # dark/duplicated strip from surviving future replays.
            self.rgb[color_obs] = warped_rgb[color_obs]
        return int(idx.sum())


def export_canvas_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Export a cleaned point cloud from aligned canvas buffers.

    This is the shared post-processing path for both the normal PLY export and
    optional live viewers.  Keeping it outside ``AlignedCanvasStream`` avoids
    viewers accidentally displaying the sparse pre-export canvas.
    """

    depth_map, rgb_map, valid_map = _fill_narrow_gaps(depth, rgb, valid, pixels=5)
    depth_map = _regularize_heightfield(depth_map, valid_map, sigma=4.0, raw_keep=0.15)
    height, width = depth_map.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    valid_map = valid_map & np.isfinite(depth_map)
    # Only remove the small interpolation fringe introduced by warping.  A
    # percentage of the whole canvas is unsafe here: with the padded data2
    # canvas, 2.5% meant a 16 px erosion and discarded legitimate newly
    # exposed geometry along the left/top frame boundary, which appeared as a
    # black missing band in the point-cloud viewer.
    trim_px = max(3, min(4, int(round(min(height, width) * 0.006))))
    valid_map = _erode_mask(valid_map, trim_px)
    scale = float(max(height, width))
    x = (xx[valid_map] - width * 0.5) / scale
    y = -(yy[valid_map] - height * 0.5) / scale
    z = depth_map[valid_map].astype(np.float32)
    z = _plane_residual(x, y, z)
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = np.clip(rgb_map[valid_map] * 255.0, 0, 255).astype(np.uint8)
    return points, colors


def estimate_homography_to_anchor(curr_rgb: np.ndarray, anchor_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a current-to-anchor homography with SIFT/ORB and RANSAC."""

    if cv2 is None:
        return np.eye(3, dtype=np.float32), {"inliers": 0, "median_error": None, "fallback": "no_cv2"}
    curr_gray = _gray_u8(curr_rgb)
    anchor_gray = _gray_u8(anchor_rgb)
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=1500)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=1500)
        norm = cv2.NORM_HAMMING
    kp_curr, des_curr = detector.detectAndCompute(curr_gray, None)
    kp_anchor, des_anchor = detector.detectAndCompute(anchor_gray, None)
    if des_curr is None or des_anchor is None or len(kp_curr) < 8 or len(kp_anchor) < 8:
        return _phase_translation(curr_gray, anchor_gray, "few_features")
    matcher = cv2.BFMatcher(norm)
    matches = matcher.knnMatch(des_curr, des_anchor, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return _phase_translation(curr_gray, anchor_gray, "few_matches")
    pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good])
    pts_anchor = np.float32([kp_anchor[m.trainIdx].pt for m in good])
    homography, inlier_mask = cv2.findHomography(pts_curr, pts_anchor, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None or int(inlier_mask.sum()) < 8:
        return _phase_translation(curr_gray, anchor_gray, "bad_homography")
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(pts_curr[inliers, None, :], homography)[:, 0, :]
    errors = np.linalg.norm(projected - pts_anchor[inliers], axis=1)
    return homography.astype(np.float32), {"inliers": int(inliers.sum()), "median_error": float(np.median(errors)), "fallback": None}


def _refine_transform_to_canvas(
    curr_rgb: np.ndarray,
    transform: np.ndarray,
    canvas_rgb: np.ndarray,
    canvas_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine current-to-canvas alignment using the already fused overlap band.

    Complexity:
        O(HW). This adds a small translation correction after the coarse
        homography so newly exposed regions share a tight boundary with the old
        canvas.
    """

    if cv2 is None:
        return transform.astype(np.float32), {"overlap_refine_dx": 0.0, "overlap_refine_dy": 0.0, "overlap_refine_response": None}
    warped = _warp(curr_rgb, transform, canvas_rgb.shape[:2], is_mask=False)
    support = _warp(_foreground_mask(curr_rgb).astype(np.float32), transform, canvas_rgb.shape[:2], is_mask=True) > 0.5
    overlap = support & canvas_valid
    if int(overlap.sum()) < 2048:
        return transform.astype(np.float32), {"overlap_refine_dx": 0.0, "overlap_refine_dy": 0.0, "overlap_refine_response": None}
    # Refine translation from the unchanged overlap only.  If the changed
    # patch participates in phase correlation it can pull the whole ROI a few
    # pixels, which later appears as a seam even when the depth alignment is
    # correct.
    diff_map = np.mean(np.abs(warped - canvas_rgb), axis=2)
    overlap_values = diff_map[overlap]
    center = float(np.median(overlap_values))
    mad = max(float(np.median(np.abs(overlap_values - center))), 1e-6)
    stable = overlap & (diff_map <= max(0.08, center + 3.0 * 1.4826 * mad))
    if int(stable.sum()) >= 2048:
        overlap = stable
    ys, xs = np.nonzero(overlap)
    pad = 24
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(overlap.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(overlap.shape[1], int(xs.max()) + pad + 1)
    if y1 - y0 < 64 or x1 - x0 < 64:
        return transform.astype(np.float32), {"overlap_refine_dx": 0.0, "overlap_refine_dy": 0.0, "overlap_refine_response": None}

    a = _gray_float(warped[y0:y1, x0:x1])
    b = _gray_float(canvas_rgb[y0:y1, x0:x1])
    m = overlap[y0:y1, x0:x1].astype(np.float32)
    if float(m.mean()) < 0.08:
        return transform.astype(np.float32), {"overlap_refine_dx": 0.0, "overlap_refine_dy": 0.0, "overlap_refine_response": None}
    a = (a - float(np.mean(a[m > 0]))) * m
    b = (b - float(np.mean(b[m > 0]))) * m
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a.astype(np.float32) * window, b.astype(np.float32) * window)
    dx = float(np.clip(shift[0], -8.0, 8.0))
    dy = float(np.clip(shift[1], -8.0, 8.0))
    if not np.isfinite(dx) or not np.isfinite(dy) or response < 0.02:
        return transform.astype(np.float32), {"overlap_refine_dx": 0.0, "overlap_refine_dy": 0.0, "overlap_refine_response": float(response)}
    correction = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)
    refined = correction @ transform.astype(np.float32)
    return refined.astype(np.float32), {
        "overlap_refine_dx": dx,
        "overlap_refine_dy": dy,
        "overlap_refine_response": float(response),
        "overlap_refine_anchor_pixels": int(overlap.sum()),
    }


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY point cloud."""

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")





def _crop_aligned_roi_pair(
    current_match_rgb: np.ndarray,
    anchor_match_rgb: np.ndarray,
    change_mask: np.ndarray,
    current_to_anchor: np.ndarray,
    max_roi_width: int,
    max_roi_height: int,
    patch_multiple: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the same anchor-canvas ROI from anchor and current aligned images."""

    ys, xs = np.nonzero(change_mask)
    canvas_h, canvas_w = change_mask.shape
    if ys.size == 0:
        x0, y0, x1, y1 = 0, 0, canvas_w, canvas_h
    else:
        x0 = max(0, int(xs.min()) - 32)
        x1 = min(canvas_w, int(xs.max()) + 33)
        y0 = max(0, int(ys.min()) - 32)
        y1 = min(canvas_h, int(ys.max()) + 33)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, canvas_w, canvas_h
    crop_w = x1 - x0
    crop_h = y1 - y0
    anchor_crop = anchor_match_rgb[y0:y1, x0:x1]
    translate = np.array([[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]], dtype=np.float32)
    current_crop = _warp(current_match_rgb, translate @ current_to_anchor.astype(np.float32), (crop_h, crop_w), is_mask=False)
    target_w, target_h = _bucket_roi_size(crop_w, crop_h, max_roi_width, max_roi_height, patch_multiple)
    anchor_roi = _resize_exact(anchor_crop, target_w, target_h)
    current_roi = _resize_exact(current_crop, target_w, target_h)
    roi_to_canvas = np.array(
        [[float(crop_w) / float(target_w), 0.0, float(x0)], [0.0, float(crop_h) / float(target_h), float(y0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return anchor_roi, current_roi, roi_to_canvas


def _resize_exact(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize RGB float image to an exact size."""

    if cv2 is not None:
        return cv2.resize(_rgb_float(rgb), (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    from PIL import Image

    return (np.asarray(Image.fromarray(np.clip(_rgb_float(rgb) * 255.0, 0, 255).astype(np.uint8)).resize((width, height))) / 255.0).astype(np.float32)


def _bucket_roi_size(
    crop_w: int,
    crop_h: int,
    max_width: int,
    max_height: int,
    patch_multiple: int,
) -> tuple[int, int]:
    """Return a patch-compatible ROI size using the full requested bucket.

    The quality path intentionally upsamples a small change crop until one
    requested bucket edge is reached.  The previous ``min(..., 1.0)`` cap
    left small ROIs at their native matching resolution, so a nominal 700
    quality run silently fell back to shapes such as 518x434.  Large crops
    are still downsampled and the result remains bounded by both dimensions.

    Complexity: O(1).
    """

    crop_w = max(int(crop_w), 1)
    crop_h = max(int(crop_h), 1)
    max_width = max(int(max_width), patch_multiple)
    max_height = max(int(max_height), patch_multiple)
    scale = min(float(max_width) / float(crop_w), float(max_height) / float(crop_h))
    target_w = max(patch_multiple, int(np.floor((crop_w * scale) / patch_multiple) * patch_multiple))
    target_h = max(patch_multiple, int(np.floor((crop_h * scale) / patch_multiple) * patch_multiple))
    target_w = min(target_w, max_width)
    target_h = min(target_h, max_height)
    return target_w, target_h

def _resize_full_canvas_for_model(
    rgb: np.ndarray,
    max_width: int,
    max_height: int,
    patch_multiple: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a full anchor canvas to a patch-compatible model image.

    The returned homography maps model pixels back into the original canvas.
    Complexity is O(HW) for the resize.
    """

    canvas_h, canvas_w = rgb.shape[:2]
    target_w, target_h = _bucket_roi_size(canvas_w, canvas_h, max_width, max_height, patch_multiple)
    model_rgb = _resize_exact(rgb, target_w, target_h)
    model_to_canvas = np.array(
        [[float(canvas_w) / float(target_w), 0.0, 0.0], [0.0, float(canvas_h) / float(target_h), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return model_rgb, model_to_canvas


def _round_down_patch(value: int, patch_multiple: int) -> int:
    """Return a positive patch-compatible input extent."""

    patch_multiple = max(int(patch_multiple), 1)
    return max(patch_multiple, (max(int(value), patch_multiple) // patch_multiple) * patch_multiple)


def _crop_roi_for_model(
    raw_rgb: np.ndarray,
    change_mask: np.ndarray,
    match_to_canvas: np.ndarray,
    match_hw: tuple[int, int],
    roi_width: int,
    patch_multiple: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop the changed region from the high-resolution matching image for model inference."""

    match_rgb = _resize_for_matching(raw_rgb, match_hw[1])
    ys, xs = np.nonzero(change_mask)
    if ys.size == 0:
        return match_rgb, match_to_canvas.astype(np.float32), (0, 0)
    canvas_h, canvas_w = change_mask.shape
    match_h, match_w = match_hw
    sx = float(match_w) / float(canvas_w)
    sy = float(match_h) / float(canvas_h)
    x0 = max(0, int(np.floor(xs.min() * sx)) - 32)
    x1 = min(match_w, int(np.ceil((xs.max() + 1) * sx)) + 32)
    y0 = max(0, int(np.floor(ys.min() * sy)) - 32)
    y1 = min(match_h, int(np.ceil((ys.max() + 1) * sy)) + 32)
    if x1 <= x0 or y1 <= y0:
        return match_rgb, match_to_canvas.astype(np.float32), (0, 0)
    crop = match_rgb[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    target_w, target_h = _bucket_roi_size(crop_w, crop_h, roi_width, roi_width, patch_multiple)
    if cv2 is not None:
        roi = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image

        roi = np.asarray(Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8)).resize((target_w, target_h))) / 255.0
    crop_to_match = np.array(
        [[float(crop_w) / float(target_w), 0.0, float(x0)], [0.0, float(crop_h) / float(target_h), float(y0)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    roi_to_match_canvas = match_to_canvas.astype(np.float32) @ crop_to_match
    return roi.astype(np.float32), roi_to_match_canvas, (x0, y0)


def _roi_to_canvas_homography(roi_to_match_canvas: np.ndarray, match_hw: tuple[int, int], canvas_hw: tuple[int, int]) -> np.ndarray:
    """Scale ROI-to-match-anchor coordinates into the active canvas coordinates."""

    match_h, match_w = match_hw
    canvas_h, canvas_w = canvas_hw
    target_scale = np.array(
        [[float(canvas_w) / float(match_w), 0.0, 0.0], [0.0, float(canvas_h) / float(match_h), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (target_scale @ roi_to_match_canvas.astype(np.float32)).astype(np.float32)

def _resize_for_matching(rgb: np.ndarray, target_width: int) -> np.ndarray:
    """Resize RGB to a wider matching canvas while preserving aspect ratio."""

    arr = _rgb_float(rgb)
    height, width = arr.shape[:2]
    if width == target_width:
        return arr
    target_height = max(1, int(round(height * float(target_width) / float(width))))
    if cv2 is not None:
        resized = cv2.resize(arr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        from PIL import Image

        resized = np.asarray(Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).resize((target_width, target_height))) / 255.0
    return resized.astype(np.float32)


def _pad_canvas(
    rgb: np.ndarray,
    pad_left: int,
    pad_top: int,
    pad_right: int,
    pad_bottom: int,
) -> np.ndarray:
    """Place a matching image inside a larger zero-background canvas."""

    arr = _rgb_float(rgb).astype(np.float32, copy=False)
    pads = (
        (max(int(pad_top), 0), max(int(pad_bottom), 0)),
        (max(int(pad_left), 0), max(int(pad_right), 0)),
        (0, 0),
    )
    return np.pad(arr, pads, mode="constant", constant_values=0.0).astype(np.float32)


def _translation_homography(dx: float, dy: float) -> np.ndarray:
    """Return a pixel translation homography."""

    return np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _scale_homography(h_match: np.ndarray, match_hw: tuple[int, int], bucket_hw: tuple[int, int]) -> np.ndarray:
    """Convert a homography estimated at match resolution to bucket resolution."""

    match_h, match_w = match_hw
    bucket_h, bucket_w = bucket_hw
    sx = float(bucket_w) / float(match_w)
    sy = float(bucket_h) / float(match_h)
    scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    inv_scale = np.array([[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (scale @ h_match.astype(np.float32) @ inv_scale).astype(np.float32)

def _single_frame_batch(rgb_f: np.ndarray) -> dict[str, Any]:
    """Build a one-frame backend batch."""

    tensor = normalize_rgb((rgb_f * 255.0).astype(np.uint8))[None, None].contiguous()
    _, _, _, height, width = tensor.shape
    return {
        "images": tensor,
        "depth": torch.zeros((1, 1, height, width, 1), dtype=torch.float32),
        "mask": torch.zeros((1, 1, height, width), dtype=torch.float32),
        "intrinsics": torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3),
        "extrinsics": torch.eye(4, dtype=torch.float32)[:3].reshape(1, 1, 3, 4),
        "camera_gt_index": [],
        "depth_gt_index": [],
    }



def _two_frame_batch(anchor_rgb: np.ndarray, current_rgb: np.ndarray) -> dict[str, Any]:
    """Build a two-frame backend batch for paired anchor/current ROI inference."""

    images = torch.stack([normalize_rgb((anchor_rgb * 255.0).astype(np.uint8)), normalize_rgb((current_rgb * 255.0).astype(np.uint8))], dim=0)[None].contiguous()
    _, s_count, _, height, width = images.shape
    return {
        "images": images,
        "depth": torch.zeros((1, s_count, height, width, 1), dtype=torch.float32),
        "mask": torch.zeros((1, s_count, height, width), dtype=torch.float32),
        "intrinsics": torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, s_count, 1, 1),
        "extrinsics": torch.eye(4, dtype=torch.float32)[:3].reshape(1, 1, 3, 4).repeat(1, s_count, 1, 1),
        "camera_gt_index": [],
        "depth_gt_index": [],
    }


def _depth_conf_from_prediction_frame(pred: Any, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract one frame's depth/conf arrays from an OmniPrediction."""

    depth = _to_numpy(pred.depth if pred.depth is not None else pred.world_points[..., 2:3])
    conf = _to_numpy(pred.depth_conf if pred.depth_conf is not None else pred.world_points_conf)
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 2:
        out_depth = depth
    else:
        out_depth = depth[min(frame_idx, depth.shape[0] - 1)]
    if conf.ndim == 5 and conf.shape[0] == 1:
        conf = conf[0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim == 2:
        out_conf = conf
    else:
        out_conf = conf[min(frame_idx, conf.shape[0] - 1)]
    return out_depth.astype(np.float32), out_conf.astype(np.float32)

def _depth_conf_from_prediction(pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract S=1 depth/conf arrays from an OmniPrediction."""

    depth = _to_numpy(pred.depth if pred.depth is not None else pred.world_points[..., 2:3])
    conf = _to_numpy(pred.depth_conf if pred.depth_conf is not None else pred.world_points_conf)
    if depth.ndim == 5 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]
    if conf.ndim == 3 and conf.shape[0] == 1:
        conf = conf[0]
    return depth.astype(np.float32), conf.astype(np.float32)


def _phase_translation(curr_gray: np.ndarray, anchor_gray: np.ndarray, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Fallback to translation-only phase correlation."""

    shift, response = cv2.phaseCorrelate(np.float32(curr_gray), np.float32(anchor_gray))
    matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]], dtype=np.float32)
    return matrix, {"inliers": 0, "median_error": float(1.0 / max(response, 1e-6)), "fallback": reason}


def _warp(value: np.ndarray, homography: np.ndarray, shape_hw: tuple[int, int], is_mask: bool) -> np.ndarray:
    """Warp an array into the anchor canvas."""

    if cv2 is None:
        return value.copy()
    height, width = shape_hw
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpPerspective(value.astype(np.float32), homography, (width, height), flags=flags)


def _model_roi_support(shape_hw: tuple[int, int], margin: int) -> np.ndarray:
    """Return reliable interior support for a resized model ROI.

    A prediction at the exact crop boundary can be influenced by the crop's
    zero-filled warp border and by resize interpolation.  Keeping a small
    interior margin avoids committing those pixels while retaining the
    context around the changed core.
    """

    height, width = (max(int(value), 1) for value in shape_hw)
    margin = max(int(margin), 0)
    if margin <= 0:
        return np.ones((height, width), dtype=bool)
    if height <= margin * 2 + 2 or width <= margin * 2 + 2:
        return np.ones((height, width), dtype=bool)
    support = np.ones((height, width), dtype=bool)
    support[:margin, :] = False
    support[-margin:, :] = False
    support[:, :margin] = False
    support[:, -margin:] = False
    return support


def _model_confidence_threshold(
    confidence: np.ndarray,
    candidate_valid: np.ndarray,
    min_conf: float,
    percentile: float,
) -> float:
    """Choose a robust per-ROI confidence floor like the offline path."""

    values = confidence[candidate_valid]
    if values.size < 256:
        return float(min_conf)
    percentile_value = float(np.percentile(values, float(np.clip(percentile, 0.0, 100.0))))
    return float(max(float(min_conf), percentile_value))


def _feather_mask(mask: np.ndarray) -> np.ndarray:
    """Build a distance-based confidence feather for a valid model support."""

    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if cv2 is None:
        return mask.astype(np.float32)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    positive = distance > 0.0
    if not positive.any():
        return mask.astype(np.float32)
    scale = max(1.0, float(np.percentile(distance[positive], 85.0)) * 0.35)
    return np.clip(distance / scale, 0.0, 1.0).astype(np.float32)


def _plane_residual(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convert depth to a normalized, clipped low-order surface residual."""

    if z.size < 3:
        return z.astype(np.float32)
    base_depth_scale = max(abs(float(np.median(z))), 1e-6)
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    z64 = z.astype(np.float64)
    a = np.stack([x64, y64, x64 * x64, y64 * y64, x64 * y64, np.ones_like(x64)], axis=1)
    keep = np.ones(z64.shape, dtype=bool)
    coeff = np.linalg.lstsq(a, z64, rcond=None)[0]
    for _ in range(3):
        coeff = np.linalg.lstsq(a[keep], z64[keep], rcond=None)[0]
        residual_iter = z64 - a @ coeff
        center_iter = float(np.median(residual_iter[keep]))
        mad_iter = max(float(np.median(np.abs(residual_iter[keep] - center_iter))), 1e-6)
        keep = np.abs(residual_iter - center_iter) <= 4.0 * 1.4826 * mad_iter
        if int(keep.sum()) < 128:
            keep[:] = True
            break
    residual = z64 - a @ coeff
    center = float(np.median(residual))
    mad = max(float(np.median(np.abs(residual - center))), 1e-6)
    limit = min(3.0 * 1.4826 * mad, 0.03 * base_depth_scale)
    clipped = np.clip(residual, center - limit, center + limit)
    return ((clipped - float(np.median(clipped))) / base_depth_scale).astype(np.float32)


def _edge_texture_correct(rgb: np.ndarray, support: np.ndarray, rim_pixels: int = 32) -> np.ndarray:
    """Reduce low-frequency optical falloff at a supported image boundary.

    The data2 camera has a dark optical/vignetting rim near the moving image
    boundary.  Treating that rim as a new RGB sample makes a valid surface look
    like two overlapped sheets after the canvas update.  This correction keeps
    the source texture and only adjusts a low-frequency luminance gain in the
    outer rim, using the same frame's interior as its reference.  It does not
    alter the support mask or the model input.

    Complexity: O(HW) with two separable Gaussian blurs.
    """

    arr = _rgb_float(rgb).astype(np.float32, copy=False)
    mask = np.asarray(support, dtype=bool)
    if cv2 is None or not mask.any() or float(mask.mean()) > 0.94:
        return arr.copy()
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    rim_pixels = max(int(rim_pixels), 8)
    interior_min = max(8, rim_pixels // 3)
    interior = mask & (distance >= float(interior_min))
    if int(interior.sum()) < 512:
        return arr.copy()

    gray = np.mean(arr, axis=2).astype(np.float32)
    sigma = max(6.0, min(18.0, float(rim_pixels) * 0.5))
    support_f = mask.astype(np.float32)
    interior_f = interior.astype(np.float32)
    local_den = cv2.GaussianBlur(support_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_num = cv2.GaussianBlur(gray * support_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    interior_den = cv2.GaussianBlur(interior_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    interior_num = cv2.GaussianBlur(gray * interior_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_mean = np.divide(local_num, np.maximum(local_den, 1e-5))
    interior_mean = np.divide(interior_num, np.maximum(interior_den, 1e-5))
    gain = np.divide(interior_mean, np.maximum(local_mean, 1e-3))
    gain = np.clip(gain, 0.80, 1.45).astype(np.float32)
    alpha = np.clip((float(rim_pixels) - distance) / float(rim_pixels), 0.0, 1.0)
    alpha *= (interior_den > 0.02).astype(np.float32)
    # Only lift a dark edge; a bright rim is not a reason to darken the real
    # texture and should remain untouched.
    alpha *= (gain > 1.01).astype(np.float32)
    corrected = arr * (1.0 + alpha[..., None] * (gain[..., None] - 1.0))
    corrected[~mask] = arr[~mask]
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate the valid non-border support region from an RGB image.

    Complexity:
        O(HW) with optional OpenCV morphology and connected components.
    """

    arr = _rgb_float(rgb)
    gray = np.mean(arr, axis=2)
    if cv2 is not None:
        otsu_thr, _ = cv2.threshold((np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(0.10, float(otsu_thr) / 255.0)
    else:
        threshold = max(0.10, float(np.percentile(gray, 35)))
    mask = gray > threshold
    if float(mask.mean()) > 0.94:
        return np.ones(gray.shape, dtype=bool)
    if cv2 is None:
        return mask.astype(bool)
    mask_u8 = mask.astype(np.uint8)
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((9, 9), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(1 + np.argmax(areas))
    if int(stats[largest, cv2.CC_STAT_AREA]) < int(0.08 * mask_u8.size):
        return mask_u8.astype(bool)
    return labels == largest


def _filter_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove small connected components from a boolean mask."""

    if cv2 is None or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, count):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_pixels:
            keep |= labels == idx
    return keep


def _seam_fusion_mask(change_mask: np.ndarray, existing_valid: np.ndarray | None, valid_warp: np.ndarray, radius: int) -> np.ndarray:
    """Expand a change mask with an overlap band to avoid seams.

    Complexity:
        O(HW). The band is deliberately small so it keeps incremental behavior
        while giving depth alignment and weighted fusion shared pixels across
        old/new boundaries.
    """

    if not change_mask.any():
        return change_mask.copy()
    radius = max(int(radius), 1)
    if cv2 is not None:
        close_kernel = np.ones((radius + 1, radius + 1), dtype=np.uint8)
        band_kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        closed = cv2.morphologyEx(change_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)
        expanded = cv2.dilate(closed.astype(np.uint8), band_kernel, iterations=1).astype(bool)
    else:
        closed = _dilate(change_mask, radius + 1)
        expanded = _dilate(closed, radius * 2 + 1)
    fusion = expanded & valid_warp
    if existing_valid is not None:
        old_band = fusion & existing_valid
        new_region = closed & valid_warp
        fusion = new_region | old_band | (expanded & ~existing_valid & valid_warp)
    return fusion


def _anchor_ring_mask(
    change_mask: np.ndarray,
    existing_valid: np.ndarray | None,
    valid_warp: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Return a stable unchanged annulus around the changed region.

    The ring is kept outside the changed core and is intersected with both the
    old canvas and the current warped support. It is therefore a geometric
    anchor for local depth alignment, not an additional surface observation.
    """

    if existing_valid is None or not change_mask.any():
        return np.zeros_like(change_mask, dtype=bool)
    inner_radius = max(int(inner_radius), 1)
    outer_radius = max(int(outer_radius), inner_radius + 1)
    if cv2 is not None:
        inner_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (inner_radius * 2 + 1, inner_radius * 2 + 1),
        )
        outer_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (outer_radius * 2 + 1, outer_radius * 2 + 1),
        )
        core = change_mask.astype(np.uint8)
        inner = cv2.dilate(core, inner_kernel, iterations=1).astype(bool)
        outer = cv2.dilate(core, outer_kernel, iterations=1).astype(bool)
    else:
        inner = _dilate(change_mask, inner_radius * 2 + 1)
        outer = _dilate(change_mask, outer_radius * 2 + 1)
    return outer & ~inner & existing_valid & valid_warp


def _propagate_seam_residual(
    residual: np.ndarray,
    seam_mask: np.ndarray,
    target_mask: np.ndarray,
    sigma: float,
    max_abs: float,
) -> np.ndarray:
    """Propagate old/new height residuals from the overlap band into a target ROI.

    Complexity:
        O(HW). This turns the overlap band into a geometric constraint rather
        than merely another weighted observation.
    """

    correction = np.zeros_like(residual, dtype=np.float32)
    if not seam_mask.any() or not target_mask.any():
        return correction
    values = np.clip(residual.astype(np.float32), -float(max_abs), float(max_abs))
    if cv2 is not None:
        values = values.copy()
        seam_values = values[seam_mask]
        center = float(np.median(seam_values))
        mad = max(float(np.median(np.abs(seam_values - center))), 1e-6)
        keep = seam_mask & (np.abs(values - center) <= 3.0 * 1.4826 * mad)
        if int(keep.sum()) >= 32:
            seam_mask = keep
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is declared in pyproject.
        logger.warning("Skipping seam residual propagation because scipy is unavailable: %s", exc)
        correction[seam_mask] = values[seam_mask]
        return correction

    _, indices = ndimage.distance_transform_edt(~seam_mask, return_indices=True)
    nearest = values[indices[0], indices[1]].astype(np.float32)
    nearest[~target_mask] = 0.0
    smooth = ndimage.gaussian_filter(nearest, sigma=float(sigma)).astype(np.float32)
    weight = ndimage.gaussian_filter(target_mask.astype(np.float32), sigma=float(sigma)).astype(np.float32)
    smooth = np.divide(smooth, np.maximum(weight, 1e-6), out=np.zeros_like(smooth), where=weight > 1e-6)
    smooth[seam_mask] = values[seam_mask]
    smooth[~target_mask] = 0.0
    return np.clip(smooth, -float(max_abs), float(max_abs)).astype(np.float32)


def _fit_spatial_seam_residual(
    residual: np.ndarray,
    seam_mask: np.ndarray,
    target_mask: np.ndarray,
    max_abs: float,
) -> np.ndarray:
    """Fit a robust local offset/tilt from the shared anchor ring.

    The scalar depth calibration fixes only scale and global bias.  A pair of
    aligned ROIs can still have a small local tilt, so the first newly exposed
    pixels inherit a depth step even when the RGB homography is correct.  Fit
    the residual ``old_canvas - aligned_current`` in normalized canvas
    coordinates and evaluate that field over the complete target ROI.  A
    robust affine field is used instead of an unconstrained polynomial: it
    transfers the shared-region alignment while avoiding a raised quadratic
    strip when the ring is narrow.
    """

    correction = np.zeros_like(residual, dtype=np.float32)
    if not seam_mask.any() or not target_mask.any():
        return correction
    ys, xs = np.nonzero(seam_mask)
    values = np.asarray(residual, dtype=np.float64)[seam_mask]
    finite = np.isfinite(values)
    if int(finite.sum()) < 64:
        return correction
    ys = ys[finite].astype(np.float64)
    xs = xs[finite].astype(np.float64)
    values = values[finite]
    height, width = residual.shape
    cx = float(np.median(xs))
    cy = float(np.median(ys))
    coord_scale = max(float(max(height, width)) * 0.25, 32.0)
    xn = (xs - cx) / coord_scale
    yn = (ys - cy) / coord_scale
    design = np.stack([np.ones_like(xn), xn, yn], axis=1)
    keep = np.ones(values.shape, dtype=bool)
    coeff = np.array([float(np.median(values)), 0.0, 0.0], dtype=np.float64)
    for _ in range(5):
        if int(keep.sum()) < 32:
            break
        try:
            coeff = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        except np.linalg.LinAlgError:
            return correction
        fit_residual = values - design @ coeff
        center = float(np.median(fit_residual[keep]))
        mad = max(float(np.median(np.abs(fit_residual[keep] - center))), 1e-6)
        keep = np.abs(fit_residual - center) <= 3.0 * 1.4826 * mad

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    target_design = np.stack(
        [
            np.ones_like(xx),
            (xx - cx) / coord_scale,
            (yy - cy) / coord_scale,
        ],
        axis=-1,
    )
    field = np.tensordot(target_design, coeff, axes=([-1], [0]))
    field = np.clip(field, -float(max_abs), float(max_abs)).astype(np.float32)
    correction[target_mask] = field[target_mask]
    return correction


def _legacy_anchor_texture_fusion(
    current_rgb: np.ndarray,
    canvas_rgb: np.ndarray,
    canvas_valid: np.ndarray,
    current_valid: np.ndarray,
    apply_mask: np.ndarray,
    support_change: np.ndarray,
    anchor_ring: np.ndarray,
    sigma: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the stable canvas color field through a newly exposed seam.

    The unchanged ring is a location constraint, but copying its nearest RGB
    value creates radial streaks.  Instead, this function estimates a smooth
    low-frequency field from the interior of the old canvas and the current
    aligned image, then transfers only their low-frequency difference into the
    committed pixels.  The high-frequency sand texture remains from the new
    frame, while the boundary follows the old surface continuously.

    Complexity: O(HW) plus separable Gaussian filtering.
    """

    result = _rgb_float(current_rgb).astype(np.float32, copy=True)
    mix = np.zeros(apply_mask.shape, dtype=np.float32)
    if cv2 is None or not apply_mask.any() or not canvas_valid.any() or not anchor_ring.any():
        return result, mix

    old_valid = np.asarray(canvas_valid, dtype=bool)
    new_valid = np.asarray(current_valid, dtype=bool)
    # Do not use the old canvas's own outermost pixels as the color source;
    # those pixels can carry exactly the rim that this operation is removing.
    interior = _erode_mask(old_valid, 10)
    if int(interior.sum()) < 512:
        interior = old_valid
    old_f = interior.astype(np.float32)
    new_f = new_valid.astype(np.float32)
    old_arr = _rgb_float(canvas_rgb).astype(np.float32, copy=False)
    new_arr = result
    old_den = cv2.GaussianBlur(old_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    new_den = cv2.GaussianBlur(new_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    old_field = np.empty_like(old_arr)
    new_field = np.empty_like(new_arr)
    for channel in range(3):
        old_num = cv2.GaussianBlur(old_arr[..., channel] * old_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        new_num = cv2.GaussianBlur(new_arr[..., channel] * new_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        old_field[..., channel] = np.divide(old_num, np.maximum(old_den, 1e-5))
        new_field[..., channel] = np.divide(new_num, np.maximum(new_den, 1e-5))

    # When a pixel is genuinely outside the old canvas there is no old sample
    # at the same XY coordinate.  Continue the *local* low-frequency field
    # estimated from the old interior rather than using one global median.
    # A global color makes a newly exposed top patch look like a rectangular
    # pasted tile even when the source texture itself is correct.
    anchor_color = np.median(old_arr[interior], axis=0).astype(np.float32)
    new_only = apply_mask & support_change & ~old_valid

    # The anchor ring controls where the correction is allowed to start.  A
    # broad Gaussian field avoids a visible hard boundary when the new area is
    # wider than the ring itself.
    ring_f = anchor_ring.astype(np.float32)
    ring_support = cv2.GaussianBlur(ring_f, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))

    # The old canvas has no sample at the same XY coordinate for a newly
    # exposed border.  ``old_field`` is therefore undefined in the outer part
    # of that region.  Fit a smooth low-frequency field on the eroded old
    # interior and extrapolate it through the new area.  The ring still
    # determines where this field is allowed to affect the new patch, but its
    # own pixels are not copied as color: they can contain the optical dark
    # rim that caused the original black strip.
    height, width = old_valid.shape
    source = interior & (old_den > 1e-4)
    extended_field = np.empty_like(old_arr)
    extended_field[...] = anchor_color
    if int(source.sum()) >= 256:
        sy, sx = np.nonzero(source)
        sample_count = min(int(sy.size), 24000)
        if sy.size > sample_count:
            selected = np.linspace(0, sy.size - 1, sample_count, dtype=np.int64)
            sy, sx = sy[selected], sx[selected]
        coord_scale = max(float(max(height, width)) * 0.5, 32.0)
        cx = float(np.median(sx))
        cy = float(np.median(sy))
        sample_x = (sx.astype(np.float64) - cx) / coord_scale
        sample_y = (sy.astype(np.float64) - cy) / coord_scale
        sample_design = np.stack(
            [
                np.ones_like(sample_x),
                sample_x,
                sample_y,
                sample_x * sample_x,
                sample_x * sample_y,
                sample_y * sample_y,
            ],
            axis=1,
        )
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        full_x = (xx - cx) / coord_scale
        full_y = (yy - cy) / coord_scale
        full_design = np.stack(
            [
                np.ones_like(full_x),
                full_x,
                full_y,
                full_x * full_x,
                full_x * full_y,
                full_y * full_y,
            ],
            axis=-1,
        )
        for channel in range(3):
            try:
                coeff = np.linalg.lstsq(
                    sample_design,
                    old_field[sy, sx, channel].astype(np.float64),
                    rcond=None,
                )[0]
                extended_field[..., channel] = np.tensordot(
                    full_design,
                    coeff,
                    axes=([-1], [0]),
                )
            except np.linalg.LinAlgError:
                extended_field[..., channel] = anchor_color[channel]
        extended_field = np.clip(extended_field, 0.0, 1.0).astype(np.float32)

    # Keep the spatial field for texture continuity, but add a modest global
    # interior reference so a far border extrapolation cannot drift toward the
    # camera's dark vignetting.  The current frame still supplies the retained
    # high-frequency detail below.
    local_anchor = (0.25 * extended_field + 0.75 * anchor_color[None, None, :]).astype(np.float32)
    # Gaussian support can be non-zero outside the old canvas.  Those pixels
    # are still extrapolated pixels, not actual old observations; only use the
    # old field directly where the canvas really was valid.
    old_field_cells = old_valid & (old_den > 1e-4)
    local_anchor[old_field_cells] = old_field[old_field_cells]
    # Keep only a fraction of the current frame's high-frequency residual so
    # the unchanged ring controls the connection without making a flat patch.
    new_only_target = local_anchor + 0.10 * (new_arr - new_field)
    base_mix = np.where(support_change, 0.72, 0.48).astype(np.float32)
    mix = base_mix * np.clip(ring_support / 0.16, 0.0, 1.0)
    mix *= np.clip(old_den / 0.015, 0.0, 1.0)
    # A newly exposed support band may extend beyond the Gaussian footprint of
    # the ring.  It still represents the same static surface, so keep the
    # anchor continuation dominant there; otherwise the raw dark optical rim
    # survives as a black strip.  The current frame contributes through the
    # 25% high-frequency residual in ``new_only_target`` above.
    new_only_mix = 0.94 * np.clip(ring_support / 0.02, 0.80, 1.0)
    mix = np.where(new_only, new_only_mix, mix)
    mix *= apply_mask.astype(np.float32)
    usable = apply_mask & (new_den > 1e-4)
    target_field = local_anchor + 0.65 * (new_arr - new_field)
    target_field[new_only] = new_only_target[new_only]
    correction = np.clip(target_field - new_arr, -0.30, 0.30)
    corrected = new_arr + mix[..., None] * correction
    corrected[~usable] = new_arr[~usable]
    result[apply_mask] = np.clip(corrected[apply_mask], 0.0, 1.0)
    mix[~apply_mask] = 0.0
    return result.astype(np.float32), mix.astype(np.float32)


def _anchor_depth_continuity(
    model_depth: np.ndarray,
    canvas_depth: np.ndarray,
    canvas_valid: np.ndarray,
    apply_mask: np.ndarray,
    anchor_ring: np.ndarray,
    transition_px: int = 96,
    model_confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a new depth patch into a smooth surface continuation.

    A support change is normally a newly exposed part of the same static
    surface, not a newly created object.  The unchanged ring therefore has
    priority throughout that added band.  The model is retained as a bounded
    residual so real local relief is not completely erased, but low-confidence
    ROI-edge samples cannot create a second sheet.  This is important when a
    frame border exposes a new image area: the model can be locally correct
    yet still have a different depth origin, which otherwise renders as a
    raised/duplicated strip.
    """

    result = np.asarray(model_depth, dtype=np.float32).copy()
    delta = np.zeros_like(result, dtype=np.float32)
    if cv2 is None or not apply_mask.any() or not anchor_ring.any():
        return result, delta

    old_valid = np.asarray(canvas_valid, dtype=bool)
    anchor = anchor_ring & old_valid & np.isfinite(canvas_depth)
    if int(anchor.sum()) < 64:
        return result, delta

    height, width = old_valid.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    ys, xs = np.nonzero(anchor)
    cx = float(np.median(xs))
    cy = float(np.median(ys))
    coord_scale = max(float(max(height, width)) * 0.25, 32.0)
    xn = (xx - cx) / coord_scale
    yn = (yy - cy) / coord_scale
    design = np.stack(
        [np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn],
        axis=-1,
    )
    anchor_design = design[anchor]
    anchor_values = np.asarray(canvas_depth, dtype=np.float64)[anchor]
    keep = np.ones(anchor_values.shape, dtype=bool)
    coeff = np.zeros(6, dtype=np.float64)
    for _ in range(4):
        if int(keep.sum()) < 32:
            break
        try:
            coeff = np.linalg.lstsq(anchor_design[keep], anchor_values[keep], rcond=None)[0]
        except np.linalg.LinAlgError:
            return result, delta
        residual = anchor_values - anchor_design @ coeff
        center = float(np.median(residual[keep]))
        mad = max(float(np.median(np.abs(residual[keep] - center))), 1e-6)
        keep = np.abs(residual - center) <= 3.0 * 1.4826 * mad

    # The quadratic fitted over the complete ring is useful as a fallback,
    # but it is not a safe continuation field for a partial border component.
    # For example, the left exposed band can be tens of pixels away from the
    # top/right parts of the same ring; a single global polynomial then bends
    # the new band onto a different depth layer.  Replace it with a local
    # normalized-convolution field derived from the old canvas.  The anchor
    # ring is still the gate for this operation and remains excluded from the
    # write mask; its unchanged surface is what supplies the local field.
    continuation = _local_anchor_depth_field(
        canvas_depth=np.asarray(canvas_depth, dtype=np.float32),
        canvas_valid=old_valid,
        fallback=np.tensordot(design, coeff, axes=([-1], [0])).astype(np.float32),
        target_mask=apply_mask,
        sigma=6.0,
    )
    finite_continuation = np.isfinite(continuation)
    if not finite_continuation.any():
        return result, delta

    # New-only support pixels are the same surface seen through a newly exposed
    # image area.  Use the old-surface continuation as their geometry directly;
    # retaining any model residual here would recreate the depth-origin error
    # that this anchor operation is meant to remove.  ``model_confidence`` is
    # kept in the signature for callers that already pass it and for future
    # non-support change modes.
    new_only = apply_mask & ~old_valid & finite_continuation & np.isfinite(result)
    if new_only.any():
        before = result.copy()
        result[new_only] = continuation[new_only]
        delta[new_only] = result[new_only] - before[new_only]

    # Changed pixels that already existed in the canvas are the overlap half
    # of a support seam.  They describe the same already reconstructed
    # surface, so keep their geometry exactly.  Replacing these cells with a
    # second model depth is what creates the parallel layer immediately beside
    # the newly exposed band.  The RGB path still decides independently which
    # texture observations are committed.
    old_overlap = apply_mask & old_valid & np.isfinite(canvas_depth) & np.isfinite(result)
    if old_overlap.any():
        before = result.copy()
        result[old_overlap] = canvas_depth[old_overlap]
        delta[old_overlap] = result[old_overlap] - before[old_overlap]

    return result.astype(np.float32), delta.astype(np.float32)


def _anchor_texture_transfer(
    current_rgb: np.ndarray,
    canvas_rgb: np.ndarray,
    canvas_valid: np.ndarray,
    current_valid: np.ndarray,
    apply_mask: np.ndarray,
    support_change: np.ndarray,
    anchor_ring: np.ndarray,
    sigma: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only an evidence-based photometric transfer at a seam.

    The current aligned image is the authoritative color observation.  The
    unchanged ring is used to estimate a smooth low-frequency residual
    ``old_canvas - current_frame`` and that residual is transferred into the
    newly exposed area.  It deliberately does *not* replace the new image with
    an extrapolated old RGB field: the latter can turn a real dark/bright edge
    in the input into a synthetic color layer.

    Complexity: O(HW) plus separable Gaussian filtering.
    """

    result = _rgb_float(current_rgb).astype(np.float32, copy=True)
    mix = np.zeros(apply_mask.shape, dtype=np.float32)
    if cv2 is None or not apply_mask.any() or not canvas_valid.any() or not anchor_ring.any():
        return result, mix

    old_valid = np.asarray(canvas_valid, dtype=bool)
    new_valid = np.asarray(current_valid, dtype=bool)
    old_arr = _rgb_float(canvas_rgb).astype(np.float32, copy=False)
    new_arr = result
    sigma = max(float(sigma), 4.0)
    old_f = old_valid.astype(np.float32)
    new_f = new_valid.astype(np.float32)
    old_den = cv2.GaussianBlur(old_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    new_den = cv2.GaussianBlur(new_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    old_field = np.empty_like(old_arr)
    new_field = np.empty_like(new_arr)
    for channel in range(3):
        old_num = cv2.GaussianBlur(old_arr[..., channel] * old_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        new_num = cv2.GaussianBlur(new_arr[..., channel] * new_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        old_field[..., channel] = np.divide(old_num, np.maximum(old_den, 1e-5))
        new_field[..., channel] = np.divide(new_num, np.maximum(new_den, 1e-5))

    stable = anchor_ring & old_valid & new_valid & np.isfinite(old_field).all(axis=2) & np.isfinite(new_field).all(axis=2)
    if int(stable.sum()) < 128:
        # A narrow ring can be clipped at a frame border.  Use valid overlap
        # only as a fallback; changed/support pixels remain excluded.
        stable = old_valid & new_valid & ~np.asarray(support_change, dtype=bool)
    if int(stable.sum()) < 128:
        return result, mix

    residual = old_field - new_field
    # Robustly remove local texture outliers before extending the transfer.
    residual_source = np.zeros_like(residual, dtype=np.float32)
    for channel in range(3):
        values = residual[..., channel][stable].astype(np.float64)
        center = float(np.median(values))
        mad = max(float(np.median(np.abs(values - center))), 1e-4)
        limit = max(0.04, 3.0 * 1.4826 * mad)
        clipped = np.clip(
            residual[..., channel],
            center - limit,
            center + limit,
        ).astype(np.float32)
        residual_source[..., channel] = np.where(stable, clipped, 0.0).astype(np.float32)

    ring_f = stable.astype(np.float32)
    ring_den = cv2.GaussianBlur(ring_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    correction = np.empty_like(residual_source)
    for channel in range(3):
        ring_num = cv2.GaussianBlur(
            residual_source[..., channel] * ring_f,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        correction[..., channel] = np.divide(
            ring_num,
            np.maximum(ring_den, 1e-5),
            out=np.zeros_like(ring_num, dtype=np.float32),
            where=ring_den > 1e-5,
        )
    # A ring can contain slight registration error and real illumination
    # variation.  Keep the measured transfer conservative; the source frame
    # must remain the dominant color evidence in newly exposed pixels.
    correction = np.clip(correction, -0.05, 0.05).astype(np.float32)
    transferred = np.clip(new_arr + correction, 0.0, 1.0).astype(np.float32)

    # The anchor ring is intentionally not written.  The current frame remains
    # authoritative in newly exposed support; the old canvas is not allowed to
    # bleed into that region because it can contain a stale optical rim.
    # A separate color-only bridge in ``push`` transitions the old overlap to
    # the new observation while leaving this ring untouched.
    support_target = np.asarray(support_change, dtype=bool)
    support_overlap = apply_mask & support_target & old_valid & new_valid
    if support_overlap.any():
        distance_to_ring = cv2.distanceTransform(
            (~anchor_ring).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        old_mix = np.clip(1.0 - distance_to_ring / 16.0, 0.65, 0.95).astype(np.float32)
        transferred[support_overlap] = (
            transferred[support_overlap] * (1.0 - old_mix[support_overlap, None])
            + old_arr[support_overlap] * old_mix[support_overlap, None]
        )
        mix[support_overlap] = np.maximum(mix[support_overlap], old_mix[support_overlap])

    usable = apply_mask & new_valid & np.isfinite(transferred).all(axis=2)
    result[usable] = transferred[usable]
    correction_mix = np.clip(np.mean(np.abs(correction[usable]), axis=1) / 0.05, 0.0, 1.0)
    mix[usable] = np.maximum(mix[usable], correction_mix)
    return result.astype(np.float32), mix.astype(np.float32)


def _local_anchor_depth_field(
    canvas_depth: np.ndarray,
    canvas_valid: np.ndarray,
    fallback: np.ndarray,
    target_mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Continue the old depth surface through a newly exposed support band.

    A normalized Gaussian over valid old pixels gives each new pixel a local
    surface estimate instead of extrapolating one polynomial across unrelated
    sides of the object.  Nearest-valid padding is used only where the
    Gaussian has insufficient support.  This keeps the value at the old/new
    boundary continuous while retaining the existing global fit as a safe
    fallback for unusual masks or missing SciPy.
    """

    result = np.asarray(fallback, dtype=np.float32).copy()
    old = np.asarray(canvas_valid, dtype=bool) & np.isfinite(canvas_depth)
    target = np.asarray(target_mask, dtype=bool)
    if not old.any() or not target.any() or sigma <= 0.0:
        return result
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is declared in pyproject.
        logger.warning("Skipping local anchor depth field because scipy is unavailable: %s", exc)
        return result

    valid_f = old.astype(np.float32)
    source = np.where(old, np.asarray(canvas_depth, dtype=np.float32), 0.0)
    numerator = ndimage.gaussian_filter(source, sigma=float(sigma))
    denominator = ndimage.gaussian_filter(valid_f, sigma=float(sigma))
    smooth = np.divide(
        numerator,
        np.maximum(denominator, 1e-6),
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 1e-6,
    )

    # The nearest old sample prevents a low-support Gaussian tail from
    # drifting toward zero at a wide frame border.  It is only a fallback:
    # using the nearest index as the main field can itself jump when the
    # boundary turns a corner and the nearest old pixel changes rows.  Instead
    # use a locally blurred Dirichlet correction from the actual old/new
    # boundary, so the first new pixels meet the old surface without a nearest
    # index seam.
    distance, indices = ndimage.distance_transform_edt(~old, return_indices=True)
    nearest = np.asarray(canvas_depth, dtype=np.float32)[indices[0], indices[1]]
    new_target = target & ~old
    distance_to_target = ndimage.distance_transform_edt(~new_target)
    boundary = old & (distance_to_target <= 2.5)
    boundary_f = boundary.astype(np.float32)
    boundary_residual = np.where(boundary, np.asarray(canvas_depth, dtype=np.float32) - smooth, 0.0)
    boundary_sigma = max(2.0, min(4.0, float(sigma) * 0.5))
    boundary_num = ndimage.gaussian_filter(boundary_residual, sigma=boundary_sigma)
    boundary_den = ndimage.gaussian_filter(boundary_f, sigma=boundary_sigma)
    boundary_correction = np.divide(
        boundary_num,
        np.maximum(boundary_den, 1e-6),
        out=np.zeros_like(boundary_num, dtype=np.float32),
        where=boundary_den > 1e-6,
    )
    local = smooth + boundary_correction
    local_support = (denominator > 0.02) | (boundary_den > 1e-4)
    local = np.where(local_support, local, nearest).astype(np.float32)
    usable = target & np.isfinite(local)
    result[usable] = local[usable]
    return result.astype(np.float32)


def _match_rgb_to_canvas(
    warped_rgb: np.ndarray,
    canvas_rgb: np.ndarray,
    overlap_mask: np.ndarray,
    apply_mask: np.ndarray,
) -> np.ndarray:
    """Photometrically match current RGB to the existing canvas over overlap.

    Complexity:
        O(N) over overlap pixels. This prevents darker oblique/edge frames from
        tinting newly added regions.
    """

    if int(overlap_mask.sum()) < 512 or not apply_mask.any():
        return warped_rgb
    source = np.clip(warped_rgb[overlap_mask].astype(np.float32), 0.0, 1.0)
    target = np.clip(canvas_rgb[overlap_mask].astype(np.float32), 0.0, 1.0)
    src_med = np.median(source, axis=0)
    tgt_med = np.median(target, axis=0)
    src_iqr = np.maximum(np.percentile(source, 75, axis=0) - np.percentile(source, 25, axis=0), 1e-4)
    tgt_iqr = np.maximum(np.percentile(target, 75, axis=0) - np.percentile(target, 25, axis=0), 1e-4)
    gain = np.clip(tgt_iqr / src_iqr, 0.75, 1.25)
    bias = np.clip(tgt_med - gain * src_med, -0.18, 0.18)
    corrected = warped_rgb.copy()
    corrected_values = np.clip(corrected[apply_mask].astype(np.float32) * gain[None, :] + bias[None, :], 0.0, 1.0)
    corrected[apply_mask] = corrected_values
    return corrected.astype(np.float32)


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Trim a boundary band from a boolean support mask."""

    if pixels <= 0 or not mask.any():
        return mask
    if cv2 is not None:
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        return eroded if eroded.any() else mask
    padded = np.pad(mask, pixels, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    size = pixels * 2 + 1
    for dy in range(size):
        for dx in range(size):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out if out.any() else mask


def _fill_narrow_gaps(depth: np.ndarray, rgb: np.ndarray, valid: np.ndarray, pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill narrow internal gaps by copying nearest valid samples.

    Complexity:
        O(HW). This is export-only and keeps the streaming map state unchanged.
    """

    if pixels <= 0 or not valid.any() or cv2 is None:
        return depth, rgb, valid
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    fill = closed & ~valid
    if not fill.any():
        return depth, rgb, valid
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is declared in pyproject.
        logger.warning("Skipping narrow-gap fill because scipy is unavailable: %s", exc)
        return depth, rgb, valid
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    nearest_y = indices[0]
    nearest_x = indices[1]
    filled_depth = depth.copy()
    filled_rgb = rgb.copy()
    filled_valid = valid.copy()
    filled_depth[fill] = depth[nearest_y[fill], nearest_x[fill]]
    filled_rgb[fill] = rgb[nearest_y[fill], nearest_x[fill]]
    filled_valid[fill] = True
    return filled_depth, filled_rgb, filled_valid


def _regularize_heightfield(depth: np.ndarray, valid: np.ndarray, sigma: float, raw_keep: float) -> np.ndarray:
    """Smooth a valid depth field with normalized convolution before export.

    Complexity:
        O(HW). This is export-only and converts block-wise streaming updates
        into a continuous height field while retaining a small fraction of raw
        local relief.
    """

    if not valid.any() or sigma <= 0:
        return depth
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover - scipy is declared in pyproject.
        logger.warning("Skipping heightfield regularization because scipy is unavailable: %s", exc)
        return depth
    valid_f = valid.astype(np.float32)
    depth_work = depth.astype(np.float32, copy=True)

    # Remove thin, high-amplitude depth ridges before Gaussian regularization.
    # A normalized Gaussian alone preserves a one- or two-pixel ROI seam as a
    # narrow parallel sheet.  Nearest-valid padding is used only to compute a
    # local median; invalid canvas cells are never changed here.
    if int(valid.sum()) >= 64:
        _, nearest = ndimage.distance_transform_edt(~valid, return_distances=True, return_indices=True)
        median_input = depth_work.copy()
        missing = ~valid
        if missing.any():
            median_input[missing] = depth_work[nearest[0][missing], nearest[1][missing]]
        local_median = ndimage.median_filter(median_input, size=5, mode="nearest").astype(np.float32)
        local_residual = depth_work - local_median
        residual_values = local_residual[valid & np.isfinite(local_residual)]
        if residual_values.size >= 64:
            residual_center = float(np.median(residual_values))
            residual_mad = max(float(np.median(np.abs(residual_values - residual_center))), 1e-6)
            ridge_limit = max(0.025, 6.0 * 1.4826 * residual_mad)
            dense_neighborhood = ndimage.uniform_filter(valid_f, size=5, mode="nearest") > 0.72
            ridges = valid & dense_neighborhood & np.isfinite(local_residual)
            ridges &= np.abs(local_residual - residual_center) > ridge_limit
            if ridges.any():
                depth_work[ridges] = local_median[ridges]

    weighted = ndimage.gaussian_filter((depth_work * valid_f).astype(np.float32), sigma=float(sigma))
    weight = ndimage.gaussian_filter(valid_f, sigma=float(sigma))
    smooth = np.divide(weighted, np.maximum(weight, 1e-6), out=depth_work.copy(), where=weight > 1e-6)
    keep = float(np.clip(raw_keep, 0.0, 1.0))
    out = smooth * (1.0 - keep) + depth_work * keep
    out[~valid] = depth[~valid]
    return out.astype(np.float32)


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save a float RGB image as PNG."""

    from PIL import Image

    Image.fromarray(np.clip(_rgb_float(rgb) * 255.0, 0, 255).astype(np.uint8)).save(path)


def _save_gray(path: Path, gray: np.ndarray) -> None:
    """Save a float scalar image as contrast-normalized PNG."""

    from PIL import Image

    arr = gray.astype(np.float32)
    if np.isfinite(arr).any():
        lo, hi = np.percentile(arr[np.isfinite(arr)], [1, 99])
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((arr - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    """Save a boolean mask as PNG."""

    from PIL import Image

    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def _save_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    """Save an RGB image with a red translucent change-mask overlay."""

    from PIL import Image

    base = np.clip(_rgb_float(rgb), 0.0, 1.0)
    overlay = base.copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([1.0, 0.0, 0.0], dtype=np.float32) * 0.55
    if cv2 is not None:
        edges = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool) & ~mask
        overlay[edges] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    Image.fromarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8)).save(path)


def _dilate(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Dilate a boolean mask."""

    if cv2 is not None:
        kernel = np.ones((max(ksize, 1), max(ksize, 1)), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    radius = max(ksize // 2, 0)
    padded = np.pad(mask, radius)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(ksize):
        for dx in range(ksize):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _gray_u8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float array to uint8 grayscale."""

    return np.clip(_rgb_float(rgb).mean(axis=2) * 255.0, 0, 255).astype(np.uint8)


def _gray_float(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float array to float32 grayscale."""

    arr = _rgb_float(rgb)
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def _rgb_float(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB to float32 [0, 1] without double-scaling float images."""

    source = np.asarray(rgb)
    arr = source.astype(np.float32, copy=False)
    if np.issubdtype(source.dtype, np.integer) or arr.max(initial=0.0) > 2.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _rgb_np(value: Any) -> np.ndarray:
    """Convert packet RGB to HxWx3 NumPy."""

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        return arr
    return np.asarray(value)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to NumPy arrays."""

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _ms(start: float) -> float:
    """Elapsed milliseconds from a perf_counter timestamp."""

    return (perf_counter() - start) * 1000.0

