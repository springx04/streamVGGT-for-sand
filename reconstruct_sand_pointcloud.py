"""
Specialized no-ghost point-cloud reconstruction for sand images captured through
a rotating observation aperture.
"""

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.image import ImgNorm


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "OmniVGGT.safetensors"


@dataclass
class FrameData:
    index: int
    path: Path
    image: np.ndarray
    mask: np.ndarray
    gray: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]


@dataclass
class EdgeData:
    src: int
    dst: int
    homography: np.ndarray
    inliers: int
    good_matches: int
    median_error: float
    overlap: float
    score: float
    accepted: bool


@dataclass
class FrameSamples:
    local_index: int
    original_index: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    weight: np.ndarray
    rgb: np.ndarray
    key: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-ghost sand point-cloud reconstruction")
    parser.add_argument("--image_folder", type=str, default="data2")
    parser.add_argument("--output_dir", type=str, default="outputs/data2_pointcloud")
    parser.add_argument("--num_keyframes", type=int, default=3)
    parser.add_argument("--mode", type=str, default="balanced", choices=["fast", "balanced", "accurate"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--matching_width", type=int, default=960)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--anchor_index", type=int, default=-1)
    parser.add_argument("--keyframe_indices", type=str, default="")
    parser.add_argument("--min_inliers", type=int, default=150)
    parser.add_argument("--max_reproj_error", type=float, default=4.0)
    parser.add_argument("--min_overlap", type=float, default=0.30)
    parser.add_argument("--conf_percentile", type=float, default=20.0)
    parser.add_argument("--depth_mad_multiplier", type=float, default=3.0)
    parser.add_argument("--depth_align", type=str, default="affine", choices=["none", "median", "affine"])
    parser.add_argument("--min_depth_overlap_cells", type=int, default=500)
    parser.add_argument("--max_aligned_depth_mad", type=float, default=0.02)
    parser.add_argument("--allow_depth_fallback_fill", action="store_true")
    parser.add_argument("--fusion_mode", type=str, default="soft_blend", choices=["median", "reference_fill", "soft_blend"])
    parser.add_argument("--reference_dilation", type=int, default=9)
    parser.add_argument("--min_fill_cells", type=int, default=5000)
    parser.add_argument("--local_depth_sigma", type=float, default=35.0)
    parser.add_argument("--depth_consistency", type=float, default=0.03)
    parser.add_argument("--feather_power", type=float, default=1.0)
    parser.add_argument("--min_feather", type=float, default=0.08)
    parser.add_argument("--update_reference_depth", action="store_true")
    parser.add_argument("--surface_model", type=str, default="plane_residual", choices=["depth", "plane_residual"])
    parser.add_argument("--height_exaggeration", type=float, default=1.0)
    parser.add_argument("--plane_residual_mad_multiplier", type=float, default=3.0)
    parser.add_argument("--output_scale", type=int, default=2)
    parser.add_argument(
        "--texture_source",
        type=str,
        default="anchor_original",
        choices=["anchor_original", "blended_original", "blended_model"],
    )
    parser.add_argument(
        "--depth_detail_source",
        type=str,
        default="none",
        choices=["none", "anchor_luma", "fused_luma", "parallax_flow"],
    )
    parser.add_argument("--depth_detail_strength", type=float, default=0.0)
    parser.add_argument("--depth_detail_sigma", type=float, default=5.0)
    parser.add_argument("--depth_detail_polarity", type=float, default=1.0)
    parser.add_argument("--parallax_flow_scale", type=float, default=0.5)
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--support_close", type=int, default=81)
    parser.add_argument("--hole_fill_max_area", type=int, default=40000)
    parser.add_argument("--boundary_trim", type=int, default=36)
    parser.add_argument("--smooth_iterations", type=int, default=2)
    parser.add_argument("--max_points", type=int, default=800000)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    aliases = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = aliases[dtype_arg]
    if device.type == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU inference only supports fp32.")
    return dtype


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    mib = 1024 ** 2
    print(
        f"[CUDA memory] {label}: "
        f"allocated={torch.cuda.memory_allocated() / mib:.0f} MiB, "
        f"reserved={torch.cuda.memory_reserved() / mib:.0f} MiB, "
        f"peak={torch.cuda.max_memory_allocated() / mib:.0f} MiB, "
        f"free={free_bytes / mib:.0f}/{total_bytes / mib:.0f} MiB"
    )


def list_images(image_folder: Path) -> List[Path]:
    image_paths = [
        p for p in sorted(image_folder.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    return image_paths


def resize_rgb_with_transform(path: Path, target_width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    original_rgb = np.asarray(img)
    new_width = target_width
    new_height = round(height * (new_width / width) / 14) * 14
    scale_x = new_width / width
    scale_y = new_height / height
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    crop_y = 0
    if new_height > target_width:
        crop_y = (new_height - target_width) // 2
        img = img.crop((0, crop_y, new_width, crop_y + target_width))
    original_to_model = np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, -float(crop_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.asarray(img), original_rgb, original_to_model


def resize_rgb(path: Path, target_width: int) -> np.ndarray:
    return resize_rgb_with_transform(path, target_width)[0]


def make_sand_mask(rgb: np.ndarray, edge_erosion: int) -> np.ndarray:
    value = rgb.max(axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(value, (0, 0), 3)
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(45, int(otsu_threshold * 0.80))
    mask = (blurred > threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = int(np.argmax(areas) + 1)
        mask = (labels == keep_label).astype(np.uint8) * 255

    if edge_erosion > 0:
        k = np.ones((edge_erosion, edge_erosion), np.uint8)
        mask = cv2.erode(mask, k, iterations=1)
    return mask.astype(bool)


def enhance_gray(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced[~mask] = 0
    return enhanced


def prepare_frames(image_paths: List[Path], width: int, edge_erosion: int) -> List[FrameData]:
    sift = cv2.SIFT_create(nfeatures=6000)
    frames = []
    for idx, path in enumerate(image_paths):
        image = resize_rgb(path, width)
        mask = make_sand_mask(image, edge_erosion=edge_erosion)
        gray = enhance_gray(image, mask)
        keypoints, descriptors = sift.detectAndCompute(gray, (mask.astype(np.uint8) * 255))
        frames.append(FrameData(idx, path, image, mask, gray, keypoints, descriptors))
    return frames


def homography_quality(
    src: FrameData,
    dst: FrameData,
    matcher: cv2.BFMatcher,
    min_inliers: int,
    max_reproj_error: float,
    min_overlap: float,
) -> Optional[EdgeData]:
    if src.descriptors is None or dst.descriptors is None:
        return None
    pairs = matcher.knnMatch(src.descriptors, dst.descriptors, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None

    pts_src = np.float32([src.keypoints[m.queryIdx].pt for m in good])
    pts_dst = np.float32([dst.keypoints[m.trainIdx].pt for m in good])
    homography, inlier_mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, max_reproj_error)
    if homography is None or inlier_mask is None:
        return None

    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 4:
        return None

    projected = cv2.perspectiveTransform(pts_src[inlier_mask, None, :], homography)[:, 0, :]
    errors = np.linalg.norm(projected - pts_dst[inlier_mask], axis=1)
    median_error = float(np.median(errors))

    warped_mask = cv2.warpPerspective(
        src.mask.astype(np.uint8),
        homography,
        (dst.mask.shape[1], dst.mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    intersection = float(np.logical_and(warped_mask, dst.mask).sum())
    overlap = intersection / max(1.0, min(float(warped_mask.sum()), float(dst.mask.sum())))
    accepted = inliers >= min_inliers and median_error <= max_reproj_error and overlap >= min_overlap
    score = (inliers * overlap) / (1.0 + median_error)
    return EdgeData(src.index, dst.index, homography, inliers, len(good), median_error, overlap, score, accepted)


def build_alignment_graph(
    frames: List[FrameData],
    min_inliers: int,
    max_reproj_error: float,
    min_overlap: float,
) -> Tuple[Dict[Tuple[int, int], EdgeData], List[dict]]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edges: Dict[Tuple[int, int], EdgeData] = {}
    report = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            edge = homography_quality(frames[i], frames[j], matcher, min_inliers, max_reproj_error, min_overlap)
            if edge is None:
                report.append({"src": i, "dst": j, "accepted": False, "reason": "no_homography"})
                continue
            inv_h = np.linalg.inv(edge.homography)
            reverse = EdgeData(
                j,
                i,
                inv_h,
                edge.inliers,
                edge.good_matches,
                edge.median_error,
                edge.overlap,
                edge.score,
                edge.accepted,
            )
            edges[(i, j)] = edge
            edges[(j, i)] = reverse
            report.append(edge_to_json(edge))
    return edges, report


def edge_to_json(edge: EdgeData) -> dict:
    return {
        "src": edge.src,
        "dst": edge.dst,
        "accepted": edge.accepted,
        "inliers": edge.inliers,
        "good_matches": edge.good_matches,
        "median_reproj_error": round(edge.median_error, 4),
        "overlap": round(edge.overlap, 4),
        "score": round(edge.score, 4),
    }


def choose_anchor(frame_count: int, edges: Dict[Tuple[int, int], EdgeData]) -> int:
    scores = np.zeros(frame_count, dtype=np.float64)
    for (src, _), edge in edges.items():
        if edge.accepted:
            scores[src] += edge.score
    return int(np.argmax(scores))


def shortest_homographies(
    frame_count: int,
    anchor: int,
    edges: Dict[Tuple[int, int], EdgeData],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    dist = {i: math.inf for i in range(frame_count)}
    transforms: Dict[int, np.ndarray] = {}
    dist[anchor] = 0.0
    transforms[anchor] = np.eye(3, dtype=np.float64)
    visited = set()

    while len(visited) < frame_count:
        current = None
        current_dist = math.inf
        for idx, value in dist.items():
            if idx not in visited and value < current_dist:
                current = idx
                current_dist = value
        if current is None:
            break
        visited.add(current)

        for nxt in range(frame_count):
            edge = edges.get((nxt, current))
            if edge is None or not edge.accepted:
                continue
            cost = 1.0 / max(edge.score, 1e-6)
            new_dist = dist[current] + cost
            if new_dist < dist[nxt]:
                dist[nxt] = new_dist
                transforms[nxt] = transforms[current] @ edge.homography
    return transforms, dist


def transformed_bounds(frames: List[FrameData], selected: List[int], transforms: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    points = []
    for idx in selected:
        h, w = frames[idx].mask.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])[:, None, :]
        warped = cv2.perspectiveTransform(corners, transforms[idx])[:, 0, :]
        points.append(warped)
    all_points = np.concatenate(points, axis=0)
    return all_points.min(axis=0), all_points.max(axis=0)


def coverage_for_selection(frames: List[FrameData], selected: List[int], transforms: Dict[int, np.ndarray]) -> Tuple[int, Tuple[np.ndarray, int, int]]:
    min_xy, max_xy = transformed_bounds(frames, selected, transforms)
    margin = 20
    width = int(math.ceil(max_xy[0] - min_xy[0] + 2 * margin))
    height = int(math.ceil(max_xy[1] - min_xy[1] + 2 * margin))
    offset = np.array([[1, 0, -min_xy[0] + margin], [0, 1, -min_xy[1] + margin], [0, 0, 1]], dtype=np.float64)
    union = np.zeros((height, width), dtype=bool)
    for idx in selected:
        warped = cv2.warpPerspective(
            frames[idx].mask.astype(np.uint8),
            offset @ transforms[idx],
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        union |= warped
    return int(union.sum()), (offset, width, height)


def support_hole_stats(
    frames: List[FrameData],
    selected: List[int],
    transforms: Dict[int, np.ndarray],
) -> Tuple[int, int, int, int, int]:
    area, (offset, width, height) = coverage_for_selection(frames, selected, transforms)
    union = np.zeros((height, width), dtype=np.uint8)
    for idx in selected:
        warped = cv2.warpPerspective(
            frames[idx].mask.astype(np.uint8),
            offset @ transforms[idx],
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        union |= warped.astype(np.uint8)

    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    invalid = ~closed
    components, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), connectivity=8)
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    hole_area = 0
    hole_count = 0
    for label in range(1, components):
        if label in border_labels:
            continue
        component_area = int(stats[label, cv2.CC_STAT_AREA])
        if component_area > 50:
            hole_area += component_area
            hole_count += 1
    return int(area), int(width), int(height), int(hole_area), int(hole_count)


def parse_keyframe_indices(indices: str, frame_count: int) -> List[int]:
    if not indices.strip():
        return []
    parsed = []
    for item in indices.split(","):
        idx = int(item.strip())
        if idx < 0 or idx >= frame_count:
            raise ValueError(f"keyframe index {idx} is out of range [0, {frame_count - 1}]")
        if idx not in parsed:
            parsed.append(idx)
    if not parsed:
        raise ValueError("--keyframe_indices did not contain any valid indices")
    return parsed[:3]


def select_keyframes(
    frames: List[FrameData],
    edges: Dict[Tuple[int, int], EdgeData],
    requested: int,
    anchor_index: int = -1,
    fixed_indices: Optional[List[int]] = None,
) -> Tuple[List[int], Dict[int, np.ndarray], Dict[int, float]]:
    requested = max(1, min(requested, len(frames), 3))
    if fixed_indices:
        anchor = fixed_indices[0]
    else:
        anchor = anchor_index if 0 <= anchor_index < len(frames) else choose_anchor(len(frames), edges)
    transforms, distances = shortest_homographies(len(frames), anchor, edges)

    if fixed_indices:
        selected = [idx for idx in fixed_indices[:requested] if idx in transforms]
        if not selected:
            selected = [anchor]
        selected_transforms = {idx: transforms[idx] for idx in selected}
        selected_distances = {idx: float(distances[idx]) for idx in selected}
        return selected, selected_transforms, selected_distances

    reachable = [idx for idx in range(len(frames)) if idx in transforms and idx != anchor]
    best_selected = [anchor]
    best_score = -math.inf
    best_tuple = None
    candidate_lengths = range(1, min(requested, len(reachable) + 1) + 1)
    for length in candidate_lengths:
        if length == 1:
            combos = [()]
        else:
            combos = itertools.combinations(reachable, length - 1)
        for combo in combos:
            candidate = [anchor] + list(combo)
            if len(candidate) < requested and len(reachable) + 1 >= requested:
                continue
            area, width, height, hole_area, hole_count = support_hole_stats(frames, candidate, transforms)
            canvas_area = max(1, width * height)
            compactness_penalty = 0.015 * (canvas_area - area)
            distance_penalty = 250.0 * sum(float(distances.get(idx, 0.0)) for idx in candidate)
            score = area - 15.0 * hole_area - 2500.0 * hole_count - compactness_penalty - distance_penalty
            ranking_tuple = (score, -hole_area, -hole_count, area)
            if ranking_tuple > (best_score, *(best_tuple or (-math.inf, -math.inf, -math.inf))):
                best_score = score
                best_tuple = (-hole_area, -hole_count, area)
                best_selected = candidate

    selected = best_selected[:requested]
    selected_transforms = {idx: transforms[idx] for idx in selected}
    selected_distances = {idx: float(distances[idx]) for idx in selected}
    return selected, selected_transforms, selected_distances


def load_model(device: torch.device, dtype: torch.dtype, checkpoint: Path) -> OmniVGGT:
    model = OmniVGGT(preload_patch_embed=False)
    state_dict = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    return model.to(device=device, dtype=dtype).eval()


def load_keyframe_inputs(
    paths: List[Path],
    target_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    images = []
    rgb_images = []
    sand_masks = []
    original_rgbs = []
    original_to_model_transforms = []
    for path in paths:
        rgb, original_rgb, original_to_model = resize_rgb_with_transform(path, target_size)
        sand_mask = make_sand_mask(rgb, edge_erosion=7)
        tensor = ImgNorm(Image.fromarray(rgb))
        images.append(tensor)
        rgb_images.append(rgb)
        sand_masks.append(sand_mask)
        original_rgbs.append(original_rgb)
        original_to_model_transforms.append(original_to_model)

    image_tensor = torch.stack(images, dim=0)
    _, height, width = image_tensor.shape[1:]
    s = len(paths)
    depth = torch.zeros((1, s, height, width, 1), dtype=torch.float32)
    mask = torch.zeros((1, s, height, width), dtype=torch.float32)
    extrinsics = torch.zeros((1, s, 3, 4), dtype=torch.float32)
    intrinsics = torch.zeros((1, s, 3, 3), dtype=torch.float32)
    input_dtype = dtype if device.type == "cuda" else torch.float32
    inputs = {
        "images": image_tensor.to(device=device, dtype=input_dtype),
        "extrinsics": extrinsics.to(device=device),
        "intrinsics": intrinsics.to(device=device),
        "depth": depth.to(device=device, dtype=input_dtype),
        "mask": mask.to(device=device, dtype=input_dtype),
        "depth_gt_index": [],
        "camera_gt_index": [],
    }
    return inputs, rgb_images, sand_masks, original_rgbs, original_to_model_transforms


def run_inference(model: OmniVGGT, inputs: dict, dtype: torch.dtype, device: torch.device) -> dict:
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        predictions = model.inference(**inputs)
    return predictions


def build_canvas_transforms(
    selected: List[int],
    model_frames: List[FrameData],
    model_edges: Dict[Tuple[int, int], EdgeData],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, int, int, Dict[int, float]]:
    anchor = selected[0]
    transforms, distances = shortest_homographies(len(model_frames), anchor, model_edges)
    usable = [idx for idx in selected if idx in transforms]
    if not usable:
        usable = [anchor]
        transforms[anchor] = np.eye(3, dtype=np.float64)
    _, canvas = coverage_for_selection(model_frames, usable, transforms)
    offset, width, height = canvas
    return {idx: offset @ transforms[idx] for idx in usable}, offset, width, height, distances


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = sorted_weights.sum() * 0.5
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def robust_mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def collect_frame_samples(
    predictions: dict,
    rgb_images: List[np.ndarray],
    sand_masks: List[np.ndarray],
    h_to_canvas: Dict[int, np.ndarray],
    selected_indices: List[int],
    canvas_width: int,
    canvas_height: int,
    conf_percentile: float,
) -> List[FrameSamples]:
    depths = predictions["depth"].detach().float().cpu().numpy()[0, ..., 0]
    confs = predictions["depth_conf"].detach().float().cpu().numpy()[0]

    frame_samples = []
    for local_idx, original_idx in enumerate(selected_indices):
        if original_idx not in h_to_canvas:
            continue
        depth = depths[local_idx]
        conf = confs[local_idx]
        rgb = rgb_images[local_idx]
        mask = sand_masks[local_idx]
        conf_threshold = np.percentile(conf[mask], conf_percentile) if np.any(mask) else np.percentile(conf, conf_percentile)
        valid = mask & np.isfinite(depth) & (depth > 0) & (conf >= conf_threshold)
        yy, xx = np.nonzero(valid)
        if len(xx) == 0:
            continue
        points = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)[:, None, :]
        warped = cv2.perspectiveTransform(points, h_to_canvas[original_idx])[:, 0, :]
        gx = np.rint(warped[:, 0]).astype(np.int64)
        gy = np.rint(warped[:, 1]).astype(np.int64)
        inside = (gx >= 0) & (gx < canvas_width) & (gy >= 0) & (gy < canvas_height)
        x = gx[inside]
        y = gy[inside]
        z = depth[yy[inside], xx[inside]].astype(np.float32)
        weight = conf[yy[inside], xx[inside]].astype(np.float32) + 1e-6
        color = rgb[yy[inside], xx[inside]].astype(np.float32)
        frame_samples.append(
            FrameSamples(
                local_index=local_idx,
                original_index=original_idx,
                x=x,
                y=y,
                z=z,
                weight=weight,
                rgb=color,
                key=y * canvas_width + x,
            )
        )

    if not frame_samples:
        raise RuntimeError("No valid point samples survived masking and confidence filtering.")
    return frame_samples


def collapse_frame_samples(sample: FrameSamples, canvas_width: int) -> FrameSamples:
    if len(sample.key) == 0:
        return sample
    order = np.argsort(sample.key)
    keys = sample.key[order]
    z = sample.z[order]
    weight = sample.weight[order]
    rgb = sample.rgb[order]

    unique_keys = []
    collapsed_z = []
    collapsed_weight = []
    collapsed_rgb = []
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    ends = np.r_[starts[1:], len(keys)]
    for start, end in zip(starts, ends):
        w_group = weight[start:end]
        z_group = z[start:end]
        rgb_group = rgb[start:end]
        unique_keys.append(keys[start])
        collapsed_z.append(weighted_median(z_group, w_group))
        collapsed_weight.append(float(w_group.sum()))
        collapsed_rgb.append(np.average(rgb_group, axis=0, weights=w_group))

    unique_keys = np.asarray(unique_keys, dtype=np.int64)
    return FrameSamples(
        local_index=sample.local_index,
        original_index=sample.original_index,
        x=(unique_keys % canvas_width).astype(np.int64),
        y=(unique_keys // canvas_width).astype(np.int64),
        z=np.asarray(collapsed_z, dtype=np.float32),
        weight=np.asarray(collapsed_weight, dtype=np.float32),
        rgb=np.asarray(collapsed_rgb, dtype=np.float32),
        key=unique_keys,
    )


def fit_depth_alignment(
    source_z: np.ndarray,
    reference_z: np.ndarray,
    weights: np.ndarray,
    mode: str,
) -> Tuple[float, float, str]:
    if mode == "none":
        return 1.0, 0.0, "none"
    if len(source_z) < 2:
        return 1.0, 0.0, "insufficient_overlap"

    offset = weighted_median(reference_z - source_z, weights)
    if mode == "median":
        return 1.0, offset, "median"

    scale = 1.0
    bias = offset
    method = "affine"
    keep = np.ones_like(source_z, dtype=bool)
    for _ in range(4):
        if keep.sum() < 20:
            method = "median_fallback"
            return 1.0, offset, method
        x = source_z[keep].astype(np.float64)
        y = reference_z[keep].astype(np.float64)
        w = np.sqrt(weights[keep].astype(np.float64))
        design = np.column_stack([x, np.ones_like(x)])
        try:
            scale, bias = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            method = "median_fallback"
            return 1.0, offset, method
        if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0 or scale > 8:
            method = "median_fallback"
            return 1.0, offset, method
        residual = reference_z - (scale * source_z + bias)
        med = np.median(residual)
        mad = robust_mad(residual)
        keep = np.abs(residual - med) <= max(1e-6, 3.0 * 1.4826 * (mad + 1e-6))
    return float(scale), float(bias), method


def merge_reference_depths(ref: FrameSamples, sample: FrameSamples, canvas_width: int) -> FrameSamples:
    merged = FrameSamples(
        local_index=-1,
        original_index=-1,
        x=np.concatenate([ref.x, sample.x]),
        y=np.concatenate([ref.y, sample.y]),
        z=np.concatenate([ref.z, sample.z]),
        weight=np.concatenate([ref.weight, sample.weight]),
        rgb=np.concatenate([ref.rgb, sample.rgb]),
        key=np.concatenate([ref.key, sample.key]),
    )
    return collapse_frame_samples(merged, canvas_width)


def align_frame_depths(
    frames: List[FrameSamples],
    canvas_width: int,
    mode: str,
    min_overlap_cells: int,
) -> Tuple[List[FrameSamples], List[dict]]:
    collapsed = [collapse_frame_samples(frame, canvas_width) for frame in frames]
    if not collapsed:
        return [], []

    aligned = []
    report = []
    reference = collapsed[0]
    for idx, frame in enumerate(collapsed):
        if idx == 0 or mode == "none":
            scale, bias, method = 1.0, 0.0, "reference" if idx == 0 else "none"
            common_count = 0
            before_median = 0.0
            after_median = 0.0
        else:
            common, ref_pos, frame_pos = np.intersect1d(reference.key, frame.key, return_indices=True)
            common_count = int(len(common))
            if common_count >= min_overlap_cells:
                ref_z = reference.z[ref_pos]
                src_z = frame.z[frame_pos]
                weights = np.minimum(reference.weight[ref_pos], frame.weight[frame_pos]) + 1e-6
                before = ref_z - src_z
                before_median = float(np.median(np.abs(before - np.median(before))))
                scale, bias, method = fit_depth_alignment(src_z, ref_z, weights, mode)
                after = ref_z - (scale * src_z + bias)
                after_median = float(np.median(np.abs(after - np.median(after))))
            else:
                ref_med = weighted_median(reference.z, reference.weight)
                src_med = weighted_median(frame.z, frame.weight)
                ref_mad = max(robust_mad(reference.z), 1e-6)
                src_mad = max(robust_mad(frame.z), 1e-6)
                scale = ref_mad / src_mad if mode == "affine" else 1.0
                bias = ref_med - scale * src_med
                method = "global_distribution_fallback"
                before_median = float("nan")
                after_median = float("nan")

        aligned_frame = FrameSamples(
            local_index=frame.local_index,
            original_index=frame.original_index,
            x=frame.x,
            y=frame.y,
            z=(scale * frame.z + bias).astype(np.float32),
            weight=frame.weight,
            rgb=frame.rgb,
            key=frame.key,
        )
        aligned.append(aligned_frame)
        reference = merge_reference_depths(reference, aligned_frame, canvas_width)
        report.append(
            {
                "local_index": frame.local_index,
                "original_index": frame.original_index,
                "method": method,
                "overlap_cells": common_count,
                "scale": round(float(scale), 8),
                "offset": round(float(bias), 8),
                "median_abs_residual_before": before_median,
                "median_abs_residual_after": after_median,
            }
        )
    return aligned, report


def apply_reference_fill(
    frames: List[FrameSamples],
    canvas_width: int,
    canvas_height: int,
    dilation: int,
    min_fill_cells: int,
) -> Tuple[List[FrameSamples], List[dict]]:
    if not frames:
        return [], []

    filtered = []
    report = []
    occupied = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    kernel_size = max(1, int(dilation))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        in_bounds = (
            (frame.x >= 0)
            & (frame.x < canvas_width)
            & (frame.y >= 0)
            & (frame.y < canvas_height)
        )
        if idx == 0:
            keep = in_bounds
            method = "reference"
        else:
            blocked = cv2.dilate(occupied, kernel, iterations=1).astype(bool)
            keep = in_bounds & ~blocked[frame.y.clip(0, canvas_height - 1), frame.x.clip(0, canvas_width - 1)]
            method = "fill_uncovered"
            if int(keep.sum()) < min_fill_cells:
                keep = np.zeros_like(keep, dtype=bool)
                method = "discard_small_fill"

        kept = FrameSamples(
            local_index=frame.local_index,
            original_index=frame.original_index,
            x=frame.x[keep],
            y=frame.y[keep],
            z=frame.z[keep],
            weight=frame.weight[keep],
            rgb=frame.rgb[keep],
            key=frame.key[keep],
        )
        filtered.append(kept)
        if len(kept.x) > 0:
            occupied[kept.y, kept.x] = 1
        report.append(
            {
                "local_index": frame.local_index,
                "original_index": frame.original_index,
                "method": method,
                "input_cells": int(len(frame.x)),
                "kept_cells": int(len(kept.x)),
                "min_fill_cells": int(min_fill_cells) if idx > 0 else 0,
            }
        )
    return filtered, report


def reject_bad_depth_aligned_frames(
    frames: List[FrameSamples],
    depth_report: List[dict],
    max_aligned_depth_mad: float,
    allow_fallback: bool,
) -> Tuple[List[FrameSamples], List[dict]]:
    filtered = []
    rejection_report = []
    for idx, (frame, report) in enumerate(zip(frames, depth_report)):
        reject_reason = None
        if idx > 0:
            method = str(report.get("method", ""))
            after = report.get("median_abs_residual_after")
            if "fallback" in method and not allow_fallback:
                reject_reason = f"depth_alignment_{method}"
            elif isinstance(after, (int, float)) and np.isfinite(after) and after > max_aligned_depth_mad:
                reject_reason = f"depth_residual_{after:.6f}_gt_{max_aligned_depth_mad:.6f}"

        if reject_reason is None:
            filtered.append(frame)
            kept_cells = int(len(frame.x))
        else:
            filtered.append(
                FrameSamples(
                    local_index=frame.local_index,
                    original_index=frame.original_index,
                    x=np.empty((0,), dtype=np.int64),
                    y=np.empty((0,), dtype=np.int64),
                    z=np.empty((0,), dtype=np.float32),
                    weight=np.empty((0,), dtype=np.float32),
                    rgb=np.empty((0, 3), dtype=np.float32),
                    key=np.empty((0,), dtype=np.int64),
                )
            )
            kept_cells = 0

        rejection_report.append(
            {
                "local_index": frame.local_index,
                "original_index": frame.original_index,
                "input_cells": int(len(frame.x)),
                "kept_cells": kept_cells,
                "reject_reason": reject_reason,
            }
        )
    return filtered, rejection_report


def smooth_z_grid(
    xy: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if iterations <= 0 or len(z) == 0:
        return xy, z, rgb

    x = xy[:, 0].astype(np.int64)
    y = xy[:, 1].astype(np.int64)
    valid = (x >= 0) & (x < canvas_width) & (y >= 0) & (y < canvas_height)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    rgb = rgb[valid]

    grid_z = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    grid_w = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    np.add.at(grid_z, (y, x), z)
    np.add.at(grid_w, (y, x), 1.0)
    mask = grid_w > 0
    grid_z[mask] /= grid_w[mask]

    for _ in range(iterations):
        weighted = cv2.GaussianBlur(grid_z * mask.astype(np.float32), (5, 5), 0)
        weights = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        smoothed = weighted / np.maximum(weights, 1e-6)
        grid_z[mask] = smoothed[mask]

    out_z = grid_z[y, x]
    out_xy = np.column_stack([x, y]).astype(np.float32)
    return out_xy, out_z.astype(np.float32), rgb


def fuse_aligned_samples(
    frames: List[FrameSamples],
    canvas_width: int,
    canvas_height: int,
    depth_mad_multiplier: float,
    max_points: int,
    smooth_iterations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_x = [frame.x for frame in frames]
    all_y = [frame.y for frame in frames]
    all_z = [frame.z for frame in frames]
    all_conf = [frame.weight for frame in frames]
    all_rgb = [frame.rgb for frame in frames]
    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)
    zs = np.concatenate(all_z).astype(np.float32)
    weights = np.concatenate(all_conf).astype(np.float32) + 1e-6
    rgbs = np.concatenate(all_rgb).astype(np.float32)
    keys = ys * canvas_width + xs
    order = np.argsort(keys)
    keys = keys[order]
    xs = xs[order]
    ys = ys[order]
    zs = zs[order]
    weights = weights[order]
    rgbs = rgbs[order]

    fused_xy = []
    fused_z = []
    fused_rgb = []
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    ends = np.r_[starts[1:], len(keys)]
    for start, end in zip(starts, ends):
        z_group = zs[start:end]
        w_group = weights[start:end]
        rgb_group = rgbs[start:end]
        med = weighted_median(z_group, w_group)
        mad = np.median(np.abs(z_group - med))
        keep = np.abs(z_group - med) <= max(1e-5, depth_mad_multiplier * (mad + 1e-6))
        if not np.any(keep):
            keep = np.ones_like(z_group, dtype=bool)
        kept_weights = w_group[keep]
        fused_xy.append((xs[start], ys[start]))
        fused_z.append(weighted_median(z_group[keep], kept_weights))
        fused_rgb.append(np.average(rgb_group[keep], axis=0, weights=kept_weights))

    xy = np.asarray(fused_xy, dtype=np.float32)
    z = np.asarray(fused_z, dtype=np.float32)
    rgb = np.clip(np.asarray(fused_rgb), 0, 255).astype(np.uint8)

    low, high = np.percentile(z, [1.0, 99.0])
    keep = (z >= low) & (z <= high)
    xy = xy[keep]
    z = z[keep]
    rgb = rgb[keep]
    xy, z, rgb = smooth_z_grid(xy, z, rgb, canvas_width, canvas_height, smooth_iterations)

    xy_scale = float(max(canvas_width, canvas_height))
    z_center = float(np.median(z))
    z_scale = max(abs(z_center), 1e-6)
    points = np.column_stack(
        [
            (xy[:, 0] - canvas_width * 0.5) / xy_scale,
            -(xy[:, 1] - canvas_height * 0.5) / xy_scale,
            (z - z_center) / z_scale,
        ]
    ).astype(np.float32)

    if len(points) > max_points:
        step = int(math.ceil(len(points) / max_points))
        points = points[::step]
        rgb = rgb[::step]
    return points, rgb


def feather_from_mask(mask: np.ndarray, power: float) -> np.ndarray:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    if np.any(dist > 0):
        scale = max(1.0, float(np.percentile(dist[dist > 0], 85)) * 0.35)
        feather = np.clip(dist / scale, 0.0, 1.0)
    else:
        feather = dist.astype(np.float32)
    if power != 1.0:
        feather = feather ** power
    return feather.astype(np.float32)


def warp_surface_maps(
    predictions: dict,
    rgb_images: List[np.ndarray],
    original_rgbs: List[np.ndarray],
    original_to_model_transforms: List[np.ndarray],
    sand_masks: List[np.ndarray],
    h_to_canvas: Dict[int, np.ndarray],
    selected_indices: List[int],
    canvas_width: int,
    canvas_height: int,
    conf_percentile: float,
    feather_power: float,
    min_feather: float,
    output_scale: int,
    texture_source: str,
) -> List[dict]:
    depths = predictions["depth"].detach().float().cpu().numpy()[0, ..., 0]
    confs = predictions["depth_conf"].detach().float().cpu().numpy()[0]
    surfaces = []
    scale = max(1, int(output_scale))
    out_width = int(canvas_width * scale)
    out_height = int(canvas_height * scale)
    output_scale_h = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    for local_idx, original_idx in enumerate(selected_indices):
        if original_idx not in h_to_canvas:
            continue
        depth = depths[local_idx].astype(np.float32)
        conf = confs[local_idx].astype(np.float32)
        rgb = rgb_images[local_idx].astype(np.float32)
        mask = sand_masks[local_idx]
        conf_threshold = np.percentile(conf[mask], conf_percentile) if np.any(mask) else np.percentile(conf, conf_percentile)
        valid = mask & np.isfinite(depth) & (depth > 0) & (conf >= conf_threshold)
        homography = output_scale_h @ h_to_canvas[original_idx]

        z = cv2.warpPerspective(depth, homography, (out_width, out_height), flags=cv2.INTER_LINEAR)
        c = cv2.warpPerspective(conf, homography, (out_width, out_height), flags=cv2.INTER_LINEAR)
        warped_valid = cv2.warpPerspective(
            valid.astype(np.uint8),
            homography,
            (out_width, out_height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        if texture_source == "blended_model":
            warped_rgb = cv2.warpPerspective(rgb, homography, (out_width, out_height), flags=cv2.INTER_LINEAR)
        else:
            texture_homography = homography @ original_to_model_transforms[local_idx]
            warped_rgb = cv2.warpPerspective(
                original_rgbs[local_idx].astype(np.float32),
                texture_homography,
                (out_width, out_height),
                flags=cv2.INTER_LINEAR,
            )
        feather = feather_from_mask(warped_valid, feather_power)
        warped_valid = warped_valid & np.isfinite(z) & (z > 0) & (feather >= min_feather)
        weight = np.where(warped_valid, c * feather + 1e-6, 0.0).astype(np.float32)
        surfaces.append(
            {
                "local_index": local_idx,
                "original_index": original_idx,
                "z": z.astype(np.float32),
                "weight": weight,
                "rgb": warped_rgb.astype(np.float32),
                "valid": warped_valid,
            }
        )
    if not surfaces:
        raise RuntimeError("No valid warped surfaces were produced.")
    return surfaces


def local_depth_correction(
    src_z: np.ndarray,
    ref_z: np.ndarray,
    overlap: np.ndarray,
    overlap_weight: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, float, float]:
    residual = np.zeros_like(src_z, dtype=np.float32)
    residual[overlap] = (ref_z[overlap] - src_z[overlap]).astype(np.float32)
    values = residual[overlap]
    if len(values) == 0:
        return np.zeros_like(src_z, dtype=np.float32), float("nan"), float("nan")

    med = float(np.median(values))
    mad = robust_mad(values)
    keep = overlap & (np.abs(residual - med) <= max(1e-6, 3.0 * 1.4826 * (mad + 1e-6)))
    weighted_residual = np.zeros_like(src_z, dtype=np.float32)
    weights = np.zeros_like(src_z, dtype=np.float32)
    weighted_residual[keep] = residual[keep] * overlap_weight[keep]
    weights[keep] = overlap_weight[keep]

    num = cv2.GaussianBlur(weighted_residual, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma)
    correction = np.divide(num, np.maximum(den, 1e-6), out=np.full_like(num, med), where=den > 1e-6)
    before = float(np.median(np.abs(values - np.median(values))))
    after_values = values - correction[overlap]
    after = float(np.median(np.abs(after_values - np.median(after_values))))
    return correction.astype(np.float32), before, after


def align_warped_surfaces(
    surfaces: List[dict],
    depth_align: str,
    min_overlap_cells: int,
    local_depth_sigma: float,
    update_reference_depth: bool,
) -> Tuple[List[dict], List[dict]]:
    reference = surfaces[0]
    reference_z = reference["z"].copy()
    reference_valid = reference["valid"].copy()
    reference_weight = reference["weight"].copy()
    aligned = []
    report = []

    for idx, surface in enumerate(surfaces):
        z = surface["z"].copy()
        valid = surface["valid"].copy()
        if idx == 0 or depth_align == "none":
            scale, bias, method = 1.0, 0.0, "reference" if idx == 0 else "none"
            before_local, after_local = 0.0, 0.0
            overlap_count = 0
        else:
            overlap = valid & reference_valid
            overlap_count = int(overlap.sum())
            if overlap_count >= min_overlap_cells:
                weights = np.minimum(surface["weight"], reference_weight) + 1e-6
                scale, bias, method = fit_depth_alignment(z[overlap], reference_z[overlap], weights[overlap], depth_align)
                z = scale * z + bias
                correction, before_local, after_local = local_depth_correction(
                    z,
                    reference_z,
                    overlap,
                    weights,
                    sigma=local_depth_sigma,
                )
                z = z + correction
                method = f"{method}+local"
            else:
                scale, bias, method = 1.0, 0.0, "insufficient_overlap"
                before_local, after_local = float("nan"), float("nan")

        aligned_surface = dict(surface)
        aligned_surface["z"] = z.astype(np.float32)
        aligned.append(aligned_surface)

        if update_reference_depth and idx > 0 and np.any(valid):
            update_weight = surface["weight"]
            combined_weight = reference_weight + update_weight
            update = valid & (update_weight > 0)
            both = reference_valid & update
            only = update & ~reference_valid
            reference_z[only] = z[only]
            reference_z[both] = (
                reference_z[both] * reference_weight[both] + z[both] * update_weight[both]
            ) / np.maximum(combined_weight[both], 1e-6)
            reference_weight[update] = np.maximum(reference_weight[update], update_weight[update])
            reference_valid |= update

        report.append(
            {
                "local_index": surface["local_index"],
                "original_index": surface["original_index"],
                "method": method,
                "overlap_cells": overlap_count,
                "scale": round(float(scale), 8),
                "offset": round(float(bias), 8),
                "local_median_abs_residual_before": before_local,
                "local_median_abs_residual_after": after_local,
            }
        )
    return aligned, report


def fit_plane_to_heightfield(
    xy: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> Tuple[np.ndarray, dict]:
    if len(z) < 3:
        return np.zeros_like(z, dtype=np.float32), {
            "method": "insufficient_points",
            "coefficients": [0.0, 0.0, 0.0],
            "inlier_count": int(len(z)),
            "residual_mad": 0.0,
        }

    scale = float(max(canvas_width, canvas_height))
    x = (xy[:, 0].astype(np.float64) - canvas_width * 0.5) / scale
    y = (xy[:, 1].astype(np.float64) - canvas_height * 0.5) / scale
    values = z.astype(np.float64)
    sample_weights = np.maximum(weights.astype(np.float64), 1e-6)
    design = np.column_stack([x, y, np.ones_like(x)])
    keep = np.isfinite(values)
    coeffs = np.array([0.0, 0.0, float(np.median(values[keep]))], dtype=np.float64)

    for _ in range(5):
        if keep.sum() < 3:
            break
        w = np.sqrt(sample_weights[keep])
        try:
            coeffs = np.linalg.lstsq(design[keep] * w[:, None], values[keep] * w, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        residual = values - design @ coeffs
        kept_residual = residual[keep]
        med = float(np.median(kept_residual))
        mad = robust_mad(kept_residual)
        keep = np.abs(residual - med) <= max(1e-6, 3.0 * 1.4826 * (mad + 1e-6))

    plane = design @ coeffs
    residual = values - plane
    kept_residual = residual[keep] if keep.any() else residual
    report = {
        "method": "robust_plane",
        "coefficients": [round(float(v), 8) for v in coeffs],
        "inlier_count": int(keep.sum()),
        "residual_mad": float(robust_mad(kept_residual)),
        "residual_p01_p99": [
            float(np.percentile(kept_residual, 1.0)),
            float(np.percentile(kept_residual, 99.0)),
        ],
    }
    return plane.astype(np.float32), report


def masked_gaussian(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    values = values.astype(np.float32)
    mask_f = mask.astype(np.float32)
    weighted = cv2.GaussianBlur(values * mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    weights = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.divide(weighted, np.maximum(weights, 1e-6), out=np.zeros_like(values), where=weights > 1e-6)


def image_guided_depth_detail(
    rgb_grid: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
    polarity: float,
) -> Tuple[np.ndarray, dict]:
    luma = (
        0.299 * rgb_grid[..., 0]
        + 0.587 * rgb_grid[..., 1]
        + 0.114 * rgb_grid[..., 2]
    ).astype(np.float32) / 255.0
    smooth = masked_gaussian(luma, valid_mask, sigma=max(0.1, float(sigma)))
    highpass = np.zeros_like(luma, dtype=np.float32)
    highpass[valid_mask] = (luma - smooth)[valid_mask]
    values = highpass[valid_mask]
    if len(values) == 0:
        return highpass, {"detail_mad": 0.0, "detail_clip_fraction": 0.0}

    center = float(np.median(values))
    mad = max(robust_mad(values), 1e-6)
    normalized = (highpass - center) / (3.0 * 1.4826 * mad)
    clipped = np.clip(normalized, -1.0, 1.0).astype(np.float32)
    clip_fraction = float(np.mean(np.abs(normalized[valid_mask]) > 1.0))
    return (clipped * float(polarity)).astype(np.float32), {
        "detail_mad": float(mad),
        "detail_clip_fraction": clip_fraction,
        "detail_highpass_p01_p99": [
            float(np.percentile(values, 1.0)),
            float(np.percentile(values, 99.0)),
        ],
    }


def enhance_flow_gray(rgb: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced[~valid_mask] = 0
    return enhanced


def parallax_flow_depth_detail(
    stack_rgb: np.ndarray,
    stack_valid: np.ndarray,
    valid_mask: np.ndarray,
    sigma: float,
    polarity: float,
    flow_scale: float,
) -> Tuple[np.ndarray, dict]:
    if len(stack_rgb) < 2:
        return np.zeros(valid_mask.shape, dtype=np.float32), {
            "parallax_pair_count": 0,
            "parallax_valid_cells": 0,
        }

    height, width = valid_mask.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = width * 0.5
    center_y = height * 0.5
    radial_x = xx - center_x
    radial_y = yy - center_y
    radial_norm = np.sqrt(radial_x * radial_x + radial_y * radial_y)
    radial_x = np.divide(radial_x, np.maximum(radial_norm, 1e-6))
    radial_y = np.divide(radial_y, np.maximum(radial_norm, 1e-6))

    anchor_valid = stack_valid[0] & valid_mask
    anchor_gray = enhance_flow_gray(stack_rgb[0], anchor_valid)
    detail_sum = np.zeros((height, width), dtype=np.float32)
    detail_weight = np.zeros((height, width), dtype=np.float32)
    pair_reports = []

    if hasattr(cv2, "DISOpticalFlow_create"):
        flow_solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    else:
        flow_solver = None

    for idx in range(1, len(stack_rgb)):
        overlap = anchor_valid & stack_valid[idx] & valid_mask
        overlap_count = int(overlap.sum())
        if overlap_count < 5000:
            pair_reports.append({"pair_index": idx, "overlap_cells": overlap_count, "used": False})
            continue

        other_gray = enhance_flow_gray(stack_rgb[idx], overlap)
        if flow_solver is not None:
            flow = flow_solver.calc(other_gray, anchor_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                other_gray,
                anchor_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

        scalar = (flow[..., 0] * radial_x + flow[..., 1] * radial_y).astype(np.float32)
        low = masked_gaussian(scalar, overlap, sigma=max(1.0, float(sigma) * 2.0))
        high = scalar - low
        values = high[overlap]
        med = float(np.median(values))
        mad = max(robust_mad(values), 1e-6)
        normalized = np.clip((high - med) / (3.0 * 1.4826 * mad), -1.0, 1.0).astype(np.float32)

        # Flow near textureless areas is unreliable; use local image gradient as a soft reliability weight.
        grad_x = cv2.Sobel(anchor_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(anchor_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        grad_weight = np.clip(grad / max(float(np.percentile(grad[overlap], 90.0)), 1e-6), 0.0, 1.0)
        weight = overlap.astype(np.float32) * (0.25 + 0.75 * grad_weight)
        detail_sum += normalized * weight
        detail_weight += weight
        pair_reports.append(
            {
                "pair_index": idx,
                "overlap_cells": overlap_count,
                "used": True,
                "flow_scalar_mad": float(mad),
                "flow_scalar_p01_p99": [
                    float(np.percentile(values, 1.0)),
                    float(np.percentile(values, 99.0)),
                ],
            }
        )

    detail = np.divide(detail_sum, np.maximum(detail_weight, 1e-6), out=np.zeros_like(detail_sum), where=detail_weight > 1e-6)
    detail = np.clip(detail * float(flow_scale) * float(polarity), -1.0, 1.0).astype(np.float32)
    values = detail[valid_mask & (detail_weight > 0)]
    return detail, {
        "parallax_pair_count": int(sum(1 for item in pair_reports if item.get("used"))),
        "parallax_valid_cells": int((valid_mask & (detail_weight > 0)).sum()),
        "parallax_flow_scale": float(flow_scale),
        "parallax_detail_p01_p99": (
            [float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))]
            if len(values) > 0 else [0.0, 0.0]
        ),
        "parallax_pairs": pair_reports,
    }


def fill_internal_holes(
    z_grid: np.ndarray,
    rgb_grid: np.ndarray,
    valid_mask: np.ndarray,
    support_close: int,
    max_hole_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    fill_mask = np.zeros_like(valid_mask, dtype=bool)
    support_added_cells = 0
    if support_close > 0 and np.any(valid_mask):
        kernel_size = int(support_close)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_support = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        close_fill = closed_support & ~valid_mask
        support_added_cells = int(close_fill.sum())
        fill_mask |= close_fill

    if max_hole_area <= 0:
        if support_added_cells == 0:
            return z_grid, rgb_grid, valid_mask, {
                "hole_fill_enabled": False,
                "hole_count": 0,
                "filled_cells": 0,
                "support_close": int(support_close),
                "support_added_cells": 0,
            }
        max_hole_area = 0

    support_mask = valid_mask | fill_mask
    if max_hole_area <= 0:
        hole_count = 0
    else:
        invalid = ~support_mask
        components, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), connectivity=8)
        hole_count = 0
        if components > 1:
            border_labels = set(np.unique(labels[0, :]))
            border_labels.update(np.unique(labels[-1, :]))
            border_labels.update(np.unique(labels[:, 0]))
            border_labels.update(np.unique(labels[:, -1]))

            for label in range(1, components):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if label in border_labels or area > max_hole_area:
                    continue
                fill_mask |= labels == label
                hole_count += 1

    filled_cells = int(fill_mask.sum())
    if filled_cells == 0:
        return z_grid, rgb_grid, valid_mask, {
            "hole_fill_enabled": True,
            "hole_count": int(hole_count),
            "filled_cells": 0,
            "support_close": int(support_close),
            "support_added_cells": 0,
        }

    inpaint_mask = (fill_mask.astype(np.uint8) * 255)
    valid_z = z_grid[valid_mask]
    lo, hi = np.percentile(valid_z, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(valid_z))
        hi = float(np.max(valid_z) + 1e-6)
    z_u8 = np.clip((z_grid - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    z_inpaint = cv2.inpaint(z_u8, inpaint_mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
    filled_z = z_grid.copy()
    filled_z[fill_mask] = lo + (z_inpaint[fill_mask] / 255.0) * (hi - lo)

    rgb_u8 = np.clip(rgb_grid, 0, 255).astype(np.uint8)
    filled_rgb = cv2.inpaint(rgb_u8, inpaint_mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
    out_rgb = rgb_grid.copy()
    out_rgb[fill_mask] = filled_rgb[fill_mask]

    out_valid = valid_mask | fill_mask
    return filled_z, out_rgb, out_valid, {
        "hole_fill_enabled": True,
        "hole_count": int(hole_count),
        "filled_cells": filled_cells,
        "support_close": int(support_close),
        "support_added_cells": int(support_added_cells),
        "hole_fill_max_area": int(max_hole_area),
    }


def blend_warped_surfaces(
    surfaces: List[dict],
    canvas_width: int,
    canvas_height: int,
    depth_consistency: float,
    max_points: int,
    smooth_iterations: int,
    surface_model: str,
    height_exaggeration: float,
    plane_residual_mad_multiplier: float,
    texture_source: str,
    depth_detail_source: str,
    depth_detail_strength: float,
    depth_detail_sigma: float,
    depth_detail_polarity: float,
    parallax_flow_scale: float,
    min_support_frames: int,
    support_close: int,
    hole_fill_max_area: int,
    boundary_trim: int,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    stack_z = np.stack([s["z"] for s in surfaces], axis=0)
    stack_w = np.stack([s["weight"] for s in surfaces], axis=0)
    stack_valid = np.stack([s["valid"] for s in surfaces], axis=0)
    stack_rgb = np.stack([s["rgb"] for s in surfaces], axis=0)
    stack_w = np.where(stack_valid, stack_w, 0.0).astype(np.float32)

    valid_count = stack_valid.sum(axis=0)
    required_support = min(max(1, int(min_support_frames)), len(surfaces))
    any_valid = valid_count >= required_support
    z_for_median = np.where(stack_valid, stack_z, np.nan)
    consensus = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    consensus[any_valid] = np.nanmedian(z_for_median[:, any_valid], axis=0)
    residual = np.abs(stack_z - consensus[None, :, :])
    residual_for_mad = np.where(stack_valid, residual, np.nan)
    local_mad = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    local_mad[any_valid] = np.nanmedian(residual_for_mad[:, any_valid], axis=0)
    tau = np.maximum(depth_consistency, 2.5 * np.nan_to_num(local_mad, nan=0.0) + 1e-6)
    consistency = np.exp(-((residual / tau[None, :, :]) ** 2)).astype(np.float32)
    weights = stack_w * consistency
    sum_w = weights.sum(axis=0)
    good = any_valid & (sum_w > 1e-6)
    fused_z = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    fused_z[good] = (weights * stack_z).sum(axis=0)[good] / sum_w[good]
    fused_rgb = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    for channel in range(3):
        fused_rgb[..., channel][good] = (weights * stack_rgb[..., channel]).sum(axis=0)[good] / sum_w[good]
    if texture_source == "anchor_original" and len(surfaces) > 0:
        anchor_valid = stack_valid[0] & good
        fused_rgb[anchor_valid] = stack_rgb[0][anchor_valid]

    fused_z, fused_rgb, good, hole_report = fill_internal_holes(
        fused_z,
        fused_rgb,
        good,
        support_close=support_close,
        max_hole_area=hole_fill_max_area,
    )
    pre_trim_cells = int(good.sum())
    boundary_removed_cells = 0
    if boundary_trim > 0 and np.any(good):
        dist = cv2.distanceTransform(good.astype(np.uint8), cv2.DIST_L2, 3)
        trimmed_good = good & (dist >= float(boundary_trim))
        boundary_removed_cells = int(good.sum() - trimmed_good.sum())
        good = trimmed_good
        fused_z, fused_rgb, good, post_trim_hole_report = fill_internal_holes(
            fused_z,
            fused_rgb,
            good,
            support_close=0,
            max_hole_area=hole_fill_max_area,
        )
    else:
        post_trim_hole_report = {
            "hole_count": 0,
            "filled_cells": 0,
            "support_added_cells": 0,
        }

    ys, xs = np.nonzero(good)
    xy = np.column_stack([xs, ys]).astype(np.float32)
    z = fused_z[ys, xs]
    rgb = np.clip(fused_rgb[ys, xs], 0, 255).astype(np.uint8)
    low, high = np.percentile(z, [1.0, 99.0])
    z = np.clip(z, low, high)
    xy, z, rgb = smooth_z_grid(xy, z, rgb, canvas_width, canvas_height, smooth_iterations)
    x = xy[:, 0].astype(np.int64)
    y = xy[:, 1].astype(np.int64)
    point_weights = sum_w[y, x].astype(np.float32)
    base_depth_scale = max(abs(weighted_median(z, point_weights + 1e-6)), 1e-6)

    surface_report = {
        "method": surface_model,
        "height_exaggeration": float(height_exaggeration),
        "base_depth_scale": float(base_depth_scale),
        "texture_source": texture_source,
        "min_support_frames": int(min_support_frames),
        "required_support_frames": int(required_support),
        "pre_boundary_trim_cells": pre_trim_cells,
        "boundary_trim": int(boundary_trim),
        "boundary_removed_cells": boundary_removed_cells,
    }
    surface_report.update(hole_report)
    surface_report.update(
        {
            "post_trim_hole_count": int(post_trim_hole_report.get("hole_count", 0)),
            "post_trim_filled_cells": int(post_trim_hole_report.get("filled_cells", 0)),
        }
    )
    if surface_model == "plane_residual":
        plane, plane_report = fit_plane_to_heightfield(xy, z, point_weights, canvas_width, canvas_height)
        residual = z - plane
        residual_center = float(np.median(residual))
        residual_mad = max(robust_mad(residual), 1e-6)
        residual_limit = float(plane_residual_mad_multiplier) * 1.4826 * residual_mad
        clipped_residual = np.clip(
            residual,
            residual_center - residual_limit,
            residual_center + residual_limit,
        )
        clipped_fraction = float(np.mean(np.abs(clipped_residual - residual) > 1e-8))
        z = clipped_residual * float(height_exaggeration)
        surface_report.update(plane_report)
        surface_report.update(
            {
                "residual_center": residual_center,
                "residual_mad_all": float(residual_mad),
                "residual_limit": residual_limit,
                "residual_clipped_fraction": clipped_fraction,
                "plane_residual_mad_multiplier": float(plane_residual_mad_multiplier),
            }
        )

    if depth_detail_source != "none" and depth_detail_strength > 0:
        if depth_detail_source == "parallax_flow":
            detail_grid, detail_report = parallax_flow_depth_detail(
                stack_rgb,
                stack_valid,
                good,
                sigma=depth_detail_sigma,
                polarity=depth_detail_polarity,
                flow_scale=parallax_flow_scale,
            )
        elif depth_detail_source == "anchor_luma":
            detail_rgb = stack_rgb[0]
            detail_valid = stack_valid[0] & good
            detail_grid, detail_report = image_guided_depth_detail(
                detail_rgb,
                detail_valid,
                sigma=depth_detail_sigma,
                polarity=depth_detail_polarity,
            )
        else:
            detail_rgb = fused_rgb
            detail_valid = good
            detail_grid, detail_report = image_guided_depth_detail(
                detail_rgb,
                detail_valid,
                sigma=depth_detail_sigma,
                polarity=depth_detail_polarity,
            )
        detail_samples = detail_grid[y, x] * (float(depth_detail_strength) * base_depth_scale)
        z = z + detail_samples.astype(np.float32)
        surface_report.update(
            {
                "depth_detail_source": depth_detail_source,
                "depth_detail_strength": float(depth_detail_strength),
                "depth_detail_sigma": float(depth_detail_sigma),
                "depth_detail_polarity": float(depth_detail_polarity),
                "parallax_flow_scale": float(parallax_flow_scale),
                "depth_detail_added_p01_p99": [
                    float(np.percentile(detail_samples, 1.0)),
                    float(np.percentile(detail_samples, 99.0)),
                ],
            }
        )
        surface_report.update(detail_report)
    else:
        surface_report.update(
            {
                "depth_detail_source": "none",
                "depth_detail_strength": 0.0,
            }
        )

    xy_scale = float(max(canvas_width, canvas_height))
    z_center = float(np.median(z))
    z_scale = base_depth_scale if surface_model == "plane_residual" else max(abs(z_center), 1e-6)
    points = np.column_stack(
        [
            (xy[:, 0] - canvas_width * 0.5) / xy_scale,
            -(xy[:, 1] - canvas_height * 0.5) / xy_scale,
            (z - z_center) / z_scale,
        ]
    ).astype(np.float32)
    if len(points) > max_points:
        step = int(math.ceil(len(points) / max_points))
        points = points[::step]
        rgb = rgb[::step]

    total_effective_weight = float(weights[:, good].sum()) if np.any(good) else 0.0
    fusion_report = []
    for idx, surface in enumerate(surfaces):
        frame_weight = weights[idx]
        used = (frame_weight > 1e-8) & good
        frame_weight_sum = float(frame_weight[good].sum()) if np.any(good) else 0.0
        consistency_values = consistency[idx][used]
        fusion_report.append(
            {
                "local_index": surface["local_index"],
                "original_index": surface["original_index"],
                "method": "soft_blend",
                "input_cells": int(surface["valid"].sum()),
                "output_cells_with_weight": int(used.sum()),
                "effective_weight_fraction": (
                    frame_weight_sum / total_effective_weight if total_effective_weight > 0 else 0.0
                ),
                "median_consistency": (
                    float(np.median(consistency_values)) if len(consistency_values) > 0 else float("nan")
                ),
            }
        )
    fusion_report.append(surface_report)
    return points, rgb, fusion_report


def fuse_soft_blend_surfaces(
    predictions: dict,
    rgb_images: List[np.ndarray],
    original_rgbs: List[np.ndarray],
    original_to_model_transforms: List[np.ndarray],
    sand_masks: List[np.ndarray],
    h_to_canvas: Dict[int, np.ndarray],
    selected_indices: List[int],
    canvas_width: int,
    canvas_height: int,
    conf_percentile: float,
    max_points: int,
    depth_align: str,
    min_depth_overlap_cells: int,
    local_depth_sigma: float,
    depth_consistency: float,
    feather_power: float,
    min_feather: float,
    update_reference_depth: bool,
    surface_model: str,
    height_exaggeration: float,
    plane_residual_mad_multiplier: float,
    output_scale: int,
    texture_source: str,
    depth_detail_source: str,
    depth_detail_strength: float,
    depth_detail_sigma: float,
    depth_detail_polarity: float,
    parallax_flow_scale: float,
    min_support_frames: int,
    support_close: int,
    hole_fill_max_area: int,
    boundary_trim: int,
    smooth_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict]]:
    surfaces = warp_surface_maps(
        predictions,
        rgb_images,
        original_rgbs,
        original_to_model_transforms,
        sand_masks,
        h_to_canvas,
        selected_indices,
        canvas_width,
        canvas_height,
        conf_percentile,
        feather_power,
        min_feather,
        output_scale,
        texture_source,
    )
    scaled_canvas_width = canvas_width * max(1, int(output_scale))
    scaled_canvas_height = canvas_height * max(1, int(output_scale))
    aligned, depth_report = align_warped_surfaces(
        surfaces,
        depth_align=depth_align,
        min_overlap_cells=int(min_depth_overlap_cells * max(1, int(output_scale)) ** 2),
        local_depth_sigma=local_depth_sigma * max(1, int(output_scale)),
        update_reference_depth=update_reference_depth,
    )
    points, colors, fusion_report = blend_warped_surfaces(
        aligned,
        scaled_canvas_width,
        scaled_canvas_height,
        depth_consistency=depth_consistency,
        max_points=max_points,
        smooth_iterations=smooth_iterations,
        surface_model=surface_model,
        height_exaggeration=height_exaggeration,
        plane_residual_mad_multiplier=plane_residual_mad_multiplier,
        texture_source=texture_source,
        depth_detail_source=depth_detail_source,
        depth_detail_strength=depth_detail_strength,
        depth_detail_sigma=depth_detail_sigma,
        depth_detail_polarity=depth_detail_polarity,
        parallax_flow_scale=parallax_flow_scale,
        min_support_frames=min_support_frames,
        support_close=support_close,
        hole_fill_max_area=hole_fill_max_area,
        boundary_trim=boundary_trim,
    )
    return points, colors, depth_report, fusion_report


def fuse_point_samples(
    predictions: dict,
    rgb_images: List[np.ndarray],
    original_rgbs: List[np.ndarray],
    original_to_model_transforms: List[np.ndarray],
    sand_masks: List[np.ndarray],
    h_to_canvas: Dict[int, np.ndarray],
    selected_indices: List[int],
    canvas_width: int,
    canvas_height: int,
    conf_percentile: float,
    depth_mad_multiplier: float,
    max_points: int,
    depth_align: str,
    min_depth_overlap_cells: int,
    max_aligned_depth_mad: float,
    allow_depth_fallback_fill: bool,
    fusion_mode: str,
    reference_dilation: int,
    min_fill_cells: int,
    local_depth_sigma: float,
    depth_consistency: float,
    feather_power: float,
    min_feather: float,
    update_reference_depth: bool,
    surface_model: str,
    height_exaggeration: float,
    plane_residual_mad_multiplier: float,
    output_scale: int,
    texture_source: str,
    depth_detail_source: str,
    depth_detail_strength: float,
    depth_detail_sigma: float,
    depth_detail_polarity: float,
    parallax_flow_scale: float,
    min_support_frames: int,
    support_close: int,
    hole_fill_max_area: int,
    boundary_trim: int,
    smooth_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[dict], List[dict]]:
    if fusion_mode == "soft_blend":
        points, colors, depth_report, fusion_report = fuse_soft_blend_surfaces(
            predictions,
            rgb_images,
            original_rgbs,
            original_to_model_transforms,
            sand_masks,
            h_to_canvas,
            selected_indices,
            canvas_width,
            canvas_height,
            conf_percentile,
            max_points,
            depth_align,
            min_depth_overlap_cells,
            local_depth_sigma,
            depth_consistency,
            feather_power,
            min_feather,
            update_reference_depth,
            surface_model,
            height_exaggeration,
            plane_residual_mad_multiplier,
            output_scale,
            texture_source,
            depth_detail_source,
            depth_detail_strength,
            depth_detail_sigma,
            depth_detail_polarity,
            parallax_flow_scale,
            min_support_frames,
            support_close,
            hole_fill_max_area,
            boundary_trim,
            smooth_iterations,
        )
        depth_rejection_report = [
            {
                "local_index": item["local_index"],
                "original_index": item["original_index"],
                "input_cells": item.get("overlap_cells", 0),
                "kept_cells": item.get("overlap_cells", 0),
                "reject_reason": None,
                "method": "soft_blend_no_frame_discard",
            }
            for item in depth_report
        ]
        return points, colors, depth_report, depth_rejection_report, fusion_report

    raw_samples = collect_frame_samples(
        predictions,
        rgb_images,
        sand_masks,
        h_to_canvas,
        selected_indices,
        canvas_width,
        canvas_height,
        conf_percentile,
    )
    aligned_samples, depth_report = align_frame_depths(
        raw_samples,
        canvas_width,
        mode=depth_align,
        min_overlap_cells=min_depth_overlap_cells,
    )
    aligned_samples, depth_rejection_report = reject_bad_depth_aligned_frames(
        aligned_samples,
        depth_report,
        max_aligned_depth_mad=max_aligned_depth_mad,
        allow_fallback=allow_depth_fallback_fill,
    )
    if fusion_mode == "reference_fill":
        aligned_samples, fusion_report = apply_reference_fill(
        aligned_samples,
        canvas_width,
        canvas_height,
        dilation=reference_dilation,
        min_fill_cells=min_fill_cells,
    )
    else:
        fusion_report = [
            {
                "local_index": sample.local_index,
                "original_index": sample.original_index,
                "method": "median_overlap",
                "input_cells": int(len(sample.x)),
                "kept_cells": int(len(sample.x)),
            }
            for sample in aligned_samples
        ]
    points, colors = fuse_aligned_samples(
        aligned_samples,
        canvas_width,
        canvas_height,
        depth_mad_multiplier,
        max_points,
        smooth_iterations,
    )
    return points, colors, depth_report, depth_rejection_report, fusion_report


def save_pointcloud(points: np.ndarray, colors: np.ndarray, output_dir: Path) -> None:
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(output_dir / "pointcloud.ply")
    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    scene.export(output_dir / "scene.glb")


def save_alignment_preview(
    frames: List[FrameData],
    selected: List[int],
    h_to_canvas: Dict[int, np.ndarray],
    width: int,
    height: int,
    output_path: Path,
) -> None:
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    colors = [(255, 80, 80), (80, 255, 80), (80, 160, 255)]
    for rank, idx in enumerate(selected):
        if idx not in h_to_canvas:
            continue
        warped = cv2.warpPerspective(frames[idx].image, h_to_canvas[idx], (width, height))
        warped_mask = cv2.warpPerspective(
            frames[idx].mask.astype(np.uint8),
            h_to_canvas[idx],
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        accum[warped_mask] += warped[warped_mask].astype(np.float32)
        weight[warped_mask] += 1.0
        contour_mask = warped_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(accum, contours, -1, colors[rank % len(colors)], 2)
    preview = np.divide(accum, np.maximum(weight, 1.0), where=np.maximum(weight, 1.0) > 0)
    preview = np.clip(preview, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    image_folder = Path(args.image_folder)
    if not image_folder.is_absolute():
        image_folder = ROOT_DIR / image_folder
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_paths = list_images(image_folder)
    print(f"Found {len(image_paths)} images in {image_folder}")
    print("Building matching masks and homography graph...")
    edge_erosion = {"fast": 7, "balanced": 9, "accurate": 11}[args.mode]
    matching_frames = prepare_frames(image_paths, args.matching_width, edge_erosion=edge_erosion)
    edges, edge_report = build_alignment_graph(
        matching_frames,
        min_inliers=args.min_inliers,
        max_reproj_error=args.max_reproj_error,
        min_overlap=args.min_overlap,
    )
    fixed_keyframes = parse_keyframe_indices(args.keyframe_indices, len(matching_frames))
    selected, matching_transforms, distances = select_keyframes(
        matching_frames,
        edges,
        args.num_keyframes,
        anchor_index=args.anchor_index,
        fixed_indices=fixed_keyframes,
    )
    if len(selected) < 2:
        print("[WARN] Fewer than 2 reliable keyframes found; falling back to single-frame point cloud.")
    selected_paths = [image_paths[idx] for idx in selected]
    print("Selected keyframes:", [p.name for p in selected_paths])

    print("Refining selected-frame homographies at model resolution...")
    model_frames = prepare_frames(selected_paths, args.target_size, edge_erosion=7)
    model_edges, selected_edge_report = build_alignment_graph(
        model_frames,
        min_inliers=max(40, args.min_inliers // 3),
        max_reproj_error=args.max_reproj_error,
        min_overlap=max(0.15, args.min_overlap * 0.75),
    )
    local_selected = list(range(len(model_frames)))
    local_canvas_transforms, _, canvas_width, canvas_height, local_distances = build_canvas_transforms(
        local_selected,
        model_frames,
        model_edges,
    )
    h_to_canvas = {selected[local_idx]: h for local_idx, h in local_canvas_transforms.items()}
    selected_for_fusion = [selected[local_idx] for local_idx in local_selected]

    print("Loading OmniVGGT and running keyframe inference...")
    model = load_model(device, dtype, Path(args.checkpoint))
    print_cuda_memory("after model load")
    inputs, model_rgbs, model_masks, original_rgbs, original_to_model_transforms = load_keyframe_inputs(
        selected_paths,
        args.target_size,
        device,
        dtype,
    )
    predictions = run_inference(model, inputs, dtype, device)
    print_cuda_memory("after inference")

    print("Fusing depth into canonical no-ghost point cloud...")
    points, colors, depth_alignment_report, depth_rejection_report, fusion_report = fuse_point_samples(
        predictions,
        model_rgbs,
        original_rgbs,
        original_to_model_transforms,
        model_masks,
        h_to_canvas,
        selected_for_fusion,
        canvas_width,
        canvas_height,
        args.conf_percentile,
        args.depth_mad_multiplier,
        args.max_points,
        args.depth_align,
        args.min_depth_overlap_cells,
        args.max_aligned_depth_mad,
        args.allow_depth_fallback_fill,
        args.fusion_mode,
        args.reference_dilation,
        args.min_fill_cells,
        args.local_depth_sigma,
        args.depth_consistency,
        args.feather_power,
        args.min_feather,
        args.update_reference_depth,
        args.surface_model,
        args.height_exaggeration,
        args.plane_residual_mad_multiplier,
        args.output_scale,
        args.texture_source,
        args.depth_detail_source,
        args.depth_detail_strength,
        args.depth_detail_sigma,
        args.depth_detail_polarity,
        args.parallax_flow_scale,
        args.min_support_frames,
        args.support_close,
        args.hole_fill_max_area,
        args.boundary_trim,
        args.smooth_iterations,
    )
    save_pointcloud(points, colors, output_dir)
    save_alignment_preview(model_frames, local_selected, local_canvas_transforms, canvas_width, canvas_height, output_dir / "alignment_preview.jpg")

    anchor_original_index = min(selected, key=lambda idx: distances.get(idx, math.inf))
    keyframe_report = {
        "image_folder": str(image_folder),
        "selected_keyframes": [
            {
                "original_index": int(idx),
                "local_index": rank,
                "filename": image_paths[idx].name,
                "path": str(image_paths[idx]),
                "graph_distance_to_anchor": distances.get(idx),
            }
            for rank, idx in enumerate(selected)
        ],
        "anchor_original_index": int(anchor_original_index),
        "anchor_index_argument": args.anchor_index,
        "keyframe_indices_argument": args.keyframe_indices,
        "dtype": str(dtype),
        "target_size": args.target_size,
        "matching_width": args.matching_width,
        "canvas_size": [canvas_width, canvas_height],
        "output_canvas_size": [canvas_width * max(1, int(args.output_scale)), canvas_height * max(1, int(args.output_scale))],
        "point_count": int(len(points)),
        "depth_align": args.depth_align,
        "fusion_mode": args.fusion_mode,
        "reference_dilation": args.reference_dilation,
        "min_fill_cells": args.min_fill_cells,
        "local_depth_sigma": args.local_depth_sigma,
        "depth_consistency": args.depth_consistency,
        "feather_power": args.feather_power,
        "min_feather": args.min_feather,
        "update_reference_depth": args.update_reference_depth,
        "surface_model": args.surface_model,
        "height_exaggeration": args.height_exaggeration,
        "plane_residual_mad_multiplier": args.plane_residual_mad_multiplier,
        "output_scale": args.output_scale,
        "texture_source": args.texture_source,
        "depth_detail_source": args.depth_detail_source,
        "depth_detail_strength": args.depth_detail_strength,
        "depth_detail_sigma": args.depth_detail_sigma,
        "depth_detail_polarity": args.depth_detail_polarity,
        "parallax_flow_scale": args.parallax_flow_scale,
        "min_support_frames": args.min_support_frames,
        "support_close": args.support_close,
        "hole_fill_max_area": args.hole_fill_max_area,
        "boundary_trim": args.boundary_trim,
        "smooth_iterations": args.smooth_iterations,
        "max_aligned_depth_mad": args.max_aligned_depth_mad,
    }
    (output_dir / "keyframes.json").write_text(json.dumps(keyframe_report, indent=2), encoding="utf-8")
    alignment_report = {
        "all_pair_edges": edge_report,
        "selected_pair_edges_at_model_resolution": selected_edge_report,
        "depth_alignment": depth_alignment_report,
        "depth_rejection": depth_rejection_report,
        "fusion": fusion_report,
        "acceptance": {
            "min_inliers": args.min_inliers,
            "max_reproj_error": args.max_reproj_error,
            "min_overlap": args.min_overlap,
            "min_depth_overlap_cells": args.min_depth_overlap_cells,
            "max_aligned_depth_mad": args.max_aligned_depth_mad,
            "fusion_mode": args.fusion_mode,
            "local_depth_sigma": args.local_depth_sigma,
            "depth_consistency": args.depth_consistency,
            "feather_power": args.feather_power,
            "min_feather": args.min_feather,
            "update_reference_depth": args.update_reference_depth,
            "surface_model": args.surface_model,
            "height_exaggeration": args.height_exaggeration,
            "plane_residual_mad_multiplier": args.plane_residual_mad_multiplier,
            "output_scale": args.output_scale,
            "texture_source": args.texture_source,
            "depth_detail_source": args.depth_detail_source,
            "depth_detail_strength": args.depth_detail_strength,
            "depth_detail_sigma": args.depth_detail_sigma,
            "depth_detail_polarity": args.depth_detail_polarity,
            "parallax_flow_scale": args.parallax_flow_scale,
            "min_support_frames": args.min_support_frames,
            "support_close": args.support_close,
            "hole_fill_max_area": args.hole_fill_max_area,
            "boundary_trim": args.boundary_trim,
            "smooth_iterations": args.smooth_iterations,
        },
    }
    (output_dir / "alignment_report.json").write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    print(f"Saved point cloud: {output_dir / 'pointcloud.ply'}")
    print(f"Saved scene: {output_dir / 'scene.glb'}")
    print(f"Saved diagnostics: {output_dir / 'keyframes.json'}, {output_dir / 'alignment_report.json'}, {output_dir / 'alignment_preview.jpg'}")
    print_cuda_memory("final")


if __name__ == "__main__":
    main()
