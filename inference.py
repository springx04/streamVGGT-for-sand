"""
OmniVGGT Inference Script

This script performs 3D reconstruction from images using the OmniVGGT model.
It supports:
- Loading images and optional depth maps and camera parameters
- Running inference to predict depth and camera poses
- Visualizing results in 3D using viser
- Exporting results to GLB format
"""

import os
import argparse
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import viser
import viser.transforms as viser_tf
from safetensors.torch import load_file
from tqdm import tqdm

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from omnivggt.utils.misc import select_first_batch
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import (
    apply_sky_segmentation,
    get_world_points_from_depth,
    load_images_and_cameras,
    predictions_to_glb,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "OmniVGGT.safetensors"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }

    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    dtype = aliases[dtype_arg]
    if device.type == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU inference only supports float32 in this script. Use --device cuda for bf16/fp16.")
    return dtype


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    mib = 1024 ** 2
    print(
        f"[CUDA memory] {label}: "
        f"allocated={allocated / mib:.0f} MiB, reserved={reserved / mib:.0f} MiB, "
        f"peak={peak / mib:.0f} MiB, free={free_bytes / mib:.0f}/{total_bytes / mib:.0f} MiB"
    )


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 20.0,  # represents percentage (e.g., 20 means filter lowest 20%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    image_folder: Optional[str] = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        mask_black_bg (bool): Whether to mask out black background pixels.
        mask_white_bg (bool): Whether to mask out white background pixels.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if use_point_map and "world_points" in pred_dict:
        # Use precomputed world points if available
        world_points = pred_dict["world_points"]  # (S, H, W, 3)
        conf = pred_dict.get("world_points_conf", depth_conf)  # (S, H, W)
    else:
        # Compute world points by unprojecting depth map
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    
    # Apply black background mask if enabled
    if mask_black_bg:
        black_bg_mask = colors_flat.sum(axis=1) >= 16
        init_conf_mask = init_conf_mask & black_bg_mask
    
    # Apply white background mask if enabled
    if mask_white_bg:
        white_bg_mask = ~((colors_flat[:, 0] > 240) & (colors_flat[:, 1] > 240) & (colors_flat[:, 2] > 240))
        init_conf_mask = init_conf_mask & white_bg_mask
    
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        
        # Apply black background mask if enabled
        if mask_black_bg:
            black_bg_mask = colors_flat.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask
        
        # Apply white background mask if enabled
        if mask_white_bg:
            white_bg_mask = ~((colors_flat[:, 0] > 240) & (colors_flat[:, 1] > 240) & (colors_flat[:, 2] > 240))
            conf_mask = conf_mask & white_bg_mask

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

# Command-line argument parser
parser = argparse.ArgumentParser(
    description="OmniVGGT demo with viser for 3D visualization"
)

# Input data arguments
parser.add_argument("--image_folder",type=str,help="Path to folder containing images")
parser.add_argument("--depth_folder",type=str,default=None,help="Path to folder containing depth maps (.npy)")
parser.add_argument("--camera_folder",type=str,default=None,help="Path to folder containing camera files (.txt)")

# Processing options
parser.add_argument("--use_point_map",action="store_true",help="Use point map instead of depth-based points")
parser.add_argument("--mask_sky",action="store_true",help="Apply sky segmentation to filter out sky points")
parser.add_argument("--mask_black_bg",action="store_true",help="Mask out black background pixels")
parser.add_argument("--mask_white_bg",action="store_true",help="Mask out white background pixels")
parser.add_argument("--target_size",type=int,default=518,help="Target size for the images")
parser.add_argument("--max_images",type=int,default=3,help="Only load the first N sorted images; keep this at 1-3 on 8GB GPUs")
parser.add_argument("--disable_point_head",action="store_true",help="Disable dense point-map head to save VRAM")

# Runtime options
parser.add_argument("--checkpoint",type=str,default=str(DEFAULT_CHECKPOINT),help="Path to OmniVGGT safetensors checkpoint")
parser.add_argument("--device",type=str,default="cuda",choices=["cuda", "cpu", "auto"],help="Device for inference")
parser.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=["auto", "float32", "fp32", "bfloat16", "bf16", "float16", "fp16"],
    help="Model/input dtype. BF16 is recommended on RTX 50-series GPUs.",
)
parser.add_argument(
    "--preload_patch_embed",
    action="store_true",
    help="Preload DINOv2 patch embedding through torch.hub before loading the checkpoint",
)

# Visualization options
parser.add_argument("--background_mode",action="store_true",help="Run the viser server in background mode")
parser.add_argument("--port",type=int,default=8080,help="Port number for the viser server")
parser.add_argument("--conf_threshold",type=float,default=25.0,help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--no_visualize",action="store_true",help="Run inference and print output shapes without starting viser")

# Export options
parser.add_argument("--save_glb",action="store_true",help="Save the output as a GLB file")

def main():
    """
    Main function for the OmniVGGT demo with viser for 3D visualization.

    This function:
    1. Loads the OmniVGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser
    """
    args = parser.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.use_point_map and args.disable_point_head:
        print("[WARN] --use_point_map was set, but --disable_point_head removes world_points. Visualization will use depth.")
    if args.max_images is not None and args.max_images > 3:
        raise ValueError("This 8GB-GPU setup is configured for 1-3 input images. Use --max_images 1, 2, or 3.")

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.reset_peak_memory_stats(device)
        print_cuda_memory("startup")

    # Initialize and load OmniVGGT model
    print("Initializing and loading OmniVGGT model...")
    model = OmniVGGT(enable_point=not args.disable_point_head, preload_patch_embed=args.preload_patch_embed)
    # Load weights from local checkpoints
    state_dict = load_file(str(checkpoint_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=not args.disable_point_head)
    del state_dict

    if args.disable_point_head:
        bad_missing = [k for k in load_result.missing_keys if not k.startswith("point_head.")]
        bad_unexpected = [k for k in load_result.unexpected_keys if not k.startswith("point_head.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Unexpected checkpoint mismatch when disabling point head: "
                f"missing={bad_missing}, unexpected={bad_unexpected}"
            )
        print(f"Skipped {len(load_result.unexpected_keys)} point_head checkpoint tensors.")

    model.to(device=device, dtype=dtype).eval()
    print_cuda_memory("after model load")

    # Load input data
    print(f"Loading images from {args.image_folder}...")
    if args.camera_folder is not None:
        print(f"Loading cameras from {args.camera_folder}...")
    if args.depth_folder is not None:
        print(f"Loading depths from {args.depth_folder}...")

    images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
        load_images_and_cameras(
            args.image_folder,
            args.camera_folder,
            args.depth_folder,
            args.target_size,
            max_images=args.max_images,
        )

    # Prepare model inputs
    input_dtype = dtype if device.type == "cuda" else torch.float32
    inputs = {
        'images': images.to(device=device, dtype=input_dtype),
        'extrinsics': extrinsics.to(device=device, dtype=torch.float32),
        'intrinsics': intrinsics.to(device=device, dtype=torch.float32),
        'depth': depthmaps.to(device=device, dtype=input_dtype),
        'mask': masks.to(device=device, dtype=input_dtype),
        'depth_gt_index': depth_indices,
        'camera_gt_index': camera_indices
    }

    # Run inference
    print("Running inference...")
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        predictions = model.inference(**inputs)
    print_cuda_memory("after inference")

    # Convert pose encoding to camera matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].float(),
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Export to GLB if requested
    if args.save_glb:
        print("Exporting scene to GLB...")
        predictions_0 = select_first_batch(predictions)
        get_world_points_from_depth(predictions_0)
        glbscene = predictions_to_glb(
            predictions_0,
            conf_thres=0.0,
            filter_by_frames='All',
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=args.image_folder,
            prediction_mode="Predicted Depth",
        )
        glb_path = os.path.join(args.image_folder, 'scene.glb')
        glbscene.export(file_obj=glb_path)
        print(f"Saved GLB file to {glb_path}")

    if args.no_visualize:
        print("Inference complete. Output tensor shapes:")
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        print_cuda_memory("final")
        return

    # Convert predictions to numpy for visualization
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].detach().float().cpu().numpy().squeeze(0)

    # Print visualization mode
    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")
    
    if args.mask_black_bg:
        print("Black background masking enabled - will filter out black background points")
    
    if args.mask_white_bg:
        print("White background masking enabled - will filter out white background points")

    # Start visualization server
    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
