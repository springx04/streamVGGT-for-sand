# OmniVGGT Project Context

Last updated: 2026-06-15

## First-read rule

Read this file before inspecting the full repository. It records the stable project map, environment notes, and low-VRAM inference decisions so future edits do not need a full rescan.

## Project shape

- `inference.py`: CLI demo/inference entry. Loads `OmniVGGT`, reads image/camera/depth folders via `visual_util.py`, runs `model.inference`, converts pose encodings, optionally exports GLB, then starts a `viser` viewer.
- `visual_util.py`: image/camera/depth loading, resize/crop logic, sky/background masks, GLB conversion helpers.
- `omnivggt/models/omnivggt.py`: top-level model. Contains `ZeroAggregator`, `CameraHead`, `DPTHead` for depth and point prediction.
- `omnivggt/models/aggregator.py`: base alternating-attention transformer and DINOv2 patch embedding construction.
- `omnivggt/models/omnivggt_aggregator.py`: OmniVGGT aggregator with camera/depth auxiliary-token injection and inference path.
- `omnivggt/heads/`: camera and dense DPT heads.
- `omnivggt/layers/`: ViT, attention, MLP, RoPE, patch embedding layers.
- `configs/`: training/test configs. Not required for CLI inference.
- `checkpoints/`: local model files. `OmniVGGT.safetensors` is present.
- `example/`: sample image/camera/depth folders.
- `reconstruct_sand_pointcloud.py`: data2-oriented no-ghost point-cloud CLI. It selects 2-3 keyframes from a rotating aperture sequence, estimates homographies, runs OmniVGGT only on selected frames, warps each depth map into a canonical anchor canvas, locks secondary frames to the anchor depth reference, soft-blends all frames into one height field, then outputs plane-residual Z so the near-planar sand surface stays continuous. It has support regularization, internal-hole filling, post-trim hole filling, two-frame support filtering, and boundary trimming for thin continuous surfaces. It also has optional training-free detail modes: `parallax_flow` from multi-view residual flow, and photometric luminance relief for visualization only.

## Current local environment findings

- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`, total 8151 MiB, compute capability 12.0.
- At inspection time, available VRAM was about 6264 MiB because other processes were using about 1636 MiB.
- `C:\Users\30738\anaconda3\envs\omnivggt` now has `torch 2.7.0+cu128`; CUDA is available.
- `C:\Users\30738\anaconda3\envs\vggt` has `torch 2.7.0+cu128`, CUDA available, and the runtime packages needed by this repo.
- `C:\Users\30738\miniconda3\envs\cu128` has CUDA PyTorch but is missing several OmniVGGT dependencies.
- Weight file: `checkpoints/OmniVGGT.safetensors`, 1505 tensors, 1,217,494,552 parameters, all `torch.float32`; raw FP32 parameter storage is about 4.54 GiB.

## Inference memory risks

- Loading FP32 weights on an 8GB GPU leaves too little room for activations; BF16 is the working path.
- BF16/FP16 model weights reduce raw parameter memory to about 2.27 GiB. RTX 5060 reports BF16 support.
- FP8 is not a practical quick fix in this PyTorch model: float8 dtypes exist, but Linear/Conv/LayerNorm/SDPA/DPT coverage and calibration are not drop-in. Prefer BF16 first.
- `inference.py` now defaults to CUDA + BF16 and loads at most 3 sorted images via `--max_images 3`.
- The dense point head is optional for visualization because depth + camera can be unprojected into points. Disabling `point_head` can save parameters and compute, but the 3-image 518px smoke test also passes with the point head enabled.
- `OmniVGGT.forward/inference` now preserves the caller's autocast state for heads. This avoids the BF16/FP32 LayerNorm dtype mismatch caused by forcing autocast off.
- Aggregator inference originally stores all 24 layer intermediates in `aggregated_tokens_list`, while the DPT heads only use layers `[4, 11, 17, 23]` and the camera head uses the final layer. Keeping only those inference intermediates can reduce activation retention.

## Recommended 8GB GPU path

1. Use `C:\Users\30738\anaconda3\envs\omnivggt\python.exe`.
2. Run inference with BF16, which is now the `inference.py` default.
3. Keep `--max_images` in the 1-3 range. The script now rejects values above 3 for this 8GB setup.
4. `--target_size 518` is feasible for 3 images in the included smoke test. Drop to `336` only if other desktop processes consume much more VRAM.
5. Use `--no_visualize` for memory smoke tests; visualization converts outputs to CPU/NumPy and starts a server.
6. Avoid FP8 unless doing a separate quantization/export path with calibration.

## Verified smoke tests

- 1 image, `target_size=336`, BF16, point head disabled: passed; peak allocated VRAM about 2512 MiB.
- 3 images, `target_size=336`, BF16, point head disabled: passed; peak allocated VRAM about 2691 MiB.
- 3 images, `target_size=518`, BF16, point head disabled: passed; peak allocated VRAM about 3240 MiB.
- 3 images, `target_size=518`, BF16, full model with point head enabled: passed; peak allocated VRAM about 3307 MiB.

## Useful commands

CUDA smoke test:

```powershell
& 'C:\Users\30738\anaconda3\envs\omnivggt\python.exe' 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\inference.py' `
  --image_folder 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\example\meetingroom\images' `
  --max_images 3 --no_visualize
```

Interactive viewer:

```powershell
& 'C:\Users\30738\anaconda3\envs\omnivggt\python.exe' 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\inference.py' `
  --image_folder '<your-image-folder>' --max_images 3
```

data2 no-ghost point-cloud reconstruction:

```powershell
& 'C:\Users\30738\anaconda3\envs\omnivggt\python.exe' 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\reconstruct_sand_pointcloud.py' `
  --image_folder 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\data2' `
  --output_dir 'C:\Users\30738\Desktop\project\3dre\VGGT\OmniVGGT\outputs\data2_pointcloud' `
  --num_keyframes 3 --mode balanced
```

Verified output for the default data2 command:

- Selected keyframes: `Image_20260423192541434.jpg`, `Image_20260423192801758.jpg`, `Image_20260423192929499.jpg`.
- Outputs: `pointcloud.ply`, `scene.glb`, `keyframes.json`, `alignment_report.json`, `alignment_preview.jpg`.
- Current default point count: 84,026.
- Peak allocated VRAM: about 3397 MiB.
- Vertical ghosting fix: default `--depth_align affine` estimates per-keyframe `z_aligned = scale * z + offset` from overlapping canonical cells before fusion. For the verified run, overlap cells were about 61.7k per non-reference frame, and median absolute depth residual dropped from about 0.022/0.027 to about 0.0067/0.0052. Normalized point-cloud Z range narrowed from about `[-0.096, 0.147]` to `[-0.018, 0.040]`.
- Current seam/ghost/bending fix after visual inspection: default `--fusion_mode soft_blend --surface_model plane_residual` no longer discards secondary frames and no longer trusts a multi-frame 3D pose fusion. It uses one selected anchor image as the canonical 2D coordinate system, keeps `--update_reference_depth` off by default so secondary frames cannot bend the anchor surface, and emits only one point per canonical XY cell.
- The default output Z is now a robust fitted-plane residual with `--plane_residual_mad_multiplier 3.0`, which clamps high residual edge artifacts while preserving continuous near-planar relief. Use `--surface_model depth` only for diagnosis; it can reintroduce apparent global bending.
- Latest verified run selected `Image_20260423192929499.jpg`, `Image_20260423192541434.jpg`, `Image_20260423192801758.jpg`; anchor original index was `7`; output point count was 94,171; duplicate canonical XY points after rounding were 0; effective fusion weight fractions were about 40.4%, 27.5%, and 32.1%, so the result is not a single-frame fallback.
- For the same run, secondary-frame local median absolute depth residuals dropped from about 0.0317 to 0.00435 and from about 0.0126 to 0.00238 before fusion. Robust plane residual MAD was about 0.00438, residual-clipped fraction about 5.8%, and normalized output Z percentiles were approximately `[-0.02497, -0.01615, -0.01100, 0.0, 0.02497, 0.02497, 0.02497]` at `[0,1,5,50,95,99,100]`. Peak allocated VRAM remained about 3397 MiB.
- Optional anchor override: pass `--anchor_index 0` to force the first file in sorted order as the anchor. Default `-1` auto-selects the best-connected anchor, which is usually safer for low-texture sand.
- Depth-detail note: BF16/FP32 precision is not the main limiter for sand-grain detail. `fp32 + target_size=700` ran successfully but peak allocated VRAM rose to about 6601 MiB and free VRAM reported 0 MiB; its plane residual MAD was only slightly lower than BF16. Use FP32 only when memory is available and numerical comparison is needed.
- Accuracy-first output with `--anchor_index 3 --target_size 700 --depth_detail_source none` is at `outputs/data2_pointcloud_anchor3_bf16_700_accurate` and `outputs/data2_pointcloud_anchor3_fp32_700_accurate`. BF16 700 accurate used three images with effective weights about 42.2%, 28.6%, 29.1%, filled 275 internal hole cells, removed 52,356 boundary cells, and output 515,634 points. This is the safer geometry output.
- Training-free geometric detail output is `outputs/data2_pointcloud_anchor3_bf16_700_parallax`, generated with `--depth_detail_source parallax_flow --depth_detail_strength 0.006 --depth_detail_sigma 7 --parallax_flow_scale 0.5`. It uses residual optical flow after homography alignment, not luminance height. It still used three images with the same weights as the accurate output and added normalized Z detail of about `[-0.0028, 0.0028]` P01-P99. This is relative micro-relief; scale/sign are not physically calibrated.
- Photometric outputs such as `outputs/data2_pointcloud_anchor3_bf16_700_detail` are not recommended for accuracy. They can make image texture visible in Z but can create a thick-looking layer and edge relief because luminance is not guaranteed to equal depth.
- BF16 `target_size=1008` ran at `outputs/data2_pointcloud_anchor3_bf16_1008_accurate` with peak allocated VRAM about 6369 MiB and free VRAM reported 0 MiB. It is a useful high-resolution diagnostic but still did not recover strong sand-grain depth detail directly from OmniVGGT.
- Latest hole/edge fix: keyframe selection is now scored by coverage, compactness, and hole penalties instead of greedy area alone. With `--anchor_index 3`, it selects `Image_20260423192657308.jpg`, `Image_20260423192801758.jpg`, and `Image_20260423192942731.jpg` instead of the earlier `...92517173.jpg` third frame. Z outlier handling in soft blending now clips height instead of deleting points, preventing artificial holes.
- Current recommended accurate output is `outputs/data2_pointcloud_anchor3_bf16_700_v5_support2`, generated with `--anchor_index 3 --target_size 700 --depth_detail_source none --min_support_frames 2 --support_close 81 --boundary_trim 36`. It outputs 440,911 points, has no detected internal holes, uses three images with effective weights about 45.7%, 31.8%, 22.5%, adds 20,098 support-regularized cells, and removes 95,798 unreliable boundary cells. This is the best current geometry-first output.
- Current optional geometric-detail output is `outputs/data2_pointcloud_anchor3_bf16_700_v5_parallax`, generated with the same support settings plus `--depth_detail_source parallax_flow --depth_detail_strength 0.006 --depth_detail_sigma 7 --parallax_flow_scale 0.5`. It keeps 440,911 points and adds residual-flow detail of about `[-0.00283, 0.00283]` normalized Z P01-P99.
