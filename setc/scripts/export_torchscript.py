#!/usr/bin/env python3
"""Export OmniVGGT to a fixed-shape TorchScript artifact for C++ inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file


def parse_indices(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise argparse.ArgumentTypeError("indices must be non-negative")
        out.append(index)
    return out


def validate_indices(indices: Iterable[int], num_images: int, name: str) -> list[int]:
    checked = sorted(set(int(i) for i in indices))
    for index in checked:
        if index >= num_images:
            raise ValueError(f"{name} index {index} is outside num_images={num_images}")
    return checked


class OmniVGGTTorchScriptWrapper(torch.nn.Module):
    """Return a stable tuple instead of the Python prediction dictionary."""

    def __init__(
        self,
        model: torch.nn.Module,
        depth_indices: list[int],
        camera_indices: list[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.depth_indices = depth_indices
        self.camera_indices = camera_indices

    def forward(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aggregated_tokens_list, patch_start_idx = self.model.aggregator.inference(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            depth_gt_index=self.depth_indices,
            camera_gt_index=self.camera_indices,
        )
        with torch.amp.autocast(
            "cuda",
            enabled=torch.is_autocast_enabled("cuda"),
            dtype=torch.get_autocast_dtype("cuda"),
        ):
            pose_enc_list = self.model.camera_head(aggregated_tokens_list)
            depth_pred, depth_conf = self.model.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            pts3d, pts3d_conf = self.model.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
        return (
            pose_enc_list[-1].float(),
            depth_pred.float(),
            depth_conf.float(),
            pts3d.float(),
            pts3d_conf.float(),
        )


class OmniVGGTObserverDepthWrapper(torch.nn.Module):
    """C++ observer path: keep camera/depth heads and omit the point head."""

    def __init__(
        self,
        model: torch.nn.Module,
        depth_indices: list[int],
        camera_indices: list[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.depth_indices = depth_indices
        self.camera_indices = camera_indices

    def forward(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aggregated_tokens_list, patch_start_idx = self.model.aggregator.inference(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth=depth,
            mask=mask,
            depth_gt_index=self.depth_indices,
            camera_gt_index=self.camera_indices,
        )
        # Keep the observer-only graph under the same head autocast scope as
        # OmniVGGT.inference().  The old observer wrapper left this scope out;
        # tracing still succeeded because the outer export context was active,
        # but the saved graph did not carry the same head dispatch behavior as
        # the Python runtime.
        with torch.amp.autocast(
            "cuda",
            enabled=torch.is_autocast_enabled("cuda"),
            dtype=torch.get_autocast_dtype("cuda"),
        ):
            pose_enc_list = self.model.camera_head(aggregated_tokens_list)
            depth_pred, depth_conf = self.model.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
        return (
            pose_enc_list[-1].float(),
            depth_pred.float(),
            depth_conf.float(),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "OmniVGGT.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "setc" / "artifacts" / "omnivggt_s2_518x518.pt",
    )
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Tracing dtype. Use float32 first; test fp16/bf16 per device.",
    )
    parser.add_argument(
        "--autocast-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help=(
            "Enable CUDA autocast while tracing. Keep --dtype=float32 for "
            "mixed-precision graphs that match Python eager inference."
        ),
    )
    parser.add_argument(
        "--observer-depth-only",
        action="store_true",
        help="Export the C++ observer graph without world-point heads.",
    )
    parser.add_argument("--camera-indices", type=parse_indices, default=[])
    parser.add_argument("--depth-indices", type=parse_indices, default=[])
    parser.add_argument(
        "--optimize-for-inference",
        action="store_true",
        help="Run torch.jit.optimize_for_inference before saving.",
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help=(
            "Keep traced submodule boundaries. This is useful for LibTorch "
            "when a fully inlined frozen transformer selects a slow CUDA path."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if args.height % 14 != 0 or args.width % 14 != 0:
        raise ValueError("--height and --width must be multiples of 14")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but torch.cuda.is_available() is False")

    camera_indices = validate_indices(args.camera_indices, args.num_images, "camera")
    depth_indices = validate_indices(args.depth_indices, args.num_images, "depth")

    repo_root = args.repo_root.resolve()
    import sys

    sys.path.insert(0, str(repo_root))

    from omnivggt.models.omnivggt import OmniVGGT

    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU tracing should use --dtype float32")

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    print(f"Loading model from {checkpoint}")
    model = OmniVGGT(preload_patch_embed=False)
    state_dict = load_file(str(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device=device, dtype=dtype)

    wrapper_type = OmniVGGTObserverDepthWrapper if args.observer_depth_only else OmniVGGTTorchScriptWrapper
    wrapper = wrapper_type(
        model=model,
        depth_indices=depth_indices,
        camera_indices=camera_indices,
    ).eval()

    shape = (1, args.num_images, 3, args.height, args.width)
    images = torch.zeros(shape, device=device, dtype=dtype)
    # Match the live Python backend: images/depth/mask use the selected model
    # dtype, while camera tensors stay FP32 even when the model runs in BF16.
    # These examples are part of the traced graph's calibration path.  The
    # live Python stream supplies identity extrinsics/intrinsics, and the C++
    # runtime does the same; tracing with all-zero cameras silently bakes a
    # different camera branch into the artifact.
    extrinsics = torch.eye(4, device=device, dtype=torch.float32)[:3].reshape(1, 1, 3, 4).repeat(1, args.num_images, 1, 1)
    intrinsics = torch.eye(3, device=device, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, args.num_images, 1, 1)
    depth = torch.zeros((1, args.num_images, args.height, args.width, 1), device=device, dtype=dtype)
    mask = torch.zeros((1, args.num_images, args.height, args.width), device=device, dtype=dtype)

    print(
        "Tracing fixed signature: "
        f"images={tuple(images.shape)}, camera_indices={camera_indices}, depth_indices={depth_indices}"
    )
    autocast_map = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    autocast_dtype = autocast_map[args.autocast_dtype]
    if autocast_dtype is None and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        autocast_dtype = dtype
    autocast_enabled = device.type == "cuda" and autocast_dtype is not None
    # The Python backend uses this same context.  OmniVGGT intentionally keeps
    # some depth-head activations in FP32; tracing without autocast makes the
    # BF16 graph fail at conv_transpose2d even though normal Python inference
    # succeeds.
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=autocast_dtype or torch.float16, enabled=autocast_enabled
    ):
        traced = torch.jit.trace(
            wrapper,
            (images, extrinsics, intrinsics, depth, mask),
            strict=False,
            check_trace=False,
        )
        traced = traced.eval()
        if not args.no_freeze:
            traced = torch.jit.freeze(traced)
        if args.optimize_for_inference:
            traced = torch.jit.optimize_for_inference(traced)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))

    manifest = {
        "format": "torchscript",
        "model": "OmniVGGT",
        "checkpoint": str(checkpoint),
        "artifact": str(args.output.resolve()),
        "num_images": args.num_images,
        "input_shape": [1, args.num_images, 3, args.height, args.width],
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "autocast_dtype": args.autocast_dtype,
        "camera_indices": camera_indices,
        "depth_indices": depth_indices,
        "inputs": ["images", "extrinsics", "intrinsics", "depth", "mask"],
        "outputs": (
            ["pose_enc", "depth", "depth_conf"]
            if args.observer_depth_only
            else [
                "pose_enc",
                "depth",
                "depth_conf",
                "world_points",
                "world_points_conf",
            ]
        ),
        "observer_depth_only": args.observer_depth_only,
        "frozen": not args.no_freeze,
        "preprocess": {
            "color": "RGB",
            "range": "[0,1]",
            "mean_std": "inside model",
            "size_contract": "fixed height/width multiples of 14",
        },
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved TorchScript model: {args.output}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
