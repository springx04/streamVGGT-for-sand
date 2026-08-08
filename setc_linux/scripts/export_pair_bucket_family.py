#!/usr/bin/env python3
"""Export exact-shape two-frame observer TorchScript buckets for C++.

The Python live path keeps the ROI aspect ratio and changes the model tensor
shape per frame.  ``torch.jit.trace`` records the output spatial shape, so one
700x700 trace cannot safely be reused for a 700x672 input.  This helper loads
OmniVGGT once and emits the small family of buckets used by a replay dataset;
it does not change the normal Python runtime entry points.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file

from export_torchscript import OmniVGGTObserverDepthWrapper


def parse_shapes(value: str) -> list[tuple[int, int]]:
    """Parse comma-separated ``WIDTHxHEIGHT`` bucket names."""

    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        width_text, separator, height_text = text.partition("x")
        if not separator:
            raise argparse.ArgumentTypeError(f"invalid bucket shape: {item}")
        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0 or width % 14 or height % 14:
            raise argparse.ArgumentTypeError(f"bucket must be positive multiples of 14: {item}")
        shapes.append((width, height))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one bucket shape is required")
    return sorted(set(shapes))


def _identity_cameras(device: torch.device, num_images: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the same identity cameras as the live Python and C++ paths."""

    extrinsics = torch.eye(4, device=device, dtype=torch.float32)[:3]
    extrinsics = extrinsics.reshape(1, 1, 3, 4).repeat(1, num_images, 1, 1)
    intrinsics = torch.eye(3, device=device, dtype=torch.float32)
    intrinsics = intrinsics.reshape(1, 1, 3, 3).repeat(1, num_images, 1, 1)
    return extrinsics, intrinsics


def _artifact_name(output_dir: Path, width: int, height: int) -> Path:
    """Return the canonical name consumed by C++ ``--model-pair-dir``."""

    return output_dir / f"omnivggt_s2_{width}x{height}_bf16_unfrozen_torch270.pt"


def export_family(
    model_source: Path | None,
    checkpoint: Path,
    output_dir: Path,
    shapes: Iterable[tuple[int, int]],
) -> None:
    """Load one model and trace each exact dynamic ROI shape."""

    package_root = Path(__file__).resolve().parents[1]
    if model_source is None:
        bundled_source = package_root / "python"
        if (bundled_source / "omnivggt").is_dir():
            model_source = bundled_source
        elif (package_root / "omnivggt").is_dir():
            model_source = package_root
    if model_source is None:
        raise RuntimeError(
            "--model-source is required when exporting: provide a directory "
            "that contains the OmniVGGT Python package (omnivggt/)."
        )
    model_source = model_source.resolve()
    if not (model_source / "omnivggt").is_dir():
        raise FileNotFoundError(
            f"OmniVGGT Python package was not found under {model_source}; "
            "expected omnivggt/"
        )
    import sys

    sys.path.insert(0, str(model_source))
    from omnivggt.models.omnivggt import OmniVGGT

    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = OmniVGGT(preload_patch_embed=False)
    state_dict = load_file(str(checkpoint.resolve()))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device=device, dtype=dtype)
    output_dir.mkdir(parents=True, exist_ok=True)

    for width, height in shapes:
        output = _artifact_name(output_dir, width, height)
        images = torch.zeros((1, 2, 3, height, width), device=device, dtype=dtype)
        depth = torch.zeros((1, 2, height, width, 1), device=device, dtype=dtype)
        mask = torch.zeros((1, 2, height, width), device=device, dtype=dtype)
        extrinsics, intrinsics = _identity_cameras(device, 2)
        # The C++ pipeline consumes only pose, depth and depth confidence.
        # Exporting the full point head here makes LibTorch copy two additional
        # large output tensors back to CPU on every frame, defeating the
        # dynamic-input speedup even though the input ROI is smaller.
        wrapper = OmniVGGTObserverDepthWrapper(
            model=model,
            depth_indices=[],
            camera_indices=[],
        ).eval()
        print(f"Tracing pair bucket {width}x{height} -> {output}", flush=True)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
            traced = torch.jit.trace(
                wrapper,
                (images, extrinsics, intrinsics, depth, mask),
                strict=False,
                check_trace=False,
            ).eval()
        traced.save(str(output))
        manifest = {
            "format": "torchscript",
            "model": "OmniVGGT",
            "artifact": str(output.resolve()),
            "num_images": 2,
            "input_shape": [1, 2, 3, height, width],
            "height": height,
            "width": width,
            "dtype": "bfloat16",
            "autocast_dtype": "bfloat16",
            "camera_indices": [],
            "depth_indices": [],
            "inputs": ["images", "extrinsics", "intrinsics", "depth", "mask"],
            "outputs": ["pose_enc", "depth", "depth_conf"],
            "frozen": False,
            "preprocess": {
                "color": "RGB",
                "range": "[0,1]",
                "mean_std": "inside model",
                "size_contract": "fixed height/width multiples of 14",
            },
        }
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        del traced, wrapper, images, depth, mask, extrinsics, intrinsics
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    package_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--model-source",
        "--repo-root",
        dest="model_source",
        type=Path,
        default=None,
        help=(
            "Directory containing the Python omnivggt/ package. The Linux "
            "C++ package does not bundle the Python model source."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=package_root / "models" / "OmniVGGT.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "models",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("700x700,672x700,644x700,700x560,700x588,700x546,700x616,630x700,560x700"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export the BF16 pair buckets")
    export_family(args.model_source, args.checkpoint, args.output_dir, args.shapes)


if __name__ == "__main__":
    main()
