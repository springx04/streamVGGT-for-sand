#!/usr/bin/env python3
"""Export the full OmniVGGT stream model for the C++ AOTInductor runner.

The generated package keeps the five outputs used by the C++ observer:
pose, depth, depth confidence, world points, and world-point confidence.
This is an optional deployment artifact; it does not alter the Python live
replay path or the original project entry points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def _patch_rope_for_fixed_export(height: int, width: int) -> None:
    """Remove a data-dependent scalar read from fixed-shape AOT export.

    The model's position grid is generated as [0..H/14-1] x [0..W/14-1].
    The original RoPE implementation computes the table length with
    ``int(positions.max())``.  AOT export rejects that data-dependent scalar,
    while a fixed table upper bound is numerically identical for valid input.
    """

    from omnivggt.layers.rope import RotaryPositionEmbedding2D

    max_position = max(int(height), int(width)) // 14 + 1

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0
        assert positions.ndim == 3 and positions.shape[-1] == 2
        feature_dim = tokens.size(-1) // 2
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    RotaryPositionEmbedding2D.forward = forward


class FullOmniWrapper(torch.nn.Module):
    """Expose the complete point-cloud inference contract as a flat tuple."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

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
            depth_gt_index=[],
            camera_gt_index=[],
        )
        aggregated_tokens_list = [tokens.to(images.dtype) for tokens in aggregated_tokens_list]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            pose_enc_list = self.model.camera_head(aggregated_tokens_list)
            depth_pred, depth_conf = self.model.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            pts3d, pts3d_conf = self.model.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
        return (
            pose_enc_list[-1].float(),
            depth_pred.float(),
            depth_conf.float(),
            pts3d.float(),
            pts3d_conf.float(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "OmniVGGT.safetensors")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "setc" / "artifacts" / "omnivggt_s1_518x518_aoti.pt2",
    )
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(root))

    if not torch.cuda.is_available():
        raise RuntimeError("AOTInductor CUDA export requires torch.cuda.is_available()")
    if args.height % 14 != 0 or args.width % 14 != 0:
        raise ValueError("height and width must be multiples of 14")

    _patch_rope_for_fixed_export(args.height, args.width)
    from omnivggt.models.omnivggt import OmniVGGT

    device = torch.device("cuda")
    model = OmniVGGT(preload_patch_embed=False)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"), strict=True)
    model.eval().to(device=device, dtype=torch.bfloat16)
    wrapper = FullOmniWrapper(model).eval()

    images = torch.zeros((1, 1, 3, args.height, args.width), device=device, dtype=torch.bfloat16)
    extrinsics = torch.zeros((1, 1, 3, 4), device=device, dtype=torch.bfloat16)
    intrinsics = torch.zeros((1, 1, 3, 3), device=device, dtype=torch.bfloat16)
    depth = torch.zeros((1, 1, args.height, args.width, 1), device=device, dtype=torch.bfloat16)
    mask = torch.zeros((1, 1, args.height, args.width), device=device, dtype=torch.bfloat16)

    print("Exporting fixed full-output AOTInductor graph...", flush=True)
    exported = torch.export.export(wrapper, (images, extrinsics, intrinsics, depth, mask), strict=False)
    print(f"Exported graph nodes: {len(list(exported.graph.nodes))}", flush=True)
    if args.export_only:
        return

    from torch._inductor import aoti_compile_and_package

    args.output.parent.mkdir(parents=True, exist_ok=True)
    package = aoti_compile_and_package(exported, package_path=str(args.output))
    print(f"Saved AOTInductor package: {package}", flush=True)


if __name__ == "__main__":
    main()
