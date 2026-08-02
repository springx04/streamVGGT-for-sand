"""Adapter for the official OmniVGGT implementation with automatic fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import torch

from stream_omnivggt.backend.mock_backend import MockOmniBackend
from stream_omnivggt.config import OmniConfig
from stream_omnivggt.types import OmniPrediction

logger = logging.getLogger(__name__)


class OmniVGGTBackend:
    """PyTorch OmniVGGT backend that falls back to MockOmniBackend on failure."""

    def __init__(self, cfg: OmniConfig | None = None) -> None:
        """Initialize the real backend if model code and weights are available."""

        self.cfg = cfg or OmniConfig()
        self._fallback: MockOmniBackend | None = None
        self._model: torch.nn.Module | None = None
        self._device = self._resolve_device(self.cfg.device)
        self._dtype = self._resolve_dtype(self.cfg.dtype, self._device)
        if self.cfg.engine_mode != "pytorch":
            logger.warning("Engine mode %s is a stub in this version; using mock fallback.", self.cfg.engine_mode)
            self._fallback = MockOmniBackend()
            return
        try:
            self._load_model()
        except Exception as exc:  # noqa: BLE001 - fallback must catch import, weight, and OOM failures.
            logger.warning("Falling back to MockOmniBackend because OmniVGGT load failed: %s", exc)
            self._fallback = MockOmniBackend()
            self._model = None

    def warmup(self, bucket_shapes: list[tuple[int, int, int, int]]) -> None:
        """Warm up either the real backend or the fallback backend."""

        if self._fallback is not None:
            self._fallback.warmup(bucket_shapes)
            return
        if self._model is None:
            return
        for shape in bucket_shapes:
            s_count, channels, height, width = shape
            images = torch.zeros((1, s_count, channels, height, width), device=self._device, dtype=self._input_dtype())
            depth = torch.zeros((1, s_count, height, width, 1), device=self._device, dtype=self._input_dtype())
            mask = torch.zeros((1, s_count, height, width), device=self._device, dtype=self._input_dtype())
            extrinsics = torch.eye(4, device=self._device, dtype=torch.float32)[:3].repeat(1, s_count, 1, 1)
            intrinsics = torch.eye(3, device=self._device, dtype=torch.float32).repeat(1, s_count, 1, 1)
            with torch.inference_mode():
                self._model.inference(images, extrinsics, intrinsics, depth, mask, [], [])
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def run_window(self, batch: dict[str, Any]) -> OmniPrediction:
        """Run the official inference API or the deterministic fallback."""

        if self._fallback is not None:
            return self._fallback.run_window(batch)
        if self._model is None:
            raise RuntimeError("OmniVGGT model was not initialized and fallback is unavailable.")

        images = self._as_tensor(batch["images"], self._input_dtype())
        extrinsics = self._as_tensor(batch["extrinsics"], torch.float32)
        intrinsics = self._as_tensor(batch["intrinsics"], torch.float32)
        depth = self._as_tensor(batch["depth"], self._input_dtype())
        mask = self._as_tensor(batch["mask"], self._input_dtype())
        camera_gt_index = list(batch.get("camera_gt_index", []))
        depth_gt_index = list(batch.get("depth_gt_index", []))

        autocast_enabled = self._device.type == "cuda" and self._dtype in (torch.float16, torch.bfloat16)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=self._dtype, enabled=autocast_enabled):
            raw = self._model.inference(
                images=images,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                depth=depth,
                mask=mask,
                depth_gt_index=depth_gt_index,
                camera_gt_index=camera_gt_index,
            )
        return OmniPrediction(
            world_points=_squeeze_batch(raw["world_points"]),
            world_points_conf=_squeeze_batch(raw["world_points_conf"]),
            depth=_squeeze_batch(raw.get("depth")),
            depth_conf=_squeeze_batch(raw.get("depth_conf")),
            pose_enc=_squeeze_batch(raw.get("pose_enc")),
            extra={"backend": self.name(), "raw_keys": list(raw.keys())},
        )

    def name(self) -> str:
        """Return the active backend name."""

        if self._fallback is not None:
            return f"omnivggt-fallback-{self._fallback.name()}"
        return "omnivggt-pytorch"

    def _load_model(self) -> None:
        """Load official model code and safetensors checkpoint."""

        from safetensors.torch import load_file
        from omnivggt.models.omnivggt import OmniVGGT

        checkpoint_path = self._resolve_checkpoint_path()
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"OmniVGGT checkpoint not found: {checkpoint_path}")
        model = OmniVGGT(preload_patch_embed=self.cfg.preload_patch_embed)
        state_dict = load_file(str(checkpoint_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.to(device=self._device, dtype=self._dtype if self._device.type == "cuda" else torch.float32).eval()
        if self.cfg.compile_mode == "reduce-overhead":
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
        elif self.cfg.compile_mode != "none":
            logger.warning("Unknown compile_mode=%s; running without torch.compile.", self.cfg.compile_mode)
        self._model = model

    def _resolve_checkpoint_path(self) -> Path | None:
        """Find a checkpoint from explicit config or common local repo paths."""

        if self.cfg.checkpoint_path:
            return Path(self.cfg.checkpoint_path)
        candidates = []
        if self.cfg.repo_root:
            candidates.append(Path(self.cfg.repo_root) / "checkpoints" / "OmniVGGT.safetensors")
        candidates.append(Path.cwd() / "checkpoints" / "OmniVGGT.safetensors")
        candidates.append(Path(__file__).resolve().parents[2] / "checkpoints" / "OmniVGGT.safetensors")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def _resolve_device(self, device_arg: str) -> torch.device:
        """Resolve auto/cuda/cpu into a torch.device."""

        if device_arg == "auto":
            device_arg = "cuda" if torch.cuda.is_available() else "cpu"
        if device_arg == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            device_arg = "cpu"
        return torch.device(device_arg)

    def _resolve_dtype(self, dtype_arg: str, device: torch.device) -> torch.dtype:
        """Resolve dtype aliases while keeping CPU on float32."""

        aliases = {
            "auto": torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        dtype = aliases.get(dtype_arg)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_arg}")
        return torch.float32 if device.type == "cpu" else dtype

    def _input_dtype(self) -> torch.dtype:
        """Return the dtype used for image and depth tensors."""

        return self._dtype if self._device.type == "cuda" else torch.float32

    def _as_tensor(self, value: Any, dtype: torch.dtype) -> torch.Tensor:
        """Convert an input value to a non-blocking tensor on the backend device."""

        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.from_numpy(np.asarray(value))
        if self._device.type == "cuda" and tensor.device.type == "cpu":
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                logger.debug("pin_memory failed; continuing with pageable memory.", exc_info=True)
        return tensor.to(device=self._device, dtype=dtype, non_blocking=True)


def _squeeze_batch(value: Any) -> Any:
    """Remove a leading batch dimension from backend outputs when present."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value

