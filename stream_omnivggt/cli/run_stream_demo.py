"""Poll an image directory and feed new frames into StreamEngine."""

from __future__ import annotations

from pathlib import Path
import logging
import time

import numpy as np
import typer
from PIL import Image
from rich.console import Console

from stream_omnivggt.backend import MockOmniBackend, OmniVGGTBackend
from stream_omnivggt.config import StreamConfig, load_stream_config
from stream_omnivggt.pipeline import StreamEngine, metrics_to_dict
from stream_omnivggt.types import InputPacket

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def main(
    image_dir: Path = typer.Option(..., "--image-dir", exists=True, file_okay=False),
    camera_dir: Path | None = typer.Option(None, "--camera-dir", exists=False, file_okay=False),
    depth_dir: Path | None = typer.Option(None, "--depth-dir", exists=False, file_okay=False),
    poll_interval: float = typer.Option(0.5, "--poll-interval"),
    mock_backend: bool = typer.Option(False, "--mock-backend"),
    real_backend: bool = typer.Option(False, "--real-backend"),
    save_snapshot: Path | None = typer.Option(None, "--save-snapshot"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    """Run a directory-polling stream demo."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = load_stream_config(config)
    backend = MockOmniBackend() if mock_backend and not real_backend else OmniVGGTBackend(cfg.omni)
    engine = StreamEngine(backend, cfg)
    seen: set[Path] = set()
    frame_id = 0
    try:
        while True:
            image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            new_paths = [path for path in image_paths if path not in seen]
            for image_path in new_paths:
                packet = _load_packet(frame_id, image_path, camera_dir, depth_dir)
                metrics = engine.push(packet)
                console.print({"frame_id": frame_id, "backend": backend.name(), **metrics_to_dict(metrics)})
                seen.add(image_path)
                frame_id += 1
                if save_snapshot is not None and frame_id % 10 == 0:
                    engine.snapshot(str(save_snapshot))
            time.sleep(max(poll_interval, 0.0))
    except KeyboardInterrupt:
        logger.info("Stopping stream demo.")
    finally:
        if save_snapshot is not None:
            engine.snapshot(str(save_snapshot))
        engine.flush()


def _load_packet(frame_id: int, image_path: Path, camera_dir: Path | None, depth_dir: Path | None) -> InputPacket:
    """Load one RGB image plus optional depth/camera sidecars."""

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    depth = _load_depth(image_path, depth_dir)
    intrinsic, extrinsic = _load_camera(image_path, camera_dir)
    return InputPacket(
        frame_id=frame_id,
        timestamp=time.time(),
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        extrinsic_c2w=extrinsic,
        meta={"source_path": str(image_path)},
    )


def _load_depth(image_path: Path, depth_dir: Path | None) -> np.ndarray | None:
    """Load a depth sidecar by image stem if present."""

    if depth_dir is None:
        return None
    for suffix in (".npy", ".npz", ".png", ".tiff"):
        candidate = depth_dir / f"{image_path.stem}{suffix}"
        if not candidate.exists():
            continue
        if suffix == ".npy":
            return np.load(candidate).astype(np.float32)
        if suffix == ".npz":
            data = np.load(candidate)
            key = "depth" if "depth" in data else data.files[0]
            return np.asarray(data[key], dtype=np.float32)
        return np.asarray(Image.open(candidate), dtype=np.float32)
    return None


def _load_camera(image_path: Path, camera_dir: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load optional camera intrinsics/extrinsics from npz/npy/txt sidecars."""

    if camera_dir is None:
        return None, None
    for suffix in (".npz", ".npy", ".txt"):
        candidate = camera_dir / f"{image_path.stem}{suffix}"
        if not candidate.exists():
            continue
        if suffix == ".npz":
            data = np.load(candidate)
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float32) if "intrinsic" in data else None
            extrinsic = np.asarray(data["extrinsic_c2w"], dtype=np.float32) if "extrinsic_c2w" in data else None
            return intrinsic, extrinsic
        arr = np.load(candidate).astype(np.float32) if suffix == ".npy" else np.loadtxt(candidate, dtype=np.float32)
        if arr.shape == (4, 4):
            return None, arr
        if arr.shape == (3, 3):
            return arr, None
    return None, None


if __name__ == "__main__":
    app()

