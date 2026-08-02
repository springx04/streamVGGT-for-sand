"""Benchmark streaming strategies with mock or real inputs."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import time
import tracemalloc

import numpy as np
import torch
import typer
from PIL import Image
from rich.console import Console

from stream_omnivggt.backend import MockOmniBackend
from stream_omnivggt.config import load_stream_config
from stream_omnivggt.pipeline import StreamEngine, metrics_to_dict, summarize_metrics
from stream_omnivggt.types import InputPacket, StreamMetrics

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def main(
    dataset_dir: Path | None = typer.Option(None, "--dataset-dir", exists=False, file_okay=False),
    strategy: str = typer.Option("block_incremental", "--strategy", case_sensitive=False),
    config: Path | None = typer.Option(None, "--config"),
    output_json: Path = typer.Option(Path("stream_benchmark.json"), "--output-json"),
) -> None:
    """Run a benchmark and write JSON plus a markdown report."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if strategy not in {"full_rebuild", "block_incremental", "keyframe_hybrid"}:
        raise typer.BadParameter("strategy must be full_rebuild, block_incremental, or keyframe_hybrid")
    cfg = load_stream_config(config)
    cfg.benchmark.strategy = strategy
    engine = StreamEngine(MockOmniBackend(), cfg)
    packets = _load_packets(dataset_dir, cfg.benchmark.frame_count)
    tracemalloc.start()
    metrics: list[StreamMetrics] = []
    for packet in packets:
        metrics.append(engine.push(packet))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engine.flush()
    summary = summarize_metrics(metrics)
    result = {
        "strategy": strategy,
        "summary": summary,
        "metrics": [metrics_to_dict(item) for item in metrics],
        "peak_memory_bytes": peak,
        "cuda_transfer_log": _cuda_transfer_log(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = output_json.with_suffix(".md")
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    console.print(result["summary"])
    console.print(f"Wrote {output_json} and {report_path}")


def _load_packets(dataset_dir: Path | None, frame_count: int) -> list[InputPacket]:
    """Load image packets from a directory or synthesize a deterministic stream."""

    if dataset_dir is not None and dataset_dir.exists():
        paths = sorted(path for path in dataset_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if paths:
            packets: list[InputPacket] = []
            for idx, path in enumerate(paths[:frame_count]):
                packets.append(InputPacket(idx, float(idx) * 0.1, np.asarray(Image.open(path).convert("RGB")), None, None, None, {"source_path": str(path)}))
            return packets
    return [_synthetic_packet(idx) for idx in range(frame_count)]


def _synthetic_packet(idx: int) -> InputPacket:
    """Create a deterministic small-change synthetic packet."""

    rgb = np.zeros((56, 56, 3), dtype=np.uint8)
    rgb[..., 0] = 40
    rgb[..., 1] = 80
    rgb[..., 2] = 120
    if idx % 5 == 2:
        rgb[20:26, 20:26] = np.array([220, 30, 30], dtype=np.uint8)
    depth = np.ones((56, 56), dtype=np.float32)
    return InputPacket(idx, float(idx) * 0.1, rgb, depth, None, None, {})


def _cuda_transfer_log() -> dict[str, float] | None:
    """Measure pinned and unpinned host-to-device copies when CUDA is available."""

    if not torch.cuda.is_available():
        return None
    tensor = torch.zeros((4, 3, 56, 56), dtype=torch.float32)
    pinned = tensor.pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = tensor.to("cuda", non_blocking=False)
    torch.cuda.synchronize()
    unpinned_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    _ = pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    pinned_ms = (time.perf_counter() - t0) * 1000.0
    logger.info("CUDA transfer pinned=%.3fms unpinned=%.3fms", pinned_ms, unpinned_ms)
    return {"pinned_ms": pinned_ms, "unpinned_ms": unpinned_ms}


def _markdown_report(result: dict[str, object]) -> str:
    """Render a compact markdown benchmark report."""

    summary = result["summary"]
    lines = [
        "# Stream OmniVGGT Benchmark",
        "",
        f"- strategy: `{result['strategy']}`",
        f"- peak_memory_bytes: `{result['peak_memory_bytes']}`",
        "",
        "| metric | avg | p90 | p99 |",
        "|---|---:|---:|---:|",
    ]
    assert isinstance(summary, dict)
    for key, values in summary.items():
        assert isinstance(values, dict)
        lines.append(f"| {key} | {values['avg']:.4f} | {values['p90']:.4f} | {values['p99']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app()

