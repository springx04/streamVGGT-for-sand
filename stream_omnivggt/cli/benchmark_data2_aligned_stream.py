"""Benchmark data2 with homography-aligned streaming fusion."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import statistics
import logging

import numpy as np
import typer
from PIL import Image
from rich.console import Console

from stream_omnivggt.backend import MockOmniBackend, OmniVGGTBackend
from stream_omnivggt.config import StreamConfig, load_stream_config
from stream_omnivggt.pipeline.aligned_canvas_stream import AlignedCanvasStream, save_ply
from stream_omnivggt.types import InputPacket

app = typer.Typer(add_completion=False)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def main(
    image_dir: Path = typer.Option(Path("data2"), "--image-dir", exists=True, file_okay=False),
    output_dir: Path = typer.Option(Path("stream_omnivggt_outputs/data2_aligned_stream"), "--output-dir"),
    config: Path | None = typer.Option(None, "--config"),
    mock_backend: bool = typer.Option(False, "--mock-backend"),
    target_width: int = typer.Option(280, "--target-width"),
    target_size: int = typer.Option(196, "--target-size"),
    save_debug: bool = typer.Option(True, "--save-debug/--no-save-debug"),
    static_reference_ply: Path | None = typer.Option(None, "--static-reference-ply", exists=False, dir_okay=False),
) -> None:
    """Run aligned-canvas streaming on data2 and export timing plus PLY."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = load_stream_config(config)
    cfg.omni.target_width = target_width
    cfg.omni.target_size = target_size
    cfg.omni.patch_multiple = 14
    cfg.omni.warmup_buckets = []
    cfg.omni.device = "cuda"
    cfg.omni.dtype = "bf16"
    cfg.omni.preload_patch_embed = False
    cfg.change.flow_mode = "none"
    cfg.fuse.min_conf = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise typer.BadParameter(f"No images found in {image_dir}")

    backend_start = perf_counter()
    backend = MockOmniBackend() if mock_backend else OmniVGGTBackend(cfg.omni)
    backend_load_ms = (perf_counter() - backend_start) * 1000.0
    stream = AlignedCanvasStream(backend, cfg)

    rows = []
    first_pointcloud_ms = None
    stream_start = perf_counter()
    for idx, path in enumerate(paths):
        read_start = perf_counter()
        rgb = np.asarray(Image.open(path).convert("RGB"))
        read_ms = (perf_counter() - read_start) * 1000.0
        packet = InputPacket(idx, float(idx), rgb, None, None, None, {"source_path": str(path)})
        metrics = stream.push(packet, read_ms=read_ms)
        debug_paths = stream.save_last_debug(output_dir / "debug", path.name) if save_debug else {}
        if first_pointcloud_ms is None and metrics.point_count > 0:
            first_pointcloud_ms = (perf_counter() - stream_start) * 1000.0
        row = metrics.to_dict()
        row["image"] = path.name
        row["debug_overlay"] = debug_paths.get("overlay")
        row["debug_change_mask"] = debug_paths.get("change_mask")
        rows.append(row)

    points, colors = stream.export_pointcloud()
    stream_ply = output_dir / "pointcloud_final.ply"
    save_ply(str(stream_ply), points, colors)
    summary = _summary(backend.name(), backend_load_ms, first_pointcloud_ms, rows, len(points), cfg)
    comparison = None
    if static_reference_ply is not None:
        comparison = _compare_static_reference(stream_ply, static_reference_ply, output_dir / "comparison")
        summary["static_reference_status"] = comparison["status"]
        summary["static_reference_z_abs_p95_ratio"] = comparison["z_abs_p95_ratio"]
    (output_dir / "timings.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2), encoding="utf-8")
    (output_dir / "timings.md").write_text(_markdown(summary, rows), encoding="utf-8")
    if comparison is not None:
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        (output_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    console.print(summary)
    console.print(f"Wrote {output_dir / 'timings.json'}")
    console.print(f"Wrote {stream_ply}")
    if save_debug:
        console.print(f"Wrote debug masks under {output_dir / 'debug'}")
    if comparison is not None:
        console.print(f"Wrote reference comparison under {output_dir / 'comparison'}")


def _summary(
    backend_name: str,
    backend_load_ms: float,
    first_pointcloud_ms: float | None,
    rows: list[dict[str, object]],
    point_count: int,
    cfg: StreamConfig,
) -> dict[str, object]:
    """Summarize aligned stream timings."""

    subsequent = rows[1:]

    def mean(key: str, data: list[dict[str, object]]) -> float | None:
        return float(statistics.fmean(float(row[key]) for row in data)) if data else None

    def p90(key: str, data: list[dict[str, object]]) -> float | None:
        if not data:
            return None
        values = sorted(float(row[key]) for row in data)
        return values[int(round((len(values) - 1) * 0.9))]

    return {
        "backend": backend_name,
        "backend_load_ms": backend_load_ms,
        "image_count": len(rows),
        "bucket_target_width": cfg.omni.target_width,
        "bucket_target_size": cfg.omni.target_size,
        "first_input_to_pointcloud_ms_excluding_backend_load": first_pointcloud_ms,
        "first_input_to_pointcloud_ms_including_backend_load": backend_load_ms + (first_pointcloud_ms or 0.0),
        "first_frame_total_ms": rows[0]["total_ms"] if rows else None,
        "first_frame_model_ms": rows[0]["model_ms"] if rows else None,
        "subsequent_avg_total_ms": mean("total_ms", subsequent),
        "subsequent_p90_total_ms": p90("total_ms", subsequent),
        "subsequent_avg_align2d_ms": mean("align2d_ms", subsequent),
        "subsequent_avg_model_ms": mean("model_ms", subsequent),
        "subsequent_avg_fuse_ms": mean("fuse_ms", subsequent),
        "subsequent_avg_changed_ratio": mean("changed_ratio", subsequent),
        "subsequent_avg_photometric_changed_ratio": mean("photometric_changed_ratio", subsequent),
        "subsequent_avg_support_changed_ratio": mean("support_changed_ratio", subsequent),
        "final_point_count": point_count,
    }


def _markdown(summary: dict[str, object], rows: list[dict[str, object]]) -> str:
    """Render a markdown timing report."""

    lines = ["# data2 aligned stream timing", "", "## Summary", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Per-frame",
        "",
        "| frame | image | read_ms | total_ms | align2d_ms | model_ms | depth_align_ms | fuse_ms | changed_ratio | photo_ratio | support_ratio | roi | fused_pixels | points | homography_inliers | fallback | overlay |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['frame_id']} | {row['image']} | {float(row['read_ms']):.2f} | {float(row['total_ms']):.2f} | "
            f"{float(row['align2d_ms']):.2f} | {float(row['model_ms']):.2f} | {float(row['depth_align_ms']):.2f} | "
            f"{float(row['fuse_ms']):.2f} | {float(row['changed_ratio']):.4f} | "
            f"{float(row['photometric_changed_ratio']):.4f} | {float(row['support_changed_ratio']):.4f} | "
            f"{row['roi_width']}x{row['roi_height']} | {row['fused_pixels']} | {row['point_count']} | {row['homography_inliers']} | {row['fallback_reason']} | "
            f"{row.get('debug_overlay')} |"
        )
    return "\n".join(lines) + "\n"


def _compare_static_reference(stream_ply: Path, static_ply: Path, output_dir: Path) -> dict[str, object]:
    """Compare stream output with a static reference PLY and save projections."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stream_pts, stream_cols = _load_ply(stream_ply)
    static_pts, static_cols = _load_ply(static_ply)
    stream_stats = _point_stats(stream_pts)
    static_stats = _point_stats(static_pts)
    z_ratio = float(stream_stats["z_abs_p95"] / max(float(static_stats["z_abs_p95"]), 1e-9))
    mad_ratio = float(stream_stats["z_mad"] / max(float(static_stats["z_mad"]), 1e-9))
    status = "ok"
    warnings: list[str] = []
    if z_ratio > 5.0:
        status = "warning"
        warnings.append("stream z_abs_p95 is more than 5x the static reference")
    if mad_ratio > 5.0:
        status = "warning"
        warnings.append("stream z_mad is more than 5x the static reference")

    projection_paths = {
        "stream_xy_height": str(_render_projection(output_dir / "stream_xy_height.png", stream_pts, stream_cols, (0, 1), True)),
        "static_xy_height": str(_render_projection(output_dir / "static_xy_height.png", static_pts, static_cols, (0, 1), True)),
        "stream_xz_side": str(_render_projection(output_dir / "stream_xz_side.png", stream_pts, stream_cols, (0, 2), True)),
        "static_xz_side": str(_render_projection(output_dir / "static_xz_side.png", static_pts, static_cols, (0, 2), True)),
    }
    return {
        "status": status,
        "warnings": warnings,
        "stream_ply": str(stream_ply),
        "static_reference_ply": str(static_ply),
        "stream": stream_stats,
        "static_reference": static_stats,
        "z_abs_p95_ratio": z_ratio,
        "z_mad_ratio": mad_ratio,
        "projection_paths": projection_paths,
    }


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a point cloud with optional vertex colors from ASCII or binary PLY."""

    import trimesh

    obj = trimesh.load(path, process=False)
    points = np.asarray(obj.vertices, dtype=np.float32)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    if hasattr(obj, "visual") and getattr(obj.visual, "vertex_colors", None) is not None:
        vc = np.asarray(obj.visual.vertex_colors)
        if vc.size:
            colors = vc[:, :3].astype(np.uint8)
    return points, colors


def _point_stats(points: np.ndarray) -> dict[str, object]:
    """Return robust point-cloud statistics used for sanity checks."""

    z = points[:, 2]
    return {
        "points": int(len(points)),
        "xyz_quantiles": np.round(np.percentile(points, [0, 1, 5, 25, 50, 75, 95, 99, 100], axis=0), 6).tolist(),
        "xyz_span": np.round(points.max(axis=0) - points.min(axis=0), 6).tolist(),
        "z_mad": round(float(np.median(np.abs(z - np.median(z)))), 6),
        "z_abs_p95": round(float(np.percentile(np.abs(z - np.median(z)), 95)), 6),
    }


def _render_projection(path: Path, points: np.ndarray, colors: np.ndarray, axes: tuple[int, int], color_by_z: bool) -> Path:
    """Render a quick point-cloud projection for visual QA."""

    from PIL import Image, ImageDraw

    width, height = 1000, 700
    a = points[:, axes[0]]
    b = points[:, axes[1]]
    amin, amax = np.percentile(a, [0.5, 99.5])
    bmin, bmax = np.percentile(b, [0.5, 99.5])
    xx = np.clip(((a - amin) / max(float(amax - amin), 1e-9) * (width - 1)).astype(np.int32), 0, width - 1)
    yy = np.clip(((1.0 - (b - bmin) / max(float(bmax - bmin), 1e-9)) * (height - 1)).astype(np.int32), 0, height - 1)
    image = np.zeros((height, width, 3), dtype=np.uint8) + 245
    if color_by_z:
        z = points[:, 2]
        z0, z1 = np.percentile(z, [1, 99])
        t = np.clip((z - z0) / max(float(z1 - z0), 1e-9), 0.0, 1.0)
        draw_colors = np.stack(
            [(255 * t).astype(np.uint8), (120 * (1.0 - np.abs(t - 0.5) * 2.0) + 80).astype(np.uint8), (255 * (1.0 - t)).astype(np.uint8)],
            axis=1,
        )
    else:
        draw_colors = colors
    step = max(1, len(points) // 250000)
    image[yy[::step], xx[::step]] = draw_colors[::step]
    rendered = Image.fromarray(image)
    ImageDraw.Draw(rendered).text((12, 12), path.stem, fill=(0, 0, 0))
    rendered.save(path)
    return path


def _comparison_markdown(comparison: dict[str, object]) -> str:
    """Render the static-reference comparison as markdown."""

    projections = comparison["projection_paths"]
    assert isinstance(projections, dict)
    lines = [
        "# Static Reference Comparison",
        "",
        f"- status: {comparison['status']}",
        f"- z_abs_p95_ratio: {float(comparison['z_abs_p95_ratio']):.3f}",
        f"- z_mad_ratio: {float(comparison['z_mad_ratio']):.3f}",
        f"- stream_ply: {comparison['stream_ply']}",
        f"- static_reference_ply: {comparison['static_reference_ply']}",
        "",
        "## Projections",
        "",
    ]
    for name, path in projections.items():
        lines.append(f"![{name}]({path})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app()
