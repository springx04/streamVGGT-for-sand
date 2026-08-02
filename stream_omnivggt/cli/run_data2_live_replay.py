"""Optional Python live inference and interactive replay for ``data2``.

This command is intentionally separate from the normal streaming demo.  It
uses the existing real Python OmniVGGT backend and aligned-canvas pipeline,
then adds a local OpenCV window with a reversible sparse history.  No model
code, weights, or normal project entry points are changed by the viewer.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Sequence
import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from stream_omnivggt.backend import MockOmniBackend, OmniVGGTBackend
from stream_omnivggt.config import StreamConfig, load_stream_config
from stream_omnivggt.pipeline.aligned_canvas_stream import AlignedCanvasStream, export_canvas_pointcloud, save_ply
from stream_omnivggt.pipeline.replay_history import ReplayHistory, capture_canvas
from stream_omnivggt.types import InputPacket

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency is declared in pyproject.
    cv2 = None  # type: ignore


logger = logging.getLogger(__name__)


# The quality viewer should show the complete cleaned canvas by default.  A
# positive value can still be supplied for a deliberately lighter display;
# zero means "do not downsample".
DEFAULT_DISPLAY_MAX_POINTS = 0
DEFAULT_TARGET_WIDTH = 700
DEFAULT_TARGET_SIZE = 700


@dataclass
class CloudGeometry:
    """Cached canonical point data used by the interactive camera."""

    xyz: np.ndarray
    colors: np.ndarray
    changed: np.ndarray


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Infer data2, leave a live point-cloud/replay window open, and log timings.

    ``argparse`` is used here instead of the optional Typer/Rich convenience
    layer so the viewer can run in the lean GPU environment already used by
    the Python backend.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required for the live viewer.")
    parser = _argument_parser()
    args = parser.parse_args(argv)
    image_dir = args.image_dir
    output_dir = args.output_dir
    config = args.config
    target_width = args.target_width
    target_size = args.target_size
    device = args.device
    dtype = args.dtype
    mock_backend = args.mock_backend
    save_debug = args.save_debug
    window_name = args.window_name
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    paths = _image_paths(image_dir)
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    viewer = LiveReplayViewer(window_name, len(paths), display_max_points=args.display_max_points)
    viewer.show_message("Loading OmniVGGT backend...", "The first frame will appear after model initialization")
    cv2.waitKey(1)

    cfg = load_stream_config(config)
    cfg.omni.target_width = int(target_width)
    cfg.omni.target_size = int(target_size)
    cfg.omni.patch_multiple = 14
    cfg.omni.warmup_buckets = []
    cfg.omni.device = device
    cfg.omni.dtype = dtype
    cfg.omni.preload_patch_embed = False
    cfg.change.flow_mode = "none"
    cfg.fuse.min_conf = 0.0

    backend_start = perf_counter()
    backend = MockOmniBackend() if mock_backend else OmniVGGTBackend(cfg.omni)
    backend_load_ms = (perf_counter() - backend_start) * 1000.0
    stream = AlignedCanvasStream(backend, cfg)
    history = ReplayHistory()
    viewer.attach(history)
    print({"backend": backend.name(), "backend_load_ms": round(backend_load_ms, 2), "frames": len(paths)}, flush=True)

    rows: list[dict[str, Any]] = []
    stream_start = perf_counter()
    first_pointcloud_ms: float | None = None
    stopped = False
    try:
        for frame_id, image_path in enumerate(paths):
            if viewer.poll_input():
                stopped = True
                break
            read_start = perf_counter()
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            read_ms = (perf_counter() - read_start) * 1000.0
            packet = InputPacket(frame_id, float(frame_id), rgb, None, None, None, {"source_path": str(image_path)})
            metrics = stream.push(packet, read_ms=read_ms)
            if first_pointcloud_ms is None and metrics.point_count > 0:
                first_pointcloud_ms = (perf_counter() - stream_start) * 1000.0
            debug_paths = stream.save_last_debug(output_dir / "debug", image_path.name) if save_debug else {}
            row = metrics.to_dict()
            row.update(
                {
                    "image": image_path.name,
                    "backend": backend.name(),
                    "debug_overlay": debug_paths.get("overlay"),
                    "debug_change_mask": debug_paths.get("change_mask"),
                }
            )
            record = history.append(
                frame_id,
                image_path.name,
                row,
                capture_canvas(stream),
                stream.last_debug.get("change_mask"),
            )
            if save_debug:
                # Keep a post-fusion diagnostic separate from the model-input
                # images.  This is the authoritative canvas/cloud state that
                # the live viewer replays, so boundary checks do not confuse
                # an input ROI artifact with a committed-map artifact.
                debug_dir = output_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_stem = f"frame_{frame_id:03d}_{image_path.stem}"
                state_after = history.state()
                cv2.imwrite(
                    str(debug_dir / f"{debug_stem}_canvas.png"),
                    _render_canvas(state_after, record.pipeline_mask, record.delta_mask),
                )
                cv2.imwrite(
                    str(debug_dir / f"{debug_stem}_cloud.png"),
                    _render_cloud_geometry(
                        _prepare_cloud_geometry(state_after, record.delta_mask, max_points=0),
                    ),
                )
                if state_after.depth is not None and state_after.valid is not None:
                    np.save(debug_dir / f"{debug_stem}_depth.npy", np.asarray(state_after.depth, dtype=np.float32))
                    np.save(debug_dir / f"{debug_stem}_valid.npy", np.asarray(state_after.valid, dtype=bool))
                if state_after.rgb is not None:
                    state_rgb = np.asarray(state_after.rgb, dtype=np.float32)
                    if state_rgb.max(initial=0.0) <= 1.5:
                        state_rgb = state_rgb * 255.0
                    Image.fromarray(np.clip(state_rgb, 0.0, 255.0).astype(np.uint8), mode="RGB").save(
                        debug_dir / f"{debug_stem}_state_rgb.png"
                    )
            row["delta_pixels"] = record.delta.changed_pixels if record.delta is not None else int(record.delta_mask.sum())
            row["history_fields"] = len(record.delta.fields) if record.delta is not None else 0
            rows.append(row)
            viewer.on_new_frame()
            viewer.render()
            print(
                {
                    "frame": frame_id,
                    "total_ms": round(float(metrics.total_ms), 2),
                    "model_ms": round(float(metrics.model_ms), 2),
                    "changed_ratio": round(float(metrics.changed_ratio), 4),
                    "delta_pixels": row["delta_pixels"],
                    "points": int(metrics.point_count),
                    "fallback": metrics.fallback_reason,
                },
                flush=True,
            )
            if viewer.poll_input():
                stopped = True
                break
    except Exception as exc:  # noqa: BLE001 - keep the visible window available for diagnosis.
        logger.exception("Python live replay stopped during inference: %s", exc)
        viewer.show_message("Python replay error", str(exc))
        viewer.wait_until_closed()
        raise

    if not stopped and history.frame_count:
        points, colors = _state_pointcloud(history.state())
        save_ply(str(output_dir / "pointcloud_final.ply"), points, colors)
        summary = _summary(backend.name(), backend_load_ms, first_pointcloud_ms, rows, len(points))
        payload = {"summary": summary, "frames": rows, "replay": {"frame_count": history.frame_count, "delta_history": True}}
        (output_dir / "timings.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output_dir / "timings.md").write_text(_markdown(summary, rows), encoding="utf-8")
        print(summary, flush=True)
        print(f"Wrote {output_dir / 'timings.json'}", flush=True)
        print(f"Wrote {output_dir / 'pointcloud_final.ply'}", flush=True)
    if stopped:
        print("Replay stopped by user; the window remains available until q/Esc.", flush=True)
    viewer.wait_until_closed()


def _argument_parser() -> argparse.ArgumentParser:
    """Build the dependency-light command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data2"))
    parser.add_argument("--output-dir", type=Path, default=Path("stream_omnivggt_outputs/data2_python_live_replay"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--target-width",
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help="Maximum model ROI width; 700 matches the latest quality review (use 280 for fast mode).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Maximum model ROI height; 700 matches the latest quality review (use 196 for fast mode).",
    )
    parser.add_argument(
        "--display-max-points",
        type=int,
        default=DEFAULT_DISPLAY_MAX_POINTS,
        help="Point-cloud display cap; zero keeps every cleaned point (positive values enable optional downsampling).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--mock-backend", action="store_true")
    parser.add_argument("--window-name", default="OmniVGGT Python Live Replay")
    parser.add_argument("--save-debug", dest="save_debug", action="store_true", default=True)
    parser.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    return parser


class LiveReplayViewer:
    """Two-panel OpenCV viewer for canvas state and projected point cloud."""

    def __init__(self, window_name: str, expected_frames: int, display_max_points: int = DEFAULT_DISPLAY_MAX_POINTS) -> None:
        self.window_name = window_name
        self.expected_frames = max(int(expected_frames), 1)
        self.display_max_points = max(int(display_max_points), 0)
        self.history: ReplayHistory | None = None
        self.requested = 0
        self.playing = False
        self.follow_tail = True
        self._updating_trackbar = False
        # The right panel is a fixed 760x540 point-cloud viewport.  Mouse
        # gestures are intentionally kept local to that panel so dragging the
        # timeline never changes the 3-D camera.
        self._cloud_panel_x = 760
        self.camera_yaw = -0.62
        self.camera_pitch = 0.34
        self.camera_zoom = 1.0
        self.camera_pan = np.zeros(2, dtype=np.float32)
        self._drag_mode: str | None = None
        self._last_mouse: tuple[int, int] | None = None
        self._cached_cursor = -1
        self._cached_canvas: np.ndarray | None = None
        self._cached_left: np.ndarray | None = None
        self._cached_geometry: CloudGeometry | None = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1440, 820)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        cv2.createTrackbar("frame", self.window_name, 0, max(self.expected_frames - 1, 1), self._on_trackbar)

    def attach(self, history: ReplayHistory) -> None:
        """Attach the live history and reset the slider."""

        self.history = history
        self.follow_tail = True

    def on_new_frame(self) -> None:
        """Follow newly committed frames until the user starts replaying."""

        if self.follow_tail and self.history is not None and self.history.frame_count:
            self.requested = self.history.cursor

    def show_message(self, title: str, detail: str) -> None:
        """Show a status card while the backend is loading."""

        canvas = np.full((720, 1440, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, title, (55, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, detail, (55, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 210, 240), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, canvas)

    def poll_input(self) -> bool:
        """Process keys and slider requests; return whether inference should stop."""

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return True
        if self.history is None or self.history.frame_count == 0:
            return False
        if key == ord(" "):
            self.playing = not self.playing
        elif key in (ord("a"), 81):
            self.follow_tail = False
            self.requested = max(self.requested - 1, 0)
        elif key in (ord("d"), 83):
            self.follow_tail = False
            self.requested = min(self.requested + 1, self.history.frame_count - 1)
        elif key == ord("l"):
            self.follow_tail = True
            self.requested = self.history.frame_count - 1
        elif key == ord("r"):
            self.reset_camera()
        self.requested = min(self.requested, max(self.history.frame_count - 1, 0))
        if self.playing and self.history.frame_count:
            self.requested = min(self.requested + 1, self.history.frame_count - 1)
        if self.requested != self.history.cursor:
            self.history.seek(self.requested)
        return False

    def render(self) -> None:
        """Render the currently selected replay state."""

        if self.history is None or self.history.frame_count == 0:
            return
        if self.requested != self.history.cursor:
            self.history.seek(self.requested)
        if self._cached_cursor != self.history.cursor or self._cached_geometry is None or self._cached_canvas is None:
            frame = self.history.current
            state = self.history.state()
            self._cached_canvas = _render_canvas(state, frame.pipeline_mask, frame.delta_mask)
            self._cached_left = _fit_panel(self._cached_canvas, 760, 540)
            self._cached_geometry = _prepare_cloud_geometry(
                state,
                frame.delta_mask,
                max_points=self.display_max_points,
            )
            self._cached_cursor = self.history.cursor
        self._render_cached_view(self.history.current)

    def _render_cached_view(self, frame: Any) -> None:
        """Render from cached frame data, changing only the camera projection."""

        if self._cached_left is None or self._cached_geometry is None or self.history is None:
            return
        cloud = _render_cloud_geometry(
            self._cached_geometry,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            zoom=self.camera_zoom,
            pan=self.camera_pan,
        )
        left = self._cached_left
        panel_height = max(left.shape[0], cloud.shape[0])
        panel_width = max(left.shape[1], cloud.shape[1])
        if left.shape[:2] != (panel_height, panel_width):
            left = _fit_panel(left, panel_width, panel_height)
        if cloud.shape[:2] != (panel_height, panel_width):
            cloud = _fit_panel(cloud, panel_width, panel_height)
        view = np.hstack([left, cloud])
        row = frame.metrics
        header = (
            f"frame {self.history.cursor}/{self.history.frame_count - 1}  "
            f"total {float(row.get('total_ms', 0.0)):.0f} ms  "
            f"model {float(row.get('model_ms', 0.0)):.0f} ms  "
            f"points {int(row.get('point_count', 0)):,}  "
            f"clean {len(self._cached_geometry.xyz):,}  "
            f"delta {int(frame.delta_mask.sum()):,} px"
        )
        cv2.rectangle(view, (0, 0), (view.shape[1] - 1, 36), (18, 18, 18), -1)
        cv2.putText(view, header, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
        footer = "mouse left-drag: rotate | right-drag: pan | wheel: zoom | r: reset view | red outline: committed change | drag frame / a,d: replay | q/Esc: close"
        cv2.putText(view, footer, (14, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 205, 220), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, view)
        self._set_trackbar(self.history.cursor)

    def wait_until_closed(self) -> None:
        """Keep the viewer open for manual inspection until q/Esc or close."""

        while True:
            if not self._window_exists():
                break
            if self.history is not None and self.history.frame_count:
                self.render()
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if self.history is not None and self.history.frame_count:
                if key == ord(" "):
                    self.playing = not self.playing
                elif key in (ord("a"), 81):
                    self.follow_tail = False
                    self.requested = max(self.requested - 1, 0)
                elif key in (ord("d"), 83):
                    self.follow_tail = False
                    self.requested = min(self.requested + 1, self.history.frame_count - 1)
                elif key == ord("l"):
                    self.follow_tail = True
                    self.requested = self.history.frame_count - 1
                elif key == ord("r"):
                    self.reset_camera()
                if self.playing:
                    self.requested = min(self.requested + 1, self.history.frame_count - 1)
                if self.requested != self.history.cursor:
                    self.history.seek(self.requested)
        cv2.destroyWindow(self.window_name)

    def _on_trackbar(self, position: int) -> None:
        if not self._updating_trackbar:
            self.follow_tail = False
            self.requested = int(position)
            if self.history is not None and self.history.frame_count:
                self.history.seek(self.requested)
                self.render()

    def _set_trackbar(self, position: int) -> None:
        self._updating_trackbar = True
        try:
            if self._window_exists():
                cv2.setTrackbarPos("frame", self.window_name, int(position))
        except cv2.error:
            # OpenCV raises here if the user closes the native window between
            # imshow() and the slider update.  A closed viewer is a normal
            # termination path, not an inference failure.
            pass
        finally:
            self._updating_trackbar = False

    def _window_exists(self) -> bool:
        """Return whether the native viewer window is still available."""

        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0
        except cv2.error:
            return False

    def reset_camera(self) -> None:
        """Restore the initial orbit camera."""

        self.camera_yaw = -0.62
        self.camera_pitch = 0.34
        self.camera_zoom = 1.0
        self.camera_pan[:] = 0.0
        if self.history is not None and self.history.frame_count:
            self.render()

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: Any) -> None:
        """Handle C++-style point-cloud orbit, pan, and wheel zoom gestures."""

        in_cloud = x >= self._cloud_panel_x and x < self._cloud_panel_x + 760 and 0 <= y < 540
        if event == cv2.EVENT_MOUSEWHEEL and in_cloud:
            delta = 0
            get_delta = getattr(cv2, "getMouseWheelDelta", None)
            if get_delta is not None:
                delta = int(get_delta(flags))
            if delta == 0:
                signed = (int(flags) >> 16) & 0xFFFF
                delta = signed - 0x10000 if signed & 0x8000 else signed
            self.camera_zoom = float(np.clip(self.camera_zoom * (1.15 if delta >= 0 else 1.0 / 1.15), 0.25, 4.0))
            return

        if event == cv2.EVENT_LBUTTONDOWN and in_cloud:
            self._drag_mode = "rotate"
            self._last_mouse = (x, y)
            return
        if event == cv2.EVENT_RBUTTONDOWN and in_cloud:
            self._drag_mode = "pan"
            self._last_mouse = (x, y)
            return
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self._drag_mode = None
            self._last_mouse = None
            return
        if event != cv2.EVENT_MOUSEMOVE or self._drag_mode is None or self._last_mouse is None:
            return

        last_x, last_y = self._last_mouse
        dx, dy = x - last_x, y - last_y
        self._last_mouse = (x, y)
        if self._drag_mode == "rotate":
            self.camera_yaw += float(dx) * 0.012
            self.camera_pitch = float(np.clip(self.camera_pitch + float(dy) * 0.012, -1.45, 1.45))
        else:
            self.camera_pan += np.asarray((dx, dy), dtype=np.float32)
            self.camera_pan = np.clip(self.camera_pan, -500.0, 500.0)


def _image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def _render_canvas(state: Any, pipeline_mask: np.ndarray, delta_mask: np.ndarray) -> np.ndarray:
    rgb = state.rgb
    if rgb is None:
        return np.full((540, 760, 3), 25, dtype=np.uint8)
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) <= 1.5:
        image = image * 255.0
    bgr = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    valid = np.asarray(state.valid, dtype=bool) if state.valid is not None else np.ones(bgr.shape[:2], dtype=bool)
    bgr[~valid] = (24, 24, 24)
    if pipeline_mask.shape == bgr.shape[:2] and pipeline_mask.any():
        # Keep the source RGB untouched.  Filling a change mask with a
        # translucent tint makes the changed sand look darker and was easy to
        # mistake for a color-fusion regression.  An outline carries the same
        # diagnostic information without modifying the underlying colors.
        pipeline_edge = pipeline_mask & ~_erode(pipeline_mask, 2)
        bgr[pipeline_edge] = (0, 210, 255)  # BGR yellow
    if delta_mask.shape == bgr.shape[:2] and delta_mask.any():
        edge = delta_mask & ~_erode(delta_mask, 2)
        bgr[edge] = (0, 0, 255)  # BGR red: actual committed cells
    cv2.putText(bgr, "aligned canvas", (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(bgr, "yellow outline: pipeline change  red outline: committed delta", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1, cv2.LINE_AA)
    return bgr


def _render_pointcloud(
    state: Any,
    delta_mask: np.ndarray,
    max_points: int = 24000,
    yaw: float = -0.62,
    pitch: float = 0.34,
    zoom: float = 1.0,
    pan: np.ndarray | None = None,
) -> np.ndarray:
    geometry = _prepare_cloud_geometry(state, delta_mask, max_points=max_points)
    return _render_cloud_geometry(geometry, yaw=yaw, pitch=pitch, zoom=zoom, pan=pan)


def _render_cloud_geometry(
    geometry: CloudGeometry,
    yaw: float = -0.62,
    pitch: float = 0.34,
    zoom: float = 1.0,
    pan: np.ndarray | None = None,
) -> np.ndarray:
    """Render cached geometry with only a cheap camera projection."""

    points = _project_geometry(geometry.xyz, yaw=yaw, pitch=pitch, zoom=zoom, pan=pan)
    colors = geometry.colors
    changed = geometry.changed
    height, width = 540, 760
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(image, "replayed point cloud", (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    if points.size == 0:
        cv2.putText(image, "waiting for valid fused points...", (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 200, 210), 1, cv2.LINE_AA)
        return image
    px = np.rint(points[:, 0]).astype(np.int32)
    py = np.rint(points[:, 1]).astype(np.int32)
    keep = (px >= 0) & (px < width) & (py >= 40) & (py < height - 25)
    px, py, colors, changed = px[keep], py[keep], colors[keep], changed[keep]
    # The canonical canvas has one sample per XY cell, but a rotated view can
    # still leave single-pixel raster gaps when every sample is drawn as one
    # screen pixel.  A small 3x3 point footprint matches the C++ viewer's
    # point-size behavior and closes display-only pinholes without changing
    # the cached geometry or its depth values.
    bgr_colors = colors[:, ::-1]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            px_i = px + dx
            py_i = py + dy
            inside = (px_i >= 0) & (px_i < width) & (py_i >= 40) & (py_i < height - 25)
            image[py_i[inside], px_i[inside]] = bgr_colors[inside]
    if changed.any():
        # Preserve the real RGB at changed points.  Mark only the outside ring
        # of the changed footprint so the replay remains color-faithful.
        changed_pixels = np.zeros((height, width), dtype=bool)
        changed_pixels[py[changed], px[changed]] = True
        changed_ring = cv2.dilate(changed_pixels.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        changed_ring &= ~changed_pixels
        image[changed_ring] = (0, 0, 255)  # BGR red outline
    cv2.line(image, (width // 2, height // 2), (width // 2 + 65, height // 2), (80, 90, 100), 1, cv2.LINE_AA)
    cv2.line(image, (width // 2, height // 2), (width // 2, height // 2 - 65), (80, 90, 100), 1, cv2.LINE_AA)
    cv2.putText(image, f"{len(px):,} rendered / {len(points):,} valid", (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 205, 215), 1, cv2.LINE_AA)
    return image


def _prepare_cloud_geometry(state: Any, delta_mask: np.ndarray, max_points: int) -> CloudGeometry:
    """Extract the same cleaned point data used by the normal PLY export.

    The raw canvas can contain sparse ROI borders and small invalid islands.
    ``export_canvas_pointcloud`` fills narrow gaps, smooths the height field,
    and trims the outer border before this viewer caches the geometry.
    """

    depth = state.depth
    valid = state.valid
    rgb = state.rgb
    if depth is None or valid is None or rgb is None:
        return CloudGeometry(np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty((0,), bool))
    points, colors = export_canvas_pointcloud(
        np.asarray(depth, dtype=np.float32),
        np.asarray(rgb, dtype=np.float32),
        np.asarray(valid, dtype=bool),
    )
    if points.size == 0:
        return CloudGeometry(np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), np.empty((0,), bool))

    # The shared exporter already returns a plane-residual height field.  Keep
    # that metric Z scale intact: normalizing this small residual to [-1, 1]
    # would turn a nearly planar surface into an artificially tall shape.
    if max_points > 0 and len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, colors = points[selected], colors[selected]
    xyz = points.astype(np.float32, copy=True)

    # Map cleaned point coordinates back to canonical cells for Delta coloring.
    height, width = np.asarray(depth).shape[:2]
    scale = float(max(height, width))
    xx = np.rint(points[:, 0] * scale + width * 0.5).astype(np.int32)
    yy = np.rint(-points[:, 1] * scale + height * 0.5).astype(np.int32)
    changed = np.zeros(len(points), dtype=bool)
    inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    if delta_mask.shape == (height, width):
        changed[inside] = delta_mask[yy[inside], xx[inside]]
    return CloudGeometry(xyz, colors, changed)


def _project_geometry(
    xyz: np.ndarray,
    yaw: float = -0.62,
    pitch: float = 0.34,
    zoom: float = 1.0,
    pan: np.ndarray | None = None,
) -> np.ndarray:
    """Project cached canonical XYZ into the point-cloud viewport."""

    if xyz.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x1 = x * cos_yaw - z * sin_yaw
    z1 = x * sin_yaw + z * cos_yaw
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    y2 = y * cos_pitch - z1 * sin_pitch
    pan_xy = np.zeros(2, dtype=np.float32) if pan is None else np.asarray(pan, dtype=np.float32).reshape(2)
    points = np.empty((xyz.shape[0], 2), dtype=np.float32)
    points[:, 0] = x1 * 540.0 * float(zoom) + 380.0 + pan_xy[0]
    points[:, 1] = -y2 * 540.0 * float(zoom) + 285.0 + pan_xy[1]
    return points


def _project_points(
    state: Any,
    delta_mask: np.ndarray,
    max_points: int,
    yaw: float = -0.62,
    pitch: float = 0.34,
    zoom: float = 1.0,
    pan: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = _prepare_cloud_geometry(state, delta_mask, max_points=max_points)
    points = _project_geometry(geometry.xyz, yaw=yaw, pitch=pitch, zoom=zoom, pan=pan)
    return points, geometry.colors, geometry.changed


def _state_pointcloud(state: Any) -> tuple[np.ndarray, np.ndarray]:
    if state.depth is None or state.rgb is None or state.valid is None:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    return export_canvas_pointcloud(
        np.asarray(state.depth, dtype=np.float32),
        np.asarray(state.rgb, dtype=np.float32),
        np.asarray(state.valid, dtype=bool),
    )


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def _erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    if not mask.any() or pixels <= 0:
        return mask
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _summary(backend_name: str, backend_load_ms: float, first_pointcloud_ms: float | None, rows: list[dict[str, Any]], point_count: int) -> dict[str, Any]:
    subsequent = rows[1:]
    return {
        "backend": backend_name,
        "backend_load_ms": backend_load_ms,
        "image_count": len(rows),
        "first_input_to_pointcloud_ms_including_backend_load": backend_load_ms + (first_pointcloud_ms or 0.0),
        "first_frame_total_ms": rows[0].get("total_ms") if rows else None,
        "first_frame_model_ms": rows[0].get("model_ms") if rows else None,
        "subsequent_avg_total_ms": _mean(subsequent, "total_ms"),
        "subsequent_p90_total_ms": _p90(subsequent, "total_ms"),
        "subsequent_avg_model_ms": _mean(subsequent, "model_ms"),
        "subsequent_avg_delta_pixels": _mean(subsequent, "delta_pixels"),
        "subsequent_avg_anchor_pixels": _mean(subsequent, "anchor_pixels"),
        "final_point_count": point_count,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.mean([float(row[key]) for row in rows])) if rows else None


def _p90(rows: list[dict[str, Any]], key: str) -> float | None:
    return float(np.percentile([float(row[key]) for row in rows], 90)) if rows else None


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = ["# data2 Python live replay", "", "## Summary", ""]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    lines += [
        "",
        "## Frames",
        "",
        "| frame | total_ms | model_ms | changed_ratio | delta_pixels | anchor_pixels | points |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['frame_id']} | {float(row['total_ms']):.2f} | {float(row['model_ms']):.2f} | {float(row['changed_ratio']):.4f} | {int(row['delta_pixels'])} | {int(row.get('anchor_pixels', 0))} | {int(row['point_count'])} |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
