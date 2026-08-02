"""Configuration objects for the streaming OmniVGGT wrapper."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class OmniConfig:
    """OmniVGGT backend and model execution settings."""

    target_width: int = 518
    target_size: int = 518
    patch_multiple: int = 14
    compile_mode: str = "none"
    engine_mode: str = "pytorch"
    device: str = "auto"
    dtype: str = "auto"
    checkpoint_path: str | None = None
    repo_root: str | None = None
    preload_patch_embed: bool = False
    warmup_buckets: list[tuple[int, int, int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Fill mutable defaults after dataclass initialization."""

        if self.warmup_buckets is None:
            self.warmup_buckets = [(1, 3, self.target_size, self.target_width), (4, 3, self.target_size, self.target_width)]


@dataclass(slots=True)
class AlignConfig:
    """External rotation and height alignment settings."""

    enable_rotation: bool = True
    enable_height: bool = True
    min_quality: float = 0.2
    fail_open: bool = True


@dataclass(slots=True)
class ChangeDetectConfig:
    """Dense change-mask thresholds and component weights."""

    flow_mode: str = "farneback"
    image_l1_thr: float = 12.0 / 255.0
    depth_rel_thr: float = 0.03
    flow_thr_px: float = 2.0
    low_conf_thr: float = 0.2
    small_change_ratio: float = 0.02
    scene_jump_ratio: float = 0.35
    no_change_ratio: float = 0.001
    lambda_image: float = 1.0
    lambda_depth: float = 1.0
    lambda_flow: float = 0.25
    lambda_conf: float = 0.25
    dilate_ksize: int = 3
    block_pixels: int = 16


@dataclass(slots=True)
class WindowConfig:
    """Active local-window selection settings."""

    allowed_buckets: tuple[int, ...] = (3, 4, 6, 8)
    default_window: int = 4
    medium_window: int = 6
    refresh_window: int = 8
    max_keyframes: int = 32
    recency_tau_sec: float = 4.0
    camera_bonus: float = 1.0
    depth_bonus: float = 0.4
    anchor_bonus: float = 2.0
    overlap_bonus: float = 1.0
    mean_conf_bonus: float = 0.5


@dataclass(slots=True)
class BlockConfig:
    """Spatial block and voxel sizing settings."""

    voxel_size: float = 0.03
    block_resolution: int = 16
    enable_open3d: bool = True


@dataclass(slots=True)
class CacheConfig:
    """Hot/cold map cache settings."""

    max_hot_blocks: int = 512
    enable_memmap: bool = True
    cold_store_dir: str = "stream_omnivggt_cold_store"
    memmap_capacity: int = 4096
    max_points_per_block: int = 2048


@dataclass(slots=True)
class FuseConfig:
    """Point, surfel, and simplified TSDF fusion settings."""

    point_mode: str = "world_points"
    min_conf: float = 0.1
    w_max: float = 32.0
    lambda_age: float = 0.02
    merge_radius: float = 0.015
    occ_z_thr: float = 0.15
    replace_on_occlusion: bool = True


@dataclass(slots=True)
class FallbackConfig:
    """Fallback triggers for jumps, low overlap, and dropped frames."""

    min_overlap_ratio: float = 0.1
    min_align_quality: float = 0.2
    dropped_frame_dt: float = 1.0
    new_segment_on_scene_jump: bool = True


@dataclass(slots=True)
class BenchmarkConfig:
    """Benchmark defaults and strategy hints."""

    strategy: str = "block_incremental"
    frame_count: int = 20
    async_render: bool = False


@dataclass(slots=True)
class StreamConfig:
    """Top-level configuration composed from focused sub-configs."""

    omni: OmniConfig = None  # type: ignore[assignment]
    align: AlignConfig = None  # type: ignore[assignment]
    change: ChangeDetectConfig = None  # type: ignore[assignment]
    window: WindowConfig = None  # type: ignore[assignment]
    block: BlockConfig = None  # type: ignore[assignment]
    cache: CacheConfig = None  # type: ignore[assignment]
    fuse: FuseConfig = None  # type: ignore[assignment]
    fallback: FallbackConfig = None  # type: ignore[assignment]
    benchmark: BenchmarkConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Create missing nested config objects."""

        self.omni = self.omni or OmniConfig()
        self.align = self.align or AlignConfig()
        self.change = self.change or ChangeDetectConfig()
        self.window = self.window or WindowConfig()
        self.block = self.block or BlockConfig()
        self.cache = self.cache or CacheConfig()
        self.fuse = self.fuse or FuseConfig()
        self.fallback = self.fallback or FallbackConfig()
        self.benchmark = self.benchmark or BenchmarkConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamConfig":
        """Build a StreamConfig from nested dictionaries."""

        cfg = cls()
        _merge_dataclass(cfg, data)
        return cfg

    def flat_thresholds(self) -> dict[str, Any]:
        """Return change-detection thresholds expected by compute_change_mask."""

        values = asdict(self.change)
        values.update({"voxel_size": self.block.voxel_size, "block_resolution": self.block.block_resolution})
        return values


def _resolve_dataclass_type(annotation: Any) -> type[Any] | None:
    """Return a dataclass type from an annotation, handling simple unions."""

    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _merge_dataclass(obj: Any, values: dict[str, Any]) -> None:
    """Recursively merge dictionary values into an existing dataclass object."""

    field_map = {f.name: f for f in fields(obj)}
    for key, value in values.items():
        if key not in field_map:
            logger.warning("Ignoring unknown config key: %s", key)
            continue
        current = getattr(obj, key)
        dataclass_type = _resolve_dataclass_type(field_map[key].type)
        if isinstance(value, dict) and is_dataclass(current):
            _merge_dataclass(current, value)
        elif isinstance(value, dict) and dataclass_type is not None:
            nested = dataclass_type()
            _merge_dataclass(nested, value)
            setattr(obj, key, nested)
        elif key == "warmup_buckets" and isinstance(value, list):
            setattr(obj, key, [tuple(item) for item in value])
        elif key == "allowed_buckets" and isinstance(value, list):
            setattr(obj, key, tuple(int(item) for item in value))
        else:
            setattr(obj, key, value)


def load_stream_config(path: str | Path | None) -> StreamConfig:
    """Load a StreamConfig from JSON or YAML, returning defaults for None."""

    if path is None:
        return StreamConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML to be installed.") from exc
        loaded = yaml.safe_load(text)
        data = loaded or {}
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")
    return StreamConfig.from_dict(data)


def dump_stream_config(cfg: StreamConfig, path: str | Path) -> None:
    """Write config as JSON for reproducible benchmark runs."""

    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

