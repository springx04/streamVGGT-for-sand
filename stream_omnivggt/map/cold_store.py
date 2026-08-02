"""Memmap-backed cold block storage."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import logging

import numpy as np

from stream_omnivggt.map.block_hash import block_key_from_str, block_key_to_str
from stream_omnivggt.types import BlockMeta

logger = logging.getLogger(__name__)


class ColdStore:
    """Fixed-slot NumPy memmap store for evicted surfel blocks."""

    def __init__(self, root: str | Path, capacity: int = 4096, max_points_per_block: int = 2048) -> None:
        """Create or open a memmap cold store."""

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.max_points_per_block = int(max_points_per_block)
        self.data_path = self.root / "blocks.dat"
        self.index_path = self.root / "index.json"
        mode = "r+" if self.data_path.exists() else "w+"
        self.data = np.memmap(self.data_path, mode=mode, dtype=np.float32, shape=(self.capacity, self.max_points_per_block, 8))
        self.index: dict[str, dict[str, Any]] = {}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def write_block(self, key: tuple[int, int, int], block: dict[str, np.ndarray], meta: BlockMeta | None = None) -> None:
        """Write one block's points/colors/weights to a memmap slot."""

        slot = self._slot_for_key(key)
        row = self.data[slot]
        row.fill(0.0)
        points = np.asarray(block.get("points", np.empty((0, 3))), dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(block.get("colors", np.zeros_like(points)), dtype=np.float32).reshape(-1, 3)
        weights = np.asarray(block.get("weights", np.ones((points.shape[0],))), dtype=np.float32).reshape(-1)
        count = min(points.shape[0], self.max_points_per_block)
        if count < points.shape[0]:
            logger.warning("Truncating cold block %s from %d to %d points.", key, points.shape[0], count)
        row[:count, 0:3] = points[:count]
        row[:count, 3:6] = colors[:count]
        row[:count, 6] = weights[:count]
        row[:count, 7] = 1.0
        entry = {"slot": slot, "count": count}
        if meta is not None:
            entry["meta"] = asdict(meta)
        self.index[block_key_to_str(key)] = entry
        self.data.flush()
        self._save_index()

    def read_block(self, key: tuple[int, int, int]) -> dict[str, np.ndarray] | None:
        """Read one block from the memmap store."""

        entry = self.index.get(block_key_to_str(key))
        if entry is None:
            return None
        slot = int(entry["slot"])
        count = int(entry["count"])
        row = np.asarray(self.data[slot, :count], dtype=np.float32)
        return {
            "points": row[:, 0:3].copy(),
            "colors": row[:, 3:6].copy(),
            "weights": row[:, 6].copy(),
            "normals": _default_normals(count),
            "timestamps": np.zeros((count,), dtype=np.float32),
        }

    def read_meta(self, key: tuple[int, int, int]) -> BlockMeta | None:
        """Read stored block metadata if present."""

        entry = self.index.get(block_key_to_str(key))
        if entry is None or "meta" not in entry:
            return None
        raw = dict(entry["meta"])
        raw["key"] = tuple(raw["key"])
        return BlockMeta(**raw)

    def keys(self) -> list[tuple[int, int, int]]:
        """Return all cold block keys."""

        return [block_key_from_str(key) for key in self.index]

    def _slot_for_key(self, key: tuple[int, int, int]) -> int:
        """Return an existing or new slot for a key."""

        encoded = block_key_to_str(key)
        if encoded in self.index:
            return int(self.index[encoded]["slot"])
        used = {int(entry["slot"]) for entry in self.index.values()}
        for slot in range(self.capacity):
            if slot not in used:
                return slot
        raise RuntimeError("ColdStore capacity exhausted.")

    def _save_index(self) -> None:
        """Persist the JSON index."""

        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")


def _default_normals(count: int) -> np.ndarray:
    """Create default +Z normals for restored surfel blocks."""

    normals = np.zeros((count, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    return normals

