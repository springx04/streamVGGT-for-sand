"""Backend protocol for local-window OmniVGGT inference."""

from __future__ import annotations

from typing import Any, Protocol

from stream_omnivggt.types import OmniPrediction


class BaseOmniBackend(Protocol):
    """Protocol implemented by real and mock OmniVGGT backends."""

    def warmup(self, bucket_shapes: list[tuple[int, int, int, int]]) -> None:
        """Pre-run representative bucket shapes to reduce first-frame latency."""

    def run_window(self, batch: dict[str, Any]) -> OmniPrediction:
        """Run one bucketed local window and return normalized predictions."""

    def name(self) -> str:
        """Return a short backend name for metrics and logging."""

