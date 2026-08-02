"""Metric helpers for stream processing and benchmarks."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Iterable
import statistics

from stream_omnivggt.types import StreamMetrics


class Stopwatch:
    """Small perf_counter-based stopwatch."""

    def __init__(self) -> None:
        """Start a new stopwatch."""

        self.start = perf_counter()

    def lap_ms(self) -> float:
        """Return elapsed milliseconds since construction."""

        return (perf_counter() - self.start) * 1000.0


def metrics_to_dict(metrics: StreamMetrics) -> dict[str, float | int | bool | str | None]:
    """Convert StreamMetrics to a plain dictionary."""

    return asdict(metrics)


def summarize_metrics(metrics: Iterable[StreamMetrics]) -> dict[str, dict[str, float]]:
    """Return average, p90, and p99 summaries for numeric metric fields."""

    rows = [metrics_to_dict(item) for item in metrics]
    if not rows:
        return {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    summary: dict[str, dict[str, float]] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        ordered = sorted(values)
        summary[key] = {
            "avg": float(statistics.fmean(values)),
            "p90": _percentile(ordered, 0.90),
            "p99": _percentile(ordered, 0.99),
        }
    return summary


def _percentile(ordered_values: list[float], q: float) -> float:
    """Compute a nearest-rank percentile from sorted values."""

    if not ordered_values:
        return 0.0
    idx = min(max(int(round((len(ordered_values) - 1) * q)), 0), len(ordered_values) - 1)
    return float(ordered_values[idx])

