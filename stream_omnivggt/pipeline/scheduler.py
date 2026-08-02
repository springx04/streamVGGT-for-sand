"""Serial and thread-pool schedulers for non-critical background work."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


class SerialScheduler:
    """Scheduler that executes tasks immediately in the caller thread."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Run a task synchronously and return a completed Future."""

        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - propagate through Future.
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """No-op shutdown for the serial scheduler."""


class ThreadedScheduler:
    """Small ThreadPoolExecutor wrapper for snapshots/render exports."""

    def __init__(self, max_workers: int = 2) -> None:
        """Create a background scheduler."""

        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stream-omnivggt")

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a background task and log failures through the Future."""

        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(_log_future_failure)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Stop the thread pool."""

        self._executor.shutdown(wait=wait)


def _log_future_failure(future: Future) -> None:
    """Log background task exceptions without blocking the stream path."""

    exc = future.exception()
    if exc is not None:
        logger.warning("Background task failed: %s", exc)

