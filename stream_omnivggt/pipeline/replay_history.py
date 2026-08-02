"""Reversible canvas history for the optional Python live viewer.

The normal streaming path keeps only the latest aligned canvas.  This module
adds a small, external history layer: frame zero stores the initial canvas and
each later frame stores sparse before/after values for cells that changed.
Seeking therefore does not rerun inference and does not mutate the inference
engine.  It is deliberately independent of the existing map and backend code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


CANVAS_FIELDS = ("rgb", "depth", "conf", "weight", "valid", "support")


@dataclass
class CanvasState:
    """Copyable view of the aligned canvas arrays."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    conf: np.ndarray | None = None
    weight: np.ndarray | None = None
    valid: np.ndarray | None = None
    support: np.ndarray | None = None

    @classmethod
    def from_stream(cls, stream: Any) -> "CanvasState":
        """Capture only the public NumPy canvas buffers from a stream object."""

        values: dict[str, np.ndarray | None] = {}
        for name in CANVAS_FIELDS:
            value = getattr(stream, name, None)
            values[name] = None if value is None else np.array(value, copy=True, order="C")
        return cls(**values)

    def copy(self) -> "CanvasState":
        """Return a deep copy suitable for independent replay state."""

        return CanvasState(
            **{
                name: None if (value := getattr(self, name)) is None else np.array(value, copy=True, order="C")
                for name in CANVAS_FIELDS
            }
        )


@dataclass
class FieldDelta:
    """Sparse or full replacement for one canvas field."""

    name: str
    index: np.ndarray | None = None
    before: np.ndarray | None = None
    after: np.ndarray | None = None
    before_full: np.ndarray | None = None
    after_full: np.ndarray | None = None
    trailing_shape: tuple[int, ...] = ()

    @property
    def is_full(self) -> bool:
        """Whether this change replaces the complete field."""

        return self.index is None

    def apply(self, state: CanvasState, forward: bool) -> None:
        """Apply this change forward or backward in place."""

        if self.is_full:
            value = self.after_full if forward else self.before_full
            setattr(state, self.name, None if value is None else np.array(value, copy=True, order="C"))
            return

        value = self.after if forward else self.before
        if value is None or self.index is None:
            return
        target = getattr(state, self.name)
        if target is None:
            raise RuntimeError(f"Cannot apply sparse delta to empty field {self.name!r}.")
        flat = target.reshape((-1,) + self.trailing_shape)
        flat[self.index] = value


@dataclass
class CanvasDelta:
    """All sparse field changes between two consecutive replay frames."""

    frame_id: int
    fields: tuple[FieldDelta, ...]
    changed_mask: np.ndarray

    @property
    def changed_pixels(self) -> int:
        """Number of canonical cells touched by this delta."""

        return int(self.changed_mask.sum())

    def apply(self, state: CanvasState, forward: bool) -> None:
        """Apply field changes in a deterministic order."""

        for field in self.fields:
            field.apply(state, forward)


@dataclass
class ReplayFrame:
    """Metadata plus the reversible delta for one committed frame."""

    frame_id: int
    image_name: str
    metrics: dict[str, Any]
    pipeline_mask: np.ndarray
    delta: CanvasDelta | None

    @property
    def delta_mask(self) -> np.ndarray:
        """Return the actual state-write mask for this frame."""

        if self.delta is None:
            return np.zeros_like(self.pipeline_mask, dtype=bool)
        return self.delta.changed_mask


class ReplayHistory:
    """Append-only history with O(number of changed cells) seeks."""

    def __init__(self) -> None:
        """Create an empty history."""

        self._frames: list[ReplayFrame] = []
        self._initial: CanvasState | None = None
        self._state: CanvasState | None = None
        self._cursor = -1

    @property
    def frames(self) -> tuple[ReplayFrame, ...]:
        """Read-only tuple of committed frame records."""

        return tuple(self._frames)

    @property
    def frame_count(self) -> int:
        """Number of committed frames."""

        return len(self._frames)

    @property
    def cursor(self) -> int:
        """Current replay cursor."""

        return self._cursor

    @property
    def current(self) -> ReplayFrame:
        """Current frame record."""

        if self._cursor < 0:
            raise IndexError("Replay history is empty.")
        return self._frames[self._cursor]

    def append(
        self,
        frame_id: int,
        image_name: str,
        metrics: dict[str, Any],
        state: CanvasState,
        pipeline_mask: np.ndarray | None,
    ) -> ReplayFrame:
        """Append a committed state and return its replay record.

        If the user was browsing an older frame while inference continued, the
        cursor is first returned to the live tail so the new delta is based on
        the actual preceding committed state.
        """

        if self._frames and self._cursor != len(self._frames) - 1:
            self.seek(len(self._frames) - 1)

        after = state.copy()
        mask = _normalise_mask(pipeline_mask, after)
        if not self._frames:
            self._initial = after.copy()
            self._state = after.copy()
            initial_mask = _state_valid_mask(after)
            if initial_mask.shape == mask.shape:
                actual_mask = initial_mask
            else:
                actual_mask = np.zeros_like(mask, dtype=bool)
            record = ReplayFrame(frame_id, image_name, dict(metrics), mask, None)
            # The first frame is a full initial state; keep its write mask for
            # display without pretending it is a later sparse delta.
            record.delta = CanvasDelta(frame_id, (), actual_mask)
            self._frames.append(record)
            self._cursor = 0
            return record

        assert self._state is not None
        delta = make_canvas_delta(frame_id, self._state, after, fallback_mask=mask)
        delta.apply(after, forward=False)  # validate the sparse representation
        # Restore the after state after the validation pass; the append path
        # must not expose an intermediate replay state to the viewer.
        after = state.copy()
        self._state = after.copy()
        record = ReplayFrame(frame_id, image_name, dict(metrics), mask, delta)
        self._frames.append(record)
        self._cursor = len(self._frames) - 1
        return record

    def seek(self, target: int) -> ReplayFrame:
        """Move to a historical frame without running inference."""

        if not self._frames:
            raise IndexError("Replay history is empty.")
        target = int(np.clip(target, 0, len(self._frames) - 1))
        assert self._state is not None
        while self._cursor < target:
            next_frame = self._frames[self._cursor + 1]
            assert next_frame.delta is not None
            next_frame.delta.apply(self._state, forward=True)
            self._cursor += 1
        while self._cursor > target:
            current_frame = self._frames[self._cursor]
            assert current_frame.delta is not None
            current_frame.delta.apply(self._state, forward=False)
            self._cursor -= 1
        return self.current

    def state(self) -> CanvasState:
        """Return a read-only-by-convention copy for rendering."""

        if self._state is None:
            raise IndexError("Replay history is empty.")
        return self._state.copy()


def make_canvas_delta(
    frame_id: int,
    before: CanvasState,
    after: CanvasState,
    fallback_mask: np.ndarray | None = None,
) -> CanvasDelta:
    """Build a reversible sparse delta between two canvas states."""

    fields: list[FieldDelta] = []
    changed_mask: np.ndarray | None = None
    for name in CANVAS_FIELDS:
        old = getattr(before, name)
        new = getattr(after, name)
        field_delta, field_mask = _make_field_delta(name, old, new)
        if field_delta is not None:
            fields.append(field_delta)
        if field_mask is not None:
            changed_mask = field_mask.copy() if changed_mask is None else _merge_masks(changed_mask, field_mask)

    if changed_mask is None:
        changed_mask = _normalise_mask(fallback_mask, after)
    elif fallback_mask is not None:
        fallback = _normalise_mask(fallback_mask, after)
        if fallback.shape == changed_mask.shape:
            # Keep the actual write mask as the primary signal.  The pipeline
            # mask is still stored on ReplayFrame and rendered separately.
            changed_mask = changed_mask.astype(bool, copy=False)
    return CanvasDelta(frame_id, tuple(fields), changed_mask.astype(bool, copy=False))


def capture_canvas(stream: Any) -> CanvasState:
    """Convenience adapter used by the live CLI."""

    return CanvasState.from_stream(stream)


def _make_field_delta(
    name: str,
    old: np.ndarray | None,
    new: np.ndarray | None,
) -> tuple[FieldDelta | None, np.ndarray | None]:
    if old is None or new is None or old.shape != new.shape:
        if _same_value(old, new):
            return None, None
        return FieldDelta(name, before_full=_copy_optional(old), after_full=_copy_optional(new)), _full_change_mask(old, new)

    cell_mask = _cell_difference(old, new)
    if not cell_mask.any():
        return None, None
    if old.ndim < 2:
        index = np.flatnonzero(cell_mask.ravel()).astype(np.int64)
        return FieldDelta(name, index=index, before=old.reshape(-1)[index].copy(), after=new.reshape(-1)[index].copy()), cell_mask
    trailing_shape = tuple(old.shape[2:])
    index = np.flatnonzero(cell_mask.ravel()).astype(np.int64)
    old_flat = old.reshape((-1,) + trailing_shape)
    new_flat = new.reshape((-1,) + trailing_shape)
    return (
        FieldDelta(
            name,
            index=index,
            before=old_flat[index].copy(),
            after=new_flat[index].copy(),
            trailing_shape=trailing_shape,
        ),
        cell_mask,
    )


def _cell_difference(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    if old.ndim < 2:
        return _element_difference(old, new)
    trailing = tuple(old.shape[2:])
    old_flat = old.reshape((-1,) + trailing)
    new_flat = new.reshape((-1,) + trailing)
    diff = _element_difference(old_flat, new_flat)
    return diff.reshape(old.shape[:2] + (() if not trailing else (int(np.prod(trailing)),))).any(axis=-1) if trailing else diff.reshape(old.shape[:2])


def _element_difference(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    equal = old == new
    if np.issubdtype(old.dtype, np.floating) or np.issubdtype(new.dtype, np.floating):
        equal = equal | (np.isnan(old) & np.isnan(new))
    return ~equal


def _same_value(old: np.ndarray | None, new: np.ndarray | None) -> bool:
    if old is None or new is None:
        return old is None and new is None
    return old.shape == new.shape and not _element_difference(old, new).any()


def _full_change_mask(old: np.ndarray | None, new: np.ndarray | None) -> np.ndarray:
    source = new if new is not None else old
    if source is None:
        return np.zeros((0, 0), dtype=bool)
    if source.ndim >= 2:
        return np.ones(source.shape[:2], dtype=bool)
    return np.ones((source.size, 1), dtype=bool)


def _normalise_mask(mask: np.ndarray | None, state: CanvasState) -> np.ndarray:
    if mask is not None:
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim == 2:
            return arr.copy()
    inferred = _state_valid_mask(state)
    return inferred.astype(bool, copy=True)


def _state_valid_mask(state: CanvasState) -> np.ndarray:
    for value in (state.valid, state.support, state.depth, state.rgb):
        if value is not None and value.ndim >= 2:
            if value.ndim == 2:
                return np.isfinite(value) & (value != 0)
            return np.isfinite(value).all(axis=tuple(range(2, value.ndim)))
    return np.zeros((0, 0), dtype=bool)


def _merge_masks(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        return left
    return left | right


def _copy_optional(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else np.array(value, copy=True, order="C")

