"""Per-item playback truth, shared by the queue and all TTS backends.

Completion means the audio sink drained normally. Interrupted sentences are not
claimed as complete text; word-level alignment is unavailable. Context-local
tracking keeps direct TTS callers compatible and concurrent calls isolated.
"""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Delivery:
    valid: Callable[[], bool] = lambda: True
    started: bool = False
    completed: bool = False
    reason: Optional[str] = None


_current: ContextVar[Optional[Delivery]] = ContextVar("speech_delivery", default=None)


@contextmanager
def track(record):
    token = _current.set(record)
    try:
        yield record
    finally:
        _current.reset(token)


def allowed() -> bool:
    record = _current.get()
    if record is None or record.valid():
        return True
    record.reason = "stale_generation"
    return False


def started() -> None:
    record = _current.get()
    if record is not None:
        record.started = True


def finish(*, canceled=False) -> None:
    record = _current.get()
    if record is not None:
        record.completed = record.started and not canceled and allowed()
        if not record.completed and record.reason is None:
            record.reason = "interrupted" if record.started else "not_started"
