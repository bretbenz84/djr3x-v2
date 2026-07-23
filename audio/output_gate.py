"""
Shared playback gate for mutually exclusive spoken-audio output.

This module provides a process-local lock so independently threaded components
cannot play two spoken audio streams at once (e.g. startup clip + TTS, or two
concurrent TTS calls from different subsystems).
"""

from contextlib import contextmanager
import threading
import time
from typing import Iterator, Optional

_playback_lock = threading.Lock()
_state_lock = threading.Lock()
_active_source: Optional[str] = None
_last_released_at: float = 0.0

# Yield hooks: fired by any BLOCKING acquirer right before it waits on the gate, so a
# preemptible holder (sound_effects) can stop early and hand the speaker over instead
# of making speech wait out a decorative chirp. Non-blocking acquirers (the effects
# themselves) do NOT fire hooks. Hooks must be fast and never raise.
_yield_hooks: "list" = []


def register_yield_hook(fn) -> None:
    """Register a callable invoked whenever a blocking caller wants the gate."""
    if fn not in _yield_hooks:
        _yield_hooks.append(fn)


def is_busy() -> bool:
    """Return True while any caller currently holds the playback gate."""
    with _state_lock:
        return _active_source is not None


def active_source() -> Optional[str]:
    """Return the current holder label, or None when idle."""
    with _state_lock:
        return _active_source


def seconds_since_release() -> float:
    """Seconds since the gate was last released. Returns inf if never held."""
    with _state_lock:
        if _active_source is not None:
            return 0.0
        if _last_released_at == 0.0:
            return float("inf")
        return time.monotonic() - _last_released_at


@contextmanager
def hold(
    source: str,
    *,
    blocking: bool = True,
    timeout: Optional[float] = None,
) -> Iterator[bool]:
    """
    Acquire the shared playback gate.

    Yields:
      True  -> lock acquired, caller may play audio
      False -> lock not acquired (only possible when blocking=False or timeout hit)
    """
    if blocking:
        for _hook in list(_yield_hooks):
            try:
                _hook()
            except Exception:
                pass
    if timeout is None:
        acquired = _playback_lock.acquire(blocking=blocking)
    else:
        acquired = _playback_lock.acquire(blocking=blocking, timeout=timeout)

    if not acquired:
        yield False
        return

    with _state_lock:
        global _active_source
        _active_source = source

    try:
        yield True
    finally:
        with _state_lock:
            global _last_released_at
            _active_source = None
            _last_released_at = time.monotonic()
        _playback_lock.release()
