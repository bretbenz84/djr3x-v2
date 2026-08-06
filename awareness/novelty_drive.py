"""
awareness/novelty_drive.py — "how long since anything new happened?"

Curiosity Phase 2. The existing boredom arc measures time-since-PERSON; this
measures time-since-NOVELTY — a different drive. Novel events (a new object in
the room model, a new person met, a learned object name, an animal, a room
change remark) reset the clock; as staleness grows, Rex's appetite for
LOOKING goes up:

  * idle micro-behaviors tilt toward ambient scanning/observation,
  * (opt-in, EXPLORE_SELF_TRIGGER_ENABLED, default OFF — it moves the robot)
    a long-stale, empty, healthy-battery room can trigger a self-directed
    exploration walk, which itself feeds the room model and the
    learn-by-asking queue.

In-memory only: the clock seeds at process start (a restart is itself a fresh
look at the world). record_novel_event() is called from the capture points and
is fail-safe; nothing here ever raises into a caller.
"""

from __future__ import annotations

import logging
import threading
import time

import config

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_last_novel_at: float = time.monotonic()
_last_kind: str = "startup"
_events_this_session: int = 0


def record_novel_event(kind: str, detail: str = "") -> None:
    """Something genuinely new happened — reset the staleness clock."""
    global _last_novel_at, _last_kind, _events_this_session
    with _lock:
        _last_novel_at = time.monotonic()
        _last_kind = str(kind or "unknown")
        _events_this_session += 1
    _log.info("[novelty] %s%s — staleness clock reset", kind,
              f" ({detail})" if detail else "")
    # Something new happening lifts Rex's day mood a little — this is the single
    # choke point every genuinely-novel event already flows through.
    try:
        from intelligence import rex_mood
        rex_mood.note("novelty")
    except Exception:
        pass


def staleness_secs() -> float:
    with _lock:
        return time.monotonic() - _last_novel_at


def is_stale() -> bool:
    """The room has offered nothing new for a while — curiosity pressure is on."""
    return staleness_secs() >= float(getattr(config, "NOVELTY_STALE_AFTER_SECS", 1800.0))


def status() -> dict:
    with _lock:
        return {
            "staleness_secs": time.monotonic() - _last_novel_at,
            "last_kind": _last_kind,
            "events_this_session": _events_this_session,
        }
