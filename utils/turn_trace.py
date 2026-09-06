"""
utils/turn_trace.py — one turn's stage stamps and model-call counts.

Lean Brain restructuring, phase 0 (docs/lean_brain_restructuring_plan.md): before
anything about the reply path changes, every turn has to be able to say WHERE its
time went and WHAT it asked a model for. The `[latency]` / `[ttfs]` /
`[character_loop]` logs already carry a turn ID, transcript-ready and first-audio
stamps; this module adds the stages between them (ASR, speaker ID, context
assembly, model request, first token, first sentence, TTS request, cancellation)
and a per-turn count of every model call — hosted OpenAI requests by client
label, local Ollama generations, and embedding requests — including the ones
made from background threads the turn spawned (the surprise classifier, the
transcription / speaker-ID pair).

Ownership: `intelligence.interaction._handle_speech_segment` begins and ends a
trace around each turn and folds `snapshot()` into its `[character_loop]` line.
Everything else only STAMPS or COUNTS through the module-level helpers, which
are no-ops when no turn is active, so a producer never needs to know whether it
is running inside a turn.

Attribution: the turn's own thread finds its trace through a contextvar. Threads
the turn spawns do not inherit that context (threading.Thread does not copy
contextvars), so `current()` falls back to the single ACTIVE turn. One turn is
processed at a time in practice (the voice loop is serial and text input is
locked), so the fallback is the right owner; if two turns ever overlap, the
background counts land on whichever began last — telemetry, not truth.

Process-wide totals by call kind are kept too (`totals()`), so idle-time work
(the conversation-arc refresh, proactive lines) is counted even though it
belongs to no turn.
"""

from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnTrace:
    turn_id: Optional[int] = None
    started_at: float = field(default_factory=time.monotonic)
    stamps: dict = field(default_factory=dict)   # stage name -> monotonic seconds
    calls: dict = field(default_factory=dict)    # call kind -> count
    values: dict = field(default_factory=dict)   # free-form measurements (chars, flags)
    cancel_reason: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def stamp(self, name: str, when: Optional[float] = None, *, overwrite: bool = False) -> bool:
        """Record when `name` happened (default: now). First stamp wins unless
        `overwrite`; returns True when the stamp was written."""
        at = time.monotonic() if when is None else float(when)
        with self._lock:
            if name in self.stamps and not overwrite:
                return False
            self.stamps[str(name)] = at
            return True

    def count(self, kind: str, n: int = 1) -> None:
        with self._lock:
            self.calls[str(kind)] = int(self.calls.get(str(kind), 0)) + int(n)

    def set_value(self, name: str, value) -> None:
        with self._lock:
            self.values[str(name)] = value

    def cancel(self, reason: str) -> None:
        """Mark the turn cancelled (first reason wins) and stamp when."""
        with self._lock:
            if self.cancel_reason is None:
                self.cancel_reason = str(reason or "cancelled")
            self.stamps.setdefault("cancelled", time.monotonic())

    def stage_ms(self, name: str, origin: Optional[float] = None) -> Optional[int]:
        """Offset of a stamp from `origin` (default: the trace start), in ms."""
        with self._lock:
            at = self.stamps.get(name)
        if at is None:
            return None
        base = self.started_at if origin is None else float(origin)
        return int(round((at - base) * 1000.0))

    def snapshot(self, origin: Optional[float] = None) -> dict:
        """JSON-safe view: every stage as ms from `origin`, the call counts, the
        recorded values, and the cancel reason."""
        with self._lock:
            names = list(self.stamps)
            calls = dict(self.calls)
            values = dict(self.values)
            reason = self.cancel_reason
        stages = {}
        for name in names:
            ms = self.stage_ms(name, origin)
            if ms is not None:
                stages[name] = ms
        return {
            "stages": stages,
            "calls": calls,
            "values": values,
            "cancel_reason": reason,
        }


# ── Current-turn plumbing ─────────────────────────────────────────────────────

_current: contextvars.ContextVar[Optional[TurnTrace]] = contextvars.ContextVar(
    "turn_trace", default=None
)
_active_lock = threading.Lock()
_active: Optional[TurnTrace] = None
_totals: dict[str, int] = {}
# token -> trace, so end() knows WHICH turn a token belongs to. A stale token
# (already ended) must be a no-op rather than clearing whatever turn is active now.
_by_token: dict = {}


def begin(turn_id: Optional[int] = None, *, started_at: Optional[float] = None):
    """Start a turn. Returns (trace, token); pass the token to `end()`."""
    global _active
    trace = TurnTrace(turn_id=turn_id)
    if started_at is not None:
        trace.started_at = float(started_at)
    token = _current.set(trace)
    with _active_lock:
        _active = trace
        _by_token[id(token)] = trace   # Tokens are unhashable; the caller holds it until end()
    return trace, token


def end(token) -> None:
    """Finish the turn that `begin()` returned `token` for. Idempotent: a token
    that was already ended does nothing (and never touches a newer turn)."""
    global _active
    with _active_lock:
        trace = _by_token.pop(id(token), None)
    if trace is None:
        return
    try:
        _current.reset(token)
    except Exception:
        # Out-of-order reset (nested begins ended in the wrong order): only drop
        # the context if it still points at THIS trace.
        try:
            if _current.get() is trace:
                _current.set(None)
        except Exception:
            pass
    with _active_lock:
        if _active is trace:
            _active = None


def current() -> Optional[TurnTrace]:
    """The trace for the turn this code is running in, or None between turns.
    Prefers the calling thread's context; background threads fall back to the
    single active turn (see the module docstring)."""
    trace = _current.get()
    if trace is not None:
        return trace
    with _active_lock:
        return _active


def stamp(name: str, when: Optional[float] = None, *, overwrite: bool = False) -> bool:
    trace = current()
    if trace is None:
        return False
    return trace.stamp(name, when, overwrite=overwrite)


def count(kind: str, n: int = 1) -> None:
    """Count a model call. Always adds to the process totals; also to the
    current turn when one is active."""
    count_for(current(), kind, n)


def count_for(trace: Optional[TurnTrace], kind: str, n: int = 1) -> None:
    """Count against an owner captured at dispatch, including late completions."""
    kind = str(kind)
    with _active_lock:
        _totals[kind] = int(_totals.get(kind, 0)) + int(n)
    if trace is not None:
        trace.count(kind, n)


def set_value(name: str, value) -> None:
    trace = current()
    if trace is not None:
        trace.set_value(name, value)


def cancel(reason: str) -> bool:
    """Mark the active turn cancelled. Returns False when no turn is active."""
    trace = current()
    if trace is None:
        return False
    trace.cancel(reason)
    return True


def totals() -> dict:
    with _active_lock:
        return dict(_totals)


def reset_for_tests() -> None:
    global _active
    with _active_lock:
        _active = None
        _totals.clear()
        _by_token.clear()
    try:
        _current.set(None)
    except Exception:
        pass
