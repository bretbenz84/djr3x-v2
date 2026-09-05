"""
intelligence/brain_context.py — the compact conversation snapshot Lean reads.

Lean Brain restructuring, phase 2. One place adapts the state Rex already keeps
into a few prompt lines for `lean_brain._system_prompt` (replies) and the
impulse (`consider_initiating`), instead of each producer inventing its own
block:

- the conversation ARC (intelligence/topic_thread.py) — the hosted running
  summary that until now reached only the classic fallback prompt. Rendered with
  the turn it covers through, and the transcript window is widened to include
  every exact message AFTER that point, so a reference beyond the fixed eight
  turns still lands (the arc has the gist; the recent messages have the words).
- the session's deterministic facts (intelligence/conversation_state.py):
  corrections, body-action outcomes, unanswered questions per target.
- presence notes from consciousness: a face that is gone because REX moved the
  camera is not a departure, a reported departure is, and plain "not on camera
  for N s" is neither.

All reads are local, in-memory, and bounded; nothing here calls a model.
"""

from __future__ import annotations

import logging
from typing import Optional

import config

_log = logging.getLogger(__name__)


def _cfg(name: str, default):
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _entry_turn_id(entry: dict, index: int) -> int:
    try:
        tid = entry.get("turn_id")
        return int(tid) if tid is not None else index + 1
    except (TypeError, ValueError):
        return index + 1


def transcript_window(transcript: Optional[list[dict]], *, base_keep: int,
                      max_keep: Optional[int] = None) -> list[dict]:
    """The recent turns Lean should see verbatim: at least `base_keep`, extended to
    every turn the arc has NOT covered yet (turn_id > covered-through), capped at
    `max_keep`. With no arc (disabled / not yet summarized) the window is simply
    `base_keep`, exactly the old behavior."""
    rows = [t for t in (transcript or []) if str((t or {}).get("text") or "").strip()]
    base_keep = max(0, int(base_keep))
    if max_keep is None:
        max_keep = int(_cfg("LEAN_BRAIN_TRANSCRIPT_TURNS_MAX", 20))
    max_keep = max(base_keep, int(max_keep))
    covered = 0
    try:
        from intelligence import topic_thread
        if topic_thread.arc_summary().strip():
            covered = int(topic_thread.arc_covered_through())
    except Exception:
        covered = 0
    keep = base_keep
    if covered > 0:
        uncovered = sum(1 for i, t in enumerate(rows) if _entry_turn_id(t, i) > covered)
        keep = max(base_keep, min(max_keep, uncovered))
    return rows[-keep:] if keep else []


def arc_lines() -> list[str]:
    """The running conversation summary as ONE prompt line, or [] when absent."""
    try:
        from intelligence import topic_thread
        if not bool(_cfg("CONVERSATION_ARC_ENABLED", True)):
            return []
        summary = topic_thread.arc_summary().strip()
    except Exception:
        return []
    if not summary:
        return []
    flat = " · ".join(s.strip() for s in summary.splitlines() if s.strip())
    return [
        "Your running notes on THIS conversation so far (everything up to the recent "
        "messages below; use them to follow references, avoid repeating a joke or "
        "question, and pick up open threads — never recite them, and if a recent message "
        "or a correction contradicts them, the recent message wins): " + flat
    ]


def presence_lines() -> list[str]:
    try:
        from intelligence import consciousness
        fn = getattr(consciousness, "presence_notes", None)
        return list(fn() or []) if fn else []
    except Exception as exc:
        _log.debug("[brain_context] presence notes skipped: %s", exc)
        return []


def lines(person_id: Optional[int], *, for_reply: bool = True) -> list[str]:
    """Every conversation-state line for one Lean call. `for_reply=False` is the
    impulse path (the arc still applies; pending questions do too)."""
    out: list[str] = []
    try:
        out += arc_lines()
    except Exception as exc:
        _log.debug("[brain_context] arc lines failed: %s", exc)
    try:
        from intelligence import conversation_state
        out += conversation_state.render_lines(person_id)
    except Exception as exc:
        _log.debug("[brain_context] state lines failed: %s", exc)
    try:
        out += presence_lines()
    except Exception as exc:
        _log.debug("[brain_context] presence lines failed: %s", exc)
    return out
