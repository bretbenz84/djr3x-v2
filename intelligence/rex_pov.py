"""
rex_pov.py - Rex's current preoccupation (a persistent, session-scoped point of view).

The conversational loop is otherwise react -> roast -> question: Rex interviews the
user and answers preference questions reactively, but he rarely VOLUNTEERS his own
substance. This module gives him ONE "current preoccupation" - a concrete thing he's
chewing on right now - that:

  * persists across turns (held for a stretch, clocked on transcript length so it
    actually CARRIES instead of being re-rolled every turn), and
  * is (re)selected with a light HYBRID context bias (room state + conversation mood)
    over a curated, authored seed pool - no LLM call, fully deterministic.

It surfaces two ways:
  1. llm.assemble_system_prompt injects current_pov_directive() so Rex volunteers it
     in normal replies. Because every proactive/idle path calls llm.get_response (which
     routes through assemble_system_prompt), this one injection point also colors them.
  2. the idle-banter "volunteer something of your own" path reads active_pov_text() so
     idle volunteering is the SAME specific preoccupation, not a generic improvised take.

Curated seeds live in config.REX_POV_SEEDS (authored in Rex's voice; deliberately
venue-neutral - no "cantina", since he usually isn't in one). Selection state is
session-scoped and wiped by clear() at session reset. Gated by config.REX_POV_ENABLED.

Unlike the conversation arc in topic_thread.py, this is deterministic and makes no
network call, so it runs directly under the test suite (no cloud-call gate needed).

Related: intelligence/rex_preferences.py holds Rex's STATIC reactive tastes; this is
the dynamic, carrying counterpart.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class _ActivePov:
    seed_id: str
    pov: str
    selected_at_exchange: int          # transcript length when chosen (the hold clock)
    context_sig: frozenset             # context signature at selection time
    surfaced: int = 0                  # times rendered into the prompt (telemetry only)


# Session-scoped state. Read from concurrent reply/proactive threads, mutated on
# (re)selection -> guard everything with a lock (mirrors topic_thread's arc lock).
_lock = threading.Lock()
_active: Optional[_ActivePov] = None
_used_ids: set[str] = set()            # seeds used this session (within-session anti-repeat)


_DIRECTIVE_TEMPLATE = (
    "Your current preoccupation: {pov} You actually care about this right now - so "
    "when there's a natural opening, VOLUNTEER it: bring up your own thing, a take, or "
    "a tangent instead of only reacting to or interviewing them. Work it in like "
    "someone with something on their mind - don't force it, don't announce it as a "
    "topic, and don't raise it every single turn. If they engage with it, run with it. "
    "Keep it in your head - an opinion, a bit, a thing you're chewing on - never invent "
    "a specific object you claim to SEE ('that copier over there', a prop in the room) "
    "to sell it; only point at something real if it's genuinely in front of you."
)


# ─────────────────────────────────────────────────────────────────────────────
# Config accessors (read lazily so config edits / test monkeypatching take effect)
# ─────────────────────────────────────────────────────────────────────────────

def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "REX_POV_ENABLED", True))
    except Exception:
        return False


def _seeds() -> list[dict]:
    """Normalized seed list: [{'id', 'pov', 'fits': (tags...)}]. Skips malformed rows."""
    try:
        import config
        raw = getattr(config, "REX_POV_SEEDS", None) or []
    except Exception:
        return []
    out: list[dict] = []
    for seed in raw:
        if not isinstance(seed, dict):
            continue
        sid = str(seed.get("id") or "").strip()
        pov = str(seed.get("pov") or "").strip()
        if not sid or not pov:
            continue
        fits = seed.get("fits") or ["any"]
        if isinstance(fits, str):
            fits = [fits]
        fits_tuple = tuple(str(f).strip().lower() for f in fits if str(f).strip())
        out.append({"id": sid, "pov": pov, "fits": fits_tuple or ("any",)})
    return out


def _min_hold() -> int:
    try:
        import config
        return max(0, int(getattr(config, "REX_POV_MIN_HOLD_EXCHANGES", 4) or 0))
    except Exception:
        return 4


def _max_hold() -> int:
    try:
        import config
        value = int(getattr(config, "REX_POV_MAX_HOLD_EXCHANGES", 12) or 0)
    except Exception:
        value = 12
    return max(_min_hold(), value)


# ─────────────────────────────────────────────────────────────────────────────
# Live signal readers (overridable via injected `context` for tests)
# ─────────────────────────────────────────────────────────────────────────────

def _exchange_count() -> int:
    """The hold clock: number of transcript entries so far. Ticks per spoken line
    (~2 per back-and-forth), monotonic within a session, 0 when unavailable. Mirrors
    how topic_thread's arc clocks itself on transcript length."""
    try:
        from memory import conversations
        return len(conversations.get_session_transcript() or [])
    except Exception:
        return 0


def _people_present() -> bool:
    try:
        from world_state import world_state
        people = world_state.snapshot().get("people") or []
        return any((p or {}).get("person_db_id") for p in people)
    except Exception:
        return False


def _arc_flat() -> bool:
    try:
        from intelligence import topic_thread
        return bool(topic_thread.arc_reads_flat())
    except Exception:
        return False


def _context_signature(context: Optional[dict] = None) -> frozenset:
    """Small context tag-set used for hybrid selection. Reuses existing signals:
    visible people (world_state) and whether the arc reads the room as falling flat
    (topic_thread.arc_reads_flat). Tests inject: {'people': bool, 'flat': bool}."""
    if context is not None:
        people = bool(context.get("people"))
        flat = bool(context.get("flat"))
    else:
        people = _people_present()
        flat = _arc_flat()
    tags = {"people"} if people else {"quiet"}
    if flat:
        tags.add("flat")
    return frozenset(tags)


# ─────────────────────────────────────────────────────────────────────────────
# Selection + holding policy
# ─────────────────────────────────────────────────────────────────────────────

def _choose(
    context_sig: frozenset,
    exchange: int,
    used: set[str],
    current_id: Optional[str] = None,
) -> Optional[dict]:
    """Pick the best-fitting seed for the context, deterministically (no random()).

    Score = number of the seed's `fits` tags present in context_sig ("any" matches
    nothing but never disqualifies). Excludes already-used ids and the current one;
    if that empties the field (anti-repeat cycle complete mid-call), recycles to all
    non-current seeds. Ties broken by a fixed rotation keyed on the exchange clock, so
    selection varies across re-selections while staying reproducible for tests.
    """
    seeds = _seeds()
    if not seeds:
        return None

    candidates = [s for s in seeds if s["id"] not in used and s["id"] != current_id]
    if not candidates:
        candidates = [s for s in seeds if s["id"] != current_id] or list(seeds)

    def score(seed: dict) -> int:
        return sum(1 for tag in seed["fits"] if tag in context_sig)

    best = max(score(s) for s in candidates)
    top = [s for s in candidates if score(s) == best]
    return top[exchange % len(top)]


def _select(context_sig: frozenset, exchange: int) -> Optional[_ActivePov]:
    """Choose and install a new active POV. Caller must hold _lock."""
    global _active, _used_ids
    current_id = _active.seed_id if _active else None
    # Restart the within-session anti-repeat cycle once every seed has been used.
    if len(_used_ids) >= len(_seeds()):
        _used_ids = set()
    seed = _choose(context_sig, exchange, _used_ids, current_id)
    if seed is None:
        return _active
    _used_ids.add(seed["id"])
    _active = _ActivePov(
        seed_id=seed["id"],
        pov=seed["pov"],
        selected_at_exchange=exchange,
        context_sig=context_sig,
    )
    _log.info(
        "[rex_pov] selected %r (context=%s exchange=%d)",
        seed["id"], sorted(context_sig), exchange,
    )
    return _active


def _ensure_active(context: Optional[dict], exchange: Optional[int]) -> Optional[_ActivePov]:
    """Apply the holding policy and return the active POV. Caller must hold _lock.

    - nothing active            -> select now
    - held < MIN_HOLD           -> keep (it must CARRY; don't thrash)
    - held >= MIN and (context changed materially OR held >= MAX_HOLD) -> re-select
    """
    if exchange is None:
        exchange = _exchange_count()
    context_sig = _context_signature(context)

    if _active is None:
        return _select(context_sig, exchange)

    held = exchange - _active.selected_at_exchange
    if held < _min_hold():
        return _active
    if held >= _max_hold() or context_sig != _active.context_sig:
        return _select(context_sig, exchange)
    return _active


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def current_pov_directive(
    context: Optional[dict] = None,
    exchange: Optional[int] = None,
) -> str:
    """Canonical accessor: (re)select per the holding policy and render the
    system-prompt line. Returns '' when disabled or the seed pool is empty. Safe to
    call from concurrent reply/proactive threads. `context`/`exchange` are injectable
    for tests; both default to live sources."""
    if not _enabled():
        return ""
    with _lock:
        active = _ensure_active(context, exchange)
        if active is None:
            return ""
        active.surfaced += 1
        pov = active.pov
    pov = pov if pov.endswith((".", "!", "?")) else pov + "."
    return _DIRECTIVE_TEMPLATE.format(pov=pov)


def active_pov_text(
    context: Optional[dict] = None,
    exchange: Optional[int] = None,
) -> str:
    """The active preoccupation text alone (for the idle volunteer path). Selects one
    if nothing is active yet (so idle volunteering still gets a concrete POV even when
    it fires before the first reply), but otherwise piggybacks on whatever the reply
    path already chose. Returns '' when disabled / pool empty."""
    if not _enabled():
        return ""
    with _lock:
        active = _ensure_active(context, exchange)
        return active.pov if active else ""


def active_seed_id() -> Optional[str]:
    """The active seed's id, or None. Pure read (no selection) - for tests/telemetry."""
    with _lock:
        return _active.seed_id if _active else None


def clear() -> None:
    """Wipe session-scoped POV state. Called from the session-reset bundle."""
    global _active, _used_ids
    with _lock:
        _active = None
        _used_ids = set()
