"""
memory/consolidation.py — the retention sweep that keeps the diary distilled.

Curiosity Phase 3 (the mechanical half). The capture side is now high-quality
(2026-07-17 diary rework), but some episode kinds still accumulate structurally:
every sighting of a person logs a person_seen row, every visit a departure row.
Left alone they bury the meaningful entries and degrade recall ranking. This
sweep enforces per-kind retention:

  person_seen           keep the NEWEST row per person per day; rows older than
                        PERSON_SEEN_RETENTION_DAYS (30) are deleted entirely
                        (the fact "I know this person" lives in people.db;
                        the diary only needs the recent rhythm).
  visit_departure       deleted past VISIT_RETENTION_DAYS (90).
  scene                 delegated to episodic_recall.prune() (existing cap).
  room question queue   'pending' objects never asked within
                        ROOM_QUESTION_PENDING_EXPIRY_DAYS (7) are dismissed —
                        the moment passed; don't interrogate next month's guest
                        about last month's box.

Deliberately NO LLM here: distillation-by-summarization stays a future,
cost-gated upgrade (see context.md); this sweep is free and runs at every
shutdown. All operations are fail-safe and test-suppressed via rex_db.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import config
from memory import rex_db

_log = logging.getLogger(__name__)


def _cutoff(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _sweep_person_seen() -> int:
    """Dedup person_seen to one row per person per day, then age out."""
    removed = 0
    try:
        # Keep the newest row per (person_id, date); delete the rest.
        rows = rex_db.fetchall(
            "SELECT id, person_id, date(created_at) AS d FROM rex_episodes "
            "WHERE kind = 'person_seen' ORDER BY created_at DESC"
        )
        keep: set = set()
        doomed = []
        for r in rows:
            key = (r["person_id"], r["d"])
            if key in keep:
                doomed.append(int(r["id"]))
            else:
                keep.add(key)
        for eid in doomed:
            rex_db.execute("DELETE FROM rex_episodes WHERE id = ?", (eid,))
        removed += len(doomed)
        days = float(getattr(config, "PERSON_SEEN_RETENTION_DAYS", 30.0))
        n = rex_db.execute(
            "DELETE FROM rex_episodes WHERE kind = 'person_seen' AND created_at < ?",
            (_cutoff(days),),
        )
        removed += 0 if n is None else 0   # execute returns lastrowid; count via changes below
    except Exception as exc:
        _log.debug("[consolidation] person_seen sweep failed: %s", exc)
    return removed


def _sweep_visit_departures() -> None:
    try:
        days = float(getattr(config, "VISIT_RETENTION_DAYS", 90.0))
        rex_db.execute(
            "DELETE FROM rex_episodes WHERE kind = 'visit_departure' AND created_at < ?",
            (_cutoff(days),),
        )
    except Exception as exc:
        _log.debug("[consolidation] visit sweep failed: %s", exc)


def _expire_stale_pending_questions() -> None:
    try:
        days = float(getattr(config, "ROOM_QUESTION_PENDING_EXPIRY_DAYS", 7.0))
        rex_db.execute(
            "UPDATE room_objects SET ask_status = 'dismissed' "
            "WHERE ask_status = 'pending' AND last_seen < ?",
            (_cutoff(days),),
        )
    except Exception as exc:
        _log.debug("[consolidation] pending-question expiry failed: %s", exc)


def run() -> None:
    """One full retention sweep. Cheap (pure SQL); called at shutdown alongside
    the existing scene prune. Never raises."""
    if rex_db.writes_suppressed():
        return
    before = _count()
    _sweep_person_seen()
    _sweep_visit_departures()
    _expire_stale_pending_questions()
    try:
        from memory import episodic_recall
        episodic_recall.prune()
    except Exception:
        pass
    after = _count()
    if before != after:
        _log.info("[consolidation] retention sweep: %d -> %d episodes", before, after)


def _count() -> int:
    try:
        row = rex_db.fetchone("SELECT COUNT(*) AS n FROM rex_episodes")
        return int(row["n"]) if row else 0
    except Exception:
        return 0
