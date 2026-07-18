"""
intelligence/open_threads.py — cross-session "did you ever...?" follow-ups.

The diary extractor (llm.generate_diary_entry, 2026-07-17 rework) stores
open_threads on each conversation_summary episode — the things a person left
unresolved ("whether the dentist appointment happened"). This module is the
CONSUMER: when that person is back and the conversation lulls, Rex asks about
ONE of them. This is the single feature most responsible for "whoa, he
remembered" — everything else in the memory stack feeds it.

Rules:
  * Freshness window: threads younger than OPEN_THREAD_MIN_AGE_HOURS feel
    like Rex forgot you just told him; older than OPEN_THREAD_MAX_AGE_DAYS
    feel like surveillance. Both configurable.
  * A thread is asked AT MOST ONCE, ever: spending rewrites the episode's
    detail JSON (threads_asked) so it survives restarts.
  * Surfacing runs through consciousness (_step_open_thread_followup) at a
    priority ABOVE lull callbacks and news — a personal follow-through beats
    banked humor and headlines.

Fail-safe throughout; reads/writes rex.db via memory.rex_db.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

import config
from memory import rex_db

_log = logging.getLogger(__name__)


def _age_days(created_at: str) -> Optional[float]:
    try:
        dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - dt).total_seconds() / 86400.0
    except Exception:
        return None


def pending_for_person(person_id: int) -> list:
    """Unasked open threads for this person, freshest-first:
    [{"episode_id", "thread", "age_days"}]. Empty when none qualify."""
    if person_id is None:
        return []
    min_age_d = float(getattr(config, "OPEN_THREAD_MIN_AGE_HOURS", 6.0)) / 24.0
    max_age_d = float(getattr(config, "OPEN_THREAD_MAX_AGE_DAYS", 21.0))
    try:
        rows = rex_db.fetchall(
            "SELECT id, created_at, detail FROM rex_episodes "
            "WHERE kind = 'conversation_summary' AND person_id = ? "
            "ORDER BY created_at DESC LIMIT 40",
            (int(person_id),),
        )
    except Exception:
        return []
    out = []
    for row in rows:
        age = _age_days(str(row["created_at"] or ""))
        if age is None or age < min_age_d or age > max_age_d:
            continue
        try:
            detail = json.loads(row["detail"] or "{}")
        except Exception:
            continue
        asked = {str(t).strip().lower() for t in (detail.get("threads_asked") or [])}
        for thread in detail.get("open_threads") or []:
            t = str(thread).strip()
            if t and t.lower() not in asked:
                out.append({"episode_id": int(row["id"]), "thread": t, "age_days": age})
    return out


def mark_asked(episode_id: int, thread: str) -> None:
    """Spend a thread permanently (persisted into the episode's detail JSON)."""
    try:
        row = rex_db.fetchone("SELECT detail FROM rex_episodes WHERE id = ?", (int(episode_id),))
        if row is None:
            return
        try:
            detail = json.loads(row["detail"] or "{}")
        except Exception:
            detail = {}
        asked = list(detail.get("threads_asked") or [])
        if thread not in asked:
            asked.append(thread)
        detail["threads_asked"] = asked
        rex_db.execute(
            "UPDATE rex_episodes SET detail = ? WHERE id = ?",
            (json.dumps(detail, ensure_ascii=False)[:4000], int(episode_id)),
        )
    except Exception as exc:
        _log.debug("[open_threads] mark_asked failed: %s", exc)


def describe_age(age_days: float) -> str:
    """Human phrasing for how long ago the thread was left open."""
    if age_days < 1.0:
        return "earlier today"
    if age_days < 2.0:
        return "yesterday"
    if age_days < 8.0:
        return f"{int(round(age_days))} days ago"
    return "a while back"
