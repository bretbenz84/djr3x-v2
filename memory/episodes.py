"""
episodes.py — Rex's first-person episodic memory (his "diary").

CAPTURE (this module): the capture hooks log salient world events to rex.db with
timestamps — people seen, things he did ("I made Bret laugh", "I saw a dog"), scenes
("the room was cluttered"), and a session summary on shutdown. Capture is gated by
`config.EPISODIC_MEMORY_ENABLED`.

RECALL (Phase 2 — IMPLEMENTED + ENABLED): `memory/episodic_recall.py` reads these
rows back into Rex's behavior, gated by the SEPARATE `config.EPISODIC_RECALL_ENABLED`
switch (default on). It feeds (a) the per-person SHARED-MEMORY hook in the reply
prompt (`intelligence/llm.py` `_pick_episodic_callback` → "I made you laugh", "we
played trivia") and (b) the idle "memory musing" behavior (`intelligence/
idle_behaviors.py`). Keep capture and recall as two independent kill switches so the
pool can build silently for A/B runs.

Every public writer is GATED:
  • `config.EPISODIC_MEMORY_ENABLED` kill switch, and
  • `rex_db.writes_suppressed()` — under the test runner on the default path, writes
    no-op, so the suite never creates/populates a real rex.db.
…and failure-safe (swallow + log; the robot must never crash because its diary
hiccuped). Person ids are SOFT references to people.db (a separate DB) — a name
snapshot is stored alongside.

A read API (`recent_episodes`, `episodes_on_date`, `count`) exists for Phase-2
exploration; it is NOT wired into any prompt/behavior.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Optional

from memory import rex_db

_log = logging.getLogger(__name__)

# One id per process run, so Phase 2 can group "a run's" episodes. Stamped lazily.
_session_id: Optional[str] = None
_session_lock = threading.Lock()
# The DB path the schema was last ensured for. PATH-AWARE (not a bare bool) so a test
# that swaps REX_DB_PATH to a fresh temp DB re-creates the schema there instead of
# wrongly assuming the previous DB's schema applies.
_schema_ready_path: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _session() -> str:
    global _session_id
    with _session_lock:
        if _session_id is None:
            _session_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        return _session_id


def reset_session(session_id: Optional[str] = None) -> None:
    """Test/diagnostic hook: force the session id (or re-stamp on next use)."""
    global _session_id
    with _session_lock:
        _session_id = session_id


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "EPISODIC_MEMORY_ENABLED", True))
    except Exception:
        return False


def _suppressed() -> bool:
    # Skip ALL capture when disabled OR when the test runner would otherwise write a
    # real rex.db on the default path.
    return (not _enabled()) or rex_db.writes_suppressed()


def ensure_ready() -> None:
    """Create rex.db + schema if needed (call once at startup). Gated + idempotent.
    Path-aware so swapping REX_DB_PATH (tests) re-ensures the new DB's schema."""
    global _schema_ready_path
    if _suppressed():
        return
    current = str(rex_db.db_path())
    if _schema_ready_path == current:
        return
    rex_db.ensure_schema()
    _schema_ready_path = current


def record_episode(
    kind: str,
    summary: str,
    *,
    person_id: Optional[int] = None,
    person_name: Optional[str] = None,
    detail: Optional[dict] = None,
    salience: float = 0.5,
) -> Optional[int]:
    """Log one episode. Returns the new row id, or None when gated/failed.

    `summary` should read first-person and human ("I saw a dog", "I made Bret laugh").
    `detail` is an optional structured payload, stored as JSON for Phase-2 use.
    """
    if _suppressed():
        return None
    summary = (summary or "").strip()
    if not summary:
        return None
    try:
        ensure_ready()
        payload = None
        if detail is not None:
            try:
                payload = json.dumps(detail, ensure_ascii=False)[:4000]
            except Exception:
                payload = None
        return rex_db.execute(
            "INSERT INTO rex_episodes "
            "(created_at, kind, summary, person_id, person_name, detail, salience, session_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                _now_iso(), str(kind), summary[:2000],
                int(person_id) if isinstance(person_id, int) else None,
                (str(person_name).strip() or None) if person_name else None,
                payload, float(salience), _session(),
            ),
        )
    except Exception as exc:
        _log.debug("record_episode failed: %s", exc)
        return None


# ── Convenience capture helpers (what the hooks call) ───────────────────────────

def record_person_seen(person_id: Optional[int], name: Optional[str]) -> Optional[int]:
    label = (name or "someone").strip() or "someone"
    return record_episode(
        "person_seen", f"I saw {label}.",
        person_id=person_id, person_name=name, salience=0.45,
    )


def record_made_laugh(person_id: Optional[int], name: Optional[str], *, kind: str = "smile") -> Optional[int]:
    label = (name or "them").strip() or "them"
    verb = "laugh" if kind in ("laugh", "laughing") else "smile"
    return record_episode(
        "made_laugh", f"I made {label} {verb}.",
        person_id=person_id, person_name=name,
        detail={"expression": kind}, salience=0.7,
    )


def record_animal(species: Optional[str], *, position: Optional[str] = None) -> Optional[int]:
    sp = (species or "creature").strip() or "creature"
    article = "an" if sp[:1].lower() in "aeiou" else "a"
    return record_episode(
        "animal", f"I saw {article} {sp}.",
        detail={"species": sp, "position": position}, salience=0.6,
    )


def record_scene(summary: str, *, detail: Optional[dict] = None) -> Optional[int]:
    return record_episode("scene", summary, detail=detail, salience=0.4)


def record_conversation_summary(
    summary: str, *, people: Optional[list] = None, salience: float = 0.8,
) -> Optional[int]:
    primary = (people or [{}])[0] if people else {}
    return record_episode(
        "conversation_summary", summary,
        person_id=primary.get("person_id") if isinstance(primary, dict) else None,
        person_name=primary.get("name") if isinstance(primary, dict) else None,
        detail={"people": people} if people else None,
        salience=salience,
    )


# ── Batch-2 convenience helpers (people/relationship/activity milestones) ────────

def record_person_enrolled(person_id: Optional[int], name: Optional[str]) -> Optional[int]:
    label = (name or "").strip() or "someone new"
    return record_episode(
        "person_enrolled", f"I met {label}.",
        person_id=person_id, person_name=name, salience=0.8,
    )


def record_game_played(
    game: str, outcome: str = "", *, person_id: Optional[int] = None,
    person_name: Optional[str] = None, detail: Optional[dict] = None,
) -> Optional[int]:
    who = (person_name or "").strip()
    with_who = f" with {who}" if who else ""
    tail = f" — {outcome}" if outcome else ""
    return record_episode(
        "game_played", f"I played {game}{with_who}{tail}.",
        person_id=person_id, person_name=person_name, detail=detail, salience=0.6,
    )


def _format_duration(secs: float) -> str:
    secs = max(0.0, float(secs or 0.0))
    mins = secs / 60.0
    if mins < 1.5:
        return "a minute"
    if mins < 50:
        return f"about {int(round(mins))} minutes"
    hours = mins / 60.0
    if hours < 1.25:
        return "about an hour"
    if hours < 1.75:
        return "about an hour and a half"
    return f"about {hours:.1f} hours".replace(".0 ", " ")


def record_visit_departure(
    person_id: Optional[int], name: Optional[str], duration_secs: float,
    *, detail: Optional[dict] = None,
) -> Optional[int]:
    # Skip fleeting glimpses — a real "visit" is worth remembering, a 10-second
    # pass-through is noise.
    if (duration_secs or 0) < 60:
        return None
    who = (name or "").strip() or "someone"
    return record_episode(
        "visit_departure", f"I spent {_format_duration(duration_secs)} with {who}.",
        person_id=person_id, person_name=name, detail=detail, salience=0.55,
    )


def record_boundary(
    person_id: Optional[int], behavior: str, topic: str, action: str,
    *, person_name: Optional[str] = None,
) -> Optional[int]:
    topic = (topic or "that").strip() or "that"
    behavior = (behavior or "bring up").strip() or "bring up"
    who = (person_name or "").strip()
    subj = who or "Someone"
    if action == "clear":
        summary = f"{subj} said it's okay to {behavior} about {topic} again."
    else:
        summary = f"{subj} asked me not to {behavior} about {topic}."
    return record_episode(
        "boundary", summary, person_id=person_id, person_name=person_name,
        detail={"behavior": behavior, "topic": topic, "action": action}, salience=0.7,
    )


def record_celebrity(
    person_id: Optional[int], name: Optional[str], celebrity: str, *, returning: bool = False,
) -> Optional[int]:
    who = (name or "").strip() or "a celebrity"
    verb = "saw" if returning else "met"
    return record_episode(
        "celebrity", f"I {verb} {who}.",
        person_id=person_id, person_name=name,
        detail={"celebrity": celebrity, "returning": bool(returning)}, salience=0.75,
    )


def record_checkin(
    person_id: Optional[int], name: Optional[str], summary: str, *, detail: Optional[dict] = None,
) -> Optional[int]:
    """An empathy check-in / heavy moment. Caller builds the (sensitive) summary."""
    return record_episode(
        "emotional_checkin", summary, person_id=person_id, person_name=name,
        detail=detail, salience=0.78,
    )


def record_greeting_event(
    kind: str, summary: str, *, person_id: Optional[int] = None,
    person_name: Optional[str] = None, detail: Optional[dict] = None,
) -> Optional[int]:
    """A memorable first-sight greeting tier (birthday/milestone/celebration/reunion).
    `kind` is one of: birthday_wish | milestone | celebration | reunion."""
    return record_episode(
        kind, summary, person_id=person_id, person_name=person_name,
        detail=detail, salience=0.65,
    )


# ── Read API — for PHASE-2 exploration only (NOT wired into behavior) ────────────

def recent_episodes(limit: int = 50, *, kind: Optional[str] = None) -> list:
    if kind:
        return rex_db.fetchall(
            "SELECT * FROM rex_episodes WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
            (kind, int(limit)),
        )
    return rex_db.fetchall(
        "SELECT * FROM rex_episodes ORDER BY created_at DESC LIMIT ?", (int(limit),)
    )


def episodes_on_date(yyyy_mm_dd: str, limit: int = 200) -> list:
    return rex_db.fetchall(
        "SELECT * FROM rex_episodes WHERE date(created_at) = ? ORDER BY created_at ASC LIMIT ?",
        (yyyy_mm_dd, int(limit)),
    )


def count() -> int:
    row = rex_db.fetchone("SELECT COUNT(*) AS n FROM rex_episodes")
    try:
        return int(row["n"]) if row is not None else 0
    except Exception:
        return 0
