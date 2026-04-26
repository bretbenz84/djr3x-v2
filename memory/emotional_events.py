"""
memory/emotional_events.py — CRUD for person_emotional_events.

Sensitive life events (grief, breakup, job loss, illness, milestones) Rex
should remember across sessions and acknowledge appropriately. Distinct from
person_events (planned/upcoming things) — access patterns differ.

Events do not delete; they decay. `sensitivity_decay_days` controls when Rex
stops *leading with* the event on a return visit. The row stays in the DB
forever so older context is available if a person brings it up themselves.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import config
from memory import database as db

_log = logging.getLogger(__name__)


# Default decay windows per category, in days. After this much time since
# `mentioned_at`, Rex stops surfacing the event proactively (it can still be
# recalled if the person mentions it). Override by passing an explicit
# sensitivity_decay_days at insert time.
_DEFAULT_DECAY_DAYS = {
    "grief":      180,
    "death":      180,
    "breakup":    90,
    "divorce":    120,
    "illness":    120,
    "health":     120,
    "job_loss":   60,
    "layoff":     60,
    "fired":      60,
    "move":       30,
    "promotion":  365,
    "new_baby":   365,
    "engagement": 365,
    "wedding":    365,
    "graduation": 365,
    "achievement": 180,
    "good_news":  90,
    "celebration": 90,
    "birthday":   30,
    "bad_day":    1,
    "work_stress": 1,
    "stress":     1,
    "default":    60,
}

_MILD_NEGATIVE_CATEGORIES = {"bad_day", "work_stress", "stress"}

_HEAVY_NEGATIVE_CATEGORIES = {
    "grief", "death", "breakup", "divorce", "illness", "health",
    "job_loss", "layoff", "fired",
}

_CELEBRATION_CATEGORIES = {
    "promotion", "new_baby", "engagement", "wedding", "graduation",
    "achievement", "good_news", "celebration", "birthday",
}

_HIGH_INTENSITY_LOSS_SUBJECTS = {
    "mom", "mother", "dad", "father", "parent",
    "son", "daughter", "child", "kid", "baby",
    "wife", "husband", "spouse", "partner",
}

_FAMILY_LOSS_SUBJECTS = {
    "grandma", "grandmother", "grandpa", "grandfather", "grandparent",
    "brother", "sister", "sibling",
}


def decay_days_for(category: str) -> int:
    """Return the default decay window for a category."""
    cat = (category or "").strip().lower()
    return int(_DEFAULT_DECAY_DAYS.get(cat, _DEFAULT_DECAY_DAYS["default"]))


def is_celebration_event(event: dict) -> bool:
    """True when an event is positive enough to deserve a happy callback."""
    try:
        valence = float(event.get("valence", 0.0) or 0.0)
    except (TypeError, ValueError):
        valence = 0.0
    category = (event.get("category") or "").strip().lower()
    return valence > 0 or category in _CELEBRATION_CATEGORIES


def is_heavy_event(event: dict) -> bool:
    """True when Rex should only raise this again if the person invited it."""
    try:
        valence = float(event.get("valence", 0.0) or 0.0)
    except (TypeError, ValueError):
        valence = 0.0
    category = (event.get("category") or "").strip().lower()
    if category in _MILD_NEGATIVE_CATEGORIES:
        return False
    return valence < 0 or category in _HEAVY_NEGATIVE_CATEGORIES


def _can_surface_event(event: dict) -> bool:
    """Return whether proactive/prompt callbacks may raise this event."""
    if not is_heavy_event(event):
        return True
    return bool(event.get("person_invited_topic"))


def can_surface_event(event: dict) -> bool:
    """Public wrapper for prompt/conversation layers."""
    return _can_surface_event(event)


def decay_days_for_event(
    category: str,
    *,
    loss_subject: Optional[str] = None,
    loss_subject_kind: Optional[str] = None,
    valence: Optional[float] = None,
    description: str = "",
) -> int:
    """Return a surfacing window calibrated by event severity.

    Rows never delete; this only controls how long Rex proactively checks in.
    A parent/child/partner loss should stay tender much longer than generic
    bad-day venting, which should cool down after roughly 24 hours.
    """
    cat = (category or "").strip().lower()
    subject = (loss_subject or "").strip().lower()
    kind = (loss_subject_kind or "").strip().lower()
    desc = (description or "").strip().lower()

    if cat in {"bad_day", "work_stress", "stress"}:
        return 1

    if cat in {"grief", "death"}:
        if kind == "pet" or any(word in subject or word in desc for word in ("dog", "cat", "pet")):
            return 120
        if subject in _HIGH_INTENSITY_LOSS_SUBJECTS:
            return 365
        if subject in _FAMILY_LOSS_SUBJECTS:
            return 180
        return decay_days_for(cat)

    # Strongly negative life disruptions stay active longer than minor stress.
    if valence is not None and float(valence) <= -0.8:
        return max(decay_days_for(cat), 90)

    return decay_days_for(cat)


def add_event(
    person_id: int,
    category: str,
    description: str,
    valence: float = -0.5,
    sensitivity_decay_days: Optional[int] = None,
    person_invited_topic: bool = True,
    loss_subject: Optional[str] = None,
    loss_subject_kind: Optional[str] = None,
    loss_subject_name: Optional[str] = None,
) -> Optional[int]:
    """Insert a new emotional event row. Returns lastrowid or None on failure.

    De-duplication: if a row already exists for this person with the same
    category and a near-duplicate description within the last 7 days, skip.
    """
    cat = (category or "other").strip().lower()
    desc = (description or "").strip()
    if not desc:
        return None
    subject = (loss_subject or "").strip() or None
    subject_kind = (loss_subject_kind or "").strip().lower() or None
    subject_name = (loss_subject_name or "").strip() or None

    existing = db.fetchall(
        "SELECT id, description FROM person_emotional_events "
        "WHERE person_id = ? AND category = ? "
        "AND mentioned_at >= datetime('now', '-7 days')",
        (person_id, cat),
    )
    for row in existing:
        if (row["description"] or "").strip().lower() == desc.lower():
            return row["id"]

    decay = (
        int(sensitivity_decay_days)
        if sensitivity_decay_days is not None
        else decay_days_for_event(
            cat,
            loss_subject=loss_subject,
            loss_subject_kind=loss_subject_kind,
            valence=valence,
            description=desc,
        )
    )

    return db.execute(
        "INSERT INTO person_emotional_events "
        "(person_id, category, valence, description, "
        " loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        " sensitivity_decay_days, person_invited_topic) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)",
        (
            person_id,
            cat,
            float(valence),
            desc,
            subject,
            subject_kind,
            subject_name,
            decay,
            1 if person_invited_topic else 0,
        ),
    )


def get_active_events(person_id: int, limit: int = 3) -> list[dict]:
    """Return events whose decay window has not yet elapsed, newest first."""
    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, "
        "       person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "AND checkins_muted_at IS NULL "
        "ORDER BY mentioned_at DESC LIMIT ?",
        (person_id, int(limit)),
    )
    return [dict(r) for r in rows]


def get_unacknowledged_since(person_id: int, since_iso: Optional[str]) -> list[dict]:
    """Active events not yet acknowledged since the given ISO timestamp.

    Pass None for `since_iso` to mean 'since beginning of time'. Used to decide
    whether Rex should open with a soft acknowledgment of a recent loss.
    """
    active = get_active_events(person_id, limit=10)
    out = []
    for ev in active:
        last_ack = ev.get("last_acknowledged_at")
        if not last_ack:
            out.append(ev)
            continue
        if since_iso and last_ack < since_iso:
            out.append(ev)
    return out


def get_due_checkins(
    person_id: int,
    limit: int = 3,
    min_ack_gap_days: int = 1,
) -> list[dict]:
    """Return active negative events due for a soft check-in.

    Acknowledgment is persisted, but check-ins can repeat gently across days
    while the event remains active. This means a recent death can be surfaced
    on a later boot, while a bad day at work naturally expires after ~24h.
    """
    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, "
        "       person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence < 0 "
        "AND (person_invited_topic = 1 OR category IN ('bad_day', 'work_stress', 'stress')) "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "AND checkins_muted_at IS NULL "
        "AND (last_acknowledged_at IS NULL "
        "     OR datetime(last_acknowledged_at, '+' || ? || ' days') <= datetime('now')) "
        "ORDER BY mentioned_at DESC LIMIT ?",
        (int(person_id), int(min_ack_gap_days), int(limit)),
    )
    return [dict(r) for r in rows]


def get_due_celebrations(
    person_id: int,
    limit: int = 2,
    min_ack_gap_days: int = 7,
) -> list[dict]:
    """Return active positive events due for a light celebratory check-in."""
    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, "
        "       person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence > 0 "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "AND checkins_muted_at IS NULL "
        "AND (last_acknowledged_at IS NULL "
        "     OR datetime(last_acknowledged_at, '+' || ? || ' days') <= datetime('now')) "
        "ORDER BY mentioned_at DESC LIMIT ?",
        (int(person_id), int(min_ack_gap_days), int(limit)),
    )
    return [dict(r) for r in rows]


def get_startup_checkins(
    person_id: int,
    process_started_iso: Optional[str],
    limit: int = 3,
) -> list[dict]:
    """Return active negative events that should lead a fresh process greeting.

    Startup is a different social moment from an in-session check-in. If Rex
    acknowledged a loss during the original disclosure, he should still check
    in on a later boot while the event is active. Once startup fires and marks
    the event acknowledged again, this query stops returning it for the rest of
    the same process.
    """
    if process_started_iso:
        ack_clause = (
            "AND (last_acknowledged_at IS NULL OR last_acknowledged_at < ?)"
        )
        params = (int(person_id), process_started_iso, int(limit))
    else:
        ack_clause = ""
        params = (int(person_id), int(limit))

    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, "
        "       person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence < 0 "
        "AND (person_invited_topic = 1 OR category IN ('bad_day', 'work_stress', 'stress')) "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "AND checkins_muted_at IS NULL "
        f"{ack_clause} "
        "ORDER BY mentioned_at DESC LIMIT ?",
        params,
    )
    return [dict(r) for r in rows]


def get_startup_celebrations(
    person_id: int,
    process_started_iso: Optional[str],
    limit: int = 1,
) -> list[dict]:
    """Return positive events that can lead a fresh-process greeting."""
    if process_started_iso:
        ack_clause = (
            "AND (last_acknowledged_at IS NULL OR last_acknowledged_at < ?)"
        )
        params = (int(person_id), process_started_iso, int(limit))
    else:
        ack_clause = ""
        params = (int(person_id), int(limit))

    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, "
        "       person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence > 0 "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "AND checkins_muted_at IS NULL "
        f"{ack_clause} "
        "ORDER BY mentioned_at DESC LIMIT ?",
        params,
    )
    return [dict(r) for r in rows]


def mark_acknowledged(event_id: int) -> None:
    db.execute(
        "UPDATE person_emotional_events SET last_acknowledged_at = datetime('now') "
        "WHERE id = ?",
        (int(event_id),),
    )


def mute_checkins(event_id: int, reason: str = "") -> None:
    """Stop proactive check-ins for one event without deleting the memory."""
    db.execute(
        "UPDATE person_emotional_events "
        "SET checkins_muted_at = datetime('now'), checkins_muted_reason = ? "
        "WHERE id = ?",
        ((reason or "").strip(), int(event_id)),
    )


def mute_recent_checkin_for_person(
    person_id: int,
    reason: str = "",
    window_minutes: int = 15,
) -> Optional[dict]:
    """
    Mute the most recent active negative event Rex acknowledged recently.

    This is the consent-boundary path for replies like "I'd rather not talk
    about it" after Rex has proactively checked in. Returns the muted event.
    """
    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence < 0 "
        "AND checkins_muted_at IS NULL "
        "AND last_acknowledged_at IS NOT NULL "
        "AND datetime(last_acknowledged_at, '+' || ? || ' minutes') >= datetime('now') "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "ORDER BY last_acknowledged_at DESC, mentioned_at DESC LIMIT 1",
        (int(person_id), int(window_minutes)),
    )
    if not rows:
        return None
    event = dict(rows[0])
    mute_checkins(int(event["id"]), reason=reason)
    return event


def mute_latest_active_negative_for_person(
    person_id: int,
    reason: str = "",
) -> Optional[dict]:
    """Mute the newest active negative event for explicit in-flow boundaries."""
    rows = db.fetchall(
        "SELECT id, category, valence, description, "
        "       loss_subject, loss_subject_kind, loss_subject_name, mentioned_at, "
        "       last_acknowledged_at, checkins_muted_at, checkins_muted_reason, "
        "       sensitivity_decay_days, person_invited_topic "
        "FROM person_emotional_events "
        "WHERE person_id = ? "
        "AND valence < 0 "
        "AND checkins_muted_at IS NULL "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now') "
        "ORDER BY mentioned_at DESC LIMIT 1",
        (int(person_id),),
    )
    if not rows:
        return None
    event = dict(rows[0])
    mute_checkins(int(event["id"]), reason=reason)
    return event


def mark_all_acknowledged_for_person(person_id: int) -> None:
    """Convenience: mark every active event acknowledged in one shot.

    Called after Rex opens the interaction with a soft acknowledgment so we
    don't repeat the same opening across consecutive turns within a session.
    """
    db.execute(
        "UPDATE person_emotional_events SET last_acknowledged_at = datetime('now') "
        "WHERE person_id = ? "
        "AND checkins_muted_at IS NULL "
        "AND datetime(mentioned_at, '+' || sensitivity_decay_days || ' days') >= datetime('now')",
        (int(person_id),),
    )


def summarize_for_prompt(
    person_id: int,
    crowd_count: int = 1,
) -> str:
    """Render active emotional events as a prompt-ready block.

    Discretion rule: when more than one person is in the scene, suppress
    sensitive callbacks — grief, illness, job loss, etc. — so Rex doesn't air
    private context in front of bystanders. Positive events may still appear.
    """
    if not getattr(config, "EMPATHY_ENABLED", True):
        return ""
    suppress_in_crowd = bool(getattr(config, "EMPATHY_DISCRETION_IN_CROWD", True))
    events = [
        ev for ev in get_active_events(person_id, limit=6)
        if _can_surface_event(ev)
        and not (suppress_in_crowd and crowd_count > 1 and is_heavy_event(ev))
    ][:3]
    if not events:
        return ""

    lines = []
    now = datetime.utcnow()
    for ev in events:
        try:
            mentioned = datetime.fromisoformat(ev["mentioned_at"].replace("Z", ""))
            days_ago = max(0, (now - mentioned).days)
        except Exception:
            days_ago = None
        when = (
            f"{days_ago} days ago" if days_ago is not None and days_ago > 0
            else "recently"
        )
        ack = ev.get("last_acknowledged_at")
        ack_clause = " (already acknowledged this session)" if ack else " (not yet acknowledged on this return)"
        tone = "celebration" if is_celebration_event(ev) else "sensitive"
        lines.append(
            f"- {tone}/{ev['category']}: {ev['description']} (mentioned {when}){ack_clause}"
        )
    return "Recent emotional/social events for this person:\n" + "\n".join(lines)
