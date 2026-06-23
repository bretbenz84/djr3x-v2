"""
memory/events.py — Upcoming events and follow-up tracking (person_events table).
"""

import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from memory import database as db

_log = logging.getLogger(__name__)

_CANCEL_PAT = re.compile(
    r"\b("
    r"not going|not gonna|not doing|not happening|"
    r"no longer going|no longer doing|"
    r"cancel(?:ed|led|s|ing)?|called off|scrubbed|"
    r"can'?t make it|won'?t make it|not anymore|not any more|"
    r"changed my mind|skip(?:ping)? it"
    r")\b",
    re.IGNORECASE,
)
# Postponement / reschedule — a DIFFERENT outcome from cancellation: the plan still
# exists, just at a new (often unknown) time. Must NOT cancel the event (that durably
# loses it); instead it's kept open + re-dated (see reschedule_event / looks_like_postponement).
_POSTPONE_PAT = re.compile(
    r"\b("
    r"postpone(?:d|s|ment)?|reschedul(?:e|ed|es|ing)?|"
    r"push(?:ed|ing)?\s+(?:it\s+|them\s+|that\s+)?back|"
    r"put\s+(?:it\s+|them\s+|that\s+)?off|"
    r"mov(?:e|ed|ing)\s+(?:it|them|that)\s+to|"
    r"new\s+date|rain\s?check(?:ed)?"
    r")\b",
    re.IGNORECASE,
)
_TOKEN_PAT = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "for", "from", "going", "i",
    "im", "i'm", "it", "my", "not", "of", "on", "or", "our", "the",
    "this", "that", "to", "we", "you", "anymore", "any", "more",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_local() -> str:
    """Return the robot host's local calendar date as YYYY-MM-DD."""
    return date.today().isoformat()


def _undated_followup_cutoff() -> str:
    days = int(getattr(config, "FOLLOWUP_UNDATED_DAYS", 7))
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


# Loose negations that match _CANCEL_PAT but do NOT cancel the event: idioms
# ("not going to lie/forget"), positive status ("not doing too bad"), and
# ongoing-travel phrases ("on my way"). Mirrors the action router's continuation
# guard so a conversational outcome reply can't silently cancel a remembered
# event before the dialogue gate ever runs.
_NOT_A_CANCELLATION_PAT = re.compile(
    r"\bnot\s+(?:going|gonna)\s+to\s+"
    r"(?:lie|forget|miss|deny|say|tell|pretend|kid|sugar-?coat|complain)\b"
    r"|\bnot\s+doing\s+(?:too|so|that|this|pretty|very|real(?:ly)?|all\s+that|half|bad)\b"
    r"|\bon\s+(?:my|our|the)\s+way\b",
    re.IGNORECASE,
)


def looks_like_cancellation(text: str) -> bool:
    """Return True when text likely cancels or retracts a planned event.

    Requires a cancellation phrase AND the absence of a known false-positive
    idiom / ongoing-status phrase, so a conversational outcome reply such as
    "not going to lie, it was amazing" or "I'm not doing too bad" is not treated
    as a cancellation (which would durably mark the event canceled).
    """
    text = text or ""
    if not _CANCEL_PAT.search(text):
        return False
    if _NOT_A_CANCELLATION_PAT.search(text):
        return False
    return True


def looks_like_postponement(text: str) -> bool:
    """Return True when text reschedules (not cancels) a planned event.

    A postponement keeps the event OPEN; the caller should reschedule_event() it
    rather than cancel it. Shares the false-positive guard with cancellation so
    "on my way to the postponed meetup" isn't treated as a reschedule signal."""
    text = text or ""
    if not _POSTPONE_PAT.search(text):
        return False
    if _NOT_A_CANCELLATION_PAT.search(text):
        return False
    return True


def _tokens(text: str) -> set[str]:
    return {
        t.strip("'").lower()
        for t in _TOKEN_PAT.findall(text or "")
        if len(t.strip("'")) >= 3 and t.strip("'").lower() not in _STOPWORDS
    }


def _event_tokens(event: dict) -> set[str]:
    return _tokens(
        " ".join([
            str(event.get("event_name") or ""),
            str(event.get("event_notes") or ""),
        ])
    )


def add_event(
    person_id: int,
    event_name: str,
    event_date: Optional[str],
    event_notes: str,
) -> Optional[int]:
    """Store an upcoming event. event_date may be None if no specific date was given.

    A planned event that matches an existing OPEN one (same/paraphrased name, via
    memory.dedup) refreshes that row instead of inserting a duplicate — so mentioning
    "the camping trip" across several turns doesn't leave four rows Rex re-asks about.
    """
    now = _now()
    try:
        from memory import dedup
        open_events = get_open_events(int(person_id))
        match = dedup.event_match(event_name or "", open_events)
    except Exception:
        match = None
    if match and match.get("id") is not None:
        db.execute(
            """UPDATE person_events
               SET event_date = COALESCE(?, event_date),
                   event_notes = COALESCE(NULLIF(?, ''), event_notes),
                   mentioned_at = ?, updated_at = ?
               WHERE id = ?""",
            (event_date, (event_notes or "").strip(), now, now, int(match["id"])),
        )
        return int(match["id"])
    return db.execute(
        """INSERT INTO person_events
           (person_id, event_name, event_date, event_notes, mentioned_at,
            followed_up, status, updated_at)
           VALUES (?, ?, ?, ?, ?, FALSE, 'planned', ?)""",
        (person_id, event_name, event_date, event_notes, now, now),
    )


def get_pending_followups(person_id: int) -> list[dict]:
    """
    Return events that are due for follow-up: followed_up is FALSE and either:
      - event_date is set and has already passed locally (event_date < today), or
      - event_date is NULL and mentioned_at is older than config.FOLLOWUP_UNDATED_DAYS.

    SQLite date('now') is UTC, which can make "tomorrow" events look due the
    evening before on a Mac in Pacific time. Use the host's local date for
    date-only plans because that is how the spoken plan was understood.
    """
    today = _today_local()
    undated_cutoff = _undated_followup_cutoff()
    rows = db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND COALESCE(status, 'planned') = 'planned'
             AND (
               (event_date IS NOT NULL AND event_date < ?)
               OR
               (event_date IS NULL AND mentioned_at < ?)
             )
           ORDER BY mentioned_at""",
        (person_id, today, undated_cutoff),
    )
    return [dict(r) for r in rows]


def mark_anticipated(event_id: int) -> None:
    """Refresh mentioned_at so the same upcoming event isn't proactively anticipated on
    every launch — the cross-session throttle behind ANTICIPATION_REPEAT_COOLDOWN_HOURS
    (the 'Juneteenth every launch' fix). Deliberately does NOT touch followed_up, so the
    post-event follow-up still fires after the date passes."""
    db.execute(
        "UPDATE person_events SET mentioned_at = ?, updated_at = ? WHERE id = ?",
        (_now(), _now(), event_id),
    )


def mark_followed_up(event_id: int, outcome: str) -> None:
    """Set followed_up to TRUE and record the outcome and follow_up_at timestamp."""
    db.execute(
        """UPDATE person_events
           SET followed_up = TRUE, outcome = ?, follow_up_at = ?,
               status = 'completed', updated_at = ?
           WHERE id = ?""",
        (outcome, _now(), _now(), event_id),
    )


def cancel_event(event_id: int, reason: str = "") -> None:
    """Mark a planned event canceled so Rex stops anticipating or following up."""
    db.execute(
        """UPDATE person_events
           SET followed_up = TRUE,
               status = 'canceled',
               canceled_at = ?,
               updated_at = ?,
               outcome = ?
           WHERE id = ?""",
        (_now(), _now(), (reason or "canceled").strip()[:500], int(event_id)),
    )


def reschedule_event(event_id: int, new_date: Optional[str] = None) -> None:
    """Keep a postponed event OPEN (status='planned', not followed up) and refresh
    mentioned_at so it stops re-prompting immediately. event_date is set to ``new_date``
    when known; when None the now-stale date is CLEARED so the event becomes an undated
    open plan rather than a perpetually-overdue follow-up. The inverse of cancel_event:
    a reschedule must never durably lose the plan."""
    db.execute(
        """UPDATE person_events
           SET event_date = ?,
               status = 'planned',
               followed_up = FALSE,
               canceled_at = NULL,
               outcome = NULL,
               mentioned_at = ?,
               updated_at = ?
           WHERE id = ?""",
        (new_date, _now(), _now(), int(event_id)),
    )


def get_upcoming_events(person_id: int) -> list[dict]:
    """Return today-or-future events that have not yet been followed up on."""
    today = _today_local()
    rows = db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND COALESCE(status, 'planned') = 'planned'
             AND event_date >= ?
           ORDER BY event_date""",
        (person_id, today),
    )
    return [dict(r) for r in rows]


def get_open_events(person_id: int) -> list[dict]:
    """Return all planned, not-yet-closed events for a person."""
    rows = db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND COALESCE(status, 'planned') = 'planned'
           ORDER BY
             CASE WHEN event_date IS NULL THEN 1 ELSE 0 END,
             event_date,
             mentioned_at DESC""",
        (person_id,),
    )
    return [dict(r) for r in rows]


def cancel_matching_events(
    person_id: int,
    text: str,
    *,
    event_hint: Optional[dict] = None,
) -> list[dict]:
    """
    Cancel planned events that the user's correction appears to retract.

    If event_hint is supplied, it wins. Otherwise a cancellation phrase must
    share a meaningful token with the stored event, or there must be exactly one
    open event and the utterance is a generic cancellation like "I'm not going
    anymore."
    """
    if person_id is None or not looks_like_cancellation(text):
        return []

    canceled: list[dict] = []
    if event_hint and event_hint.get("id") is not None:
        cancel_event(int(event_hint["id"]), text)
        canceled.append(dict(event_hint))
        return canceled

    open_events = get_open_events(person_id)
    if not open_events:
        return []

    hint_text = ""
    if event_hint:
        hint_text = str(event_hint.get("event_name") or event_hint.get("event_notes") or "")
    text_tokens = _tokens(" ".join([text or "", hint_text]))
    for ev in open_events:
        overlap = text_tokens & _event_tokens(ev)
        if overlap:
            cancel_event(int(ev["id"]), text)
            canceled.append(ev)

    if not canceled and len(open_events) == 1:
        cancel_event(int(open_events[0]["id"]), text)
        canceled.append(open_events[0])

    return canceled


def postpone_matching_events(
    person_id: int,
    text: str,
    *,
    event_hint: Optional[dict] = None,
    new_date: Optional[str] = None,
) -> list[dict]:
    """Reschedule (keep open) planned events the user said were postponed/moved.

    Mirrors cancel_matching_events' matching (event_hint wins, else token overlap,
    else the single open event) but calls reschedule_event instead of cancel_event,
    so a postponed plan survives and Rex keeps anticipating it."""
    if person_id is None or not looks_like_postponement(text):
        return []

    rescheduled: list[dict] = []
    if event_hint and event_hint.get("id") is not None:
        reschedule_event(int(event_hint["id"]), new_date)
        rescheduled.append(dict(event_hint))
        return rescheduled

    open_events = get_open_events(person_id)
    if not open_events:
        return []

    hint_text = ""
    if event_hint:
        hint_text = str(event_hint.get("event_name") or event_hint.get("event_notes") or "")
    text_tokens = _tokens(" ".join([text or "", hint_text]))
    for ev in open_events:
        if text_tokens & _event_tokens(ev):
            reschedule_event(int(ev["id"]), new_date)
            rescheduled.append(ev)

    if not rescheduled and len(open_events) == 1:
        reschedule_event(int(open_events[0]["id"]), new_date)
        rescheduled.append(open_events[0])

    return rescheduled


def delete_events(person_id: int) -> None:
    """Remove all events for a person."""
    db.execute("DELETE FROM person_events WHERE person_id = ?", (person_id,))
