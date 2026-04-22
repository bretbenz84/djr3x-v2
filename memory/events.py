"""
memory/events.py — Upcoming events and follow-up tracking (person_events table).
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from memory import database as db

_log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_event(
    person_id: int,
    event_name: str,
    event_date: Optional[str],
    event_notes: str,
) -> Optional[int]:
    """Store an upcoming event. event_date may be None if no specific date was given."""
    return db.execute(
        """INSERT INTO person_events
           (person_id, event_name, event_date, event_notes, mentioned_at, followed_up)
           VALUES (?, ?, ?, ?, ?, FALSE)""",
        (person_id, event_name, event_date, event_notes, _now()),
    )


def get_pending_followups(person_id: int) -> list[dict]:
    """
    Return events that are due for follow-up: followed_up is FALSE and either:
      - event_date is set and has already passed (event_date <= today), or
      - event_date is NULL and mentioned_at is older than config.FOLLOWUP_UNDATED_DAYS.
    """
    rows = db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND (
               (event_date IS NOT NULL AND event_date <= date('now'))
               OR
               (event_date IS NULL AND mentioned_at < datetime('now', ?))
             )
           ORDER BY mentioned_at""",
        (person_id, f"-{config.FOLLOWUP_UNDATED_DAYS} days"),
    )
    return [dict(r) for r in rows]


def mark_followed_up(event_id: int, outcome: str) -> None:
    """Set followed_up to TRUE and record the outcome and follow_up_at timestamp."""
    db.execute(
        """UPDATE person_events
           SET followed_up = TRUE, outcome = ?, follow_up_at = ?
           WHERE id = ?""",
        (outcome, _now(), event_id),
    )


def get_upcoming_events(person_id: int) -> list[dict]:
    """Return future events that have not yet been followed up on."""
    rows = db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND event_date > date('now')
           ORDER BY event_date""",
        (person_id,),
    )
    return [dict(r) for r in rows]


def delete_events(person_id: int) -> None:
    """Remove all events for a person."""
    db.execute("DELETE FROM person_events WHERE person_id = ?", (person_id,))
