"""
memory/relationships.py — Q&A history and question depth management (person_qa table).
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
from memory import people as people_db

_log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_qa_history(person_id: int) -> list[dict]:
    """Return all Q&A pairs for a person, ordered by asked_at."""
    rows = db.fetchall(
        "SELECT * FROM person_qa WHERE person_id = ? ORDER BY asked_at",
        (person_id,),
    )
    return [dict(r) for r in rows]


def get_answered_question_keys(person_id: int) -> set[str]:
    """Return question_keys with a non-empty human answer."""
    rows = db.fetchall(
        """SELECT question_key FROM person_qa
           WHERE person_id = ?
             AND answer_text IS NOT NULL
             AND TRIM(answer_text) != ''""",
        (person_id,),
    )
    return {row["question_key"] for row in rows}


def get_asked_question_keys(person_id: int) -> set[str]:
    """Return every question_key already asked, including unanswered ones."""
    rows = db.fetchall(
        "SELECT question_key FROM person_qa WHERE person_id = ?",
        (person_id,),
    )
    return {row["question_key"] for row in rows}


def save_qa(
    person_id: int,
    question_key: str,
    question_text: str,
    answer_text: str,
    depth_level: int,
) -> None:
    """Store a Q&A pair and apply the familiarity increment for the given depth level."""
    db.execute(
        """INSERT INTO person_qa
           (person_id, question_key, question_text, answer_text, asked_at, depth_level)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (person_id, question_key, question_text, answer_text, _now(), depth_level),
    )
    increment_key = f"qa_depth_{depth_level}"
    increment = config.FAMILIARITY_INCREMENTS.get(increment_key)
    if increment is not None:
        people_db.update_familiarity(person_id, increment)
    else:
        _log.warning("No familiarity increment defined for depth level %d", depth_level)


def save_question_asked(
    person_id: int,
    question_key: str,
    question_text: str,
    depth_level: int,
) -> None:
    """Store that Rex asked a question. Familiarity increases only once answered."""
    existing = db.fetchone(
        """SELECT id FROM person_qa
           WHERE person_id = ? AND question_key = ?
           ORDER BY asked_at DESC LIMIT 1""",
        (person_id, question_key),
    )
    if existing:
        return
    db.execute(
        """INSERT INTO person_qa
           (person_id, question_key, question_text, answer_text, asked_at, depth_level)
           VALUES (?, ?, ?, '', ?, ?)""",
        (person_id, question_key, question_text, _now(), depth_level),
    )


def get_latest_pending_question(person_id: int) -> Optional[dict]:
    """Return the newest asked-but-unanswered question for this person."""
    row = db.fetchone(
        """SELECT * FROM person_qa
           WHERE person_id = ?
             AND (answer_text IS NULL OR TRIM(answer_text) = '')
           ORDER BY asked_at DESC
           LIMIT 1""",
        (person_id,),
    )
    return dict(row) if row else None


def answer_latest_pending_question(person_id: int, answer_text: str) -> Optional[dict]:
    """
    Attach a human answer to the newest pending question.
    Returns the answered row, or None if there was no pending question.
    """
    answer = (answer_text or "").strip()
    if not answer:
        return None
    pending = get_latest_pending_question(person_id)
    if not pending:
        return None
    db.execute(
        "UPDATE person_qa SET answer_text = ? WHERE id = ?",
        (answer, pending["id"]),
    )
    increment_key = f"qa_depth_{pending.get('depth_level', 1)}"
    increment = config.FAMILIARITY_INCREMENTS.get(increment_key)
    if increment is not None:
        people_db.update_familiarity(person_id, increment)
    else:
        _log.warning("No familiarity increment defined for depth level %s", pending.get("depth_level"))
    pending["answer_text"] = answer
    return pending


def get_next_question(person_id: int, friendship_tier: str) -> Optional[dict]:
    """
    Return the next unanswered question appropriate for the current friendship tier,
    drawn from config.QUESTION_POOL. Questions are returned in pool order.
    Returns None if all tier-appropriate questions have already been answered.
    """
    max_depth = config.TIER_MAX_DEPTH.get(friendship_tier, 1)
    asked = get_asked_question_keys(person_id)
    for question in config.QUESTION_POOL:
        if question["depth"] <= max_depth and question["key"] not in asked:
            return question
    return None


def delete_qa(person_id: int) -> None:
    """Remove all Q&A records for a person."""
    db.execute("DELETE FROM person_qa WHERE person_id = ?", (person_id,))
