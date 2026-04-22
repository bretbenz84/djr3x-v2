"""
memory/facts.py — Factual knowledge about a person (person_facts table).
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_fact(
    person_id: int,
    category: str,
    key: str,
    value: str,
    source: str,
    confidence: float = 1.0,
) -> None:
    """Insert or update a fact. On (person_id, key) conflict, updates value and sets updated_at."""
    now = _now()
    existing = db.fetchone(
        "SELECT id FROM person_facts WHERE person_id = ? AND key = ?",
        (person_id, key),
    )
    if existing:
        db.execute(
            """UPDATE person_facts
               SET category = ?, value = ?, source = ?, confidence = ?, updated_at = ?
               WHERE person_id = ? AND key = ?""",
            (category, value, source, confidence, now, person_id, key),
        )
    else:
        db.execute(
            """INSERT INTO person_facts
               (person_id, category, key, value, confidence, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (person_id, category, key, value, confidence, source, now, now),
        )


def get_facts(person_id: int) -> list[dict]:
    """Return all facts for a person."""
    rows = db.fetchall(
        "SELECT * FROM person_facts WHERE person_id = ? ORDER BY category, key",
        (person_id,),
    )
    return [dict(r) for r in rows]


def get_facts_by_category(person_id: int, category: str) -> list[dict]:
    """Return all facts for a person filtered by category."""
    rows = db.fetchall(
        "SELECT * FROM person_facts WHERE person_id = ? AND category = ? ORDER BY key",
        (person_id, category),
    )
    return [dict(r) for r in rows]


def get_stale_facts(person_id: int, days: int) -> list[dict]:
    """Return facts where updated_at is older than the given number of days."""
    rows = db.fetchall(
        """SELECT * FROM person_facts
           WHERE person_id = ?
             AND updated_at < datetime('now', ?)
           ORDER BY updated_at""",
        (person_id, f"-{days} days"),
    )
    return [dict(r) for r in rows]


def delete_facts(person_id: int) -> None:
    """Remove all facts for a person."""
    db.execute("DELETE FROM person_facts WHERE person_id = ?", (person_id,))
