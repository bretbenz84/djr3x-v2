"""
memory/preferences.py — Typed preference memory for people.

Preferences supplement generic person_facts with structured likes, dislikes,
boundaries, and interaction style hints that Rex can use more carefully.
"""

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)

_VALID_TYPES = {"likes", "dislikes", "prefers", "avoids", "boundary"}
_BOUNDARY_IMPORTANCE_FLOOR = 0.95


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _clean_token(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _normalize_type(value: str) -> str:
    pref_type = _clean_token(value)
    return pref_type if pref_type in _VALID_TYPES else "prefers"


def upsert_preference(
    person_id: int,
    domain: str,
    preference_type: str,
    key: str,
    value: Optional[str] = None,
    *,
    confidence: float = 1.0,
    importance: float = 0.5,
    source: str = "explicit",
    ask_cooldown_until: Optional[str] = None,
) -> Optional[int]:
    """Insert or update a typed preference and return its row id."""
    domain_clean = _clean_token(domain) or "general"
    type_clean = _normalize_type(preference_type)
    key_clean = _clean_token(key)
    if not key_clean:
        return None

    confidence_clean = _clamp(confidence, 0.0, 1.0)
    importance_clean = _clamp(importance, 0.0, 1.0)
    if type_clean == "boundary":
        importance_clean = max(importance_clean, _BOUNDARY_IMPORTANCE_FLOOR)
    source_clean = _clean_token(source) or "explicit"
    now = _now()

    existing = db.fetchone(
        """SELECT id, confidence, importance FROM person_preferences
           WHERE person_id = ? AND domain = ? AND preference_type = ? AND key = ?""",
        (int(person_id), domain_clean, type_clean, key_clean),
    )
    if existing:
        row_id = int(existing["id"])
        db.execute(
            """UPDATE person_preferences
               SET value = ?, confidence = ?, importance = ?, source = ?,
                   updated_at = ?, ask_cooldown_until = COALESCE(?, ask_cooldown_until)
               WHERE id = ?""",
            (
                (value or "").strip(),
                max(confidence_clean, float(existing["confidence"] or 0.0)),
                max(importance_clean, float(existing["importance"] or 0.0)),
                source_clean,
                now,
                ask_cooldown_until,
                row_id,
            ),
        )
        return row_id

    return db.execute(
        """INSERT INTO person_preferences
           (person_id, domain, preference_type, key, value, confidence,
            importance, source, created_at, updated_at, last_used_at,
            ask_cooldown_until)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
        (
            int(person_id),
            domain_clean,
            type_clean,
            key_clean,
            (value or "").strip(),
            confidence_clean,
            importance_clean,
            source_clean,
            now,
            now,
            ask_cooldown_until,
        ),
    )


def get_preferences_for_prompt(person_id: int, limit: int = 10) -> list[dict]:
    """Return highest-signal preferences for prompt injection."""
    rows = db.fetchall(
        """SELECT * FROM person_preferences
           WHERE person_id = ?
           ORDER BY
             CASE WHEN preference_type = 'boundary' THEN 1 ELSE 0 END DESC,
             importance DESC,
             confidence DESC,
             COALESCE(updated_at, created_at) DESC
           LIMIT ?""",
        (int(person_id), max(0, int(limit))),
    )
    return [dict(row) for row in rows]


def find_preference(
    person_id: int,
    domain: Optional[str] = None,
    key: Optional[str] = None,
) -> list[dict]:
    """Find preferences for a person, optionally filtered by domain and/or key."""
    clauses = ["person_id = ?"]
    params: list[object] = [int(person_id)]
    if domain:
        clauses.append("domain = ?")
        params.append(_clean_token(domain))
    if key:
        clauses.append("key = ?")
        params.append(_clean_token(key))
    rows = db.fetchall(
        f"""SELECT * FROM person_preferences
            WHERE {' AND '.join(clauses)}
            ORDER BY domain, preference_type, key""",
        tuple(params),
    )
    return [dict(row) for row in rows]


def delete_preference(
    person_id: int,
    domain: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Delete preferences for a person, optionally filtered by domain/key."""
    clauses = ["person_id = ?"]
    params: list[object] = [int(person_id)]
    if domain:
        clauses.append("domain = ?")
        params.append(_clean_token(domain))
    if key:
        clauses.append("key = ?")
        params.append(_clean_token(key))
    db.execute(
        f"DELETE FROM person_preferences WHERE {' AND '.join(clauses)}",
        tuple(params),
    )


def mark_preference_used(preference_id: int) -> None:
    """Mark a preference as used in a prompt/reply decision."""
    db.execute(
        "UPDATE person_preferences SET last_used_at = ? WHERE id = ?",
        (_now(), int(preference_id)),
    )


def format_preference_for_prompt(pref: dict) -> str:
    """Render a compact prompt line like music.dislikes: country."""
    domain = pref.get("domain") or "general"
    pref_type = pref.get("preference_type") or "prefers"
    key = (pref.get("key") or "").replace("_", " ")
    value = (pref.get("value") or "").strip()
    if pref_type == "boundary":
        detail = value or key
        return f"{domain}.boundary: {detail}"
    return f"{domain}.{pref_type}: {value or key}"


def delete_preferences(person_id: int) -> None:
    """Remove all preference rows for a person."""
    db.execute("DELETE FROM person_preferences WHERE person_id = ?", (int(person_id),))
