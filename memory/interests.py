"""
memory/interests.py — Durable hobby and interest memory.

Interests are conversation hooks: what a person enjoys, builds, follows, or
practices. They intentionally live outside person_facts so Rex can avoid
re-asking basic discovery questions and instead deepen known threads.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)

_VALID_STRENGTHS = {"low", "medium", "high"}
_VALID_SOURCES = {"explicit", "inferred", "observed", "corrected"}


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> str:
    return _now_dt().isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        cleaned = str(value).replace("Z", "+00:00")
        if "T" not in cleaned and " " in cleaned:
            cleaned = cleaned.replace(" ", "T", 1)
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _clamp(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _clean_name(value: str) -> str:
    cleaned = " ".join((value or "").strip().split())
    return cleaned.strip(".,;:!?\"'")


def _clean_token(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _normalize_strength(value: str) -> str:
    strength = _clean_token(value)
    return strength if strength in _VALID_STRENGTHS else "medium"


def _normalize_source(value: str) -> str:
    source = _clean_token(value)
    return source if source in _VALID_SOURCES else "explicit"


def _cooldown_active(row: dict) -> bool:
    until = _parse_dt(row.get("ask_cooldown_until"))
    return bool(until and until > _now_dt())


def _annotate(row: dict) -> dict:
    row["cooldown_active"] = _cooldown_active(row)
    return row


def upsert_interest(
    person_id: int,
    name: str,
    category: str = "hobby",
    interest_strength: str = "medium",
    *,
    confidence: float = 1.0,
    source: str = "explicit",
    notes: Optional[str] = None,
    associated_people: Optional[str] = None,
    associated_stories: Optional[str] = None,
) -> Optional[int]:
    """Insert or update an interest and return its row id."""
    name_clean = _clean_name(name)
    if not name_clean:
        return None
    category_clean = _clean_token(category) or "hobby"
    strength_clean = _normalize_strength(interest_strength)
    confidence_clean = _clamp(confidence)
    source_clean = _normalize_source(source)
    now = _now()

    existing = db.fetchone(
        "SELECT * FROM person_interests WHERE person_id = ? AND lower(name) = lower(?)",
        (int(person_id), name_clean),
    )
    if existing:
        row = dict(existing)
        row_id = int(row["id"])
        db.execute(
            """UPDATE person_interests
               SET category = ?, interest_strength = ?, confidence = ?, source = ?,
                   last_mentioned_at = ?, notes = COALESCE(NULLIF(?, ''), notes),
                   associated_people = COALESCE(NULLIF(?, ''), associated_people),
                   associated_stories = COALESCE(NULLIF(?, ''), associated_stories)
               WHERE id = ?""",
            (
                category_clean,
                _stronger(strength_clean, row.get("interest_strength")),
                max(confidence_clean, _clamp(row.get("confidence"))),
                source_clean,
                now,
                (notes or "").strip(),
                (associated_people or "").strip(),
                (associated_stories or "").strip(),
                row_id,
            ),
        )
        return row_id

    return db.execute(
        """INSERT INTO person_interests
           (person_id, name, category, interest_strength, confidence, source,
            first_mentioned_at, last_mentioned_at, last_asked_about_at,
            ask_cooldown_until, notes, associated_people, associated_stories)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)""",
        (
            int(person_id),
            name_clean,
            category_clean,
            strength_clean,
            confidence_clean,
            source_clean,
            now,
            now,
            (notes or "").strip(),
            (associated_people or "").strip(),
            (associated_stories or "").strip(),
        ),
    )


def get_interests_for_prompt(person_id: int, limit: int = 8) -> list[dict]:
    """Return highest-signal interests for prompt injection."""
    rows = db.fetchall(
        """SELECT * FROM person_interests
           WHERE person_id = ?
           ORDER BY
             CASE interest_strength WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
             confidence DESC,
             COALESCE(last_mentioned_at, first_mentioned_at) DESC
           LIMIT ?""",
        (int(person_id), max(0, int(limit))),
    )
    return [_annotate(dict(row)) for row in rows]


def get_interest_hooks(person_id: int) -> list[dict]:
    """Return interests that are ready for deeper follow-up questions."""
    rows = db.fetchall(
        """SELECT * FROM person_interests
           WHERE person_id = ?
           ORDER BY
             CASE interest_strength WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
             COALESCE(last_asked_about_at, '1970-01-01') ASC,
             COALESCE(last_mentioned_at, first_mentioned_at) DESC""",
        (int(person_id),),
    )
    hooks = [_annotate(dict(row)) for row in rows]
    return [row for row in hooks if not row.get("cooldown_active")]


def mark_interest_asked(
    person_id: int,
    interest_name: str,
    cooldown_days: int = 30,
) -> None:
    """Record that Rex asked about an interest and set a follow-up cooldown."""
    now_dt = _now_dt()
    until = now_dt + timedelta(days=max(0, int(cooldown_days)))
    db.execute(
        """UPDATE person_interests
           SET last_asked_about_at = ?, ask_cooldown_until = ?
           WHERE person_id = ? AND lower(name) = lower(?)""",
        (now_dt.isoformat(), until.isoformat(), int(person_id), _clean_name(interest_name)),
    )


def delete_interest(
    person_id: int,
    interest_name: Optional[str] = None,
) -> None:
    """Delete all interests for a person, or one named interest."""
    if interest_name:
        db.execute(
            "DELETE FROM person_interests WHERE person_id = ? AND lower(name) = lower(?)",
            (int(person_id), _clean_name(interest_name)),
        )
        return
    db.execute("DELETE FROM person_interests WHERE person_id = ?", (int(person_id),))


def format_interest_for_prompt(interest: dict) -> str:
    """Render a compact interest line for prompt injection."""
    name = interest.get("name") or "unknown interest"
    strength = interest.get("interest_strength") or "medium"
    last = (interest.get("last_mentioned_at") or interest.get("first_mentioned_at") or "")[:10]
    pieces = [f"{name}, {strength} interest"]
    if last:
        pieces.append(f"last mentioned {last}")
    if interest.get("cooldown_active") and interest.get("ask_cooldown_until"):
        pieces.append(f"ask cooldown active until {str(interest['ask_cooldown_until'])[:10]}")
    notes = (interest.get("notes") or "").strip()
    if notes:
        pieces.append(f"notes: {notes[:120]}")
    return ", ".join(pieces)


def _stronger(new_strength: str, old_strength: Optional[str]) -> str:
    rank = {"low": 1, "medium": 2, "high": 3}
    old = _normalize_strength(old_strength or "medium")
    return new_strength if rank[new_strength] >= rank[old] else old
