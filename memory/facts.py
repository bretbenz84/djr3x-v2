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


def _clamp_confidence(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


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


def _age_days(fact: dict) -> Optional[int]:
    dt = (
        _parse_dt(fact.get("last_confirmed_at"))
        or _parse_dt(fact.get("updated_at"))
        or _parse_dt(fact.get("created_at"))
    )
    if dt is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def _freshness_label(age_days: Optional[int]) -> str:
    if age_days is None:
        return "unknown"
    if age_days <= 90:
        return "fresh"
    if age_days <= 365:
        return "aging"
    return "stale"


def add_fact(
    person_id: int,
    category: str,
    key: str,
    value: str,
    source: str,
    confidence: float = 1.0,
) -> None:
    """
    Insert or update a fact.

    Repeated matching evidence strengthens confidence and increments
    evidence_count. A changed value replaces the old value but starts a new
    evidence count so Rex treats the updated memory with appropriate caution.
    """
    now = _now()
    confidence = _clamp_confidence(confidence)
    existing = db.fetchone(
        "SELECT * FROM person_facts WHERE person_id = ? AND key = ?",
        (person_id, key),
    )
    if existing:
        row = dict(existing)
        prior_value = (row.get("value") or "").strip()
        same_value = prior_value.lower() == (value or "").strip().lower()
        prior_conf = _clamp_confidence(row.get("confidence", 0.5))
        prior_evidence = int(row.get("evidence_count") or 1)
        if same_value:
            new_confidence = min(1.0, max(prior_conf, confidence) + 0.05)
            evidence_count = prior_evidence + 1
        else:
            new_confidence = confidence
            evidence_count = 1
        db.execute(
            """UPDATE person_facts
               SET category = ?, value = ?, source = ?, confidence = ?,
                   updated_at = ?, last_confirmed_at = ?, evidence_count = ?
               WHERE person_id = ? AND key = ?""",
            (
                category,
                value,
                source,
                new_confidence,
                now,
                now,
                evidence_count,
                person_id,
                key,
            ),
        )
    else:
        db.execute(
            """INSERT INTO person_facts
               (person_id, category, key, value, confidence, source,
                created_at, updated_at, last_confirmed_at, evidence_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (person_id, category, key, value, confidence, source, now, now, now, 1),
        )


def get_facts(person_id: int) -> list[dict]:
    """Return all facts for a person."""
    rows = db.fetchall(
        "SELECT * FROM person_facts WHERE person_id = ? ORDER BY category, key",
        (person_id,),
    )
    return [_annotate_fact(dict(r)) for r in rows]


def get_facts_by_category(person_id: int, category: str) -> list[dict]:
    """Return all facts for a person filtered by category."""
    rows = db.fetchall(
        "SELECT * FROM person_facts WHERE person_id = ? AND category = ? ORDER BY key",
        (person_id, category),
    )
    return [_annotate_fact(dict(r)) for r in rows]


def get_stale_facts(person_id: int, days: int) -> list[dict]:
    """Return facts where updated_at is older than the given number of days."""
    rows = db.fetchall(
        """SELECT * FROM person_facts
           WHERE person_id = ?
             AND COALESCE(last_confirmed_at, updated_at, created_at) < datetime('now', ?)
           ORDER BY COALESCE(last_confirmed_at, updated_at, created_at)""",
        (person_id, f"-{days} days"),
    )
    return [_annotate_fact(dict(r)) for r in rows]


def get_prompt_facts(person_id: int, *, limit: int = 12) -> list[dict]:
    """Return facts sorted for prompt use, with confidence/freshness metadata."""
    facts = [f for f in get_facts(person_id) if f.get("key") != "skin_color"]
    facts.sort(
        key=lambda f: (
            -float(f.get("confidence") or 0.0),
            f.get("age_days") if f.get("age_days") is not None else 99999,
            f.get("category") or "",
            f.get("key") or "",
        )
    )
    return facts[: max(0, int(limit))]


def format_fact_for_prompt(fact: dict) -> str:
    key = fact.get("key") or "fact"
    value = fact.get("value") or ""
    confidence_label = fact.get("confidence_label") or "medium"
    freshness_label = fact.get("freshness_label") or "unknown"
    age_days = fact.get("age_days")
    pieces = [f"{key}: {value}"]
    qualifiers = []
    if confidence_label != "high":
        qualifiers.append(f"{confidence_label} confidence")
    if freshness_label in {"aging", "stale", "unknown"}:
        if isinstance(age_days, int):
            qualifiers.append(f"{freshness_label}; last confirmed {age_days}d ago")
        else:
            qualifiers.append(f"{freshness_label} freshness")
    evidence_count = int(fact.get("evidence_count") or 1)
    if evidence_count > 1:
        qualifiers.append(f"{evidence_count} confirmations")
    if qualifiers:
        pieces.append(f"({'; '.join(qualifiers)})")
    return " ".join(pieces)


def _annotate_fact(fact: dict) -> dict:
    confidence = _clamp_confidence(fact.get("confidence", 0.5))
    age_days = _age_days(fact)
    fact["confidence"] = confidence
    fact["confidence_label"] = _confidence_label(confidence)
    fact["age_days"] = age_days
    fact["freshness_label"] = _freshness_label(age_days)
    fact["evidence_count"] = int(fact.get("evidence_count") or 1)
    fact["memory_quality"] = (
        f"{fact['confidence_label']} confidence, {fact['freshness_label']} freshness"
    )
    return fact


def delete_facts(person_id: int) -> None:
    """Remove all facts for a person."""
    db.execute("DELETE FROM person_facts WHERE person_id = ?", (person_id,))
