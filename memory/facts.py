"""
memory/facts.py — Factual knowledge about a person (person_facts table).
"""

import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)

_SOURCE_DEFAULT_CONFIDENCE = {
    "explicit": 0.95,
    "corrected": 1.0,
    "inferred": 0.55,
    "observed": 0.75,
}
_SOURCE_RANK = {
    "inferred": 1,
    "observed": 2,
    "explicit": 3,
    "corrected": 4,
}
_PERMANENT_KEYS = {
    "birthday",
    "birth_year",
    "pronouns",
    "last_name_declined",
}
_HIGH_IMPORTANCE_CATEGORIES = {
    "birthday",
    "identity",
    "relationship",
    "family",
    "pet",
    "preference",
    "boundary",
    "inside_joke",
}
_NOISY_CATEGORIES = {"interest_note", "other"}
_DECAY_DEFAULT_DAYS = {
    "fast": 30,
    "normal": 365,
    "permanent": None,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_confidence(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _clamp(value: float, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_source(source: str) -> str:
    cleaned = (source or "").strip().lower()
    if cleaned in _SOURCE_DEFAULT_CONFIDENCE:
        return cleaned
    if any(token in cleaned for token in ("correct", "repair", "rename")):
        return "corrected"
    if any(token in cleaned for token in ("observed", "vision", "appearance")):
        return "observed"
    if any(token in cleaned for token in ("infer", "thread", "pattern")):
        return "inferred"
    return "explicit"


def _default_confidence(source: str) -> float:
    return _SOURCE_DEFAULT_CONFIDENCE.get(_normalize_source(source), 0.95)


def _decay_rate(category: str, key: str, source: str, explicit_decay: Optional[str] = None) -> str:
    if explicit_decay in _DECAY_DEFAULT_DAYS:
        return str(explicit_decay)
    category = (category or "").lower()
    key = (key or "").lower()
    normalized_source = _normalize_source(source)
    if key in _PERMANENT_KEYS or category in {"birthday", "identity", "relationship"}:
        return "permanent"
    if normalized_source == "inferred" or category in _NOISY_CATEGORIES:
        return "fast"
    return "normal"


def _default_stale_after_days(decay_rate: str, stale_after_days: Optional[int] = None) -> Optional[int]:
    if stale_after_days is not None:
        try:
            return max(1, int(stale_after_days))
        except (TypeError, ValueError):
            pass
    return _DECAY_DEFAULT_DAYS.get(decay_rate)


def _default_importance(
    category: str,
    key: str,
    source: str,
    value: str,
    explicit_importance: Optional[float] = None,
) -> float:
    if explicit_importance is not None:
        return _clamp(explicit_importance)
    category = (category or "").lower()
    key = (key or "").lower()
    value_l = (value or "").lower()
    normalized_source = _normalize_source(source)
    if key in _PERMANENT_KEYS or category in _HIGH_IMPORTANCE_CATEGORIES:
        return 0.85
    if category == "preference" and any(
        token in value_l for token in ("favorite", "hate", "love", "prefer", "avoid")
    ):
        return 0.8
    if normalized_source == "corrected":
        return 0.9
    if normalized_source == "inferred":
        return 0.35
    if category in _NOISY_CATEGORIES:
        return 0.25
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


def _used_age_days(fact: dict) -> Optional[int]:
    dt = _parse_dt(fact.get("last_used_at"))
    if dt is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 86400))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def _freshness_label(age_days: Optional[int], stale_after_days: Optional[int] = None, decay_rate: str = "normal") -> str:
    if decay_rate == "permanent":
        return "permanent"
    if age_days is None:
        return "unknown"
    stale_at = stale_after_days or _DECAY_DEFAULT_DAYS.get(decay_rate) or 365
    if age_days <= max(7, int(stale_at * 0.25)):
        return "fresh"
    if age_days <= stale_at:
        return "aging"
    return "stale"


def add_fact(
    person_id: int,
    category: str,
    key: str,
    value: str,
    source: str,
    confidence: Optional[float] = None,
    importance: Optional[float] = None,
    decay_rate: Optional[str] = None,
    stale_after_days: Optional[int] = None,
) -> None:
    """
    Insert or update a fact.

    Repeated matching evidence strengthens confidence and increments
    evidence_count. A changed value replaces the old value but starts a new
    evidence count so Rex treats the updated memory with appropriate caution.
    """
    now = _now()
    normalized_source = _normalize_source(source)
    confidence = _clamp_confidence(
        _default_confidence(normalized_source) if confidence is None else confidence
    )
    importance_value = _default_importance(
        category,
        key,
        normalized_source,
        value,
        importance,
    )
    decay_value = _decay_rate(category, key, normalized_source, decay_rate)
    stale_days_value = _default_stale_after_days(decay_value, stale_after_days)
    existing = db.fetchone(
        "SELECT * FROM person_facts WHERE person_id = ? AND key = ?",
        (person_id, key),
    )
    if existing:
        row = dict(existing)
        prior_value = (row.get("value") or "").strip()
        same_value = prior_value.lower() == (value or "").strip().lower()
        prior_conf = _clamp_confidence(row.get("confidence", 0.5))
        prior_source = _normalize_source(row.get("source") or "")
        prior_importance = _clamp(row.get("importance", 0.5))
        prior_evidence = int(row.get("evidence_count") or 1)
        if same_value:
            new_confidence = min(1.0, max(prior_conf, confidence) + 0.05)
            evidence_count = prior_evidence + 1
            corrected_at = row.get("corrected_at")
        else:
            if (
                _SOURCE_RANK.get(normalized_source, 2) < _SOURCE_RANK.get(prior_source, 2)
                and prior_source in {"explicit", "corrected"}
            ):
                _log.debug(
                    "skipping weaker fact overwrite person_id=%s key=%r old_source=%s new_source=%s",
                    person_id,
                    key,
                    prior_source,
                    normalized_source,
                )
                return
            new_confidence = confidence
            evidence_count = 1
            corrected_at = now if normalized_source == "corrected" else row.get("corrected_at")
        db.execute(
            """UPDATE person_facts
               SET category = ?, value = ?, source = ?, confidence = ?,
                   updated_at = ?, last_confirmed_at = ?, evidence_count = ?,
                   importance = ?, decay_rate = ?, stale_after_days = ?,
                   corrected_at = ?
               WHERE person_id = ? AND key = ?""",
            (
                category,
                value,
                normalized_source,
                new_confidence,
                now,
                now,
                evidence_count,
                max(prior_importance, importance_value),
                decay_value,
                stale_days_value,
                corrected_at,
                person_id,
                key,
            ),
        )
    else:
        db.execute(
            """INSERT INTO person_facts
               (person_id, category, key, value, confidence, source,
                created_at, updated_at, last_confirmed_at, evidence_count,
                importance, decay_rate, stale_after_days, corrected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                person_id,
                category,
                key,
                value,
                confidence,
                normalized_source,
                now,
                now,
                now,
                1,
                importance_value,
                decay_value,
                stale_days_value,
                now if normalized_source == "corrected" else None,
            ),
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
    """Return facts that are stale or low-confidence, sorted by confirmation value."""
    facts = [
        f for f in get_facts(person_id)
        if f.get("decay_rate") != "permanent"
        and (
            f.get("freshness_label") == "stale"
            or float(f.get("confidence") or 0.0) < 0.60
            or (
                f.get("age_days") is not None
                and f.get("age_days") >= max(1, int(days))
            )
        )
    ]
    facts.sort(
        key=lambda f: (
            -float(f.get("importance") or 0.0),
            float(f.get("confidence") or 0.0),
            -(f.get("age_days") or 0),
        )
    )
    return facts


def get_prompt_facts(person_id: int, *, limit: int = 12) -> list[dict]:
    """Return facts sorted for prompt use, with confidence/freshness metadata."""
    return get_prompt_worthy_facts(person_id, limit=limit)


def get_prompt_worthy_facts(person_id: int, limit: int = 12) -> list[dict]:
    """Return prompt-worthy facts ranked by importance, confidence, recency, and use."""
    facts = [f for f in get_facts(person_id) if f.get("key") != "skin_color"]
    facts.sort(
        key=lambda f: -score_fact_for_prompt(f)
    )
    return facts[: max(0, int(limit))]


def score_fact_for_prompt(fact: dict) -> float:
    """Score a fact for prompt injection."""
    confidence = _clamp(fact.get("confidence", 0.5))
    importance = _clamp(fact.get("importance", 0.5))
    age_days = fact.get("age_days")
    used_age = fact.get("last_used_age_days")
    freshness = fact.get("freshness_label")
    source = _normalize_source(fact.get("source") or "")

    age_penalty = 0.0
    if freshness == "stale":
        age_penalty = 0.30
    elif freshness == "aging":
        age_penalty = 0.12
    elif freshness == "unknown":
        age_penalty = 0.18

    overuse_penalty = 0.0
    if used_age is not None:
        overuse_penalty = max(0.0, 0.18 - min(0.18, used_age / 30.0 * 0.18))

    source_bonus = {
        "corrected": 0.15,
        "explicit": 0.08,
        "observed": 0.0,
        "inferred": -0.12,
    }.get(source, 0.0)
    permanence_bonus = 0.08 if fact.get("decay_rate") == "permanent" else 0.0
    recency_bonus = 0.0
    if isinstance(age_days, int):
        recency_bonus = max(0.0, 0.10 - math.log1p(age_days) / 60.0)

    return (
        importance * 0.45
        + confidence * 0.35
        + recency_bonus
        + source_bonus
        + permanence_bonus
        - age_penalty
        - overuse_penalty
    )


def format_fact_for_prompt(fact: dict) -> str:
    key = fact.get("key") or "fact"
    value = fact.get("value") or ""
    confidence_label = fact.get("confidence_label") or "medium"
    freshness_label = fact.get("freshness_label") or "unknown"
    age_days = fact.get("age_days")
    pieces = [f"{key}: {value}"]
    qualifiers = []
    source = _normalize_source(fact.get("source") or "")
    if source == "inferred":
        qualifiers.append("inferred; hedge this")
    elif source == "corrected":
        qualifiers.append("corrected by the person")
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
    decay_rate = fact.get("decay_rate") or _decay_rate(
        fact.get("category", ""),
        fact.get("key", ""),
        fact.get("source", ""),
    )
    stale_after_days = fact.get("stale_after_days")
    if stale_after_days is None:
        stale_after_days = _default_stale_after_days(decay_rate)
    fact["confidence"] = confidence
    fact["source"] = _normalize_source(fact.get("source") or "")
    fact["confidence_label"] = _confidence_label(confidence)
    fact["importance"] = _clamp(fact.get("importance", 0.5))
    fact["decay_rate"] = decay_rate
    fact["stale_after_days"] = stale_after_days
    fact["age_days"] = age_days
    fact["last_used_age_days"] = _used_age_days(fact)
    fact["freshness_label"] = _freshness_label(age_days, stale_after_days, decay_rate)
    fact["evidence_count"] = int(fact.get("evidence_count") or 1)
    fact["prompt_score"] = score_fact_for_prompt(fact)
    fact["memory_quality"] = (
        f"{fact['confidence_label']} confidence, {fact['freshness_label']} freshness, "
        f"importance {fact['importance']:.2f}, source {_normalize_source(fact.get('source') or '')}"
    )
    return fact


def mark_fact_used(fact_id: int) -> None:
    """Mark a fact as used in prompt/reply construction."""
    db.execute(
        "UPDATE person_facts SET last_used_at = ? WHERE id = ?",
        (_now(), int(fact_id)),
    )


def apply_fact_correction(
    person_id: int,
    key: str,
    value: str,
    *,
    category: str = "other",
    importance: Optional[float] = None,
    decay_rate: Optional[str] = None,
) -> None:
    """Apply a user correction as a high-confidence corrected fact."""
    add_fact(
        person_id,
        category,
        key,
        value,
        source="corrected",
        confidence=1.0,
        importance=0.9 if importance is None else importance,
        decay_rate=decay_rate,
    )


def delete_facts(person_id: int) -> None:
    """Remove all facts for a person."""
    db.execute("DELETE FROM person_facts WHERE person_id = ?", (person_id,))
