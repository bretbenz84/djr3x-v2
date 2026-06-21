"""
memory/facts.py — Factual knowledge about a person (person_facts table).
"""

import logging
import math
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

_SOURCE_DEFAULT_CONFIDENCE = {
    "explicit": 0.95,
    "corrected": 1.0,
    "inferred": 0.55,
    "observed": 0.75,
    # Told to Rex by a third party about someone who wasn't there ("tell me
    # about" pre-briefings, gossip). Ranks below explicit so the person's own
    # firsthand statements always win on conflict.
    "secondhand": 0.6,
}
_SOURCE_RANK = {
    "inferred": 1,
    "secondhand": 1,
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
    "belief",
    "worldview",
}
_NOISY_CATEGORIES = {"interest_note", "other"}
_DECAY_DEFAULT_DAYS = {
    "fast": 30,
    "normal": 365,
    "permanent": None,
}


# Relative-day phrases that pin a statement to the day it was captured. A fact
# whose value contains one ("today is the speaker's birthday") was only true
# then — recited as a standing memory it makes Rex think every day is that day
# (e.g. wishing happy birthday a week later). Such statements belong in the
# events table, not as durable person traits, so they're dropped from prompt
# injection. Anchored to the canonical structured fact instead (e.g. the
# 'birthday' MM-DD key, which the dedicated birthday-window path owns).
_EPHEMERAL_TIME_RE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (?:morning|afternoon|evening)|"
    r"last night|right now)\b",
    re.IGNORECASE,
)


def _is_ephemeral_statement(fact: dict) -> bool:
    return bool(_EPHEMERAL_TIME_RE.search(str(fact.get("value") or "")))


# Week/month-scale future-relative phrases. Unlike the day-scale _EPHEMERAL_TIME_RE
# (which drops the fact entirely), these are real but TIME-BOUND ("training for a
# marathon next month", "camping next week") — true now, wrong once the window passes.
# They get a short stale horizon + fast decay so they age out of prompt injection
# instead of being recited as standing traits forever.
_FUTURE_RELATIVE_PATTERNS = (
    (re.compile(r"\bin\s+(\d{1,2})\s+days?\b", re.IGNORECASE), lambda m: int(m.group(1)) + 3),
    (re.compile(r"\bin\s+(\d{1,2})\s+weeks?\b", re.IGNORECASE), lambda m: int(m.group(1)) * 7 + 5),
    (re.compile(r"\bin\s+(\d{1,2})\s+months?\b", re.IGNORECASE), lambda m: int(m.group(1)) * 31 + 7),
    (re.compile(r"\b(?:this|next)\s+weekend\b", re.IGNORECASE), lambda m: 10),
    (re.compile(r"\bnext\s+week\b", re.IGNORECASE), lambda m: 12),
    (re.compile(r"\b(?:later\s+)?this\s+week\b", re.IGNORECASE), lambda m: 8),
    (re.compile(r"\bnext\s+month\b", re.IGNORECASE), lambda m: 38),
    (re.compile(r"\bin\s+a\s+(?:few\s+)?weeks?\b", re.IGNORECASE), lambda m: 20),
    (re.compile(r"\bin\s+a\s+month\b", re.IGNORECASE), lambda m: 38),
)


def _time_bound_stale_days(value: str) -> Optional[int]:
    """Return a short stale horizon (days) when a value pins to a future window, else None."""
    text = str(value or "")
    best: Optional[int] = None
    for pattern, days_fn in _FUTURE_RELATIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                days = max(1, int(days_fn(m)))
            except Exception:
                continue
            best = days if best is None else min(best, days)
    return best


def _reconfirm_min_hours() -> float:
    try:
        import config
        return float(getattr(config, "MEMORY_RECONFIRM_MIN_HOURS", 6.0))
    except Exception:
        return 6.0


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
    if any(token in cleaned for token in ("secondhand", "told_about", "gossip", "hearsay")):
        return "secondhand"
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
    if key in _PERMANENT_KEYS or category in {"birthday", "identity", "relationship", "worldview"}:
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
    fact_kind: Optional[str] = None,
    kindness: Optional[float] = None,
    told_by: Optional[int] = None,
) -> None:
    """
    Insert or update a fact.

    Repeated matching evidence strengthens confidence and increments
    evidence_count. A changed value replaces the old value but starts a new
    evidence count so Rex treats the updated memory with appropriate caution.

    fact_kind ('fact'|'gossip'), kindness (-1 mean .. +1 kind), and told_by
    (person_id of the teller) classify secondhand "tell me about" material so
    prompt formatting can hedge it and keep unkind gossip from being recited.
    """
    now = _now()
    normalized_source = _normalize_source(source)
    fact_kind_value = fact_kind if fact_kind in ("fact", "gossip") else None
    kindness_value = None
    if kindness is not None:
        try:
            kindness_value = max(-1.0, min(1.0, float(kindness)))
        except (TypeError, ValueError):
            kindness_value = None
    told_by_value = None
    if told_by is not None:
        try:
            told_by_value = int(told_by)
        except (TypeError, ValueError):
            told_by_value = None
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
    # A time-bound value ("next month") gets a short horizon + fast decay so it expires
    # out of prompt injection once its window passes (never override a permanent fact).
    time_bound_days = _time_bound_stale_days(value)
    if time_bound_days is not None and decay_value != "permanent":
        decay_value = "fast"
        stale_days_value = (
            time_bound_days if stale_days_value is None else min(stale_days_value, time_bound_days)
        )
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
            # Corroboration must be SPACED to count: a fact repeated within the reconfirm
            # window (same conversation) refreshes recency but does NOT inflate evidence
            # or confidence — that's what produced "13 confirmations" on idle chatter. A
            # genuine later-session re-mention (window elapsed) still strengthens it.
            last_conf_dt = _parse_dt(row.get("last_confirmed_at"))
            within_window = (
                last_conf_dt is not None
                and (datetime.now(timezone.utc) - last_conf_dt)
                < timedelta(hours=_reconfirm_min_hours())
            )
            if within_window:
                new_confidence = prior_conf
                evidence_count = prior_evidence
                confirmed_at = row.get("last_confirmed_at") or now
            else:
                new_confidence = min(1.0, max(prior_conf, confidence) + 0.05)
                evidence_count = prior_evidence + 1
                confirmed_at = now
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
            confirmed_at = now
            corrected_at = now if normalized_source == "corrected" else row.get("corrected_at")
        db.execute(
            """UPDATE person_facts
               SET category = ?, value = ?, source = ?, confidence = ?,
                   updated_at = ?, last_confirmed_at = ?, evidence_count = ?,
                   importance = ?, decay_rate = ?, stale_after_days = ?,
                   corrected_at = ?, fact_kind = ?, kindness = ?, told_by = ?
               WHERE person_id = ? AND key = ?""",
            (
                category,
                value,
                normalized_source,
                new_confidence,
                now,
                confirmed_at,
                evidence_count,
                max(prior_importance, importance_value),
                decay_value,
                stale_days_value,
                corrected_at,
                fact_kind_value or row.get("fact_kind") or "fact",
                kindness_value if kindness_value is not None else row.get("kindness"),
                told_by_value if told_by_value is not None else row.get("told_by"),
                person_id,
                key,
            ),
        )
    else:
        db.execute(
            """INSERT INTO person_facts
               (person_id, category, key, value, confidence, source,
                created_at, updated_at, last_confirmed_at, evidence_count,
                importance, decay_rate, stale_after_days, corrected_at,
                fact_kind, kindness, told_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                fact_kind_value or "fact",
                kindness_value,
                told_by_value,
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


def fact_topic_overlap(fact: dict, topic_tokens) -> int:
    """How many live-topic words this fact's key/value/category mention — the cheap
    relevance signal for topic-aware retrieval. Stems both sides (memory.text_match) so
    "dogs" matches a "dog" fact. 0 when no tokens / no match."""
    if not topic_tokens:
        return 0
    from memory import text_match
    text = " ".join(str(fact.get(k) or "") for k in ("key", "value", "category"))
    return text_match.overlap_count(text, topic_tokens)


def _is_expired_provisional(fact: dict) -> bool:
    """True for a fast-decay fact that's gone stale and was never corroborated — the
    'decay queue': a one-off inference or a passed time-bound plan that should no longer
    be proactively recited. Corroborated facts (evidence >= 2) and permanent/normal
    facts are kept."""
    return (
        fact.get("decay_rate") == "fast"
        and fact.get("freshness_label") == "stale"
        and int(fact.get("evidence_count") or 1) < 2
    )


def _fact_hits_mute(fact: dict, mute_terms: set) -> bool:
    if not mute_terms:
        return False
    text = " ".join(str(fact.get(k) or "") for k in ("key", "value", "category")).lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    return bool(words & mute_terms)


def get_prompt_worthy_facts(
    person_id: int, limit: int = 12, *, topic_tokens=None, mute_terms=None
) -> list[dict]:
    """Return prompt-worthy facts ranked by importance, confidence, recency, and use.

    When `topic_tokens` is given, a relevance bonus lifts facts that mention what the
    person JUST said, so an on-topic fact outranks a higher-importance but off-topic
    one (and can make the cut it otherwise wouldn't). With no tokens this is the
    original static importance ranking, unchanged.

    Skips relative-day statements ("today is …") — see _is_ephemeral_statement —
    so a one-day-true line isn't recited as a standing fact forever. Also drops
    expired-provisional facts (the decay queue) and, when `mute_terms` is supplied,
    facts whose topic an active boundary has asked Rex not to raise. These filters
    affect PROACTIVE injection only — get_facts stays unfiltered for direct recall.
    """
    drop_provisional = True
    try:
        import config
        drop_provisional = bool(getattr(config, "MEMORY_DROP_STALE_PROVISIONAL", True))
    except Exception:
        pass
    mute = set(mute_terms) if mute_terms else None
    facts = [
        f for f in get_facts(person_id)
        if f.get("key") != "skin_color"
        and not _is_ephemeral_statement(f)
        and not (drop_provisional and _is_expired_provisional(f))
        and not _fact_hits_mute(f, mute)
    ]
    boost = float(_relevance_boost()) if topic_tokens else 0.0
    cap = int(_relevance_max_matches())

    def _rank(f: dict) -> float:
        score = score_fact_for_prompt(f)
        if topic_tokens:
            score += boost * min(fact_topic_overlap(f, topic_tokens), cap)
        return score

    facts.sort(key=lambda f: -_rank(f))
    return facts[: max(0, int(limit))]


def _relevance_boost() -> float:
    try:
        import config
        return float(getattr(config, "MEMORY_TOPIC_RELEVANCE_BOOST", 0.5))
    except Exception:
        return 0.5


def _relevance_max_matches() -> int:
    try:
        import config
        return int(getattr(config, "MEMORY_TOPIC_RELEVANCE_MAX_MATCHES", 3))
    except Exception:
        return 3


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
        "secondhand": -0.05,
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
    elif source == "secondhand":
        qualifiers.append("secondhand; heard from someone else, hedge this")
    if (fact.get("fact_kind") or "fact") == "gossip":
        kindness = fact.get("kindness")
        try:
            kindness = float(kindness) if kindness is not None else 0.0
        except (TypeError, ValueError):
            kindness = 0.0
        if kindness <= -0.25:
            qualifiers.append(
                "unkind gossip — NEVER repeat or hint at this to the person; "
                "background context only"
            )
        else:
            qualifiers.append("gossip; don't recite it back to them")
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
    fact["fact_kind"] = fact.get("fact_kind") or "fact"
    if fact.get("kindness") is not None:
        try:
            fact["kindness"] = max(-1.0, min(1.0, float(fact["kindness"])))
        except (TypeError, ValueError):
            fact["kindness"] = None
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
