"""
memory/forgetting.py — targeted deletion of specific remembered details.

Whole-person wipes live in memory.people. This module handles requests like
"forget about my dog Scout" by deleting matching rows from the durable memory
tables that can later feed prompts or proactive callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from memory import database as db


_VAGUE_TARGETS = {
    "it", "that", "this", "that thing", "this thing", "everything", "everyone",
    "everybody", "me", "myself", "you",
}
_STOPWORDS = {
    "about", "all", "and", "any", "delete", "erase", "forget", "from", "memory",
    "memories", "my", "of", "please", "remove", "the", "thing", "your",
}
_GENERIC_TARGET_TOKENS = {
    "cat", "dad", "dog", "father", "friend", "husband", "mom", "mother",
    "partner", "pet", "wife",
}


@dataclass
class ForgetResult:
    target: str
    terms: set[str] = field(default_factory=set)
    deleted: dict[str, int] = field(default_factory=dict)

    @property
    def total_deleted(self) -> int:
        return sum(max(0, int(v)) for v in self.deleted.values())

    def summary(self) -> str:
        parts = [f"{name}={count}" for name, count in self.deleted.items() if count]
        return ", ".join(parts) if parts else "no matching rows"


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (value or "").lower()))


def extract_specific_forget_target(text: str) -> str | None:
    """Return the target phrase in a specific forget request, or None."""
    raw = (text or "").strip()
    if not raw:
        return None
    match = re.match(
        r"^(?:please\s+)?(?:forget|delete|remove|erase)\s+"
        r"(?:(?:all|the)\s+)?(?:(?:memory|memories|things?|stuff)\s+)?"
        r"(?:(?:about|of|for|related\s+to)\s+)?(.+?)"
        r"(?:\s+from\s+(?:your\s+)?memor(?:y|ies))?\s*[?.!]*$",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return None
    target = match.group(1).strip(" \t\r\n?.!,;:\"'`()[]{}")
    target = re.sub(r"^(?:about|of|for)\s+", "", target, flags=re.IGNORECASE)
    norm = _normalize(target)
    if not norm or norm in _VAGUE_TARGETS:
        return None
    return target


def target_terms(target: str) -> set[str]:
    """Build searchable terms from a spoken target phrase."""
    norm = _normalize(target)
    terms: set[str] = set()
    if norm and norm not in _VAGUE_TARGETS:
        terms.add(norm)
    tokens = [t for t in norm.split() if len(t) >= 3 and t not in _STOPWORDS]
    specific_tokens = [t for t in tokens if t not in _GENERIC_TARGET_TOKENS]
    terms.update(specific_tokens or tokens)
    if len(tokens) >= 2:
        terms.add(" ".join(tokens))
    return {term for term in terms if term and term not in _VAGUE_TARGETS}


def text_matches_terms(text: str, terms: set[str]) -> bool:
    haystack = _normalize(text)
    if not haystack or not terms:
        return False
    return any(term in haystack for term in terms)


def row_matches_terms(row: dict, fields: tuple[str, ...], terms: set[str]) -> bool:
    return text_matches_terms(
        " ".join(str(row.get(field) or "") for field in fields),
        terms,
    )


def _delete_matching(
    table: str,
    person_id: int,
    fields: tuple[str, ...],
    terms: set[str],
) -> int:
    rows = db.fetchall(f"SELECT * FROM {table} WHERE person_id = ?", (int(person_id),))
    ids = [
        int(row["id"])
        for row in [dict(r) for r in rows]
        if row.get("id") is not None and row_matches_terms(row, fields, terms)
    ]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    db.execute(
        f"DELETE FROM {table} WHERE person_id = ? AND id IN ({placeholders})",
        (int(person_id), *ids),
    )
    return len(ids)


def forget_specific_memory(person_id: int, target: str) -> ForgetResult:
    """Delete stored memories for one person's target phrase."""
    terms = target_terms(target)
    result = ForgetResult(target=(target or "").strip(), terms=terms)
    if not terms:
        return result

    result.deleted["facts"] = _delete_matching(
        "person_facts",
        person_id,
        ("category", "key", "value", "source"),
        terms,
    )
    result.deleted["events"] = _delete_matching(
        "person_events",
        person_id,
        ("event_name", "event_notes", "outcome"),
        terms,
    )
    result.deleted["emotional_events"] = _delete_matching(
        "person_emotional_events",
        person_id,
        ("category", "description", "loss_subject", "loss_subject_kind", "loss_subject_name"),
        terms,
    )
    result.deleted["conversations"] = _delete_matching(
        "conversations",
        person_id,
        ("summary", "topics", "emotion_tone"),
        terms,
    )
    result.deleted["qa"] = _delete_matching(
        "person_qa",
        person_id,
        ("question_key", "question_text", "answer_text"),
        terms,
    )
    result.deleted["preferences"] = _delete_matching(
        "person_preferences",
        person_id,
        ("domain", "preference_type", "key", "value", "source"),
        terms,
    )
    result.deleted["interests"] = _delete_matching(
        "person_interests",
        person_id,
        (
            "name",
            "category",
            "interest_strength",
            "source",
            "notes",
            "associated_people",
            "associated_stories",
        ),
        terms,
    )
    return result


def forget_memory_detail(person_id: int, target: str) -> ForgetResult:
    """Delete matching facts/preferences/interests for a person."""
    terms = target_terms(target)
    result = ForgetResult(target=(target or "").strip(), terms=terms)
    if not terms:
        return result

    result.deleted["facts"] = _delete_matching(
        "person_facts",
        person_id,
        ("category", "key", "value", "source"),
        terms,
    )
    result.deleted["preferences"] = _delete_matching(
        "person_preferences",
        person_id,
        ("domain", "preference_type", "key", "value", "source"),
        terms,
    )
    result.deleted["interests"] = _delete_matching(
        "person_interests",
        person_id,
        (
            "name",
            "category",
            "interest_strength",
            "source",
            "notes",
            "associated_people",
            "associated_stories",
        ),
        terms,
    )
    return result


def fact_or_event_matches(payload: dict, terms: set[str]) -> bool:
    """Return True when an extracted fact/event payload mentions forgotten terms."""
    return row_matches_terms(
        payload,
        (
            "category",
            "domain",
            "preference_type",
            "key",
            "value",
            "name",
            "interest_strength",
            "notes",
            "associated_people",
            "associated_stories",
            "event_name",
            "event_notes",
            "description",
        ),
        terms,
    )
