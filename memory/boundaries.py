"""
memory/boundaries.py - conversational boundaries and preferences.

These are not factual biography. They are consent/preferences for how Rex should
talk with a person: topics not to ask about, jokes not to make, and appearance
or check-in areas to avoid unless the person reopens them.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from memory import database as db

_log = logging.getLogger(__name__)

_DEFAULT_TOPIC = "current topic"

_TOPIC_ALIASES = {
    "how i am doing": "how are you",
    "how i'm doing": "how are you",
    "how im doing": "how are you",
    "how are you": "how are you",
    "how i'm feeling": "how are you",
    "how i feel": "how are you",
    "my appearance": "appearance",
    "appearance": "appearance",
    "my body": "body",
    "body": "body",
    "my weight": "body",
    "weight": "body",
    "work": "work",
    "my job": "work",
    "job": "work",
    "shirt": "clothing",
    "my shirt": "clothing",
    "clothes": "clothing",
    "clothing": "clothing",
}

_BOUNDARY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("roast", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:roast|tease|make fun of|joke about)\s+"
        r"(?:me\s+)?(?:about|for|over)?\s*(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:ask|question|bring up)\s+"
        r"(?:me\s+)?(?:about|on)?\s*(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("mention", re.compile(
        r"\b(?:don'?t|do not|stop|please don'?t|please do not)\s+"
        r"(?:mention|comment on|talk about|bring up)\s+"
        r"(?:my\s+|the\s+)?(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
    ("ask", re.compile(
        r"\b(?:i hate|i don'?t like|i do not like)\s+"
        r"(?:being\s+)?(?:asked|getting asked)\s+"
        r"(?:about\s+)?(?P<topic>[^.?!,;]+)",
        re.IGNORECASE,
    )),
]

_CLEAR_PAT = re.compile(
    r"\b(?:you can|it's okay to|it is okay to|feel free to|you may)\s+"
    r"(?P<behavior>ask|mention|roast|tease|joke about|talk about)\s+"
    r"(?:me\s+)?(?:about\s+|on\s+)?(?P<topic>[^.?!,;]+)",
    re.IGNORECASE,
)

_TRAILING_JUNK = re.compile(
    r"\s+(again|anymore|any more|please|okay|ok|with me|to me)$",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_boundary(
    person_id: int,
    behavior: str,
    topic: str,
    *,
    description: Optional[str] = None,
    source_text: str = "",
) -> Optional[int]:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    if not behavior or not topic:
        return None

    desc = description or _description_for(behavior, topic)
    now = _now()
    existing = db.fetchone(
        "SELECT id FROM person_conversation_boundaries "
        "WHERE person_id = ? AND behavior = ? AND topic = ?",
        (int(person_id), behavior, topic),
    )
    if existing:
        db.execute(
            "UPDATE person_conversation_boundaries "
            "SET description = ?, source_text = ?, active = 1, updated_at = ? "
            "WHERE id = ?",
            (desc, source_text.strip(), now, int(existing["id"])),
        )
        return int(existing["id"])
    return db.execute(
        "INSERT INTO person_conversation_boundaries "
        "(person_id, behavior, topic, description, source_text, active, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
        (int(person_id), behavior, topic, desc, source_text.strip(), now, now),
    )


def deactivate_boundary(person_id: int, behavior: str, topic: str) -> None:
    db.execute(
        "UPDATE person_conversation_boundaries SET active = 0, updated_at = ? "
        "WHERE person_id = ? AND behavior = ? AND topic = ?",
        (_now(), int(person_id), _normalize_behavior(behavior), _normalize_topic(topic)),
    )


def get_boundaries(person_id: int, active_only: bool = True) -> list[dict]:
    clause = "AND active = 1 " if active_only else ""
    rows = db.fetchall(
        "SELECT * FROM person_conversation_boundaries "
        "WHERE person_id = ? "
        f"{clause}"
        "ORDER BY updated_at DESC, created_at DESC",
        (int(person_id),),
    )
    return [dict(r) for r in rows]


def summarize_for_prompt(person_id: int) -> str:
    rows = get_boundaries(person_id, active_only=True)
    if not rows:
        return ""
    lines = [
        f"- {row['description']}"
        for row in rows[:8]
        if row.get("description")
    ]
    if not lines:
        return ""
    return (
        "Conversation boundaries/preferences for this person:\n"
        + "\n".join(lines)
        + "\nThese are consent boundaries, not jokes. Follow them even when roasting."
    )


def is_blocked(person_id: int, behavior: str, topic: str) -> bool:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    for row in get_boundaries(person_id, active_only=True):
        row_behavior = row.get("behavior") or ""
        row_topic = row.get("topic") or ""
        if row_behavior not in {behavior, "mention"} and behavior != "roast":
            continue
        if behavior == "roast" and row_behavior not in {"roast", "mention"}:
            continue
        if _topics_overlap(topic, row_topic):
            return True
    return False


def detect_boundary(
    text: str,
    *,
    fallback_topic: Optional[str] = None,
) -> Optional[dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    clear = _CLEAR_PAT.search(cleaned)
    if clear:
        return {
            "action": "clear",
            "behavior": _normalize_behavior(clear.group("behavior")),
            "topic": _normalize_topic(clear.group("topic")),
            "source_text": cleaned,
        }

    for behavior, pattern in _BOUNDARY_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        topic = _normalize_topic(match.groupdict().get("topic") or fallback_topic or _DEFAULT_TOPIC)
        return {
            "action": "add",
            "behavior": behavior,
            "topic": topic,
            "description": _description_for(behavior, topic),
            "source_text": cleaned,
        }
    return None


def apply_detected_boundary(person_id: int, detected: dict) -> Optional[dict]:
    if not detected:
        return None
    action = detected.get("action")
    behavior = detected.get("behavior") or "mention"
    topic = detected.get("topic") or _DEFAULT_TOPIC
    if action == "clear":
        deactivate_boundary(person_id, behavior, topic)
        _log.info(
            "[boundaries] cleared boundary person_id=%s behavior=%s topic=%s",
            person_id, behavior, topic,
        )
        return {"action": "clear", "behavior": behavior, "topic": topic}
    if action == "add":
        row_id = add_boundary(
            person_id,
            behavior,
            topic,
            description=detected.get("description"),
            source_text=detected.get("source_text", ""),
        )
        _log.info(
            "[boundaries] saved boundary id=%s person_id=%s behavior=%s topic=%s",
            row_id, person_id, behavior, topic,
        )
        return {"action": "add", "id": row_id, "behavior": behavior, "topic": topic}
    return None


def _normalize_behavior(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"tease", "joke about", "make fun of"}:
        return "roast"
    if v in {"talk about", "bring up", "comment on"}:
        return "mention"
    if v in {"question"}:
        return "ask"
    return v or "mention"


def _normalize_topic(value: str) -> str:
    topic = (value or _DEFAULT_TOPIC).strip().lower()
    topic = re.sub(r"^(me\s+)?(about|for|over|on)\s+", "", topic)
    topic = re.sub(r"^(my|the|that|it|this)\s+", "", topic)
    topic = _TRAILING_JUNK.sub("", topic).strip()
    topic = re.sub(r"\s+", " ", topic)
    return _TOPIC_ALIASES.get(topic, topic or _DEFAULT_TOPIC)


def _description_for(behavior: str, topic: str) -> str:
    behavior = _normalize_behavior(behavior)
    topic = _normalize_topic(topic)
    if topic == "how are you":
        return "Do not proactively ask how they are doing."
    if behavior == "roast":
        return f"Do not roast or tease them about {topic}."
    if behavior == "ask":
        return f"Do not proactively ask them about {topic}."
    return f"Do not proactively mention or comment on {topic}."


def _topics_overlap(a: str, b: str) -> bool:
    a = _normalize_topic(a)
    b = _normalize_topic(b)
    if a == b:
        return True
    clusters = [
        {"appearance", "body", "clothing", "hair", "shirt", "clothes"},
        {"work", "job", "boss", "office"},
        {"how are you", "feelings", "mood", "check in"},
    ]
    return any(a in cluster and b in cluster for cluster in clusters)
