"""
conversation_steering.py — interest-led conversation continuity.

When someone says they are into a topic, Rex should treat that as an invitation
to talk about the thing they actually enjoy. This module keeps that steering
lightweight: detect explicit interest declarations, remember the active topic
per person, and provide prompt directives that encourage skill/knowledge
curiosity instead of generic interview questions.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from memory import boundaries as boundary_memory
from memory import facts as facts_memory

_log = logging.getLogger(__name__)

_TTL_SECS = 15 * 60
_MAX_TOPIC_CHARS = 80
_TRAILING_JUNK = re.compile(
    r"\s+(?:a lot|so much|these days|right now|lately|for fun|as a hobby)\.?$",
    re.IGNORECASE,
)
_BAD_TOPIC = {
    "it", "that", "this", "things", "stuff", "you", "him", "her", "them",
    "myself", "everything", "nothing",
}

_INTEREST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:i\s*(?:really\s*)?(?:like|love|enjoy|dig)|"
        r"i'?m\s+(?:really\s+)?into|i\s+am\s+(?:really\s+)?into|"
        r"i'?m\s+(?:really\s+)?obsessed\s+with|"
        r"i\s+am\s+(?:really\s+)?obsessed\s+with|"
        r"i\s+do\s+(?!not\b))(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<topic>[^.?!,;]{3,90})\s+is\s+my\s+"
        r"(?:favorite|favourite|hobby|main hobby|thing)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:favorite|favourite)\s+(?:thing|hobby|subject|topic|activity)"
        r"\s+is\s+(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+hobby\s+is|one\s+of\s+my\s+hobbies\s+is)\s+"
        r"(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
]

_AVOID_PAT = re.compile(
    r"\b(?:don'?t|do not|stop|no more|not)\s+"
    r"(?:talk|ask|bring|mention|continue)\b|"
    r"\b(?:change the subject|talk about something else|drop it)\b",
    re.IGNORECASE,
)
_SUBSTANTIVE_PAT = re.compile(
    r"\b(?:because|actually|usually|started|learned|built|made|work|client|"
    r"camera|printer|print|style|cut|color|design|process|favorite|hardest|"
    r"best|worst|trick|technique|gear|tool)\b",
    re.IGNORECASE,
)


@dataclass
class SteeringContext:
    topic: str
    source: str
    fresh: bool
    fact_key: str
    directive: str


_active: dict[Optional[int], dict] = {}


def clear(person_id: Optional[int] = None) -> None:
    if person_id is None:
        _active.clear()
    else:
        _active.pop(person_id, None)


def detect_interest(text: str) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    if _AVOID_PAT.search(cleaned):
        return None
    for pat in _INTEREST_PATTERNS:
        match = pat.search(cleaned)
        if not match:
            continue
        topic = _clean_topic(match.group("topic"))
        if topic:
            return topic
    return None


def note_user_turn(
    person_id: Optional[int],
    text: str,
    *,
    suppress_memory_learning: bool = False,
) -> Optional[SteeringContext]:
    """Update interest steering and persist explicit interests/notes."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if _AVOID_PAT.search(cleaned):
        clear(person_id)
        return None

    topic = detect_interest(cleaned)
    fresh = bool(topic)
    if topic:
        if person_id is not None and _topic_blocked(int(person_id), topic):
            clear(person_id)
            return None
        _active[person_id] = {
            "topic": topic,
            "ts": time.monotonic(),
            "source": "explicit_interest",
        }
        if person_id is not None and not suppress_memory_learning:
            _store_interest_fact(person_id, topic, source="interest_declaration")
    else:
        active = _read_active(person_id)
        topic = active.get("topic") if active else None
        if not topic:
            return None

    if person_id is not None and not suppress_memory_learning:
        _maybe_store_interest_note(person_id, topic, cleaned, fresh=fresh)

    return build_context(person_id, topic=topic, fresh=fresh)


def build_context(
    person_id: Optional[int],
    *,
    topic: Optional[str] = None,
    fresh: bool = False,
) -> Optional[SteeringContext]:
    active = _read_active(person_id)
    resolved_topic = topic or (active.get("topic") if active else None)
    if not resolved_topic:
        return None
    if person_id is not None and _topic_blocked(person_id, resolved_topic):
        return None
    fact_key = _interest_key(resolved_topic)
    source = "explicit_interest" if fresh else ((active or {}).get("source") or "known_interest")
    return SteeringContext(
        topic=resolved_topic,
        source=source,
        fresh=fresh,
        fact_key=fact_key,
        directive=_directive_for(resolved_topic, fresh=fresh),
    )


def build_directive(person_id: Optional[int], user_text: str) -> str:
    ctx = note_user_turn(person_id, user_text)
    return ctx.directive if ctx else ""


def _read_active(person_id: Optional[int]) -> Optional[dict]:
    active = _active.get(person_id)
    if not active:
        return None
    if time.monotonic() - float(active.get("ts") or 0.0) > _TTL_SECS:
        _active.pop(person_id, None)
        return None
    return active


def _directive_for(topic: str, *, fresh: bool) -> str:
    lead = (
        "The human just volunteered a genuine interest"
        if fresh else
        "The current thread matches a known/active interest"
    )
    return (
        f"Conversation steering: {lead}: {topic!r}. Keep this turn steered "
        "toward that subject unless the human asks for something else. Rex should "
        "sound curious about their skill, taste, tools, process, or knowledge. "
        "Use the main LLM to add one compact subject-specific observation or "
        "'did you know' style tidbit when you can do it confidently, then ask at "
        "most one natural follow-up about their experience with it. Keep it "
        "funny and in-character; light roasts are allowed only about the hobby "
        "or Rex's ignorance, not the person's competence."
    )


def _store_interest_fact(person_id: int, topic: str, *, source: str) -> None:
    try:
        facts_memory.add_fact(
            int(person_id),
            "interest",
            _interest_key(topic),
            topic,
            source,
            confidence=0.95,
        )
        _log.info(
            "[conversation_steering] stored interest person_id=%s topic=%r",
            person_id,
            topic,
        )
    except Exception as exc:
        _log.debug("interest fact save failed: %s", exc)


def _maybe_store_interest_note(
    person_id: int,
    topic: str,
    text: str,
    *,
    fresh: bool,
) -> None:
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < 5:
        return
    if "?" in text and not fresh:
        return
    if not fresh and not _SUBSTANTIVE_PAT.search(text):
        return
    try:
        facts_memory.add_fact(
            int(person_id),
            "interest_note",
            _interest_note_key(topic),
            text[:220],
            "interest_thread",
            confidence=0.75 if fresh else 0.85,
        )
    except Exception as exc:
        _log.debug("interest note save failed: %s", exc)


def _topic_blocked(person_id: int, topic: str) -> bool:
    try:
        return (
            boundary_memory.is_blocked(person_id, "ask", topic)
            or boundary_memory.is_blocked(person_id, "mention", topic)
            or boundary_memory.is_blocked(person_id, "ask", "questions")
        )
    except Exception as exc:
        _log.debug("interest boundary check failed: %s", exc)
        return False


def _clean_topic(topic: str) -> Optional[str]:
    cleaned = " ".join((topic or "").strip(" .?!,;:-").split())
    cleaned = re.sub(r"^(?:to|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _TRAILING_JUNK.sub("", cleaned).strip(" .?!,;:-")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _BAD_TOPIC:
        return None
    if len(cleaned) > _MAX_TOPIC_CHARS:
        cleaned = cleaned[:_MAX_TOPIC_CHARS].rsplit(" ", 1)[0].strip()
    return cleaned


def _slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug[:40] or hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]


def _interest_key(topic: str) -> str:
    return f"interest_{_slug(topic)}"


def _interest_note_key(topic: str) -> str:
    return f"interest_note_{_slug(topic)}"
