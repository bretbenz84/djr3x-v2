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
from memory import interests as interests_memory

_log = logging.getLogger(__name__)

# How long an "active interest" stays steered before it goes stale. 15 min was
# long enough that Rex kept steering toward a topic the conversation had clearly
# moved on from; 8 min still covers a continued conversation.
_TTL_SECS = 8 * 60
_MAX_TOPIC_CHARS = 80
_TRAILING_JUNK = re.compile(
    r"\s+(?:a lot|so much|these days|right now|lately|for fun|as a hobby)\.?$",
    re.IGNORECASE,
)
_BAD_TOPIC = {
    "it", "that", "this", "things", "stuff", "you", "him", "her", "them",
    "myself", "everything", "nothing",
}
_BARE_TOPIC_MAX_WORDS = 6

# Profile/curiosity questions whose answer names a thing the person is into, so a
# short reply ("astrophotography") is a high-value topic seed to deepen — not a
# low-energy throwaway. Keys match config.QUESTION_POOL. Emotional/biographical
# keys (proudest_moment, fears, values, …) are deliberately excluded.
# NOTE: "favorite_music" is intentionally absent — that answer is owned by the
# music-offer flow ("Want me to play some classical?"), which is already an
# engaging response; seeding it as a steering topic would double-handle it.
INTEREST_SEED_QUESTION_KEYS = {
    "obsession",
    "hobbies",
    "favorite_movie",
    "travel",
    "job",
}
_SEED_REFUSAL_RE = re.compile(
    r"^(?:yes|yeah|yep|no|nope|nah|okay|ok|sure|nothing|none|"
    r"i don'?t know|dunno|idk|not sure|maybe|whatever)\.?$",
    re.IGNORECASE,
)

_INTEREST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:i\s*(?:really\s+)?(?:want|wanna|would like)\s+to\s+talk\s+about|"
        r"let'?s\s+talk\s+about|can\s+we\s+talk\s+about)\s+"
        r"(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s*(?:really\s*)?(?:like|love|enjoy|dig)|"
        r"i'?m\s+(?:really\s+)?into|i\s+am\s+(?:really\s+)?into|"
        r"i'?m\s+(?:really\s+)?obsessed\s+with|"
        r"i\s+am\s+(?:really\s+)?obsessed\s+with|"
        # "I do yoga / woodworking" is a hobby; "I do think/believe/need…" is an
        # emphatic AUXILIARY, not a hobby — exclude a following common verb so the
        # interest store isn't poisoned with sentence fragments ("think the weather…").
        r"i\s+do\s+(?!not\b)(?!(?:think|thought|believe|feel|felt|need|want|have|had|"
        r"know|knew|remember|recall|love|like|enjoy|hate|wish|hope|agree|disagree|"
        r"understand|care|mind|see|get|got|wonder|guess|suppose|appreciate|prefer|"
        r"miss|realize|realise|notice|admit|expect|find|found|consider|mean|meant|"
        r"say|said|tell|told|talk|speak|spoke|plan|intend|tend|happen|seem|sound|"
        r"look|use|hope)\b))(?P<topic>[^.?!,;]{3,90})",
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
        r"\bmy\s+(?:favorite|favourite)\s+kind\s+of\s+"
        r"(?P<category>[^.?!,;]{3,40})\s+is\s+(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+hobby\s+is|one\s+of\s+my\s+hobbies\s+is)\s+"
        r"(?P<topic>[^.?!,;]{3,90})",
        re.IGNORECASE,
    ),
    # "I'm building/making/working on X" — a thing they're actively into, so the
    # active topic UPDATES instead of getting stuck on an earlier interest (the
    # live "still steering toward Apple TV while we talk about the droid" bug).
    # Direct verb only (no "trying to") so "I'm trying to make him funny" is skipped.
    re.compile(
        r"\bi(?:'?m|\s+am)\s+(?:building|making|creating|designing|developing|"
        r"coding|programming|writing|painting|growing|restoring|fixing|"
        r"working\s+on|learning|studying)\s+"
        r"(?P<topic>(?:a|an|the|my)\s+[^.?!,;]{3,80}|[^.?!,;]{3,80})",
        re.IGNORECASE,
    ),
]
_TOPIC_KNOWLEDGE_PAT = re.compile(
    r"\b(?:what\s+do\s+you\s+know|do\s+you\s+know\s+anything|"
    r"tell\s+me|explain)\s+(?:about\s+)?(?P<topic>[^?.,!;]{3,90})",
    re.IGNORECASE,
)

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



# A bare low-content reply ("yeah", "sure", "I guess", "not really") signals the
# subject isn't generating elaboration. Two in a row on the same topic and Rex
# should pivot rather than keep probing. Must match the WHOLE utterance so a real
# short answer ("mostly nebulae") is not mistaken for disengagement.
_GENERIC_REPLY_RE = re.compile(
    r"^\s*(?:yeah|yep|yup|yes|sure|okay|ok|absolutely|totally|definitely|"
    r"i guess|i suppose|maybe|kinda|kind of|sorta|sort of|not really|nope|no|"
    r"nah|i don'?t know|i dunno|dunno|idk|not sure|cool|nice|fine|fair|"
    r"true|right|exactly|pretty much|i think so|sometimes|mostly|whatever|"
    r"meh|whatever you say)\b[\s.!,]*$",
    re.IGNORECASE,
)
# Consecutive disengaged turns on the same active interest before Rex pivots.
_PIVOT_AFTER_MISSES = 2


def _looks_disengaged(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if "?" in cleaned:  # the human asking something back is engagement
        return False
    if _AVOID_PAT.search(cleaned):
        return True
    return bool(_GENERIC_REPLY_RE.match(cleaned))


@dataclass
class SteeringContext:
    topic: str
    source: str
    fresh: bool
    fact_key: str
    directive: str
    # "deepen" = stay on / dig into the topic; "pivot" = the subject isn't
    # landing, swing to a related subject or open a new one.
    mode: str = "deepen"


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
        topic_text = match.group("topic")
        category = match.groupdict().get("category")
        if category:
            topic_text = f"{topic_text} {category}"
        topic = _clean_topic(topic_text)
        if topic:
            return topic
    return None


def detect_topic_question(text: str) -> Optional[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned or _AVOID_PAT.search(cleaned):
        return None
    match = _TOPIC_KNOWLEDGE_PAT.search(cleaned)
    if not match:
        return None
    return _clean_topic(match.group("topic"))


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

    # A compliment/affection statement ("I love you", "you're the best") is not an
    # interest declaration. Without this, the interest regex captured "you now"
    # from "I love you now" and persisted it as a steered interest at confidence
    # 0.95. Mirror the guard the reply path already uses (interaction.py). Do NOT
    # clear() — a warm aside shouldn't wipe a legitimate active topic.
    from intelligence import personality
    if personality.is_obvious_compliment(cleaned):
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
        topic = detect_topic_question(cleaned)
        if topic:
            if person_id is not None and _topic_blocked(int(person_id), topic):
                clear(person_id)
                return None
            _active[person_id] = {
                "topic": topic,
                "ts": time.monotonic(),
                "source": "topic_question",
            }
        else:
            active = _read_active(person_id)
            topic = active.get("topic") if active else None
            if not topic:
                return None
            # Track engagement: bare low-content replies mean the subject isn't
            # landing. After a couple in a row, pivot away and drop the topic so
            # Rex doesn't keep probing a dead subject.
            if _looks_disengaged(cleaned):
                active["misses"] = int(active.get("misses", 0)) + 1
            else:
                active["misses"] = 0
            if int(active.get("misses", 0)) >= _PIVOT_AFTER_MISSES:
                pivot_ctx = _build_pivot_context(person_id, topic)
                clear(person_id)
                return pivot_ctx

    if person_id is not None and not suppress_memory_learning:
        _maybe_store_interest_note(person_id, topic, cleaned, fresh=fresh)

    return build_context(person_id, topic=topic, fresh=fresh)


def note_bare_interest_answer(
    person_id: Optional[int],
    text: str,
    *,
    source: str = "interest_answer",
    suppress_memory_learning: bool = False,
) -> Optional[SteeringContext]:
    """Treat a short answer to Rex's opener as the topic they want to discuss."""
    topic = _clean_bare_topic(text)
    if not topic:
        return None
    if person_id is not None and _topic_blocked(int(person_id), topic):
        clear(person_id)
        return None
    _active[person_id] = {
        "topic": topic,
        "ts": time.monotonic(),
        "source": source,
    }
    if person_id is not None and not suppress_memory_learning:
        _store_interest_fact(int(person_id), topic, source=source)
        _maybe_store_interest_note(int(person_id), topic, text.strip(), fresh=True)
    return build_context(person_id, topic=topic, fresh=True)


def is_interest_seed_question(question_key: Optional[str]) -> bool:
    """True for profile questions whose answer names a deepenable interest."""
    return bool(question_key) and str(question_key) in INTEREST_SEED_QUESTION_KEYS


def looks_like_interest_seed_answer(text: str, question_key: Optional[str]) -> bool:
    """True when this turn answers an interest-seeking question with real content.

    Looser than ``_clean_bare_topic``: it accepts longer answers too, so the
    length/energy layers give the share room even when the exact topic slug is
    not extracted. A refusal ("I don't know") or boundary still returns False.
    """
    if not is_interest_seed_question(question_key):
        return False
    cleaned = " ".join((text or "").strip().split())
    if not cleaned or "?" in cleaned:
        return False
    if _SEED_REFUSAL_RE.match(cleaned) or _AVOID_PAT.search(cleaned):
        return False
    return bool(re.search(r"[A-Za-z]", cleaned))


def seed_from_answer(
    person_id: Optional[int],
    text: str,
    question_key: Optional[str],
    *,
    suppress_memory_learning: bool = False,
) -> Optional[SteeringContext]:
    """Register a short answer to one of Rex's interest questions as the active
    topic, so the steering machinery deepens it instead of letting the turn
    collapse into a 12-word throwaway. No-op for non-interest-seed keys."""
    if not is_interest_seed_question(question_key):
        return None
    return note_bare_interest_answer(
        person_id,
        text,
        source=f"interest_answer:{question_key}",
        suppress_memory_learning=suppress_memory_learning,
    )


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
        mode="deepen",
    )


def _build_pivot_context(person_id: Optional[int], topic: str) -> SteeringContext:
    """A 'this subject stalled — change the channel' steering context."""
    return SteeringContext(
        topic=topic,
        source="pivot",
        fresh=False,
        fact_key=_interest_key(topic),
        directive=_pivot_directive_for(topic),
        mode="pivot",
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
        "most one natural follow-up on a FRESH angle — e.g. what first got them "
        "into it, their favorite or most frustrating part, the best thing they've "
        "done with it, or what they're chasing next — and do not re-ask an angle "
        "you've already covered this conversation. Keep it "
        "funny and in-character; do not confuse franchises or fields as if they "
        "are the same thing, and ask instead of bluffing if you are unsure. "
        "If the topic is Star Trek, answer as Star Trek first; a Star Wars "
        "self-aware joke is okay only as a quick aside, never as the substance "
        "of the answer. "
        "If the human asked what you know about the topic, answer from general "
        "knowledge first instead of saying it is missing from personal memory. "
        "You may ask if this is a subject they are into before treating it as "
        "a remembered interest. "
        "Light roasts are allowed only about the hobby or Rex's ignorance, not "
        "the person's competence."
    )


def _pivot_directive_for(topic: str) -> str:
    category = _category_for_topic(topic)
    return (
        f"Conversation steering: {topic!r} has stopped landing — the human has "
        "given a couple of flat, low-energy replies on it, so STOP probing this "
        "subject. Change the channel naturally: give one brief reaction to what "
        "they just said, then either (a) swing to a RELATED subject — an adjacent "
        f"angle, a sibling hobby, or the wider {category} space around it — or (b) "
        "open a genuinely new topic / ask about something else they're into. Make "
        "it a smooth, in-character conversational turn, not an abrupt jump or an "
        "interrogation, and do NOT keep asking about the stalled subject."
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
    try:
        interests_memory.upsert_interest(
            int(person_id),
            topic,
            _category_for_topic(topic),
            "high",
            confidence=0.95,
            source="explicit",
        )
    except Exception as exc:
        _log.debug("typed interest save failed: %s", exc)


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
    try:
        interests_memory.upsert_interest(
            int(person_id),
            topic,
            _category_for_topic(topic),
            "medium",
            confidence=0.85 if not fresh else 0.75,
            source="explicit" if fresh else "inferred",
            notes=text[:220],
        )
    except Exception as exc:
        _log.debug("typed interest note save failed: %s", exc)


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


# Pronoun/function-word tokens that must never stand in for a real topic. The
# interest regex ("i love (?P<topic>...)") can capture affection or filler
# ("you now", "me too", "it now") — a real interest carries at least one content
# word. Without this gate, "I love you now" minted the steered interest "you now".
_TOPIC_FUNCTION_WORDS = {
    "i", "you", "we", "they", "he", "she", "it", "me", "us", "him", "her",
    "them", "that", "this", "these", "those", "my", "your", "our", "their",
    "his", "its", "a", "an", "the", "to", "of", "and", "or",
    "now", "then", "too", "also", "just", "really", "very", "here", "there",
    "again", "still", "much", "more", "what", "when", "so", "yeah", "okay",
}


def _topic_is_substantive(topic: str) -> bool:
    """Reject pure pronoun/function-word fragments the interest regex can capture
    from affection or filler ('you now', 'me too', 'it now'). A real topic must
    carry at least one token that is not a bare function word — short real nouns
    ('art', 'tea', 'cars') still pass."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'+#-]*", topic.lower())
    if not tokens:
        return False
    return any(t not in _TOPIC_FUNCTION_WORDS for t in tokens)


def _clean_topic(topic: str) -> Optional[str]:
    cleaned = " ".join((topic or "").strip(" .?!,;:-").split())
    cleaned = re.sub(r"^(?:to|the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _TRAILING_JUNK.sub("", cleaned).strip(" .?!,;:-")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in _BAD_TOPIC:
        return None
    if not _topic_is_substantive(cleaned):
        return None
    if len(cleaned) > _MAX_TOPIC_CHARS:
        cleaned = cleaned[:_MAX_TOPIC_CHARS].rsplit(" ", 1)[0].strip()
    return cleaned


def _clean_bare_topic(text: str) -> Optional[str]:
    cleaned = " ".join((text or "").strip(" .?!,;:-").split())
    if not cleaned or _AVOID_PAT.search(cleaned):
        return None
    if "?" in cleaned:
        return None
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'+#-]*", cleaned)
    if not words or len(words) > _BARE_TOPIC_MAX_WORDS:
        return None
    if len(words) > 3 and re.search(
        r"\b(?:i|you|we|they|he|she|did|didn'?t|do|don'?t|am|are|is|was|were)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return None
    lowered = cleaned.lower()
    if lowered in _BAD_TOPIC:
        return None
    if re.fullmatch(r"(?:yes|yeah|yep|no|nope|okay|ok|sure|nothing|i don'?t know)", lowered):
        return None
    return _clean_topic(cleaned)


def _slug(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug[:40] or hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]


def _interest_key(topic: str) -> str:
    return f"interest_{_slug(topic)}"


def _interest_note_key(topic: str) -> str:
    return f"interest_note_{_slug(topic)}"


def _category_for_topic(topic: str) -> str:
    lowered = (topic or "").lower()
    if any(word in lowered for word in ("star wars", "star trek", "marvel", "disney")):
        return "fandom"
    if any(word in lowered for word in ("volleyball", "soccer", "basketball", "football", "baseball")):
        return "sport"
    if any(word in lowered for word in ("3d print", "printing", "telescope", "robot", "droid", "coding", "programming")):
        return "technical"
    if any(word in lowered for word in ("music", "guitar", "piano", "band", "dj")):
        return "music"
    if any(word in lowered for word in ("camping", "hiking", "travel")):
        return "hobby"
    if any(word in lowered for word in ("art", "paint", "writing", "craft", "photo")):
        return "creative"
    return "hobby"
