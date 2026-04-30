"""
intelligence/topic_thread.py - lightweight in-session topic continuity.

This module tracks the "soft thread" of the current conversation: what the
conversation is roughly about, whether the user seems engaged or avoidant, and
whether Rex has a question hanging in the air. It is intentionally heuristic and
session-local; durable memories belong in memory/*.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from typing import Optional


_AVOID_PAT = re.compile(
    r"\b(rather not|don'?t want to|do not want to|change (the )?subject|"
    r"talk about something else|drop it|leave it|not talk about|not now|"
    r"don'?t ask|do not ask|stop asking)\b",
    re.IGNORECASE,
)
_PLAYFUL_PAT = re.compile(r"\b(lol|haha|funny|joke|roast|kidding|teasing)\b", re.I)
_DEPTH_PAT = re.compile(r"\b(because|actually|honestly|i think|i feel|it was|we were)\b", re.I)
_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|is|are|am|should)\b",
    re.IGNORECASE,
)
_SHORT_CONFIRMATION_PAT = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|right|affirmative|sure|"
    r"no|nope|nah|negative)(?:[,.! ]|$)",
    re.IGNORECASE,
)
_POLAR_QUESTION_START = re.compile(
    r"^\s*(?:is|are|am|was|were|do|does|did|will|would|can|could|"
    r"should|have|has|had|didn'?t|don'?t|doesn'?t|isn'?t|aren'?t|"
    r"won'?t|wouldn'?t|can'?t|couldn'?t)\b",
    re.IGNORECASE,
)
_EXPLICIT_INTEREST_SWITCH_PAT = re.compile(
    r"\b("
    r"i (?:really )?(?:like|love|am into|enjoy)|"
    r"i'?d like to talk about|i would like to talk about|"
    r"let'?s talk about|my favorite|favorite kind of"
    r")\b",
    re.IGNORECASE,
)

_TOPIC_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("grief/loss", re.compile(r"\b(died|death|dead|passed|loss|grief|funeral)\b", re.I), "heavy"),
    ("health", re.compile(r"\b(sick|ill|hospital|health|doctor|pain|diagnos)\w*\b", re.I), "heavy"),
    ("work", re.compile(r"\b(work|job|office|boss|coworker|meeting|project)\b", re.I), "mild"),
    ("pets", re.compile(r"\b(dog|cat|pet|puppy|kitten)\b", re.I), "mild"),
    ("music", re.compile(r"\b(music|song|track|album|artist|band|dj|playlist)\b", re.I), "light"),
    ("family", re.compile(r"\b(mom|dad|parent|grandpa|grandma|kid|child|wife|husband|partner)\b", re.I), "mild"),
    ("visual detail", re.compile(r"\b(shirt|hat|jacket|screen|camera|room|desk|poster|light)\b", re.I), "light"),
    ("identity", re.compile(r"\b(name|who am i|who is|who'?s|recognize)\b", re.I), "light"),
    ("plans", re.compile(r"\b(plan|weekend|tomorrow|today|tonight|trip|event)\b", re.I), "light"),
]

_STOPWORDS = {
    "the", "and", "but", "for", "with", "that", "this", "you", "your",
    "about", "have", "just", "what", "when", "where", "yeah", "yes",
    "no", "not", "really", "pretty", "good", "okay", "like",
}


@dataclass
class TopicThread:
    label: str
    emotional_weight: str
    user_stance: str
    summary: str
    started_at: float
    updated_at: float
    turn_count: int = 0
    unresolved_question: Optional[str] = None
    last_user_text: str = ""
    last_assistant_question: str = ""


_current: Optional[TopicThread] = None


def clear() -> None:
    global _current
    _current = None


def snapshot() -> Optional[dict]:
    if _current is None:
        return None
    return asdict(_current)


def note_assistant_turn(text: str) -> None:
    """Remember Rex's latest question so the next user turn can answer it."""
    global _current
    cleaned = (text or "").strip()
    if not cleaned:
        return
    if _current is None:
        now = time.monotonic()
        _current = TopicThread(
            label="conversation",
            emotional_weight="light",
            user_stance="neutral",
            summary="conversation opened by Rex",
            started_at=now,
            updated_at=now,
        )
    if "?" in cleaned:
        question = _last_question_sentence(cleaned)
        _current.unresolved_question = question
        _current.last_assistant_question = question
    _current.updated_at = time.monotonic()


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    answered_question: Optional[dict] = None,
) -> None:
    del person_id  # reserved for later person-specific topic preferences
    global _current

    cleaned = (text or "").strip()
    if not cleaned:
        return

    unresolved_before = _current.unresolved_question if _current is not None else ""
    answers_unresolved = _answers_unresolved_question(cleaned, unresolved_before)
    now = time.monotonic()
    label, weight = _classify_topic(cleaned)
    stance = _classify_stance(
        cleaned,
        answered_question=answered_question,
        answers_unresolved=answers_unresolved,
    )

    if _current is None or _should_start_new_thread(cleaned, label, stance):
        _current = TopicThread(
            label=label,
            emotional_weight=weight,
            user_stance=stance,
            summary=_summarize_text(cleaned, label),
            started_at=now,
            updated_at=now,
            turn_count=1,
            last_user_text=cleaned,
        )
    else:
        _current.label = _merged_label(_current.label, label)
        _current.emotional_weight = _heavier(_current.emotional_weight, weight)
        _current.user_stance = stance
        _current.summary = _summarize_text(cleaned, _current.label)
        _current.updated_at = now
        _current.turn_count += 1
        _current.last_user_text = cleaned

    if answered_question or answers_unresolved or stance in {"engaged", "avoidant"}:
        _current.unresolved_question = None


def note_answered_question(answered_question: Optional[dict] = None) -> None:
    """Mark Rex's outstanding question as answered without adding another turn."""
    if _current is None:
        return
    _current.unresolved_question = None
    _current.user_stance = "engaged"
    _current.updated_at = time.monotonic()
    if answered_question:
        answer = (answered_question.get("answer_text") or "").strip()
        if answer:
            _current.summary = _summarize_text(answer, _current.label)


def build_directive() -> str:
    if _current is None:
        return ""
    age = time.monotonic() - _current.updated_at
    if age > 300:
        return ""

    lines = [
        "Topic thread: keep continuity with the current conversational thread.",
        f"Current topic: {_current.label}.",
        f"Thread summary: {_current.summary}.",
        f"User stance: {_current.user_stance}; emotional weight: {_current.emotional_weight}.",
    ]
    if _current.unresolved_question:
        lines.append(
            f"Rex's unresolved question: {_current.unresolved_question!r}. "
            "Treat the user's latest utterance as a likely answer if it fits; "
            "do not ask an unrelated new question in the same breath."
        )
    if _current.user_stance == "avoidant":
        lines.append(
            "The user is steering away from this topic. Briefly acknowledge the "
            "boundary and let the topic drop unless they reopen it."
        )
    elif _current.user_stance == "terse":
        lines.append(
            "The user gave a short/low-energy reply. Do not interrogate. Either "
            "leave space, make one gentle follow-up, or shift softly."
        )
    elif _current.user_stance == "playful":
        lines.append(
            "The user is playful. Banter is welcome, but keep the thread connected "
            "instead of random-topic hopping."
        )
    elif _current.user_stance == "engaged":
        lines.append(
            "The user is engaging with the thread. Continue or deepen this topic "
            "before introducing anything new."
        )
    if _current.emotional_weight == "heavy":
        lines.append(
            "This topic is emotionally heavy. Prioritize care, consent, and pacing; "
            "no roasts about the vulnerable subject."
        )
    return "\n".join(lines)


def _classify_topic(text: str) -> tuple[str, str]:
    for label, pat, weight in _TOPIC_PATTERNS:
        if pat.search(text):
            return label, weight
    keywords = _keywords(text)
    if keywords:
        return " / ".join(keywords[:2]), "light"
    return "current exchange", "light"


def _classify_stance(
    text: str,
    *,
    answered_question: Optional[dict],
    answers_unresolved: bool = False,
) -> str:
    if _AVOID_PAT.search(text):
        return "avoidant"
    if _PLAYFUL_PAT.search(text):
        return "playful"
    words = re.findall(r"[A-Za-z']+", text)
    if answered_question or answers_unresolved or len(words) >= 8 or _DEPTH_PAT.search(text):
        return "engaged"
    if len([w for w in words if len(w) > 2]) <= 2:
        return "terse"
    return "neutral"


def _should_start_new_thread(text: str, label: str, stance: str) -> bool:
    if _current is None:
        return True
    if stance == "avoidant":
        return False
    if label == _current.label or label == "current exchange":
        return False
    if _current.emotional_weight == "heavy" and label not in {"grief/loss", "health"}:
        return _looks_like_explicit_switch(text)
    if _current.turn_count <= 1:
        return False
    return _looks_like_explicit_switch(text) or len(text.split()) >= 6


def _looks_like_explicit_switch(text: str) -> bool:
    lowered = text.lower()
    return (
        "speaking of" in lowered
        or "by the way" in lowered
        or "anyway" in lowered
        or "new subject" in lowered
        or "let's talk about" in lowered
        or bool(_EXPLICIT_INTEREST_SWITCH_PAT.search(text))
        or bool(_QUESTION_START.search(text))
    )


def _merged_label(current: str, incoming: str) -> str:
    if incoming in {"current exchange", current}:
        return current
    if current == "conversation":
        return incoming
    return current


def _heavier(a: str, b: str) -> str:
    order = {"light": 0, "mild": 1, "heavy": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _summarize_text(text: str, label: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > 140:
        cleaned = cleaned[:137].rstrip() + "..."
    return f"{label}: {cleaned}"


def _keywords(text: str) -> list[str]:
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", text)
        if w.lower() not in _STOPWORDS
    ]
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def _last_question_sentence(text: str) -> str:
    parts = re.findall(r"[^?]*\?", text)
    if not parts:
        return text[-180:]
    return parts[-1].strip()[-180:]


def _answers_unresolved_question(text: str, question: Optional[str]) -> bool:
    cleaned = (text or "").strip()
    q = (question or "").strip()
    if not cleaned or not q or "?" in cleaned:
        return False
    if _AVOID_PAT.search(cleaned):
        return True
    words = re.findall(r"[A-Za-z']+", cleaned)
    if len(words) >= 8 or _DEPTH_PAT.search(cleaned):
        return True
    if _SHORT_CONFIRMATION_PAT.match(cleaned) and _is_polar_or_tag_question(q):
        return True
    return False


def _is_polar_or_tag_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if _POLAR_QUESTION_START.match(q):
        return True
    return bool(
        re.search(
            r"(?:,\s*)?(?:right|correct|yeah|yes|no|okay|ok|huh)\?\s*$",
            q,
            re.IGNORECASE,
        )
    )
