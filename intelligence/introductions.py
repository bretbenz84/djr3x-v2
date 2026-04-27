"""
intelligence/introductions.py - explicit social introduction handling.

This is deliberately separate from generic unknown-face curiosity. When a known
person says "this is my partner JT" or "I'd like you to meet my coworker", Rex
should treat it as an introduction, not as random small talk.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Optional


INTRO_CONTEXT_TTL_SECS = 45.0
INTRO_FOLLOWUP_TTL_SECS = 90.0

_REL_WORDS = (
    "friend|best friend|father|dad|mother|mom|parent|coworker|co-worker|"
    "colleague|boss|supervisor|manager|aunt|uncle|partner|girlfriend|"
    "boyfriend|fiancee|fiance|wife|husband|spouse|sister|brother|sibling|cousin|"
    "roommate|neighbor|neighbour|dog|cat|pet"
)
_INTRO_PAT = re.compile(
    rf"\b("
    rf"i'?d like (you )?to meet|i would like (you )?to meet|"
    rf"let me introduce|introduce you to|meet my|meet our|"
    rf"this is my|this is our|this is|that'?s my|that is my|"
    rf"that'?s our|that is our|say hi to"
    rf")\b",
    re.IGNORECASE,
)
_NAME_TOKEN_PAT = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")
_DECLINE_PAT = re.compile(
    r"\b(never mind|don'?t worry|do not worry|forget it|no one|nobody|"
    r"not important|skip it)\b",
    re.IGNORECASE,
)

_PET_RELATIONSHIPS = {"dog", "cat", "pet"}
_REL_NORMALIZE = {
    "best friend": "best_friend",
    "co-worker": "coworker",
    "colleague": "coworker",
    "dad": "father",
    "mom": "mother",
    "fiancee": "fiance",
    "manager": "supervisor",
    "neighbour": "neighbor",
}


@dataclass
class IntroductionParse:
    is_introduction: bool
    name: Optional[str] = None
    relationship: Optional[str] = None
    subject_kind: str = "person"
    needs_name: bool = False
    confidence: float = 0.0
    reason: str = ""


def detect(text: str, *, has_unknown_face: bool = False) -> IntroductionParse:
    cleaned = (text or "").strip()
    if not cleaned:
        return IntroductionParse(False, reason="empty")
    if _DECLINE_PAT.search(cleaned):
        return IntroductionParse(False, reason="decline")

    intro_match = _INTRO_PAT.search(cleaned)
    if not intro_match:
        return IntroductionParse(False, reason="no intro cue")

    parsed = _parse_intro_text(cleaned)
    if parsed.name or parsed.relationship:
        parsed.is_introduction = True
        parsed.confidence = max(parsed.confidence, 0.85)
        parsed.reason = parsed.reason or "intro cue with name/relationship"
        return parsed

    # "I'd like you to meet somebody" has no name yet, but if a mystery face is
    # visible it should open an introduction slot instead of generic chat.
    if has_unknown_face or re.search(r"\b(someone|somebody|a friend|my friend)\b", cleaned, re.I):
        return IntroductionParse(
            True,
            needs_name=True,
            confidence=0.75,
            reason="intro cue without name",
        )

    return IntroductionParse(False, reason="intro cue too vague")


def parse_pending_answer(
    text: str,
    *,
    default_relationship: Optional[str] = None,
) -> IntroductionParse:
    cleaned = (text or "").strip()
    if not cleaned or _DECLINE_PAT.search(cleaned):
        return IntroductionParse(False, reason="decline/empty")

    parsed = _parse_intro_text(cleaned)
    if not parsed.name:
        bare = _normalize_name(cleaned)
        if bare:
            parsed.name = bare
    if not parsed.relationship and default_relationship:
        parsed.relationship = _normalize_relationship(default_relationship)
    parsed.subject_kind = _subject_kind(parsed.relationship)
    parsed.is_introduction = bool(parsed.name or parsed.relationship)
    parsed.needs_name = not bool(parsed.name)
    parsed.confidence = 0.85 if parsed.name else 0.45
    parsed.reason = "pending intro answer"
    return parsed


def should_capture_followup(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or _DECLINE_PAT.search(cleaned):
        return False
    words = re.findall(r"[A-Za-z']+", cleaned)
    return len(words) >= 3


def _parse_intro_text(text: str) -> IntroductionParse:
    rel = None
    name = None

    patterns = [
        rf"\b(?:this is|that'?s|that is|meet|say hi to)\s+(?:my|our)\s+(?P<rel>{_REL_WORDS})(?:[\s,]+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}}))?",
        rf"\b(?:i'?d like (?:you )?to meet|i would like (?:you )?to meet|introduce you to|let me introduce(?: you to)?)\s+(?:my|our)\s+(?P<rel>{_REL_WORDS})(?:[\s,]+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}}))?",
        rf"\b(?:this is|meet)\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}}),?\s+(?:my|our)\s+(?P<rel>{_REL_WORDS})\b",
        rf"\b(?:my|our)\s+(?P<rel>{_REL_WORDS})[\s,]+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
        rf"\b(?:i'?d like to introduce you to|i would like to introduce you to|(?:i'?m|i am) going to introduce you to|introduce you to|let me introduce you to)\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
        rf"\b(?:his|her|their)\s+name\s+is\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
        rf"\b(?:it'?s|it is|that'?s|that is)\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
        rf"\b(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}}),?\s+(?:my|our)\s+(?P<rel>{_REL_WORDS})\b",
        rf"\b(?:this is|meet|say hi to)\s+(?P<name>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){{0,2}})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        rel = _normalize_relationship((m.groupdict().get("rel") or "").strip())
        name = _normalize_name((m.groupdict().get("name") or "").strip())
        break

    if rel in _PET_RELATIONSHIPS:
        # "this is my dog" without a name should not store "Dog" as a person.
        if name and name.lower() in _PET_RELATIONSHIPS:
            name = None

    return IntroductionParse(
        is_introduction=bool(name or rel),
        name=name,
        relationship=rel,
        subject_kind=_subject_kind(rel),
        needs_name=not bool(name),
        confidence=0.8 if (name or rel) else 0.0,
        reason="parsed intro text" if (name or rel) else "",
    )


def _normalize_relationship(value: str) -> Optional[str]:
    rel = (value or "").strip().lower().replace("-", " ")
    rel = re.sub(r"\s+", " ", rel)
    if not rel:
        return None
    return _REL_NORMALIZE.get(rel, rel).replace(" ", "_")


def _subject_kind(relationship: Optional[str]) -> str:
    if relationship in _PET_RELATIONSHIPS:
        return "pet"
    return "person"


def _normalize_name(value: str) -> Optional[str]:
    text = (value or "").strip()
    if not text:
        return None
    text = re.split(r"[,.!?;:]", text, maxsplit=1)[0].strip()
    tokens = []
    for raw in text.split():
        token = re.sub(r"[^A-Za-z'\-]", "", raw).strip("'-")
        if token:
            tokens.append(token)
    if not tokens or len(tokens) > 3:
        return None
    if any(t.lower() in {"my", "our", "name", "is", "this", "meet"} for t in tokens):
        return None
    if len(tokens) == 1 and tokens[0].lower() in {"friend", "someone", "somebody"}:
        return None
    if all(t.islower() for t in tokens):
        tokens = [t.capitalize() for t in tokens]
    return " ".join(tokens)


def context_fresh(ctx: Optional[dict], *, now: Optional[float] = None) -> bool:
    if not ctx:
        return False
    now = time.monotonic() if now is None else now
    return (now - float(ctx.get("asked_at") or ctx.get("created_at") or 0.0)) <= INTRO_CONTEXT_TTL_SECS


def followup_fresh(ctx: Optional[dict], *, now: Optional[float] = None) -> bool:
    if not ctx:
        return False
    now = time.monotonic() if now is None else now
    return (now - float(ctx.get("asked_at") or 0.0)) <= INTRO_FOLLOWUP_TTL_SECS
