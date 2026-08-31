"""
intelligence/introductions.py - explicit social introduction handling.

This is deliberately separate from generic unknown-face curiosity. When a known
person says "this is my partner Alex" or "I'd like you to meet my coworker", Rex
should treat it as an introduction, not as random small talk.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Optional

from memory.name_validation import normalize_person_name


INTRO_CONTEXT_TTL_SECS = 45.0
INTRO_FOLLOWUP_TTL_SECS = 90.0

_REL_WORDS = (
    "friend|best friend|father|dad|mother|mom|parent|coworker|co-worker|"
    "colleague|boss|supervisor|manager|aunt|uncle|nephew|niece|partner|girlfriend|"
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


def should_capture_followup(
    text: str,
    *,
    introduced_name: Optional[str] = None,
) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or _DECLINE_PAT.search(cleaned):
        return False
    if denies_introduction(cleaned, introduced_name=introduced_name):
        return False
    words = re.findall(r"[A-Za-z']+", cleaned)
    return len(words) >= 3


# A reply that DENIES the introduction's premise is not connection color, and it
# is not the newcomer's voice sample either — it is the human telling Rex he got
# the whole frame wrong. Field 2026-08-29 11:21-11:22: Rex was told "say hi to
# PJ", opened a voice-capture window, enrolled BRET'S clip onto PJ (person 7,
# biometric 56), asked how they know each other, and then filed Bret's correction
# "PJ is not here. This is Bret." as the Bret<->PJ connection story (person_facts
# 132/133). Every one of those writes was downstream of not reading a denial as a
# denial.
_PREMISE_DENIAL_PAT = re.compile(
    r"(?:"
    r"\b(?:is|was|are|were)\s+not\s+(?:here|there|around|present|in\s+the\s+room)\b|"
    r"\b(?:isn'?t|wasn'?t|aren'?t|weren'?t|ain'?t)\s+(?:here|there|around|present|in\s+the\s+room)\b|"
    r"\bnot\s+(?:here|there|around)\s+(?:right\s+now|anymore|any\s+more|yet)\b|"
    r"\b(?:nobody|no\s+one)\s+(?:else\s+)?(?:is\s+)?(?:here|there|around)\b|"
    r"\bthere(?:'?s|\s+is)\s+no\s+(?:one|body)\b|"
    r"\bwrong\s+(?:person|name|guy|girl|voice|one)\b|"
    r"\b(?:you'?ve|you\s+have)\s+got\s+the\s+wrong\b|"
    r"\b(?:that\s+was|that'?s|this\s+is|it'?s|it\s+is)\s+(?:still\s+|just\s+)*me\b|"
    r"\b(?:still|only)\s+(?:just\s+)?me\s+(?:here|talking)\b|"
    r"\bjust\s+me\s+(?:here|in\s+here)\b|"
    r"\bi'?m\s+the\s+only\s+one\b"
    r")",
    re.IGNORECASE,
)

# "This is Bret." / "I'm Bret." during PJ's window: the speaker is naming
# THEMSELF as somebody other than the person Rex is capturing. Case-sensitive on
# the name token on purpose — the ASR capitalizes proper nouns, and a lowercase
# match would swallow ordinary sentences ("it's not that big a deal").
_SELF_NAME_CLAIM_PAT = re.compile(
    r"\b(?i:this is|that is|that'?s|it'?s|it is|i'?m|i am)\s+"
    r"(?P<name>[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)?)\b"
)


def _first_token(name: Optional[str]) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", name or "")
    return tokens[0].lower() if tokens else ""


def denies_introduction(
    text: str,
    *,
    introduced_name: Optional[str] = None,
) -> bool:
    """True when a reply DENIES the introduction's premise instead of answering it.

    Covers three shapes, in rising specificity:
      1. presence/identity denial with no name needed ("she isn't here",
         "wrong person", "that was me"),
      2. the newcomer named as absent ("PJ is not here", "that's not PJ"),
      3. the speaker naming THEMSELF as somebody else ("This is Bret").

    Callers pass ``introduced_name`` when the open window knows who it is
    capturing; without it only shape 1 can fire.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _PREMISE_DENIAL_PAT.search(cleaned):
        return True

    first = _first_token(introduced_name)
    if not first:
        return False
    esc = re.escape(first)
    if re.search(
        rf"\b{esc}\s+(?:is|was)\s+(?:not|no\s+longer)\b|"
        rf"\b{esc}\s+(?:isn'?t|wasn'?t|ain'?t)\b|"
        rf"\bnot\s+{esc}\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True

    claim = _SELF_NAME_CLAIM_PAT.search(cleaned)
    if claim:
        claimed = _normalize_name(claim.group("name"))
        if claimed and _first_token(claimed) != first:
            return True
    return False


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
    return normalize_person_name(value, allow_single=True)


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


# When the user actively introduces a newcomer ("this is my partner JT"), Rex is already
# capturing that identity — so the "urgent group identity handoff / who's the mystery
# guest?" agenda should stand down for a window instead of badgering on every turn while
# voice/face enrollment catches up (the JT run looped that question ~5 times).
_last_introduction_at: float = 0.0


def note_introduction(*, now: Optional[float] = None) -> None:
    """Mark that an explicit introduction just happened (for intro_recent)."""
    global _last_introduction_at
    _last_introduction_at = time.monotonic() if now is None else now


def intro_recent(within_secs: float = 45.0, *, now: Optional[float] = None) -> bool:
    """True if an explicit introduction happened within the last ``within_secs``."""
    if _last_introduction_at <= 0.0:
        return False
    now = time.monotonic() if now is None else now
    return (now - _last_introduction_at) <= max(0.0, float(within_secs))
