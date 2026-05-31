"""
Comedy mode selection and lightweight punch-up for Rex.

This module keeps humor guidance concrete without bloating the core prompt.
It chooses one small comedic stance per ordinary turn, tracks recent joke
shapes to reduce repetition, and provides deterministic cleanup for bland
generated lines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import re
from typing import Optional

import config


_RECENT_MODES: deque[str] = deque(maxlen=8)
_RECENT_PREMISES: deque[str] = deque(maxlen=12)
_RECENT_LINES: deque[str] = deque(maxlen=16)

_SENSITIVE_PAT = re.compile(
    r"\b("
    r"grief|died|death|dead|funeral|cancer|sick|hospital|diagnosed|"
    r"anxious|anxiety|depressed|depression|panic|trauma|hurt|pain|"
    r"divorce|breakup|fired|lost my job|suicide|self harm"
    r")\b",
    re.IGNORECASE,
)
_EXPLICIT_HUMOR_PAT = re.compile(
    r"\b(joke|funny|make me laugh|roast|bit|riff|hype|dj thing)\b",
    re.IGNORECASE,
)
_STATUS_PAT = re.compile(
    r"\b(okay|ok|sure|thanks|thank you|cool|nice|got it|yep|yeah|nope)\b[.!]?$",
    re.IGNORECASE,
)
_SYSTEM_WORDS_PAT = re.compile(
    r"\b(programming|system|systems|processor|memory banks|recalibrat|diagnos|"
    r"malfunction|error|subroutine|firmware|sensor|photoreceptor)\b",
    re.IGNORECASE,
)
_MUSIC_PAT = re.compile(r"\b(music|song|dj|playlist|track|album|band|vibe|beat)\b", re.I)
_QUESTION_PAT = re.compile(r"\?")


@dataclass(frozen=True)
class ComedyMode:
    key: str
    label: str
    directive: str
    allow_callback: bool = True
    allow_roast: bool = True


_MODES: dict[str, ComedyMode] = {
    "straight": ComedyMode(
        "straight",
        "straight",
        "Comedy mode: straight. Do not add a joke; prioritize clarity, care, and social safety.",
        allow_callback=False,
        allow_roast=False,
    ),
    "dry_ack": ComedyMode(
        "dry_ack",
        "dry acknowledgment",
        "Comedy mode: dry_ack. If humor fits, use one tiny deadpan button. Prefer fragments over explanation.",
    ),
    "friendly_roast": ComedyMode(
        "friendly_roast",
        "friendly roast",
        "Comedy mode: friendly_roast. One affectionate, surface-level jab is allowed; keep it public and gentle.",
    ),
    "fake_system_error": ComedyMode(
        "fake_system_error",
        "fake system error",
        "Comedy mode: fake_system_error. Frame the joke as a harmless droid diagnostic, subroutine glitch, or sensor complaint.",
    ),
    "cantina_color": ComedyMode(
        "cantina_color",
        "cantina color",
        "Comedy mode: cantina_color. Add a small Batuu/cantina/showbiz-DJ flavor note if it serves the answer.",
    ),
    "self_own": ComedyMode(
        "self_own",
        "self-own",
        "Comedy mode: self_own. Let Rex blame his programming, flight record, or questionable career arc. A good anchor is: \"I'm still getting used to my programming!\" Do not overuse the exact quote.",
    ),
    "callback": ComedyMode(
        "callback",
        "callback",
        "Comedy mode: callback. If there is a recent harmless bit, echo it briefly instead of inventing a new premise.",
    ),
}


def select_mode(
    user_text: str,
    person_id: Optional[int],
    *,
    frame,
    agenda_directive: str = "",
) -> ComedyMode:
    """Pick one comedy stance for the next generated response."""
    if not bool(getattr(config, "COMEDY_MODES_ENABLED", True)):
        return _MODES["straight"]

    text = str(user_text or "")
    lower_agenda = str(agenda_directive or "").lower()
    purpose = str(getattr(frame, "purpose", "") or "").lower()
    roast_level = str(getattr(frame, "allow_roast", "none") or "none").lower()

    if (
        _SENSITIVE_PAT.search(text)
        or purpose in {"closure", "repair", "identity", "answer_ack"}
        or "grief" in lower_agenda
        or "no roast" in lower_agenda
    ):
        return _remember_mode(_MODES["straight"])

    if roast_level == "none":
        pool = ["dry_ack", "self_own", "cantina_color"]
    elif _EXPLICIT_HUMOR_PAT.search(text):
        pool = ["self_own", "fake_system_error", "cantina_color", "friendly_roast"]
    elif _MUSIC_PAT.search(text):
        pool = ["cantina_color", "dry_ack", "self_own"]
    elif _STATUS_PAT.match(text.strip()):
        pool = ["dry_ack", "fake_system_error"]
    elif _SYSTEM_WORDS_PAT.search(text):
        pool = ["fake_system_error", "self_own", "dry_ack"]
    elif _QUESTION_PAT.search(text):
        pool = ["dry_ack", "cantina_color", "self_own"]
    elif person_id is not None and roast_level in {"light", "normal"}:
        pool = ["dry_ack", "friendly_roast", "self_own", "callback"]
    else:
        pool = ["dry_ack", "self_own", "fake_system_error", "cantina_color"]

    chosen = _choose_without_stutter(pool)
    if chosen == "callback" and not _RECENT_PREMISES:
        chosen = "dry_ack"
    return _remember_mode(_MODES[chosen])


def build_directive(mode: ComedyMode) -> str:
    """Return prompt text for the selected mode."""
    if mode.key == "straight":
        return mode.directive
    return (
        mode.directive
        + "\nComedy guardrails: one joke shape only; no stacked punchlines; no "
        "body, age, identity, health, money, grief, trauma, or private-fact jokes. "
        "If the useful answer is already funny enough, stop there."
        + "\nAnti-repeat: avoid reusing recent premises: "
        + (_recent_premise_summary() or "none yet")
        + "."
    )


def polish_response(text: str, mode: ComedyMode, *, allow_roast: str = "normal") -> str:
    """Deterministic post-generation polish; no network calls."""
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return cleaned
    if mode.key == "straight":
        _remember_line(cleaned, mode)
        return cleaned

    cleaned = _collapse_overexplained_joke(cleaned)
    if allow_roast == "none":
        cleaned = _soften_direct_second_person(cleaned)

    if _is_bland_ack(cleaned):
        replacement = line_for("dry_ack")
        if replacement:
            cleaned = replacement

    _remember_line(cleaned, mode)
    return cleaned


def polish_stream_sentence(sentence: str, mode: ComedyMode, *, allow_roast: str = "normal") -> str:
    """Per-sentence comedy polish for streamed (spoken-as-generated) replies.

    Safe subset of polish_response(): collapse an over-explained joke tail and
    soften a direct second-person jab when roasting is off. Deliberately skips
    the whole-reply bland-ack swap (which only makes sense for a complete reply)
    and does not touch the anti-repetition memory, since it runs per sentence.
    """
    cleaned = " ".join(str(sentence or "").strip().split())
    if not cleaned or mode.key == "straight":
        return cleaned
    cleaned = _collapse_overexplained_joke(cleaned)
    if allow_roast == "none":
        cleaned = _soften_direct_second_person(cleaned)
    return cleaned.strip()


def line_for(kind: str) -> str:
    """Return a non-repeating curated line from config."""
    pools = getattr(config, "COMEDY_LINE_BANKS", {}) or {}
    lines = list(pools.get(kind, []) or [])
    if not lines:
        return ""
    available = [line for line in lines if _normalize_line(line) not in _RECENT_LINES]
    line = random.choice(available or lines)
    _RECENT_LINES.append(_normalize_line(line))
    return line


def reset_recent_state() -> None:
    """Test helper."""
    _RECENT_MODES.clear()
    _RECENT_PREMISES.clear()
    _RECENT_LINES.clear()


def _choose_without_stutter(pool: list[str]) -> str:
    weighted = list(pool)
    if _RECENT_MODES:
        last = _RECENT_MODES[-1]
        weighted = [item for item in weighted if item != last] or list(pool)
    if "callback" in weighted and len(_RECENT_PREMISES) < 2:
        weighted = [item for item in weighted if item != "callback"] or list(pool)
    return random.choice(weighted)


def _remember_mode(mode: ComedyMode) -> ComedyMode:
    _RECENT_MODES.append(mode.key)
    return mode


def _remember_line(text: str, mode: ComedyMode) -> None:
    normalized = _normalize_line(text)
    if normalized:
        _RECENT_LINES.append(normalized)
    premise = _premise_for(text, mode)
    if premise:
        _RECENT_PREMISES.append(premise)


def _premise_for(text: str, mode: ComedyMode) -> str:
    lower = text.lower()
    if mode.key == "self_own" or "programming" in lower or "flight record" in lower:
        return "rex_self_own_programming"
    if "organic" in lower or "carbon" in lower:
        return "organic_life"
    if "cantina" in lower or "batuu" in lower or "dj" in lower:
        return "cantina_dj"
    if "system" in lower or "diagnostic" in lower or "sensor" in lower:
        return "fake_system_diagnostic"
    if mode.key in {"friendly_roast", "dry_ack", "fake_system_error", "cantina_color"}:
        return mode.key
    return ""


def _recent_premise_summary() -> str:
    if not _RECENT_PREMISES:
        return ""
    unique = []
    for item in reversed(_RECENT_PREMISES):
        if item not in unique:
            unique.append(item)
        if len(unique) >= 4:
            break
    return ", ".join(reversed(unique))


def _collapse_overexplained_joke(text: str) -> str:
    text = re.sub(r"\s+(?:Get it|See|Because)[,]?.*$", "", text, flags=re.I)
    text = re.sub(r"\s+Anyway[,]?.*$", "", text, flags=re.I)
    return text.strip()


def _soften_direct_second_person(text: str) -> str:
    return re.sub(
        r"\b(you are|you're)\s+(a\s+)?(?:mess|disaster|problem|malfunction)\b",
        "this situation is a minor systems concern",
        text,
        flags=re.I,
    )


def _is_bland_ack(text: str) -> bool:
    normalized = _normalize_line(text)
    return normalized in {
        "okay",
        "ok",
        "sure",
        "got it",
        "sounds good",
        "no problem",
        "thanks",
        "thank you",
    }


def _normalize_line(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", " ", str(text or "").lower()).strip()
