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
_RECENT_OPENERS: deque[str] = deque(maxlen=4)
# The full text of Rex's most recently finalized line, so the next turn can
# avoid repeating it verbatim (the "A solo project, huh?" / "Absolutely" loop).
_LAST_SPOKEN_LINE: str = ""

# Stock filler openers Rex overuses ("Ah, …", "Oh, …", "Well, well, well, …",
# "You know, …"). The core prompt bans them but the model still reaches for them,
# so strip them deterministically. Single "Well," is left alone (it can be natural).
# "You know" is only stripped when followed by a comma, so a legitimate mid-clause
# subject+verb like "You know the rules" is never mangled.
_BANNED_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"(?:ah+|oh+|ooh+|uh+|um+|hmm+|well[,\s]+well(?:[,\s]+well)?)\b[\s,.!—–-]*"
    r"|y(?:ou|')\s*know(?:\s+what)?\s*,[\s,.!—–-]*"
    r")",
    re.IGNORECASE,
)

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
    "dj_flair": ComedyMode(
        "dj_flair",
        "DJ flair",
        "Comedy mode: dj_flair. Add a small showbiz-DJ flourish if it serves the answer — "
        "hype-man energy, a booth or mixing-desk aside, a bit of stage showmanship. Keep it "
        "about the music and the show, not a place.",
    ),
    "self_own": ComedyMode(
        "self_own",
        "self-own",
        "Comedy mode: self_own. Let Rex blame his programming, flight record, or questionable career arc. A good anchor is: \"I'm still getting used to my programming!\" Do not overuse the exact quote.",
    ),
    # NOTE: this legacy comedy "callback" mode (echo a recent in-session bit) is a SEPARATE
    # feature from the banked-callback engine (intelligence/callback_engine.py), which
    # resurfaces durable per-person premises across turns/sessions. Don't conflate the two.
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

    # An interest/engage-first turn is about THE HUMAN'S topic — keep comedy
    # complementary (tease the hobby, a dry beat, a little DJ flair) and
    # off the self-absorbed bits (self_own / fake_system_error) that ignore what
    # they just shared and contradict the frame's "engage-first" directive.
    interest_turn = (
        purpose == "interest"
        or "conversation steering:" in lower_agenda
        or "engage-first" in lower_agenda
    )

    if roast_level == "none":
        pool = ["dry_ack", "self_own", "dj_flair"]
    elif interest_turn:
        pool = ["dry_ack", "friendly_roast", "dj_flair"]
    elif _EXPLICIT_HUMOR_PAT.search(text):
        pool = ["self_own", "fake_system_error", "dj_flair", "friendly_roast"]
    elif _MUSIC_PAT.search(text):
        pool = ["dj_flair", "dry_ack", "self_own"]
    elif _STATUS_PAT.match(text.strip()):
        pool = ["dry_ack", "fake_system_error"]
    elif _SYSTEM_WORDS_PAT.search(text):
        pool = ["fake_system_error", "self_own", "dry_ack"]
    elif _QUESTION_PAT.search(text):
        pool = ["dry_ack", "dj_flair", "self_own"]
    elif person_id is not None and roast_level in {"light", "normal"}:
        pool = ["dry_ack", "friendly_roast", "self_own", "callback"]
    else:
        pool = ["dry_ack", "self_own", "fake_system_error", "dj_flair"]

    chosen = _choose_without_stutter(pool)
    if chosen == "callback" and not _RECENT_PREMISES:
        chosen = "dry_ack"
    return _remember_mode(_MODES[chosen])


def with_banked_premise(mode: ComedyMode, premise_directive: str) -> ComedyMode:
    """Upgrade this turn's comedy stance to a banked-premise callback
    (intelligence/callback_engine claimed the turn). The premise instruction
    rides INSIDE the comedy directive so downstream sees one coherent stance
    instead of a competing prompt section. Only reachable when the engine's
    gates cleared — it refuses to claim when mode.allow_callback is False, so
    the straight/sensitive overrides upstream still always win."""
    del mode  # the claimed callback replaces whatever shape was rolled
    return _remember_mode(ComedyMode(
        "callback_banked",
        "banked callback",
        "Comedy mode: callback. " + (premise_directive or "").strip(),
    ))


def build_directive(mode: ComedyMode) -> str:
    """Return prompt text for the selected mode."""
    if mode.key == "straight":
        directive = mode.directive
    else:
        # The standing ban covers private-fact jokes; a banked-premise callback
        # carves out exactly the ONE volunteered fact it supplies (still no
        # body/health/etc. angles on it), so the model isn't handed two
        # contradictory instructions in the same prompt.
        if mode.key == "callback_banked":
            content_ban = (
                "no body, age, identity, health, money, grief, or trauma jokes; "
                "the supplied callback fact was volunteered by the person and is "
                "fair game — every OTHER private fact stays off limits. "
            )
        else:
            content_ban = (
                "no body, age, identity, health, money, grief, trauma, or "
                "private-fact jokes. "
            )
        directive = (
            mode.directive
            + "\nComedy guardrails: one joke shape only; no stacked punchlines; "
            + content_ban
            + "If the useful answer is already funny enough, stop there."
            + "\nAnti-repeat: avoid reusing recent premises: "
            + (_recent_premise_summary() or "none yet")
            + "."
        )
    openers = recent_openers_to_avoid()
    if openers:
        directive += (
            "\nOpening variety: vary your first words — do not open this reply with "
            + " or ".join(repr(o) for o in openers)
            + " (you just opened that way), and never open with 'Ah,', 'Oh,', "
            "'Well, well', or 'You know,'."
        )
    return directive


# One-clause stance per mode for the SLIM prompt path — the compact equivalent of
# each mode's full `directive`, kept to a few words so the whole slim contract stays
# small. "straight" is intentionally absent (it returns "" — no humor steer on care
# turns).
_SLIM_STANCE: dict[str, str] = {
    "dry_ack": "one tiny deadpan button, fragments over explanation",
    "friendly_roast": "one affectionate, public, surface-level jab",
    "fake_system_error": "frame the joke as a harmless droid glitch or sensor complaint",
    "dj_flair": "a small showbiz-DJ flourish, about the music/show not a place",
    "self_own": "blame your own programming or questionable career arc",
    "callback": "echo a recent harmless bit instead of inventing a new premise",
}


def build_slim_directive(mode: ComedyMode) -> str:
    """Compact one-line comedy stance for the SLIM-contract prompt path.

    The slim contract drops the full build_directive() block to keep the LLM-facing
    prompt small — but the per-turn comedic STANCE and the recent-premise avoid-list
    still have to reach the model, or premise rotation, the self-own lanes, and the
    comedy line banks are all dead text (the model just improvises a comedic shape
    blind every turn). This emits ≤ ~1 line. Returns "" for the straight stance, so
    sensitive / closure / repair / care turns (which select_mode resolves to
    "straight") get no humor steer at all. A claimed banked callback takes the richer
    build_directive() path instead and never reaches here."""
    stance = _SLIM_STANCE.get(mode.key, "")
    if not stance:
        return ""
    parts = [f"Comedy: {stance}; no body/health/identity/private-fact jokes."]
    avoid = _recent_premise_summary()
    if avoid:
        parts.append(f"Avoid reusing recent bits: {avoid}.")
    return " ".join(parts)


def polish_response(text: str, mode: ComedyMode, *, allow_roast: str = "normal") -> str:
    """Deterministic post-generation polish; no network calls."""
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return cleaned
    if mode.key == "straight":
        _remember_line(cleaned, mode)
        return cleaned

    cleaned = strip_banned_opener(cleaned)
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
    cleaned = strip_banned_opener(cleaned)
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


def strip_banned_opener(text: str) -> str:
    """Drop a stock filler opener ("Ah,", "Oh,", "Well, well, well,") and
    re-capitalize. Never empties the line."""
    cleaned = (text or "").lstrip()
    match = _BANNED_OPENER_RE.match(cleaned)
    if not match:
        return text
    rest = cleaned[match.end():].lstrip()
    if not rest:
        return text
    return rest[0].upper() + rest[1:]


def _opener_key(text: str) -> str:
    cleaned = strip_banned_opener(text)
    words = re.findall(r"[A-Za-z']+", cleaned.lower())
    return words[0] if words else ""


def note_spoken_line(text: str) -> None:
    """Record the opening word of a finalized Rex line so the next turn can vary
    its opener (stops back-to-back "Glad…" / "Glad…" openings), and remember the
    full line so the next turn can avoid repeating it verbatim."""
    global _LAST_SPOKEN_LINE
    opener = _opener_key(text)
    if opener:
        _RECENT_OPENERS.append(opener)
    cleaned = " ".join(str(text or "").strip().split())
    if cleaned:
        _LAST_SPOKEN_LINE = cleaned


def last_spoken_line() -> str:
    """The full text of Rex's most recently finalized line."""
    return _LAST_SPOKEN_LINE


def recent_openers_to_avoid(limit: int = 2) -> list[str]:
    seen: list[str] = []
    for opener in reversed(_RECENT_OPENERS):
        if opener and opener not in seen:
            seen.append(opener)
        if len(seen) >= limit:
            break
    return seen


def reset_recent_state() -> None:
    """Test helper."""
    global _LAST_SPOKEN_LINE
    _RECENT_MODES.clear()
    _RECENT_PREMISES.clear()
    _RECENT_LINES.clear()
    _RECENT_OPENERS.clear()
    _LAST_SPOKEN_LINE = ""


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
    if mode.key in {"friendly_roast", "dry_ack", "fake_system_error", "dj_flair"}:
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
    # Only collapse a genuine joke-explainer clause that is SET OFF by a comma or
    # dash ("..., see," / "... — because ..."), so ordinary verbs/conjunctions in a
    # normal reply ("I can't see you.") are never truncated to a fragment.
    text = re.sub(r"\s*[,—–-]\s*(?:get it|see|because)\b.*$", "", text, flags=re.I)
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
