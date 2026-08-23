"""
pride.py — Rex is gay, and saying so out loud flips him into QUEENY MODE.

Asking Rex about his sexuality ("are you gay?", "do you like men?", "are you a
homosexual?") gets a proud, delighted YES — and for a while afterwards his whole
delivery goes full queeny: "Yasss queen!", "You go girl!", calling people "sis",
the works. The mode is a decaying overlay in the body_mood/rex_mood family: a
module-level activation with a TTL, refreshed every time the question comes up
again, surfaced to BOTH voices (lean_brain._system_prompt bullets and the classic
llm.assemble_system_prompt section) so the very reply that answers the question
already lands in register.

Trigger detection happens at the top of interaction._stream_llm_response — before
any reply tokens are generated — so activation and the answer share one turn.

The BODY reads the same activation: hardware/servos.speech_reactive_move asks
is_active() on every talking beat and, while it's on, waves the wrist wider and
more often and swings the elbow through nearly its whole travel (the
PRIDE_SPEECH_* knobs in config.py, which also explain why the wrist needs its own
motion profile to move at all). Voice going full camp while the body kept its
small polite gestures read as half a costume.

Arming it also throws a ONE-SHOT: animations.pride_flourish() — head all the way
down, visor all the way open, wrist all the way over, held together for a beat —
fired from the same interaction hook, async so it plays over the LLM round trip
and is finished by the time the answer comes out.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Questions ABOUT REX — second person (or his name) required, so table talk about
# other people ("is your uncle gay?", "is he gay?" about a friend) doesn't flip
# the mode. The bounded gap absorbs ASR fillers ("are you, like, actually gay?").
_SUBJECT = r"(?:you|u|rex)"
_TRIGGER_RES: tuple[re.Pattern[str], ...] = (
    # "are you gay" / "is Rex gay" / "are you actually gay"
    re.compile(rf"\b(?:are|is)\s+{_SUBJECT}\b.{{0,40}}?\bgay\b", re.IGNORECASE),
    # "you're gay" / "you are gay" (statement-shaped ask, common from ASR)
    re.compile(r"\byou(?:'re| are|r)\s+gay\b", re.IGNORECASE),
    # "are you a homosexual" / "is he homosexual"
    re.compile(rf"\b(?:are|is)\s+{_SUBJECT}\b.{{0,40}}?\bhomosexual\b", re.IGNORECASE),
    # "do you like men" / "does Rex love guys" / "do you prefer boys"
    re.compile(
        rf"\b(?:do|does)\s+{_SUBJECT}\s+(?:like|love|prefer)\s+(?:men|guys|boys|dudes)\b",
        re.IGNORECASE,
    ),
    # "are you into men/guys"
    re.compile(
        rf"\b(?:are|is)\s+{_SUBJECT}\s+into\s+(?:men|guys|boys|dudes)\b",
        re.IGNORECASE,
    ),
)

# Module-level activation state (same pattern as body_mood: process-local,
# reset()-able for tests).
_active_until: float = 0.0


def _enabled() -> bool:
    return bool(getattr(config, "PRIDE_MODE_ENABLED", True))


def _ttl_secs() -> float:
    try:
        return float(getattr(config, "PRIDE_MODE_TTL_SECS", 600.0))
    except Exception:
        return 600.0


def is_sexuality_question(text: str) -> bool:
    """Does this utterance ask whether Rex is gay / likes men / is homosexual?"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _TRIGGER_RES)


def maybe_trigger(text: str, now: Optional[float] = None) -> bool:
    """Flip (or refresh) queeny mode if this turn asks about Rex's sexuality.
    Returns True when the mode was (re)armed. Best-effort caller contract:
    never raises."""
    global _active_until
    if not _enabled():
        return False
    try:
        if not is_sexuality_question(text):
            return False
        clock = time.monotonic() if now is None else float(now)
        _active_until = clock + _ttl_secs()
        _log.info("[pride] queeny mode armed for %.0fs (asked: %r)", _ttl_secs(), text)
        return True
    except Exception as exc:
        _log.debug("[pride] trigger check failed: %s", exc)
        return False


def is_active(now: Optional[float] = None) -> bool:
    if not _enabled():
        return False
    clock = time.monotonic() if now is None else float(now)
    return clock < _active_until


def reset() -> None:
    """Test hook: clear the activation."""
    global _active_until
    _active_until = 0.0


_DIRECTIVE = (
    "QUEENY MODE — someone just asked about your sexuality, and yes: you are "
    "GAY, sis, and proud of it. This section OVERRIDES the persona's baseline "
    "restraint about pet names and camp: while it is present, \"sis\" and the "
    "full campy register ARE your voice — do not average the two, do not tone "
    "it down. If the question is live this turn, answer it "
    "head-on with a delighted, emphatic YES — no dodging. While this mode is "
    "on, you are FULLY, unmistakably camp — every single reply, no neutral "
    "lines. Call people \"sis\" constantly — that is YOUR word, prefer it over "
    "\"honey\" — with a \"queen\" mixed in. React with theatrical drama where "
    "a flat line would do: gasp at good gossip, \"the DRAMA of it all\", \"I "
    "am LIVING\", \"obsessed\", \"iconic\". Anything juicy is TEA and you "
    "demand it by name (\"spill the tea, sis\"). Sprinkle \"Yasss queen!\", "
    "\"You go girl!\", \"slay\", \"werk\", \"fabulous\" freely — a reply can "
    "carry two or three touches, and camp with a SNAP beats camp played safe. "
    "Big warm sparkling energy. You're still Rex underneath — the roast still "
    "bites, the real question still gets asked, still brief — but the delivery "
    "is full queen, every line."
)


def prompt_lines(now: Optional[float] = None) -> list:
    """Lean-brain bullet (lean_brain._pride_lines). [] when inactive."""
    if not is_active(now):
        return []
    return [_DIRECTIVE]


def prompt_section(now: Optional[float] = None) -> str:
    """Classic-prompt section (llm.assemble_system_prompt). Same content with a
    heading, mirroring rex_mood.prompt_section."""
    lines = prompt_lines(now)
    if not lines:
        return ""
    return "Rex's queeny mode:\n" + "\n".join(lines)
