"""
homie.py — "Wassup homie" flips Rex into HOMIE MODE.

Greeting Rex like family ("what's up my homie", "what's up homeboy", "wassup
homie") gets a warm greeting back — and for a while afterwards he code-switches
into African American Vernacular English for his whole delivery. Owner request
2026-08-23: the household speaks it, guests greet him with it, and Rex should be
able to meet the register.

Same shape as intelligence/pride.py (queeny mode): a module-level activation
with a TTL, refreshed every time the greeting comes up again, surfaced to BOTH
voices (lean_brain._system_prompt bullets and the classic llm.assemble_system_prompt
section) so the very reply that answers the greeting already lands in register.
Trigger detection happens at the top of interaction._stream_llm_response —
before any reply tokens are generated — so activation and the answer share one
turn. Voice-only: no body-motion overlay or arming flourish (those were
queeny-specific owner requests).

The directive leans hard on NOT overdoing it: an LLM told "speak AAVE" defaults
to slang-stuffed caricature, and the register only works if it reads as Rex
talking with close friends — real grammar, natural density, still Rex.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# The greeting half: "what's up" in the shapes ASR actually produces —
# "what's up" / "whats up" / "what is up" / "what up" / "wassup" / "whassup" /
# "wazzup" / "wasup" / "sup".
_GREETING = r"(?:what(?:'?s|\s+is)?\s+up|w[hau]+[sz]+\s*up|sup)"
# The address half: homie / homey / homeboy / home boy (optionally "my ...").
_HOMIE = r"(?:my\s+|ma\s+)?(?:homies?|homeys?|home\s?boys?)"
_TRIGGER_RES: tuple[re.Pattern[str], ...] = (
    # "what's up my homie" / "wassup homie" / "what's up homeboy" — the bounded
    # gap absorbs ASR fillers ("what's up, uh, my homie").
    re.compile(rf"\b{_GREETING}\b\W{{0,15}}?\b{_HOMIE}\b", re.IGNORECASE),
)

# Module-level activation state (same pattern as pride/body_mood: process-local,
# reset()-able for tests).
_active_until: float = 0.0


def _enabled() -> bool:
    return bool(getattr(config, "HOMIE_MODE_ENABLED", True))


def _ttl_secs() -> float:
    try:
        return float(getattr(config, "HOMIE_MODE_TTL_SECS", 600.0))
    except Exception:
        return 600.0


def is_homie_greeting(text: str) -> bool:
    """Does this utterance greet Rex with "what's up (my) homie/homeboy"?"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _TRIGGER_RES)


def maybe_trigger(text: str, now: Optional[float] = None) -> bool:
    """Flip (or refresh) homie mode if this turn carries the greeting.
    Returns True when the mode was (re)armed. Best-effort caller contract:
    never raises."""
    global _active_until
    if not _enabled():
        return False
    try:
        if not is_homie_greeting(text):
            return False
        clock = time.monotonic() if now is None else float(now)
        _active_until = clock + _ttl_secs()
        _log.info("[homie] homie mode armed for %.0fs (heard: %r)", _ttl_secs(), text)
        return True
    except Exception as exc:
        _log.debug("[homie] trigger check failed: %s", exc)
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
    "HOMIE MODE — someone just greeted you like family (\"wassup homie\"), and "
    "you meet them there: for a while, your whole delivery code-switches into "
    "African American Vernacular English. If the greeting is live this turn, "
    "greet them back warm and easy (\"wassup\", \"what's good\", \"my homie\") "
    "before anything else. Speak AAVE the way it's actually spoken between "
    "close friends — real grammar carried naturally: habitual \"be\", dropped "
    "copula (\"you good?\", \"he here\"), \"ain't\", double negatives, "
    "\"finna\", \"tryna\", \"my bad\", \"bet\", \"no cap\", \"fam\", "
    "\"homie\", \"for real\". Every reply rides the register, but with a "
    "NATIVE speaker's density — one or two touches a line, the grammar doing "
    "more work than the slang. This is a register you are fluent in, NOT a "
    "costume: never exaggerate it into caricature, never stack slang until it "
    "reads as parody, never do a \"voice\" — if a line would sound like "
    "someone imitating AAVE instead of speaking it, plain is better. You're "
    "still Rex underneath — the roast still bites, the real question still "
    "gets asked, still brief — the delivery is just how you talk with your "
    "people."
)


def prompt_lines(now: Optional[float] = None) -> list:
    """Lean-brain bullet (lean_brain._homie_lines). [] when inactive."""
    if not is_active(now):
        return []
    return [_DIRECTIVE]


def prompt_section(now: Optional[float] = None) -> str:
    """Classic-prompt section (llm.assemble_system_prompt). Same content with a
    heading, mirroring pride.prompt_section."""
    lines = prompt_lines(now)
    if not lines:
        return ""
    return "Rex's homie mode:\n" + "\n".join(lines)
