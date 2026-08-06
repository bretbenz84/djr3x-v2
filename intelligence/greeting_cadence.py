"""
greeting_cadence.py — how recently Rex greeted you, and whether he's already asked.

Two owner gripes, one instinct (2026-08-05):

  1. "Repeat visits in the same day are somewhat boring" — Rex re-ran the full hello
     on every boot, because every gate that could have stopped him was IN-MEMORY.
     `_greeted_this_session` is a set wiped at process start; `_should_fire_presence`
     uses `time.monotonic()` cooldowns that reset with the process. So the one event
     most likely to happen twice in an hour — a restart — was also the one thing
     guaranteed to defeat every "don't repeat yourself" guard in the system.
  2. "He shouldn't ask how I'm doing again" — nothing anywhere recorded that he HAD
     asked, so the reciprocal loop ran fresh every time: he asks, you answer, you
     bounce it back, he says he's operating within normal parameters.

Both are fixed by reading PERSISTED per-person timestamps instead of process state:
`people.last_greeted_at` (already written by every greeting) and the new
`people.last_wellbeing_ask_at`. A reboot changes nothing about what the database
knows, which is precisely the property that was missing.

The wellbeing ask is detected from Rex's ACTUAL spoken line rather than from which
prompt-builder ran, so an LLM that improvises "how've you been?" into a greeting that
never asked for one still gets recorded. Detecting from the output is the only way to
be right about what he actually said.

Consumers:
  * consciousness._step_presence_tracking — the greeting ladder consults recency()
    before building a prompt (P3.4), and passes the no-ask constraint downward.
  * speech_engine.generate_and_speak_presence — records the ask next to the existing
    record_greeting() call, once the line is final.
  * interaction._register_rex_utterance — records it for mid-conversation asks too.
  * lean_brain._system_prompt — injects suppression_line() so the live voice knows.

Gated by config.GREETING_CADENCE_ENABLED. No LLM call, no network, no module state:
every answer is derived from the DB, so it is correct across processes by construction.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config accessors (lazy, so tests can monkeypatch)
# ─────────────────────────────────────────────────────────────────────────────

def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "GREETING_CADENCE_ENABLED", True))
    except Exception:
        return False


def _snap_secs() -> float:
    try:
        import config
        return float(getattr(config, "GREETING_CADENCE_SNAP_SECS", 1200) or 0)
    except Exception:
        return 1200.0


def _recent_secs() -> float:
    try:
        import config
        return float(getattr(config, "GREETING_CADENCE_RECENT_SECS", 10800) or 0)
    except Exception:
        return 10800.0


def _wellbeing_cooldown() -> float:
    try:
        import config
        return float(getattr(config, "WELLBEING_ASK_COOLDOWN_SECS", 14400) or 0)
    except Exception:
        return 14400.0


# ─────────────────────────────────────────────────────────────────────────────
# Recency
# ─────────────────────────────────────────────────────────────────────────────

SNAP = "snap"        # minutes ago — barely acknowledge, no question at all
RECENT = "recent"    # a few hours ago — warm nod, but no "how are you"


def greeted_age_secs(person_db_id: Optional[int]) -> Optional[float]:
    """Seconds since Rex last greeted this person, across reboots. None if never."""
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import people as people_mod
        return people_mod.last_greeted_age_secs(person_db_id)
    except Exception as exc:
        _log.debug("[greeting_cadence] greeted-age lookup failed: %s", exc)
        return None


def recency(person_db_id: Optional[int]) -> tuple:
    """(bucket, age_secs) — bucket is SNAP, RECENT, or None.

    None means "long enough ago that the normal greeting ladder should run" and is
    also what a disabled feature, an unknown person, or a never-greeted person get:
    every unknown falls back to the pre-existing behavior, never to silence.
    """
    if not _enabled():
        return (None, None)
    age = greeted_age_secs(person_db_id)
    if age is None:
        return (None, None)
    if age < _snap_secs():
        return (SNAP, age)
    if age < _recent_secs():
        return (RECENT, age)
    return (None, age)


def describe_gap(age_secs: Optional[float]) -> str:
    """A short human phrase for how long it's been — for prompt text, not display."""
    if age_secs is None:
        return ""
    minutes = age_secs / 60.0
    if minutes < 2:
        return "barely a minute ago"
    if minutes < 45:
        return f"about {int(round(minutes))} minutes ago"
    hours = minutes / 60.0
    if hours < 1.6:
        return "about an hour ago"
    return f"about {int(round(hours))} hours ago"


# ─────────────────────────────────────────────────────────────────────────────
# The wellbeing ask
# ─────────────────────────────────────────────────────────────────────────────
#
# Matches Rex asking after THEM. Deliberately does not match him being asked, or him
# answering: "how are you?" said BY Rex is the ask; the same words quoted back are
# not, but Rex's own lines are the only text this ever sees, so that ambiguity can't
# arise. Requires a question mark somewhere in the line — a bare "how are you" inside
# a statement ("you never ask how are you") isn't an ask.
_WELLBEING_ASK_PAT = re.compile(
    r"\b(?:"
    r"how(?:'s|\s+is|\s+are|\s+have|'ve|\s+has|\s+was|\s+were)?\s+"
    r"(?:you|ya|things|it|everything|life|your\s+day|your\s+week|your\s+weekend|"
    r"your\s+evening|your\s+morning|your\s+night)\b"
    r"|how\s+(?:you\s+)?(?:doing|holding\s+up|feeling|been|goes\s+it|'?re\s+you)\b"
    r"|how'?d\s+your\s+(?:day|week|weekend|night|evening)\b"
    r"|what'?s\s+(?:up|new|good|going\s+on)\b"
    r"|you\s+(?:doing\s+)?(?:ok|okay|alright|all\s+right)\b"
    r"|everything\s+(?:ok|okay|alright|all\s+right)\b"
    r")",
    re.IGNORECASE,
)


def looks_like_wellbeing_ask(text: str) -> bool:
    """True when one of Rex's own finished lines asked the human how THEY are."""
    line = " ".join((text or "").strip().split())
    if not line or "?" not in line:
        return False
    return bool(_WELLBEING_ASK_PAT.search(line))


def note_wellbeing_ask(person_db_id: Optional[int], text: str = "") -> bool:
    """Record the ask if `text` is one (or unconditionally when `text` is empty).

    Returns True when something was recorded. Never raises — a bookkeeping failure
    must not take down a speech path.
    """
    if not _enabled() or not isinstance(person_db_id, int):
        return False
    if text and not looks_like_wellbeing_ask(text):
        return False
    try:
        from memory import people as people_mod
        people_mod.record_wellbeing_ask(person_db_id)
        _log.debug("[greeting_cadence] wellbeing ask recorded for person %s", person_db_id)
        return True
    except Exception as exc:
        _log.debug("[greeting_cadence] wellbeing ask record failed: %s", exc)
        return False


def wellbeing_ask_age_secs(person_db_id: Optional[int]) -> Optional[float]:
    if not isinstance(person_db_id, int):
        return None
    try:
        from memory import people as people_mod
        return people_mod.last_wellbeing_ask_age_secs(person_db_id)
    except Exception as exc:
        _log.debug("[greeting_cadence] wellbeing-age lookup failed: %s", exc)
        return None


def wellbeing_ask_spent(person_db_id: Optional[int]) -> tuple:
    """(spent, age_secs) — True when Rex asked recently enough that asking again would
    read as forgetting he already did."""
    if not _enabled():
        return (False, None)
    age = wellbeing_ask_age_secs(person_db_id)
    if age is None:
        return (False, None)
    return (age < _wellbeing_cooldown(), age)


def suppression_line(person_db_id: Optional[int]) -> str:
    """The lean-brain bullet telling Rex he already asked. "" when he hasn't.

    Bans only the RITUAL re-ask. Following up on something they actually told him
    ("you said the install was going badly — did it land?") is the opposite of
    redundant, so it stays explicitly allowed.
    """
    spent, age = wellbeing_ask_spent(person_db_id)
    if not spent:
        return ""
    when = describe_gap(age)
    return (
        f"You ALREADY asked them how they're doing {when} and they answered. Do NOT "
        f"ask again — no \"how are you\", \"how's it going\", \"what's up\", \"how's "
        f"your day\", or any reworded version. Picking up something specific they "
        f"actually told you is still welcome; the generic check-in is spent."
    )


def greeting_constraint(bucket: Optional[str], age_secs: Optional[float]) -> str:
    """The prompt clause for a quick return, appended by the greeting ladder.

    SNAP is the reboot case the owner hit: he was here minutes ago, so the greeting
    should be the two-word acknowledgment a human gives someone walking back in for
    their keys — not a fresh hello, and definitely not a fresh question.
    """
    when = describe_gap(age_secs)
    if bucket == SNAP:
        return (
            f"You saw them {when} — they have barely been gone. Say the SHORTEST "
            f"possible thing: a half-sentence acknowledgment, the way you'd greet "
            f"someone walking back in for their keys. Do NOT run a hello, do NOT use "
            f"their name as a full greeting, and ask NOTHING — no \"how are you\", "
            f"\"what's up\", or \"what's new\". Amused or deadpan, never annoyed. "
            f"Under eight words. Examples of the register: \"That was quick.\", "
            f"\"Back so soon.\", \"Miss me?\", \"Oh good, you again.\""
        )
    if bucket == RECENT:
        return (
            f"You already saw and greeted them {when}, so this is a RETURN, not a "
            f"first hello. Acknowledge it warmly in ONE short line. You already know "
            f"how their day is going — do NOT ask \"how are you\", \"what's up\", "
            f"\"what's new\", or \"how's your day\" again. If you say anything beyond "
            f"the acknowledgment, make it about something specific from earlier, not "
            f"a fresh check-in."
        )
    return ""
