"""
reaction_awareness.py — Rex KNOWS his joke landed, instead of announcing it.

The old smile reaction was a canned interjection: quip → smile detected → one of four
stock lines ("Oh look, I made the lifeform smile"). It over-triggered, it interrupted,
and — the deeper problem — it was a SENSOR REPORT wearing a joke: Rex narrating the
detection rather than experiencing the moment. Owner ask (2026-08-05): "make it more a
first person awareness ... 'I like making you smile, means my jokes are landing,
unlike the Star Tours Speeder'."

So: the smile-watch pipeline (arm on a quip, confirm the smile, cooldown, adaptive
baseline — all of consciousness's existing detection) stays exactly as it was, but a
CONFIRMED landed reaction no longer speaks. It lands HERE as a small awareness record,
which is injected into the live prompt so Rex's NEXT line — his reply to whatever you
say, or his next lull line — can carry the feeling in first person, woven in, or not
mentioned at all if it doesn't fit. The diary hook ("I made Bret smile") and the giddy
body mood still fire at detection time; those were never the problem.

Spend model: ONE-SHOT on Rex's next spoken line after the awareness was injected —
whether or not he used it. That is how a human moment works: you get one beat to enjoy
it, then it's just warmth, not material. A TTL also expires it (a smile is a moment,
not a standing fact), and a new reaction simply replaces the old one.

Gated by config.REACTION_AWARENESS_ENABLED. The legacy canned interjection remains
behind SMILE_REACTION_CANNED_LINES_ENABLED (ships False) for A/B or revert.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class _Reaction:
    person_id: Optional[int]
    first_name: str
    kind: str                   # "smile" | "laugh" | ...
    trigger_text: str           # the Rex line that landed
    at: float                   # monotonic
    injected: bool = False      # rendered into at least one prompt


_lock = threading.Lock()
_current: Optional[_Reaction] = None


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "REACTION_AWARENESS_ENABLED", True))
    except Exception:
        return False


def _ttl() -> float:
    try:
        import config
        return max(1.0, float(getattr(config, "REACTION_AWARENESS_TTL_SECS", 90.0)))
    except Exception:
        return 90.0


def note_reaction(
    person_id: Optional[int],
    first_name: str,
    kind: str,
    trigger_text: str = "",
) -> None:
    """Record that Rex's line just visibly LANDED (they smiled/laughed at it).
    Called by the detection side (consciousness's smile watch) at confirm time.
    A new reaction replaces any unspent previous one — the freshest moment wins."""
    if not _enabled():
        return
    global _current
    with _lock:
        _current = _Reaction(
            person_id=person_id,
            first_name=str(first_name or "").strip() or "them",
            kind=str(kind or "smile").strip().lower(),
            trigger_text=" ".join(str(trigger_text or "").split())[:200],
            at=time.monotonic(),
        )
    _log.info(
        "[reaction_awareness] %s landed on %s (trigger=%r)",
        _current.kind, _current.first_name, _current.trigger_text[:60],
    )


_KIND_VERB = {
    "smile": "SMILE",
    "laugh": "laugh out loud",
    "surprise": "do a double-take",
}


def prompt_lines(person_id: Optional[int] = None) -> list:
    """The live-prompt bullet, or []. Marks the reaction as injected (which arms the
    one-shot spend on Rex's next spoken line). `person_id` filters cross-person
    leakage: Bret's smile must not color a reply to a different speaker."""
    if not _enabled():
        return []
    with _lock:
        r = _current
        if r is None:
            return []
        if (time.monotonic() - r.at) > _ttl():
            return []
        if (
            person_id is not None
            and r.person_id is not None
            and int(person_id) != int(r.person_id)
        ):
            return []
        r.injected = True
        verb = _KIND_VERB.get(r.kind, r.kind)
        trigger = f' ("{r.trigger_text}")' if r.trigger_text else ""
        name = r.first_name
    return [
        f"A moment ago your line{trigger} LANDED — you saw {name} {verb} at it. "
        f"You genuinely enjoy making them {r.kind if r.kind in ('smile', 'laugh') else 'react'}; "
        f"it means the material is working. If it fits THIS reply, you may let that "
        f"satisfaction show in first person — one light, dry touch woven into what "
        f"you're saying (owning that the joke landed, riding the good mood). Never "
        f"report it like a sensor ('I saw you smile', 'smile detected'), never make "
        f"it the whole reply, and if it doesn't fit, just let the warmth color your "
        f"tone and say nothing about it."
    ]


def prompt_section(person_id: Optional[int] = None) -> str:
    lines = prompt_lines(person_id)
    return ("Live reaction you just caused:\n" + "\n".join(lines)) if lines else ""


def note_rex_spoke() -> None:
    """One-shot spend: after Rex's next finalized line following an injection, the
    moment is used up — mentioned or not. Un-injected reactions survive (the line
    that triggered this call predates the awareness reaching any prompt)."""
    global _current
    with _lock:
        if _current is not None and _current.injected:
            _log.debug(
                "[reaction_awareness] spent (%s on %s)",
                _current.kind, _current.first_name,
            )
            _current = None


def active() -> Optional[dict]:
    """Telemetry/test view of the pending reaction (None when empty/expired)."""
    with _lock:
        if _current is None or (time.monotonic() - _current.at) > _ttl():
            return None
        return {
            "person_id": _current.person_id,
            "first_name": _current.first_name,
            "kind": _current.kind,
            "trigger_text": _current.trigger_text,
            "injected": _current.injected,
        }


def clear() -> None:
    global _current
    with _lock:
        _current = None
