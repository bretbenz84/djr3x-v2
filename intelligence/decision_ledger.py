"""A programmatically-true record of what Rex just DID and why.

"Why did you do that?" used to be answered by the reply model inventing a
mechanism — plausible, in character, and false. This module is a small
in-memory ring of decisions the code actually made, each with a Rex-legible
`why` written by the site that made it, plus a directive builder that hands the
last few to the reply model when the person asks.

Design limits, on purpose (owner 2026-08-18: only what is feasible and easy):

- Only sites that already know their reason record here — proactive lines (the
  governor purpose + label), the lean impulse (which cue won, how long the room
  had been quiet), the reply frame (purpose / comedy stance / roast level), the
  pet-name guess, an unprompted impression, an idle head wander, a speaker gaze
  search, a flinch retreat. Nothing is inferred after the fact.
- Anything not on record is answered with "honestly, I'm not sure" — the
  directive says so explicitly. An honest blank beats a confident story.
- Session memory only. It is not persisted; the question is always about the
  last minute or two.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Optional

import config

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_ring: "deque[dict]" = deque(maxlen=int(getattr(config, "DECISION_LEDGER_SIZE", 80)))


def record(kind: str, why: str, *, said: str = "", detail: Optional[dict] = None) -> None:
    """Add one decision. `why` is written FOR Rex, first person, plain words —
    it is pasted into his prompt verbatim ("I spotted a dog and Bret was here a
    minute ago, so I asked if it was Max"). `said` is the line, if it spoke one."""
    why = " ".join(str(why or "").split())
    if not why:
        return
    entry = {
        "t": time.time(),
        "mono": time.monotonic(),
        "kind": str(kind or "decision"),
        "why": why,
        "said": " ".join(str(said or "").split()),
        "detail": dict(detail or {}),
    }
    with _lock:
        _ring.append(entry)
    _log.info("[decision_ledger] %s: %s%s", entry["kind"], why,
              (f" said={entry['said']!r}" if entry["said"] else ""))


def recent(*, max_age_secs: float = 240.0, limit: int = 8, kinds=None) -> list[dict]:
    now = time.monotonic()
    with _lock:
        items = list(_ring)
    out = [
        e for e in reversed(items)
        if now - e["mono"] <= max_age_secs and (not kinds or e["kind"] in kinds)
    ]
    return out[:limit]


def clear() -> None:
    with _lock:
        _ring.clear()


# ── The question ─────────────────────────────────────────────────────────────

_WHY_RE = re.compile(
    r"\b(?:"
    r"why(?:'d| did| do| are| were| would| was|'re)? (?:you|u|ya)\b"
    r"|why (?:that|the (?:turn|look|voice|joke|impression|roast|bit|silence))\b"
    r"|what (?:made|makes|possessed) you\b"
    r"|how come (?:you|u)\b"
    r"|what was that (?:about|for|all about)\b"
    r"|what (?:are|were) you (?:doing|looking at|looking for|staring at|turning (?:for|around for))\b"
    r"|where (?:are|were) you (?:looking|going)\b"
    r"|who (?:are|were) you (?:talking to|looking at|looking for)\b"
    r"|explain yourself\b"
    r")",
    re.IGNORECASE,
)


def looks_like_why_question(text: str) -> bool:
    return bool(_WHY_RE.search(" ".join((text or "").split())))


def _age_phrase(secs: float) -> str:
    secs = max(0.0, secs)
    if secs < 8:
        return "just now"
    if secs < 60:
        return f"about {int(round(secs / 5.0) * 5)} seconds ago"
    mins = int(round(secs / 60.0))
    return f"about {mins} minute{'s' if mins != 1 else ''} ago"


def why_directive(user_text: str) -> Optional[str]:
    """When the person is asking why Rex did something, the reply-model directive:
    the real record, and an order to say 'not sure' past its edge."""
    if not bool(getattr(config, "DECISION_LEDGER_ENABLED", True)):
        return None
    if not looks_like_why_question(user_text):
        return None
    now = time.monotonic()
    window = float(getattr(config, "DECISION_LEDGER_WHY_WINDOW_SECS", 240.0))
    limit = int(getattr(config, "DECISION_LEDGER_WHY_LIMIT", 6))
    # Every reply records its frame, which would crowd out the rarer, more
    # interesting decisions (a turn, a bit, a look). Keep the two newest frames
    # (the reply they are probably asking about, and the one before) and fill the
    # rest with everything else.
    frames = recent(max_age_secs=window, limit=2, kinds={"reply_frame"})
    others = [e for e in recent(max_age_secs=window, limit=limit + 2)
              if e["kind"] != "reply_frame"][:limit]
    items = sorted(frames + others, key=lambda e: -e["mono"])[:limit]
    if not items:
        return (
            "They seem to be asking WHY you did something. You have NO record of "
            "deciding anything on your own in the last few minutes — if they mean "
            "a move, a look, a remark, or a bit you volunteered, say honestly that "
            "you're not sure why (in character, briefly); do NOT invent a reason or "
            "a mechanism. If they're asking about an opinion, a preference, or "
            "something you said in reply to them, just answer normally."
        )
    lines = []
    for e in items:
        piece = f"- {_age_phrase(now - e['mono'])}: {e['why']}"
        if e.get("said"):
            piece += f' (I said: "{e["said"][:140]}")'
        lines.append(piece)
    return (
        "They seem to be asking WHY you did something. This is your ACTUAL record "
        "of what you decided on your own recently and the real reason each time "
        "(newest first):\n" + "\n".join(lines) + "\n"
        "Answer FROM THIS RECORD, in character, briefly — one or two sentences, and "
        "you may own it with attitude. If what they're asking about is NOT in this "
        "record, say honestly that you're not sure why (do NOT invent a reason or a "
        "mechanism). If they're asking about an opinion, a preference, or something "
        "you said in direct reply to them, answer normally."
    )


# ── Purpose → plain words, for the proactive-speech path ────────────────────

_PURPOSE_WHY = {
    "world.animal_arrival": "I spotted an animal",
    "presence_reaction": "I reacted to someone showing up or a face I saw",
    "reengagement": "it had gone quiet and I tried to re-engage",
    "emotional_checkin": "I remembered something emotional they'd told me and checked in",
    "relationship_inquiry": "I wanted to know how two people here know each other",
    "identity_prompt": "I didn't know who someone was and asked",
    "memory_followup": "I remembered something they'd told me and followed up",
    "celebration_checkin": "a remembered event or occasion of theirs came due",
    "startup_empty_room": "I booted up into an empty room",
    "battery_status": "my battery level",
    "held_object_remark": "I saw something in someone's hands",
    "lull_callback": "a lull, so I called back to an earlier joke",
    "visual_curiosity": "something I saw caught my eye",
    "people_roast": "a quiet moment, so I ribbed someone for a laugh",
    "small_talk": "the room was quiet and I made small talk",
    "weather.proactive_comment": "the weather",
    "world_reaction": "something I saw or heard changed",
    "ambient_observation": "something in the room I noticed",
    "appearance_riff": "how someone looked, for a laugh",
    "idle_monologue": "I was bored and talking to myself",
    "boredom": "boredom, plain and simple",
    "memory_musing": "a memory drifted up and I mused on it",
    "direct_speech": "I decided to say it",
    "overheard_chime_in": "I heard my name / people talking about me and chimed in",
    "third_party_awareness": "people were talking about me",
    "group_turn_invite": "someone in the group had gone quiet and I pulled them in",
    "personal_space": "someone was very close to me",
    "exploration": "I was exploring the room",
}


def why_for_purpose(purpose: Optional[str], label: str = "") -> str:
    p = str(purpose or "").strip()
    base = _PURPOSE_WHY.get(p) or (p.replace(".", " ").replace("_", " ") if p else "a proactive impulse")
    lab = " ".join(str(label or "").split())
    if lab and lab.lower() not in base.lower():
        return f"{base} ({lab})"
    return base
