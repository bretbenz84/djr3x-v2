"""
intelligence/room_questions.py — the learn-by-asking loop for the room.

Curiosity Phase 1 (owner direction 2026-07-17: "R3X asks a lot of personal
questions but doesn't really ask about novelties in a room"). The room model
(memory/room_model.py) queues genuinely-new-to-the-room objects as 'pending'
questions; this module owns:

  ASK    next_room_question() — the best pending object phrased as a short
         question, consumed by interaction's idle-question path under the
         STARVATION RULE: a pending room question OUTRANKS the personal
         profile-question pool (when the room is stale, personal curiosity
         resumes). Shares the existing question_budget pacing.
  LATCH  note_asked() — arms a short-lived awaiting-answer window.
  LEARN  maybe_capture_answer() — passively watches the next human turns for
         an identity ("that's my sourdough starter") and writes it back to the
         room model with corroboration counting. Cheap regex extraction — no
         LLM call; a miss just leaves the question answered-by-conversation
         (the reply LLM still responds naturally either way).

Everything is fail-safe and behind config.ROOM_QUESTIONS_ENABLED.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Awaiting-answer latch: {"label", "armed_at", "turns_left"}. Module-local —
# one pending room answer at a time is plenty.
_latch: Optional[dict] = None
_last_asked_at: float = 0.0

_TEMPLATES = (
    "Hey — what's the story with that {thing}{where}?",
    "I keep noticing a {thing}{where}. What's the deal with it?",
    "Okay, I have to ask — what's that {thing}{where} about?",
)

# Identity shapes in a natural answer. Ordered most-specific first.
_ANSWER_PATTERNS = (
    re.compile(r"\b(?:it|that|this)(?:'s| is) (?:called |named )(?P<name>[^.,!?]{2,60})", re.I),
    re.compile(r"\b(?:it|that|this)(?:'s| is) (?:my|our|a|an|the) (?P<name>[^.,!?]{2,60})", re.I),
    re.compile(r"\bthose are (?:my|our|the) (?P<name>[^.,!?]{2,60})", re.I),
    re.compile(r"\bwe call (?:it|that) (?P<name>[^.,!?]{2,60})", re.I),
)

_NON_ANSWERS = (
    "i don't know", "i dont know", "no idea", "not sure", "dunno",
    "nothing", "don't worry", "dont worry", "never mind", "nevermind",
    "none of your", "wouldn't you like to know",
)


def _enabled() -> bool:
    return bool(getattr(config, "ROOM_QUESTIONS_ENABLED", True))


def next_room_question() -> Optional[dict]:
    """The best pending room question, or None. Requires: enabled, global
    cooldown clear, and a confirmed pending object in the room model. Returns
    {"label", "text"} — the caller speaks `text` and calls note_asked(label)."""
    if not _enabled():
        return None
    if (time.monotonic() - _last_asked_at) < float(
        getattr(config, "ROOM_QUESTION_COOLDOWN_SECS", 600.0)
    ):
        return None
    try:
        from memory import room_model
        row = room_model.pending_question(
            min_sightings=int(getattr(config, "ROOM_CHANGE_MIN_SIGHTINGS", 2))
        )
    except Exception as exc:
        _log.debug("[room_questions] pending lookup failed: %s", exc)
        return None
    if not row:
        return None
    label = str(row.get("label") or "").strip()
    if not label:
        return None
    bucket = str(row.get("location_bucket") or "").strip()
    where = f" over on the {bucket}" if bucket and bucket != "unknown" else ""
    text = random.choice(_TEMPLATES).format(thing=label, where=where)
    return {"label": label, "text": text}


def note_asked(label: str) -> None:
    """Mark the question asked (room model) and arm the answer-capture latch."""
    global _latch, _last_asked_at
    _last_asked_at = time.monotonic()
    _latch = {
        "label": str(label).strip().lower(),
        "armed_at": time.monotonic(),
        "turns_left": int(getattr(config, "ROOM_QUESTION_ANSWER_TURNS", 2)),
    }
    try:
        from memory import room_model
        room_model.note_question_asked(label)
    except Exception as exc:
        _log.debug("[room_questions] note_question_asked failed: %s", exc)


def _extract_identity(text: str) -> Optional[str]:
    """Pull the object identity out of a natural-language answer, or None."""
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return None
    low = cleaned.lower()
    if any(p in low for p in _NON_ANSWERS):
        return None
    if cleaned.endswith("?"):
        return None                      # a question back is not an answer
    for pat in _ANSWER_PATTERNS:
        m = pat.search(cleaned)
        if m:
            name = m.group("name").strip(" .")
            if 2 <= len(name) <= 60:
                return name
    # A short direct reply ("sourdough starter", "my telescope case") counts —
    # but not casual chatter: no question-words, no leading filler/deflection.
    words = cleaned.split()
    fillers = ("what", "why", "how", "who", "when", "where", "anyway", "so ",
               "well", "okay", "ok ", "yeah", "yes", "no ", "nope", "hmm",
               "huh", "sure", "right", "hey", "oh ", "uh")
    if 1 <= len(words) <= 6 and not low.startswith(fillers):
        return cleaned.strip(" .!")
    return None


def maybe_capture_answer(text: str) -> bool:
    """Passively observe one HUMAN turn while the latch is armed. Never consumes
    the turn (normal routing continues); returns True when an identity was
    captured and written back. Expires by TTL or after N observed turns."""
    global _latch
    latch = _latch
    if latch is None:
        return False
    ttl = float(getattr(config, "ROOM_QUESTION_ANSWER_TTL_SECS", 90.0))
    if (time.monotonic() - latch["armed_at"]) > ttl:
        _latch = None
        return False

    name = _extract_identity(text)
    if name is None:
        latch["turns_left"] -= 1
        if latch["turns_left"] <= 0:
            # Nobody bit — close the question quietly so it doesn't re-ask forever.
            _latch = None
            try:
                from memory import room_model
                room_model.dismiss_question(latch["label"])
            except Exception:
                pass
        return False

    _latch = None
    try:
        from memory import room_model
        ok = room_model.record_answer(latch["label"], name, note=str(text or "")[:400])
    except Exception as exc:
        _log.debug("[room_questions] record_answer failed: %s", exc)
        return False
    if ok:
        _log.info("[room_questions] learned: %s -> %r", latch["label"], name)
    return ok


def reset() -> None:
    """Test hook."""
    global _latch, _last_asked_at
    _latch = None
    _last_asked_at = 0.0
