"""
intelligence/object_qa.py — cross-generator ledger of object questions + answers.

Field incident 2026-08-08 09:43/09:45: visual curiosity asked "what's in the
bowl?", Bret answered "frosted mini wheats", and two minutes later visual
curiosity asked the identical question again — the four object-question
generators (visual curiosity, held-object remark, room-change ask, templated
room questions) each had their own cooldowns but shared no per-object memory,
and the answer itself was never stored anywhere.

This module is that shared memory, session-scoped:

  note_asked(label)          — any generator records "I asked about this" and
                               arms a passive answer latch.
  mark_asked_labels(text, …) — freeform-LLM generators (visual curiosity)
                               don't know which object their question targeted;
                               this detects candidate labels in the spoken
                               question and records those.
  maybe_capture_answer(text) — interaction calls this on every human turn
                               (never consumes it). While the latch is armed,
                               the reply is stored verbatim as the answer.
  known_answer(label)        — what the human said, for prompt injection so
                               Rex riffs on the answer instead of re-asking.
  was_asked(label)           — asked this session, answered or not.

Answers also write through to the room model's human_note (ask_status →
'answered') so they survive the session, WITHOUT touching human_name — "what's
in the bowl?" → "frosted mini wheats" is an answer about the bowl, not a
rename of it. Identity renames stay room_questions' job.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import config

_log = logging.getLogger(__name__)

# label -> {"asked_at", "source", "question", "answer", "answered_at"}
_asked: dict[str, dict] = {}
# One pending answer at a time (same convention as room_questions):
# {"label", "armed_at", "turns_left"}
_latch: Optional[dict] = None

_NON_ANSWERS = (
    "i don't know", "i dont know", "no idea", "not sure", "dunno",
    "nothing", "don't worry", "dont worry", "never mind", "nevermind",
    "none of your", "wouldn't you like to know", "i already told you",
)


def _clean(label) -> str:
    return str(label or "").strip().lower()


def note_asked(label: str, *, source: str = "", question: str = "") -> None:
    """Record that Rex just asked about `label` and arm the answer latch."""
    global _latch
    key = _clean(label)
    if not key:
        return
    entry = _asked.setdefault(key, {})
    entry.update({
        "asked_at": time.monotonic(),
        "source": source,
        "question": str(question or "")[:200],
    })
    _latch = {
        "label": key,
        "armed_at": time.monotonic(),
        "turns_left": int(getattr(config, "OBJECT_QA_ANSWER_TURNS", 2)),
    }
    try:
        from memory import room_model
        room_model.note_question_asked(key)
    except Exception:
        pass
    # Lifelike delivery: glance at the thing being asked about. One hook here
    # covers every generator; it no-ops unless the object was seen recently.
    try:
        from intelligence import consciousness
        consciousness.request_object_glance(key, source=source or "object_qa")
    except Exception:
        pass
    _log.info("[object_qa] asked about %r (source=%s)", key, source or "?")


def mark_asked_labels(question_text: str, candidate_labels, *, source: str = "") -> list[str]:
    """Freeform generators: find which candidate labels the spoken question
    actually names and record those. Returns the matched labels."""
    text = str(question_text or "").lower()
    if not text:
        return []
    matched = []
    for label in candidate_labels or []:
        key = _clean(label)
        if key and re.search(r"\b" + re.escape(key) + r"\b", text):
            matched.append(key)
    for key in matched:
        note_asked(key, source=source, question=question_text)
    return matched


def was_asked(label: str) -> bool:
    return _clean(label) in _asked


def known_answer(label: str) -> Optional[str]:
    """The human's session answer about this object, if any. Falls back to the
    room model's persisted note so an answer survives a restart."""
    entry = _asked.get(_clean(label))
    if entry and entry.get("answer"):
        return str(entry["answer"])
    try:
        from memory import rex_db
        row = rex_db.fetchone(
            "SELECT human_note FROM room_objects WHERE label = ? "
            "AND ask_status = 'answered' AND human_note IS NOT NULL",
            (_clean(label),),
        )
        if row and row["human_note"]:
            return str(row["human_note"])
    except Exception:
        pass
    return None


def _looks_like_answer(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return False
    low = cleaned.lower()
    if any(p in low for p in _NON_ANSWERS):
        return False
    if cleaned.endswith("?"):
        return False
    return True


def maybe_capture_answer(text: str) -> Optional[str]:
    """Observe one HUMAN turn; never consumes it. While the latch is armed the
    reply is stored verbatim as the answer for the asked-about object. Returns
    the label on capture, else None."""
    global _latch
    latch = _latch
    if latch is None:
        return None
    ttl = float(getattr(config, "OBJECT_QA_ANSWER_TTL_SECS", 90.0))
    if (time.monotonic() - latch["armed_at"]) > ttl:
        _latch = None
        return None
    if not _looks_like_answer(text):
        latch["turns_left"] -= 1
        if latch["turns_left"] <= 0:
            _latch = None
        return None

    _latch = None
    key = latch["label"]
    answer = " ".join(str(text or "").split()).strip(" .!")[:300]
    entry = _asked.setdefault(key, {"asked_at": time.monotonic()})
    entry["answer"] = answer
    entry["answered_at"] = time.monotonic()
    try:
        from memory import rex_db
        rex_db.execute(
            "UPDATE room_objects SET human_note = ?, ask_status = 'answered' "
            "WHERE label = ?",
            (answer, key),
        )
    except Exception:
        pass
    _log.info("[object_qa] learned about %r: %r", key, answer)
    return key


def reset() -> None:
    """Test hook / session boundary."""
    global _asked, _latch
    _asked = {}
    _latch = None
