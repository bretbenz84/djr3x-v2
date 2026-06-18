"""
intelligence/end_thread.py - session-local end-of-thread grace.

Sometimes the best conversational move is to let a thread land. This module
detects user closure cues and gives Rex a short grace period where optional
follow-ups, visual curiosity, and idle chatter back off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import threading
import time
from typing import Optional

import config


# Strong closure cues — explicit goodbyes / "we're done here" signals only.
# Bare politeness ("thanks", "thank you", "sounds good", "got it", "cool") was
# REMOVED from here: it falsely closed conversational replies like "Well thank
# you" (a reply to a compliment, not a goodbye), arming the grace window that
# muzzled Rex's proactive re-engagement. Those soft acks now close the thread
# ONLY via _SHORT_ACK_PAT below — i.e. when the WHOLE utterance is just the ack
# AND Rex had just asked a question.
_CLOSURE_PAT = re.compile(
    r"\b(that'?s all|that is all|that'?s it|that is it|all good|i'?m good|"
    r"i am good|nothing else|no more|let'?s leave it there|leave it there|"
    r"we can stop|let'?s stop|moving on|anyway,? never ?mind|never ?mind|"
    r"don'?t want to talk about (?:it|that|this)(?: anymore| again)?|"
    r"do not want to talk about (?:it|that|this)(?: anymore| again)?|"
    r"i told you i didn'?t want to talk about (?:it|that|this)|"
    r"i told you i did not want to talk about (?:it|that|this)|"
    r"bye|goodbye|good-bye|"
    r"see you|see ya|later|talk to you later|talk later|nice speaking|"
    r"nice talking|nice chatting|i'?m\s+going\s+to\s+go|"
    r"i\s+am\s+going\s+to\s+go|i\s+have\s+to\s+go|gotta\s+go)\b",
    re.IGNORECASE,
)
_SHORT_ACK_PAT = re.compile(
    r"^\s*(ok|okay|cool|nice|yeah|yep|alright|right|gotcha|thanks|thank you)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# Genuine sign-offs — "I'm leaving / signing off" cues, a deliberately narrower
# subset of _CLOSURE_PAT. Only these arm the farewell-departure latch: a topic
# closure like "moving on" or "that's all" should NOT make Rex go dormant if the
# person then steps off-camera, but an actual goodbye should.
_FAREWELL_PAT = re.compile(
    r"\b(bye|goodbye|good-bye|see you|see ya|"
    r"talk to you later|talk later|catch you later|"
    r"nice (?:talking|chatting|speaking)|"
    r"i'?m\s+(?:gonna|going to)\s+(?:go|head out|take off|get going)|"
    r"i\s+(?:gotta|have to|need to|hafta)\s+(?:go|head out|take off|get going|run)|"
    r"gotta\s+go|heading\s+out|i'?m\s+off|i'?m\s+out|i'?m\s+leaving|"
    r"take care)\b",
    re.IGNORECASE,
)
_THANKS_FOR_ASKING_PAT = re.compile(r"\bthanks?(?:\s+you)?\s+for\s+asking\b", re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should)\b",
    re.IGNORECASE,
)
_SWITCH_PAT = re.compile(
    r"\b(by the way|speaking of|new subject|change the subject|let'?s talk about)\b",
    re.IGNORECASE,
)


@dataclass
class EndThreadState:
    closing_pending: bool
    reason: str
    user_text: str
    quiet_until: float
    detected_at: float


_lock = threading.Lock()
_state: Optional[EndThreadState] = None
_last_assistant_had_question: bool = False
# Monotonic time of the last explicit verbal farewell, and the latch set once that
# farewell is followed by the person leaving the camera view. While the latch is
# live, Rex treats the conversation as fully closed — no proactive re-engagement —
# until they come back (a new turn or a presence return) or the safety cap lapses.
_farewell_at: Optional[float] = None
_conversation_closed_at: Optional[float] = None


def clear() -> None:
    global _state, _last_assistant_had_question, _farewell_at, _conversation_closed_at
    with _lock:
        _state = None
        _last_assistant_had_question = False
        _farewell_at = None
        _conversation_closed_at = None


def note_assistant_turn(text: str) -> None:
    global _last_assistant_had_question
    cleaned = (text or "").strip()
    if not cleaned:
        return
    with _lock:
        _last_assistant_had_question = "?" in cleaned


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    answered_question: Optional[dict] = None,
) -> Optional[dict]:
    del person_id  # reserved for future per-person pacing
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    if _starts_new_thread(cleaned):
        clear()
        return None

    reason = _closure_reason(cleaned, answered_question=answered_question)
    if not reason:
        # A real new user turn means the old grace period has done its job.
        if len(re.findall(r"[A-Za-z']+", cleaned)) >= 4:
            clear()
        return None

    now = time.monotonic()
    state = EndThreadState(
        closing_pending=True,
        reason=reason,
        user_text=cleaned,
        quiet_until=now + _grace_secs(),
        detected_at=now,
    )
    is_farewell = bool(_FAREWELL_PAT.search(cleaned))
    global _state, _farewell_at
    with _lock:
        _state = state
        if is_farewell:
            _farewell_at = now
    return asdict(state)


def mark_closure_spoken() -> None:
    global _state
    with _lock:
        if _state is not None:
            _state.closing_pending = False
            _state.quiet_until = max(_state.quiet_until, time.monotonic() + _grace_secs())


def snapshot() -> Optional[dict]:
    with _lock:
        if _state is None:
            return None
        return asdict(_state)


def pending_closure() -> Optional[dict]:
    with _lock:
        if _state is None or not _state.closing_pending:
            return None
        if time.monotonic() > _state.quiet_until:
            return None
        return asdict(_state)


def recent_farewell(within_secs: Optional[float] = None) -> bool:
    """True when the user gave an explicit verbal goodbye recently enough that a
    camera departure now should be read as 'they said bye and left.'"""
    window = _farewell_window_secs() if within_secs is None else float(within_secs)
    with _lock:
        if _farewell_at is None:
            return False
        return (time.monotonic() - _farewell_at) <= window


def note_farewell_departure() -> bool:
    """Latch the conversation closed: the person left the camera view shortly after
    an explicit goodbye. Keeps Rex from re-engaging an empty room until they come
    back. Returns True if it latched (i.e. a recent farewell was on record)."""
    now = time.monotonic()
    global _conversation_closed_at
    with _lock:
        if _farewell_at is None or (now - _farewell_at) > _farewell_window_secs():
            return False
        _conversation_closed_at = now
        return True


def note_presence_return() -> None:
    """A departed person came back into view — a clean slate. The prior thread is
    over and Rex re-greets fresh (the return reaction handles that), so drop both
    the farewell dormancy AND any lingering end-of-thread grace from the goodbye
    so normal proactive life resumes once he's said hello."""
    clear()


def is_conversation_closed() -> bool:
    """True while a farewell-then-departure has Rex dormant. Self-expires after the
    safety cap so a missed return can never wedge him permanently silent."""
    global _conversation_closed_at
    with _lock:
        if _conversation_closed_at is None:
            return False
        if (time.monotonic() - _conversation_closed_at) > _farewell_closed_max_secs():
            _conversation_closed_at = None
            return False
        return True


def is_grace_active() -> bool:
    # A closed conversation (explicit goodbye + left view) is a hard, longer-lived
    # form of grace: every inline proactive path already backs off on this flag, so
    # folding it in here muzzles idle banter / monologue / re-engagement for free.
    if is_conversation_closed():
        return True
    with _lock:
        return _state is not None and time.monotonic() < _state.quiet_until


def can_proactive_purpose(purpose: str) -> bool:
    # Conversation closed → nobody's there; allow nothing, not even check-ins.
    if is_conversation_closed():
        return False
    if not is_grace_active():
        return True
    return purpose in {"emotional_checkin", "identity_prompt", "relationship_inquiry"}


def build_directive() -> str:
    state = pending_closure()
    if not state:
        return ""
    return (
        "End-of-thread grace:\n"
        f"- The user appears to be closing this thread ({state['reason']}).\n"
        "- Primary purpose: give one short landing acknowledgement. Do not ask "
        "a new question, do not pivot topics, and do not add visual curiosity. "
        "Let the silence be acceptable."
    )


def _grace_secs() -> float:
    return max(5.0, float(getattr(config, "END_OF_THREAD_GRACE_SECS", 35.0)))


def _farewell_window_secs() -> float:
    return max(5.0, float(getattr(config, "FAREWELL_DEPART_WINDOW_SECS", 120.0)))


def _farewell_closed_max_secs() -> float:
    return max(30.0, float(getattr(config, "FAREWELL_CLOSED_MAX_SECS", 600.0)))


def _starts_new_thread(text: str) -> bool:
    return "?" in text or bool(_QUESTION_START.search(text)) or bool(_SWITCH_PAT.search(text))


def _closure_reason(text: str, *, answered_question: Optional[dict]) -> str:
    if _THANKS_FOR_ASKING_PAT.search(text):
        return ""
    if _CLOSURE_PAT.search(text):
        return "explicit closure cue"
    if _SHORT_ACK_PAT.match(text):
        with _lock:
            last_was_question = _last_assistant_had_question
        if answered_question or last_was_question:
            return "short acknowledgement after Rex prompt"
    return ""
