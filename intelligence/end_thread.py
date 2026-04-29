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


_CLOSURE_PAT = re.compile(
    r"\b(that'?s all|that is all|that'?s it|that is it|all good|i'?m good|"
    r"i am good|nothing else|no more|let'?s leave it there|leave it there|"
    r"we can stop|let'?s stop|moving on|anyway,? never ?mind|never ?mind|"
    r"thanks|thank you|appreciate it|sounds good|fair enough|got it|"
    r"makes sense|okay,? cool|ok,? cool|bye|goodbye|good-bye|"
    r"see you|see ya|later|talk to you later|talk later|nice speaking|"
    r"nice talking)\b",
    re.IGNORECASE,
)
_SHORT_ACK_PAT = re.compile(
    r"^\s*(ok|okay|cool|nice|yeah|yep|alright|right|gotcha|thanks|thank you)\s*[.!]?\s*$",
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


def clear() -> None:
    global _state, _last_assistant_had_question
    with _lock:
        _state = None
        _last_assistant_had_question = False


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
    global _state
    with _lock:
        _state = state
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


def is_grace_active() -> bool:
    with _lock:
        return _state is not None and time.monotonic() < _state.quiet_until


def can_proactive_purpose(purpose: str) -> bool:
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
