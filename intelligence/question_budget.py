"""
intelligence/question_budget.py - session-local pacing for Rex's questions.

Rex should be curious, but not interview-y. This module tracks recent assistant
questions and recent user engagement so optional follow-ups can back off when
the conversation already has enough open prompts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import re
import threading
import time
from typing import Optional

import config


_DEPTH_PAT = re.compile(
    r"\b(because|actually|honestly|i think|i feel|it feels|the thing is|"
    r"what happened|i remember|i've been|i have been)\b",
    re.IGNORECASE,
)
_PLAYFUL_PAT = re.compile(r"\b(lol|haha|funny|joke|kidding|messing with you)\b", re.I)
_AVOID_PAT = re.compile(
    r"\b(rather not|change the subject|drop it|leave it|not now|don't ask|"
    r"do not ask|stop asking|don't want to talk|do not want to talk)\b",
    re.IGNORECASE,
)

_URGENT_KINDS = {
    "emotional_checkin",
    "identity_prompt",
    "relationship_inquiry",
    "off_camera_identity",
    "face_reveal",
    "newcomer_identity",
    # First-meeting onboarding burst rides its own private MIN/MAX cap
    # (intelligence/onboarding.py), so it must bypass the global pacing budget
    # rather than be muzzled by the deliberately-tight friend-protecting cap.
    "newcomer_baseline",
}


@dataclass
class QuestionBudgetSnapshot:
    window_secs: float
    max_questions: int
    recent_questions: int
    recent_engaged_replies: int
    remaining: int
    relaxed_extra_available: bool
    exhausted: bool


_lock = threading.Lock()
_question_times: deque[float] = deque()
_engaged_reply_times: deque[float] = deque()
_last_user_turn_at: float = 0.0
# Anti-interview cadence: count of consecutive Rex turns that ENDED with a question
# (reset by a statement turn or a long gap). The time-window budget above rations
# new-topic questions; this separately stops an on-topic question-every-turn
# interrogation once a topic has opened.
_consecutive_question_turns: int = 0
_last_rex_turn_at: float = 0.0


def clear() -> None:
    global _last_user_turn_at, _consecutive_question_turns, _last_rex_turn_at
    with _lock:
        _question_times.clear()
        _engaged_reply_times.clear()
        _last_user_turn_at = 0.0
        _consecutive_question_turns = 0
        _last_rex_turn_at = 0.0


def note_rex_utterance(text: str) -> None:
    """Record one assistant turn: time-window question budget (questions only) AND the
    consecutive-question-turn counter (every turn — a statement turn resets it)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return
    now = time.monotonic()
    is_question = "?" in cleaned
    with _lock:
        global _consecutive_question_turns, _last_rex_turn_at
        _prune_locked(now)
        # A long pause between Rex turns means the prior interview cooled off.
        reset_secs = max(0.0, float(getattr(config, "INTERVIEW_CADENCE_RESET_SECS", 120.0)))
        if reset_secs and _last_rex_turn_at and (now - _last_rex_turn_at) > reset_secs:
            _consecutive_question_turns = 0
        _last_rex_turn_at = now
        if is_question:
            _question_times.append(now)
            _consecutive_question_turns += 1
        else:
            _consecutive_question_turns = 0  # a statement turn breaks the streak


def consecutive_question_turns() -> int:
    """Consecutive Rex turns that ended with a question (0 if the streak has cooled)."""
    now = time.monotonic()
    with _lock:
        reset_secs = max(0.0, float(getattr(config, "INTERVIEW_CADENCE_RESET_SECS", 120.0)))
        if reset_secs and _last_rex_turn_at and (now - _last_rex_turn_at) > reset_secs:
            return 0
        return _consecutive_question_turns


def should_force_statement_turn() -> bool:
    """True when Rex has ended too many turns in a row with a question — the next reply
    should be a statement/reaction instead, to break an interview cadence."""
    if not bool(getattr(config, "INTERVIEW_CADENCE_CLAMP_ENABLED", True)):
        return False
    k = max(1, int(getattr(config, "INTERVIEW_CADENCE_MAX_CONSECUTIVE_QUESTIONS", 3)))
    return consecutive_question_turns() >= k


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    answered_question: Optional[dict] = None,
) -> None:
    """Record whether the user's reply suggests appetite for follow-ups."""
    del person_id  # reserved for future per-person calibration
    global _last_user_turn_at

    cleaned = (text or "").strip()
    if not cleaned:
        return

    now = time.monotonic()
    engaged = _looks_engaged(cleaned, answered_question=answered_question)
    with _lock:
        _prune_locked(now)
        _last_user_turn_at = now
        if engaged:
            _engaged_reply_times.append(now)


def note_answered_question(answered_question: Optional[dict]) -> None:
    if not answered_question:
        return
    answer = answered_question.get("answer_text") or ""
    note_user_turn(answer, answered_question=answered_question)


def can_ask(kind: str = "optional") -> bool:
    """
    Return True when Rex may ask another question right now.

    Urgent social/safety questions bypass the budget. Optional questions get a
    small grace slot when the user has recently answered with real engagement.
    """
    if kind in _URGENT_KINDS or kind.startswith("urgent:"):
        return True

    snap = snapshot()
    if snap["recent_questions"] < snap["max_questions"]:
        return True
    return bool(snap["relaxed_extra_available"])


def snapshot() -> dict:
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        window = _window_secs()
        max_questions = _max_questions()
        question_count = len(_question_times)
        engaged_count = len(_engaged_reply_times)
        relaxed_limit = max_questions + _engaged_extra_slots(now)
        relaxed_extra_available = question_count < relaxed_limit
        remaining = max(0, max_questions - question_count)
        return asdict(
            QuestionBudgetSnapshot(
                window_secs=window,
                max_questions=max_questions,
                recent_questions=question_count,
                recent_engaged_replies=engaged_count,
                remaining=remaining,
                relaxed_extra_available=relaxed_extra_available
                and question_count >= max_questions,
                exhausted=question_count >= relaxed_limit,
            )
        )


def build_directive() -> str:
    snap = snapshot()
    if snap["recent_questions"] <= 0 and snap["recent_engaged_replies"] <= 0:
        return ""

    if snap["exhausted"]:
        shape = (
            "Question budget is spent: do not ask another optional or unrelated "
            "question this turn. Answer, acknowledge, reflect, or make a brief "
            "observation instead. Only urgent identity, safety, or emotional "
            "care questions may bypass this."
        )
    elif snap["remaining"] <= 0 and snap["relaxed_extra_available"]:
        shape = (
            "Base question budget is full, but the user has been engaging. One "
            "more tightly related follow-up is okay; avoid a new interview topic."
        )
    else:
        shape = (
            f"{snap['remaining']} optional question(s) remain in the current "
            "window. Ask at most one, and only if it naturally serves this turn."
        )

    return (
        "Question budget:\n"
        f"- Recent assistant questions: {snap['recent_questions']}/"
        f"{snap['max_questions']} in {snap['window_secs']:.0f}s; "
        f"engaged user replies: {snap['recent_engaged_replies']}.\n"
        f"- Response shape: {shape}"
    )


def _window_secs() -> float:
    return max(10.0, float(getattr(config, "QUESTION_BUDGET_WINDOW_SECS", 90.0)))


def _max_questions() -> int:
    return max(1, int(getattr(config, "QUESTION_BUDGET_MAX_QUESTIONS", 2)))


def _engaged_extra_slots(now: float) -> int:
    if not _engaged_reply_times:
        return 0
    grace_secs = max(10.0, float(getattr(config, "QUESTION_BUDGET_ENGAGED_GRACE_SECS", 45.0)))
    if (now - _engaged_reply_times[-1]) <= grace_secs:
        return max(0, int(getattr(config, "QUESTION_BUDGET_ENGAGED_EXTRA", 1)))
    return 0


def _prune_locked(now: float) -> None:
    cutoff = now - _window_secs()
    while _question_times and _question_times[0] < cutoff:
        _question_times.popleft()
    while _engaged_reply_times and _engaged_reply_times[0] < cutoff:
        _engaged_reply_times.popleft()


def _looks_engaged(text: str, *, answered_question: Optional[dict]) -> bool:
    if _AVOID_PAT.search(text):
        return False
    words = re.findall(r"[A-Za-z']+", text)
    if answered_question:
        return len(words) >= 2 or bool(_DEPTH_PAT.search(text))
    if len(words) >= 8:
        return True
    if _DEPTH_PAT.search(text) or _PLAYFUL_PAT.search(text):
        return True
    return False
