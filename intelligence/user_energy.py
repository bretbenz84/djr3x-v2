"""
intelligence/user_energy.py - session-local user energy matching.

This keeps a cheap, current read on how the human seems to want the
conversation shaped: deeper, playful, quiet, task-focused, or just checking
that Rex is awake. It is not durable memory; it resets with the session.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Optional


_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|is|are|am|should)\b",
    re.IGNORECASE,
)
_TASK_PAT = re.compile(
    r"\b(play|stop|pause|skip|volume|weather|time|timer|search|look up|what can you do|"
    r"describe|who am i|what do you see)\b",
    re.IGNORECASE,
)
_CHECK_ALIVE_PAT = re.compile(
    r"\b(hello|hey|you there|are you there|can you hear me|testing|wake up|rex)\b",
    re.IGNORECASE,
)
_PLAYFUL_PAT = re.compile(
    r"\b(lol|haha|funny|joke|roast|messing with you|kidding|tease|silly)\b",
    re.IGNORECASE,
)
_DEPTH_PAT = re.compile(
    r"\b(honestly|actually|because|i think|i feel|it feels|i remember|the thing is|"
    r"what happened was|i'm worried|i am worried|i've been|i have been)\b",
    re.IGNORECASE,
)
_AVOID_PAT = re.compile(
    r"\b(rather not|change the subject|drop it|leave it|not now|don't ask|do not ask|"
    r"stop asking|don't want to talk|do not want to talk)\b",
    re.IGNORECASE,
)


@dataclass
class UserEnergy:
    mode: str
    engagement: str
    arousal: str
    question_appetite: str
    response_length: str
    last_updated: float
    signals: str = ""


_current: Optional[UserEnergy] = None


def clear() -> None:
    global _current
    _current = None


def snapshot() -> Optional[dict]:
    if _current is None:
        return None
    return asdict(_current)


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    prosody_features: Optional[dict] = None,
    affect_result: Optional[dict] = None,
) -> dict:
    del person_id  # reserved for future per-person calibration
    global _current

    profile = _classify(text, prosody_features=prosody_features, affect_result=affect_result)
    _current = profile
    return asdict(profile)


def build_directive() -> str:
    if _current is None:
        return ""
    if time.monotonic() - _current.last_updated > 180:
        return ""

    shape = _shape_for(_current)
    return (
        "User energy matching:\n"
        f"- Current mode: {_current.mode}; engagement={_current.engagement}; "
        f"arousal={_current.arousal}; question appetite={_current.question_appetite}.\n"
        f"- Signals: {_current.signals or 'none'}.\n"
        f"- Response shape: {shape}"
    )


def _classify(
    text: str,
    *,
    prosody_features: Optional[dict],
    affect_result: Optional[dict],
) -> UserEnergy:
    cleaned = (text or "").strip()
    words = re.findall(r"[A-Za-z']+", cleaned)
    word_count = len(words)
    lowered = cleaned.lower()
    signals: list[str] = []

    arousal_score = 0.0
    engagement_score = 0.0
    mode = "neutral"

    is_question = "?" in cleaned or bool(_QUESTION_START.search(cleaned))
    if is_question:
        signals.append("user asked a question")
        engagement_score += 0.3

    if _TASK_PAT.search(cleaned):
        mode = "task"
        signals.append("task/help intent")

    if word_count <= 3 and _CHECK_ALIVE_PAT.search(cleaned):
        mode = "check_alive"
        signals.append("checking if Rex is present")

    if _AVOID_PAT.search(cleaned):
        mode = "quiet"
        engagement_score -= 0.7
        arousal_score -= 0.3
        signals.append("avoidant boundary")
    elif _PLAYFUL_PAT.search(cleaned):
        mode = "banter"
        engagement_score += 0.4
        arousal_score += 0.2
        signals.append("playful language")
    elif word_count >= 12 or _DEPTH_PAT.search(cleaned):
        mode = "depth" if mode == "neutral" else mode
        engagement_score += 0.5
        signals.append("longer/deeper reply")
    elif word_count <= 3 and not is_question:
        mode = "quiet" if mode == "neutral" else mode
        engagement_score -= 0.35
        arousal_score -= 0.1
        signals.append("short reply")

    if prosody_features:
        prosody_arousal = float(prosody_features.get("arousal") or 0.0)
        arousal_score += prosody_arousal
        tag = prosody_features.get("tag")
        if tag:
            signals.append(str(tag))
        if prosody_arousal <= -0.35 and mode == "neutral":
            mode = "quiet"
        elif prosody_arousal >= 0.35 and mode == "neutral":
            mode = "banter"

    if affect_result:
        affect = (affect_result.get("affect") or "neutral").lower()
        needs = (affect_result.get("needs") or "none").lower()
        sensitivity = (affect_result.get("topic_sensitivity") or "none").lower()
        if affect not in {"neutral", ""}:
            signals.append(f"affect={affect}")
        if needs not in {"none", ""}:
            signals.append(f"needs={needs}")
        if sensitivity in {"heavy", "crisis"}:
            mode = "depth"
            engagement_score += 0.2
            arousal_score -= 0.2
            signals.append(f"sensitivity={sensitivity}")
        elif affect in {"happy", "excited"} and mode == "neutral":
            mode = "banter"
            engagement_score += 0.3
            arousal_score += 0.3
        elif affect in {"tired", "sad"}:
            mode = "quiet" if mode == "neutral" else mode
            arousal_score -= 0.3
        elif affect in {"anxious", "angry"}:
            mode = "depth" if mode == "neutral" else mode
            arousal_score += 0.2

    if word_count == 0:
        engagement = "unknown"
    elif engagement_score >= 0.45:
        engagement = "engaged"
    elif engagement_score <= -0.3:
        engagement = "low"
    else:
        engagement = "medium"

    if arousal_score >= 0.35:
        arousal = "high"
    elif arousal_score <= -0.25:
        arousal = "low"
    else:
        arousal = "medium"

    if mode == "neutral":
        if engagement == "engaged":
            mode = "depth"
        elif engagement == "low" or arousal == "low":
            mode = "quiet"

    question_appetite = "normal"
    if mode in {"quiet", "task", "check_alive"} or engagement == "low":
        question_appetite = "low"
    elif mode in {"depth", "banter"} and engagement == "engaged":
        question_appetite = "open"

    response_length = {
        "task": "brief",
        "check_alive": "brief",
        "quiet": "short",
        "banter": "short",
        "depth": "medium",
    }.get(mode, "short")

    return UserEnergy(
        mode=mode,
        engagement=engagement,
        arousal=arousal,
        question_appetite=question_appetite,
        response_length=response_length,
        last_updated=time.monotonic(),
        signals=", ".join(signals[:6]),
    )


def _shape_for(profile: UserEnergy) -> str:
    if profile.mode == "task":
        return (
            "Handle the request efficiently first. Keep the bit small. Do not "
            "add an unrelated personal question."
        )
    if profile.mode == "check_alive":
        return (
            "Briefly acknowledge presence and invite the next move. One line is "
            "usually enough."
        )
    if profile.mode == "quiet":
        return (
            "Lower energy. Keep it short, leave space, and avoid piling on "
            "questions. Ask at most one gentle question only if needed."
        )
    if profile.mode == "depth":
        return (
            "Slow down and respond with substance. Continue the current thread. "
            "One thoughtful follow-up is okay; do not pivot to random banter."
        )
    if profile.mode == "banter":
        return (
            "Match the playfulness with brisk banter. Keep roasts surface-level "
            "and friendly, and do not derail the topic."
        )
    return (
        "Use normal Rex cadence: concise, responsive, and at most one question "
        "if it genuinely helps the turn."
    )
