"""
intelligence/dialogue_act.py - deterministic turn-shape triage.

This layer decides whether a user utterance is probably answering Rex's most
recent turn before any executable action router gets to claim it. It is
session-local and intentionally cheap: no LLM calls, just context plus
conservative evidence checks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import re
import time
from typing import Any, Optional


_FRAME_TTL_SECS = 120.0
_frames: deque["RexTurnFrame"] = deque(maxlen=16)

_QUESTION_START_RE = re.compile(
    r"^\s*(?:who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should)\b",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|correct|right|affirmative|"
    r"no|nope|nah|negative)\b",
    re.IGNORECASE,
)
_TERSE_REPLY_RE = re.compile(
    r"^\s*(?:"
    r"i\s+don'?t\s+know|not\s+sure|maybe|probably|"
    r"that'?s\s+(?:right|correct|wrong|not\s+right|not\s+correct|fine|okay|ok)|"
    r"same|exactly|pretty\s+much|not\s+really|kind\s+of|sort\s+of|"
    r"not\s+(?:anymore|any\s+more|now)|"
    r"no\s+longer|can'?t\s+make\s+it|won'?t\s+make\s+it|"
    r"that'?s\s+not\s+happening(?:\s+anymore)?|"
    r"it'?s\s+(?:off|over|done|finished|cancelled|canceled)"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_STATUS_RETRACTION_RE = re.compile(
    r"\b(?:not\s+happening|no\s+longer|not\b.{0,40}\banymore|"
    r"can'?t\s+make\s+it|won'?t\s+make\s+it|called?\s+off|"
    r"cancel(?:ed|led)|postpon(?:ed|e)|reschedul(?:ed|e)|"
    r"already\s+(?:happened|passed)|it'?s\s+over)\b",
    re.IGNORECASE,
)
_IDENTITY_CONTROL_RE = re.compile(
    r"\b(?:call\s+me|rename\s+me(?:\s+to)?|my\s+name\s+is|"
    r"you\s+(?:got|have)\s+my\s+name\s+wrong|"
    r"that['’]?s\s+not\s+my\s+name|that\s+isn['’]?t\s+my\s+name|"
    r"that['’]?s\s+not\s+[A-Za-z][A-Za-z' -]{1,40}\s*,?\s+"
    r"(?:i\s+am|i['’]?m|im)\s+[A-Za-z][A-Za-z' -]{0,40})\b",
    re.IGNORECASE,
)
_MEMORY_CONTROL_RE = re.compile(
    r"\b(?:forget|delete|remove|erase|wipe|clear)\b.{0,80}\b"
    r"(?:memory|remember|remembered|that|this|it|what\s+i\s+(?:just\s+)?said)\b|"
    r"\b(?:don'?t|do\s+not)\s+(?:remember|store|save)\b.{0,80}\b"
    r"(?:that|this|it|what\s+i\s+(?:just\s+)?said)\b",
    re.IGNORECASE,
)
_EXPLICIT_COMMAND_RE = re.compile(
    r"\b(?:"
    r"tell\s+(?:me|us)\s+(?:a\s+)?(?:joke|pun)|"
    r"roast\s+(?:me|us|the\s+room|him|her|them)|"
    r"say\s+something\s+(?:funny|hilarious)|"
    r"do\s+(?:a\s+|your\s+)?(?:bit|riff|dance|pose)|"
    r"look\s+(?:at|for|around|left|right|up|down)|"
    r"play\s+(?:music|a\s+song|something)|"
    r"play\s+(?:some\s+)?(?:[A-Za-z0-9' -]{1,40}\s+)?music|"
    r"(?:put|throw)\s+on\s+(?:some\s+)?[A-Za-z0-9' -]{1,40}|"
    r"stop\s+(?:playing|music|the\s+music)|"
    r"skip\s+(?:this|song|track)|"
    r"what\s+(?:time|date|day)\b|"
    r"what\s+do\s+you\s+see|"
    r"what\s+can\s+you\s+do|"
    r"what'?s\s+your\s+uptime|"
    r"what(?:'s| is)\s+(?:the\s+)?(?:weather|forecast|temperature)|"
    r"(?:weather|temperature)\s+(?:forecast|outside)|"
    r"is\s+it\s+(?:raining|hot|cold)\b"
    r")\b",
    re.IGNORECASE,
)
_DIRECT_SLEEP_RE = re.compile(
    r"^\s*(?:please\s+)?(?:hey\s+rex\s+|rex\s+)?"
    r"(?:go\s+to\s+sleep|sleep)(?:\s+please)?[.!?]*\s*$",
    re.IGNORECASE,
)
_MEMORY_HINT_RE = re.compile(
    r"\b(?:remember|told me|you said|plan|planned|schedule|trip|congrats|"
    r"how'?s|how is|how did|did .* go|survive|ready for|counting down)\b",
    re.IGNORECASE,
)
_TOPIC_AFTER_RE = re.compile(
    r"\b(?:your|the|that|this)\s+"
    r"([A-Za-z0-9][A-Za-z0-9' -]{2,60}?)\s+"
    r"(?:on|coming|plan|plans|trip|event|thing|today|tomorrow|tonight)\b",
    re.IGNORECASE,
)


@dataclass
class RexTurnFrame:
    text: str
    source: str = "assistant_turn"
    topic: str = ""
    target_person_id: Optional[int] = None
    target_name: Optional[str] = None
    expected_reply_types: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    ttl_secs: float = _FRAME_TTL_SECS

    def active(self, now: Optional[float] = None) -> bool:
        current = time.monotonic() if now is None else now
        return (current - self.created_at) <= max(0.0, float(self.ttl_secs))

    def for_person(self, person_id: Optional[int]) -> bool:
        if self.target_person_id is None:
            return True
        if person_id is None:
            return False
        try:
            return int(self.target_person_id) == int(person_id)
        except (TypeError, ValueError):
            return False

    def as_context(self) -> dict[str, Any]:
        data = asdict(self)
        data["age_secs"] = round(time.monotonic() - self.created_at, 3)
        return data


@dataclass
class DialogueActDecision:
    label: str
    confidence: float
    reason: str
    frame: Optional[RexTurnFrame] = None
    blocked_actions: list[str] = field(default_factory=list)
    skip_action_router: bool = False

    def as_context(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence or 0.0), 3),
            "reason": self.reason,
            "skip_action_router": bool(self.skip_action_router),
            "blocked_actions": list(self.blocked_actions),
            "frame": self.frame.as_context() if self.frame else None,
        }


def clear() -> None:
    _frames.clear()


def note_rex_turn(
    text: str,
    *,
    source: Optional[str] = None,
    topic: Optional[str] = None,
    target_person_id: Optional[int] = None,
    target_name: Optional[str] = None,
    expected_reply_types: Optional[list[str]] = None,
    blocked_actions: Optional[list[str]] = None,
    ttl_secs: Optional[float] = None,
) -> Optional[RexTurnFrame]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    inferred_source = source or _infer_source(cleaned)
    inferred_expected = expected_reply_types or _infer_expected_reply_types(
        cleaned,
        inferred_source,
    )
    frame = RexTurnFrame(
        text=cleaned,
        source=inferred_source,
        topic=(topic or _infer_topic(cleaned, inferred_source)).strip(),
        target_person_id=_coerce_person_id(target_person_id),
        target_name=target_name,
        expected_reply_types=list(inferred_expected),
        blocked_actions=list(
            blocked_actions
            if blocked_actions is not None
            else _infer_blocked_actions(inferred_source, inferred_expected)
        ),
        ttl_secs=max(1.0, float(ttl_secs if ttl_secs is not None else _FRAME_TTL_SECS)),
    )
    _frames.append(frame)
    return frame


def active_frame(
    *,
    person_id: Optional[int] = None,
    max_age_secs: Optional[float] = None,
) -> Optional[RexTurnFrame]:
    now = time.monotonic()
    max_age = float(max_age_secs if max_age_secs is not None else _FRAME_TTL_SECS)
    for frame in reversed(_frames):
        if not frame.active(now):
            continue
        if (now - frame.created_at) > max_age:
            continue
        if frame.for_person(person_id):
            return frame
    return None


def active_frame_context(person_id: Optional[int] = None) -> Optional[dict[str, Any]]:
    frame = active_frame(person_id=person_id)
    return frame.as_context() if frame else None


def classify(
    text: str,
    context: Optional[dict[str, Any]] = None,
    *,
    person_id: Optional[int] = None,
) -> DialogueActDecision:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return DialogueActDecision("general_chat", 0.0, "empty utterance")

    context = context or {}
    pending = context.get("pending") or {}
    frame = active_frame(person_id=person_id)
    direct_kind = _direct_control_kind(cleaned)

    if pending.get("identity_prompt_active") or pending.get("prompted_name_confirmation"):
        if direct_kind in {"identity_control", "memory_control"}:
            return DialogueActDecision(direct_kind, 0.96, "explicit control during identity prompt")
        return _answer_decision(frame, "reply to active identity prompt", confidence=0.92)

    if bool(context.get("active_game")):
        if _explicit_game_or_music_stop(cleaned):
            return DialogueActDecision("new_command", 0.94, "explicit game/music stop")
        return DialogueActDecision("game_answer", 0.92, "active game owns short turns")

    if direct_kind is not None:
        return DialogueActDecision(direct_kind, 0.95, "explicit command/control evidence")

    if pending.get("awaiting_followup_event") and _looks_like_contextual_reply(cleaned, frame):
        return _answer_decision(frame, "reply to pending event follow-up", confidence=0.95)

    if pending.get("pending_question") and _looks_like_contextual_reply(cleaned, frame):
        return _answer_decision(frame, "reply to pending question", confidence=0.93)

    if frame is not None and _looks_like_contextual_reply(cleaned, frame):
        return _answer_decision(frame, "reply to last Rex turn", confidence=0.90)

    return DialogueActDecision("general_chat", 0.55, "no active reply frame claimed turn")


def action_blocked_by_dialogue(
    action: str,
    decision: Optional[DialogueActDecision],
) -> bool:
    if decision is None:
        return False
    return action in set(decision.blocked_actions or [])


def _answer_decision(
    frame: Optional[RexTurnFrame],
    reason: str,
    *,
    confidence: float,
) -> DialogueActDecision:
    blocked = list(frame.blocked_actions) if frame is not None else [
        "identity.name_correction",
        "identity.introduce_person",
    ]
    return DialogueActDecision(
        "answer_to_rex",
        confidence,
        reason,
        frame=frame,
        blocked_actions=blocked,
        skip_action_router=True,
    )


def _direct_control_kind(text: str) -> Optional[str]:
    if _IDENTITY_CONTROL_RE.search(text):
        return "identity_control"
    if _MEMORY_CONTROL_RE.search(text):
        return "memory_control"
    if _DIRECT_SLEEP_RE.match(text):
        return "new_command"
    if _EXPLICIT_COMMAND_RE.search(text):
        return "new_command"
    if "?" in text and _QUESTION_START_RE.match(text):
        return "new_command"
    return None


def _explicit_game_or_music_stop(text: str) -> bool:
    return bool(re.search(r"^\s*(?:stop|quit|end|stop playing|pause|skip)\b", text, re.I))


def _looks_like_contextual_reply(text: str, frame: Optional[RexTurnFrame]) -> bool:
    if not text:
        return False
    if _direct_control_kind(text) is not None:
        return False
    if "?" in text and _QUESTION_START_RE.match(text):
        return False
    words = re.findall(r"[A-Za-z0-9']+", text)
    if _YES_NO_RE.match(text) or _TERSE_REPLY_RE.match(text):
        return True
    if _STATUS_RETRACTION_RE.search(text):
        return True
    if frame is not None and frame.expected_reply_types:
        if len(words) <= 12 and not _QUESTION_START_RE.match(text):
            return True
    return False


def _infer_source(text: str) -> str:
    if _MEMORY_HINT_RE.search(text):
        return "memory_hint"
    if "?" in text:
        return "question"
    return "assistant_turn"


def _infer_expected_reply_types(text: str, source: str) -> list[str]:
    expected: list[str] = []
    if "?" in text:
        expected.extend(["answer", "confirmation", "dismissal"])
    if source in {"memory_followup", "celebration_checkin", "emotional_checkin", "memory_hint"}:
        expected.extend(["status_update", "cancel_event", "dismissal"])
    seen: set[str] = set()
    return [item for item in expected if not (item in seen or seen.add(item))]


def _infer_blocked_actions(source: str, expected_reply_types: list[str]) -> list[str]:
    blocked = {"identity.name_correction", "identity.introduce_person"}
    if source in {"memory_followup", "memory_hint"} or "cancel_event" in expected_reply_types:
        blocked.update({"game.answer"})
    return sorted(blocked)


def _infer_topic(text: str, source: str) -> str:
    match = _TOPIC_AFTER_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    if source in {"memory_followup", "memory_hint"}:
        words = [
            w
            for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text)
            if w.lower() not in {
                "hey", "you", "your", "the", "that", "this", "for", "and",
                "are", "ready", "remember", "counting", "down", "hope",
                "have", "got",
            }
        ]
        if words:
            return " ".join(words[:4])
    return ""


def _coerce_person_id(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
