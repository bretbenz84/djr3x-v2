"""
intelligence/action_router.py - higher-level action selection.

The legacy interaction loop routes through a mix of command parsing, intent
classification, and feature-specific branches. This module is the first step
toward a single "given this utterance and context, what action should Rex take?"
layer. Most actions are still observe-only; a small allowlist can execute after
the router reaches the configured confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import threading
from typing import Any

import apikeys
import config
from openai import OpenAI


_log = logging.getLogger(__name__)
_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


ACTION_CATALOG: dict[str, str] = {
    "conversation.reply": "Normal conversational response; no tool/feature should run.",
    "memory.query": "User asks what Rex remembers or knows about a person, relationship, or themselves. Not for general topic knowledge.",
    "memory.forget_specific": "User asks Rex to forget/delete a specific remembered detail or topic.",
    "memory.forget_person": "User asks Rex to forget a whole person, themselves, or everyone.",
    "event.cancel": "User says a remembered plan/event is canceled, stale, or no longer happening.",
    "emotional.boundary": "User asks not to discuss a sensitive topic anymore or rejects an emotional check-in.",
    "identity.who_is_speaking": "User asks who they are, who is speaking, or whether Rex recognizes them.",
    "identity.introduce_person": "User introduces a person or relationship, such as 'this is my dad Jeff'. Not for professions, jobs, hobbies, or other personal facts.",
    "game.start": "User asks to start/play a game.",
    "game.stop": "User asks to stop/quit/end the current game.",
    "game.answer": "User is answering or choosing inside an active game.",
    "music.play": "User asks Rex to play music, a song, artist, genre, vibe, or station.",
    "music.stop": "User asks Rex to stop/pause music.",
    "music.skip": "User asks Rex to skip the current track.",
    "music.options": "User asks what music, genres, stations, or songs Rex can play.",
    "vision.describe_scene": "User asks what Rex sees or asks Rex to look/inspect something.",
    "time.query": "User asks for the current clock time.",
    "date.query": "User asks for today's date or day of week.",
    "weather.query": "User asks for weather.",
    "status.capabilities": "User asks what Rex can do.",
    "status.uptime": "User asks how long Rex has been running/awake.",
    "system.sleep": "User asks Rex to sleep, wake, quiet down, or shut down.",
}

_VALID_ACTIONS = set(ACTION_CATALOG)
EXECUTABLE_ACTIONS = {
    "memory.query",
    "memory.forget_specific",
    "event.cancel",
    "emotional.boundary",
    "identity.who_is_speaking",
    "game.start",
    "game.stop",
    "game.answer",
    "music.play",
    "music.stop",
    "music.skip",
    "music.options",
    "vision.describe_scene",
    "time.query",
    "date.query",
    "weather.query",
    "status.capabilities",
    "status.uptime",
    "system.sleep",
}

_SYSTEM_PROMPT = """You are DJ-R3X's action router.
Choose the single best action for the user's latest utterance using the catalog.
Return JSON only. Do not write a conversational reply.

Rules:
- Prefer the user's actual intent over keyword matching.
- If context.pending.pending_question exists, treat short fragments as answers
  to Rex's pending question, not as new feature commands.
- If the pending question key is favorite_music, a bare genre/artist/style like
  "classical music" is a preference answer. Use conversation.reply unless the
  user explicitly asks to play/put on/start music.
- Only use memory.forget_specific when the utterance explicitly asks to forget,
  delete, remove, erase, wipe, or clear a remembered thing. Preference statements
  like "I like Disneyland" are conversation.reply and may be learned as interests.
- If the utterance asks what you remember or know about someone, use memory.query.
  If it asks what Rex generally knows about a topic, franchise, place, hobby,
  object, or field, use conversation.reply so the main LLM can answer.
- If the utterance says a remembered plan is no longer happening, use event.cancel.
- For event.cancel, put the plan/topic being canceled in args.event_hint when possible.
- Only use emotional.boundary when the user explicitly asks not to talk about,
  ask about, mention, or bring up a topic. A bare health/sad topic like "back pain"
  is conversation.reply unless the user says not to discuss it.
- If a game is active and the utterance asks to stop, quit, end, or stop playing, use game.stop.
- If music is active and the utterance asks to stop, pause, or stop playing music, use music.stop.
- If the utterance asks for the clock time, use time.query.
- If the utterance asks for today's date or day of week, use date.query.
- If a game is active and the utterance is a short fragment that is not clearly a stop/control command, prefer game.answer over identity or general actions.
- Do not use identity.introduce_person for first-person facts like "I'm an IT systems administrator"; those are normal conversation.reply turns so memory extraction can learn them.
- If the utterance is normal chat, use conversation.reply.
- Use requires_confirmation=true when an action is broad/destructive or ambiguous. A specific forget request with a clear target does not require confirmation.
- Confidence is 0.0 to 1.0.
"""

_MUSIC_PLAY_REQUEST_RE = re.compile(
    r"\b(play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue|turn\s+on)\b",
    re.IGNORECASE,
)
_FORGET_SPECIFIC_REQUEST_RE = re.compile(
    r"\b("
    r"forget|delete|remove|erase|wipe|clear"
    r")\b.{0,80}\b("
    r"memory|remember|remembered|about|that|this|it|from your memory"
    r")\b|"
    r"\b("
    r"forget|delete|remove|erase|wipe|clear"
    r")\b\s+.+",
    re.IGNORECASE,
)
_BOUNDARY_REQUEST_RE = re.compile(
    r"\b("
    r"don'?t|do not|stop|quit|please don'?t|please do not"
    r")\b.{0,80}\b("
    r"talk|ask|bring|mention|discuss"
    r")\b|"
    r"\b(rather not|don'?t want to|do not want to|can we not|"
    r"change the subject|talk about something else|drop it|leave it alone|"
    r"no more check-?ins?)\b",
    re.IGNORECASE,
)
_TOPIC_KNOWLEDGE_QUERY_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+know|do\s+you\s+know\s+anything|"
    r"tell\s+me|explain)\s+(?:about\s+)?(?P<topic>[^?.,!;]{3,100})",
    re.IGNORECASE,
)
_PERSON_MEMORY_QUERY_RE = re.compile(
    r"\b("
    r"me|myself|me\?|my\s+|mine|i\s+told\s+you|i'?ve\s+told\s+you|"
    r"remember|memory|memories|person|people|friend|partner|wife|husband|"
    r"mom|mother|dad|father|brother|sister|kid|child|son|daughter|"
    r"jeff|joy|jt|bret"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class ActionDecision:
    action: str = "conversation.reply"
    confidence: float = 0.0
    args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    reason: str = ""

    def as_log_fields(self) -> tuple[str, float, bool, str, str]:
        return (
            self.action,
            self.confidence,
            self.requires_confirmation,
            _compact_json(self.args, max_chars=600),
            self.reason,
        )


def _compact_json(value: Any, *, max_chars: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _coerce_decision(payload: Any) -> ActionDecision:
    if not isinstance(payload, dict):
        return ActionDecision(reason="router returned non-object JSON")

    action = str(payload.get("action") or "conversation.reply").strip()
    if action not in _VALID_ACTIONS:
        action = "conversation.reply"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    args = payload.get("args")
    if not isinstance(args, dict):
        args = {}

    requires_confirmation = bool(payload.get("requires_confirmation", False))
    if action == "memory.forget_person":
        requires_confirmation = True
    if action == "memory.forget_specific" and not str(args.get("target") or "").strip():
        confidence = min(confidence, 0.45)

    reason = str(payload.get("reason") or "").strip()
    if len(reason) > 240:
        reason = reason[:237] + "..."

    return ActionDecision(
        action=action,
        confidence=confidence,
        args=args,
        requires_confirmation=requires_confirmation,
        reason=reason,
    )


def _pending_question_context(context: dict[str, Any]) -> dict[str, Any] | None:
    pending = (context or {}).get("pending")
    if not isinstance(pending, dict):
        return None
    question = pending.get("pending_question")
    return question if isinstance(question, dict) else None


def _apply_context_overrides(
    decision: ActionDecision,
    text: str,
    context: dict[str, Any],
) -> ActionDecision:
    """Deterministic safety rails for contexts the LLM router often misses."""
    if (
        decision.action == "memory.forget_specific"
        and not _FORGET_SPECIFIC_REQUEST_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="preference/topic mention is not an explicit forget request",
        )

    if (
        decision.action == "emotional.boundary"
        and not _BOUNDARY_REQUEST_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="sensitive topic mention is not an explicit boundary request",
        )

    if decision.action == "memory.query":
        topic_match = _TOPIC_KNOWLEDGE_QUERY_RE.search(text or "")
        topic = (topic_match.group("topic") if topic_match else "").strip()
        if topic and not _PERSON_MEMORY_QUERY_RE.search(topic):
            return ActionDecision(
                action="conversation.reply",
                confidence=min(float(decision.confidence or 0.0), 0.40),
                args={},
                requires_confirmation=False,
                reason="general topic knowledge question should use LLM conversation",
            )

    pending_question = _pending_question_context(context)
    if not pending_question:
        return decision

    question_key = str(pending_question.get("question_key") or "").strip()
    if (
        question_key == "favorite_music"
        and decision.action == "music.play"
        and not _MUSIC_PLAY_REQUEST_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason=(
                "pending favorite_music answer should be stored/acknowledged; "
                "no explicit play request"
            ),
        )

    return decision


def decide(text: str, context: dict[str, Any] | None = None) -> ActionDecision:
    """Return the router's best action decision for this turn."""
    if not text or not text.strip():
        return ActionDecision(reason="empty utterance")

    context = context or {}
    max_context_chars = int(getattr(config, "ACTION_ROUTER_MAX_CONTEXT_CHARS", 5000))
    user_payload = {
        "utterance": text,
        "context": context,
        "action_catalog": ACTION_CATALOG,
        "output_schema": {
            "action": "one action_catalog key",
            "confidence": "number 0.0 to 1.0",
            "args": "object; include target/game/music_query/person_name/event_hint when relevant",
            "requires_confirmation": "boolean",
            "reason": "short internal routing reason",
        },
    }
    prompt = _compact_json(user_payload, max_chars=max_context_chars)

    try:
        resp = _client.chat.completions.create(
            model=getattr(config, "ACTION_ROUTER_MODEL", config.LLM_MODEL),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=240,
        )
        raw = resp.choices[0].message.content or ""
        payload = json.loads(_strip_code_fence(raw))
        return _apply_context_overrides(_coerce_decision(payload), text, context)
    except Exception as exc:
        _log.debug("[action_router] decision failed: %s", exc)
        return ActionDecision(reason=f"router error: {type(exc).__name__}")


def log_decision(
    decision: ActionDecision,
    context: dict[str, Any] | None = None,
    *,
    mode: str = "shadow",
) -> None:
    """Write a compact action-router decision log line."""
    if not bool(getattr(config, "ACTION_ROUTER_LOG_DECISIONS", True)):
        return
    action, confidence, confirm, args, reason = decision.as_log_fields()
    legacy = (context or {}).get("legacy") or {}
    _log.info(
        "[action_router] %s action=%s confidence=%.2f confirm=%s "
        "args=%s reason=%s legacy_command=%s active_game=%s active_music=%s",
        mode,
        action,
        confidence,
        confirm,
        args,
        reason or "-",
        legacy.get("command_key"),
        (context or {}).get("active_game"),
        (context or {}).get("active_music"),
    )


def start_shadow_decision(text: str, context: dict[str, Any] | None = None) -> None:
    """Launch a background shadow decision and log the result."""
    if not bool(getattr(config, "ACTION_ROUTER_SHADOW_ENABLED", False)):
        return

    def _run() -> None:
        decision = decide(text, context)
        log_decision(decision, context, mode="shadow")

    thread = threading.Thread(target=_run, daemon=True, name="action-router-shadow")
    thread.start()
