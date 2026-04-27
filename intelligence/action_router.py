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
import threading
from typing import Any

import apikeys
import config
from openai import OpenAI


_log = logging.getLogger(__name__)
_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


ACTION_CATALOG: dict[str, str] = {
    "conversation.reply": "Normal conversational response; no tool/feature should run.",
    "memory.query": "User asks what Rex remembers or knows about a person, relationship, or themselves.",
    "memory.forget_specific": "User asks Rex to forget/delete a specific remembered detail or topic.",
    "memory.forget_person": "User asks Rex to forget a whole person, themselves, or everyone.",
    "event.cancel": "User says a remembered plan/event is canceled, stale, or no longer happening.",
    "emotional.boundary": "User asks not to discuss a sensitive topic anymore or rejects an emotional check-in.",
    "identity.who_is_speaking": "User asks who they are, who is speaking, or whether Rex recognizes them.",
    "identity.introduce_person": "User introduces a person or relationship, such as 'this is my dad Jeff'.",
    "game.start": "User asks to start/play a game.",
    "game.stop": "User asks to stop/quit/end the current game.",
    "game.answer": "User is answering or choosing inside an active game.",
    "music.play": "User asks Rex to play music, a song, artist, genre, vibe, or station.",
    "music.stop": "User asks Rex to stop/pause music.",
    "music.skip": "User asks Rex to skip the current track.",
    "music.options": "User asks what music, genres, stations, or songs Rex can play.",
    "vision.describe_scene": "User asks what Rex sees or asks Rex to look/inspect something.",
    "time.query": "User asks for the time or date.",
    "weather.query": "User asks for weather.",
    "status.capabilities": "User asks what Rex can do.",
    "status.uptime": "User asks how long Rex has been running/awake.",
    "system.sleep": "User asks Rex to sleep, wake, quiet down, or shut down.",
}

_VALID_ACTIONS = set(ACTION_CATALOG)
EXECUTABLE_ACTIONS = {
    "event.cancel",
    "emotional.boundary",
}

_SYSTEM_PROMPT = """You are DJ-R3X's action router.
Choose the single best action for the user's latest utterance using the catalog.
Return JSON only. Do not write a conversational reply.

Rules:
- Prefer the user's actual intent over keyword matching.
- If the utterance asks to forget/delete/remove a specific memory, use memory.forget_specific and put the target phrase in args.target.
- If the utterance says a remembered plan is no longer happening, use event.cancel.
- For event.cancel, put the plan/topic being canceled in args.event_hint when possible.
- If the utterance asks not to talk about a topic anymore, use emotional.boundary.
- If the utterance is normal chat, use conversation.reply.
- Use requires_confirmation=true when an action is destructive or ambiguous.
- Confidence is 0.0 to 1.0.
"""


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
    if action in {"memory.forget_specific", "memory.forget_person"}:
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
        return _coerce_decision(payload)
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
