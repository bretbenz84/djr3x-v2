"""
intelligence/tool_router.py — Phase 0 (SHADOW ONLY) of the tool-calling router.

See docs/tool_router_scope.md. This module never executes anything: for each
routed turn it asks the conversation model to pick a tool for the same
utterance/context the shipping regex+JSON router saw, and logs the choice NEXT
TO the shipped decision so the two can be compared over real traffic. Cutover
happens category-by-category later, on measured agreement — not here.

Design notes:
  * Tool schemas are keyed off action_router.ACTION_SPECS (the catalog source of
    truth). For Phase 0 the per-action parameter schemas + "when" hints live in
    this module's _TOOL_DEFS table; tests/test_tool_router.py enforces that the
    table covers every spec, so a new ActionSpec without a tool definition fails
    CI instead of silently missing from the shadow. At Phase 4 cleanup these
    merge into ActionSpec itself.
  * Args are judged SEMANTICALLY in the report (did it extract "left"/"90"),
    not against the executor's exact kwarg names — arg-contract alignment is
    cutover work, not shadow work.
  * OFF by default (config.TOOL_ROUTER_SHADOW_ENABLED): the shadow costs one
    small hosted call per routed turn. Enable it in user_config.py for a
    collection week, then run tools/tool_router_report.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

import apikeys
import config
from intelligence.action_router import ACTION_SPECS
from openai import OpenAI

_log = logging.getLogger(__name__)
from intelligence import connectivity as _connectivity
_client = _connectivity.guard_client(OpenAI(api_key=apikeys.OPENAI_API_KEY), "tool_router")

# ── tool definitions: action key → (when-hint, JSON-schema properties, required) ──
# Keep "when" to ONE sentence — it is appended to the spec description and is the
# model's routing hint. Empty params = a no-argument tool.

_NUM = {"type": "number"}
_STR = {"type": "string"}

_TOOL_DEFS: dict[str, tuple[str, dict, list]] = {
    "conversation.reply": (
        "DEFAULT: the user is just talking — reply in words, call no tool.", {}, []),
    "conversation.repair": (
        "The user says Rex misheard/misunderstood or asks him to try again.", {}, []),
    "memory.query": (
        "Recall stored memory about a person ('what do you remember about me/Jeff?').",
        {"subject": {**_STR, "description": "who or what to recall"}}, []),
    "memory.forget_specific": (
        "Delete one stored fact/preference the user names.",
        {"statement": {**_STR, "description": "the thing to forget, as said"}}, ["statement"]),
    "memory.recent_discard": (
        "The user disowns or is baffled by something Rex just attributed to them.", {}, []),
    "memory.forget_person": (
        "Forget an entire person (requires confirmation downstream).",
        {"person_name": _STR}, ["person_name"]),
    "event.cancel": (
        "A planned event the user says is off/cancelled.",
        {"event_hint": {**_STR, "description": "which event"}}, []),
    "emotional.boundary": (
        "The user sets a topic boundary ('don't ask about my ex').", {}, []),
    "identity.who_is_speaking": (
        "'Who am I?' / 'do you know who's speaking?' — immediate identity check.", {}, []),
    "identity.name_correction": (
        "The user corrects their own name.",
        {"correct_name": _STR}, ["correct_name"]),
    "identity.introduce_person": (
        "The user introduces someone new who is present.",
        {"person_name": _STR}, []),
    "humor.tell_joke": ("An explicit request for a joke.", {}, []),
    "humor.roast": (
        "An explicit invitation to roast someone.",
        {"target": {**_STR, "description": "who to roast; empty = the speaker"}}, []),
    "humor.free_bit": ("An explicit request to 'do a bit' / riff freely.", {}, []),
    "performance.dj_bit": ("An explicit request for a DJ bit/announcement.", {}, []),
    "performance.body_beat": ("An explicit request to dance/move to the music.", {}, []),
    "performance.mood_pose": (
        "An explicit request to strike a pose/act out a mood.",
        {"mood": _STR}, []),
    "performance.impersonate": (
        "An explicit request to impersonate someone.",
        {"who": {**_STR, "description": "person to impersonate; 'me' = the speaker"}}, ["who"]),
    "character.preference_query": (
        "Asking Rex about his OWN tastes/preferences.",
        {"topic": _STR}, []),
    "game.start": (
        "Start a verbal game (Jeopardy, Trivia, I Spy, 20 Questions, Word Association).",
        {"game": _STR}, ["game"]),
    "game.stop": ("Stop/quit the current game.", {}, []),
    "game.answer": (
        "An answer/guess for the ACTIVE game (context shows active_game).",
        {"answer": _STR}, ["answer"]),
    "music.play": (
        "Play music: a song, artist, genre, vibe, or station.",
        {"music_query": {**_STR, "description": "what to play, as said"}}, ["music_query"]),
    "music.stop": ("Stop the music that is playing.", {}, []),
    "music.skip": ("Skip to the next track.", {}, []),
    "music.options": ("Asking what music is available.", {}, []),
    "vision.describe_scene": ("Asking what Rex can SEE right now.", {}, []),
    "vision.snapshot": ("An explicit request to take a picture.", {}, []),
    "time.query": ("Asking the current clock time.", {}, []),
    "date.query": ("Asking today's date/day (NOT holiday explanations).", {}, []),
    "weather.query": (
        "Asking about the weather, forecast, or temperature — outdoor OR "
        "indoor ('what temperature is it inside?' reads the onboard climate "
        "sensor).", {}, []),
    "status.capabilities": ("Asking what Rex can do.", {}, []),
    "status.uptime": ("Asking how long Rex has been running.", {}, []),
    "motion.turn": (
        "Turn the drive base in place.",
        {"direction": {"type": "string", "enum": ["left", "right", "around"]},
         "degrees": {**_NUM, "description": "turn size if the user gave one"}},
        ["direction"]),
    "motion.move": (
        "Drive straight forward or backward.",
        {"direction": {"type": "string", "enum": ["forward", "backward"]},
         "distance": {**_NUM, "description": "distance if given"},
         "unit": {"type": "string", "enum": ["feet", "meters", "inches"]}},
        ["direction"]),
    "motion.arc": (
        "Drive in a curve while moving ('swing left as you go').",
        {"direction": {"type": "string", "enum": ["left", "right"]}}, ["direction"]),
    "motion.come": ("'Come here' — find the speaker and approach them.", {}, []),
    "motion.stop": ("Stop moving RIGHT NOW ('stop', 'halt' while driving).", {}, []),
    "motion.explore": ("An invitation to wander/explore the room.", {}, []),
    "web.search": (
        "The user asks about news, current events, or anything that needs LIVE "
        "up-to-date information Rex cannot know — wars, elections, scores, "
        "prices, product launches, 'what's going on with X', follow-up "
        "questions about a news story Rex mentioned. Runs a real web search "
        "and answers from the results. NOT for things Rex already knows or "
        "can sense (weather, time, what he sees).",
        {"query": {**_STR, "description":
                   "what to search for — the topic, not the full sentence"}}, []),
    "system.sleep": (
        "An explicit instruction to go to sleep / quiet mode ('go to sleep', "
        "'quiet mode') — NOT a full power-down.", {}, []),
    "system.shutdown": (
        "An explicit instruction to fully power down ('shut down', 'power off', "
        "'turn yourself off') — including polite forms like 'can you shut down, "
        "please?'. Never for shutting down some OTHER thing (music, a server).",
        {}, []),
}


class ToolCallRequested(Exception):
    """Raised by the lean reply stream when the model chose a LIVE tool instead
    of prose. Deliberately an exception: it unwinds the streaming/TTS machinery
    before any text is spoken, and the reply pipeline catches it and dispatches
    to the existing executor for that action."""

    def __init__(self, action: str, args: dict):
        super().__init__(action)
        self.action = str(action)
        self.args = dict(args or {})


# Phase 1 live set (docs/tool_router_scope.md): the intent-backed actions where
# every measured shipped-miss lived, all served by the existing
# _handle_classified_intent executor. Humor/character keep their working fast
# lanes and stay shadow-only for now.
_DEFAULT_LIVE_ACTIONS = (
    "time.query", "date.query", "weather.query",
    "status.capabilities", "status.uptime",
    "vision.describe_scene", "music.options",
    "system.sleep", "system.shutdown", "web.search",
    "event.cancel", "memory.query", "identity.who_is_speaking",
    "music.play", "music.stop", "music.skip",
)


def live_actions() -> "set[str]":
    if not bool(getattr(config, "TOOL_ROUTER_LIVE_ENABLED", True)):
        return set()
    return {str(a) for a in getattr(config, "TOOL_ROUTER_LIVE_ACTIONS",
                                    _DEFAULT_LIVE_ACTIONS)}


def live_reply_tools() -> "list[dict] | None":
    """Tool schemas for the LIVE subset only, or None when cutover is off.
    Attached to the lean reply call — routing rides the call that already
    happens, so a live tool costs zero extra LLM round-trips."""
    live = live_actions()
    if not live:
        return None
    tools = [t for t in tool_schemas()
             if _NAME_TO_KEY.get(t["function"]["name"]) in live]
    return tools or None


def resolve_tool_call(name: str, arguments: str) -> "tuple[str, dict] | None":
    """(action_key, args) for an accumulated streamed tool call, or None when the
    name is unknown or the action isn't live (never execute a non-live tool)."""
    key = _NAME_TO_KEY.get(str(name or "").strip())
    if key is None or key not in live_actions():
        return None
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            args = {}
    except json.JSONDecodeError:
        args = {}
    return key, args


def _tool_name(key: str) -> str:
    return key.replace(".", "_")


_NAME_TO_KEY = {_tool_name(spec.key): spec.key for spec in ACTION_SPECS}


def tool_schemas() -> list[dict]:
    """OpenAI tools array derived from ACTION_SPECS + _TOOL_DEFS.

    conversation.reply is deliberately NOT a tool — "no tool call" IS the reply
    decision, which keeps the model's default path identical to today's.
    """
    tools: list[dict] = []
    for spec in ACTION_SPECS:
        if spec.key == "conversation.reply":
            continue
        when, props, required = _TOOL_DEFS[spec.key]
        tools.append({
            "type": "function",
            "function": {
                "name": _tool_name(spec.key),
                "description": f"{spec.description} {when}".strip(),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        })
    return tools


_SYSTEM = (
    "You are the action-selection layer of DJ R3X, a physical droid with a drive "
    "base, camera, music player, games, and a person-memory. Given ONE user "
    "utterance plus context, decide whether it asks for an ACTION. If it clearly "
    "does, call the matching tool (extract arguments from the utterance). If it is "
    "ordinary conversation — including banter that merely MENTIONS jokes, music, "
    "moving, or memory without requesting them — call NO tool and reply with the "
    "single word: reply. When context shows an active game, bare answers belong to "
    "game_answer. Never call a tool speculatively."
)


def shadow_decide(text: str, context: dict[str, Any] | None = None) -> dict:
    """One tool-choice decision (no execution). Returns
    {"action", "args", "secs", "error"?} — action is an ACTION_SPECS key."""
    from intelligence import llm_compat

    model = str(getattr(config, "TOOL_ROUTER_SHADOW_MODEL", "") or "") or llm_compat.conversation_model()
    payload = {"utterance": text, "context": context or {}}
    t0 = time.perf_counter()
    try:
        resp = llm_compat.create(
            _client,
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)[:4000]},
            ],
            max_tokens=96,
            temperature=0,
            timeout=float(getattr(config, "TOOL_ROUTER_SHADOW_TIMEOUT_SECS", 8.0)),
            extra={"tools": tool_schemas(), "tool_choice": "auto"},
        )
        secs = time.perf_counter() - t0
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return {"action": "conversation.reply", "args": {}, "secs": secs}
        fn = calls[0].function
        key = _NAME_TO_KEY.get(str(fn.name or ""), "conversation.reply")
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": str(fn.arguments)[:200]}
        return {"action": key, "args": args, "secs": secs}
    except Exception as exc:
        return {"action": None, "args": {}, "secs": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


def start_shadow(text: str, context: dict[str, Any] | None, shipped_action: str) -> None:
    """Fire-and-forget shadow comparison for one live turn. Never blocks the turn."""
    if not bool(getattr(config, "TOOL_ROUTER_SHADOW_ENABLED", False)):
        return

    def _run() -> None:
        result = shadow_decide(text, context)
        record = {
            "utterance": text,
            "shipped": shipped_action,
            "tool": result.get("action"),
            "args": result.get("args"),
            "agree": result.get("action") == shipped_action,
            "secs": round(float(result.get("secs") or 0.0), 3),
        }
        if result.get("error"):
            record["error"] = result["error"]
        # Single JSON payload per line — tools/tool_router_report.py parses these.
        _log.info("[tool_router_shadow] %s", json.dumps(record, ensure_ascii=False, default=str))

    threading.Thread(target=_run, daemon=True, name="tool-router-shadow").start()
