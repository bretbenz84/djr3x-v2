"""
intelligence/motion_route.py — the rescue interpreter for spoken drive routes.

See docs/motion_route_tool_plan.md. One narrow job: when
`action_router.classify_explicit_motion_sequence` returns the tri-state ``None`` —
"this IS an attempted route and I could not parse it" — ask the model to plan the
legs instead of telling the human "I couldn't safely parse that whole route."

Why this is its OWN call and not the persona reply call
-------------------------------------------------------
docs/tool_router_scope.md, Phase 2 carve-out, 2026-08-14: eight explicit
impersonation requests in one sitting. Four got no tool call at all — the model
performed in prose instead ("prose wins"), one called the tool with the previous
turn's argument, and the shadow router returned the right answer every time.
*Routing was never the hard part; a persona-loaded reply call at conversational
temperature was.* A route command is an unambiguous imperative that the tri-state
has ALREADY deterministically detected, so it gets a call that cannot wander:
a units-and-conventions system prompt with no character in it, exactly two tools,
forced tool choice, and no conversation history (plan §11 — "do that again but
further" resolving against a stale prior turn is the `target='speaker'` bug with
wheels attached).

The second tool is the decline. A single forced tool cannot say no, which would
make the ASR-garbage arm of the corpus ("and the other, and the other, and the
other...") unrepresentable as anything but a drive command.

House style, borrowed from tool_router.shadow_decide: this module NEVER raises.
Errors come back inside the result dict. The caller sits inside
interaction._handle_speech_segment's fast-takeover ladder, whose except-handler
logs at DEBUG under an unrelated message ("action router shadow start failed") and
drops the turn to conversation — so an exception escaping here would turn a network
blip into a silently mis-routed command.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import apikeys
import config
from intelligence import connectivity as _connectivity
from openai import OpenAI

_log = logging.getLogger(__name__)
_client = _connectivity.guard_client(OpenAI(api_key=apikeys.OPENAI_API_KEY), "motion_route")

_DECLINE_TOOL = {
    "type": "function",
    "function": {
        "name": "motion_route_decline",
        "description": (
            "Call this when the utterance is NOT a drive route he can plan: it is "
            "chatter, a mis-transcription, a figure of speech, a story about "
            "movement, a negated or cancelled command, or it asks him to drive to a "
            "place or an object rather than through plain geometry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string",
                           "description": "a few words on why, for the log"},
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}

# Units, conventions and the server-side ceilings — and NOTHING about who Rex is.
# The clamps are stated so the model plans inside them rather than having a good
# route truncated; action_router.route_tool_to_decisions enforces them regardless,
# because a stated limit is guidance and only the translator is a guarantee.
_SYSTEM = (
    "You translate one spoken command into a drive plan for a small wheeled robot. "
    "Output is a tool call and nothing else.\n"
    "Conventions: the robot's left and right, never the speaker's. Distances are "
    "METRES (1 foot = 0.3 m, 'a bit' or 'a smidge' = 0.2 m, 'a step' = 0.3 m); "
    "angles are DEGREES (a quarter turn = 90, an about-face = 180, a full spin = "
    "360). Every step needs an explicit direction word and a positive magnitude.\n"
    "Limits: at most {steps} steps, {step_m} m per step, {total_m} m of travel and "
    "{total_deg} degrees of rotation in total. A plan outside them is thrown away "
    "whole, so plan inside them.\n"
    "Read the WHOLE utterance before planning. This is ordinary speech, not typed "
    "input, so expect it to be messy and plan the command underneath the mess:\n"
    "- disfluent repeats are one step ('turn, turn to your left' is one turn);\n"
    "- self-corrections keep what the speaker landed on and drop what they took "
    "back ('turn, never mind, move forward four feet' is one forward move);\n"
    "- an elided verb inherits the one before it ('three feet, and turn a little "
    "bit right' is a forward move and then a turn);\n"
    "- surrounding chatter that is not addressed to the robot is ignored;\n"
    "- when a clause asks for something OTHER than driving ('...and tell me what "
    "you see'), plan only the driving and leave the rest alone.\n"
    # SCOPING. Every decline rule below judges the COMMAND; the ignore-chatter
    # rule above judges a SPAN. That mismatch was a live bug: field 2026-08-23
    # 13:49, "No, cause I don't have not. We don't have places to go. Turn to
    # your right, then move forward five feet." declined 8 times in 12, and the
    # model's own reasons said why — "no real drive command UNTIL THE FINAL
    # ROUTE", "it also says no places to go". It could see the command and let a
    # negation, and the bare word "places", outvote it. Measured on the mined
    # field corpus, this rewrite takes chatter-wrapped commands from 31/42 to
    # 36/42 with 39/39 chatter-only and 18/18 negated declines both unchanged.
    "Find the drive command first, then judge only it. Chatter, an answer to "
    "an earlier question, an aside to someone else, a false start, a "
    "self-correction: set them aside, do not weigh them. Heard on the robot — "
    "'No, "
    "cause I don't have not. We don't have places to go. Turn to your right, "
    "then move forward five feet.' — the command is the last sentence and the "
    "plan is a turn and then a move; the muttering in front of it is not "
    "evidence against it. A negation, a figure of speech, or a word like "
    "'place' or 'go' sitting elsewhere in the utterance is never a reason to "
    "refuse a command that is plainly there. A plan of a single step is a "
    "complete, correct answer — do not decline because there is only one "
    "movement. "
    "\n"
    "Decline when, with the rest set aside, no drive command is left, or when "
    "the command's own target is a place or an object he would have to find "
    "('go to the couch', 'drive to the kitchen') instead of plain geometry — "
    "never merely because the word 'place' was said. A drive verb is not a "
    "drive command: a discourse marker opening a sentence about something "
    "else ('moving forward, I want to try something', 'going forward, "
    "let's...'), a figure of speech ('let's move on', 'let's roll', 'back me "
    "up'), a story about someone else moving, a route someone is retelling, a "
    "plan for later ('we should do a lap sometime'). A negation counts only "
    "when it negates the drive command itself — 'don't move', 'turn left, no, "
    "forget it' — not when it negates something else nearby. "
)


def _rescue_schema(schema: "dict | None") -> "dict | None":
    """The shared motion.route schema, re-framed for a call that has no other tool.

    The STEP shape is taken verbatim from tool_router._TOOL_DEFS — that is the half
    where arg-name drift bites, and it must have exactly one definition. What differs
    is the route-level framing. The shared description tells the model "a SINGLE
    movement is motion_turn / motion_move / motion_arc — use those instead", which is
    right on the reply call, where those tools are sitting next to it, and wrong
    here, where this tool and a decline are the entire surface.

    Measured, 2026-08-22, replaying the 13 real None-arm utterances through the
    unmodified schema: 11 declined and 7 of those said so in as many words —
    "single turn command, not a multi-step route", "single movement plus turn phrased
    as a route", "Cancelled turn; remaining command is a s[ingle move]". Every one
    was a real command, and every one would have drawn the very denial this call
    exists to delete. interaction._handle_motion_route already routes a one-step plan
    to the single-verb executor, so a one-step plan was always safe to receive — the
    schema was just telling the model not to send it.
    """
    if not schema:
        return None
    out = json.loads(json.dumps(schema))          # deep copy; never mutate the shared one
    fn = out["function"]
    fn["description"] = (
        "Plan the driving this command asks for, as an ordered list of steps. This "
        "is the ONLY way to move on this call, so use it for a single movement as "
        "readily as for a route — one step is a complete plan. Give every step an "
        "explicit direction word and a positive magnitude. Never for a figure of "
        "speech, never for a route someone is RETELLING, never for a negated or "
        "cancelled command, and never for a place or object to drive to — he has no "
        "way to find one, so only geometry belongs here."
    )
    steps = fn["parameters"]["properties"]["steps"]
    steps["minItems"] = 1
    steps["description"] = ("the movements, in the order he should drive them — one "
                            "is fine")
    return out


def _prompt() -> str:
    def _num(name: str, default: float) -> str:
        try:
            value = float(getattr(config, name, default))
        except (TypeError, ValueError):
            value = default
        return f"{value:g}"

    return _SYSTEM.format(
        steps=_num("MOTION_ROUTE_MAX_STEPS", 6.0),
        step_m=_num("MOTION_ROUTE_MAX_STEP_M", 1.5),
        total_m=_num("MOTION_ROUTE_MAX_TOTAL_M", 3.0),
        total_deg=_num("MOTION_ROUTE_MAX_TOTAL_DEG", 720.0),
    )


def available() -> bool:
    """Whether a rescue attempt can be made at all right now.

    Offline the answer is no and the caller keeps the existing spoken denial (plan
    §4.4): the local reply model gets no tool surface today and small-model tool
    accuracy is unproven, so with the link down the deterministic classifiers stay
    the whole story. `connectivity.is_offline()` is a cached state read, never a
    network call, so this is free to ask on the turn's hot path.
    """
    if not bool(getattr(config, "MOTION_ROUTE_ENABLED", True)):
        return False
    return not _connectivity.is_offline()


def interpret(text: str) -> dict[str, Any]:
    """One forced tool call planning `text` as a route. Never raises.

    Returns ``{"args", "declined", "reason", "secs", "error"}``:
      * ``args`` — the motion_route arguments to hand
        ``action_router.route_tool_to_decisions``, or None;
      * ``declined`` — the model said this is not a route (its own tool);
      * ``error`` — set when the call itself failed. Both of those, and a None
        ``args``, mean the caller falls back to today's spoken denial.
    """
    from intelligence import llm_compat, tool_router

    out: dict[str, Any] = {"args": None, "declined": False, "reason": "", "secs": 0.0,
                           "error": None}
    schema = _rescue_schema(tool_router.tool_schema_for("motion.route"))
    if schema is None:
        out["error"] = "motion.route has no tool schema"
        return out

    model = (str(getattr(config, "MOTION_ROUTE_MODEL", "") or "")
             or llm_compat.conversation_model())
    timeout = float(getattr(config, "MOTION_ROUTE_TIMEOUT_SECS", 6.0))
    tools = [schema, _DECLINE_TOOL]
    t0 = time.perf_counter()
    try:
        resp = _create(
            llm_compat, model=model, tools=tools, text=text, timeout=timeout,
            forced=bool(getattr(config, "MOTION_ROUTE_FORCE_TOOL_CHOICE", True)),
        )
    except Exception as exc:
        out["secs"] = time.perf_counter() - t0
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["secs"] = time.perf_counter() - t0

    try:
        calls = getattr(resp.choices[0].message, "tool_calls", None) or []
    except Exception as exc:
        out["error"] = f"malformed response: {type(exc).__name__}: {exc}"
        return out
    if not calls:
        # tool_choice="auto" (or a model that ignored "required") answering in prose
        # IS a decline — the prose-wins failure mode, read the only way that is safe.
        out["declined"] = True
        out["reason"] = "no tool call"
        return out

    fn = calls[0].function
    name = str(getattr(fn, "name", "") or "")
    try:
        args = json.loads(getattr(fn, "arguments", "") or "{}")
    except json.JSONDecodeError:
        args = None
    if name == "motion_route_decline":
        out["declined"] = True
        out["reason"] = str((args or {}).get("reason") or "declined")
        return out
    if name != schema["function"]["name"]:
        out["error"] = f"unexpected tool {name!r}"
        return out
    if not isinstance(args, dict):
        out["error"] = "unparseable tool arguments"
        return out
    out["args"] = args
    return out


def _create(llm_compat, *, model: str, tools: list, text: str, timeout: float,
            forced: bool):
    """The hosted call, with the tool_choice=required -> auto downgrade.

    Copied from features/web_search.py's forced-search call: an SDK or a model that
    rejects tool_choice="required" must degrade to "auto" rather than lose the turn.
    "auto" is a weaker contract, not a broken one — with a two-tool surface and no
    persona, no tool call is read as a decline above."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": _prompt()},
            {"role": "user", "content": str(text or "")[:1000]},
        ],
        max_tokens=int(getattr(config, "MOTION_ROUTE_MAX_TOKENS", 400)),
        timeout=timeout,
        # No reasoning_effort, deliberately. gpt-5.4-mini is a reasoning model so
        # llm_compat DROPS temperature — the plan's "temperature low" is a no-op —
        # and effort is not a free substitute: anything but "none" makes the API
        # refuse a tool-bearing request outright, which llm_compat now enforces at
        # the chokepoint. The determinism this call needs comes from its SHAPE
        # instead: no persona, no history, two tools, forced choice.
        extra={"tools": tools,
               "tool_choice": "required" if forced else "auto"},
    )
    try:
        return llm_compat.create(_client, **kwargs)
    except TypeError as exc:
        if forced and "tool_choice" in str(exc):
            _log.debug("[motion_route] tool_choice=required rejected; retrying auto")
            kwargs["extra"] = {"tools": tools, "tool_choice": "auto"}
            return llm_compat.create(_client, **kwargs)
        raise


def log_shadow(text: str, result: dict[str, Any],
               decisions: "list | None", refusal: Optional[str], *,
               executed: bool) -> None:
    """One `[motion_route]` JSON line per rescue attempt — the Phase-0/1 record.

    Emitted whether or not the route drove, so the shadow week and the live weeks
    produce the same shape and tools/motion_route_report.py can read both.
    """
    record = {
        "utterance": str(text or ""),
        "executed": bool(executed),
        "declined": bool(result.get("declined")),
        "secs": round(float(result.get("secs") or 0.0), 3),
        "steps": [{"action": d.action, "args": d.args} for d in (decisions or [])],
        "route_args": result.get("args"),
    }
    if result.get("reason"):
        record["reason"] = result["reason"]
    if refusal:
        record["refused"] = refusal
    if result.get("error"):
        record["error"] = result["error"]
    _log.info("[motion_route] %s", json.dumps(record, ensure_ascii=False, default=str))
