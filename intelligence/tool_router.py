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
from intelligence import performance_plan
from openai import OpenAI

_log = logging.getLogger(__name__)
from intelligence import connectivity as _connectivity
_client = _connectivity.guard_client(OpenAI(api_key=apikeys.OPENAI_API_KEY), "tool_router")

# ── tool definitions: action key → (when-hint, JSON-schema properties, required) ──
# Keep "when" to ONE sentence — it is appended to the spec description and is the
# model's routing hint. Empty params = a no-argument tool.

_NUM = {"type": "number"}
_STR = {"type": "string"}

# The physical-performance enums are GENERATED from performance_plan, never
# retyped here. Both executors coerce an unrecognized name to a default
# (canonical_body_beat(beat) or "thinking_tilt"), so a free-text arg lets the
# model invent "spin the mystery servo" and Rex answers with a head tilt — which
# reads as broken hardware, not a bad schema. Generating the list also keeps it
# from rotting the day a new beat is added to performance_plan.
_BODY_BEATS = sorted(performance_plan.BODY_BEAT_NAMES)
_MOOD_POSES = sorted(performance_plan.MOOD_POSE_NAMES)

_TOOL_DEFS: dict[str, tuple[str, dict, list]] = {
    "conversation.reply": (
        "DEFAULT: the user is just talking — reply in words, call no tool.", {}, []),
    "conversation.repair": (
        "The user says Rex misheard/misunderstood or asks him to try again.", {}, []),
    # The in-session caveat is new 2026-08-13. This tool answers from the stored
    # person dossier and CANNOT see the current conversation, so calling it for
    # something said a minute ago throws the transcript away: "Do you remember what
    # I said that they were gonna let you do?" ("they" = the radar sensors, Rex's
    # own topic 19 seconds earlier) resolved to no person and Rex answered "Who's
    # 'they,' Bret?" about a thing he had just been talking about.
    "memory.query": (
        "Recall STORED memory about a person from earlier sessions ('what do you "
        "remember about me/Jeff?', 'what's my sister's name?'). NOT for anything "
        "said in the CURRENT conversation — that is already in the transcript in "
        "front of you, so just answer it, and resolve pronouns like 'they'/'it' "
        "against what was said a moment ago rather than calling this.",
        {"subject": {**_STR, "description": "the PERSON to recall"}}, []),
    # memory.* / emotional.boundary went LIVE 2026-08-13 (Phase 2b). These hints
    # carry the negative examples the regex family learned the expensive way,
    # because the model is now the first thing standing between an idiom and a
    # delete.
    #
    # The arg is "target", not "statement": ActionSpec, the JSON-prose router prompt,
    # _apply_context_overrides, _handle_router_takeover_action and _execute_command
    # ALL say args.target, so a "statement" arg arrived in a key nobody read and the
    # executor took its empty-target branch. Same arg-name drift class as
    # performance.impersonate's who->target fix.
    "memory.forget_specific": (
        "The user asks Rex to forget a specific thing he has STORED about them — a "
        "named fact, preference, person, pet, or topic. Never a dismissive idiom "
        "('forget it, I'll do it myself', 'forget the traffic, we made it') and "
        "never ordinary housekeeping ('remove the lid', 'delete that file'). Rex "
        "does not delete on this call: he reads back what would go and asks for a "
        "yes first.",
        {"target": {**_STR, "description":
                    "the thing to forget, in the user's own words "
                    "('my dog Scout', 'what I said about my job')"}}, ["target"]),
    "memory.recent_discard": (
        "The user disowns, retracts, or is baffled by something Rex just attributed "
        "to them ('forget I said that', \"don't store that\") — scoped to the last "
        "turn, never to a named stored fact, and never 'don't forget X', which is "
        "the opposite request.", {}, []),
    "memory.forget_person": (
        "Forget an entire person (requires confirmation downstream).",
        {"person_name": _STR}, ["person_name"]),
    "event.cancel": (
        "A planned event the user says is off/cancelled.",
        {"event_hint": {**_STR, "description": "which event"}}, []),
    # topic/behavior are new 2026-08-13: with no parameters the model could not name
    # what was being closed, so _handle_router_emotional_boundary guessed from
    # _boundary_fallback_topic() — the guessing that let an unattributed "Drop it."
    # mute the wrong topic (audit 2026-08-13) — and always wrote behavior="mention",
    # the BROADEST kind (boundaries.is_blocked treats a mention row as blocking ask
    # and roast too), so "don't joke about my weight" silently became "never mention
    # weight". Both stay optional: an unnamed topic still falls back exactly as before.
    "emotional.boundary": (
        "The user asks Rex to STOP raising a topic for good ('don't ask about my "
        "ex', 'stop bringing up my job'). This writes a durable consent record, so "
        "it is not for a passing mood, not for 'let's talk about something else' "
        "(that is just conversation — follow them), not for an invitation dressed "
        "as a refusal ('don't ask how I got it, long story' WANTS the question), "
        "and never for the release of a boundary ('you can ask about that again').",
        {"topic": {**_STR, "description":
                   "what to stop raising, in a word or two; omit if unclear"},
         "behavior": {"type": "string", "enum": ["mention", "ask", "roast"],
                      "description":
                      "mention = don't bring it up at all (broadest), ask = don't "
                      "ask about it, roast = don't joke about it"}}, []),
    "identity.who_is_speaking": (
        "'Who am I?' / 'do you know who's speaking?' — immediate identity check.", {}, []),
    "identity.name_correction": (
        "The user corrects their own name.",
        {"correct_name": _STR}, ["correct_name"]),
    "identity.introduce_person": (
        "The user introduces someone new who is present.",
        {"person_name": _STR}, []),
    # humor.* / performance.* went LIVE 2026-08-13 (config.TOOL_ROUTER_LIVE_ACTIONS).
    # These hints carry the negative examples the regex families had learned the
    # hard way, because the model is now the only thing standing between banter
    # and a performance.
    "humor.tell_joke": (
        "An explicit request for a joke, pun, or one-liner — never banter that "
        "merely MENTIONS jokes.", {}, []),
    "humor.roast": (
        "An explicit invitation for Rex to roast or tease a PERSON ('roast me', "
        "'roast Dave') — never narration or an idiom ('this heat could roast a "
        "turkey' fired the regex, audit 2026-08-13).",
        {"target": {**_STR, "description":
                    "'speaker' for the person talking, 'room' for everyone "
                    "present, otherwise the name they said; empty = the speaker"}},
        []),
    "humor.free_bit": (
        "An open 'be funny' request ('say something funny', 'do a bit', 'make me "
        "laugh') with no joke format and no roast target.", {}, []),
    "performance.dj_bit": (
        "A request for DJ patter, hype, or a station-break line — music_play is "
        "the tool that actually starts audio.", {}, []),
    # body_beat/mood_pose take a CANONICAL name: performance_plan coerces anything
    # it doesn't recognize to thinking_tilt/thinking, so a free-text arg would let
    # an invented pose reach the servos as a shrug. The enum makes that
    # unrepresentable; interaction._router_execution_block_reason is the backstop
    # that declines rather than performing the default.
    "performance.body_beat": (
        "A request for ONE named physical gesture — pick a beat from the enum, "
        "and if nothing listed fits, call no tool rather than inventing a name.",
        {"body_beat": {"type": "string", "enum": _BODY_BEATS,
                       "description": "the beat to perform"}},
        ["body_beat"]),
    "performance.mood_pose": (
        "A request to physically ACT OUT an emotion ('act embarrassed', 'look "
        "annoyed') — pick a mood from the enum, and if nothing listed fits, call "
        "no tool.",
        {"mood": {"type": "string", "enum": _MOOD_POSES,
                  "description": "the emotion to pose"}},
        ["mood"]),
    # The arg is "target", not "who": ActionSpec, the JSON-prose router prompt and
    # the regex classifier all say args.target, and the executor reads target
    # first. One arg name across all three routers — arg-name drift is the same
    # failure class as the tool_args/args bug documented below.
    "performance.impersonate": (
        "An explicit request to impersonate, imitate, or 'talk like' someone — a "
        "passing compliment about an impression is not one.",
        {"target": {**_STR, "description":
                    "who to imitate: 'speaker' for the person talking, "
                    "otherwise the name they said"}}, ["target"]),
    # game.* went LIVE 2026-08-13. These hints carry the negatives the guards in
    # action_router.game_request_refusal_reason enforce, because the model is now
    # the first thing standing between reminiscing about a game and starting one.
    "game.start": (
        "An explicit request to PLAY a verbal game now — Jeopardy, Trivia, I Spy, "
        "20 Questions or Word Association ('quiz me', 'how about a game', 'fire up "
        "trivia'). Never reminiscing ('we played trivia last night'), never an "
        "idiom ('he's playing games with my head'), and never 'what games do you "
        "have', which asks for the LIST. If they did not name a game, leave the "
        "argument empty rather than picking one for them.",
        {"game": {**_STR, "description":
                  "the game they named, as said; empty if they named none"}}, []),
    "game.stop": (
        "An explicit request to end the game that is running ('stop the game', "
        "\"I'm done with this\", 'wrap it up') — never a refusal ('don't stop "
        "now') and never narration about some other game ending.", {}, []),
    "game.answer": (
        "An answer/guess for the ACTIVE game (context shows active_game).",
        {"answer": _STR}, ["answer"]),
    "music.play": (
        "Play music: a song, artist, genre, vibe, or station.",
        {"music_query": {**_STR, "description": "what to play, as said"}}, ["music_query"]),
    "music.stop": ("Stop the music that is playing.", {}, []),
    "music.skip": ("Skip to the next track.", {}, []),
    "music.options": ("Asking what music is available.", {}, []),
    # Widened 2026-08-13 after a field failure: "What do you see me holding?" and
    # "I'm holding it right in front of you." both drew "I can't tell from here."
    # while the camera was working — the very next turn, "What do you see?",
    # returned "a colorful braided toy". The shadow collector picked
    # vision.describe_scene for the holding phrasing on that same turn, so the
    # ROUTING was right and the reply call was what declined. The old hint only
    # described the generic scene case, so a question about ONE object read as
    # something else.
    "vision.describe_scene": (
        "Asking what Rex can SEE right now — the room, or what someone is HOLDING, "
        "wearing, showing him or pointing at, or what an object is. Returns the "
        "live camera frame including objects in someone's hand, so call it rather "
        "than saying you cannot tell.", {}, []),
    "vision.snapshot": ("An explicit request to take a picture.", {}, []),
    # Offered ONLY when interaction's addressee hint says the line may not have
    # been aimed at Rex (intelligence/addressee.py) — never in a one-on-one room.
    "conversation.stay_quiet": (
        "The last line was side conversation between the humans, not addressed to "
        "Rex, and nothing is worth adding — stay quiet and keep listening.", {}, []),
    "time.query": ("Asking the current clock time.", {}, []),
    "date.query": ("Asking today's date/day (NOT holiday explanations).", {}, []),
    "weather.query": (
        "Asking about the weather, forecast, or temperature — outdoor OR "
        "indoor ('what temperature is it inside?' reads the onboard climate "
        "sensor).", {}, []),
    "status.capabilities": ("Asking what Rex can do.", {}, []),
    "status.uptime": ("Asking how long Rex has been running.", {}, []),
    "status.battery": (
        "Asking about Rex's OWN battery, charge level, or state of charge.",
        {}, [],
    ),
    # motion.turn/move/arc/come went LIVE 2026-08-13 (Phase 3, the last family).
    # Unlike every other migration the regex fast lane KEEPS the first claim
    # (docs/tool_router_scope.md §3), so these hints describe only what it misses,
    # and they state UNITS, because the executor reads a bare number.
    #
    # EVERY arg name below is now the exact key interaction._handle_router_motion_
    # action reads. The shadow-era schemas drifted on three of them and each failed
    # SILENTLY: `degrees` was read by nobody (the executor reads `deg`), so a
    # commanded angle became the default 90; `distance`+`unit` were read by nobody
    # (it reads `dist_m`), so a commanded distance became the default 0.30 m nudge;
    # and motion.arc's lone `direction` was read by nobody (it reads `ang_dir` and
    # `lin_dir`), so EVERY tool-routed arc would have curved forward-and-LEFT no
    # matter which way was asked. Worse than any of those, the move enum said
    # "backward" while the executor tests `== "back"` and otherwise falls through to
    # move_forward — "back up" would have driven him FORWARD, into the person who
    # just asked him to move away. Same drift class as performance.impersonate
    # who->target and memory.forget_specific statement->target, with wheels attached.
    "motion.turn": (
        "Rotate the drive base in place. Wheels only — a request to LOOK somewhere "
        "is not a turn, and neither is a figure of speech ('the meeting turned into "
        "a disaster').",
        # Phase 3 of docs/motion_route_tool_plan.md (§2.2): a full spin had no
        # encoding. "'around' means a 180" was the whole hint, so "spin around" and
        # "do a 360" had nowhere to land but a half turn — the executor and the
        # firmware have taken 360 all along (motion_controller.turn clamps at ±360),
        # only the schema never said so. Description change only; no executor change.
        {"direction": {"type": "string", "enum": ["left", "right", "around"],
                       "description":
                       "'around' means keep spinning past a quarter turn — a 180 by "
                       "default, or whatever deg says (use it with deg=360 for a "
                       "full spin)"},
         "deg": {**_NUM, "description":
                 "how far to rotate, in DEGREES (90 = a quarter turn, 180 = about "
                 "face, 360 = one full spin, the maximum); omit when they did not "
                 "say an amount"}},
        ["direction"]),
    "motion.move": (
        "Drive the base straight forward or backward on the floor.",
        {"direction": {"type": "string", "enum": ["forward", "back"]},
         "dist_m": {**_NUM, "description":
                    "how far, in METRES. Omit it unless they gave an amount — Rex "
                    "re-reads any distance they actually said out of their own "
                    "words, so never convert feet or inches yourself"}},
        ["direction"]),
    "motion.arc": (
        "Drive a brief curve toward one side — the base cannot strafe, so this is "
        "what 'scoot over to your right' / 'slide left' / 'swing left as you go' "
        "become.",
        {"ang_dir": {"type": "string", "enum": ["left", "right"],
                     "description": "which side to curve toward"},
         "lin_dir": {"type": "string", "enum": ["forward", "back"],
                     "description": "curve while driving forward or while backing up"},
         "small": {"type": "boolean",
                   "description": "true when they asked for a little / a bit"}},
        ["ang_dir"]),
    "motion.come": (
        "'Come here' / 'come closer' / 'roll over to me' — find the person speaking "
        "and drive to them. Never the idioms ('come on', 'come to think of it') and "
        "never someone else's invitation being retold.", {}, []),
    # motion.stop and motion.explore are catalog tools for the SHADOW only and are
    # deliberately absent from the live sets. Stop: docs/tool_router_scope.md 2.2 —
    # a stop that waits for a reply-call round trip is a stop that arrives late, and
    # the deterministic escape (interaction._errand_stop_demanded +
    # motion_controller.is_moving(), watched by the eager endpointer) already claims
    # it before any LLM sees the turn. Explore: an accepted invite seizes the floor
    # for minutes via the autonomous worker, and classify_explicit_exploration is
    # already a purpose-built "imperative addressed to Rex" test — the same thing the
    # motion gate had to be rebuilt into — so there is nothing for a tool to add yet.
    "motion.stop": ("Stop moving RIGHT NOW ('stop', 'halt' while driving).", {}, []),
    "motion.explore": ("An invitation to wander/explore the room.", {}, []),
    # motion.route (docs/motion_route_tool_plan.md) — the multi-step gap the regex
    # sequence parser leaves behind. Two callers share this ONE schema: the reply
    # call (organic path, gated on config.MOTION_ROUTE_ORGANIC_ENABLED) and the
    # focused rescue interpreter in intelligence/motion_route.py, which is why the
    # step properties are spelled out here rather than in the interpreter's prompt.
    #
    # Every step key is the key the executors READ (motion_sequence._issue and
    # interaction._handle_router_motion_action) — the Phase-3 arg-name drift class,
    # four instances of it, every one silent. Two deviations from the plan's §4.1
    # sketch, both deliberate and both load-bearing:
    #   * The magnitudes are POSITIVE and the direction is a separate enum word. The
    #     plan sketched signed numbers (+ = left, - = back); neither executor reads a
    #     sign (turn_left/move_back both take abs()), and the move enum "backward" —
    #     one letter off from the "back" the executor tests — falls through to
    #     FORWARD. action_router.route_tool_to_decisions still accepts a signed
    #     magnitude as a fallback when the direction word is missing, but it consumes
    #     the sign there and never forwards it.
    #   * No `target` field, ever (plan §9). Target-relative motion ("face the
    #     window") needs a bearing source; with no field for it the model cannot
    #     pretend it has one.
    "motion.route": (
        "A MULTI-STEP drive request: two or more movements the user wants driven in "
        "order, in ONE command ('go forward a bit, then turn around and come back', "
        "'back up, swing left, then roll forward two feet'). A SINGLE movement is "
        "motion_turn / motion_move / motion_arc — use those instead. Give every step "
        "an explicit direction word and a positive magnitude. Never for a figure of "
        "speech ('let's move on'), never for a route someone is RETELLING, never for "
        "a negated command, and never for a place or object he should drive to — he "
        "has no way to find one, so only geometry belongs here.",
        {"steps": {
            "type": "array", "minItems": 2, "maxItems": 6,
            "description": "the movements, in the order he should drive them",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["turn", "move", "arc"],
                           "description":
                           "turn = rotate in place, move = drive straight, "
                           "arc = a brief curve toward one side (the base cannot "
                           "strafe, so 'scoot over' is an arc)"},
                    "direction": {"type": "string",
                                  "enum": ["left", "right", "around",
                                           "forward", "back"],
                                  "description":
                                  "turn steps: left / right / around (an about-face). "
                                  "move steps: forward / back. Required on every turn "
                                  "and move step — 'back', never 'backward'"},
                    "deg": {**_NUM, "description":
                            "turn steps only: how far to rotate, in DEGREES, as a "
                            "POSITIVE number (90 = a quarter turn, 180 = an "
                            "about-face, 360 = one full spin). Omit for a plain "
                            "'turn left' with no stated amount"},
                    "dist_m": {**_NUM, "description":
                               "move steps only: how far to drive, in METRES, as a "
                               "POSITIVE number — convert what they said (a foot is "
                               "0.3, a 'bit' or a 'smidge' is about 0.2, a 'step' is "
                               "about 0.3). Omit for a plain 'move forward' with no "
                               "stated amount"},
                    "ang_dir": {"type": "string", "enum": ["left", "right"],
                                "description": "arc steps only: which side to curve toward"},
                    "lin_dir": {"type": "string", "enum": ["forward", "back"],
                                "description":
                                "arc steps only: curve while driving forward or while backing up"},
                    "small": {"type": "boolean",
                              "description": "arc steps only: true when they asked for a little / a bit"},
                    "pace": {"type": "string", "enum": ["slow", "normal"],
                             "description": "optional: 'slow' when they asked him to take it easy"},
                },
                "required": ["op"],
            },
        }},
        ["steps"]),
    # motion.face (2026-08-22). ZERO ARGS, deliberately: the requester comes from
    # voice ID, never from a model argument. The impersonation carve-out is the
    # precedent — the reply call once passed target='speaker', the previous turn's
    # argument, and Rex performed the wrong person. A bearing is worse: a model
    # asserting one it cannot observe would turn him at a wall while announcing that
    # he had turned to face you. Whose voice it was is the ONE thing the model is not
    # asked, because the deterministic layer already knows.
    "motion.face": (
        "'Turn to face me' / 'face me' / 'turn towards me' / 'point yourself at me' "
        "— rotate the drive base to point at whoever is speaking, then STOP. Not "
        "come here (that drives across the room to them), not a head or eye "
        "movement, and never a figure of speech ('face the music', 'face your "
        "fears', 'face me in chess').",
        {}, []),
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
    to the existing executor for that action.

    The tool arguments live on ``tool_args`` — NOT ``args``. ``args`` is
    BaseException's reserved attribute: assigning a dict to it silently stores
    ``tuple(dict)`` = a tuple of the KEYS (field 2026-08-03 18:00: web.search
    args became ``('query',)``, the executor's ``.get`` raised AttributeError,
    and the crash killed the speech loop — Rex went deaf until a manual
    shutdown). Every argument-less tool had masked the bug: ``()`` is falsy,
    so ``args or {}`` papered over it."""

    def __init__(self, action: str, tool_args: dict):
        super().__init__(action)
        self.action = str(action)
        self.tool_args = dict(tool_args or {})


# Phase 1 live set (docs/tool_router_scope.md): the intent-backed actions where
# every measured shipped-miss lived, all served by the existing
# _handle_classified_intent executor. Humor/character keep their working fast
# lanes and stay shadow-only for now.
_DEFAULT_LIVE_ACTIONS = (
    "time.query", "date.query", "weather.query",
    "status.capabilities", "status.uptime", "status.battery",
    "vision.describe_scene", "music.options",
    "system.sleep", "system.shutdown", "web.search",
    "event.cancel", "memory.query", "identity.who_is_speaking",
    "music.play", "music.stop", "music.skip", "vision.snapshot",
    "identity.name_correction", "memory.forget_person",
    "humor.tell_joke", "humor.roast", "humor.free_bit",
    "performance.dj_bit", "performance.body_beat", "performance.mood_pose",
    "performance.impersonate",
    "memory.forget_specific", "memory.recent_discard", "emotional.boundary",
    # conversation.stay_quiet (2026-09-05): live, but OPTIONAL — attached to the
    # reply call only when the addressee hint asks for it (see live_reply_tools).
    "conversation.stay_quiet",
    # Phase 2 games (2026-08-13): game.start is the win — command_parser was
    # the only thing that ever started a game and it is blind to "quiz me",
    # "game time", "fire up trivia", "deal me in". game.stop is live for the
    # no-game-running case; mid-game the deterministic escape keeps the claim.
    # game.answer is NOT live and must not be (scope doc 2.2).
    "game.start", "game.stop",
    # Phase 3 motion (2026-08-13) — the last family, and the only one where the regex
    # KEEPS the first claim: motion.* is NOT in action_router.TOOL_ROUTER_OWNED_ACTIONS,
    # so a >=0.95 classifier match still executes immediately at today's latency and
    # the tool governs only what it missed (docs/tool_router_scope.md §3). Measured
    # misses on this checkout, all currently answered as conversation: "rotate ninety
    # degrees", "rotate 90 degrees", "back yourself up a bit", "scoot a little
    # closer", "get closer", "back it up", "back away", "drive up here", "go straight",
    # "face me", "swivel left", "hang a left", "veer right", "why don't you scoot
    # forward", "scootch to your right".
    # motion.stop and motion.explore are ABSENT ON PURPOSE — see _TOOL_DEFS above.
    "motion.turn", "motion.move", "motion.arc", "motion.come",
    # motion.face ships LIVE rather than phase-gated like motion.route below: its
    # worst case is ONE bounded turn onto a sensed bearing, and the status quo it
    # replaces is worse than that — "turn to face me" reaching motion.turn, whose
    # schema forces left/right/around, so he guessed a side and then suppressed
    # realign for 45 s.
    "motion.face",
    # motion.route is listed so this tuple stays the honest catalog of the reply
    # call's motion surface, but it is PHASE-GATED in live_actions() below: the
    # organic path is the half of docs/motion_route_tool_plan.md that can invent a
    # route out of banter, and it stays off until live decoy numbers say otherwise.
    "motion.route",
)


def live_actions() -> "set[str]":
    if not bool(getattr(config, "TOOL_ROUTER_LIVE_ENABLED", True)):
        return set()
    live = {str(a) for a in getattr(config, "TOOL_ROUTER_LIVE_ACTIONS",
                                   _DEFAULT_LIVE_ACTIONS)}
    if not motion_route_organic_enabled():
        # Phase 2 of docs/motion_route_tool_plan.md. Dropped HERE rather than left
        # out of the tuple so that config.TOOL_ROUTER_LIVE_ACTIONS still reads as the
        # full surface and one flag — not two lists that can disagree — decides.
        live.discard("motion.route")
    return live


def motion_route_organic_enabled() -> bool:
    """Whether motion.route may be chosen by the REPLY call (plan §7 Phase 2).

    Separate from MOTION_ROUTE_ENABLED, which governs the rescue path: that one only
    ever fires on a turn the sequence classifier already identified as an attempted
    route and today answers with a flat refusal, while this one puts a six-step drive
    plan on the same conversational call that produced the prose-wins impersonation
    record (docs/tool_router_scope.md, Phase 2 carve-out)."""
    return (bool(getattr(config, "MOTION_ROUTE_ENABLED", True))
            and bool(getattr(config, "MOTION_ROUTE_ORGANIC_ENABLED", False)))


def tool_schema_for(action: str) -> "dict | None":
    """The single tool schema for one action key, or None when it has no spec.

    Exists so intelligence/motion_route.py's focused interpreter call can hand the
    model the EXACT same motion.route schema the reply call sees — one definition,
    so the two paths cannot drift into disagreeing about arg names."""
    for schema in tool_schemas():
        if _NAME_TO_KEY.get(schema["function"]["name"]) == action:
            return schema
    return None


# Live tools that are attached to a reply call only when the caller asks for
# them (situational, not every turn). conversation.stay_quiet must never be on
# offer in a one-on-one conversation — see intelligence/addressee.py.
_OPTIONAL_LIVE = frozenset({"conversation.stay_quiet"})


def live_reply_tools(optional: "set[str] | None" = None) -> "list[dict] | None":
    """Tool schemas for the LIVE subset only, or None when cutover is off.
    Attached to the lean reply call — routing rides the call that already
    happens, so a live tool costs zero extra LLM round-trips. Tools in
    _OPTIONAL_LIVE are included only when named in `optional`."""
    live = live_actions()
    if not live:
        return None
    wanted = set(optional or ())
    tools = [t for t in tool_schemas()
             if (_NAME_TO_KEY.get(t["function"]["name"]) in live
                 and (_NAME_TO_KEY.get(t["function"]["name"]) not in _OPTIONAL_LIVE
                      or _NAME_TO_KEY.get(t["function"]["name"]) in wanted))]
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
