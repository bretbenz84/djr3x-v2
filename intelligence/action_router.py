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
from intelligence.person_memory_targets import references_person_memory_target
from intelligence import performance_plan, rex_preferences
from memory.name_validation import normalize_person_name
from openai import OpenAI


_log = logging.getLogger(__name__)
from intelligence import connectivity as _connectivity
_client = _connectivity.guard_client(OpenAI(api_key=apikeys.OPENAI_API_KEY), "action_router")


@dataclass(frozen=True)
class ActionSpec:
    """Stable action-router catalog entry.

    The router should classify recurring classes of intent, not every tiny
    conversational edge case. New keys should stay broad enough to map onto a
    durable handler or performance plan.
    """

    key: str
    category: str
    description: str
    executable: bool = False


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "conversation.reply",
        "conversation",
        "Normal conversational response; no tool, feature, or special performance action should run.",
    ),
    ActionSpec(
        "conversation.repair",
        "conversation",
        "User corrects Rex, says he misunderstood, or asks him to try again. Use for repair, not ordinary disagreement.",
        executable=True,
    ),
    ActionSpec(
        "memory.query",
        "memory",
        "User asks what Rex remembers or knows about a person, relationship, or themselves. Not for general topic knowledge.",
        executable=True,
    ),
    ActionSpec(
        "memory.forget_specific",
        "memory",
        "User asks Rex to forget/delete a specific remembered detail or topic.",
        executable=True,
    ),
    ActionSpec(
        "memory.recent_discard",
        "memory",
        "User asks Rex not to store or remember the immediately recent thing they said.",
        executable=True,
    ),
    ActionSpec(
        "memory.forget_person",
        "memory",
        "User asks Rex to forget a whole person, themselves, or everyone. Always requires confirmation.",
    ),
    ActionSpec(
        "event.cancel",
        "memory",
        "User says a remembered plan/event is canceled, stale, or no longer happening.",
        executable=True,
    ),
    ActionSpec(
        "emotional.boundary",
        "boundary",
        "User asks not to discuss a sensitive topic anymore or rejects an emotional check-in.",
        executable=True,
    ),
    ActionSpec(
        "identity.who_is_speaking",
        "identity",
        "User asks who they are, who is speaking, or whether Rex recognizes them.",
        executable=True,
    ),
    ActionSpec(
        "identity.name_correction",
        "identity",
        "User corrects the current speaker identity/name, such as 'that's not Bret', 'I'm Daniel', or 'call me X'.",
        executable=True,
    ),
    ActionSpec(
        "identity.introduce_person",
        "identity",
        "User introduces a person or relationship, such as 'this is my dad Jeff'. Not for professions, jobs, hobbies, or other personal facts.",
    ),
    ActionSpec(
        "humor.tell_joke",
        "humor",
        "User explicitly asks for a joke, pun, one-liner, or canned funny line.",
        executable=True,
    ),
    ActionSpec(
        "humor.roast",
        "humor",
        "User explicitly asks Rex to roast or tease someone, the speaker, the room, or a named target. Put the target in args.target.",
        executable=True,
    ),
    ActionSpec(
        "humor.free_bit",
        "humor",
        "User asks Rex to be funny, riff, do a bit, or make them laugh without a specific joke or roast target.",
        executable=True,
    ),
    ActionSpec(
        "performance.dj_bit",
        "performance",
        "User asks Rex for DJ/cantina patter, hype, an announcement, or a station-break line without requesting actual music playback.",
        executable=True,
    ),
    ActionSpec(
        "performance.body_beat",
        "performance",
        "User asks Rex to perform a physical gesture, pose, dance, look, tilt, peek, or other embodied beat.",
        executable=True,
    ),
    ActionSpec(
        "performance.mood_pose",
        "performance",
        "User asks Rex to physically act or look like an emotion, such as embarrassed, annoyed, proud, suspicious, or thinking.",
        executable=True,
    ),
    ActionSpec(
        "performance.impersonate",
        "performance",
        "User explicitly asks Rex to do an impersonation/impression of themselves or "
        "a named person, or to copy/imitate/'talk like' someone's voice. Put who to "
        "imitate in args.target: use 'speaker' for the user themselves, otherwise the "
        "provided name.",
        executable=True,
    ),
    ActionSpec(
        "character.preference_query",
        "character",
        "User asks Rex about Rex's own likes, dislikes, favorites, beliefs, taste, or preference between options.",
        executable=True,
    ),
    ActionSpec(
        "game.start",
        "game",
        "User asks to start/play a game.",
        executable=True,
    ),
    ActionSpec(
        "game.stop",
        "game",
        "User asks to stop/quit/end the current game.",
        executable=True,
    ),
    ActionSpec(
        "game.answer",
        "game",
        "User is answering or choosing inside an active game.",
        executable=True,
    ),
    ActionSpec(
        "music.play",
        "music",
        "User asks Rex to play music, a song, artist, genre, vibe, or station.",
        executable=True,
    ),
    ActionSpec(
        "music.stop",
        "music",
        "User asks Rex to stop/pause music.",
        executable=True,
    ),
    ActionSpec(
        "music.skip",
        "music",
        "User asks Rex to skip the current track.",
        executable=True,
    ),
    ActionSpec(
        "music.options",
        "music",
        "User asks what music, genres, stations, or songs Rex can play.",
        executable=True,
    ),
    ActionSpec(
        "vision.describe_scene",
        "vision",
        "User asks what Rex sees or asks Rex to look/inspect something.",
        executable=True,
    ),
    ActionSpec(
        "vision.snapshot",
        "vision",
        "User asks Rex to remember, save, or keep in mind what he currently sees. Privacy-sensitive; do not execute without confirmation.",
    ),
    ActionSpec(
        "time.query",
        "world",
        "User asks for the current clock time.",
        executable=True,
    ),
    ActionSpec(
        "date.query",
        "world",
        "User asks for today's date or day of week.",
        executable=True,
    ),
    ActionSpec(
        "weather.query",
        "world",
        "User asks for weather.",
        executable=True,
    ),
    ActionSpec(
        "status.capabilities",
        "status",
        "User asks what Rex can do.",
        executable=True,
    ),
    ActionSpec(
        "status.uptime",
        "status",
        "User asks how long Rex has been running/awake.",
        executable=True,
    ),
    ActionSpec(
        "status.battery",
        "status",
        "User asks about Rex's OWN battery / charge level / state of charge / "
        "whether he's charging. Not for other devices' batteries.",
        executable=True,
    ),
    ActionSpec(
        "motion.turn",
        "motion",
        "User asks Rex to physically rotate the drive base in place — turn/spin/pivot left or right, or turn around. Not for head/look gestures.",
        executable=True,
    ),
    ActionSpec(
        "motion.move",
        "motion",
        "User asks Rex to physically drive the base forward or backward — move forward, back up, reverse, roll ahead. Not for head/look gestures.",
        executable=True,
    ),
    ActionSpec(
        "motion.arc",
        "motion",
        "User asks Rex to physically move SIDEWAYS or diagonally with the drive base — "
        "'move to your left', 'go right', 'scoot over', 'move forward and to the left'. "
        "The base can't strafe, so it drives a brief curve toward that side. "
        "Args: ang_dir ('left'/'right'), lin_dir ('forward'/'back', default forward). "
        "Not for head/look gestures.",
        executable=True,
    ),
    ActionSpec(
        "motion.come",
        "motion",
        "User asks Rex to physically come to them / roll over here / come closer (drive base).",
        executable=True,
    ),
    ActionSpec(
        "motion.stop",
        "motion",
        "User asks Rex to stop moving / halt / freeze the drive base while it is driving.",
        executable=True,
    ),
    ActionSpec(
        "motion.explore",
        "motion",
        "User INVITES Rex to autonomously explore / look around / wander the room on "
        "his own — 'feel free to explore', 'look around a little', 'wander around', "
        "'check the place out', 'make yourself at home'. Rex drives around and takes "
        "in the room himself. NOT a directed 'look left and tell me what you see' "
        "(that's a vision query), NOT a search errand ('look around for my keys').",
        executable=True,
    ),
    ActionSpec(
        "web.search",
        "web",
        "User asks about news, current events, or anything needing live "
        "up-to-date information — wars, elections, scores, prices, launches, "
        "'what's going on with X', follow-ups on a news story Rex raised.",
        executable=True,
    ),
    ActionSpec(
        "system.sleep",
        "system",
        "User asks Rex to sleep, wake, quiet down, or mute.",
        executable=True,
    ),
    ActionSpec(
        "system.shutdown",
        "system",
        "User asks Rex to fully power down — 'shut down', 'power off', 'turn "
        "yourself off' — including polite requests ('can you shut down, "
        "please?'). NOT for shutting down something else (music, a server).",
        executable=True,
    ),
)

ACTION_CATALOG: dict[str, str] = {
    spec.key: spec.description for spec in ACTION_SPECS
}
ACTION_CATEGORIES: dict[str, str] = {
    spec.key: spec.category for spec in ACTION_SPECS
}
PERFORMANCE_ACTIONS = {
    spec.key
    for spec in ACTION_SPECS
    if spec.category in {"humor", "performance"}
}
_VALID_ACTIONS = set(ACTION_CATALOG)
EXECUTABLE_ACTIONS = {
    spec.key for spec in ACTION_SPECS if spec.executable
}

_SYSTEM_PROMPT = """You are DJ-R3X's action router.
Choose the single best action for the user's latest utterance using the catalog.
Return JSON only. Do not write a conversational reply.

Rules:
- Prefer the user's actual intent over keyword matching.
- Pick exactly one stable action key. Do not invent one-off actions for narrow
  conversational snafus; use conversation.reply or conversation.repair unless a
  catalog action clearly fits.
- If context.pending.pending_question exists, treat short fragments as answers
  to Rex's pending question, not as new feature commands.
- If the pending question key is favorite_music, a bare genre/artist/style like
  "classical music" is a preference answer. Use conversation.reply unless the
  user explicitly asks to play/put on/start music.
- Only use memory.forget_specific when the utterance explicitly asks to forget,
  delete, remove, erase, wipe, or clear a remembered thing. Preference statements
  like "I like Disneyland" are conversation.reply and may be learned as interests.
- Use memory.recent_discard for "forget I said that", "don't remember that",
  "don't store that", or "don't save that" when the scope is the immediately
  recent utterance rather than a named stored fact.
- If the utterance asks what you remember or know about someone, use memory.query.
  If it asks what Rex generally knows about a topic, franchise, place, hobby,
  object, or field, use conversation.reply so the main LLM can answer.
- If the utterance explicitly says a remembered plan is canceled, stale, over,
  or no longer happening, use event.cancel. Status updates like "we're still
  driving home" are normal conversation.reply.
- For event.cancel, put the plan/topic being canceled in args.event_hint when possible.
- Only use emotional.boundary when the user explicitly asks not to talk about,
  ask about, mention, or bring up a topic. A bare health/sad topic like "back pain"
  is conversation.reply unless the user says not to discuss it.
- Use conversation.repair when the user corrects Rex, says Rex misunderstood, or
  asks Rex to try that again. Do not use it for ordinary topic disagreement.
- Use identity.name_correction when the user corrects who Rex thinks is speaking
  or what to call the current speaker, e.g. "that's not Bret, I'm Daniel" or
  "call me JT". Put the corrected name in args.name when present. Use
  conversation.repair if the correction has no identity/name content. Do not use
  identity.name_correction for plan/status retractions like "that's not
  happening anymore".
- Use humor.tell_joke only for explicit joke/pun/one-liner requests like
  "tell me a joke"; do not treat general mentions of jokes as a joke request.
- Use humor.roast only for explicit roast/tease requests. Put the roast target in
  args.target, e.g. "speaker", "room", or a provided name.
- Use humor.free_bit for broader requests like "say something funny", "do a bit",
  or "make me laugh" when no specific joke format or roast target is requested.
- Use performance.dj_bit for DJ patter, hype lines, cantina banter, or station
  breaks. Use music.play only when the user asks to actually play audio.
- Use performance.body_beat for explicit physical pose/gesture/dance/look/tilt
  requests. Put one of these exact names in args.body_beat:
  agreement_nod, anger_flash, disagreement_shake, disbelief_stare,
  disgust_recoil, giddy_wiggle, happy_bounce, sad_droop, surprise_pop,
  suspicious_glance, proud_dj_pose, offended_recoil, thinking_tilt,
  dramatic_visor_peek, tiny_victory_dance, eye_roll, double_take, mic_drop,
  spit_take. Do not use it for ordinary
  "look at this" vision requests.
- Use performance.mood_pose for emotion-driven physical acting requests such as
  "act embarrassed", "look annoyed", or "look proud". Put one of these exact
  mood names in args.mood: agreement, disagreement, disbelief, disgusted,
  embarrassed, annoyed, angry, proud, suspicious, thinking, happy, giddy,
  sad, surprised, offended.
- Use performance.impersonate when the user explicitly asks Rex to do an
  impersonation/impression of someone, to copy/imitate someone's voice, or to
  "talk/sound like" a person: "do an impersonation of me", "impersonate Jimmy
  Carter", "can you do my voice", "talk like Patrick Stewart". Put who to imitate
  in args.target — use "speaker" for the user themselves, otherwise the provided
  name. Do NOT use it for a passing compliment about an impression ("that was a
  good impression").
- Use character.preference_query when the user asks Rex about Rex's own taste,
  favorites, beliefs, or preferences: "do you like X?", "do you hate X?",
  "how do you feel about X?", "what's your favorite X?", "do you prefer X or Y?".
  Put args.topic when there is one, args.verb for like/hate/dislike/prefer
  questions, args.mode="favorite" for favorites, and args.options for X-or-Y
  comparisons. Do not use it when the human states their own preference.
- Use vision.snapshot only when the user asks Rex/you to remember, save, store,
  or keep in mind what Rex currently sees, such as "remember what you see" or
  "take a look and keep that in mind". Set requires_confirmation=true. Do not
  use it for the user's own first-person plans like "I want to take a picture"
  or "I'm going to save this view"; those are conversation.reply. Do not use it
  for ordinary "what do you see?" questions; those are vision.describe_scene.
- If a game is active and the utterance asks to stop, quit, end, or stop playing, use game.stop.
- If music is active and the utterance asks to stop, pause, or stop playing music, use music.stop.
- If the utterance asks for the clock time, use time.query.
- If the utterance asks for today's date or day of week, use date.query.
- If a game is active and the utterance is a short fragment that is not clearly a stop/control command, prefer game.answer over identity or general actions. If no game is active, do not use game.answer.
- Do not use identity.introduce_person for first-person facts like "I'm an IT systems administrator"; those are normal conversation.reply turns so memory extraction can learn them.
- Do not use identity.introduce_person for pronoun-only fragments like "me and
  you", "you and me", "us", or "me"; treat them as answers to the current
  conversation unless a real introduced name is present.
- If the utterance is normal chat, use conversation.reply.
- Use requires_confirmation=true when an action is broad/destructive or ambiguous. A specific forget request with a clear target does not require confirmation.
- Confidence is 0.0 to 1.0.
"""

_MUSIC_PLAY_REQUEST_RE = re.compile(
    r"(?:^|[.!?]\s+)\s*(?:please\s+)?"
    r"(?:play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue|turn\s+on)\b|"
    r"\b(?:can|could|would)\s+you\s+(?:please\s+)?(?:play|put\s+on|queue|cue)\b|"
    r"\b(?:play|put\s+on|throw\s+on|spin|queue|cue)\s+(?:me|us)\b",
    re.IGNORECASE,
)
_MUSIC_STOP_REQUEST_RE = re.compile(
    r"\b(?:stop|pause|turn\s+off)\b.{0,30}\b(?:music|song|track|playlist|audio|playing)\b|"
    r"^\s*(?:stop|pause)\s+(?:the\s+)?(?:music|song|track|playlist)\b",
    re.IGNORECASE,
)
_MUSIC_SKIP_REQUEST_RE = re.compile(
    r"\b(?:skip|next)\b.{0,20}\b(?:this|that|song|track|music|one)\b|^\s*skip\s*$",
    re.IGNORECASE,
)
_MUSIC_OPTIONS_REQUEST_RE = re.compile(
    r"\b(?:what|which)\b.{0,40}\b(?:music|songs|tracks|playlists|stations)\b|"
    r"\b(?:music|songs|tracks|playlists|stations)\b.{0,40}\b(?:available|options|have)\b",
    re.IGNORECASE,
)
_GAME_START_REQUEST_RE = re.compile(
    r"\b(?:play|start|run|do)\b.{0,40}\b(?:game|trivia|jeopardy|i\s+spy|"
    r"20\s+questions|twenty\s+questions|word\s+association)\b|"
    r"\b(?:let'?s|lets|can\s+we|could\s+we|i\s+want\s+to)\s+"
    r"(?:play|start|do)\b.{0,40}\b(?:game|trivia|jeopardy|i\s+spy|"
    r"20\s+questions|twenty\s+questions|word\s+association)\b",
    re.IGNORECASE,
)
_GAME_STOP_REQUEST_RE = re.compile(
    r"^\s*(?:stop|quit|end)(?:\s+(?:the\s+)?game)?\s*$|"
    r"\b(?:stop|quit|end)\b.{0,30}\b(?:game|trivia|jeopardy|i\s+spy|"
    r"20\s+questions|twenty\s+questions|word\s+association)\b",
    re.IGNORECASE,
)
_TIME_QUERY_RE = re.compile(
    r"\b(?:what(?:'s| is)?|tell me|give me|do you know)\b.{0,30}\b(?:time|clock)\b|"
    r"\b(?:time|clock)\b.{0,20}\b(?:now|is it)\b",
    re.IGNORECASE,
)
_DATE_QUERY_RE = re.compile(
    # Keep in sync with intent_classifier._DATE_QUERY_RE: explicit date / day-of-week
    # questions only, NOT "what are you doing today" / "what are you up to today".
    r"\bwhat(?:'s| is)?\s+(?:the\s+|today'?s\s+|current\s+|exact\s+)*date\b|"
    r"\bwhat(?:'s| is)?\s+the\s+weekday\b|"
    r"\bwhat\s+weekday\s+is\s+it\b|"
    r"\bwhat\s+day\s+(?:of\s+the\s+week\s+)?(?:is\s+it|is\s+today|are\s+we\s+on)\b|"
    r"\bwhat(?:'s| is)?\s+(?:the\s+)?day\s+of\s+the\s+week\b|"
    r"\b(?:tell me|give me|do you know)\s+(?:the\s+|today'?s\s+)?(?:date|weekday|day\s+of\s+the\s+week)\b",
    re.IGNORECASE,
)
_WEATHER_QUERY_RE = re.compile(
    r"\b(?:what(?:'s| is)|tell me|give me|do you know)\b.{0,35}\b"
    r"(?:weather|temperature|forecast|raining|hot|cold|outside)\b|"
    r"\b(?:weather|temperature)\s+(?:forecast|outside)\b|"
    # "What temperature is it (inside)?" — 'what' directly followed by
    # 'temperature' missed the first alternation (field 2026-08-01: the indoor
    # BME280 branch was never reached; Rex claimed he couldn't read the room).
    r"\bwhat\s+temperature\b|"
    r"\bhow\s+(?:hot|cold|warm|humid|muggy)\s+is\s+it\b|"
    r"\bis\s+it\s+(?:raining|hot|cold)\b",
    re.IGNORECASE,
)
_CAPABILITIES_QUERY_RE = re.compile(
    r"\b(?:what can you do|what are you capable of|capabilities|"
    r"what do you do|what can i ask you|what should i ask you|commands)\b",
    re.IGNORECASE,
)
_UPTIME_QUERY_RE = re.compile(
    r"\b(?:how long have you been|uptime|been running|been awake|when did you start)\b",
    re.IGNORECASE,
)
# Evidence that an utterance is a genuine "use your camera" question. Kept
# deliberately broader than the original (which missed live phrasings like
# "can you see what I'm holding" and "look at my telescope" — both blocked as
# missing_vision_query_evidence in real runs and answered with hallucinated
# "I see it" text). Conversational idioms ("see what I mean", "look at the
# bright side") are excluded by _VISION_IDIOM_RE below, checked first.
_VISION_DESCRIBE_RE = re.compile(
    r"\b(?:"
    r"what (?:do|can|did) you see"
    r"|what(?:'s| is| are) (?:in front of you|on (?:the|your) camera|that|this|these|those)"
    r"|(?:do|can|did) you see"
    r"|what am i (?:holding|wearing|showing|pointing)"
    r"|what (?:i'm|i am) (?:holding|wearing|showing|pointing)"
    r"|what (?:is|are) (?:he|she|they) (?:holding|wearing)"
    r"|what(?:'s| is) (?:in|on) my hand"
    r"|look around"
    r"|look at (?:my|this|that|the|these|those|him|her|them|me|us|it|what)"
    r"|take a look(?: at)?"
    r"|check (?:this|that|it) out"
    r"|describe (?:the |this |my )?(?:room|scene|view|place)"
    r"|(?:describe|tell me) what you (?:see|can see|are seeing)"
    r"|use your (?:camera|eyes)"
    r")\b",
    re.IGNORECASE,
)
# Figure-of-speech guard: these contain "see"/"look at" but are conversation,
# not camera requests. Mirrors the stale-event-cancel idiom guard pattern.
_VISION_IDIOM_RE = re.compile(
    r"\b(?:"
    r"(?:do you |can you |you )?see (?:what i mean|my point|your point|the point|why|how it goes)"
    r"|look at (?:the bright side|it this way|the big picture|the time|you go)"
    r"|we(?:'ll)? see"
    r"|i see\b"
    r"|see you (?:later|soon|around|tomorrow)"
    r")",
    re.IGNORECASE,
)
# Physical gaze imperatives ("look to your right", "look at this") — evidence
# for vision.directed_look (the servo head-turn + directed camera analysis).
_DIRECTED_LOOK_RE = re.compile(
    r"\b(?:"
    r"look (?:to (?:the |your )?)?(?:left|right|up|down|behind(?: you)?|ahead|forward|"
    r"straight(?: ahead)?|center|centre|around|over (?:here|there))"
    r"|look (?:at )?(?:this|that|here|there)"
    r"|look at (?:my|the|his|her|their|these|those|him|her|them|me|us|it)\b"
    r"|look for\b"
    r"|look the other way"
    r"|turn (?:your head|around)"
    r")",
    re.IGNORECASE,
)
_LOOK_IDIOM_RE = re.compile(
    r"\b(?:"
    r"look at (?:the bright side|it this way|the big picture|the time|you go)"
    r"|look (?:sharp|alive|out)\b"
    r"|looking (?:good|sharp|forward)"
    r")",
    re.IGNORECASE,
)


def has_vision_query_evidence(text: str) -> bool:
    """Deterministic check that an utterance is a real camera question.

    Single source of truth shared by the central evidence policy below and the
    dialogue-act breakout in interaction.py, so a vision query passes or fails
    the same bar everywhere.
    """
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    if _VISION_IDIOM_RE.search(cleaned) and not _VISION_DESCRIBE_RE.search(
        _VISION_IDIOM_RE.sub(" ", cleaned)
    ):
        return False
    return bool(_VISION_DESCRIBE_RE.search(cleaned))


def has_directed_look_evidence(text: str) -> bool:
    """Deterministic check that an utterance is a physical look/gaze command."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    if _LOOK_IDIOM_RE.search(cleaned) and not _DIRECTED_LOOK_RE.search(
        _LOOK_IDIOM_RE.sub(" ", cleaned)
    ):
        return False
    return bool(_DIRECTED_LOOK_RE.search(cleaned))
_WHO_SPEAKING_RE = re.compile(
    r"\b(?:who am i|who is speaking|who'?s speaking|do you recognize me|"
    r"do you know who i am|what(?:'s| is) my name|do you know my name|"
    r"can you see me|who do you think i am)\b",
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
_RECENT_DISCARD_REQUEST_RE = re.compile(
    r"\b("
    r"forget|don'?t\s+remember|do\s+not\s+remember|don'?t\s+store|"
    r"do\s+not\s+store|don'?t\s+save|do\s+not\s+save|discard"
    r")\b.{0,80}\b("
    r"that|this|it|what\s+i\s+(?:just\s+)?said|i\s+(?:just\s+)?said\s+that"
    r")\b",
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
_REPAIR_REQUEST_RE = re.compile(
    r"\b(?:you\s+(?:misheard|misunderstood|got\s+that\s+wrong|got\s+it\s+wrong)|"
    r"that'?s\s+(?:wrong|incorrect|not\s+what\s+i\s+said)|"
    r"no\s*,?\s+that'?s\s+wrong|"
    r"no\s*,?\s+i\s+(?:said|meant)|"
    r"actually\s*,?\s+i\s+(?:said|meant)|"
    r"try\s+again)\b",
    re.IGNORECASE,
)
_NAME_CORRECTION_REQUEST_RE = re.compile(
    r"\b(?:call\s+me|rename\s+me(?:\s+to)?|my\s+name\s+is|"
    r"you\s+(?:got|have)\s+my\s+name\s+wrong|"
    r"that['’]?s\s+not\s+(?:my\s+name|[A-Za-z][A-Za-z' -]{1,60}))\b",
    re.IGNORECASE,
)
_NAME_FROM_TEXT_RE = re.compile(
    r"\b(?:call\s+me|rename\s+me(?:\s+to)?|my\s+name\s+is|"
    r"i\s+am|i['’]?m|im)\s+"
    r"(?P<name>[A-Za-z][A-Za-z' -]{0,60})",
    re.IGNORECASE,
)
_TOPIC_KNOWLEDGE_QUERY_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+know|do\s+you\s+know\s+anything|"
    r"tell\s+me|explain)\s+(?:about\s+)?(?P<topic>[^?.,!;]{3,100})",
    re.IGNORECASE,
)
_NAMED_DAY_EXPLANATION_RE = re.compile(
    r"\b(?:what(?:'s| is)?|tell me about|explain|describe)\s+"
    r"(?:the\s+)?(?:holiday\s+(?:called|named)\s+)?"
    r"(?!(?:today|today's|todays|date|the\s+date|day|weekday|day\s+of\s+week)\b)"
    r"(?:[a-z0-9][a-z0-9'’.-]*\s+){0,6}day\b",
    re.IGNORECASE,
)
_EVENT_CANCEL_OR_STALE_RE = re.compile(
    r"\b("
    r"cancel(?:ed|led|s|ing)?|called?\s+off|not\s+happening|"
    r"no\s+longer\s+happening|won['’]?t\s+happen|will\s+not\s+happen|"
    r"can['’]?t\s+make\s+it|cannot\s+make\s+it|not\s+going|"
    r"no\s+longer|not\b.{0,40}\banymore|instead\s+of|"
    r"scrap(?:ped|ping)?|ditch(?:ed|ing)?|postpon(?:e|ed|ing)|"
    r"reschedul(?:e|ed|ing)|already\s+happened|already\s+passed|"
    r"is\s+over|it['’]?s\s+over|was\s+over|ended|finished|wrapped\s+up"
    r")\b",
    re.IGNORECASE,
)
_EVENT_CONTINUATION_STATUS_RE = re.compile(
    r"\b("
    r"still|currently|right\s+now|on\s+(?:my|our|the)\s+way|"
    r"heading|headed|driving|riding|walking|flying|going"
    r")\b",
    re.IGNORECASE,
)
_PRONOUN_ONLY_INTRO_RE = re.compile(
    r"^\s*(?:me|you|us|me\s+and\s+you|you\s+and\s+me|me\s*&\s*you|"
    r"you\s*&\s*me|between\s+me\s+and\s+you|between\s+you\s+and\s+me)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_NAMED_PERSON_FACT_STATEMENT_RE = re.compile(
    r"^\s*[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}\s+"
    r"(?:is|has|works|likes|loves|hates|prefers|plays|collects)\b",
)
_NAMED_RELATION_INTRO_RE = re.compile(
    r"\bis\s+my\s+(?:"
    r"best\s+friend|partner|spouse|wife|husband|girlfriend|boyfriend|"
    r"fianc[eé]e?|father|dad|mother|mom|parent|son|daughter|child|"
    r"brother|sister|sibling|friend|boss|manager|supervisor|employee|"
    r"coworker|co[-\s]?worker|colleague|roommate|neighbor|neighbour"
    r")\b",
    re.IGNORECASE,
)
_RELATIONSHIP_SCORE_QUERY_RE = re.compile(
    r"\b("
    r"friendship\s+score|relationship\s+score|our\s+score|"
    r"score\s+between\s+(?:me\s+and\s+you|you\s+and\s+me)|"
    r"(?:me\s+and\s+you|you\s+and\s+me).{0,40}\bscore"
    r")\b",
    re.IGNORECASE,
)
_THATS_NOT_NAME_RE = re.compile(
    r"\bthat['’]?s\s+not\s+(?!my\s+name\b)(?P<name>[A-Za-z][A-Za-z' -]{0,60})",
    re.IGNORECASE,
)
_NON_NAME_THATS_NOT_TOKENS = {
    "any",
    "anymore",
    "bad",
    "case",
    "correct",
    "doing",
    "fine",
    "going",
    "good",
    "happen",
    "happening",
    "it",
    "more",
    "point",
    "right",
    "that",
    "thing",
    "this",
    "true",
    "what",
}
_TELL_JOKE_RE = re.compile(
    r"\b(?:tell|give|hit)\s+(?:me|us|the room)?\s*(?:with\s+)?"
    r"(?:a|another|one)?\s*(?:joke|pun|one[- ]liner)\b|"
    r"\bcrack\s+(?:me|us)?\s*(?:a|another)?\s*(?:joke|pun)\b|"
    r"\bgot\s+(?:any|a)\s+(?:jokes?|puns?)\b",
    re.IGNORECASE,
)
_ROAST_REQUEST_RE = re.compile(
    r"\b(?:roast|tease|mock|trash\s*talk)\s+"
    r"(?P<target>me|us|the room|this room|yourself|him|her|them|"
    r"[a-z][a-z .'-]{0,40})\b|"
    r"\bmake\s+fun\s+of\s+"
    r"(?P<target2>me|us|the room|this room|yourself|him|her|them|"
    r"[a-z][a-z .'-]{0,40})\b|"
    r"\bgive\s+(?:me|us)\s+(?:a\s+)?roast\b|"
    r"\bhit\s+(?:me|us)\s+with\s+(?:a\s+)?roast\b",
    re.IGNORECASE,
)
_FREE_HUMOR_RE = re.compile(
    r"\b(?:say\s+something\s+(?:funny|hilarious|amusing)|"
    r"make\s+(?:me|us)\s+laugh|"
    r"crack\s+(?:me|us)\s+up|"
    r"do\s+(?:a\s+|your\s+)?(?:bit|riff)|"
    r"riff\s+(?:for\s+)?(?:me|us)?|"
    r"be\s+funny)\b",
    re.IGNORECASE,
)
_DJ_BIT_RE = re.compile(
    r"\b(?:do|give|hit|drop)\s+(?:me|us|the\s+room)?\s*(?:with\s+)?"
    r"(?:your\s+)?(?:dj\s+thing|dj\s+bit|dj\s+riff|cantina\s+patter|"
    r"station[- ]break|hype\s+line|announcement)\b|"
    r"\bhype\s+(?:me|us|the\s+room)\s+up\b|"
    r"\bhype\s+the\s+room\b|"
    r"\bmake\s+(?:an|a)\s+announcement\b|"
    r"\bgive\s+(?:me|us)\s+(?:some\s+)?cantina\s+patter\b",
    re.IGNORECASE,
)
# Impersonation — deterministic, so an explicit "impersonate me" beats BOTH the
# dialogue-act answer-binding (which frames the turn as a reply to Rex's last
# question) and a hesitant LLM route. Each pattern captures the target. Kept to
# unambiguous verb shapes; softer phrasings ("talk like X") stay on the LLM route.
_IMPERSONATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "do an impersonation/impression of X", "give us your impression of X"
    re.compile(
        r"\b(?:do|give\s+(?:me|us)|perform)\s+(?:an?\s+|your\s+(?:best\s+)?)?"
        r"(?:impersonation|impression)\s+of\s+(?P<target>.+)$",
        re.IGNORECASE,
    ),
    # "impersonate X", "imitate X", "mimic X"
    re.compile(r"\b(?:impersonate|imitate|mimic)\s+(?P<target>.+)$", re.IGNORECASE),
    # "do/copy/clone my voice", "do Jimmy Carter's voice"
    re.compile(
        r"\b(?:do|copy|clone|steal)\s+(?P<target>.+?)(?:'s)?\s+voice\b",
        re.IGNORECASE,
    ),
)
_IMPERSONATE_NEGATION_RE = re.compile(
    r"\b(?:don'?t|do\s+not|never|stop|quit|no\s+more)\b", re.IGNORECASE
)
# The verb with NOBODY named — "Impersonate." on its own, usually because the
# speaker was cut off before finishing the sentence (field 2026-08-04). This used
# to classify as nothing at all, so the turn fell through to the LLM, which
# answered a bare "impersonate" by declining to impersonate anyone. It routes to
# the same action with an empty target, and the handler asks who.
_IMPERSONATE_BARE_RE = re.compile(
    r"^(?:(?:do|give\s+(?:me|us)|perform)\s+(?:an?\s+|your\s+(?:best\s+)?)?"
    r"(?:impersonation|impression)|impersonate|imitate|mimic)"
    r"\s*[.!?]*$",
    re.IGNORECASE,
)
_IMPERSONATE_SELF_RE = re.compile(r"^(?:me|myself|my|mine)$", re.IGNORECASE)


def _clean_impersonate_target(raw: str) -> str:
    """Normalize a captured impersonation target: strip punctuation/filler, map
    self-references to the canonical 'speaker'."""
    target = " ".join((raw or "").strip().split())
    target = re.sub(r"[.!?,;:]+$", "", target).strip()
    # Trailing politeness/filler: "impersonate me please", "... for me"
    target = re.sub(r"\b(?:please|for\s+(?:me|us)|right\s+now|now)$", "", target,
                    flags=re.IGNORECASE).strip()
    target = re.sub(r"[.!?,;:]+$", "", target).strip()
    if _IMPERSONATE_SELF_RE.match(target):
        return "speaker"
    return target


def classify_explicit_impersonation(text: str) -> ActionDecision | None:
    """Deterministically classify an explicit impersonation request, or None."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    verb_seen = False
    for pattern in _IMPERSONATE_PATTERNS:
        m = pattern.search(cleaned)
        if not m:
            continue
        # "don't impersonate me" / "stop imitating him" must not fire.
        if _IMPERSONATE_NEGATION_RE.search(cleaned[: m.start()]):
            return None
        target = _clean_impersonate_target(m.group("target"))
        if target:
            return ActionDecision(
                action="performance.impersonate",
                confidence=0.95,
                args={"target": target},
                reason="explicit impersonation request",
            )
        # The verb matched but nothing usable came with it ("impersonate please").
        verb_seen = True
        break
    if verb_seen or _IMPERSONATE_BARE_RE.match(cleaned):
        if _IMPERSONATE_NEGATION_RE.search(cleaned):
            return None
        return ActionDecision(
            action="performance.impersonate",
            confidence=0.9,
            args={"target": ""},
            reason="impersonation request with no target named",
        )
    return None
_BODY_BEAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:nod|agree|say\s+yes)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:agreement\s+nod|yes\s+nod)\b",
            re.IGNORECASE,
        ),
        "agreement_nod",
    ),
    (
        re.compile(
            r"\b(?:shake\s+(?:your\s+)?head|disagree|say\s+no)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:disagreement\s+shake|head\s+shake|no\s+shake)\b",
            re.IGNORECASE,
        ),
        "disagreement_shake",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+(?:surprised|shocked|startled)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:surprise\s+pop|surprised\s+pop|shock\s+reaction)\b",
            re.IGNORECASE,
        ),
        "surprise_pop",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+(?:disgusted|grossed\s+out)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:disgust\s+recoil|grossed\s+out\s+recoil)\b",
            re.IGNORECASE,
        ),
        "disgust_recoil",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+(?:angry|mad|furious)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:anger\s+flash|angry\s+flash)\b",
            re.IGNORECASE,
        ),
        "anger_flash",
    ),
    (
        re.compile(
            r"\b(?:look|act|be)\s+(?:giddy|joyful)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:giddy\s+wiggle|giddy\s+joy)\b",
            re.IGNORECASE,
        ),
        "giddy_wiggle",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+(?:sad|dejected)\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:sad\s+droop|sadness\s+droop)\b",
            re.IGNORECASE,
        ),
        "sad_droop",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+(?:in\s+)?disbelief\b|"
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:disbelief\s+stare|incredulous\s+stare)\b",
            re.IGNORECASE,
        ),
        "disbelief_stare",
    ),
    (
        re.compile(
            r"\b(?:do|perform|give|show|hit|drop)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:tiny\s+)?victory\s+dance\b|"
            r"\b(?:celebrate|do\s+a\s+little\s+dance)\b",
            re.IGNORECASE,
        ),
        "tiny_victory_dance",
    ),
    (
        re.compile(
            r"\b(?:look|act)\s+suspicious\b|"
            r"\b(?:do|perform|give|shoot|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:suspicious\s+glance|side\s+eye)\b",
            re.IGNORECASE,
        ),
        "suspicious_glance",
    ),
    (
        re.compile(
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:offended\s+recoil|insult\s+recoil)\b|"
            r"\b(?:look|act)\s+offended\b",
            re.IGNORECASE,
        ),
        "offended_recoil",
    ),
    (
        re.compile(
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:thinking\s+tilt|think\s+tilt)\b|"
            r"\b(?:look|act)\s+(?:thoughtful|confused|like\s+you'?re\s+thinking)\b",
            re.IGNORECASE,
        ),
        "thinking_tilt",
    ),
    (
        re.compile(
            r"\b(?:do|perform|give|show)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:dramatic\s+visor\s+peek|visor\s+peek)\b|"
            r"\bpeek\s+(?:the\s+)?visor\b",
            re.IGNORECASE,
        ),
        "dramatic_visor_peek",
    ),
    (
        re.compile(
            r"\b(?:do|perform|give|show|strike)\s+(?:me|us)?\s*(?:a|an|the|your)?\s*"
            r"(?:proud\s+dj\s+pose|dj\s+pose|proud\s+pose)\b",
            re.IGNORECASE,
        ),
        "proud_dj_pose",
    ),
)
_MOOD_POSE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:act|look|be)\s+(?:surprised|shocked|startled)\b", re.IGNORECASE),
        "surprised",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:angry|mad|furious)\b", re.IGNORECASE),
        "angry",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:disgusted|grossed\s+out)\b", re.IGNORECASE),
        "disgusted",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:giddy|joyful)\b", re.IGNORECASE),
        "giddy",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:sad|dejected)\b", re.IGNORECASE),
        "sad",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:in\s+)?disbelief\b", re.IGNORECASE),
        "disbelief",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:a\s+little\s+)?(?:embarrassed|sheepish|bashful)\b", re.IGNORECASE),
        "embarrassed",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:annoyed|irritated|fed\s+up)\b", re.IGNORECASE),
        "annoyed",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:proud|smug)\b", re.IGNORECASE),
        "proud",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:happy|excited|delighted)\b", re.IGNORECASE),
        "happy",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:suspicious|skeptical)\b", re.IGNORECASE),
        "suspicious",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:thoughtful|confused|like\s+you'?re\s+thinking)\b", re.IGNORECASE),
        "thinking",
    ),
    (
        re.compile(r"\b(?:act|look|be)\s+(?:offended|insulted)\b", re.IGNORECASE),
        "offended",
    ),
)
_VISION_SNAPSHOT_RE = re.compile(
    r"\b(?:remember|save|store|keep)\b.{0,80}\b(?:what\s+you\s+see|the\s+scene|this\s+view|that\s+view)\b|"
    r"\btake\s+a\s+look\b.{0,80}\b(?:keep|remember|save|store)\b.{0,60}\b(?:mind|memory|that|this)\b",
    re.IGNORECASE,
)
_HUMAN_VISUAL_PLAN_RE = re.compile(
    r"\b(?:i|we)\s+(?:really\s+)?"
    r"(?:wanna|want\s+to|need\s+to|plan\s+to|hope\s+to|would\s+like\s+to|should)\s+"
    r"(?:take|shoot|snap|capture|photograph|get)\s+"
    r"(?:a\s+|some\s+|the\s+)?"
    r"(?:pictures?|photos?|photographs?|images?|shots?|snapshots?)\b|"
    r"\b(?:i['’]?m|im|i\s+am|we['’]?re|we\s+are)\s+(?:really\s+)?"
    r"(?:going\s+to|gonna|about\s+to)\s+"
    r"(?:take|shoot|snap|capture|photograph|get)\s+"
    r"(?:a\s+|some\s+|the\s+)?"
    r"(?:pictures?|photos?|photographs?|images?|shots?|snapshots?)\b|"
    r"\b(?:i|we)\s+(?:really\s+)?"
    r"(?:wanna|want\s+to|need\s+to|plan\s+to|hope\s+to|would\s+like\s+to|should)\s+"
    r"(?:remember|save|store|keep)\s+(?:this|that|the|my|our)\s+"
    r"(?:scene|view|moment|picture|photo|image|shot|snapshot)\b|"
    r"\b(?:i['’]?m|im|i\s+am|we['’]?re|we\s+are)\s+(?:really\s+)?"
    r"(?:going\s+to|gonna|about\s+to)\s+"
    r"(?:remember|save|store|keep)\s+(?:this|that|the|my|our)\s+"
    r"(?:scene|view|moment|picture|photo|image|shot|snapshot)\b",
    re.IGNORECASE,
)
_ROAST_FOOD_TARGETS = {
    "beef",
    "chicken",
    "coffee",
    "pork",
    "turkey",
    "vegetables",
}


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


def _clean_roast_target(raw: str) -> str:
    target = " ".join(str(raw or "").strip(" .?!").split())
    lowered = target.lower()
    if lowered in {"me", "myself"}:
        return "speaker"
    if lowered in {"us", "we", "the room", "this room"}:
        return "room"
    if lowered in {"yourself", "you"}:
        return "rex"
    return target


def _clean_name_arg(raw: str) -> str:
    text = " ".join(str(raw or "").strip(" .?!").split())
    text = re.split(
        r"\b(?:instead|from\s+now\s+on|please|thanks|thank\s+you)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" .?!")
    words = []
    for raw_word in text.split():
        word = re.sub(r"[^A-Za-z'-]", "", raw_word).strip("'-")
        if word:
            words.append(word)
    if not words or len(words) > 3:
        return ""
    if any(word.lower() in {"i", "im", "i'm", "me", "my", "name"} for word in words):
        return ""
    if all(word.islower() for word in words):
        words = [word.capitalize() for word in words]
    return " ".join(words)


def _plausible_name_arg(raw: str) -> bool:
    """Return True when a candidate looks like a person name."""
    name = normalize_person_name(raw, allow_single=True)
    if not name:
        return False
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z'\-]*", name)
    ]
    if not tokens:
        return False
    if any(token in _NON_NAME_THATS_NOT_TOKENS for token in tokens):
        return False
    return True


def _plausible_thats_not_name_candidate(raw: str) -> bool:
    """Return True when a "that's not X" tail looks like a person name."""
    return _plausible_name_arg(raw)


def _text_has_identity_name_correction_content(
    text: str,
    decision: ActionDecision | None = None,
) -> bool:
    """Separate real name corrections from generic "that's not ..." replies."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False

    raw_name = ""
    if decision is not None:
        raw_name = str(
            decision.args.get("name")
            or decision.args.get("new_name")
            or decision.args.get("person_name")
            or ""
        ).strip()
    if raw_name and _plausible_name_arg(raw_name):
        return True

    named_match = _NAME_FROM_TEXT_RE.search(cleaned)
    if named_match and _plausible_name_arg(named_match.group("name")):
        return True

    if re.search(
        r"\b(?:you\s+(?:got|have)\s+my\s+name\s+wrong|"
        r"that['’]?s\s+not\s+my\s+name|that\s+isn['’]?t\s+my\s+name|"
        r"you\s+called\s+me\s+the\s+wrong\s+name)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return True

    wrong_name = _THATS_NOT_NAME_RE.search(cleaned)
    if wrong_name is not None:
        return _plausible_thats_not_name_candidate(wrong_name.group("name"))

    return False


def classify_explicit_control(text: str) -> ActionDecision | None:
    """Classify obvious non-performance control requests without an LLM call."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    if _RECENT_DISCARD_REQUEST_RE.search(cleaned):
        return ActionDecision(
            action="memory.recent_discard",
            confidence=0.97,
            args={"scope": "recent"},
            reason="explicit recent-memory discard request",
        )

    if _NAME_CORRECTION_REQUEST_RE.search(cleaned):
        name = ""
        match = _NAME_FROM_TEXT_RE.search(cleaned)
        if match:
            name = _clean_name_arg(match.group("name"))
        if not name:
            wrong_name = _THATS_NOT_NAME_RE.search(cleaned)
            if wrong_name is not None and not _plausible_thats_not_name_candidate(
                wrong_name.group("name")
            ):
                return None
        return ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={"name": name} if name else {},
            reason="explicit speaker name correction",
        )

    if _VISION_SNAPSHOT_RE.search(cleaned) and not _HUMAN_VISUAL_PLAN_RE.search(cleaned):
        return ActionDecision(
            action="vision.snapshot",
            confidence=0.94,
            args={"scope": "current_view"},
            requires_confirmation=True,
            reason="privacy-sensitive request to remember current view",
        )

    return None


# Drive-base motion (deterministic; only acted on when a base is connected — the
# interaction layer gates these on motion_controller.available()).
_MOTION_COME_RE = re.compile(
    r"\b(come\s+(?:here|over\s+here|to\s+me|closer|on\s+over)|over\s+here|"
    r"roll\s+over\s+here|get\s+over\s+here|come\s+to\s+(?:me|daddy))\b",
    re.I,
)
# Cardinal directions (true compass headings; executed against the calibrated
# QMC5883's fused yaw at run time). Diagonals accept solid/spaced/hyphenated forms.
_CARDINAL_DEG = {
    "north": 0.0, "northeast": 45.0, "east": 90.0, "southeast": 135.0,
    "south": 180.0, "southwest": 225.0, "west": 270.0, "northwest": 315.0,
}
_CARDINAL_PAT = (
    r"(?P<card>north\s*-?\s*east|north\s*-?\s*west|south\s*-?\s*east|"
    r"south\s*-?\s*west|north|south|east|west)"
)
def _normalize_cardinal(raw: str) -> str:
    return re.sub(r"[\s\-]+", "", str(raw or "").strip().lower())
# An optional "small amount / manner" phrase between the move verb and the direction,
# so "move a little forward" / "ease slightly back" classify. Kept a strict whitelist
# (not ".*") so "move the box forward" still does NOT false-positive as a drive command.
_MOTION_AMOUNT = (
    r"(?:a\s+(?:little|bit|tad|touch|smidge)|slightly|just|gently|slowly|"
    r"kinda|kind\s+of|tiny\s+bit)"
)
# "ahead" is intentionally NOT a forward trigger — "go ahead and …" is almost always
# figurative. The trailing negative lookahead drops the figurative "move forward
# with/in/on the plan / in life" while still allowing "move forward", "… 2 feet",
# "… a little", and "… and to your right".
_MOTION_FWD_RE = re.compile(
    r"\b(?:move|go|roll|drive|scoot|head|creep|come|ease|inch|edge|pull)"
    rf"(?:\s+{_MOTION_AMOUNT})?\s+(?:forward|forwards)\b"
    r"(?!\s+(?:with|in|on|through|into|towards?)\b)",
    re.I,
)
_MOTION_BACK_RE = re.compile(
    r"\b(?:back\s*up|backup|reverse|"
    r"(?:move|go|roll|drive|scoot|head|ease|inch|edge|pull)"
    rf"(?:\s+{_MOTION_AMOUNT})?\s+back(?:ward|wards)?)\b",
    re.I,
)
# "turn/face/... north" — rotate in place to the true heading. "due north" tolerated.
_MOTION_COMPASS_TURN_RE = re.compile(
    r"\b(?:turn|face|point|look|rotate|spin)\s+"
    r"(?:to\s+|toward[s]?\s+|to\s+the\s+|to\s+face\s+|yourself\s+)?"
    rf"(?:due\s+)?{_CARDINAL_PAT}\b",
    re.I,
)
# "go/move/... north [two feet]" — face the heading, then advance.
_MOTION_COMPASS_GO_RE = re.compile(
    r"\b(?:move|go|roll|drive|scoot|head|creep|ease|inch|edge)\s+"
    rf"(?:{_MOTION_AMOUNT}\s+)?(?:to\s+the\s+|toward[s]?\s+(?:the\s+)?|due\s+)?{_CARDINAL_PAT}\b",
    re.I,
)
# "a little / a bit / nudge / inch …" with no explicit distance => a SMALL move.
_MOTION_SMALL_RE = re.compile(rf"\b(?:{_MOTION_AMOUNT}|nudge|inch)\b", re.I)
_MOTION_SMALL_MOVE_M = 0.15
# A direct "turn left/right a little" is still a useful deliberate reorientation,
# not the tiny 15° conversational continuation used by "a little more".
_MOTION_SMALL_TURN_DEG = 45.0
# Compound arc: a forward/back move + a left/right component joined by "and" in ONE
# utterance ("move a little forward and to your right") => a simultaneous curve. The
# left/right must follow the "and" (within a short window) so it's the arc's turn part.
_MOTION_ARC_LAT_RE = re.compile(r"\band\b.{0,18}?\b(?P<lr>left|right)\b", re.I)
# LATERAL move: "move to your left", "go left", "scoot over to the right" — a move
# verb aimed at a SIDE, with no forward/back word (field-logged 2026-07-11: "Move to
# your left" fell through to conversation and got a quip instead of motion). A
# differential base can't strafe, so this executes as a small forward arc toward the
# side (motion.arc). Turn verbs (turn/rotate/spin/...) are deliberately NOT in the
# verb list — "turn left" stays a pure motion.turn.
_MOTION_LATERAL_RE = re.compile(
    r"\b(?:move|go|roll|drive|scoot|slide|shift|step|shimmy|edge|ease|inch|head)"
    rf"(?:\s+{_MOTION_AMOUNT})?"
    r"(?:\s+over)?"
    r"(?:\s+(?:to|toward|towards))?(?:\s+(?:your|the|my))?"
    r"\s+(?P<lr>left|right)\b",
    re.I,
)
_MOTION_TURN_RE = re.compile(
    r"\b(?:turn|rotate|spin|pivot|swing|face)\b.{0,20}?\b"
    r"(?P<dir>left|right|around|clockwise|counter[-\s]?clockwise)\b",
    re.I,
)
_MOTION_STOP_RE = re.compile(
    r"\b(halt|freeze|stop\s+(?:moving|driving|rolling)|"
    r"stop\s+the\s+(?:robot|base|droid|wheels|car)|hold\s+still|"
    # "don't move" is a stop; "don't move FORWARD" is a prohibition on a heading —
    # letting the stop branch claim it would defeat the negation guard and drive him
    # forward. A direction after the verb disqualifies it.
    r"(?:don'?t|do\s+not|stop)\s+mov(?:e|ing)\b"
    r"(?!\s+(?:forward|forwards|back|backward|backwards|left|right|up|down|"
    r"closer|away|toward|towards|into|onto|to|past|around|any)\b)|"
    r"quit\s+moving|stay\s+(?:there|put|still))\b",
    re.I,
)
_MOTION_DEG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)", re.I)
_MOTION_BARE_TURN_DEG_RE = re.compile(
    r"\b(?:turn|rotate|spin|pivot)\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:°)?\s*$",
    re.I,
)
_MOTION_DIST_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:er|re)s?|mm|millimet(?:er|re)s?|"
    r"m|met(?:er|re)s?|ft|foot|feet|in|inch|inches)\b",
    re.I,
)
_MOTION_EXPLANATION_RE = re.compile(
    r"^\s*(?:so\s+)?(?:why\b|how\s+come\b|what\s+made\s+(?:you|him|her|it|them)\b)",
    re.I,
)
_MOTION_NEGATED_RE = re.compile(
    r"\b(?:don'?t|do\s+not|never|shouldn'?t|mustn'?t|can'?t|cannot)\b"
    r".{0,24}\b(?:move|go|roll|drive|turn|rotate|spin|pivot|back\s*up|come)\b",
    re.I,
)
_MOTION_MORE_RE = re.compile(
    r"^\s*(?P<small>(?:a\s+)?(?:little|bit)\s+)?more\s*[.!]*\s*$", re.I,
)
_MOTION_KEEP_TURNING_RE = re.compile(
    r"^\s*(?:keep|continue)\s+(?:on\s+)?turning\s*[.!]*\s*$", re.I,
)
_MOTION_KEEP_MOVING_RE = re.compile(
    r"^\s*(?:keep|continue)\s+(?:on\s+)?moving\s*[.!]*\s*$", re.I,
)
_MOTION_KEEP_GOING_RE = re.compile(
    r"^\s*(?:keep|continue)\s+(?:on\s+)?going\s*[.!]*\s*$", re.I,
)
_MOTION_SEQUENCE_SEP_RE = re.compile(
    r"\s*(?:"
    r"(?:[;,]\s*)?\b(?:and\s+then|then)\b|"
    r"[;,]|"
    # "come" was missing, so "turn around and come forward 5 feet" never split —
    # the whole utterance fell through to the single-command path and only the
    # FORWARD half ran (the turn was silently dropped; field 2026-07-24).
    r"\band\b(?=\s+(?:turn|rotate|spin|pivot|move|go|roll|drive|scoot|come|"
    r"advance|creep|head|ease|inch|edge|pull|back\s*up|reverse)\b)"
    r")\s*",
    re.I,
)


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "a": 1, "an": 1, "half a": 0.5, "half an": 0.5,
}
_MOTION_UNIT_PAT = (
    r"(cm|centimet(?:er|re)s?|mm|millimet(?:er|re)s?|"
    r"m|met(?:er|re)s?|ft|foot|feet|in|inch|inches)"
)
# Whisper transcribes small counts as WORDS ("four feet", "half a meter"), and voice
# is the only way distances arrive — the digit-only regex meant spoken distances were
# silently dropped and every move fell back to the default nudge (field 2026-07-21).
_MOTION_WORD_DIST_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\s+"
    + _MOTION_UNIT_PAT + r"\b",
    re.I,
)


def _unit_to_m(val: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("mm") or unit.startswith("millim"):
        return val / 1000.0
    if unit.startswith("cm") or unit.startswith("centim"):
        return val / 100.0
    if unit in ("ft", "foot", "feet"):
        return val * 0.3048
    if unit in ("in", "inch", "inches"):
        return val * 0.0254
    return val  # metres


def _motion_dist_to_m(text: str) -> "float | None":
    m = _MOTION_DIST_RE.search(text)
    if m:
        return _unit_to_m(float(m.group(1)), m.group(2))
    m = _MOTION_WORD_DIST_RE.search(text)
    if m:
        return _unit_to_m(float(_WORD_NUMBERS[m.group(1).lower()]), m.group(2))
    return None


def classify_motion_continuation(
    text: str,
    previous: ActionDecision | None,
    *,
    small_turn_deg: float = 15.0,
    small_move_m: float = 0.15,
) -> ActionDecision | None:
    """Bind a short follow-up to the most recent successful finite motion.

    Deliberately requires caller-supplied live context: these phrases are not motion
    commands on their own. "Keep turning" only binds to a turn and "keep moving"
    only to a move/arc; generic "keep going" and bare "more" repeat any of those.
    "A little more" keeps direction but substitutes a small bounded increment.
    """
    if previous is None or previous.action not in {"motion.turn", "motion.move", "motion.arc"}:
        return None
    cleaned = " ".join((text or "").strip().split())
    more = _MOTION_MORE_RE.match(cleaned)
    if _MOTION_KEEP_TURNING_RE.match(cleaned):
        if previous.action != "motion.turn":
            return None
    elif _MOTION_KEEP_MOVING_RE.match(cleaned):
        if previous.action not in {"motion.move", "motion.arc"}:
            return None
    elif _MOTION_KEEP_GOING_RE.match(cleaned):
        pass
    elif not more:
        return None

    args = dict(previous.args or {})
    if more and more.group("small"):
        if previous.action == "motion.turn":
            args["deg"] = abs(float(small_turn_deg))
        elif previous.action == "motion.move":
            args["dist_m"] = abs(float(small_move_m))
        else:
            args["small"] = True
    return ActionDecision(
        action=previous.action,
        confidence=0.97,
        args=args,
        reason=f"continuation of previous {previous.action} command",
    )


# Non-motion ACTIONS that may legitimately trail a route ("turn left, sing"). These
# must never be mistaken for a vocative — the whole point of the tri-state refusal is
# that Rex must not run half of "turn left then sing".
_NON_MOTION_ACTION_WORDS = frozenset({
    "sing", "dance", "play", "stop", "halt", "wait", "hold", "freeze", "talk",
    "speak", "listen", "look", "watch", "jump", "shut", "quiet", "sleep", "wake",
    "dj", "rap", "beatbox", "joke", "laugh", "scan", "search", "find", "follow",
})

# A trailing address ("..., Rex", "..., buddy") or a one-word Whisper garble
# ("...5 feet" heard as "..., Ozzie"). Captured so it can be dropped before the
# route parser treats it as an unsupported clause and refuses the WHOLE command.
_TRAILING_VOCATIVE_RE = re.compile(
    r",\s*(?P<frag>[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)?)\s*[.!?]*$"
)


# A bare MAGNITUDE that follows a comma qualifies the clause BEFORE it — "turn left,
# 15 degrees" is one command, not a route whose second leg is the noun phrase
# "15 degrees". Number (digits or words) + a distance/angle unit, optionally hedged.
_MOTION_MAGNITUDE_TAIL_RE = re.compile(
    r",\s*(?P<mag>(?:about|around|roughly|approximately|like)?\s*"
    r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\s*"
    r"(?:deg|degree|degrees|°|" + _MOTION_UNIT_PAT + r"))"
    r"(?=\s*(?:$|[.!?,;]|\b(?:and|then)\b))",
    re.I,
)


def _rejoin_magnitude_clauses(text: str) -> str:
    """Drop the comma in "<motion>, <magnitude>" so the magnitude stays with its verb.

    Field 2026-07-24: "Turn left, 15 degrees." drew "I couldn't safely parse that
    whole route" and Rex never moved — the comma split it into ["Turn left",
    "15 degrees"], the second clause classified as nothing, and the mixed
    motion/non-motion guard refused the WHOLE utterance. The single-command
    classifier parses the un-comma'd form perfectly (left, 15 deg), so the comma is
    pure punctuation here. Unlike _strip_trailing_vocative this REJOINS rather than
    discards — the angle/distance must survive.
    """
    return _MOTION_MAGNITUDE_TAIL_RE.sub(lambda m: " " + m.group("mag").strip(), text or "")


def _strip_trailing_vocative(text: str) -> str:
    """Drop a trailing comma-address / one-word garble from a route command.

    Field 2026-07-24: "Turn around and come forward, Ozzie" (Whisper's take on
    "...5 feet") was split into a valid motion clause plus the junk clause
    "Ozzie", which tripped the no-partial-execution refusal and NOTHING ran. The
    same shape broke plain politeness — "turn left then move forward five feet,
    Rex" refused too. Only a comma-introduced fragment of 1-2 alphabetic words
    that is neither a motion clause nor a known non-motion action is removed, so
    "turn left, sing" still refuses as designed.
    """
    match = _TRAILING_VOCATIVE_RE.search(text or "")
    if not match:
        return text
    frag = match.group("frag").strip()
    words = frag.lower().split()
    if any(w.strip(".,!?'-") in _NON_MOTION_ACTION_WORDS for w in words):
        return text
    if classify_explicit_motion(frag) is not None:
        return text          # a real (if terse) motion clause — keep it
    stripped = text[: match.start()].strip(" ,.")
    return stripped or text


def classify_explicit_motion_sequence(
    text: str,
    *,
    max_steps: int = 8,
) -> list[ActionDecision] | None:
    """Parse an ordered chain of explicit finite motion clauses.

    Returns ``[]`` when the utterance is not a sequence, ``None`` when it looks like
    a sequence but any clause is invalid/unsupported, and 2+ decisions on success.
    The tri-state prevents partial execution: "turn left then sing" must not execute
    its first clause. Plain ``and`` splits only when another motion verb follows, so
    "move forward and to your right" remains the existing single arc command.
    """
    cleaned = " ".join((text or "").strip().split())
    # Re-attach magnitudes BEFORE stripping a vocative: ", 15 degrees" is part of the
    # command, ", Rex" is not. Both run before the separator scan so a comma that is
    # only punctuation never looks like a route boundary.
    cleaned = _rejoin_magnitude_clauses(cleaned)
    cleaned = _strip_trailing_vocative(cleaned)
    if not cleaned or not _MOTION_SEQUENCE_SEP_RE.search(cleaned):
        return []
    # Negation/explanation makes this NOT-a-route — but which tri-state arm depends
    # on whether the utterance contains any actual motion clause, so the check moves
    # BELOW clause classification. Returning None up here meant any comma-containing
    # chatter with a "don't" in it was announced as an unparseable route (field
    # 2026-08-05 21:23: "I don't know. Hey, I'm gonna go now. Can you shut down,
    # please?" → "I couldn't safely parse that whole route" — the shutdown request
    # was eaten by a rejection for a route nobody asked for).
    negated_or_explaining = bool(
        _MOTION_EXPLANATION_RE.search(cleaned)
        or (_MOTION_NEGATED_RE.search(cleaned) and not _MOTION_STOP_RE.search(cleaned))
    )
    clauses = [c.strip(" .()") for c in _MOTION_SEQUENCE_SEP_RE.split(cleaned)]
    # A LEADING/TRAILING connective leaves empty fragments ("and move backwards" ->
    # ["", "move backwards"]). Drop them; if a single real clause remains this is NOT
    # a sequence — return [] so the caller falls through to the plain single-command
    # path instead of rejecting the whole utterance (field 2026-07-21: "and move
    # backwards" got "I couldn't safely parse that whole route" and nothing moved).
    clauses = [c for c in clauses if c]
    # Fewer than 2 real clauses is NOT a sequence — a lone fragment ("and move
    # backwards") or a pure connective/disfluency ("and then,", "then,") must fall
    # through to [] so the caller tries the plain single-command path / normal
    # conversation, NOT the route-rejection line (field 2026-07-23: "and then," drew
    # "I couldn't safely parse that whole route" — the 0-clause case reached None).
    if len(clauses) < 2:
        return []
    if len(clauses) > max(2, int(max_steps)):
        return None
    decisions: list[ActionDecision] = []
    misses = 0
    for clause in clauses:
        decision = classify_explicit_motion(clause)
        if decision is None or decision.action not in {"motion.turn", "motion.move", "motion.arc"}:
            misses += 1
            continue
        decisions.append(decision)
    if not decisions:
        # ZERO motion clauses: plain conversation that happens to contain a comma/'then'
        # ("yeah that sounds great, thanks") — not a sequence at all. Returning None here
        # made Rex say "I couldn't safely parse that whole route" at casual chatter.
        # This ALSO covers the negated/explaining case: "I don't know, I'm gonna go
        # now" has a negation and a comma but no motion clause — conversation.
        return []
    if negated_or_explaining:
        # A negation/explanation over an utterance that DOES contain motion clauses
        # ("don't turn left then move forward", "why didn't you move, then turn?") —
        # refuse the whole thing so nothing executes (the guard's original purpose).
        return None
    if misses:
        # MIXED motion + non-motion ("turn left then sing"): refuse the whole thing so
        # no partial execution — the original purpose of the tri-state.
        return None
    return decisions


def classify_explicit_motion(text: str) -> ActionDecision | None:
    """Classify explicit drive-base motion commands without an LLM call.

    Conservative/high-precision: only clear directional phrases. Bare 'stop' is
    intentionally NOT claimed here (the interaction layer routes it to the base
    only while it is actually moving, so 'stop' still means stop-music/game/talk
    otherwise)."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    # Questions about prior behavior and explicit negations are conversation, not
    # commands. This guard must precede every motion family because takeover runs
    # before the general dialogue router (field log: "How come you didn't move
    # forward?" otherwise drove the robot while the user was diagnosing it).
    # A STOP phrase outranks the negation guard. "Don't move" is BOTH — a negated
    # motion verb and an explicit halt — and the guard (there to stop "don't turn
    # left" from turning left) swallowed it entirely, so the phrase did nothing even
    # when heard perfectly. Field 2026-07-25: the owner said "don't move" twice at a
    # base grinding against carpet and was ignored both times.
    if _MOTION_EXPLANATION_RE.search(cleaned) or (
        _MOTION_NEGATED_RE.search(cleaned) and not _MOTION_STOP_RE.search(cleaned)
    ):
        return None

    # Cardinal-direction commands (needs the calibrated compass at execution time).
    # "turn/face/point/look/rotate north" -> rotate to the true heading;
    # "go/move/drive/roll/head north [two feet]" -> face it, then advance.
    m = _MOTION_COMPASS_TURN_RE.search(cleaned)
    if m:
        card = _normalize_cardinal(m.group("card"))
        return ActionDecision(
            action="motion.turn", confidence=0.95,
            args={"compass": card, "compass_deg": _CARDINAL_DEG[card]},
            reason="explicit compass turn",
        )
    m = _MOTION_COMPASS_GO_RE.search(cleaned)
    if m:
        # Figurative guard: "this could go south", "it all went south" are conversation.
        # An imperative drive command isn't preceded by an auxiliary/subject word.
        prefix_words = cleaned[: m.start()].lower().split()
        if prefix_words and prefix_words[-1] in (
            "could", "would", "might", "may", "will", "can", "to", "gonna",
            "things", "it", "this", "that", "everything", "all",
        ):
            pass
        else:
            card = _normalize_cardinal(m.group("card"))
            args: dict[str, Any] = {"compass": card, "compass_deg": _CARDINAL_DEG[card],
                                    "direction": "forward"}
            dist = _motion_dist_to_m(cleaned)
            if dist is not None:
                args["dist_m"] = dist
            return ActionDecision(
                action="motion.move", confidence=0.95, args=args,
                reason="explicit compass move",
            )

    if _MOTION_COME_RE.search(cleaned):
        return ActionDecision(
            action="motion.come", confidence=0.95, args={},
            reason="explicit come-here request",
        )

    # Compound arc — a forward/back move AND a left/right turn joined by "and" in a
    # single utterance drives a brief simultaneous curve. (Two separate utterances —
    # "move forward" then "turn right" — are NOT merged; each is its own finite command
    # via the per-utterance pipeline.)
    if re.search(r"\band\b", cleaned):
        fwd = bool(_MOTION_FWD_RE.search(cleaned))
        back = bool(_MOTION_BACK_RE.search(cleaned))
        lat = _MOTION_ARC_LAT_RE.search(cleaned)
        if (fwd or back) and lat:
            return ActionDecision(
                action="motion.arc", confidence=0.95,
                args={
                    "lin_dir": "forward" if fwd else "back",
                    "ang_dir": lat.group("lr").lower(),
                    "small": bool(_MOTION_SMALL_RE.search(cleaned)),
                },
                reason="explicit compound move+turn arc",
            )

    if _MOTION_FWD_RE.search(cleaned):
        args: dict[str, Any] = {"direction": "forward"}
        dist = _motion_dist_to_m(cleaned)
        if dist is None and _MOTION_SMALL_RE.search(cleaned):
            dist = _MOTION_SMALL_MOVE_M
        if dist is not None:
            args["dist_m"] = dist
        return ActionDecision(
            action="motion.move", confidence=0.95, args=args,
            reason="explicit move-forward request",
        )

    if _MOTION_BACK_RE.search(cleaned):
        args = {"direction": "back"}
        dist = _motion_dist_to_m(cleaned)
        if dist is None and _MOTION_SMALL_RE.search(cleaned):
            dist = _MOTION_SMALL_MOVE_M
        if dist is not None:
            args["dist_m"] = dist
        return ActionDecision(
            action="motion.move", confidence=0.95, args=args,
            reason="explicit move-back request",
        )

    lateral = _MOTION_LATERAL_RE.search(cleaned)
    if lateral:
        return ActionDecision(
            action="motion.arc", confidence=0.95,
            args={
                "lin_dir": "forward",
                "ang_dir": lateral.group("lr").lower(),
                "small": True,   # lateral repositioning: always a brief, gentle curve
            },
            reason="explicit lateral move request",
        )

    # Natural spoken shorthand commonly omits the unit: "turn 180". Keep this
    # narrow (turn verb + one trailing number) so unrelated numeric phrases never
    # acquire control of the drive base.
    bare_turn = _MOTION_BARE_TURN_DEG_RE.search(cleaned)
    if bare_turn:
        degrees = float(bare_turn.group("deg"))
        return ActionDecision(
            action="motion.turn", confidence=0.95,
            args={"direction": "around" if degrees == 180.0 else "left", "deg": degrees},
            reason="explicit numeric turn request",
        )

    turn = _MOTION_TURN_RE.search(cleaned)
    if turn:
        direction = turn.group("dir").lower().replace(" ", "").replace("-", "")
        args = {}
        if direction == "around":
            args["direction"] = "around"
            args["deg"] = 180.0
        elif direction in ("right", "clockwise"):
            args["direction"] = "right"
        else:  # left / counterclockwise
            args["direction"] = "left"
        deg = _MOTION_DEG_RE.search(cleaned)
        if deg is not None:
            args["deg"] = float(deg.group(1))
        elif _MOTION_SMALL_RE.search(cleaned):
            args["deg"] = _MOTION_SMALL_TURN_DEG
        return ActionDecision(
            action="motion.turn", confidence=0.95, args=args,
            reason="explicit turn request",
        )

    if _MOTION_STOP_RE.search(cleaned):
        return ActionDecision(
            action="motion.stop", confidence=0.97, args={},
            reason="explicit motion-stop request",
        )

    return None


# ── Room-exploration invitation (deterministic; motion.explore) ───────────────
# "feel free to explore the room", "look around a little", "wander around",
# "check the place out", "make yourself at home". HIGH-PRECISION and safety-critical:
# a false positive drives a physical base around the room and seizes the floor, and it
# runs BEFORE the dialogue-act gate, so nothing downstream can undo a misfire. The
# guard is that the invite must be an IMPERATIVE ADDRESSED TO REX — the core verb
# phrase must be verb-first at the START of the utterance, after only an optional
# whitelist of invitation lead-ins / vocatives ("feel free to", "why don't you",
# "hey Rex, …"). This is what keeps a first-person answer ("I love to wander around
# the city") or third-party narration ("the dog likes to roam around the yard") from
# launching a wander. It also declines negations, directed-vision queries, and search
# errands.
#
# Invitation lead-ins that mark the phrase as ADDRESSED TO REX. Stripped (repeatedly)
# off the FRONT before the imperative core is matched. First-person subjects ("I",
# "we") and third-party subjects ("the dog") are deliberately NOT here, so a
# declarative statement never reduces to a bare imperative core.
_EXPLORE_LEAD_RE = re.compile(
    r"^(?:"
    r"hey|ok|okay|alright|so|now|please|rex|c'?mon|come\s+on|"
    r"feel\s+free\s+to|go\s+ahead\s+and|go\s+ahead|why\s+don'?t\s+you|why\s+not|"
    r"you\s+can|you\s+could|you\s+should|you\s+might|how\s+about\s+you|how\s+about|"
    r"maybe\s+you|maybe|just|go\s+and|go|wanna|want\s+to|"
    r"i'?d\s+like\s+you\s+to|i\s+want\s+you\s+to|let'?s|"
    r"feel\s+free|be\s+my\s+guest\s+and|be\s+my\s+guest"
    r")\b[\s,!.]*",
    re.I,
)
# The imperative invite core, anchored to the START of the (lead-stripped) text.
# The bare "explore" branch is restricted to a room-ish object or end-of-clause so
# "explore my feelings" / "explore your options" do NOT match.
_EXPLORE_CORE_RE = re.compile(
    r"^(?:"
    r"explore(?:\s+(?:the|your|this)\s+(?:new\s+|whole\s+|entire\s+|rest\s+of\s+the\s+)?"
    r"(?:room|place|space|area|surroundings|environment|"
    r"home|domain|joint|pad|apartment|house|studio|office|garage|yard|garden)"
    r"|\s+(?:around|room|place|space|area|surroundings|environment)"
    r"|[\s.!?]*$)"
    r"|look(?:ing)?\s+around\b"
    r"|have\s+a\s+look\s+around\b"
    r"|take\s+a\s+look\s+around\b"
    r"|wander\b|roam\b|scout\b"
    r"|nose\s+around\b|poke\s+around\b|nose\s+about\b|poke\s+about\b"
    r"|check\s+(?:out\s+)?(?:the|this|your)\s+(?:room|place|space|joint|pad|surroundings)"
    r"|check\s+(?:the|this|your)\s+(?:room|place|space|joint|pad|surroundings)\s+out"
    r"|scope\s+(?:out\s+)?(?:the|this|your)\s+(?:room|place|space|joint|pad|surroundings)"
    r"|case\s+(?:the|this|your)\s+(?:room|place|space|joint|pad)"
    r"|survey\s+(?:the|this|your)\s+(?:room|place|space|surroundings|domain)"
    r"|scan\s+(?:the|this|your)\s+(?:room|place|space|surroundings)"
    r"|take\s+a\s+(?:lap|tour|spin|stroll|walk)\b"
    r"|make\s+yourself\s+at\s+home\b"
    r")",
    re.I,
)
# A see/describe request → this is the existing directed-vision path, not an invite.
_EXPLORE_SEE_REQUEST_RE = re.compile(
    r"\b(what\s+(?:do|can)\s+you\s+see|tell\s+me\s+what|describe|what'?s\s+(?:there|in\s+(?:front|the))"
    r"|what\s+do\s+you\s+notice)\b",
    re.I,
)
# A search errand ("look around for my keys", "scan the room for it") → not a wander.
# Broad on both the verb and the object (incl. pronoun objects) so a "find X" errand is
# never mistaken for an open-ended wander.
_EXPLORE_SEARCH_ERRAND_RE = re.compile(
    r"\b(?:look|search|hunt|check|scan|survey|scope|case|find|locate|spot)\b[^.?!]*?"
    # "for <target>" marks an errand — but NOT a duration/manner ("for a while", "for fun").
    r"\bfor\s+(?!(?:a\s+|the\s+)?(?:while|bit|moment|sec|second|minute|fun|now|once|good|kicks|"
    r"a\s+laugh)\b)"
    r"(?:my|the|a|an|some|any|it|him|her|them|us|me|his|your|our|that|those|\w)",
    re.I,
)
# A negation at the very front of the (lead-stripped) core reverses the intent.
_EXPLORE_NEGATION_RE = re.compile(
    r"^(?:don'?t|do\s+not|never|no\b|not\b|stop|quit|cut\s+it|knock\s+it|hold\s+off|"
    r"instead\s+of|rather\s+than|without)\b",
    re.I,
)


def classify_explicit_exploration(text: str) -> ActionDecision | None:
    """Classify an invitation for Rex to autonomously explore the room (no LLM).

    HIGH-PRECISION: only fires when the utterance is an IMPERATIVE invitation ADDRESSED
    TO REX — the core verb phrase must be verb-first after only an optional whitelist of
    invitation lead-ins/vocatives. This is what stops a first-person answer ("I love to
    wander around the city") or third-party narration ("the dog roams around the yard")
    from launching a physical wander. Also declines negations, directed-vision queries,
    and search errands. Runs AFTER classify_explicit_motion in the takeover, so a
    'turn around' has already been claimed as a turn.
    """
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    # Yield outright to the existing directed-vision path and search errands.
    if _EXPLORE_SEE_REQUEST_RE.search(cleaned):
        return None
    if _EXPLORE_SEARCH_ERRAND_RE.search(cleaned):
        return None
    # Strip invitation lead-ins ("feel free to", "why don't you", "hey Rex,") off the
    # FRONT so the imperative core is at the start; a declarative subject ("I", "the
    # dog") is not a lead-in, so it will fail the verb-first core match below.
    core = cleaned
    for _ in range(5):
        m = _EXPLORE_LEAD_RE.match(core)
        if not m or m.end() == 0:
            break
        core = core[m.end():]
    if not core:
        return None
    # A negation at the front of the remaining core reverses the intent.
    if _EXPLORE_NEGATION_RE.match(core):
        return None
    # The remaining text must START with an imperative invite core.
    if not _EXPLORE_CORE_RE.match(core):
        return None
    return ActionDecision(
        action="motion.explore", confidence=0.93, args={},
        reason="explicit room-exploration invitation",
    )


def classify_explicit_humor(text: str) -> ActionDecision | None:
    """Classify obvious humor-performance requests without an LLM call."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    roast = _ROAST_REQUEST_RE.search(cleaned)
    if roast:
        raw_target = (
            roast.groupdict().get("target")
            or roast.groupdict().get("target2")
            or "speaker"
        )
        target = _clean_roast_target(raw_target)
        if target.lower() not in _ROAST_FOOD_TARGETS:
            return ActionDecision(
                action="humor.roast",
                confidence=0.96,
                args={"target": target},
                reason="explicit roast request",
            )

    if _TELL_JOKE_RE.search(cleaned):
        return ActionDecision(
            action="humor.tell_joke",
            confidence=0.96,
            args={},
            reason="explicit joke request",
        )

    if _FREE_HUMOR_RE.search(cleaned):
        return ActionDecision(
            action="humor.free_bit",
            confidence=0.94,
            args={},
            reason="explicit free humor request",
        )

    return None


def classify_explicit_performance(text: str) -> ActionDecision | None:
    """Classify obvious non-music performance requests without an LLM call."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None

    impersonation = classify_explicit_impersonation(cleaned)
    if impersonation is not None:
        return impersonation

    if _DJ_BIT_RE.search(cleaned):
        return ActionDecision(
            action="performance.dj_bit",
            confidence=0.95,
            args={},
            reason="explicit DJ performance request",
        )

    for pattern, beat in _BODY_BEAT_PATTERNS:
        if pattern.search(cleaned):
            canonical = performance_plan.canonical_body_beat(beat)
            if canonical:
                return ActionDecision(
                    action="performance.body_beat",
                    confidence=0.95,
                    args={"body_beat": canonical},
                    reason="explicit body beat performance request",
                )

    for pattern, mood in _MOOD_POSE_PATTERNS:
        if pattern.search(cleaned):
            canonical = performance_plan.canonical_mood_pose(mood)
            if canonical:
                return ActionDecision(
                    action="performance.mood_pose",
                    confidence=0.94,
                    args={"mood": canonical},
                    reason="explicit emotion-driven physical pose request",
                )

    return None


def classify_explicit_character_preference(text: str) -> ActionDecision | None:
    """Classify obvious questions about Rex's own preferences without an LLM call."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return None
    parsed = rex_preferences.extract_preference_query(cleaned)
    if not parsed:
        return None
    return ActionDecision(
        action="character.preference_query",
        confidence=0.95,
        args=parsed,
        reason="explicit Rex preference/opinion question",
    )


def missing_required_evidence_reason(
    text: str,
    decision: ActionDecision | None,
    *,
    context: dict[str, Any] | None = None,
) -> str | None:
    """Return a block reason when an executable action lacks direct evidence.

    The action router may use an LLM, but execution should still require an
    action-shaped utterance. Ambiguous conversational replies fall through to
    normal conversation unless they contain this deterministic evidence.
    """
    context = context or {}
    if decision is None:
        return None
    cleaned = " ".join((text or "").strip().split())
    if not cleaned or decision.action == "conversation.reply":
        return None

    action = decision.action

    if action == "identity.name_correction":
        return (
            None
            if _text_has_identity_name_correction_content(cleaned, decision)
            else "missing_identity_name_evidence"
        )
    if action == "conversation.repair":
        return None if _REPAIR_REQUEST_RE.search(cleaned) else "missing_repair_evidence"
    if action == "memory.recent_discard":
        return None if _RECENT_DISCARD_REQUEST_RE.search(cleaned) else "missing_recent_discard_evidence"
    if action == "memory.forget_specific":
        return None if _FORGET_SPECIFIC_REQUEST_RE.search(cleaned) else "missing_forget_evidence"
    if action == "humor.tell_joke":
        return None if _TELL_JOKE_RE.search(cleaned) else "missing_joke_request_evidence"
    if action == "humor.roast":
        explicit = classify_explicit_humor(cleaned)
        return None if explicit and explicit.action == action else "missing_roast_request_evidence"
    if action == "humor.free_bit":
        explicit = classify_explicit_humor(cleaned)
        return None if explicit and explicit.action == action else "missing_free_bit_request_evidence"
    if action in {"performance.dj_bit", "performance.body_beat", "performance.mood_pose"}:
        explicit = classify_explicit_performance(cleaned)
        return None if explicit and explicit.action == action else "missing_performance_request_evidence"
    if action == "motion.explore":
        # A floor-seizing physical wander must not fire on an ambient LLM read of the
        # turn — require the deterministic imperative-invite classifier to agree.
        explicit = classify_explicit_exploration(cleaned)
        return None if explicit and explicit.action == action else "missing_explore_invite_evidence"
    if action == "character.preference_query":
        return (
            None
            if classify_explicit_character_preference(cleaned)
            else "missing_rex_preference_query_evidence"
        )
    if action == "identity.who_is_speaking":
        return None if _WHO_SPEAKING_RE.search(cleaned) else "missing_identity_query_evidence"
    if action == "music.play":
        return None if _MUSIC_PLAY_REQUEST_RE.search(cleaned) else "missing_music_play_evidence"
    if action == "music.stop":
        active_music = bool(context.get("active_music"))
        bare_stop = bool(re.match(r"^\s*(?:stop|pause)\s*$", cleaned, re.IGNORECASE))
        return (
            None
            if _MUSIC_STOP_REQUEST_RE.search(cleaned) or (active_music and bare_stop)
            else "missing_music_stop_evidence"
        )
    if action == "music.skip":
        return None if _MUSIC_SKIP_REQUEST_RE.search(cleaned) else "missing_music_skip_evidence"
    if action == "music.options":
        return None if _MUSIC_OPTIONS_REQUEST_RE.search(cleaned) else "missing_music_options_evidence"
    if action == "game.start":
        return None if _GAME_START_REQUEST_RE.search(cleaned) else "missing_game_start_evidence"
    if action == "game.stop":
        active_game = bool(context.get("active_game"))
        bare_stop = bool(re.match(r"^\s*(?:stop|quit|end)\s*$", cleaned, re.IGNORECASE))
        return (
            None
            if _GAME_STOP_REQUEST_RE.search(cleaned) or (active_game and bare_stop)
            else "missing_game_stop_evidence"
        )
    if action == "time.query":
        return None if _TIME_QUERY_RE.search(cleaned) else "missing_time_query_evidence"
    if action == "date.query":
        return None if _DATE_QUERY_RE.search(cleaned) else "missing_date_query_evidence"
    if action == "weather.query":
        return None if _WEATHER_QUERY_RE.search(cleaned) else "missing_weather_query_evidence"
    if action == "status.capabilities":
        return None if _CAPABILITIES_QUERY_RE.search(cleaned) else "missing_capabilities_query_evidence"
    if action == "status.uptime":
        return None if _UPTIME_QUERY_RE.search(cleaned) else "missing_uptime_query_evidence"
    if action == "status.battery":
        from intelligence.intent_classifier import _BATTERY_QUERY_RE
        return None if _BATTERY_QUERY_RE.search(cleaned) else "missing_battery_query_evidence"
    if action == "vision.describe_scene":
        return None if has_vision_query_evidence(cleaned) else "missing_vision_query_evidence"
    if action == "vision.directed_look":
        return None if has_directed_look_evidence(cleaned) else "missing_directed_look_evidence"
    if action == "system.sleep":
        return (
            None
            if re.match(
                r"^\s*(?:go\s+to\s+sleep|sleep|wake\s+up|resume(?:\s+talking)?|"
                r"talk\s+again|speak\s+again|stop\s+being\s+quiet|"
                r"exit\s+quiet\s+mode|be\s+quiet|quiet\s+mode|go\s+quiet|"
                r"shut\s*down|shutdown|power\s+off|turn\s+off)\s*$",
                cleaned,
                re.IGNORECASE,
            )
            else "missing_system_mode_evidence"
        )

    return None


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
    if action == "vision.snapshot":
        requires_confirmation = True
    if action == "memory.forget_specific" and not str(args.get("target") or "").strip():
        confidence = min(confidence, 0.45)
    if action == "identity.name_correction":
        raw_name = str(
            args.get("name")
            or args.get("new_name")
            or args.get("person_name")
            or ""
        ).strip()
        cleaned_name = _clean_name_arg(raw_name)
        if cleaned_name:
            args = dict(args)
            args["name"] = cleaned_name
    if action == "performance.body_beat":
        raw_beat = str(
            args.get("body_beat")
            or args.get("beat")
            or args.get("gesture")
            or args.get("pose")
            or ""
        ).strip()
        canonical = performance_plan.canonical_body_beat(raw_beat)
        if canonical:
            args = dict(args)
            args["body_beat"] = canonical
        else:
            confidence = min(confidence, 0.45)
    if action == "performance.mood_pose":
        raw_mood = str(
            args.get("mood")
            or args.get("emotion")
            or args.get("pose")
            or ""
        ).strip()
        canonical = performance_plan.canonical_mood_pose(raw_mood)
        if canonical:
            args = dict(args)
            args["mood"] = canonical
        else:
            confidence = min(confidence, 0.45)
    if action == "character.preference_query":
        parsed = rex_preferences.extract_preference_query(
            str(args.get("text") or args.get("utterance") or "")
        )
        if parsed:
            merged = dict(parsed)
            merged.update(args)
            args = merged
        topic = str(args.get("topic") or args.get("domain") or "").strip()
        options = args.get("options") or []
        if not topic and not options:
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
    dialogue = (context or {}).get("dialogue_act") or {}
    if isinstance(dialogue, dict) and dialogue.get("label") == "answer_to_rex":
        blocked = {
            str(item)
            for item in (dialogue.get("blocked_actions") or [])
            if str(item).strip()
        }
        if decision.action in blocked:
            return ActionDecision(
                action="conversation.reply",
                confidence=min(float(decision.confidence or 0.0), 0.40),
                args={},
                requires_confirmation=False,
                reason="dialogue act says utterance is a reply to Rex",
            )

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
        decision.action == "memory.recent_discard"
        and not _RECENT_DISCARD_REQUEST_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="recent discard requires an explicit do-not-store/forget-that request",
        )

    if (
        decision.action == "vision.snapshot"
        and _HUMAN_VISUAL_PLAN_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="first-person photo/visual-memory plan is not a command to Rex",
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
        if topic and not references_person_memory_target(topic):
            return ActionDecision(
                action="conversation.reply",
                confidence=min(float(decision.confidence or 0.0), 0.40),
                args={},
                requires_confirmation=False,
                reason="general topic knowledge question should use LLM conversation",
            )

    if (
        decision.action == "date.query"
        and _NAMED_DAY_EXPLANATION_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="named holiday explanation is not a current date query",
        )

    if (
        decision.action == "identity.name_correction"
        and _EVENT_CANCEL_OR_STALE_RE.search(text or "")
        and not _text_has_identity_name_correction_content(text or "", decision)
    ):
        return ActionDecision(
            action="event.cancel",
            confidence=min(max(float(decision.confidence or 0.0), 0.80), 0.92),
            args={},
            requires_confirmation=False,
            reason="plan/status retraction is not an identity name correction",
        )

    if (
        decision.action == "event.cancel"
        and _EVENT_CONTINUATION_STATUS_RE.search(text or "")
        and not _EVENT_CANCEL_OR_STALE_RE.search(text or "")
    ):
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="ongoing status update is not an event cancellation",
        )

    if decision.action == "identity.introduce_person":
        introduced_names = {
            " ".join(str(decision.args.get(key) or "").strip().lower().split())
            for key in ("name", "person_name", "new_person_name")
            if str(decision.args.get(key) or "").strip()
        }
        if _PRONOUN_ONLY_INTRO_RE.match(text or "") or bool(introduced_names & {
            "me",
            "you",
            "us",
            "me and you",
            "you and me",
        }):
            return ActionDecision(
                action="conversation.reply",
                confidence=min(float(decision.confidence or 0.0), 0.40),
                args={},
                requires_confirmation=False,
                reason="pronoun-only fragment is not a person introduction",
            )
        if (
            _NAMED_PERSON_FACT_STATEMENT_RE.match(text or "")
            and not _NAMED_RELATION_INTRO_RE.search(text or "")
        ):
            return ActionDecision(
                action="conversation.reply",
                confidence=min(float(decision.confidence or 0.0), 0.40),
                args={},
                requires_confirmation=False,
                reason="named third-person fact should be learned as conversation",
            )

    active_game = bool((context or {}).get("active_game"))
    if decision.action == "game.answer" and not active_game:
        if _RELATIONSHIP_SCORE_QUERY_RE.search(text or ""):
            return ActionDecision(
                action="memory.query",
                confidence=min(max(float(decision.confidence or 0.0), 0.85), 0.95),
                args={},
                requires_confirmation=False,
                reason="score question outside an active game is a relationship memory query",
            )
        return ActionDecision(
            action="conversation.reply",
            confidence=min(float(decision.confidence or 0.0), 0.40),
            args={},
            requires_confirmation=False,
            reason="game answer requires an active game",
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


# Action-domain cue words: any hit keeps the LLM router in the loop (the utterance is
# plausibly actionable even though the explicit regexes missed). Deliberately GENEROUS --
# a false cue just costs the old ~0.8s router call; a missed cue on a fuzzy action
# phrase falls through to conversation (recoverable: the user re-asks explicitly, and
# canonical command forms are still caught by the explicit classifiers above).
_ACTION_CUE_RE = re.compile(
    r"\b(look|watch|see|turn|spin|rotate|move|drive|roll|come|follow|stop|halt|freeze|"
    r"forward|backward|back ?up|"
    r"play|pause|skip|song|music|sing|dance|dj|beat|pose|volume|louder|quieter|softer|mute|"
    r"game|trivia|jeopardy|twenty questions|quiz|guess|"
    r"joke|roast|impression|bit|"
    r"remember|forget|memory|memories|recall|"
    r"who'?s|who is|name|introduce|"
    r"picture|photo|snapshot|describe|camera|scene|"
    r"sleep|wake|shut ?down|power|capabilit\w*|uptime|"
    r"weather|forecast|temperature|time|date|day|o'?clock|"
    r"cancel|favorite|favourite)\b",
    re.IGNORECASE,
)


# Deterministic self-knowledge intents that the intent classifier answers from
# real local data (clock, wttr cache, uptime, capability list, biometric speaker
# ID). For these the LLM router can only ever agree ("What day is it?" burned a
# 0.91s routing call to return conversation.reply at 0.00 and was discarded,
# live-logged 2026-08-02 13:03). Excluded on purpose: play_music /
# query_music_options (router owns args + pending favorite_music override),
# query_memory (forget/boundary disambiguation), query_what_do_you_see (vision
# evidence + consent rules).
# Maps each skippable intent to its stable action key so the router's own
# evidence regexes can vet the claim (keep in sync with the same entries in
# interaction._INTENT_ACTION_MAP). query_games has no action mapping and no
# evidence rule — its deterministic regex is the whole claim.
_SELF_QUERY_SKIP_INTENTS = {
    "query_time": "time.query",
    "query_date": "date.query",
    "query_weather": "weather.query",
    "query_uptime": "status.uptime",
    "query_battery": "status.battery",
    "query_capabilities": "status.capabilities",
    "query_games": None,
    "query_who_is_speaking": "identity.who_is_speaking",
}


def _deterministic_self_query_intent(text: str, context: dict[str, Any]) -> str | None:
    """Return the deterministic self-knowledge intent claiming this turn, if any.

    Active games keep full routing: Jeopardy answers are phrased "what is ..."
    and could regex-match a query (game.answer must win those turns). The claim
    must also pass the router's evidence regexes — the intent classifier's
    patterns are looser ("something about the weather maybe" classifies as
    query_weather but is not a weather question), and the downstream execution
    gate would block exactly the same way.
    """
    if not bool(getattr(config, "ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED", True)):
        return None
    if context.get("active_game"):
        return None
    try:
        from intelligence import intent_classifier
        intent = intent_classifier.classify_deterministic(text)
    except Exception:
        return None
    if intent not in _SELF_QUERY_SKIP_INTENTS:
        return None
    action = _SELF_QUERY_SKIP_INTENTS[intent]
    if action is not None:
        evidence_reason = missing_required_evidence_reason(
            text,
            ActionDecision(action=action, confidence=0.94),
            context=context,
        )
        if evidence_reason:
            return None
    return intent


def _clearly_conversational(text: str, context: dict[str, Any]) -> bool:
    """True when this turn is deterministically plain conversation -- safe to skip the
    LLM routing call (~0.8s, the single largest fixed cost on chat turns, measured
    2026-07-06). Requires ALL of: no action-domain cue word, no active game/music
    (mid-game answers and bare 'stop' must keep full routing), and the deterministic
    intent classifier agreeing it's 'general'."""
    if not bool(getattr(config, "ACTION_ROUTER_DETERMINISTIC_SKIP_ENABLED", True)):
        return False
    if context.get("active_game") or context.get("active_music"):
        return False
    if _ACTION_CUE_RE.search(text):
        return False
    try:
        from intelligence import intent_classifier
        if intent_classifier.classify_deterministic(text) != "general":
            return False
    except Exception:
        return False
    return True


def decide(text: str, context: dict[str, Any] | None = None) -> ActionDecision:
    """Return the router's best action decision for this turn."""
    if not text or not text.strip():
        return ActionDecision(reason="empty utterance")

    context = context or {}
    # Direct shutdown/sleep requests route deterministically — the LLM router
    # scored "I will talk to you later, and I would like you to shut down." as
    # conversation (0.20), the closure-cue agenda took over, and the reply model
    # generated "Powering down." as a FAREWELL QUIP without powering down
    # (field 2026-08-03 00:05). command_parser owns the safety guards
    # (negation, object-scoped "shut down the music", hypotheticals).
    try:
        from intelligence import command_parser as _cp
        if _cp.is_shutdown_request(text):
            return _apply_context_overrides(
                ActionDecision(
                    action="system.shutdown",
                    confidence=0.95,
                    reason="deterministic: direct shutdown request",
                ),
                text,
                context,
            )
    except Exception as exc:
        _log.debug("[action_router] shutdown pre-pass failed: %s", exc)
    explicit_control = classify_explicit_control(text)
    if explicit_control is not None:
        return _apply_context_overrides(explicit_control, text, context)
    explicit_humor = classify_explicit_humor(text)
    if explicit_humor is not None:
        return _apply_context_overrides(explicit_humor, text, context)
    explicit_performance = classify_explicit_performance(text)
    if explicit_performance is not None:
        return _apply_context_overrides(explicit_performance, text, context)
    explicit_character_preference = classify_explicit_character_preference(text)
    if explicit_character_preference is not None:
        return _apply_context_overrides(explicit_character_preference, text, context)

    self_query_intent = _deterministic_self_query_intent(text, context)
    if self_query_intent is not None:
        _log.info(
            "[action_router] deterministic self-query skip -- intent=%s, "
            "LLM routing call saved",
            self_query_intent,
        )
        # conversation.reply falls through to the intent classifier, which
        # executes the same handler the router's executable action would have —
        # identical final path (intent_classifier.<intent>), minus the LLM call.
        return _apply_context_overrides(
            ActionDecision(
                action="conversation.reply",
                confidence=0.6,
                reason=f"deterministic: self-query {self_query_intent}; "
                       "intent classifier owns it",
            ),
            text,
            context,
        )

    if _clearly_conversational(text, context):
        _log.info(
            "[action_router] deterministic conversational skip -- no action cues, "
            "LLM routing call saved"
        )
        return _apply_context_overrides(
            ActionDecision(
                action="conversation.reply",
                confidence=0.6,
                reason="deterministic: conversational, no action cues",
            ),
            text,
            context,
        )

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
        # llm_compat translates the GPT-5-family param contract (max_completion_tokens,
        # temperature dropped, reasoning_effort) — a no-op for classic models like
        # gpt-4o-mini, so rollback via ACTION_ROUTER_MODEL alone stays valid.
        from intelligence import llm_compat
        resp = llm_compat.create(
            _client,
            model=getattr(config, "ACTION_ROUTER_MODEL", config.LLM_MODEL),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=240,
            reasoning_effort=str(getattr(config, "ACTION_ROUTER_REASONING_EFFORT", "none")),
        )
        raw = resp.choices[0].message.content or ""
        payload = json.loads(_strip_code_fence(raw))
        return _apply_context_overrides(_coerce_decision(payload), text, context)
    except Exception as exc:
        _log.debug("[action_router] decision failed: %s", exc)
        return ActionDecision(reason=f"router error: {type(exc).__name__}")


def warmup() -> bool:
    """Open the action-router's OpenAI connection pool (a separate client from
    llm._client) so the first ambiguous turn doesn't pay cold TLS / HTTP setup.
    """
    try:
        from intelligence import llm_compat
        # max_tokens=16, not 1: GPT-5-family models 400 on a cap they cannot
        # finish within ("Could not finish the message...") instead of
        # truncating like gpt-4o-mini does.
        llm_compat.create(
            _client,
            model=getattr(config, "ACTION_ROUTER_MODEL", config.LLM_MODEL),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=16,
            reasoning_effort=str(getattr(config, "ACTION_ROUTER_REASONING_EFFORT", "none")),
        )
        _log.info("[action_router] OpenAI connection warmed")
        return True
    except Exception as exc:
        _log.debug("[action_router] OpenAI warmup failed (non-fatal): %s", exc)
        return False


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
