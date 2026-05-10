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
_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


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
        "system.sleep",
        "system",
        "User asks Rex to sleep, wake, quiet down, mute, shut down, or power off.",
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
  dramatic_visor_peek, tiny_victory_dance. Do not use it for ordinary
  "look at this" vision requests.
- Use performance.mood_pose for emotion-driven physical acting requests such as
  "act embarrassed", "look annoyed", or "look proud". Put one of these exact
  mood names in args.mood: agreement, disagreement, disbelief, disgusted,
  embarrassed, annoyed, angry, proud, suspicious, thinking, happy, giddy,
  sad, surprised, offended.
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
    r"^\s*(?:please\s+)?(?:play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue|turn\s+on)\b|"
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
    r"\b(?:what(?:'s| is)?|tell me|give me|do you know)\b.{0,35}\b"
    r"(?:date|day|today|weekday)\b|\bwhat day is it\b",
    re.IGNORECASE,
)
_WEATHER_QUERY_RE = re.compile(
    r"\b(?:what(?:'s| is)|tell me|give me|do you know)\b.{0,35}\b"
    r"(?:weather|temperature|forecast|raining|hot|cold|outside)\b|"
    r"\b(?:weather|temperature)\s+(?:forecast|outside)\b|"
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
_VISION_DESCRIBE_RE = re.compile(
    r"\b(?:what do you see|what can you see|look around|describe (?:the )?"
    r"(?:room|scene)|what am i holding|what's in front of you)\b",
    re.IGNORECASE,
)
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
    if action == "vision.describe_scene":
        return None if _VISION_DESCRIBE_RE.search(cleaned) else "missing_vision_query_evidence"
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


def decide(text: str, context: dict[str, Any] | None = None) -> ActionDecision:
    """Return the router's best action decision for this turn."""
    if not text or not text.strip():
        return ActionDecision(reason="empty utterance")

    context = context or {}
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
