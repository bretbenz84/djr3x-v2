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
from intelligence import performance_plan
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
- If the utterance says a remembered plan is no longer happening, use event.cancel.
- For event.cancel, put the plan/topic being canceled in args.event_hint when possible.
- Only use emotional.boundary when the user explicitly asks not to talk about,
  ask about, mention, or bring up a topic. A bare health/sad topic like "back pain"
  is conversation.reply unless the user says not to discuss it.
- Use conversation.repair when the user corrects Rex, says Rex misunderstood, or
  asks Rex to try that again. Do not use it for ordinary topic disagreement.
- Use identity.name_correction when the user corrects who Rex thinks is speaking
  or what to call the current speaker, e.g. "that's not Bret, I'm Daniel" or
  "call me JT". Put the corrected name in args.name when present. Use
  conversation.repair if the correction has no identity/name content.
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
  suspicious_glance, proud_dj_pose, offended_recoil, thinking_tilt,
  dramatic_visor_peek, tiny_victory_dance. Do not use it for ordinary
  "look at this" vision requests.
- Use performance.mood_pose for emotion-driven physical acting requests such as
  "act embarrassed", "look annoyed", or "look proud". Put one of these exact
  mood names in args.mood: embarrassed, annoyed, proud, suspicious, thinking,
  happy, offended.
- Use vision.snapshot for privacy-sensitive requests to remember or save what
  Rex sees, such as "remember what you see" or "take a look and keep that in
  mind". Set requires_confirmation=true. Do not use it for ordinary "what do
  you see?" questions; those are vision.describe_scene.
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
_PERSON_MEMORY_QUERY_RE = re.compile(
    r"\b("
    r"me|myself|me\?|my\s+|mine|i\s+told\s+you|i'?ve\s+told\s+you|"
    r"remember|memory|memories|person|people|friend|partner|wife|husband|"
    r"mom|mother|dad|father|brother|sister|kid|child|son|daughter|"
    r"jeff|joy|jt|bret"
    r")\b",
    re.IGNORECASE,
)
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
        return ActionDecision(
            action="identity.name_correction",
            confidence=0.95,
            args={"name": name} if name else {},
            reason="explicit speaker name correction",
        )

    if _VISION_SNAPSHOT_RE.search(cleaned):
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
    explicit_control = classify_explicit_control(text)
    if explicit_control is not None:
        return _apply_context_overrides(explicit_control, text, context)
    explicit_humor = classify_explicit_humor(text)
    if explicit_humor is not None:
        return _apply_context_overrides(explicit_humor, text, context)
    explicit_performance = classify_explicit_performance(text)
    if explicit_performance is not None:
        return _apply_context_overrides(explicit_performance, text, context)

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
