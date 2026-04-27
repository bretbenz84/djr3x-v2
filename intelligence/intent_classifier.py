"""
intelligence/intent_classifier.py — Fast intent classification for the LLM
fallback path in interaction.py.

The command parser uses exact / fuzzy string matching, which misses natural
phrasing like "hey what time is it again" or "so what can you actually do".
This module sits between the parser and the full LLM call: a single tiny
GPT-4o-mini request returns one of a fixed set of intent labels (or
'general'), so the interaction loop can answer self-knowledge questions
locally with real data instead of letting Rex hallucinate over them.
"""

import logging
import re

import config
import apikeys
from openai import OpenAI

_log = logging.getLogger(__name__)

_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)

_VALID_INTENTS = {
    "query_time",
    "query_date",
    "query_weather",
    "query_games",
    "query_capabilities",
    "query_uptime",
    "query_what_do_you_see",
    "query_who_is_speaking",
    "query_memory",
    "play_music",
    "query_music_options",
    "general",
}

_MUSIC_OPTION_CONTEXT_RE = re.compile(
    r"\b("
    r"music|songs?|tracks?|albums?|artists?|bands?|playlists?|stations?|radio|"
    r"genres?|dj|tunes?"
    r")\b",
    re.IGNORECASE,
)

_MUSIC_PLAY_ACTION_RE = re.compile(
    r"\b("
    r"play|start\s+playing|put\s+on|throw\s+on|spin|queue|cue|turn\s+on"
    r")\b",
    re.IGNORECASE,
)

_NON_MUSIC_PLAY_CONTEXT_RE = re.compile(
    r"\b("
    r"game|games|trivia|i\s+spy|20\s+questions|twenty\s+questions|"
    r"jeopardy|word\s+association|movie|video|clip"
    r")\b",
    re.IGNORECASE,
)

_MUSIC_NEGATION_RE = re.compile(
    r"\b("
    r"didn['’]?t|did\s+not|don['’]?t|do\s+not|wasn['’]?t|was\s+not|"
    r"isn['’]?t|is\s+not|not|no"
    r")\b.{0,40}\b("
    r"music|songs?|tracks?|albums?|artists?|bands?|playlists?|stations?|radio|"
    r"genres?|dj|tunes?"
    r")\b",
    re.IGNORECASE,
)

_MUSIC_VIBE_FALLBACKS = {
    "ambient", "chill", "relaxing", "downtempo", "mellow", "space",
    "electronic", "house", "dance", "jazz", "rock", "metal", "reggae",
    "country", "folk", "hiphop", "hip hop", "lofi", "lo-fi", "classical",
    "synthpop", "pop", "lounge", "retro", "upbeat", "quiet",
}


def _known_music_vibes() -> set[str]:
    vibes = set(_MUSIC_VIBE_FALLBACKS)
    for station in getattr(config, "RADIO_STATIONS", []) or []:
        for vibe in station.get("vibes") or []:
            vibe = str(vibe).strip().lower()
            if vibe:
                vibes.add(vibe)
    return vibes


def _contains_known_music_vibe(text: str) -> bool:
    normalized = " ".join(re.sub(r"[^a-z0-9\s-]", " ", text.lower()).split())
    return any(
        re.search(rf"\b{re.escape(vibe)}\b", normalized)
        for vibe in _known_music_vibes()
    )


def _music_intent_allowed(text: str, label: str) -> bool:
    """Deterministic guardrail for high-impact music intents.

    The classifier has no conversation context, so labels like
    query_music_options must require explicit music wording. This prevents
    generic closure/correction turns such as "nevermind" from dumping the DJ
    genre list.
    """
    if label == "query_music_options":
        if _MUSIC_NEGATION_RE.search(text):
            return False
        return bool(_MUSIC_OPTION_CONTEXT_RE.search(text))

    if label == "play_music":
        if _MUSIC_NEGATION_RE.search(text):
            return False
        if not _MUSIC_PLAY_ACTION_RE.search(text):
            return False
        if _NON_MUSIC_PLAY_CONTEXT_RE.search(text) and not _MUSIC_OPTION_CONTEXT_RE.search(text):
            return False
        return bool(
            _MUSIC_OPTION_CONTEXT_RE.search(text)
            or _contains_known_music_vibe(text)
            or re.search(r"\bsomething\b", text, re.IGNORECASE)
        )

    return True


_PROMPT_TEMPLATE = (
    'Classify this input into exactly one category. Reply with only the '
    'category name. Categories: query_time, query_date, query_weather, query_games, '
    'query_capabilities, query_uptime, query_what_do_you_see, '
    'query_who_is_speaking, query_memory, play_music, query_music_options, general. '
    'Note: query_time covers clock-time questions like "what time is it?" '
    'or "tell me the time". query_date covers date/day questions like '
    '"what day is it?", "what is today?", "tell me today\'s date". '
    'Note: query_who_is_speaking covers "who\'s speaking?", "who am I?", '
    '"do you know who I am?", "you know who that is?", "can you tell who I am?". '
    'Note: query_memory covers requests to recall stored memory about a person '
    'or relationship, e.g. "tell me what you know about me", "what do you '
    'remember about myself?", "tell me about my partner", "what do you know '
    'about Jeff?", "what have I told you about Exudica?". Do NOT use '
    'query_memory for immediate identity recognition like "who am I?" or '
    '"do you know who is speaking?" — those are query_who_is_speaking. '
    'Note: play_music covers any request to play music, a song, a track, an '
    'artist, a genre, a vibe, or a radio station — e.g. "play some jazz", '
    '"can you play jazz music?", "put on something chill", "play me a song", '
    '"play the Beatles", "throw on some lo-fi". Do NOT use play_music for '
    'games such as Jeopardy, Trivia, I Spy, 20 Questions, or Word Association; '
    'volume / skip / stop controls are general. '
    'Note: query_music_options covers asking what music is available, e.g. '
    '"what kind of music can you play?", "what genres do you have?", '
    '"what stations can you play?". Do NOT classify "what can you play?" as '
    'music unless the input explicitly says music, songs, stations, radio, '
    'genres, artist, track, or playlist. Closure/correction phrases like '
    '"nevermind", "no", "that was not about music" are general. '
    'Input: "{text}"'
)


def classify(text: str) -> str:
    """Return one of _VALID_INTENTS for the given user utterance.

    Falls back to 'general' on any error or unrecognized label so a misfire
    never blocks the normal LLM path.
    """
    if not text or not text.strip():
        return "general"

    try:
        resp = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(text=text)}],
            temperature=0,
            max_tokens=20,
        )
        label = (resp.choices[0].message.content or "").strip().lower()
    except Exception as exc:
        _log.debug("intent_classifier classify failed: %s", exc)
        return "general"

    # Tolerate stray punctuation / quotes from the model.
    label = label.strip(' "\'.`')
    if label in _VALID_INTENTS:
        if label in {"play_music", "query_music_options"} and not _music_intent_allowed(text, label):
            _log.info(
                "[intent_classifier] overriding %s → general; no explicit music intent in %r",
                label,
                text,
            )
            return "general"
        return label

    for candidate in _VALID_INTENTS:
        if candidate in label:
            if (
                candidate in {"play_music", "query_music_options"}
                and not _music_intent_allowed(text, candidate)
            ):
                _log.info(
                    "[intent_classifier] overriding %s → general; no explicit music intent in %r",
                    candidate,
                    text,
                )
                return "general"
            return candidate

    return "general"
