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
from intelligence import local_llm
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
_MUSIC_OPTIONS_REQUEST_RE = re.compile(
    r"\b("
    r"(?:what|which)\s+(?:kind|kinds|type|types|sort|sorts)\s+of\s+"
    r"(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)|"
    r"(?:what|which)\s+(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)\s+"
    r"(?:can|could|do|are|have|you)|"
    r"(?:what|which)\s+(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)"
    r".{0,40}\b(?:play|have|available|options|offer|do)|"
    r"(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)"
    r".{0,40}\b(?:can you play|do you have|available|options|offer)|"
    r"(?:can|could)\s+you\s+(?:play|do)\s+"
    r"(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)|"
    r"(?:list|tell me|show me)\s+.{0,20}"
    r"(?:music|songs?|tracks?|stations?|radio|genres?|playlists?)"
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
_MUSIC_PREFERENCE_QUESTION_RE = re.compile(
    r"\b("
    r"(?:what|which|any)\s+(?:favorite|favourite)\s+"
    r"(?:music|songs?|tracks?|albums?|artists?|bands?|genres?|playlists?)|"
    r"(?:favorite|favourite)\s+"
    r"(?:music|songs?|tracks?|albums?|artists?|bands?|genres?|playlists?)|"
    r"(?:music|songs?|tracks?|albums?|artists?|bands?|genres?|playlists?)"
    r".{0,30}\b(?:you|she|he|they)\s+(?:like|enjoy|prefer|love|spin)"
    r")\b",
    re.IGNORECASE,
)

_MUSIC_VIBE_FALLBACKS = {
    "ambient", "chill", "relaxing", "downtempo", "mellow", "space",
    "electronic", "house", "dance", "jazz", "rock", "metal", "reggae",
    "country", "folk", "hiphop", "hip hop", "lofi", "lo-fi", "classical",
    "synthpop", "pop", "lounge", "retro", "upbeat", "quiet",
}
_TOPIC_KNOWLEDGE_QUERY_RE = re.compile(
    r"\b(?:what\s+do\s+you\s+know|do\s+you\s+know\s+anything|"
    r"tell\s+me|explain)\s+(?:about\s+)?(?P<topic>[^?.,!;]{3,100})",
    re.IGNORECASE,
)
_PERSON_MEMORY_QUERY_RE = re.compile(
    r"\b("
    r"me|myself|my\s+|mine|i\s+told\s+you|i'?ve\s+told\s+you|"
    r"remember|memory|memories|person|people|friend|partner|wife|husband|"
    r"mom|mother|dad|father|brother|sister|kid|child|son|daughter|"
    r"jeff|joy|jt|bret"
    r")\b",
    re.IGNORECASE,
)
_BARE_TOPIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '&:-]{2,60}$")

_TIME_QUERY_RE = re.compile(
    r"\b(what(?:'s| is)?|tell me|give me|do you know)\b.{0,30}\b(time|clock)\b|"
    r"\b(time|clock)\b.{0,20}\b(now|is it)\b",
    re.IGNORECASE,
)
_DATE_QUERY_RE = re.compile(
    r"\b(what(?:'s| is)?|tell me|give me|do you know)\b.{0,35}\b("
    r"date|day|today|weekday"
    r")\b|"
    r"\bwhat day is it\b|"
    r"\bwhat day\b.{0,35}\b(dealing with|today|is it|are we|we are)\b",
    re.IGNORECASE,
)
_WEATHER_QUERY_RE = re.compile(
    r"\b(weather|temperature|forecast|rain|raining|hot|cold|outside)\b",
    re.IGNORECASE,
)
_GAMES_QUERY_RE = re.compile(
    r"\b(what games|which games|games can you|play a game|start a game|"
    r"trivia|jeopardy|i spy|20 questions|twenty questions|word association)\b",
    re.IGNORECASE,
)
_CAPABILITIES_QUERY_RE = re.compile(
    r"\b(what can you do|what are you capable of|capabilities|"
    r"what do you do|what (?:sort|kind) of (?:stuff|things) are you good for|"
    r"what are you good (?:for|at)|what are you useful for|what can i ask you|"
    r"what should i ask you|help me|commands)\b",
    re.IGNORECASE,
)
_UPTIME_QUERY_RE = re.compile(
    r"\b(how long have you been|how long are you|uptime|been running|"
    r"been awake|when did you start)\b",
    re.IGNORECASE,
)
_VISION_QUERY_RE = re.compile(
    r"\b(what do you see|what can you see|look at|take a look|"
    r"what am i holding|describe (?:the )?(room|scene)|see me)\b",
    re.IGNORECASE,
)
_WHO_QUERY_RE = re.compile(
    r"\b(who am i|who'?s speaking|who is speaking|do you know who i am|"
    r"recognize (?:me|my voice)|can you tell who i am|"
    r"identify whoever is talking|identify who(?:ever)? is talking|"
    r"who(?:ever)? is talking right now)\b",
    re.IGNORECASE,
)
_CLOSURE_RE = re.compile(
    r"\b("
    r"bye|goodbye|good-bye|see you|see ya|talk to you later|talk later|"
    r"catch you later|later|nice speaking|nice talking|that'?s all|"
    r"that is all|never ?mind|forget it|we can stop|let'?s stop"
    r")\b",
    re.IGNORECASE,
)
_CONTEXTUAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:what|how|and)\s+(?:about|bout)\s+[^?!.]{2,80}\??\s*$",
    re.IGNORECASE,
)
_MEMORY_SELF_QUERY_RE = re.compile(
    r"\b("
    r"tell me about myself|tell me about me|what do you know about me|"
    r"what do you remember about me|what do you remember about myself|"
    r"tell me what you know about me|tell me what you remember about me|"
    r"what are my plans|what(?:'s| is) my plan|what do i have planned|"
    r"what am i doing"
    r")\b",
    re.IGNORECASE,
)


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
        return bool(
            _MUSIC_OPTION_CONTEXT_RE.search(text)
            and _MUSIC_OPTIONS_REQUEST_RE.search(text)
        )

    if label == "play_music":
        if _MUSIC_NEGATION_RE.search(text):
            return False
        if _MUSIC_PREFERENCE_QUESTION_RE.search(text):
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


def _deterministic_label(text: str) -> str:
    """Cheap rules for common routing intents.

    The old classifier made an OpenAI call even for ordinary chat that was
    almost certainly going to return "general". These rules preserve the
    latency-sensitive local handlers for obvious requests and let everything
    else fall straight through to the main response path.
    """
    if not text or not text.strip():
        return "general"

    cleaned = " ".join(text.strip().split())
    if _CLOSURE_RE.search(cleaned):
        return "general"
    if _CONTEXTUAL_FOLLOWUP_RE.match(cleaned):
        return "general"
    if _MEMORY_SELF_QUERY_RE.search(cleaned):
        return "query_memory"
    if _WHO_QUERY_RE.search(cleaned):
        return "query_who_is_speaking"
    if _VISION_QUERY_RE.search(cleaned):
        return "query_what_do_you_see"
    if _UPTIME_QUERY_RE.search(cleaned):
        return "query_uptime"
    if _CAPABILITIES_QUERY_RE.search(cleaned):
        return "query_capabilities"
    if _DATE_QUERY_RE.search(cleaned):
        return "query_date"
    if _TIME_QUERY_RE.search(cleaned):
        return "query_time"
    if _WEATHER_QUERY_RE.search(cleaned):
        return "query_weather"
    if _GAMES_QUERY_RE.search(cleaned):
        return "query_games"
    if re.search(
        r"\b(anything by|something by)\b",
        cleaned,
        re.IGNORECASE,
    ) and _contains_known_music_vibe(cleaned):
        return "play_music"
    if _music_intent_allowed(cleaned, "query_music_options"):
        return "query_music_options"
    if _music_intent_allowed(cleaned, "play_music"):
        return "play_music"
    if re.search(
        r"\b(what do you remember|what do you know about me|"
        r"tell me what you know about|remember about)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return "query_memory"
    return "general"


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

_LOCAL_SYSTEM_PROMPT = (
    "You are a strict intent classifier. Return exactly one category name and "
    "nothing else. Never explain your choice."
)


def classify(text: str) -> str:
    """Return one of _VALID_INTENTS for the given user utterance.

    Falls back to 'general' on any error or unrecognized label so a misfire
    never blocks the normal LLM path.
    """
    if not text or not text.strip():
        return "general"

    cleaned = " ".join(text.strip().split())
    label = _deterministic_label(cleaned)
    if label != "general":
        return label

    topic_match = _TOPIC_KNOWLEDGE_QUERY_RE.search(cleaned)
    topic = (topic_match.group("topic") if topic_match else "").strip()
    if topic and not _PERSON_MEMORY_QUERY_RE.search(topic):
        return "general"

    if (
        _BARE_TOPIC_RE.match(cleaned)
        and "?" not in cleaned
        and 2 <= len(re.findall(r"[A-Za-z0-9']+", cleaned)) <= 5
        and not _MUSIC_PLAY_ACTION_RE.search(cleaned)
    ):
        return "general"
    if not bool(getattr(config, "INTENT_CLASSIFIER_LLM_FALLBACK_ENABLED", True)):
        return "general"

    try:
        label = _classify_with_llm(cleaned)
    except Exception as exc:
        _log.debug("intent_classifier classify failed: %s", exc)
        return "general"

    # Tolerate stray punctuation / quotes from the model.
    label = label.strip(' "\'.`')
    if label in _VALID_INTENTS:
        if label != "general" and _llm_label_blocked(cleaned, label):
            _log.info(
                "[intent_classifier] overriding %s → general; deterministic guard for %r",
                label,
                text,
            )
            return "general"
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
            if candidate != "general" and _llm_label_blocked(cleaned, candidate):
                _log.info(
                    "[intent_classifier] overriding %s → general; deterministic guard for %r",
                    candidate,
                    text,
                )
                return "general"
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


def _llm_label_blocked(text: str, label: str) -> bool:
    """Reject feature intents the fallback LLM often invents for ordinary chat."""
    if _CLOSURE_RE.search(text):
        return True
    if _CONTEXTUAL_FOLLOWUP_RE.match(text):
        return True
    if label == "query_date" and not _DATE_QUERY_RE.search(text):
        return True
    if label == "query_time" and not _TIME_QUERY_RE.search(text):
        return True
    if label == "query_weather" and not _WEATHER_QUERY_RE.search(text):
        return True
    if label == "query_games" and not _GAMES_QUERY_RE.search(text):
        return True
    if label == "query_capabilities" and not _CAPABILITIES_QUERY_RE.search(text):
        return True
    if label == "query_what_do_you_see" and not _VISION_QUERY_RE.search(text):
        return True
    if label == "query_who_is_speaking" and not _WHO_QUERY_RE.search(text):
        return True
    return False


def _classify_with_llm(text: str) -> str:
    backend = str(getattr(config, "INTENT_CLASSIFIER_LLM_BACKEND", "ollama")).lower()
    prompt = _PROMPT_TEMPLATE.format(text=text)

    if backend == "ollama":
        return local_llm.generate(
            prompt,
            system=_LOCAL_SYSTEM_PROMPT,
            temperature=0,
            max_tokens=8,
            timeout_secs=float(getattr(config, "INTENT_CLASSIFIER_LOCAL_TIMEOUT_SECS", 0.75)),
        ).strip().lower()

    resp = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=20,
        timeout=float(getattr(config, "INTENT_CLASSIFIER_OPENAI_TIMEOUT_SECS", 1.5)),
    )
    return (resp.choices[0].message.content or "").strip().lower()
