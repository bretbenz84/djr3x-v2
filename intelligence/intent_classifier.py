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

import config
import apikeys
from openai import OpenAI

_log = logging.getLogger(__name__)

_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)

_VALID_INTENTS = {
    "query_time",
    "query_weather",
    "query_games",
    "query_capabilities",
    "query_uptime",
    "query_what_do_you_see",
    "query_who_is_speaking",
    "general",
}

_PROMPT_TEMPLATE = (
    'Classify this input into exactly one category. Reply with only the '
    'category name. Categories: query_time, query_weather, query_games, '
    'query_capabilities, query_uptime, query_what_do_you_see, '
    'query_who_is_speaking, general. '
    'Note: query_who_is_speaking covers "who\'s speaking?", "who am I?", '
    '"do you know who I am?", "you know who that is?", "can you tell who I am?". '
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
        return label

    for candidate in _VALID_INTENTS:
        if candidate in label:
            return candidate

    return "general"
