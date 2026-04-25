"""
awareness/address_mode.py — Address-mode classification for DJ-R3X.

Distinguishes utterances that mention Rex into:
    direct_address  — spoken TO Rex (respond normally)
    referential     — spoken ABOUT Rex to a third party ("Rex is so fun")
    instructional   — Rex is the OBJECT of an instruction ("say hi to Rex")
    unrelated       — keyword present but not actually about Rex

Pipeline:
  1. Cheap keyword pre-filter — if text has no rex-keyword, return direct_address
     immediately (the caller decides whether to even invoke this module).
  2. Hard rules — common patterns we can label without an LLM call.
  3. LLM fallback — single tiny GPT-4o-mini call (~3-token output) when
     hard rules don't fire. Returns 'direct_address' on any error.

Sentiment is co-classified by the LLM call when it's invoked: positive,
neutral, or negative. The consciousness loop uses this to bias its
chime-in probability.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import config
import apikeys
from openai import OpenAI

_log = logging.getLogger(__name__)

_client = OpenAI(api_key=apikeys.OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

ADDRESS_DIRECT        = "direct_address"
ADDRESS_REFERENTIAL   = "referential"
ADDRESS_INSTRUCTIONAL = "instructional"
ADDRESS_UNRELATED     = "unrelated"

_VALID_LABELS = {
    ADDRESS_DIRECT,
    ADDRESS_REFERENTIAL,
    ADDRESS_INSTRUCTIONAL,
    ADDRESS_UNRELATED,
}

_VALID_SENTIMENTS = {"positive", "neutral", "negative"}


@dataclass
class AddressClassification:
    label: str            # one of the four ADDRESS_* constants
    sentiment: str        # positive / neutral / negative
    rule: str             # short label of which rule/path produced this
    contained_keyword: bool


# ─────────────────────────────────────────────────────────────────────────────
# Keyword pre-filter
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_KEYWORDS = (
    "rex", "r3x", "r 3 x", "r-3-x",
    "droid", "robot", "dj rex", "dj r3x", "deejay rex",
)


def _keywords() -> tuple[str, ...]:
    return tuple(getattr(config, "ADDRESS_MODE_KEYWORDS", _DEFAULT_KEYWORDS))


def contains_rex_keyword(text: str) -> bool:
    """Cheap word-boundary check for any rex-related keyword. Caller can use this
    to decide whether classification is worth running at all."""
    if not text:
        return False
    lower = text.lower()
    for kw in _keywords():
        if not kw:
            continue
        # Word-boundary for short alpha keywords; substring for compound forms.
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Hard-rule layer
# ─────────────────────────────────────────────────────────────────────────────

# "say hi to rex", "tell rex …", "show rex …", "ask rex …" — Rex is the object.
_INSTRUCTIONAL_VERBS = (
    "say hi to", "say hello to", "say goodbye to", "wave at",
    "introduce", "show", "tell", "ask", "bring", "take",
    "meet", "greet",
)

# 2nd-person directives that begin with the rex name = direct address.
# e.g. "Rex, what time is it", "Rex play something", "hey rex …"
_DIRECT_PREFIXES = (
    "hey rex", "hi rex", "hello rex",
    "rex,", "rex?", "rex!",
    "dj rex", "dj r3x", "yo rex", "yo robot",
)


def _hard_rule(text: str) -> Optional[str]:
    """Return an address label if a high-confidence rule fires, else None."""
    if not text:
        return None
    t = text.strip().lower()

    # Direct prefix: starts with "hey rex", "rex," etc.
    for p in _DIRECT_PREFIXES:
        if t.startswith(p):
            return ADDRESS_DIRECT

    # Instructional: "say hi to rex / robot / droid".
    keywords = _keywords()
    for verb in _INSTRUCTIONAL_VERBS:
        if verb not in t:
            continue
        # Look for a rex-keyword within ~6 words after the verb.
        idx = t.find(verb)
        tail = t[idx + len(verb): idx + len(verb) + 60]
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", tail):
                return ADDRESS_INSTRUCTIONAL

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM fallback
# ─────────────────────────────────────────────────────────────────────────────

_LLM_PROMPT = (
    'You are classifying a transcribed utterance for a robot character named Rex '
    '(also called DJ Rex, DJ-R3X, R3X, "the droid", "the robot"). Decide whether '
    'the speaker is talking TO Rex, ABOUT Rex, instructing someone else regarding '
    'Rex, or merely uttering the word in passing.\n\n'
    'Return ONE JSON object on a single line with exactly these keys:\n'
    '  "label": one of "direct_address", "referential", "instructional", "unrelated"\n'
    '  "sentiment": one of "positive", "neutral", "negative" — sentiment toward Rex\n\n'
    'Definitions:\n'
    '  direct_address — speaker is talking TO Rex (questions, requests, comments aimed at him)\n'
    '  referential — speaker is talking ABOUT Rex to a third party '
    '("Rex is so fun", "this droid is cool", "he\'s a real character")\n'
    '  instructional — Rex is the object of an action directed at someone else '
    '("say hi to Rex", "show them the droid", "introduce Rex to her")\n'
    '  unrelated — Rex/droid/robot appears but the utterance is not actually about him '
    '(e.g. talking about a different robot, a movie title, a coincidental word)\n\n'
    'Context: {context}\n'
    'Utterance: "{text}"\n\n'
    'Reply with the JSON object only.'
)


def _llm_classify(text: str, context: str) -> Optional[tuple[str, str]]:
    """Single tiny LLM call. Returns (label, sentiment) or None on error."""
    import json

    prompt = _LLM_PROMPT.format(text=text, context=context or "(no extra context)")
    try:
        resp = _client.chat.completions.create(
            model=getattr(config, "LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=40,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        _log.debug("address_mode LLM call failed: %s", exc)
        return None

    # Strip code fences if model added them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # Drop leading "json\n" if present
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        data = json.loads(raw)
    except Exception:
        # Last-ditch: extract any of the four label words from the raw response.
        lowered = raw.lower()
        for cand in (ADDRESS_REFERENTIAL, ADDRESS_INSTRUCTIONAL, ADDRESS_UNRELATED, ADDRESS_DIRECT):
            if cand in lowered:
                return (cand, "neutral")
        _log.debug("address_mode could not parse LLM reply: %r", raw)
        return None

    label = str(data.get("label", "")).strip().lower()
    sentiment = str(data.get("sentiment", "neutral")).strip().lower()
    if label not in _VALID_LABELS:
        return None
    if sentiment not in _VALID_SENTIMENTS:
        sentiment = "neutral"
    return (label, sentiment)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify(
    text: str,
    *,
    context: Optional[str] = None,
    skip_llm: bool = False,
) -> AddressClassification:
    """
    Classify a transcribed utterance's address mode.

    Cheap path: if no rex-keyword, returns direct_address (caller should
    typically not even call this in that case).

    `context` is an optional one-line description of the social setting
    ("2 people present, Bret is dominant speaker, Rex is in active conversation").
    `skip_llm` forces hard-rule-only operation (used for tests / latency-critical paths).
    """
    has_kw = contains_rex_keyword(text)
    if not has_kw:
        return AddressClassification(
            label=ADDRESS_DIRECT,
            sentiment="neutral",
            rule="no_keyword",
            contained_keyword=False,
        )

    rule = _hard_rule(text)
    if rule is not None:
        return AddressClassification(
            label=rule,
            sentiment="neutral",
            rule="hard_rule",
            contained_keyword=True,
        )

    if skip_llm:
        return AddressClassification(
            label=ADDRESS_DIRECT,
            sentiment="neutral",
            rule="skip_llm_default",
            contained_keyword=True,
        )

    result = _llm_classify(text, context or "")
    if result is None:
        # Fail-safe: when the classifier can't decide, default to direct_address
        # so Rex still responds (a missed referential is better than ignoring
        # someone who actually did address him).
        return AddressClassification(
            label=ADDRESS_DIRECT,
            sentiment="neutral",
            rule="llm_fallback_default",
            contained_keyword=True,
        )

    label, sentiment = result
    return AddressClassification(
        label=label,
        sentiment=sentiment,
        rule="llm",
        contained_keyword=True,
    )
