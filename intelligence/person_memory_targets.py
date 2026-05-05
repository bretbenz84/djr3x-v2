"""Helpers for separating person-memory targets from general topic questions."""

from __future__ import annotations

from collections import Counter
import logging
import re

_log = logging.getLogger(__name__)

PERSON_MEMORY_TERM_RE = re.compile(
    r"\b("
    r"me|myself|me\?|my\s+|mine|i\s+told\s+you|i'?ve\s+told\s+you|"
    r"remember|memory|memories|person|people|friend|partner|wife|husband|"
    r"mom|mother|dad|father|brother|sister|kid|child|son|daughter"
    r")\b",
    re.IGNORECASE,
)
_NAME_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\u2019][a-z0-9]+)?")
_UNSAFE_NAME_START_TOKENS = {
    "a",
    "am",
    "an",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "dont",
    "don't",
    "how",
    "i",
    "is",
    "me",
    "my",
    "no",
    "not",
    "should",
    "straight",
    "tell",
    "the",
    "what",
    "when",
    "where",
    "who",
    "why",
    "would",
    "you",
    "your",
}


def _normalize_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for raw in _NAME_TOKEN_RE.findall((value or "").lower()):
        token = raw.strip("'\u2019")
        if token.endswith("'s") or token.endswith("\u2019s"):
            token = token[:-2]
        if token:
            tokens.append(token)
    return tokens


def _contains_sequence(haystack: list[str], needle: tuple[str, ...]) -> bool:
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    span = len(needle)
    return any(tuple(haystack[index:index + span]) == needle for index in range(len(haystack) - span + 1))


def _load_known_person_names() -> list[str]:
    try:
        from memory import people as people_memory

        return people_memory.list_person_names()
    except Exception as exc:
        _log.debug("known person lookup unavailable for memory routing: %s", exc)
        return []


def _references_known_person(text: str) -> bool:
    text_tokens = _normalize_tokens(text)
    if not text_tokens:
        return False

    known_names = [
        tuple(tokens)
        for name in _load_known_person_names()
        if (tokens := _normalize_tokens(name))
        and tokens[0] not in _UNSAFE_NAME_START_TOKENS
    ]
    if not known_names:
        return False

    first_counts = Counter(tokens[0] for tokens in known_names if tokens)
    for name_tokens in known_names:
        if _contains_sequence(text_tokens, name_tokens):
            return True
        if first_counts[name_tokens[0]] == 1 and name_tokens[0] in text_tokens:
            return True
    return False


def references_person_memory_target(text: str) -> bool:
    """Return True when text points at a person, relationship, or known name."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    if PERSON_MEMORY_TERM_RE.search(cleaned):
        return True
    return _references_known_person(cleaned)
