"""Cached English pronunciation comparisons for already-transcribed game answers.

CMUdict ships with the Python dependency: no service or ML inference is needed.
Exact homophones can be accepted; nearby sounds only justify asking again.
"""
from functools import lru_cache
from itertools import product
import logging
import re

from rapidfuzz.distance import Levenshtein

_log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _dictionary():
    try:
        import cmudict
        return cmudict.dict()
    except ImportError:
        _log.warning("Spoken-answer pronunciation lookup unavailable; run setup_assets.py to install cmudict")
        return {}


def _unstressed(phones):
    return tuple(re.sub(r"\d", "", phone) for phone in phones)


@lru_cache(maxsize=2048)
def pronunciations(text: str) -> frozenset[tuple[str, ...]]:
    words = text.lower().split()
    if not words or len(words) > 8 or any(not word.isalpha() for word in words):
        return frozenset()
    dictionary = _dictionary()
    choices = []
    for word in words:
        phones = {_unstressed(p) for p in dictionary.get(word, [])}
        # A spoken singular/plural is sufficient (symbol/cymbals). Only remove
        # a real plural ending whose dictionary pronunciation extends its stem;
        # do not blindly strip the last sound of arbitrary words.
        if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
            for stem in (word[:-1], word[:-2] if word.endswith("es") else ""):
                for p in dictionary.get(stem, []):
                    base = _unstressed(p)
                    if any(p == base + suffix for p in phones for suffix in (("S",), ("Z",), ("IH", "Z"))):
                        phones.add(base)
        if not phones:
            return frozenset()
        choices.append(sorted(phones)[:4])
    # Bound combinations for ambiguous multi-word pronunciations.
    result = set()
    for variants in product(*choices):
        result.add(tuple(phone for variant in variants for phone in variant))
        if len(result) >= 64:
            break
    return frozenset(result)


def same_pronunciation(user: str, expected: str) -> bool:
    return bool(pronunciations(user) & pronunciations(expected))


def possible_mishearing(user: str, expected: str) -> bool:
    """Close short words need a repeat, never automatic points.

    'flag' and 'flood' share an onset but differ in two phones. In a noisy
    room that is ambiguous. 'struts' and 'frets' have different onsets and
    remain distinct answers. Missing pieces of multi-part answers do not qualify.
    """
    if len(user.split()) != 1 or len(expected.split()) != 1:
        return False
    for left in pronunciations(user):
        for right in pronunciations(expected):
            if (3 <= min(len(left), len(right)) <= max(len(left), len(right)) <= 8
                    and left[:2] == right[:2]
                    and 0 < Levenshtein.distance(left, right) <= 2):
                return True
    return False
