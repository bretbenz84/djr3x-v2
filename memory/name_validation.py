"""Shared person-name normalization and validation."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

_MAX_NAME_WORDS = 3
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_PREFERRED_NAME_SPLIT_RE = re.compile(
    r"\b(?:but\s+)?(?:you can\s+)?call me\b",
    re.IGNORECASE,
)
_INTRO_SPLIT_RE = re.compile(
    r"\s+\b(?:and\s+)?(?:this|that)\s+is\b|\s+\b(?:and\s+)?(?:meet|say hi to)\b",
    re.IGNORECASE,
)
_TRAILING_FILLER_RE = re.compile(
    r"\b(?:hi|hello|hey|wait|hold on|actually|instead|from now on|please|thanks|thank you)\b.*$",
    re.IGNORECASE,
)

_FILLER_UTTERANCES = {
    "mmm",
    "mm",
    "hmm",
    "hm",
    "uh",
    "uhh",
    "um",
    "umm",
    "ah",
    "ahh",
    "er",
    "err",
    "huh",
    "mhm",
    "mmhmm",
    "uhhuh",
    "yeah",
    "yep",
    "yup",
    "nah",
    "wow",
    "whoa",
    "ha",
    "haha",
    "hehe",
}
# A "name" whose every token is a backchannel/laugh syllable is a transcribed
# non-verbal noise, not a person ("Mm-hmm", "Uh-huh", "Ha ha") — live incident
# 2026-07-26: Whisper heard "Mm-hmm" at an identity prompt and a phantom person
# was enrolled with the speaker's own face and voice, which then OUTSCORED the
# real person on their own speech every session after.
_BACKCHANNEL_TOKENS = _FILLER_UTTERANCES | {"he", "ho", "hah", "heh", "hmmm"}
_BAD_SINGLE_TOKENS = {
    "again",
    "back",
    "both",
    "everybody",
    "everyone",
    "fire",
    "fine",
    "good",
    "great",
    "have",
    "has",
    "headed",
    "heading",
    "here",
    "home",
    "hi",
    "hello",
    "hey",
    "i",
    "im",
    "i'm",
    "me",
    "my",
    "name",
    "nah",
    "naw",
    "nobody",
    "no",
    "nope",
    "okay",
    "ok",
    "someone",
    "somebody",
    "ready",
    "sorry",
    "there",
    "unbelievable",
    "you",
    "your",
    "whoever",
}
_BAD_PHRASE_STARTS = {
    "a",
    "am",
    "an",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "don't",
    "dont",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "should",
    "straight",
    "tell",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "who",
    "why",
    "would",
    "you",
    "your",
}
_BAD_PHRASE_TOKENS = {
    "about",
    "again",
    "bit",
    "break",
    "chances",
    "down",
    "funny",
    "gonna",
    # "I'm headed home" → candidate "headed home" minted phantom person
    # "Headed Home" with a real speaker's voice+face (live 2026-08-23 18:26).
    "headed",
    "heading",
    "home",
    "know",
    "manual",
    "override",
    "people",
    "sit",
    "there",
}
# Sentence-initial discourse markers / fillers that get mistaken for a bare name
# when an utterance opens with one and is split at the first comma
# (e.g. "Also, what are you doing today?" -> "Also").
_DISCOURSE_MARKERS = {
    "also",
    "so",
    "well",
    "anyway",
    "anyways",
    "anyhow",
    "actually",
    "basically",
    "honestly",
    "literally",
    "alright",
    "right",
    "now",
    "um",
    "uh",
    "oh",
    "yeah",
    "yes",
    "but",
    "and",
    "or",
    "then",
    "plus",
    "besides",
    "however",
    "though",
    "like",
    "incidentally",
    "frankly",
}


def normalized_name_key(value: str) -> str:
    """Return a lowercase lookup key for names and aliases."""
    tokens = []
    for raw in re.findall(r"[a-z0-9]+(?:['\u2019][a-z0-9]+)?", (value or "").lower()):
        token = raw.strip("'\u2019")
        if token.endswith("'s") or token.endswith("\u2019s"):
            token = token[:-2]
        if token:
            tokens.append(token)
    return " ".join(tokens)


# Dictated initials arrive spelled out, with or without periods: "JT" is heard as
# "J.T.", "J. T." or "J T". The dotted spellings used to be TRUNCATED to the first
# letter by the sentence-splitting below (it cuts at the first "."), which is how the
# live 2026-08-23 session enrolled a phantom person named "J" off someone answering
# "It was JT" — that row then sat one fuzzy match away from the real JT. Collapse an
# initial run back into one token BEFORE the punctuation split so the letters survive.
_DOTTED_INITIALS_RE = re.compile(r"\b(?:[A-Za-z]\.\s*){2,}|\b(?:[A-Za-z]\.\s*)+[A-Za-z]\b")


def _collapse_dotted_initials(text: str) -> str:
    def _join(match: re.Match) -> str:
        run = match.group(0)
        letters = "".join(re.findall(r"[A-Za-z]", run))
        # Keep the separator that followed the run, or "A. J. Foyt" welds into "AJFoyt".
        return letters + (" " if run[-1:].isspace() else "")

    return _DOTTED_INITIALS_RE.sub(_join, text)


def _merge_initial_tokens(tokens: list[str]) -> list[str]:
    """Fold a run of consecutive single-letter tokens into one ("J T" -> "JT").

    Only ADJACENT single letters merge, so a middle initial between real words
    ("Bret M Benziger") is left alone, while "J T" / "J J Watt" resolve correctly.
    """
    merged: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) > 1:
            merged.append("".join(run).upper())
        elif run:
            merged.append(run[0])
        run = []
        merged.append(token)
    if len(run) > 1:
        merged.append("".join(run).upper())
    elif run:
        merged.append(run[0])
    return merged


def _clean_candidate(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    call_me_parts = _PREFERRED_NAME_SPLIT_RE.split(text, maxsplit=1)
    if len(call_me_parts) > 1:
        # ASR may close the sentence right after the verb ("Call me. Playa P"),
        # leaving leading punctuation that would make the sentence-split below
        # return an empty candidate — strip it before splitting.
        text = call_me_parts[1].strip().lstrip(".,!?;: ").strip()
    text = _collapse_dotted_initials(text)
    text = re.split(r"[,.!?;:]", text, maxsplit=1)[0].strip()
    text = _INTRO_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    text = _TRAILING_FILLER_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def looks_like_initials(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z]", "", value or "")
    return 2 <= len(compact) <= 4 and compact.isupper()


def normalize_person_name(value: str, *, allow_single: bool = True) -> Optional[str]:
    """Return a storage-ready person name, or None for non-name fragments."""
    text = _clean_candidate(value)
    if not text:
        return None

    key = normalized_name_key(text)
    if not key or key in _FILLER_UTTERANCES:
        return None
    if all(part in _BACKCHANNEL_TOKENS for part in key.split()):
        return None

    raw_tokens = _NAME_TOKEN_RE.findall(text)
    if not raw_tokens:
        return None
    if len(raw_tokens) == 1 and not allow_single:
        return None
    if len(raw_tokens) > _MAX_NAME_WORDS:
        return None

    tokens = [token.strip("'-") for token in raw_tokens if token.strip("'-")]
    lowered = [token.lower() for token in tokens]
    if not tokens:
        return None
    if len(tokens) == 1 and (
        lowered[0] in _BAD_SINGLE_TOKENS or lowered[0] in _DISCOURSE_MARKERS
    ):
        return None
    if len(tokens) > 1:
        if lowered[0] in _BAD_PHRASE_STARTS or lowered[0] in _DISCOURSE_MARKERS:
            return None
        if any(token in _BAD_SINGLE_TOKENS or token in _BAD_PHRASE_TOKENS for token in lowered):
            return None

    # Applied after the guards above on purpose: merging earlier would shrink a
    # rejected phrase ("It was J T") back under the word limit and let it through.
    tokens = _merge_initial_tokens(tokens)

    if all(token.islower() for token in tokens):
        tokens = [token.capitalize() for token in tokens]
    return " ".join(tokens)


def is_single_token_name(value: str) -> bool:
    name = normalize_person_name(value)
    return bool(name and len(name.split()) == 1)


def names_are_similar(left: str, right: str, *, threshold: float = 0.84) -> bool:
    left_key = normalized_name_key(left)
    right_key = normalized_name_key(right)
    if not left_key or not right_key or left_key == right_key:
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= threshold


def full_names_are_similar(
    left: str,
    right: str,
    *,
    first_threshold: float = 0.84,
    surname_threshold: float = 0.6,
) -> bool:
    """Token-aware near-match for two multi-token names.

    The whole-string ratio misses ASR-garbled surnames: "Bret Bender" vs
    "Bret Benziger" scores 0.833, just under the 0.84 bar, so no fuzzy tier
    fires and a phantom person gets minted. Compare the first token and the
    surname remainder separately instead — the first name must match (exactly
    or fuzzily) and the surnames must be clearly related, while genuinely
    different surnames ("Bret Smith" vs "Bret Jones") stay well apart.
    """
    left_key = normalized_name_key(left)
    right_key = normalized_name_key(right)
    if not left_key or not right_key or left_key == right_key:
        return False
    left_tokens = left_key.split()
    right_tokens = right_key.split()
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    if (
        left_tokens[0] != right_tokens[0]
        and SequenceMatcher(None, left_tokens[0], right_tokens[0]).ratio() < first_threshold
    ):
        return False
    left_rest = " ".join(left_tokens[1:])
    right_rest = " ".join(right_tokens[1:])
    return SequenceMatcher(None, left_rest, right_rest).ratio() >= surname_threshold
