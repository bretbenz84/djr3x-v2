"""Knowledge base for R3X's 20 Questions game (R3X is the guesser).

Distilled from the allenai/twentyquestions dataset by `assets/20questions/build_kb.py`
into `assets/20questions/r3x_kb.json`. Provides two things the runtime uses:

  * a SPINE of proven yes/no discriminator questions for strong, varied openings, with
    light branch-skipping so R3X doesn't ask redundant questions (e.g. "is it an animal?"
    after "is it alive?" came back NO), and
  * a SUBJECT VOCABULARY (the real things people pick) to ground/clean R3X's final guess.

Everything is failure-safe: if the asset is missing the helpers degrade gracefully and the
game falls back to the LLM alone.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from rapidfuzz import fuzz, process

_log = logging.getLogger("djr3x.games.20q")

_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "20questions", "r3x_kb.json",
)

_lock = threading.Lock()
_kb: Optional[dict] = None
_loaded = False


def _load() -> dict:
    """Lazy-load the KB once. Returns an empty-but-valid structure if the asset is absent."""
    global _kb, _loaded
    if _loaded:
        return _kb or {}
    with _lock:
        if _loaded:
            return _kb or {}
        try:
            with open(_KB_PATH, encoding="utf-8") as f:
                _kb = json.load(f)
            _log.info(
                "[20q] loaded KB: %d spine questions, %d subjects",
                len(_kb.get("spine", [])), len(_kb.get("subjects", [])),
            )
        except Exception as exc:  # missing/corrupt asset — degrade to LLM-only
            _log.warning("[20q] KB unavailable (%s); falling back to LLM-only", exc)
            _kb = {"spine": [], "subjects": []}
        _loaded = True
    return _kb or {}


def spine() -> list[dict]:
    """The ordered spine of opener questions (each: concept, parent, question, yes_rate)."""
    return list(_load().get("spine", []))


def is_loaded() -> bool:
    return bool(_load().get("spine"))


def next_spine_question(answers: dict, asked: set) -> Optional[dict]:
    """Pick the next spine question to ask, given concept->bool answers so far and the set
    of already-asked (normalized) question strings. Skips branches an earlier answer made
    redundant. Returns the spine entry dict, or None if the spine is exhausted/inapplicable.

    `answers` maps spine concept -> True/False (only yes/no answers; skip unsure ones).
    """
    alive = answers.get("alive")
    is_person = answers.get("person") is True
    is_animal = answers.get("animal") is True
    is_plant = answers.get("plant") is True

    for entry in _load().get("spine", []):
        concept = entry.get("concept")
        if concept in answers:
            continue
        if entry.get("question", "") in asked:
            continue

        parent = entry.get("parent")
        if parent == "alive" and alive is not True:
            continue          # only ask the living-thing branch once "alive?" is YES
        if parent == "not_alive" and alive is not False:
            continue          # only ask the object branch once "alive?" is NO

        # Mutual-exclusion pruning: once a category is established, don't waste a question on
        # a sibling category the answer already rules out (the redundant "is it a plant?" after
        # animal=yes). NOTE: we deliberately do NOT skip "is it a place?" after man-made=yes —
        # man-made PLACES (Coney Island, the Eiffel Tower, a stadium) need that signal, and
        # losing it made the guesser chase buildings/towers. Only a living thing is never a
        # place, so "place" is pruned on alive=yes alone.
        if concept == "person" and (is_animal or is_plant):
            continue
        if concept == "animal" and (is_person or is_plant):
            continue
        if concept == "plant" and (is_person or is_animal):
            continue
        if concept == "place" and alive is True:
            continue          # a living thing isn't a "place"

        return entry
    return None


def subjects() -> list[str]:
    return _load().get("subjects", [])


def snap_guess(text: str, min_score: float = 90.0) -> Optional[str]:
    """If `text` closely matches a real subject in the vocabulary, return that canonical
    subject; otherwise None. Used to clean up R3X's free-text guess (e.g. "a guitar" ->
    "guitar") without distorting guesses the dataset has never seen.
    """
    if not text:
        return None
    vocab = subjects()
    if not vocab:
        return None
    cleaned = text.strip().lower()
    for prefix in ("a ", "an ", "the ", "is it ", "it's ", "its "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    cleaned = cleaned.strip(" .!?\"'")
    if not cleaned:
        return None
    if cleaned in vocab:
        return cleaned
    match = process.extractOne(cleaned, vocab, scorer=fuzz.ratio)
    if match and match[1] >= min_score:
        return match[0]
    return None
