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


# Ask-order for the dataset spine. Size/holdability moved AHEAD of "is it a place?" so a
# holdable answer prunes the place question entirely (a thing you can hold is never a
# place), and ahead of the material/category probes so they inherit maximum context.
_SPINE_ORDER = [
    "alive", "person", "animal", "plant", "manmade", "bigger", "handheld",
    "place", "edible", "indoors", "metal", "electronic", "tool", "wearable", "vehicle",
]

# Authored TIER-2 branch questions — the proven category splitters the dataset spine lacks
# (live-logged 2026-07-07: "is it a toy?" not reaching the table until Q18 lost a game whose
# answer was a rubber ducky). Each entry applies only when every `requires` concept has been
# answered with exactly that value, and never when any `not_true` concept is True — so a
# pruned/unanswered concept doesn't block the branch, but a contradicting YES does.
_TIER2: list[dict] = [
    # Man-made object branch.
    {"concept": "toy", "question": "is it a toy or a game?",
     "requires": {"manmade": True}, "not_true": ["place", "edible"]},
    {"concept": "kitchen", "question": "is it mainly used in the kitchen?",
     "requires": {"manmade": True, "indoors": True}, "not_true": ["place", "edible", "toy"]},
    {"concept": "bathroom", "question": "is it mainly found in the bathroom?",
     "requires": {"manmade": True, "indoors": True}, "not_true": ["place", "edible", "kitchen"]},
    {"concept": "sports", "question": "is it used for sports or exercise?",
     "requires": {"manmade": True}, "not_true": ["place", "edible", "toy"]},
    {"concept": "decorative", "question": "is it mainly for decoration?",
     "requires": {"manmade": True}, "not_true": ["place", "edible", "toy", "sports"]},
    {"concept": "furniture", "question": "is it a piece of furniture?",
     "requires": {"manmade": True, "handheld": False}, "not_true": ["place", "edible", "toy"]},
    {"concept": "plastic", "question": "is it mostly made of plastic?",
     "requires": {"manmade": True, "metal": False}, "not_true": ["place", "edible"]},
    # Food branch.
    {"concept": "drink", "question": "is it a drink?",
     "requires": {"edible": True}, "not_true": []},
    {"concept": "sweet", "question": "is it sweet?",
     "requires": {"edible": True}, "not_true": []},
    {"concept": "fruit_veg", "question": "is it a fruit or a vegetable?",
     "requires": {"edible": True}, "not_true": ["drink"]},
    {"concept": "hot_food", "question": "is it usually eaten hot?",
     "requires": {"edible": True}, "not_true": ["drink", "fruit_veg"]},
    # Animal branch.
    {"concept": "pet", "question": "is it commonly kept as a pet?",
     "requires": {"animal": True}, "not_true": []},
    {"concept": "four_legs", "question": "does it walk on four legs?",
     "requires": {"animal": True}, "not_true": []},
    {"concept": "flies", "question": "can it fly?",
     "requires": {"animal": True}, "not_true": ["four_legs"]},
    {"concept": "water_animal", "question": "does it live in water?",
     "requires": {"animal": True}, "not_true": ["four_legs", "flies"]},
    # Person branch.
    {"concept": "real_person", "question": "are they a real person, not fictional?",
     "requires": {"person": True}, "not_true": []},
    {"concept": "famous", "question": "are they famous?",
     "requires": {"person": True}, "not_true": []},
    {"concept": "performer", "question": "are they known for movies, music, or TV?",
     "requires": {"person": True, "famous": True}, "not_true": []},
    {"concept": "athlete", "question": "are they an athlete?",
     "requires": {"person": True, "famous": True}, "not_true": ["performer"]},
    {"concept": "known_personally", "question": "do you know them personally?",
     "requires": {"person": True, "famous": False}, "not_true": []},
    # Place branch.
    {"concept": "real_place", "question": "is it a real place?",
     "requires": {"place": True}, "not_true": []},
    {"concept": "landmark", "question": "is it a famous landmark?",
     "requires": {"place": True}, "not_true": []},
    {"concept": "building", "question": "is it a building or a structure?",
     "requires": {"place": True}, "not_true": []},
    {"concept": "city_country", "question": "is it a city or a country?",
     "requires": {"place": True}, "not_true": ["building", "landmark"]},
    # Plant branch.
    {"concept": "tree", "question": "is it a tree?",
     "requires": {"plant": True}, "not_true": []},
]


def spine() -> list[dict]:
    """The dataset spine in authored ask-order (each: concept, parent, question, yes_rate)."""
    entries = list(_load().get("spine", []))
    rank = {c: i for i, c in enumerate(_SPINE_ORDER)}
    entries.sort(key=lambda e: rank.get(e.get("concept"), len(_SPINE_ORDER)))
    return entries


def is_loaded() -> bool:
    return bool(_load().get("spine"))


def _entry_applicable(entry: dict, answers: dict, asked: set) -> bool:
    """Shared applicability check for dataset-spine and tier-2 entries: not yet answered or
    asked, parent branch open, `requires` satisfied, no `not_true` contradiction, and none of
    the mutual-exclusion prunes triggered."""
    concept = entry.get("concept")
    if concept in answers:
        return False
    if entry.get("question", "") in asked:
        return False

    alive = answers.get("alive")
    manmade = answers.get("manmade")
    handheld = answers.get("handheld")
    is_person = answers.get("person") is True
    is_animal = answers.get("animal") is True
    is_plant = answers.get("plant") is True
    is_place = answers.get("place") is True
    is_edible = answers.get("edible") is True

    parent = entry.get("parent")
    if parent == "alive" and alive is not True:
        return False          # only ask the living-thing branch once "alive?" is YES
    if parent == "not_alive" and alive is not False:
        return False          # only ask the object branch once "alive?" is NO

    for req_concept, req_value in (entry.get("requires") or {}).items():
        if answers.get(req_concept) is not req_value:
            return False
    for blocker in entry.get("not_true") or []:
        if answers.get(blocker) is True:
            return False

    # Mutual-exclusion pruning: once a category is established, don't waste a question on
    # a sibling category the answer already rules out (the redundant "is it a plant?" after
    # animal=yes). NOTE: we deliberately do NOT skip "is it a place?" after man-made=yes —
    # man-made PLACES (Coney Island, the Eiffel Tower, a stadium) need that signal, and
    # losing it made the guesser chase buildings/towers.
    if concept == "person":
        if is_animal or is_plant:
            return False
        # A person answers "alive?" yes — EXCEPT dead/fictional people, who show up as
        # alive=no + man-made=no. Gating on that (instead of asking person unconditionally
        # right after alive=no) saves a wasted question on every man-made object game
        # (live-logged 2026-07-07: Q2 "is it a person?" after alive=no, answer rubber ducky)
        # while still discovering Lincoln and Darth Vader via the not-man-made path.
        if alive is False and manmade is not False:
            return False
    if concept == "animal" and (is_person or is_plant):
        return False
    if concept == "plant" and (is_person or is_animal):
        return False
    if concept == "place":
        # A living thing isn't a place, a person isn't a place, and neither is anything
        # you can hold in your hands.
        if alive is True or is_person or handheld is True:
            return False
    if concept == "edible" and (is_person or is_place):
        return False
    # Object-shaped probes are wasted on people and places — their tier-2 branches ask
    # the right questions instead.
    if concept in ("bigger", "handheld", "indoors") and (is_person or is_place):
        return False
    # Nothing edible (and no place, and no person — a DEAD person passes the not_alive
    # parent gate) is metal/electronic/a tool/wearable/a vehicle.
    if concept in ("metal", "electronic", "tool", "wearable", "vehicle") and (
        is_edible or is_place or is_person
    ):
        return False
    # A vehicle you can hold is a toy, not a vehicle — the toy question covers it.
    if concept == "vehicle" and handheld is True:
        return False
    return True


def next_spine_question(answers: dict, asked: set) -> Optional[dict]:
    """Pick the next spine question to ask, given concept->bool answers so far and the set
    of already-asked (normalized) question strings. Walks the dataset spine (in authored
    order), then the authored tier-2 branch questions. Skips branches an earlier answer made
    redundant. Returns the entry dict, or None if the spine is exhausted/inapplicable.

    `answers` maps spine concept -> True/False (only yes/no answers; skip unsure ones).
    """
    for entry in spine() + _TIER2:
        if _entry_applicable(entry, answers, asked):
            return entry
    return None


def spine_menu(answers: dict, asked: set, limit: int = 4) -> list[str]:
    """Up to `limit` applicable, unasked spine/tier-2 questions — a vetted menu the LLM
    endgame can prefer over improvising a redundant or low-information question."""
    menu: list[str] = []
    for entry in spine() + _TIER2:
        if _entry_applicable(entry, answers, asked):
            menu.append(entry["question"])
            if len(menu) >= limit:
                break
    return menu


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
