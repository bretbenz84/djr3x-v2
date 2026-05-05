"""
Shared profile-question selection for sparse known-person profiles.

This keeps Rex's curiosity grounded in the tiered QUESTION_POOL without letting
face-enrollment appearance facts count as "knowing" someone.
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from memory import boundaries as boundary_memory
from memory import facts as facts_memory
from memory import people as people_memory
from memory import relationships as rel_memory

_log = logging.getLogger(__name__)

PROFILE_FACT_EXCLUDED_CATEGORIES = {
    "appearance",
    "boundary",
    "identity",
    "relationship",
}

PROFILE_FACT_EXCLUDED_KEYS = {
    "age_category",
    "age_range",
    "build",
    "hair_color",
    "hair_style",
    "height_estimate",
    "notable_features",
    "skin_color",
}

QUESTION_BOUNDARY_TOPICS = {
    "hometown": "hometown",
    "job": "work",
    "favorite_movie": "movies",
    "favorite_music": "music",
    "how_found_rex": "rex",
    "hobbies": "hobbies",
    "travel": "travel",
    "proudest_moment": "personal history",
    "biggest_challenge": "personal history",
    "obsession": "interests",
    "relationships": "relationships",
    "values": "values",
    "fears": "fears",
    "life_changing": "personal history",
    "regret": "regret",
    "meaning_of_life": "philosophy",
    "free_will": "philosophy",
    "consciousness": "philosophy",
    "good_life": "philosophy",
}


def is_profile_fact(fact: dict) -> bool:
    """Return True for biographical/user-interest facts, not visual metadata."""
    if not fact:
        return False
    key = str(fact.get("key") or "").strip().lower()
    category = str(fact.get("category") or "").strip().lower()
    if not key and not str(fact.get("value") or "").strip():
        return False
    if key in PROFILE_FACT_EXCLUDED_KEYS:
        return False
    if category in PROFILE_FACT_EXCLUDED_CATEGORIES:
        return False
    return True


def profile_fact_count(person_id: int) -> int:
    """Count facts that should satisfy low-memory profile curiosity."""
    try:
        return sum(1 for fact in facts_memory.get_facts(person_id) if is_profile_fact(fact))
    except Exception as exc:
        _log.debug("profile fact count failed for person_id=%s: %s", person_id, exc)
        return 0


def question_blocked_by_boundary(person_id: Optional[int], question: dict) -> bool:
    if person_id is None or not question:
        return False
    topic = QUESTION_BOUNDARY_TOPICS.get(question.get("key") or "")
    if not topic:
        return False
    try:
        return (
            boundary_memory.is_blocked(person_id, "ask", topic)
            or boundary_memory.is_blocked(person_id, "mention", topic)
            or boundary_memory.is_blocked(person_id, "ask", "questions")
        )
    except Exception as exc:
        _log.debug("question boundary check failed: %s", exc)
        return False


def next_profile_question(person_id: int) -> Optional[dict]:
    """
    Return the next tier-appropriate QUESTION_POOL item this person has not
    answered, been asked, covered with a profile fact, or blocked by boundary.
    """
    try:
        person = people_memory.get_person(person_id)
        tier = person.get("friendship_tier", "stranger") if person else "stranger"
        max_depth = config.TIER_MAX_DEPTH.get(tier, 1)
        asked = rel_memory.get_asked_question_keys(person_id)
        known_fact_keys: set[str] = set()
        known_fact_categories: set[str] = set()
        for fact in facts_memory.get_facts(person_id):
            if not is_profile_fact(fact):
                continue
            key = str(fact.get("key") or "").strip()
            category = str(fact.get("category") or "").strip()
            if key:
                known_fact_keys.add(key)
            if category:
                known_fact_categories.add(category)

        for candidate in config.QUESTION_POOL:
            q_key = candidate.get("key")
            if candidate.get("depth", 1) > max_depth:
                continue
            if q_key in asked or q_key in known_fact_keys or q_key in known_fact_categories:
                continue
            if question_blocked_by_boundary(person_id, candidate):
                continue
            return candidate
    except Exception as exc:
        _log.debug("next profile question failed for person_id=%s: %s", person_id, exc)
    return None
