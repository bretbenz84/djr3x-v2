"""
memory/admin.py — editor/admin data access for the Memory Banks GUI.

One place that wraps the read/write operations the Memory Banks window needs across
BOTH databases — Rex's own episodic memory (rex.db) and the people store (people.db) —
including the id-based edit/delete operations the per-feature modules don't expose.
Kept separate from the GUI so the data layer is unit-testable without a display, and
separate from the per-feature memory modules so admin/destructive editing is in one
auditable spot. All functions are failure-safe and return simple dicts/bools.
"""

from __future__ import annotations

import logging
from typing import Optional

from memory import database as db
from memory import rex_db
from memory import people as people_mem
from memory import facts as facts_mem

_log = logging.getLogger(__name__)


# Fact categories Rex's memory actually recognizes. Category is NOT free-form: it drives
# how a fact decays and how important it is (memory/facts.py) — e.g. birthday / identity /
# relationship never decay, and family / pet / preference / belief / worldview are
# high-importance. This is the union of the extractor's enum and the special-behavior
# categories in facts.py, ordered most-common-first. The GUI offers these in a dropdown
# (still editable, so a custom category is possible — it just won't get special handling).
FACT_CATEGORIES = [
    "preference",
    "family",
    "pet",
    "job",
    "hometown",
    "birthday",
    "relationship",
    "belief",
    "worldview",
    "identity",
    "inside_joke",
    "other",
]

# Keys are free-form snake_case labels (favorite_<x>, nephew, …), so these are only
# SUGGESTIONS offered as autocomplete — the user can still type anything.
COMMON_FACT_KEYS = [
    "birthday", "pronouns", "hometown", "job_title", "worldview",
    "favorite_music", "favorite_food", "favorite_movie", "favorite_color",
    "favorite_band", "favorite_drink", "favorite_team",
    "pet_name", "spouse", "child", "sibling", "nephew", "niece", "parent",
]

# Per-category key suggestions, so picking a category like "relationship" shows the
# conventional keys (boss, coworker, mentor, …) instead of a blank box. The KEY is the
# kind of thing; the VALUE is the specifics — e.g. category=relationship, key=boss,
# value="Daniel"; category=family, key=nephew, value="Wade"; category=preference,
# key=favorite_music, value="classical". Keys stay free text — these are just the menu.
FACT_KEYS_BY_CATEGORY = {
    "preference": [
        "favorite_music", "favorite_food", "favorite_movie", "favorite_show",
        "favorite_color", "favorite_band", "favorite_drink", "favorite_team",
        "favorite_book", "favorite_game", "favorite_place",
    ],
    "family": [
        "spouse", "partner", "child", "son", "daughter", "sibling", "brother",
        "sister", "mother", "father", "parent", "nephew", "niece",
        "grandparent", "grandchild", "cousin", "in_law",
    ],
    "relationship": [
        # Romantic / partner (the program mirrors these symmetrically — see
        # memory/social._SYMMETRIC_LABELS — and resolves them as synonyms in queries).
        "partner", "boyfriend", "girlfriend", "husband", "wife", "spouse",
        "fiance", "fiancee", "significant_other",
        # Social / professional
        "boss", "manager", "coworker", "colleague", "mentor", "mentee",
        "friend", "best_friend", "neighbor", "roommate", "teammate",
        "classmate", "ex", "rival", "acquaintance",
    ],
    "pet": ["pet_name", "pet_type", "pet_breed"],
    "job": ["job_title", "employer", "industry", "workplace"],
    "hometown": ["hometown", "current_city", "country"],
    "birthday": ["birthday", "birth_year"],
    "identity": ["pronouns", "full_name", "age", "nationality"],
    "worldview": ["worldview", "religion", "politics", "values"],
    "belief": ["belief", "opinion", "value"],
    "inside_joke": ["inside_joke"],
    "other": [],
}


def suggested_keys_for_category(category: str) -> list[str]:
    """Conventional key suggestions for a fact category (empty list = free text)."""
    return list(FACT_KEYS_BY_CATEGORY.get((category or "").strip().lower(), []))


# ─────────────────────────────────────────────────────────────────────────────
# Rex's own memories (rex.db → rex_episodes)
# ─────────────────────────────────────────────────────────────────────────────

def list_rex_memories(limit: int = 1000) -> list[dict]:
    """Rex's own episodic memories, newest first."""
    rows = rex_db.fetchall(
        "SELECT id, created_at, kind, summary, person_name, salience, session_id "
        "FROM rex_episodes ORDER BY created_at DESC, id DESC LIMIT ?",
        (int(limit),),
    )
    return [dict(r) for r in rows]


def update_rex_memory(
    memory_id: int,
    *,
    summary: Optional[str] = None,
    kind: Optional[str] = None,
    salience: Optional[float] = None,
) -> bool:
    """Edit a single Rex memory. Only the provided fields change."""
    sets: list[str] = []
    params: list[object] = []
    if summary is not None:
        sets.append("summary = ?")
        params.append(str(summary).strip())
    if kind is not None:
        sets.append("kind = ?")
        params.append(str(kind).strip() or "other")
    if salience is not None:
        try:
            params.append(max(0.0, min(1.0, float(salience))))
            sets.append("salience = ?")
        except (TypeError, ValueError):
            pass
    if not sets:
        return False
    params.append(int(memory_id))
    rex_db.execute(f"UPDATE rex_episodes SET {', '.join(sets)} WHERE id = ?", tuple(params))
    return True


def delete_rex_memory(memory_id: int) -> bool:
    """Delete a single Rex memory."""
    rex_db.execute("DELETE FROM rex_episodes WHERE id = ?", (int(memory_id),))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# People (people.db)
# ─────────────────────────────────────────────────────────────────────────────

def list_people() -> list[dict]:
    """All known people with the fields the list view shows, ordered by name."""
    rows = db.fetchall(
        "SELECT id, name, nickname, friendship_tier, visit_count, last_seen, "
        "warmth_score, antagonism_score FROM people ORDER BY name COLLATE NOCASE"
    )
    return [dict(r) for r in rows]


def create_person(name: str) -> Optional[int]:
    """Create a new person from the GUI form. Returns the new id, or None if the
    name was rejected (empty / not a usable display name)."""
    return people_mem.enroll_person(name)


def delete_person(person_id: int) -> bool:
    """Delete a person and ALL of their rows across every table."""
    people_mem.delete_person(int(person_id))
    return True


# Person columns the editor may write directly. Excludes name (use rename_person, which
# validates) and the derived/relationship score columns (edited via their own paths).
_EDITABLE_PERSON_FIELDS = {
    "nickname", "height", "build", "hair_color", "hair_style", "skin_color",
    "age_range", "age_category", "notable_features",
}


def get_person_detail(person_id: int) -> Optional[dict]:
    """Everything stored about one person: the people row plus facts, interests,
    and preferences — for the person editor."""
    person = people_mem.get_person(int(person_id))
    if not person:
        return None
    facts = [dict(r) for r in db.fetchall(
        "SELECT * FROM person_facts WHERE person_id = ? ORDER BY category, key",
        (int(person_id),),
    )]
    interests = [dict(r) for r in db.fetchall(
        "SELECT * FROM person_interests WHERE person_id = ? ORDER BY name",
        (int(person_id),),
    )]
    preferences = [dict(r) for r in db.fetchall(
        "SELECT * FROM person_preferences WHERE person_id = ? ORDER BY domain, key",
        (int(person_id),),
    )]
    biometrics = {
        "face": people_mem.count_biometrics(int(person_id), "face"),
        "voice": people_mem.count_biometrics(int(person_id), "voice"),
    }
    return {
        "person": person,
        "facts": facts,
        "interests": interests,
        "preferences": preferences,
        "biometrics": biometrics,
    }


def clear_biometrics(person_id: int, kind: str) -> bool:
    """Delete a person's stored face encodings or voiceprints (kind: 'face'|'voice').
    Useful to wipe a mis-enrolled biometric that's causing wrong recognition."""
    if kind not in ("face", "voice"):
        return False
    if kind == "voice":
        from audio import voice_score
        kind = voice_score.biometric_type()
    db.execute(
        "DELETE FROM biometrics WHERE person_id = ? AND type = ?",
        (int(person_id), kind),
    )
    return True


def update_person_fields(person_id: int, *, name: Optional[str] = None, **fields) -> bool:
    """Update a person's editable fields. `name` is routed through rename_person (which
    validates + keeps aliases consistent); other whitelisted columns are written directly."""
    ok = True
    if name is not None and str(name).strip():
        ok = people_mem.rename_person(int(person_id), str(name).strip()) and ok
    sets: list[str] = []
    params: list[object] = []
    for col, val in fields.items():
        if col not in _EDITABLE_PERSON_FIELDS:
            continue
        sets.append(f"{col} = ?")
        params.append(None if val is None else str(val).strip())
    if sets:
        params.append(int(person_id))
        db.execute(f"UPDATE people SET {', '.join(sets)} WHERE id = ?", tuple(params))
    return ok


# ── Facts ────────────────────────────────────────────────────────────────────

def add_person_fact(
    person_id: int, category: str, key: str, value: str, *, importance: float = 0.5,
) -> bool:
    """Add (or strengthen) a fact via the normal fact path, marked user-sourced."""
    category = (category or "other").strip() or "other"
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return False
    facts_mem.add_fact(
        int(person_id), category, key, value,
        source="user_edited", confidence=0.95, importance=float(importance),
    )
    return True


def update_fact(
    fact_id: int,
    *,
    category: Optional[str] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
    importance: Optional[float] = None,
) -> bool:
    """Edit one fact in place by id."""
    sets: list[str] = []
    params: list[object] = []
    if category is not None:
        sets.append("category = ?"); params.append(str(category).strip() or "other")
    if key is not None:
        sets.append("key = ?"); params.append(str(key).strip())
    if value is not None:
        sets.append("value = ?"); params.append(str(value).strip())
    if importance is not None:
        try:
            params.append(max(0.0, min(1.0, float(importance))))
            sets.append("importance = ?")
        except (TypeError, ValueError):
            pass
    if not sets:
        return False
    params.append(int(fact_id))
    db.execute(f"UPDATE person_facts SET {', '.join(sets)} WHERE id = ?", tuple(params))
    return True


def delete_fact(fact_id: int) -> bool:
    db.execute("DELETE FROM person_facts WHERE id = ?", (int(fact_id),))
    return True


# ── Interests / preferences (view + delete by id) ────────────────────────────

def delete_interest(interest_id: int) -> bool:
    db.execute("DELETE FROM person_interests WHERE id = ?", (int(interest_id),))
    return True


def delete_preference(preference_id: int) -> bool:
    db.execute("DELETE FROM person_preferences WHERE id = ?", (int(preference_id),))
    return True
