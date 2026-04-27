"""
intelligence/social_scene.py - lightweight visible-group reasoning.

This module deliberately stays small: it turns the current WorldState people
list plus the social relationship graph into a compact description other
intelligence layers can use without reimplementing relationship lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from world_state import world_state
from memory import social as social_memory


_SYMMETRIC_PAIR_LABELS = {
    "partner": "partners",
    "spouse": "couple",
    "wife": "couple",
    "husband": "couple",
    "girlfriend": "couple",
    "boyfriend": "couple",
    "fiance": "engaged duo",
    "fiancee": "engaged duo",
    "friend": "friends",
    "bestfriend": "best friends",
    "best_friend": "best friends",
    "sibling": "siblings",
    "brother": "siblings",
    "sister": "siblings",
    "cousin": "cousins",
    "coworker": "work duo",
    "co_worker": "work duo",
    "colleague": "work duo",
    "roommate": "roommates",
    "flatmate": "roommates",
    "neighbor": "neighbors",
    "neighbour": "neighbors",
}

_DIRECTIONAL_PAIR_LABELS = {
    "father": "father-son duo",
    "dad": "father-son duo",
    "mother": "mother-child duo",
    "mom": "mother-child duo",
    "parent": "parent-child duo",
    "aunt": "family duo",
    "uncle": "family duo",
    "boss": "work duo",
    "supervisor": "work duo",
    "manager": "work duo",
    "employee": "work duo",
    "child": "family duo",
    "son": "family duo",
    "daughter": "family duo",
    "owner": "person-and-pet duo",
    "dog": "person-and-dog duo",
    "cat": "person-and-cat duo",
    "pet": "person-and-pet duo",
}


@dataclass(frozen=True)
class VisiblePerson:
    person_id: int
    name: str
    first_name: str
    slot_id: str


@dataclass(frozen=True)
class SocialScene:
    known: tuple[VisiblePerson, ...]
    unknown_count: int
    crowd_count: int

    @property
    def is_group(self) -> bool:
        return len(self.known) >= 2

    @property
    def signature(self) -> str:
        ids = "-".join(str(p.person_id) for p in self.known)
        return ids or f"unknowns:{self.unknown_count}"

    @property
    def first_names(self) -> list[str]:
        return [p.first_name for p in self.known]


def from_snapshot(snapshot: Optional[dict] = None) -> SocialScene:
    snap = snapshot if snapshot is not None else world_state.snapshot()
    known: list[VisiblePerson] = []
    unknown_count = 0
    for idx, person in enumerate(snap.get("people", []) or []):
        pid = person.get("person_db_id")
        name = person.get("face_id") or person.get("voice_id") or ""
        if pid is None:
            unknown_count += 1
            continue
        if not name:
            name = f"person_{pid}"
        first = name.split()[0] if name else f"person_{pid}"
        known.append(
            VisiblePerson(
                person_id=int(pid),
                name=name,
                first_name=first,
                slot_id=str(person.get("id") or f"person_{idx + 1}"),
            )
        )
    known.sort(key=lambda p: p.person_id)
    try:
        crowd_count = int((snap.get("crowd") or {}).get("count", len(known) + unknown_count))
    except Exception:
        crowd_count = len(known) + unknown_count
    return SocialScene(tuple(known), unknown_count, crowd_count)


def _pair_relationship(a: VisiblePerson, b: VisiblePerson) -> Optional[dict]:
    try:
        edges = social_memory.get_between(a.person_id, b.person_id)
    except Exception:
        return None
    return edges[0] if edges else None


def pair_label(a: VisiblePerson, b: VisiblePerson) -> str:
    """Return a natural short label for two visible known people."""
    edge = _pair_relationship(a, b)
    if not edge:
        return f"{a.first_name} and {b.first_name}"

    rel = (edge.get("relationship") or "").replace("_", " ").strip().lower()
    rel_key = rel.replace(" ", "_")
    from_id = int(edge.get("from_person_id") or 0)
    to_id = int(edge.get("to_person_id") or 0)
    from_name = edge.get("from_name") or ""
    to_name = edge.get("to_name") or ""
    from_first = (from_name.split() or [""])[0]
    to_first = (to_name.split() or [""])[0]

    if rel_key in {"father", "dad"}:
        child = from_first if from_id != b.person_id else b.first_name
        father = to_first if to_id != a.person_id else a.first_name
        return f"father-son duo {child} and {father}"
    if rel_key in {"mother", "mom"}:
        child = from_first if from_id != b.person_id else b.first_name
        mother = to_first if to_id != a.person_id else a.first_name
        return f"mother-child duo {child} and {mother}"
    if rel_key == "parent":
        child = from_first if from_id != b.person_id else b.first_name
        parent = to_first if to_id != a.person_id else a.first_name
        return f"parent-child duo {child} and {parent}"

    if rel_key in _SYMMETRIC_PAIR_LABELS:
        return f"{_SYMMETRIC_PAIR_LABELS[rel_key]} {a.first_name} and {b.first_name}"
    if rel_key in _DIRECTIONAL_PAIR_LABELS:
        return f"{_DIRECTIONAL_PAIR_LABELS[rel_key]} {a.first_name} and {b.first_name}"
    if rel_key:
        return f"{a.first_name} and {b.first_name} ({rel.replace('_', ' ')})"
    return f"{a.first_name} and {b.first_name}"


def visible_group_label(scene: SocialScene) -> str:
    if len(scene.known) == 2:
        return pair_label(scene.known[0], scene.known[1])
    names = scene.first_names
    if len(names) <= 1:
        return names[0] if names else "the room"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def startup_group_prompt(scene: SocialScene) -> str:
    label = visible_group_label(scene)
    names = ", ".join(scene.first_names)
    return (
        f"You just started up and can see these known people together: {names}. "
        f"The natural group label is: {label}. Greet them as a group in one "
        f"short in-character Rex line. If the label mentions a relationship, "
        f"use it naturally. Ask how they are all doing today. Do NOT bring up "
        f"individual memories, trips, old plans, or private callbacks in this "
        f"group startup greeting."
    )
