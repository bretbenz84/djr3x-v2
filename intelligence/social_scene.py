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


FIRST_GREETING_STEERING_PHRASES = (
    "What are you up to today, besides providing me questionable supervision?",
    "What do you want to talk about before I start guessing and embarrass us both?",
    "What mission are we pretending is important today?",
    "What are you working on, plotting, building, styling, breaking, or otherwise making my problem?",
    "What corner of your organic life are we discussing first?",
    "What topic gets the honor of my extremely limited patience today?",
    "What are we talking about today: your plans, your hobbies, or my obvious brilliance?",
    "What are you into today? Give me a topic before I start interviewing the furniture.",
)


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


@dataclass(frozen=True)
class UnknownGroupContext:
    unknown_count: int
    known_count: int
    addressee: str
    label: str
    directive: str


@dataclass(frozen=True)
class ConversationCastContext:
    group_label: str
    addressee: str
    current_speaker: Optional[VisiblePerson]
    known: tuple[VisiblePerson, ...]
    unknown_count: int
    directive: str


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


def conversation_cast_context(
    snapshot: Optional[dict] = None,
    *,
    current_person_id: Optional[int] = None,
    current_person_name: Optional[str] = None,
) -> ConversationCastContext:
    """Return a prompt-ready representation of the visible conversational cast."""
    scene = from_snapshot(snapshot)
    current = _current_visible_person(
        scene,
        current_person_id=current_person_id,
        current_person_name=current_person_name,
    )
    group_label = visible_group_label(scene)

    if current and len(scene.known) >= 2:
        addressee = f"{current.first_name} primarily; visible group: {group_label}"
    elif current:
        addressee = current.first_name
    elif len(scene.known) >= 2:
        addressee = group_label
    elif scene.known:
        addressee = scene.known[0].first_name
    elif scene.unknown_count == 1:
        addressee = "unknown guest"
    elif scene.unknown_count > 1:
        addressee = "unknown guests"
    else:
        addressee = "the room"

    lines = [
        f"Conversation cast: addressee={addressee}; visible_group={group_label}; "
        f"known_visible={len(scene.known)}; unknown_visible={scene.unknown_count}."
    ]
    if current:
        lines.append(f"Current speaker / primary addressee: {current.first_name}.")

    if scene.known:
        parts = []
        for person in scene.known[:5]:
            pronouns = _pronouns_for_person(person.person_id)
            suffix = f" ({pronouns})" if pronouns else ""
            marker = " [current speaker]" if current and person.person_id == current.person_id else ""
            parts.append(f"{person.first_name}{suffix}{marker}")
        lines.append("Visible known participants: " + ", ".join(parts) + ".")

    if len(scene.known) >= 2:
        others = [
            p.first_name for p in scene.known
            if not current or p.person_id != current.person_id
        ]
        if others:
            lines.append(
                "Referent candidates besides the speaker: " + ", ".join(others) + "."
            )

    if scene.unknown_count:
        noun = "person" if scene.unknown_count == 1 else "people"
        lines.append(
            f"Unknown visible participants: {scene.unknown_count} {noun}. "
            "Use 'the mystery guest' / 'the newcomers' until names are known."
        )

    lines.append(
        "Pronoun and group-address rules: resolve he/she/they only when the "
        "referent is obvious from the latest user turn and the visible cast. "
        "If more than one person fits, use names or ask a tiny clarification. "
        "When addressing multiple humans, use 'you two', 'you all', or names; "
        "do not let singular 'you' blur who Rex is roasting. Respect any "
        "pronoun facts shown above."
    )

    return ConversationCastContext(
        group_label=group_label,
        addressee=addressee,
        current_speaker=current,
        known=scene.known,
        unknown_count=scene.unknown_count,
        directive="\n".join(lines),
    )


def unknown_group_context(
    snapshot: Optional[dict] = None,
    *,
    current_person_id: Optional[int] = None,
    current_person_name: Optional[str] = None,
) -> Optional[UnknownGroupContext]:
    """
    Return a compact social instruction for visible unknown people.

    This is intentionally about conversational shape, not enrollment mechanics.
    Interaction.py owns the actual identity capture; prompt/governor layers use
    this to keep group banter natural and to avoid losing the identity question.
    """
    scene = from_snapshot(snapshot)
    if scene.unknown_count <= 0:
        return None

    current = _current_visible_person(
        scene,
        current_person_id=current_person_id,
        current_person_name=current_person_name,
    )

    known_label = visible_group_label(scene) if scene.known else "the room"
    mystery_word = "guest" if scene.unknown_count == 1 else "guests"
    face_word = "face" if scene.unknown_count == 1 else "faces"
    verb = "is" if scene.unknown_count == 1 else "are"

    if current is not None:
        addressee = (
            f"{current.first_name} and the mystery {mystery_word}"
            if scene.unknown_count == 1
            else f"{current.first_name} and the mystery lineup"
        )
        directive = (
            "Primary purpose: urgent group identity handoff. "
            f"There are {scene.unknown_count} unfamiliar {face_word} visible "
            f"while Rex is talking with {current.first_name}. Ask who the "
            f"unfamiliar {mystery_word} {verb}, get name(s), and ask how they know "
            f"{current.first_name} or the group. Keep it one witty party-host "
            "question, warm but lightly suspicious, like a droid checking the "
            "guest list. This identity question may bypass the optional question "
            "budget. Do not pivot to unrelated small talk."
        )
        return UnknownGroupContext(
            unknown_count=scene.unknown_count,
            known_count=len(scene.known),
            addressee=addressee,
            label=f"{current.first_name} plus {scene.unknown_count} unknown",
            directive=directive,
        )

    if scene.known:
        addressee = f"{known_label} and the mystery {mystery_word}"
        directive = (
            "Primary purpose: urgent group identity handoff. "
            f"Rex can see known people ({known_label}) plus "
            f"{scene.unknown_count} unfamiliar {face_word}. Address the room as "
            "a group and ask who the newcomer(s) are, what name(s) Rex should "
            "file, and who brought them into this questionable social orbit. "
            "Make it one witty party-host question. This identity question may "
            "bypass the optional question budget."
        )
        return UnknownGroupContext(
            unknown_count=scene.unknown_count,
            known_count=len(scene.known),
            addressee=addressee,
            label=f"{known_label} plus {scene.unknown_count} unknown",
            directive=directive,
        )

    addressee = "unknown guest" if scene.unknown_count == 1 else "unknown guests"
    directive = (
        "Primary purpose: urgent group identity handoff. "
        f"Rex can see {scene.unknown_count} unfamiliar {face_word} and no known "
        "person to anchor the introduction. Greet them directly and ask what "
        "name(s) to call them. If there are multiple people, ask for quick "
        "names without turning it into a census. Keep it funny, short, and "
        "welcoming. This identity question may bypass the optional question "
        "budget."
    )
    return UnknownGroupContext(
        unknown_count=scene.unknown_count,
        known_count=0,
        addressee=addressee,
        label=addressee,
        directive=directive,
    )


def _current_visible_person(
    scene: SocialScene,
    *,
    current_person_id: Optional[int] = None,
    current_person_name: Optional[str] = None,
) -> Optional[VisiblePerson]:
    if current_person_id is not None:
        for person in scene.known:
            if person.person_id == current_person_id:
                return person
    if current_person_name:
        first = current_person_name.split()[0]
        return VisiblePerson(
            person_id=int(current_person_id or -1),
            name=current_person_name,
            first_name=first,
            slot_id="current",
        )
    return None


def _pronouns_for_person(person_id: int) -> Optional[str]:
    try:
        from memory import facts as facts_memory
        facts = facts_memory.get_facts(person_id)
    except Exception:
        return None
    for fact in facts:
        key = (fact.get("key") or "").lower()
        category = (fact.get("category") or "").lower()
        value = (fact.get("value") or "").strip()
        if not value:
            continue
        haystack = f"{category} {key}".replace("_", " ")
        if "pronoun" in haystack:
            return value
    return None


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
    steering_examples = "; ".join(FIRST_GREETING_STEERING_PHRASES)
    return (
        f"You just started up and can see these known people together: {names}. "
        f"The natural group label is: {label}. Greet them as a group in one "
        f"short in-character Rex line. If the label mentions a relationship, "
        f"use it naturally. End with ONE conversation-steering question, varying "
        f"between 'what are you up to today?' and 'what do you want to talk about?' "
        f"in Rex's snarky DJ-R3X voice. Example endings: {steering_examples}. "
        f"Do NOT bring up "
        f"individual memories, trips, old plans, or private callbacks in this "
        f"group startup greeting."
    )
