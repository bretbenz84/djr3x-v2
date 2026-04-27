"""
intelligence/memory_query.py — resolve and assemble person-memory answers.

This module is intentionally LLM-free: it finds the target person and gathers
grounded memory snippets. interaction.py then asks the conversational LLM to
phrase those snippets in Rex's voice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Optional

from memory import conversations as conv_memory
from memory import database as db
from memory import emotional_events
from memory import events as events_memory
from memory import facts as facts_memory
from memory import people as people_memory
from memory import social as social_memory


_SELF_RE = re.compile(
    r"\b("
    r"about\s+me|about\s+myself|know\s+about\s+me|know\s+about\s+myself|"
    r"remember\s+about\s+me|remember\s+about\s+myself|tell\s+me\s+about\s+me|"
    r"tell\s+me\s+about\s+myself|my\s+memory|what\s+do\s+you\s+remember"
    r")\b",
    re.IGNORECASE,
)

_RELATION_RE = re.compile(
    r"\bmy\s+("
    r"best\s+friend|partner|spouse|wife|husband|girlfriend|boyfriend|"
    r"fianc[eé]e?|father|dad|mother|mom|parent|son|daughter|child|"
    r"brother|sister|sibling|friend|boss|manager|supervisor|"
    r"employee|coworker|co[-\s]?worker|colleague|roommate|neighbor|neighbour"
    r")\b",
    re.IGNORECASE,
)

_NAMED_PATTERNS = (
    re.compile(
        r"\b(?:tell\s+me\s+(?:what\s+you\s+know\s+)?about|"
        r"what\s+do\s+you\s+(?:know|remember)\s+about|"
        r"what\s+have\s+i\s+told\s+you\s+about|"
        r"what\s+do\s+we\s+know\s+about|"
        r"who\s+is)\s+(.+?)\s*[?.!]*$",
        re.IGNORECASE,
    ),
    re.compile(r"\babout\s+([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,3})\b"),
)

_BAD_TARGETS = {
    "me", "myself", "you", "yourself", "my partner", "my friend", "my spouse",
    "my wife", "my husband", "my girlfriend", "my boyfriend", "my dad",
    "my father", "my mom", "my mother", "memory", "memories",
}

_FACT_CATEGORY_EXCLUDES = {"appearance", "relationship"}
_FACT_KEY_EXCLUDES = {
    # Event memories have status-aware storage in person_events /
    # person_emotional_events; old extracted facts can linger after cancellation.
    "event",
    "upcoming_event",
    "skin_color",
}

_RELATION_ALIASES = {
    "best friend": {"best_friend", "bestfriend", "friend"},
    "partner": {"partner", "spouse", "wife", "husband", "girlfriend", "boyfriend", "fiance", "fiancee"},
    "spouse": {"spouse", "partner", "wife", "husband"},
    "wife": {"wife", "spouse", "partner"},
    "husband": {"husband", "spouse", "partner"},
    "girlfriend": {"girlfriend", "partner"},
    "boyfriend": {"boyfriend", "partner"},
    "fiance": {"fiance", "fiancee", "partner"},
    "fiancee": {"fiancee", "fiance", "partner"},
    "father": {"father", "dad", "parent"},
    "dad": {"dad", "father", "parent"},
    "mother": {"mother", "mom", "parent"},
    "mom": {"mom", "mother", "parent"},
    "parent": {"parent", "father", "mother", "dad", "mom"},
    "son": {"son", "child"},
    "daughter": {"daughter", "child"},
    "child": {"child", "son", "daughter"},
    "brother": {"brother", "sibling"},
    "sister": {"sister", "sibling"},
    "sibling": {"sibling", "brother", "sister"},
    "friend": {"friend", "best_friend", "bestfriend"},
    "boss": {"boss", "manager", "supervisor"},
    "manager": {"manager", "boss", "supervisor"},
    "supervisor": {"supervisor", "boss", "manager"},
    "employee": {"employee"},
    "coworker": {"coworker", "co_worker", "colleague"},
    "co_worker": {"coworker", "co_worker", "colleague"},
    "colleague": {"colleague", "coworker", "co_worker"},
    "roommate": {"roommate", "flatmate"},
    "neighbor": {"neighbor", "neighbour"},
    "neighbour": {"neighbour", "neighbor"},
}


@dataclass
class MemoryTarget:
    person_id: Optional[int] = None
    name: Optional[str] = None
    mode: str = "unknown"
    detail: str = ""
    relation_label: Optional[str] = None
    ambiguous_names: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.person_id is not None


@dataclass
class MemoryContext:
    target: MemoryTarget
    sections: list[str]
    has_memory: bool

    def as_prompt_text(self) -> str:
        return "\n".join(self.sections) if self.sections else "(no memory found)"


def _normalize_label(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_").replace("é", "e")


def _normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (value or "").lower()))


def _clean_named_target(value: str) -> str:
    text = (value or "").strip(" \t\r\n?.!,;:\"'`()[]{}")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:please|again|for\s+me)$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _all_people() -> list[dict]:
    rows = db.fetchall(
        """
        SELECT p.*,
               SUM(CASE WHEN b.type = 'face' THEN 1 ELSE 0 END) AS face_count,
               SUM(CASE WHEN b.type = 'voice' THEN 1 ELSE 0 END) AS voice_count
        FROM people p
        LEFT JOIN biometrics b ON b.person_id = p.id
        GROUP BY p.id
        ORDER BY p.last_seen DESC, p.id ASC
        """
    )
    return [dict(r) for r in rows]


def _find_person_fuzzy(name: str) -> Optional[dict]:
    query = _normalize_name(name)
    if len(query) < 4:
        return None
    best: tuple[float, dict] | None = None
    for person in _all_people():
        stored = _normalize_name(person.get("name") or "")
        if not stored:
            continue
        ratio = SequenceMatcher(None, query, stored).ratio()
        stored_first = stored.split()[0] if stored.split() else stored
        query_first = query.split()[0] if query.split() else query
        first_ratio = SequenceMatcher(None, query_first, stored_first).ratio()
        score = max(ratio, first_ratio - 0.05)
        if best is None or score > best[0]:
            best = (score, person)
    if best and best[0] >= 0.82:
        return best[1]
    return None


def _extract_named_target(text: str) -> Optional[str]:
    for pattern in _NAMED_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        candidate = _clean_named_target(match.group(1))
        if not candidate:
            continue
        if _normalize_name(candidate) in _BAD_TARGETS:
            continue
        if _RELATION_RE.fullmatch(candidate):
            continue
        return candidate
    return None


def _resolve_person_name(name: str) -> MemoryTarget:
    person = people_memory.find_person_by_name(name) or _find_person_fuzzy(name)
    if not person:
        return MemoryTarget(mode="named", name=name, detail="no_person_match")
    return MemoryTarget(
        person_id=int(person["id"]),
        name=person.get("name") or name,
        mode="named",
        detail=f"matched named target '{name}'",
    )


def _resolve_relationship(current_person_id: int, relation_phrase: str) -> MemoryTarget:
    requested = _normalize_label(relation_phrase)
    labels = _RELATION_ALIASES.get(requested, {requested})
    matches: dict[int, dict] = {}

    for edge in social_memory.get_outbound(current_person_id):
        if _normalize_label(edge.get("relationship") or "") in labels:
            matches[int(edge["to_person_id"])] = edge

    # Directional relationships are sometimes only stored from the other
    # person's perspective. Include inbound edges when their label matches.
    for edge in social_memory.get_inbound(current_person_id):
        if _normalize_label(edge.get("relationship") or "") in labels:
            matches[int(edge["from_person_id"])] = edge

    if not matches:
        return MemoryTarget(
            mode="relationship",
            relation_label=requested,
            detail="no_relationship_match",
        )

    if len(matches) > 1:
        names: list[str] = []
        for other_id, edge in matches.items():
            name = edge.get("to_name") or edge.get("from_name")
            if not name:
                person = people_memory.get_person(other_id) or {}
                name = person.get("name") or f"person {other_id}"
            names.append(str(name))
        return MemoryTarget(
            mode="relationship",
            relation_label=requested,
            detail="ambiguous_relationship_match",
            ambiguous_names=sorted(set(names)),
        )

    other_id, edge = next(iter(matches.items()))
    person = people_memory.get_person(other_id) or {}
    return MemoryTarget(
        person_id=other_id,
        name=person.get("name") or edge.get("to_name") or edge.get("from_name") or f"person {other_id}",
        mode="relationship",
        relation_label=requested,
        detail=f"resolved my {relation_phrase}",
    )


def resolve_target(text: str, current_person_id: Optional[int]) -> MemoryTarget:
    """Resolve the person a memory question is asking about."""
    raw = text or ""

    rel_match = _RELATION_RE.search(raw)
    if rel_match:
        if current_person_id is None:
            return MemoryTarget(
                mode="relationship",
                relation_label=_normalize_label(rel_match.group(1)),
                detail="relationship_query_without_current_person",
            )
        return _resolve_relationship(int(current_person_id), rel_match.group(1))

    named = _extract_named_target(raw)
    if named:
        return _resolve_person_name(named)

    if current_person_id is not None and (
        _SELF_RE.search(raw)
        or re.search(r"\bwhat\s+do\s+you\s+(?:know|remember)\b", raw, re.IGNORECASE)
    ):
        person = people_memory.get_person(int(current_person_id)) or {}
        return MemoryTarget(
            person_id=int(current_person_id),
            name=person.get("name") or "you",
            mode="self",
            detail="current speaker",
        )

    return MemoryTarget(mode="self", detail="self_query_without_current_person")


def _format_fact_lines(person_id: int, *, limit: int = 20) -> list[str]:
    facts = []
    for fact in facts_memory.get_prompt_facts(person_id, limit=limit * 2):
        category = str(fact.get("category") or "").strip().lower()
        key = str(fact.get("key") or "").strip().lower()
        if category in _FACT_CATEGORY_EXCLUDES or key in _FACT_KEY_EXCLUDES:
            continue
        facts.append(fact)
        if len(facts) >= limit:
            break
    return [facts_memory.format_fact_for_prompt(fact) for fact in facts]


def _format_events(person_id: int, *, limit: int = 5) -> list[str]:
    out: list[str] = []
    for event in events_memory.get_open_events(person_id)[:limit]:
        name = event.get("event_name") or "event"
        date = event.get("event_date") or "no date"
        notes = event.get("event_notes") or ""
        line = f"{name} ({date})"
        if notes:
            line += f": {notes}"
        out.append(line)
    return out


def _format_conversations(person_id: int, *, limit: int = 3) -> list[str]:
    out: list[str] = []
    for conv in conv_memory.get_conversation_history(person_id, limit=limit):
        summary = conv.get("summary") or ""
        if not summary:
            continue
        date = conv.get("session_date") or "unknown date"
        tone = conv.get("emotion_tone") or ""
        line = f"{date}: {summary}"
        if tone:
            line += f" (tone: {tone})"
        out.append(line)
    return out


def _format_emotional_events(person_id: int, *, limit: int = 3) -> list[str]:
    out: list[str] = []
    for ev in emotional_events.get_active_events(person_id, limit=limit):
        desc = ev.get("description") or ""
        if not desc:
            continue
        valence = ev.get("valence") or "unknown"
        out.append(f"{desc} (valence: {valence})")
    return out


def build_context(target: MemoryTarget, requester_person_id: Optional[int]) -> MemoryContext:
    if target.person_id is None:
        return MemoryContext(target=target, sections=[], has_memory=False)

    person_id = int(target.person_id)
    person = people_memory.get_person(person_id) or {}
    name = target.name or person.get("name") or f"person {person_id}"
    sections = [
        f"Target person: {name} (person_id={person_id}, tier={person.get('friendship_tier') or 'unknown'})"
    ]

    if requester_person_id is not None and int(requester_person_id) != person_id:
        between = social_memory.get_between(int(requester_person_id), person_id)
        if between:
            lines = []
            for edge in between[:4]:
                lines.append(
                    f"{edge.get('from_name') or edge.get('from_person_id')} -> "
                    f"{edge.get('to_name') or edge.get('to_person_id')}: "
                    f"{edge.get('relationship')}"
                )
            sections.append("Relationship to current speaker:\n- " + "\n- ".join(lines))

    rel_summary = social_memory.summarize_for_prompt(person_id, name)
    if rel_summary:
        sections.append(f"Known relationships: {rel_summary}")

    facts = _format_fact_lines(person_id)
    if facts:
        sections.append("Stored facts:\n- " + "\n- ".join(facts))

    events = _format_events(person_id)
    if events:
        sections.append("Open or upcoming event memories:\n- " + "\n- ".join(events))

    conversations = _format_conversations(person_id)
    if conversations:
        sections.append("Recent conversation summaries:\n- " + "\n- ".join(conversations))

    emotional = _format_emotional_events(person_id)
    if emotional:
        sections.append("Active emotional memories:\n- " + "\n- ".join(emotional))

    return MemoryContext(target=target, sections=sections, has_memory=len(sections) > 1)


def build_response_prompt(raw_text: str, context: MemoryContext) -> str:
    target = context.target
    target_name = target.name or "that person"
    relation_note = ""
    if target.mode == "relationship" and target.relation_label:
        relation_note = f"The user referred to this person as their {target.relation_label}. "
    return (
        f"The user asked a memory question: {raw_text!r}.\n"
        f"Resolved target: {target_name}. {relation_note}\n"
        "Use ONLY the retrieved memory below. Do not invent facts, relationships, "
        "or history. If memory is thin, say that clearly instead of padding.\n\n"
        f"Retrieved memory:\n{context.as_prompt_text()}\n\n"
        "Answer in Rex's voice in 1-3 short sentences. For 'tell me about myself', "
        "speak to the user directly as 'you'. For named/relationship targets, name "
        "the person. Be helpful, not evasive."
    )
