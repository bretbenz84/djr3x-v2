"""
memory/person_summary.py — Compact review summaries for a person's memory.
"""

from __future__ import annotations

from memory import emotional_events
from memory import events
from memory import facts
from memory import interests
from memory import people
from memory import preferences
from memory import social


def _join_limited(items: list[str], limit: int = 3) -> str:
    cleaned = [item.strip() for item in items if item and item.strip()]
    if not cleaned:
        return ""
    shown = cleaned[:limit]
    extra = len(cleaned) - len(shown)
    text = "; ".join(shown)
    if extra > 0:
        text += f"; plus {extra} more"
    return text


def summarize_for_review(
    person_id: int,
    *,
    include_sensitive: bool = False,
    fact_limit: int = 6,
) -> str:
    """Return a short, trustworthy Rex-voice memory review for one person."""
    person = people.get_person(int(person_id))
    name = (person or {}).get("name") or "this biological"

    fact_rows = facts.get_prompt_worthy_facts(int(person_id), limit=fact_limit)
    fact_lines = [
        facts.format_fact_for_prompt(row)
        for row in fact_rows
        if row.get("category") not in {"skin_color"}
    ]

    pref_rows = preferences.get_preferences_for_prompt(int(person_id), limit=5)
    pref_lines = [preferences.format_preference_for_prompt(row) for row in pref_rows]

    interest_rows = interests.get_interests_for_prompt(int(person_id), limit=5)
    interest_lines = [interests.format_interest_for_prompt(row) for row in interest_rows]

    relationship_line = social.summarize_for_prompt(int(person_id), name) or ""

    event_rows = events.get_open_events(int(person_id))[:3]
    event_lines = []
    for ev in event_rows:
        event_name = ev.get("event_name") or "upcoming thing"
        event_date = ev.get("event_date")
        event_lines.append(f"{event_name} on {event_date}" if event_date else event_name)

    sensitive_lines: list[str] = []
    if include_sensitive:
        try:
            for ev in emotional_events.get_active_events(int(person_id), limit=3):
                desc = ev.get("description") or ev.get("category") or "sensitive event"
                sensitive_lines.append(str(desc))
        except Exception:
            sensitive_lines = []

    chunks = [f"Memory banks on {name}:"]
    if fact_lines:
        chunks.append(f"facts: {_join_limited(fact_lines, 4)}.")
    if pref_lines:
        chunks.append(f"preferences: {_join_limited(pref_lines, 3)}.")
    if interest_lines:
        chunks.append(f"interests: {_join_limited(interest_lines, 3)}.")
    if relationship_line:
        chunks.append(f"relationships: {relationship_line}.")
    if event_lines:
        chunks.append(f"upcoming: {_join_limited(event_lines, 3)}.")
    if sensitive_lines:
        chunks.append(f"sensitive: {_join_limited(sensitive_lines, 2)}.")

    if len(chunks) == 1:
        chunks.append("nothing durable yet. Tragic, but fixable.")
    else:
        chunks.append("That is the useful version, minus the dramatic lighting.")
    return " ".join(chunks)
