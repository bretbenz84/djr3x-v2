"""
friendship_patterns.py - lightweight learning for social style and running jokes.

This module keeps "real friendship" behavior on top of existing person_facts:
preference facts for style, and inside_joke facts for shared bits. It avoids a
new schema while still giving the prompt durable, per-person social memory.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from memory import facts as facts_memory
from memory import people as people_memory
from world_state import world_state

_log = logging.getLogger(__name__)


_STYLE_PATTERNS: tuple[tuple[re.Pattern, str, str, float], ...] = (
    (
        re.compile(r"\b(?:roast me|you can roast me|keep roasting|i like (?:the )?roasts?|that roast was funny)\b", re.I),
        "roast_style",
        "likes light roasts",
        0.88,
    ),
    (
        re.compile(r"\b(?:don'?t roast me|do not roast me|stop roasting|that was rude|that was mean|too mean|too harsh|not funny)\b", re.I),
        "roast_style",
        "dislikes sharp roasts",
        0.9,
    ),
    (
        re.compile(r"\b(?:less banter|just answer|be direct|straight answer|no jokes?|skip the bit|get to the point)\b", re.I),
        "answer_style",
        "prefers direct answers",
        0.86,
    ),
    (
        re.compile(r"\b(?:i like the banter|keep the banter|mess with me|joke with me|tease me)\b", re.I),
        "banter_style",
        "likes playful banter",
        0.84,
    ),
    (
        re.compile(r"\b(?:i like star wars|love star wars|keep the star wars|more star wars|star wars bits are funny)\b", re.I),
        "star_wars_bits",
        "likes Star Wars references",
        0.86,
    ),
    (
        re.compile(r"\b(?:less star wars|stop with the star wars|no more star wars|enough star wars)\b", re.I),
        "star_wars_bits",
        "prefers fewer Star Wars references",
        0.88,
    ),
    (
        re.compile(r"\b(?:nice callback|good callback|i like when you remember|bring that up again)\b", re.I),
        "callback_style",
        "likes occasional callbacks",
        0.82,
    ),
    (
        re.compile(r"\b(?:too many callbacks|stop bringing that up|don'?t keep bringing that up|you mention that too much)\b", re.I),
        "callback_style",
        "prefers callback restraint",
        0.88,
    ),
)

_JOKE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"\b(?:our|the)\s+(?:inside joke|running joke|bit)\s+(?:is|about|that)?\s*(?P<topic>[^.?!]{3,90})",
        re.I,
    ),
    re.compile(
        r"\b(?:we always joke about|we keep joking about|remember the joke about)\s+(?P<topic>[^.?!]{3,90})",
        re.I,
    ),
    re.compile(
        r"\bthat'?s\s+(?:our|the)\s+(?:bit|inside joke|running joke)\s*(?P<topic>[^.?!]{0,90})",
        re.I,
    ),
)

_TRAILING_FILLER = re.compile(r"\s+(?:right|okay|ok|lol|haha)\s*$", re.I)


def learn_from_turn(person_id: Optional[int], text: str) -> int:
    """Persist social-style preferences and explicit running jokes from a turn."""
    if not isinstance(person_id, int) or not text or not text.strip():
        return 0

    saved = 0
    for pattern, key, value, confidence in _STYLE_PATTERNS:
        if pattern.search(text):
            facts_memory.add_fact(
                person_id,
                "preference",
                key,
                value,
                source="social_style",
                confidence=confidence,
            )
            saved += 1

    joke = _extract_running_joke(text)
    if joke:
        scope = _visible_scope(person_id)
        value = f"with {scope}: {joke}" if scope else joke
        facts_memory.add_fact(
            person_id,
            "inside_joke",
            _joke_key(value),
            value,
            source="running_joke",
            confidence=0.82,
        )
        saved += 1

    if saved:
        _log.info("[friendship] learned %d social pattern(s) for person_id=%s", saved, person_id)
    return saved


def summarize_for_prompt(person_id: Optional[int]) -> str:
    """Return prompt-ready friendship preferences and inside jokes."""
    if not isinstance(person_id, int):
        return ""

    try:
        prefs = facts_memory.get_facts_by_category(person_id, "preference")
        jokes = facts_memory.get_facts_by_category(person_id, "inside_joke")
    except Exception as exc:
        _log.debug("friendship pattern summary failed: %s", exc)
        return ""

    lines: list[str] = []
    style_values = []
    for fact in prefs:
        key = fact.get("key") or ""
        if key in {
            "roast_style", "answer_style", "banter_style",
            "star_wars_bits", "callback_style",
        }:
            style_values.append(str(fact.get("value") or "").strip())
    style_values = [v for v in dict.fromkeys(style_values) if v]
    if style_values:
        lines.append("Social style preferences: " + "; ".join(style_values[:6]) + ".")

    joke_values = []
    for fact in jokes:
        value = str(fact.get("value") or "").strip()
        if value:
            joke_values.append(value)
    joke_values = list(dict.fromkeys(joke_values))
    if joke_values:
        lines.append(
            "Running/inside jokes available for rare use: "
            + "; ".join(joke_values[:3])
            + "."
        )

    if not lines:
        return ""
    lines.append(
        "Friendship-pattern rule: follow style preferences, and use at most one "
        "inside joke or callback in a reply. If the human seems tired, upset, or "
        "direct, skip the bit."
    )
    return "\n".join(lines)


def _extract_running_joke(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    for pattern in _JOKE_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        topic = (match.groupdict().get("topic") or "").strip(" :-,")
        topic = _TRAILING_FILLER.sub("", topic).strip(" :-,")
        topic = re.sub(r"^(?:that|about)\s+", "", topic, flags=re.I).strip(" :-,")
        if not topic:
            topic = cleaned.strip(" :-,")
        if len(topic) >= 3:
            return topic[:100]
    return ""


def _joke_key(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return "inside_joke_" + (slug[:48] or "shared_bit")


def _visible_scope(person_id: int) -> str:
    """Return names of other known visible people for pair/group joke scope."""
    try:
        snapshot = world_state.snapshot()
        names = []
        for p in snapshot.get("people", []) or []:
            other_id = p.get("person_db_id")
            if other_id is None or int(other_id) == int(person_id):
                continue
            person = people_memory.get_person(int(other_id))
            name = (person or {}).get("name") or p.get("face_id") or ""
            first = name.split()[0] if name else ""
            if first:
                names.append(first)
        names = list(dict.fromkeys(names))
        if names:
            return ", ".join(names[:3])
    except Exception as exc:
        _log.debug("visible scope lookup failed: %s", exc)
    return ""
