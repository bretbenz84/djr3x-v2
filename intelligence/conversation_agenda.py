"""
conversation_agenda.py — choose one conversational purpose for Rex's next turn.

Rex has many instincts: answer, roast, observe, ask questions, follow up, and
notice people. This module keeps those instincts from all speaking at once by
turning the current context into a single short directive for the LLM.
"""

from __future__ import annotations

import re
from typing import Optional

import config
from memory import facts as facts_memory
from memory import people as people_memory
from memory import relationships as rel_memory
from world_state import world_state


_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|is|are|am|should)\b",
    re.IGNORECASE,
)


def _looks_like_user_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and ("?" in cleaned or bool(_QUESTION_START.search(cleaned)))


def _known_fact_keys(person_id: int) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    categories: set[str] = set()
    try:
        for fact in facts_memory.get_facts(person_id):
            if fact.get("key"):
                keys.add(fact["key"])
            if fact.get("category"):
                categories.add(fact["category"])
    except Exception:
        pass
    return keys, categories


def _next_useful_question(person_id: int) -> Optional[dict]:
    person = people_memory.get_person(person_id)
    if not person:
        return None
    tier = person.get("friendship_tier", "stranger")
    max_depth = config.TIER_MAX_DEPTH.get(tier, 1)
    asked = rel_memory.get_asked_question_keys(person_id)
    fact_keys, fact_categories = _known_fact_keys(person_id)
    for candidate in config.QUESTION_POOL:
        if candidate["depth"] > max_depth:
            continue
        key = candidate["key"]
        if key in asked or key in fact_keys or key in fact_categories:
            continue
        return candidate
    return None


def build_turn_directive(
    user_text: str,
    person_id: Optional[int],
    *,
    answered_question: Optional[dict] = None,
) -> str:
    """
    Return a compact directive that gives the next generated reply one job.

    The directive is intentionally plain. The Rex voice still comes from the
    core prompt; this just decides what the turn is for.
    """
    text = (user_text or "").strip()
    ws = world_state.snapshot()

    lines = [
        "Conversation agenda: choose ONE purpose for this turn. Do not stack "
        "multiple follow-up questions, presence reactions, and environment remarks."
    ]

    if answered_question:
        q_text = answered_question.get("question_text") or "your previous question"
        a_text = answered_question.get("answer_text") or text
        lines.append(
            "Primary purpose: the human just answered a question Rex asked. "
            f"Question: {q_text!r}. Answer: {a_text!r}. "
            "Briefly acknowledge the answer and use it naturally. Do not ask "
            "another unrelated interview question in the same breath."
        )
        return "\n".join(lines)

    if _looks_like_user_question(text):
        lines.append(
            "Primary purpose: answer the human's question directly first. "
            "After answering, ask at most one short follow-up only if it flows "
            "from their question or from something currently visible."
        )
        return "\n".join(lines)

    # Unknown people are socially urgent: identify them before generic small talk.
    people = ws.get("people", []) or []
    unknown_count = sum(1 for p in people if p.get("person_db_id") is None)
    if person_id is not None and unknown_count:
        known_name = "them"
        try:
            row = people_memory.get_person(person_id)
            if row and row.get("name"):
                known_name = row["name"].split()[0]
        except Exception:
            pass
        lines.append(
            f"Primary purpose: there are {unknown_count} unfamiliar face(s) visible "
            f"while talking with {known_name}. If Rex has not just asked, briefly "
            "ask who the unfamiliar person is and how they know each other. "
            "This beats generic small talk."
        )
        return "\n".join(lines)

    if person_id is not None:
        pending = rel_memory.get_latest_pending_question(person_id)
        if pending:
            lines.append(
                "Primary purpose: Rex is waiting for an answer to his last "
                f"question: {pending.get('question_text')!r}. Do not ask a new "
                "question yet; respond to what the human just said and leave "
                "space for them to answer if they have not."
            )
            return "\n".join(lines)

        next_q = _next_useful_question(person_id)
        if next_q:
            lines.append(
                "Primary purpose: keep the conversation moving with curiosity. "
                f"If the user's utterance does not demand a direct answer, weave "
                f"in this one question naturally: {next_q['text']!r}. "
                "Ask only this one question, and make it feel motivated by the turn."
            )
        else:
            lines.append(
                "Primary purpose: respond to the human's latest thought. Use known "
                "facts and the environment if relevant, but do not force a question."
            )
    else:
        lines.append(
            "Primary purpose: respond to an unknown person. If you need a name to "
            "continue naturally, ask for it once; otherwise answer normally."
        )

    env = ws.get("environment", {}) or {}
    if env.get("description"):
        lines.append(
            f"Available environmental cue: {env['description']}. Mention it only "
            "if it genuinely connects to the user's turn."
        )

    return "\n".join(lines)
