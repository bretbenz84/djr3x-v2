"""
conversation_agenda.py — choose one conversational purpose for Rex's next turn.

Rex has many instincts: answer, roast, observe, ask questions, follow up, and
notice people. This module keeps those instincts from all speaking at once by
turning the current context into a single short directive for the LLM.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
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


@dataclass
class _ProactiveClaim:
    token: str
    purpose: str
    priority: int
    label: str
    expires_at: float


_PROACTIVE_RULES: dict[str, tuple[int, str]] = {
    "emotional_checkin": (
        100,
        "check in about the sensitive emotional context only. No roasts, no "
        "extra small talk, no visual riff unless the human invites it.",
    ),
    "relationship_inquiry": (
        95,
        "identify or ask about the unfamiliar person only. Do not add unrelated "
        "banter or a second question.",
    ),
    "identity_prompt": (
        92,
        "ask the unknown person who they are only. Do not stack another topic.",
    ),
    "presence_reaction": (
        80,
        "react to the person entering or leaving only. Keep it to one line.",
    ),
    "overheard_chime_in": (
        75,
        "briefly chime in because Rex was being discussed. Do not start an "
        "interview or change topics.",
    ),
    "third_party_awareness": (
        72,
        "acknowledge the nearby third party only. Do not redirect the whole "
        "conversation or ask another question.",
    ),
    "reengagement": (
        70,
        "recapture attention with one line only. Do not ask an unrelated question.",
    ),
    "memory_followup": (
        65,
        "follow up on the remembered plan or date only. No extra question.",
    ),
    "visual_curiosity": (
        55,
        "ask one question based on the visible scene only. Do not also bring up "
        "memory, holidays, or emotional check-ins.",
    ),
    "small_talk": (
        45,
        "ask one small-talk question only. Do not stack a second prompt.",
    ),
    "world_reaction": (
        40,
        "react to the world-state change only. No follow-up question unless the "
        "prompt explicitly requires one.",
    ),
    "ambient_observation": (
        30,
        "make one ambient observation only. Do not ask a question.",
    ),
    "appearance_riff": (
        28,
        "make one appearance or style observation only. Keep it non-sensitive.",
    ),
    "idle_monologue": (
        15,
        "say one idle/private line only. Do not pull in another topic.",
    ),
}

_proactive_lock = threading.Lock()
_active_proactive_claim: Optional[_ProactiveClaim] = None


def _looks_like_user_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and ("?" in cleaned or bool(_QUESTION_START.search(cleaned)))


def claim_proactive_purpose(
    purpose: str,
    *,
    priority: Optional[int] = None,
    label: str = "",
    ttl_secs: float = 18.0,
) -> Optional[str]:
    """
    Reserve the next proactive speech slot for one conversational purpose.

    Background behaviors often launch LLM calls in parallel. A claim gives the
    highest-priority purpose ownership while the line is being generated, so a
    lower-priority visual riff or idle thought cannot sneak in underneath an
    emotional check-in or identity prompt.
    """
    global _active_proactive_claim

    now = time.monotonic()
    rule_priority = _PROACTIVE_RULES.get(purpose, (20, ""))[0]
    requested_priority = int(rule_priority if priority is None else priority)

    with _proactive_lock:
        if (
            _active_proactive_claim is not None
            and _active_proactive_claim.expires_at <= now
        ):
            _active_proactive_claim = None

        if _active_proactive_claim is not None:
            if requested_priority <= _active_proactive_claim.priority:
                return None

        token = uuid.uuid4().hex
        _active_proactive_claim = _ProactiveClaim(
            token=token,
            purpose=purpose,
            priority=requested_priority,
            label=label,
            expires_at=now + max(1.0, float(ttl_secs)),
        )
        return token


def proactive_claim_is_current(token: Optional[str]) -> bool:
    if not token:
        return True
    now = time.monotonic()
    with _proactive_lock:
        return (
            _active_proactive_claim is not None
            and _active_proactive_claim.token == token
            and _active_proactive_claim.expires_at > now
        )


def release_proactive_claim(token: Optional[str]) -> None:
    global _active_proactive_claim
    if not token:
        return
    with _proactive_lock:
        if (
            _active_proactive_claim is not None
            and _active_proactive_claim.token == token
        ):
            _active_proactive_claim = None


def proactive_purpose_directive(purpose: str) -> str:
    rule = _PROACTIVE_RULES.get(purpose)
    if not rule:
        return (
            "Proactive agenda: this unsolicited line must have exactly ONE "
            "purpose. Do not stack a question, a memory callback, a roast, and "
            "an environment remark together."
        )
    return (
        "Proactive agenda: this unsolicited line must have exactly ONE purpose. "
        f"Primary purpose: {purpose}. Instruction: {rule[1]}"
    )


def with_proactive_directive(prompt: str, purpose: str) -> str:
    return f"{proactive_purpose_directive(purpose)}\n\n{prompt}"


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
    try:
        from intelligence import topic_thread
        topic_directive = topic_thread.build_directive()
        if topic_directive:
            lines.append(topic_directive)
    except Exception:
        pass

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
