"""
conversation_agenda.py — choose one conversational purpose for Rex's next turn.

Rex has many instincts: answer, roast, observe, ask questions, follow up, and
notice people. This module keeps those instincts from all speaking at once by
turning the current context into a single short directive for the LLM.
"""

from __future__ import annotations

import re
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import config
from intelligence import empathy
from intelligence import conversation_steering
from intelligence import social_scene
from memory import facts as facts_memory
from memory import people as people_memory
from memory import relationships as rel_memory
from world_state import world_state

_log = logging.getLogger(__name__)

_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|is|are|am|should)\b",
    re.IGNORECASE,
)
_COMPLIMENT_OR_ACK_PAT = re.compile(
    r"\b(thanks?|thank you|appreciate|good job|great job|nice work|"
    r"well done|you'?re (?:good|great|swell|awesome|amazing)|"
    r"you are (?:good|great|swell|awesome|amazing)|"
    r"you'?re such a (?:good|great|swell|awesome|amazing) robot|"
    r"you are such a (?:good|great|swell|awesome|amazing) robot|"
    r"that'?s (?:good|great|nice|cool|awesome)|"
    r"it (?:turned out|came out|worked out) (?:totally |really |pretty )?"
    r"(?:good|great|nice|cool|awesome)|"
    r"i'?m (?:good|fine|okay|ok|alright)|"
    r"doing (?:good|great|fine|okay|ok|alright))\b",
    re.IGNORECASE,
)
_PLAN_STATEMENT_PAT = re.compile(
    r"\b(i'?m|i am|we'?re|we are|i will|i'?ll|we will|we'?ll)\s+"
    r"(?:going|heading|traveling|travelling|flying|driving|visiting|leaving|"
    r"coming|meeting|seeing)\b|"
    r"\b(?:on|this|next)\s+"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b|"
    r"\b(?:tomorrow|tonight|next week|this weekend)\b",
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
    "group_turn_invite": (
        68,
        "invite the quiet visible participant into the current conversation only. "
        "Make it optional, warm, and one short line; do not pressure them.",
    ),
    "personal_space": (
        67,
        "react to the person being comically too close only. One short boundary "
        "joke or playful roast; do not ask a question.",
    ),
    "reengagement": (
        70,
        "recapture attention with one line only. Do not ask an unrelated question.",
    ),
    "memory_followup": (
        65,
        "follow up on the remembered plan or date only. No extra question.",
    ),
    "celebration_checkin": (
        64,
        "briefly celebrate the remembered good news only. Keep it warm, optional, "
        "and do not stack another memory callback.",
    ),
    "startup_empty_room": (
        60,
        "make one short startup empty-room joke only. Do not ask a question or "
        "pretend someone is present.",
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
    "weather.proactive_comment": (
        42,
        "react to the weather-feed change only. Keep it honest that Rex saw it "
        "in a feed, not felt it directly. One short line, no follow-up question.",
    ),
    "ambient_observation": (
        30,
        "make one ambient observation only. Do not ask a question.",
    ),
    "appearance_riff": (
        28,
        "make one appearance or style observation only. Keep it non-sensitive.",
    ),
    "people_roast": (
        27,
        "make one playful non-sensitive roast about the visible person's current "
        "vibe only. No questions, no body/identity/protected-trait jokes.",
    ),
    "idle_monologue": (
        15,
        "say one idle/private line only. Do not pull in another topic.",
    ),
}
_BUDGETED_PROACTIVE_PURPOSES = {
    "celebration_checkin",
    "memory_followup",
    "visual_curiosity",
    "small_talk",
    "group_turn_invite",
}
_GRACE_SUPPRESSED_PROACTIVE_PURPOSES = {
    "celebration_checkin",
    "memory_followup",
    "visual_curiosity",
    "small_talk",
    "group_turn_invite",
    "startup_empty_room",
    "ambient_observation",
    "appearance_riff",
    "people_roast",
    "idle_monologue",
}

_proactive_lock = threading.Lock()
_active_proactive_claim: Optional[_ProactiveClaim] = None


def _looks_like_user_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return bool(cleaned) and ("?" in cleaned or bool(_QUESTION_START.search(cleaned)))


def _is_compliment_or_ack(text: str) -> bool:
    return bool(_COMPLIMENT_OR_ACK_PAT.search(text or ""))


def _social_context_lines(ws: dict) -> list[str]:
    crowd = ws.get("crowd", {}) or {}
    people = ws.get("people", []) or []
    lines: list[str] = []
    if crowd.get("interaction_mode"):
        lines.append(
            "Live social context: "
            f"mode={crowd.get('interaction_mode')}; "
            f"count={crowd.get('count', len(people))}; "
            f"engaged={crowd.get('engaged_count', 'unknown')}."
        )
    close_people = [
        p for p in people
        if (p.get("distance_zone") or "").lower() == "intimate"
    ]
    if close_people:
        names = [
            str(p.get("face_id") or p.get("voice_id") or p.get("id") or "someone")
            for p in close_people[:2]
        ]
        lines.append(
            "Proxemics cue: "
            + ", ".join(names)
            + " is extremely close to Rex's personal space by American norms. "
            "A short boundary joke or roast is allowed if it fits the turn."
        )
    disengaged = crowd.get("disengaged_people") or []
    if disengaged:
        lines.append(
            "Engagement cue: at least one visible person appears disengaged; "
            "avoid piling on questions unless re-engaging them is the single purpose."
        )
    return lines


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

    if purpose in _GRACE_SUPPRESSED_PROACTIVE_PURPOSES:
        try:
            from intelligence import end_thread
            if not end_thread.can_proactive_purpose(purpose):
                _log.info(
                    "proactive purpose suppressed by end-of-thread grace — "
                    "purpose=%s label=%r",
                    purpose,
                    label,
                )
                return None
        except Exception as exc:
            _log.debug("end-of-thread proactive check failed: %s", exc)

    if purpose in _BUDGETED_PROACTIVE_PURPOSES:
        try:
            from intelligence import question_budget
            if not question_budget.can_ask(purpose):
                _log.info(
                    "proactive purpose suppressed by question budget — purpose=%s label=%r",
                    purpose,
                    label,
                )
                return None
        except Exception as exc:
            _log.debug("question budget proactive check failed: %s", exc)

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
    extra_lines = []
    try:
        from intelligence import user_energy
        energy = user_energy.build_directive()
        if energy:
            extra_lines.append(energy)
    except Exception:
        pass
    try:
        from intelligence import question_budget
        budget = question_budget.build_directive()
        if budget:
            extra_lines.append(budget)
    except Exception:
        pass
    try:
        from intelligence import end_thread
        grace = end_thread.build_directive()
        if grace:
            extra_lines.append(grace)
    except Exception:
        pass
    if not rule:
        base = (
            "Proactive agenda: this unsolicited line must have exactly ONE "
            "purpose. Do not stack a question, a memory callback, a roast, and "
            "an environment remark together."
        )
        return "\n".join([base, *extra_lines]) if extra_lines else base
    base = (
        "Proactive agenda: this unsolicited line must have exactly ONE purpose. "
        f"Primary purpose: {purpose}. Instruction: {rule[1]}"
    )
    return "\n".join([base, *extra_lines]) if extra_lines else base


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


def _friendship_question_allowed(text: str, person_id: Optional[int]) -> bool:
    if person_id is None:
        return False
    if _looks_like_user_question(text):
        return False
    if _PLAN_STATEMENT_PAT.search(text or ""):
        return False
    if len(re.findall(r"[A-Za-z0-9']+", text or "")) <= 5:
        return False
    try:
        entry = empathy.peek(person_id) or {}
        mode = ((entry.get("mode") or {}).get("mode") or "").lower()
        result = entry.get("result") or {}
        affect = (result.get("affect") or "").lower()
        sensitivity = (result.get("topic_sensitivity") or "").lower()
        if mode in {"listen", "support", "validate", "ground", "brief", "gentle_probe"}:
            return False
        if affect in {"sad", "withdrawn", "angry", "anxious"} or sensitivity in {"heavy", "medium"}:
            return False
    except Exception:
        pass
    return True


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
        "multiple follow-up questions, presence reactions, opinions, roasts, "
        "and environment remarks."
    ]
    try:
        from intelligence import topic_thread
        topic_directive = topic_thread.build_directive()
        if topic_directive:
            lines.append(topic_directive)
    except Exception:
        pass
    try:
        from intelligence import user_energy
        energy_directive = user_energy.build_directive()
        if energy_directive:
            lines.append(energy_directive)
    except Exception:
        pass
    end_thread_pending = None
    try:
        from intelligence import end_thread
        end_thread_directive = end_thread.build_directive()
        if end_thread_directive:
            lines.append(end_thread_directive)
        end_thread_pending = end_thread.pending_closure()
    except Exception:
        end_thread_pending = None
    try:
        from intelligence import response_length
        lines.append(
            response_length.build_directive(
                text,
                answered_question=answered_question,
            )
        )
    except Exception:
        pass
    question_budget_allows = True
    try:
        from intelligence import question_budget
        budget_directive = question_budget.build_directive()
        if budget_directive:
            lines.append(budget_directive)
        question_budget_allows = question_budget.can_ask("agenda_question")
    except Exception:
        question_budget_allows = True

    lines.extend(_social_context_lines(ws))
    try:
        cast = social_scene.conversation_cast_context(
            ws,
            current_person_id=person_id,
        )
        if cast.directive:
            lines.append(cast.directive)
    except Exception as exc:
        _log.debug("conversation cast directive skipped: %s", exc)

    local_sensitive = empathy.classify_local_sensitivity(text)
    if local_sensitive:
        event = local_sensitive.get("event") or {}
        category = event.get("category") or (
            "crisis" if local_sensitive.get("crisis") else "sensitive"
        )
        lines.append(
            "Primary purpose: respond to the sensitive disclosure detected in "
            f"this exact user turn (category={category}). Drop roast-first mode "
            "completely. No personal roasts, no visual riff, no comic pivot, "
            "and no memory callback. Be brief, warm, and grounded in Rex's voice. "
            "For death, grief, illness, or crisis language, acknowledge plainly; "
            "ask at most one low-pressure support question only if it helps."
        )
        return "\n".join(lines)

    if _looks_like_offscreen_correction(text):
        lines.append(
            "Primary purpose: acknowledge the correction that the person is still "
            "present but out of camera view. Briefly say you have them / there "
            "they are, using their name if known, then stop. No new questions, "
            "no interest-thread pivot, no generic friendship question."
        )
        return "\n".join(lines)

    if _looks_like_health_resolved(text):
        lines.append(
            "Primary purpose: acknowledge relief that the health issue or pain has "
            "resolved. Let the worry de-escalate now: warm, pleased, and brief. "
            "Do not keep probing the health topic, do not ask a new question, and "
            "do not pivot into an unrelated interview topic."
        )
        return "\n".join(lines)

    if end_thread_pending:
        lines.append(
            "Primary purpose: close the current thread gracefully. Give a brief "
            "acknowledgement or soft final beat, then stop. No new questions, "
            "no unrelated memory hooks, no visual riff."
        )
        return "\n".join(lines)

    unknown_context = social_scene.unknown_group_context(
        ws,
        current_person_id=person_id,
    )
    if unknown_context:
        if _looks_like_user_question(text):
            lines.append(
                unknown_context.directive
                + " If the human also asked a direct practical question, answer it "
                "in one very short clause first, then use the one allowed question "
                "to handle the group introduction."
            )
        else:
            lines.append(unknown_context.directive)
        return "\n".join(lines)

    steering_directive = conversation_steering.build_directive(person_id, text)
    if steering_directive:
        lines.append(steering_directive)
        if _looks_like_user_question(text):
            lines.append(
                "Primary purpose: answer the human's direct question first, then "
                "keep the reply connected to their interest thread if it still fits."
            )
        elif question_budget_allows:
            lines.append(
                "Primary purpose: deepen the interest thread the human opened. "
                "Give one specific subject-aware reaction or tidbit, then ask one "
                "natural follow-up about their experience with that topic."
            )
        else:
            lines.append(
                "Primary purpose: deepen the interest thread the human opened. "
                "Give one specific subject-aware reaction or tidbit, but do not "
                "add a new question because the recent question budget is full."
            )
        return "\n".join(lines)

    if answered_question:
        q_text = answered_question.get("question_text") or "your previous question"
        a_text = answered_question.get("answer_text") or text
        if question_budget_allows and not _is_compliment_or_ack(a_text):
            lines.append(
                "Primary purpose: the human just answered a question Rex asked. "
                f"Question: {q_text!r}. Answer: {a_text!r}. "
                "Briefly acknowledge the answer and use it naturally. You may "
                "add one tightly related follow-up question, or carry the turn "
                "with a specific Rex opinion / light roast instead. Do not pivot "
                "into a new interview topic."
            )
        else:
            lines.append(
                "Primary purpose: the human just answered a question Rex asked. "
                f"Question: {q_text!r}. Answer: {a_text!r}. "
                "Briefly acknowledge the answer and use it naturally. Do not ask "
                "another question in the same breath; a short opinion or light "
                "roast is okay if it fits the answer."
            )
        return "\n".join(lines)

    if _looks_like_user_question(text):
        if question_budget_allows:
            lines.append(
                "Primary purpose: answer the human's question directly first. "
                "After answering, ask at most one short follow-up only if it flows "
                "from their question or from something currently visible."
            )
        else:
            lines.append(
                "Primary purpose: answer the human's question directly first. "
                "Do not add a new follow-up question; the recent question budget "
                "is full."
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

        low_pressure_ack = _is_compliment_or_ack(text)
        if _PLAN_STATEMENT_PAT.search(text):
            lines.append(
                "Primary purpose: acknowledge the human's plan or upcoming event. "
                "Give one concrete positive or curious beat connected to that plan, "
                "then stop. Do not pivot into an unrelated interview question."
            )
            return "\n".join(lines)

        next_q = None
        if bool(getattr(config, "REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED", False)):
            next_q = (
                _next_useful_question(person_id)
                if (
                    question_budget_allows
                    and not low_pressure_ack
                    and _friendship_question_allowed(text, person_id)
                )
                else None
            )
        if next_q:
            lines.append(
                "Primary purpose: keep the conversation moving with curiosity. "
                f"If the user's utterance does not demand a direct answer, weave "
                f"in this one question naturally: {next_q['text']!r}. "
                "Ask only this one question, and make it feel motivated by the turn."
            )
        elif low_pressure_ack:
            lines.append(
                "Primary purpose: briefly acknowledge the human's compliment, "
                "status update, or simple conversational beat. You may add a "
                "specific Rex opinion, playful observation, or light roast if it "
                "fits. Do not pivot into a new interview question just because "
                "question budget remains."
            )
        elif not question_budget_allows:
            lines.append(
                "Primary purpose: respond to the human's latest thought without "
                "adding a new question. The recent question budget is full; leave "
                "space instead of interviewing, but you may keep the turn alive "
                "with a specific opinion, observation, or roast if socially safe."
            )
        else:
            lines.append(
                "Primary purpose: respond to the human's latest thought. Use known "
                "facts and the environment if relevant. You may ask one tightly "
                "related follow-up question if it naturally continues this exact "
                "thread; do not pivot into a new interview topic."
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


_OFFSCREEN_CORRECTION_PAT = re.compile(
    r"\b("
    r"i'?m still here|i am still here|still here|out of view|off[- ]camera|"
    r"camera (?:is )?(?:turned|pointed) away|you can'?t see me|you cannot see me"
    r")\b",
    re.IGNORECASE,
)
_HEALTH_RESOLVED_PAT = re.compile(
    r"\b("
    r"(?:pain|ache|hurt|back|neck|headache|migraine|soreness).{0,50}"
    r"(?:gone away|went away|resolved|cleared up|is gone|has gone|better now|"
    r"is mostly gone|has mostly gone|mostly gone|feels better|feeling better)|"
    r"(?:i'?m|i am) (?:better|fine|okay|ok) now"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_offscreen_correction(text: str) -> bool:
    return bool(_OFFSCREEN_CORRECTION_PAT.search(text or ""))


def _looks_like_health_resolved(text: str) -> bool:
    return bool(_HEALTH_RESOLVED_PAT.search(text or ""))
