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
from intelligence.turn_plan import TurnPlan
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
    "lull_callback": (
        58,
        "resurface the one supplied banked fact as a single dry, affectionate "
        "callback line only. No question, no second topic, no other memories, "
        "and nothing sensitive.",
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
    "lull_callback",
    "small_talk",
    "group_turn_invite",
    "startup_empty_room",
    "ambient_observation",
    "appearance_riff",
    "people_roast",
    "idle_monologue",
}

def proactive_grace_blocks(purpose: str) -> bool:
    """True when end-of-thread grace currently suppresses this proactive purpose.

    Extracted from _claim_proactive_purpose so the action governor (the single
    decider under ENFORCE, which bypasses the claim) can apply the SAME gate — it
    is not subsumed by arbitration. Side-effect-free."""
    if purpose not in _GRACE_SUPPRESSED_PROACTIVE_PURPOSES:
        return False
    try:
        from intelligence import end_thread
        return not end_thread.can_proactive_purpose(purpose)
    except Exception:
        return False


def proactive_budget_blocks(purpose: str) -> bool:
    """True when the question budget is exhausted for this budgeted proactive
    purpose. Companion to proactive_grace_blocks — same rationale. Side-effect-free
    (question_budget.can_ask only reads the current count)."""
    if purpose not in _BUDGETED_PROACTIVE_PURPOSES:
        return False
    try:
        from intelligence import question_budget
        return not question_budget.can_ask(purpose)
    except Exception:
        return False


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

    if proactive_grace_blocks(purpose):
        _log.info(
            "proactive purpose suppressed by end-of-thread grace — "
            "purpose=%s label=%r",
            purpose,
            label,
        )
        return None

    if proactive_budget_blocks(purpose):
        _log.info(
            "proactive purpose suppressed by question budget — purpose=%s label=%r",
            purpose,
            label,
        )
        return None

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
    from intelligence import profile_questions
    if profile_questions.person_is_minor(person_id):
        return False  # don't run the reactive friendship interview on a child/teen
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


def _finish(plan: TurnPlan, lines: list, purpose: Optional[str] = None) -> TurnPlan:
    """Render the directive into the plan, populate any question-signals the branch
    didn't set, and return it. A `purpose` passed here is only a default — an inline
    `plan.purpose = ...` set by a branch takes precedence."""
    plan.directive = "\n".join(lines)
    if purpose is not None and plan.purpose is None:
        plan.purpose = purpose
    _populate_signals(plan)
    return plan


def _populate_signals(plan: TurnPlan) -> None:
    """Fill any question-signals the branch left unset by regex-deriving them from the
    rendered directive — the SAME mapping build_frame's fallback uses (social_frame.
    derive_signals) — so the live build_frame reads structured TurnPlan fields instead
    of reparsing the prose. Never raises; on error the fields stay None and build_frame
    falls back to its own regex."""
    try:
        from intelligence import social_frame
        sig = social_frame.derive_signals(plan.directive, plan.purpose or "")
    except Exception:
        return
    for name in (
        "ask_allowed", "hard_no_question", "explicit_followup",
        "fresh_interest_followup", "urgent_identity",
    ):
        if getattr(plan, name) is None:
            setattr(plan, name, sig[name])


def build_turn_plan(
    user_text: str,
    person_id: Optional[int],
    *,
    answered_question: Optional[dict] = None,
) -> TurnPlan:
    """
    Build the turn's TurnPlan: the structured decisions (purpose, …) the agenda
    makes, plus the rendered `directive` string that gives the next reply one job.

    The directive is intentionally plain. The Rex voice still comes from the core
    prompt; this just decides what the turn is for. social_frame reads the
    structured fields instead of regex-reparsing the directive (Bet 2).
    """
    text = (user_text or "").strip()
    ws = world_state.snapshot()

    plan = TurnPlan()
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
        return _finish(plan, lines)

    if _looks_like_offscreen_correction(text):
        lines.append(
            "Primary purpose: acknowledge the correction that the person is still "
            "present but out of camera view. Briefly say you have them / there "
            "they are, using their name if known, then stop. No new questions, "
            "no interest-thread pivot, no generic friendship question."
        )
        return _finish(plan, lines)

    if _looks_like_health_resolved(text):
        lines.append(
            "Primary purpose: acknowledge relief that the health issue or pain has "
            "resolved. Let the worry de-escalate now: warm, pleased, and brief. "
            "Do not keep probing the health topic, do not ask a new question, and "
            "do not pivot into an unrelated interview topic."
        )
        return _finish(plan, lines)

    try:
        from intelligence import social_frame as _sf
        _is_boundary = _sf._looks_like_boundary(text)
    except Exception:
        _is_boundary = False
    if _is_boundary:
        # The user set a boundary / asked for space ("I'll be quiet", "I'd rather
        # not", "give me a minute"). The quality eval showed Rex RESISTS these —
        # "silence isn't my jam, I thrive on noise" — even with the roast eased to
        # none, because nothing told him to RESPECT it. This positive directive does.
        lines.append(
            "Primary purpose: the human just set a boundary or asked for space "
            "('I'll be quiet', 'I'd rather not', 'give me a minute', 'let's change "
            "the subject'). RESPECT it immediately: one short, warm acknowledgement "
            "that genuinely gives them room, then stop. Do NOT push back, protest, "
            "talk them out of it, complain that you prefer noise/chatter, needle "
            "them, or roast the boundary — and do not pivot into a new question. "
            "Letting it land gracefully is the whole move."
        )
        plan.purpose = "boundary"
        return _finish(plan, lines)

    if _looks_like_reassurance(text):
        lines.append(
            "Primary purpose: the human is reassuring you or de-escalating ('I'm "
            "not sad', 'it's okay', 'no worries'). Take it at face value — respond "
            "warmly and lightly and move on. Do NOT roast or needle them for it, do "
            "NOT imply they are repressing or hiding feelings, and do not insist the "
            "mood is worse than they say. A brief, genuine beat is the whole move."
        )
        return _finish(plan, lines)

    if end_thread_pending:
        lines.append(
            "Primary purpose: close the current thread gracefully. Give a brief "
            "acknowledgement or soft final beat, then stop. No new questions, "
            "no unrelated memory hooks, no visual riff."
        )
        plan.purpose = "closure"
        return _finish(plan, lines)

    if _looks_like_phatic_answer(text):
        # A short, friendly throwaway ("good", "I've been good", "it's going good").
        # Rex kept turning these into a clever bit — roasting the BREVITY ("the classic
        # response", "droid-approved script") or dragging in a remembered detail
        # ("…those piles of boxes"). Match their easy energy with a brief, warm beat.
        lines.append(
            "Primary purpose: the human gave a short, friendly throwaway answer about "
            "how they're doing ('good', 'I've been good', 'it's going good'). Match "
            "their easy energy with ONE brief, warm, natural beat — and if they asked "
            "how YOU are, answer that lightly. Do NOT analyze, roast, or make a bit out "
            "of the fact that the answer was short or generic (no 'the classic "
            "response', 'droid-approved script', 'that's code for…'), do NOT drag in a "
            "remembered fact or their surroundings to pad it, and do NOT interrogate "
            "them. A short genuine reply — optionally one light, open question like "
            "'what's good with you?' — is the whole move."
        )
        plan.purpose = "small_talk"
        return _finish(plan, lines)

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
        return _finish(plan, lines)

    steering_ctx = None
    try:
        steering_ctx = conversation_steering.note_user_turn(person_id, text)
    except Exception as exc:
        _log.debug("conversation steering note failed: %s", exc)
    if steering_ctx and steering_ctx.directive:
        lines.append(steering_ctx.directive)
        if getattr(steering_ctx, "mode", "deepen") == "pivot":
            # The subject stopped landing — change the channel. Offer a concrete
            # fresh thing to ask about so the pivot lands somewhere instead of
            # being a vague "let's talk about something else".
            next_q = (
                _next_useful_question(person_id) if person_id is not None else None
            )
            pivot_line = (
                "Primary purpose: that subject stalled — pivot. Lead with a brief "
                "reaction to what they just said, then steer to a RELATED subject "
                "or open a new one, and ask one natural follow-up about that fresh "
                "subject. Do not keep probing the stalled topic."
            )
            if next_q:
                pivot_line += (
                    " A natural fresh question to use if it fits: "
                    f"{next_q['text']!r}."
                )
            lines.append(pivot_line)
            plan.explicit_followup = True
        elif _looks_like_user_question(text):
            lines.append(
                "Primary purpose: answer the human's direct question first, then "
                "keep the reply connected to their interest thread if it still fits."
            )
        else:
            # A follow-up about the interest they JUST raised is earned curiosity,
            # not an interview pivot — so it is allowed even when the recent
            # question budget is full (the budget rations new-topic questions).
            lines.append(
                "Primary purpose: deepen the interest thread the human opened. "
                "Give one specific subject-aware reaction or tidbit, then ask one "
                "natural follow-up about their experience with that topic — what "
                "got them into it, how they got into it, or their favorite part."
            )
            plan.explicit_followup = True
        plan.purpose = "interest"
        return _finish(plan, lines)

    if answered_question:
        q_text = answered_question.get("question_text") or "your previous question"
        a_text = answered_question.get("answer_text") or text
        if not _is_compliment_or_ack(a_text):
            # They just answered a real question — a single tightly-related
            # follow-up is earned curiosity, not an interview pivot, so it is
            # allowed even when the question budget is full.
            lines.append(
                "Primary purpose: the human just answered a question Rex asked. "
                f"Question: {q_text!r}. Answer: {a_text!r}. "
                "React to the actual content with genuine, specific interest — "
                "show you find it interesting, not just that you logged it. After "
                "answering, ask at most one short follow-up that stays on this "
                "exact topic, or carry the turn with a specific Rex opinion / "
                "light roast instead. Do not pivot into a new interview topic."
            )
            plan.explicit_followup = True
        else:
            lines.append(
                "Primary purpose: the human just answered a question Rex asked. "
                f"Question: {q_text!r}. Answer: {a_text!r}. "
                "Briefly acknowledge the answer and use it naturally. Do not ask "
                "another question in the same breath; a short opinion or light "
                "roast is okay if it fits the answer."
            )
        plan.purpose = "answer_ack"
        return _finish(plan, lines)

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
        plan.purpose = "answer"
        return _finish(plan, lines)

    if person_id is not None:
        pending = rel_memory.get_latest_pending_question(person_id)
        if pending:
            lines.append(
                "Primary purpose: Rex is waiting for an answer to his last "
                f"question: {pending.get('question_text')!r}. Do not ask a new "
                "question yet; respond to what the human just said and leave "
                "space for them to answer if they have not."
            )
            return _finish(plan, lines)

        low_pressure_ack = _is_compliment_or_ack(text)
        if _PLAN_STATEMENT_PAT.search(text):
            lines.append(
                "Primary purpose: acknowledge the human's plan or upcoming event. "
                "Give one concrete positive or curious beat connected to that plan, "
                "then stop. Do not pivot into an unrelated interview question."
            )
            return _finish(plan, lines)

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
                "Primary purpose: REACT first — land a specific opinion, jab, or "
                "roast on what they just said. Then, ONLY if it fits naturally and "
                "doesn't make the turn feel like an interview, you may fold in this "
                f"one question: {next_q['text']!r}. The reaction matters more than "
                "the question — skip the question entirely when the funnier move is "
                "to land the line and stop."
            )
        elif low_pressure_ack:
            lines.append(
                "Primary purpose: react to the human's compliment, status update, "
                "or simple beat with a specific Rex opinion, playful observation, or "
                "sharp roast — lead with the funny. Do not pivot into a new "
                "interview question just because question budget remains."
            )
        elif not question_budget_allows:
            lines.append(
                "Primary purpose: respond to the human's latest thought without "
                "adding a new question. The recent question budget is full; leave "
                "space instead of interviewing, and land a specific opinion, "
                "observation, or roast to keep the turn alive."
            )
        else:
            lines.append(
                "Primary purpose: react to what the human actually said with a "
                "specific, in-character beat — genuine interest, a real opinion, a "
                "dry observation, or a sharp roast when you have an actual angle on "
                "it. Lead with substance, not a reflexive joke: do not force a "
                "punchline onto a plain or sincere statement, and never answer it "
                "with a non-sequitur. If they shared something real, show you find "
                "it interesting before (or instead of) teasing. Use known facts and "
                "what you see if relevant. At most one tightly related follow-up "
                "question, only if it continues this exact thread; never pivot into "
                "a new interview topic."
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

    return _finish(plan, lines)


def build_turn_directive(
    user_text: str,
    person_id: Optional[int],
    *,
    answered_question: Optional[dict] = None,
) -> str:
    """Back-compat string accessor: render just the agenda directive from the
    TurnPlan. Prefer build_turn_plan() where the structured fields are wanted."""
    return build_turn_plan(
        user_text, person_id, answered_question=answered_question
    ).directive


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
_REASSURANCE_PAT = re.compile(
    r"\b("
    r"i'?m not (?:sad|upset|mad|angry|worried|bothered|hurt|stressed|down)|"
    r"i am not (?:sad|upset|mad|angry|worried|bothered|hurt|stressed|down)|"
    r"it'?s (?:okay|ok|fine|all good|alright|no big deal)|"
    r"it is (?:okay|ok|fine|all good|alright|no big deal)|"
    r"no worries|don'?t worry|do not worry|"
    r"it'?s not (?:a big deal|that bad|that serious)|"
    r"really[,]? it'?s fine|i'?m good[,]? (?:really|honestly)"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_offscreen_correction(text: str) -> bool:
    return bool(_OFFSCREEN_CORRECTION_PAT.search(text or ""))


def _looks_like_health_resolved(text: str) -> bool:
    return bool(_HEALTH_RESOLVED_PAT.search(text or ""))


def _looks_like_reassurance(text: str) -> bool:
    return bool(_REASSURANCE_PAT.search(text or ""))


# A short, friendly throwaway answer to "how are you / how's it going" — "good",
# "I've been good", "it's going good", "doing well", "can't complain". The WHOLE
# utterance must be phatic (no real content), so "good, just back from camping" does
# NOT match (that has substance to engage). A trailing reciprocal "you?" is allowed.
_PHATIC_ANSWER_PAT = re.compile(
    r"^(?:"
    r"(?:i'?m|i am|i'?ve|i have|things?(?:'re| are)?|everything(?:'s| is)?|it'?s|it is)\s+"
    r"(?:been\s+|going\s+|doing\s+)?"
    r"|just\s+|doing\s+|going\s+|been\s+|pretty\s+|really\s+|so\s+|all\s+|quite\s+"
    r")*"
    r"(?:good|great|fine|okay|ok|alright|well|chill|cool|grand|decent|"
    r"not\s+bad|not\s+much|nothing\s+much|can'?t\s+complain|same\s+(?:old(?:\s+same\s+old)?|as\s+usual)|"
    r"hanging\s+in\s+there|making\s+it|surviving)"
    r"[.!,]*$",
    re.IGNORECASE,
)
# Strip a tacked-on reciprocal ("…, you?", "how about you?", "and yourself?") before matching.
_RECIPROCAL_TAIL_PAT = re.compile(
    r"[,]?\s*(?:how\s+about\s+you|what\s+about\s+you|and\s+you|and\s+yourself|"
    r"you|yourself|hbu|wbu)\s*\??$",
    re.IGNORECASE,
)


def _looks_like_phatic_answer(text: str) -> bool:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return False
    core = _RECIPROCAL_TAIL_PAT.sub("", cleaned).strip()
    if not core:
        return False
    if len(re.findall(r"[A-Za-z']+", core)) > 6:
        return False
    return bool(_PHATIC_ANSWER_PAT.match(core))
