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
# Proactive purposes still throttled by the question budget. The silence-FILLING
# re-engagement paths (visual_curiosity, small_talk) are deliberately NOT here: they
# fire only when the conversation has gone quiet, so asking then is re-engagement, not
# interviewing — the budget would otherwise leave dead air after a few turns (live-logged
# 2026-06-19: visual curiosity rejected as "question_budget_exhausted" every lull).
_BUDGETED_PROACTIVE_PURPOSES = {
    "celebration_checkin",
    "memory_followup",
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


def _post_web_search_steer() -> str:
    """If Rex just web-searched something for the person and they've gone quiet, steer
    the proactive lull line to be INQUISITIVE about the topic — why they asked, whether
    they're into it, what they think — instead of repeating the answer or piling on more
    facts/opinions. Empty string when there's no fresh search. Failure-safe."""
    try:
        from intelligence import web_search
        topic = web_search.recent_search()
    except Exception:
        return ""
    if not topic:
        return ""
    return (
        f"POST-SEARCH FOLLOW-UP: you JUST looked up \"{topic}\" out loud for them a "
        "moment ago, and now they've gone quiet. Do NOT repeat or re-summarize what you "
        "already told them, and do NOT keep piling on facts or hot takes about it. "
        "Instead be INQUISITIVE: ask ONE short, genuine question about THEM and this "
        "topic — what got them asking about it, whether they're into it, what they think "
        "or remember about it — then let them lead. One dry opinion is fine ONLY if it is "
        "attached to that question; lead with the curiosity, not another lecture."
    )


def with_proactive_directive(prompt: str, purpose: str) -> str:
    directive = proactive_purpose_directive(purpose)
    steer = _post_web_search_steer()
    if steer:
        directive = f"{directive}\n\n{steer}" if directive else steer
    return f"{directive}\n\n{prompt}"


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


# ── What-if / plans state (per session; cleared by reset_plans_state) ─────────────
# Coarse per-(person, plan_key) dedupe so Rex clarifies a plan once and suggests once,
# never nagging. _pending_plan_clarify bridges the clarify→answer→suggest handoff across
# turns ("I'm going camping" → "Where?" → "Fraser Flats" → "what if…").
_plans_clarified: set = set()
_plans_suggested: set = set()
_pending_plan_clarify: dict = {}  # person_id -> {"key": str, "at": monotonic}


def reset_plans_state() -> None:
    """Clear the what-if/plans dedupe + pending state (called on session reset)."""
    _plans_clarified.clear()
    _plans_suggested.clear()
    _pending_plan_clarify.clear()


# NOTE on wording: each directive matches social_frame's _ASK_ALLOWED_PAT +
# _EXPLICIT_FOLLOWUP_PAT ("give one … then ask one … follow-up", "one question") so the
# one earned question/what-if is allowed even when the question budget is full — AND so
# social_frame's structured (TurnPlan) and regex paths derive the SAME explicit_followup
# (the Bet-2 equivalence invariant). purpose is left unset (the generic conversational
# default) for the same reason; it carries no behavioral weight for plans.
def _plan_clarify_directive() -> str:
    return (
        "Primary purpose: the human mentioned a plan but kept it vague. Do NOT give a "
        "generic 'that sounds fun' riff. Give one genuine reaction, then ask one specific "
        "clarifying follow-up question a curious friend would ask to pin down the key "
        "detail — where they're going, when, or who with. One question, in Rex's voice."
    )


def _plan_suggest_directive(location: str, *, place_hint: str = "") -> str:
    loc = f" or near {location}" if location else ""
    anchor = f" ({place_hint})" if place_hint else ""
    return (
        f"Primary purpose: the human shared a specific plan{anchor}. Give one quick "
        "reaction, then ask one what-if follow-up question that floats a concrete thing "
        "to do, see, or try there (\"what if you …?\") grounded in that place or activity "
        "— a suggestion, not a claim you've been there. Only suggest something you are "
        f"genuinely confident exists or fits there{loc}; if you're not sure what or where "
        "it is, ask what it's near instead of inventing. One question, specific and dry — "
        "Rex, not a travel brochure."
    )


def _no_plans_directive(location: str) -> str:
    loc = location or "the local area"
    return (
        "Primary purpose: the human has no plans. Don't just riff about it. Give one "
        "quick beat, then ask one what-if follow-up question that floats ONE concrete, "
        f"specific thing to do near {loc} — a real place, activity, or kind of event "
        "(\"what if you …?\"), framed as a friendly suggestion. Only suggest places you "
        f"are genuinely confident exist near {loc}. One idea, in Rex's voice, not a list."
    )


def _plan_branch(plan: TurnPlan, lines: list, text: str, person_id: Optional[int]):
    """Split a plan statement: sparse→clarify, specific→suggest, no-plans→suggest.
    Returns a finished TurnPlan, or None to fall back to the generic acknowledgment
    (feature off, not actually a plan, classifier error, or already handled)."""
    if not bool(getattr(config, "WHAT_IF_PLANS_ENABLED", True)):
        return None
    try:
        from intelligence import plan_intent
        info = plan_intent.classify(text)
    except Exception as exc:
        _log.debug("[plans] classify failed: %s", exc)
        return None
    if not (info.get("is_plan") or info.get("is_no_plans")):
        return None
    location = str(getattr(config, "WEATHER_LOCATION", "") or "").strip()
    key = (person_id, info.get("plan_key") or "plan")

    if info.get("is_no_plans"):
        if key in _plans_suggested:
            return None
        _plans_suggested.add(key)
        lines.append(_no_plans_directive(location))
        return _finish(plan, lines)

    if info.get("specificity") == "specific":
        if key in _plans_suggested:
            return None
        _plans_suggested.add(key)
        lines.append(_plan_suggest_directive(location, place_hint=info.get("place") or ""))
        return _finish(plan, lines)

    # Sparse plan → clarify once (then a later place-answer triggers the suggestion via
    # the answered_question handoff below).
    if key in _plans_clarified or key in _plans_suggested:
        return None
    _plans_clarified.add(key)
    if person_id is not None:
        _pending_plan_clarify[person_id] = {"key": info.get("plan_key") or "plan", "at": time.monotonic()}
    lines.append(_plan_clarify_directive())
    return _finish(plan, lines)


def _plan_clarify_answer(plan: TurnPlan, lines: list, text: str, person_id: Optional[int],
                         answered_question: Optional[dict]):
    """If this turn answers a plan clarifier Rex just asked, offer the what-if suggestion
    grounded in the place they named. Returns a finished TurnPlan or None."""
    if person_id is None or not bool(getattr(config, "WHAT_IF_PLANS_ENABLED", True)):
        return None
    pend = _pending_plan_clarify.pop(person_id, None)
    if not pend:
        return None
    ttl = float(getattr(config, "PLANS_CLARIFY_TTL_SECS", 300.0))
    if (time.monotonic() - float(pend.get("at") or 0.0)) > ttl:
        return None
    key = (person_id, pend.get("key") or "plan")
    if key in _plans_suggested:
        return None
    _plans_suggested.add(key)
    location = str(getattr(config, "WEATHER_LOCATION", "") or "").strip()
    a_text = (answered_question or {}).get("answer_text") or text
    loc = f" or near {location}" if location else ""
    lines.append(
        "Primary purpose: the human just told you the specifics of their plan: "
        f"{a_text!r}. Give one quick reaction, then ask one what-if follow-up question "
        "that floats a concrete thing to do, see, or try there (\"what if you …?\") "
        "grounded in that specific place or activity — a suggestion, not a claim you've "
        f"been there. Only suggest something you are genuinely confident exists or fits "
        f"there{loc}; if you're unsure what or where it is, ask instead of inventing. "
        "One question, specific and dry — Rex."
    )
    return _finish(plan, lines)


# The human is signalling they want OFF this thread — bored, the bit/metaphor isn't landing,
# or they explicitly asked for something else. These are NOT on-topic answers to be deepened;
# they mean DROP the subject and change direction. Field 2026-06-30: Rex ground a "bed/mattress"
# metaphor for five straight turns, and when Bret said "Don't you have anything else to say?" the
# system processed it as a normal answer and stayed ON the bed bit. (Bare "what?" is deliberately
# excluded — it's handled by the misheard-repair path, not a topic pivot.)
_NEW_DIRECTION_PAT = re.compile(
    r"(?:do|don'?t)\s+you\s+have\s+(?:anything|something)\s+else"
    r"|(?:anything|something)\s+else\s+(?:to\s+)?(?:say|talk|add|do)\b"
    r"|say\s+something\s+else|talk\s+about\s+something\s+else"
    r"|change\s+the\s+(?:subject|topic)|change\s+(?:subjects|topics)"
    r"|(?:new|different|another)\s+(?:subject|topic)|(?:let'?s\s+)?move\s+on\b|moving\s+on\b|next\s+topic"
    r"|you\s+(?:keep|already)\s+(?:saying|said)|(?:said|asked)\s+that\s+already"
    r"|you'?re\s+repeating|stop\s+(?:saying|repeating|talking\s+about)"
    r"|enough\s+(?:about|of|with)\b|you'?re\s+stuck\s+on|same\s+thing\s+over"
    r"|beating\s+a\s+dead\s+horse|let\s+it\s+(?:go|die|rest)\b|drop\s+it\b"
    r"|you'?ve\s+lost\s+(?:me|the\s+(?:metaphor|plot|thread|point))|losing\s+me\b"
    r"|lost\s+the\s+(?:metaphor|plot|thread)"
    r"|you'?re\s+(?:rambling|not\s+making\s+(?:any\s+)?sense|making\s+no\s+sense)"
    r"|(?:that|this|it)\s+(?:doesn'?t|does\s+not)\s+make\s+(?:any\s+)?sense"
    r"|this\s+is\s+(?:boring|pointless|going\s+nowhere)|i'?m\s+(?:bored|lost|confused)\b"
    r"|you'?re\s+(?:boring|being\s+weird)|talking\s+(?:so\s+)?weird",
    re.IGNORECASE,
)


def _wants_new_direction(text: str) -> bool:
    """True when the human wants OFF the current thread (bored / the bit isn't landing /
    'don't you have anything else to say?' / 'you've lost the metaphor' / 'you keep saying
    that'). Such a turn must DROP the topic and change direction, not be answered on-topic."""
    t = (text or "").strip()
    return bool(t) and bool(_NEW_DIRECTION_PAT.search(t))


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
    invitation_accepted = False
    try:
        from intelligence import end_thread
        end_thread_directive = end_thread.build_directive()
        if end_thread_directive:
            lines.append(end_thread_directive)
        end_thread_pending = end_thread.pending_closure()
        invitation_accepted = end_thread.consume_invitation_acceptance()
    except Exception:
        end_thread_pending = None
        invitation_accepted = False
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
        try:
            from intelligence import conversation_state as _cstate
            _cstate.note_correction(
                "presence",
                "They said they are still here, just out of camera view — "
                "do not treat them as having left",
                person_id=person_id,
            )
        except Exception:
            pass
        lines.append(
            "Primary purpose: acknowledge the correction that the person is still "
            "present but out of camera view. Briefly say you have them / there "
            "they are, using their name if known, then stop. No new questions, "
            "no interest-thread pivot, no generic friendship question."
        )
        return _finish(plan, lines)

    if _looks_like_grounding_correction(text):
        try:
            from intelligence import conversation_state as _cstate
            _cstate.note_correction(
                "grounding",
                f"They corrected something you got wrong: \"{text}\" — drop the bad guess",
                person_id=person_id,
            )
        except Exception:
            pass
        # Safety net for corrections that reach llm.stream instead of the
        # deterministic repair path: make the "drop the bad guess, don't
        # re-litigate" contract reach the LLM. Must sit ABOVE the
        # _looks_like_user_question branch so "What? That makes no sense" is
        # caught here, not answered as a generic question.
        lines.append(
            "Primary purpose: you guessed or invented a detail and the human is "
            "correcting you. Acknowledge the miss in ONE short beat and drop that "
            "thread entirely. Do NOT re-explain or defend how you got there, do "
            "NOT restate your reasoning, and do NOT re-ask the question you built "
            "on the wrong detail. No new question. Then continue only from what "
            "they actually said."
        )
        plan.purpose = "grounding_repair"
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

    if invitation_accepted:
        # Field 2026-08-27 13:37:05 — Rex asked "Want to sit with me a minute?",
        # Bret said "Yeah", and the yes got a jab ("You're spared from your own bad
        # timing for another minute") followed by 47 seconds of nothing. A yes to an
        # invitation needs its own purpose: land the yes, then just be there.
        lines.append(
            "Primary purpose: they just said YES to the invitation you extended — "
            "to sit with you, to stay a minute, or to take up whatever you just "
            "offered. Nothing more is required of them. Give ONE short, warm, "
            "genuine beat that accepts the yes and settles in ('good', 'stay as "
            "long as you want'), then stop. No new questions, no topic pivot, and "
            "do not tease or needle them for saying yes. Companionable quiet after "
            "this is the point, not a failure."
        )
        plan.purpose = "companionable"
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

    # MOVE-ON OVERRIDE: the human wants off this thread (bored / the bit isn't landing / asked
    # for something else). DROP the topic and change direction — this beats the "stay on this
    # exact topic, do not pivot" agendas below that were grinding one dead metaphor (field
    # 2026-06-30). Ban the just-dropped topic so the reply AND idle banter both stop re-raising it.
    if bool(getattr(config, "SUBJECT_CHANGE_ON_CUE_ENABLED", True)) and _wants_new_direction(text):
        try:
            from intelligence import interaction as _interaction
            _ban = str((answered_question or {}).get("question_text") or "") if answered_question else ""
            if _ban:
                _interaction._record_banned_topic(_ban)
        except Exception as exc:
            _log.debug("move-on topic ban failed: %s", exc)
        next_q = _next_useful_question(person_id) if person_id is not None else None
        pivot = (
            "Primary purpose: the human is telling you to MOVE ON — they're done with this "
            "thread, or the bit/metaphor isn't landing. DROP the current topic, bit, and "
            "metaphor ENTIRELY: do not continue it, defend it, explain it, or reference it "
            "again. Give ONE quick, warm, self-aware beat that you're switching gears (no "
            "sulking, no 'fair, ...' loop), then genuinely CHANGE THE SUBJECT — ask about "
            "something they're actually into or open a fresh, specific topic. Do NOT take "
            "another lap on the dead one."
        )
        if next_q:
            pivot += f" A fresh thing to ask about if it fits: {next_q['text']!r}."
        lines.append(pivot)
        plan.explicit_followup = True
        return _finish(plan, lines, purpose="interest")

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
        # If they just answered Rex's plan clarifier ("Where are you camping?" → "Fraser
        # Flats"), turn that into the what-if suggestion instead of a generic ack.
        plan_suggest = _plan_clarify_answer(plan, lines, text, person_id, answered_question)
        if plan_suggest is not None:
            return plan_suggest
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
                "answering, ask at most one short follow-up that builds on what "
                "they said, or carry the turn with a specific Rex opinion / light "
                "roast instead. If their answer was TERSE — a bare thing like 'my "
                "car' — the curious follow-up about the THING they named is almost "
                "always the right move (what kind, what happened, how it went); a "
                "quip with no question is a dead-end there. "
                "Don't re-ask what they just answered. You don't have "
                "to stay welded to this exact subject — if the thread is clearly "
                "thinning, a brief, natural change of tack is welcome (not a fresh interview)."
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
            # What-if/plans: sparse plan → clarifying question; specific plan or
            # no-plans → concrete suggestion. Falls back to the generic acknowledgment
            # when the feature is off, it's not really a plan, or the plan was already
            # handled this session.
            planned = _plan_branch(plan, lines, text, person_id)
            if planned is not None:
                return planned
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
                "question that continues this thread — but don't grind one topic or "
                "metaphor turn after turn; if it's clearly run its course, a natural "
                "change of tack beats another lap (just don't launch a fresh interview)."
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


_GROUNDING_CORRECTION_PAT = re.compile(
    r"\b("
    r"(?:that|this|it)\s+(?:makes no sense|doesn'?t make (?:any |much )?sense|"
    r"made no sense|makes zero sense)|"
    r"none of (?:that|this) (?:makes sense|is right)|"
    r"you'?re (?:just )?making (?:that|this|it|stuff) up|"
    r"you (?:just )?(?:invented|assumed|made (?:that|this|it|the|a)? ?up|made up)\b|"
    r"i never said (?:that|this)|i didn'?t (?:say|mention)|"
    r"where are you getting (?:that|this)|"
    r"you mean my \w+\s*\??$|"
    r"that'?s not what i (?:was |said i was )?(?:building|doing|making|talking about)"
    r")",
    re.IGNORECASE,
)


def _looks_like_offscreen_correction(text: str) -> bool:
    return bool(_OFFSCREEN_CORRECTION_PAT.search(text or ""))


def _looks_like_grounding_correction(text: str) -> bool:
    return bool(_GROUNDING_CORRECTION_PAT.search(text or ""))


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
