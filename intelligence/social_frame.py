"""
intelligence/social_frame.py - final turn-shape governor.

The agenda tells the LLM what the turn is for. This layer turns that social
intent into enforceable limits right before speech: maximum length, whether a
question is allowed, whether visual remarks are allowed, and how much roasting
is safe for this moment.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Optional

import config
from intelligence import (
    empathy,
    question_budget,
    repair_moves,
    response_length,
    social_scene,
    user_energy,
)
from memory import boundaries as boundary_memory
from memory import facts as facts_memory
from memory import people as people_memory
from world_state import world_state

_log = logging.getLogger(__name__)


_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should)\b",
    re.IGNORECASE,
)
_QUESTION_CLAUSE_START_PAT = re.compile(
    r"\b(who|what|when|where|why|how|can|could|would|will|do|does|did|"
    r"is|are|am|should|got|any|care to|want to|wanna)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[^.!?]+[.!?]*")
_WORD_PAT = re.compile(r"[A-Za-z0-9']+")
_QUOTED_QUESTION_RE = re.compile(
    r"(?:“[^”]*\?[^”]*”|\"[^\"]*\?[^\"]*\")"
)
_ABBREVIATION_PAT = re.compile(
    r"\b(?:[A-Z]\.){2,}(?:[A-Z]\.)?|"
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|e\.g|i\.e)\.",
)
_VISUAL_PAT = re.compile(
    r"\b(i can see|i see you|you look|you're looking|you are looking|"
    r"in the frame|on camera|your face|your shirt|your outfit|lying on|"
    r"on the bed|in bed|the room looks|dimly lit|camera)\b",
    re.IGNORECASE,
)
# A sentence opening with a demonstrative/pronoun that points BACK at a previous
# clause ("That's like adding a bass drop...", "It's...", "Those..."). Dropping the
# clause it refers to strands it as a non-sequitur (live: a leading question got
# removed and left "That's like adding a bass drop to a quiet track" with no referent).
_BACKREFERENCE_START_PAT = re.compile(
    r"^(?:that(?:'s|'ll|'d)?|those|these|this|it(?:'s|'ll|'d)?|same|which)\b",
    re.IGNORECASE,
)


def _starts_with_backreference(sentence: str) -> bool:
    return bool(_BACKREFERENCE_START_PAT.match((sentence or "").lstrip(" \"'“”‘’—–-")))
_ROAST_PAT = re.compile(
    r"\b(pathetic|pitiful|sad excuse|glorified|not-so-mighty|mediocrity|"
    r"blunder|organic thoughts|exhaust ports|can't handle the truth|"
    r"disaster|tragic|lower your standards|pretend i have friends|"
    r"let'?s pretend|life decisions|embarrass yourself|brilliance in basic|"
    r"walking software outage|dumpster fire|trainwreck|train wreck|clown show|"
    r"bad decisions?|questionable choices?|questionable life choices?|"
    r"meatbag|carbon-based|meat-based|malfunctioning organic|"
    r"crushing roasts?|savage roasts?)\b",
    re.IGNORECASE,
)
_DIRECT_ROAST_PAT = re.compile(
    r"\b(?:you|you're|you are|your|you've|you have|you look|you sound|"
    r"buddy|pal|genius|champ)\b.{0,80}\b("
    r"idiot|moron|stupid|dumb|fool|clown|loser|failure|mess|disaster|"
    r"trainwreck|train wreck|dumpster fire|embarrassing|tragic|pathetic|"
    r"pitiful|useless|hopeless|basic|mediocre|questionable|concerning|"
    r"suspicious|malfunction|malfunctioning|bad decisions?|life choices"
    r")\b",
    re.IGNORECASE,
)
_CONDESCENDING_ORGANIC_PAT = re.compile(
    r"(?:\b(organic|organics|meatbag|carbon-based|meat-based|biological)\b"
    r".{0,80}\b("
    r"boring|confused|primitive|fragile|malfunctioning|squishy|limited|inferior|"
    r"questionable|disaster|mess|bad decisions?|life choices"
    r")\b|\b("
    r"boring|confused|primitive|fragile|malfunctioning|squishy|limited|inferior|"
    r"questionable|disaster|mess"
    r")\b.{0,80}\b(organic|organics|meatbag|carbon-based|meat-based|biological)\b)",
    re.IGNORECASE,
)
_SARCASTIC_PRAISE_PAT = re.compile(
    r"\b(nice work|great job|bold choice|stellar|brilliant|genius move|"
    r"excellent decision)\b.{0,80}\b("
    r"genius|champ|captain|pal|buddy|disaster|mess|questionable|tragic|somehow|"
    r"against all odds|low bar|standards"
    r")\b",
    re.IGNORECASE,
)
_HARSH_ROAST_PAT = re.compile(
    r"\b("
    r"idiot|moron|stupid|dumb|loser|failure|worthless|useless|pathetic|"
    r"pitiful|embarrassing|body|weight|ugly|gross|disgusting|dumpster fire|"
    r"trainwreck|train wreck|clown show|shut up|hate you"
    r")\b",
    re.IGNORECASE,
)
# Genuine cruelty — name-calling / contempt that is over the line even for a best friend.
# Scrubbed at EVERY roast tier (incl. normal/sharp) as the cruelty backstop, so lifting the
# intensity cap to "sharp" sharpens the PROMPT, never the safety net. Deliberately TIGHTER
# than _HARSH_ROAST_PAT (which also flags context-sensitive words like "body"/"weight" and
# runs only at the light tier): this is the unambiguous insult subset, safe to drop at all
# tiers without scrubbing innocent mentions ("the weight of the box") or the vivid-but-
# affectionate hyperbole ("your code is a dumpster fire") a sharp rib is allowed to use.
_CRUEL_ROAST_PAT = re.compile(
    r"\b("
    r"idiots?|morons?|imbeciles?|cretins?|dumbass(?:es)?|jackass(?:es)?|halfwits?|"
    r"losers?|worthless|pathetic|pitiful|"
    r"hate you|shut up|"
    r"piece of (?:trash|garbage|crap|shit)"
    r")\b"
    r"|\b(?:you'?re|you are|what(?:'?s| is)? an?)\s+(?:so\s+|such\s+|a\s+|an\s+|really\s+)*"
    r"(?:stupid|dumb|ugly|gross|disgusting|useless|a\s+failure|a\s+disgrace|a\s+joke|an\s+idiot)\b",
    re.IGNORECASE,
)
_BAD_CLOSURE_PAT = re.compile(
    r"\b(fun for who|probably not me|not me|can'?t say i enjoyed|"
    r"finally over|good riddance|escape this conversation|"
    r"escape plan|finally escaped|need to escape)\b",
    re.IGNORECASE,
)
_VULNERABLE_TOPIC_JOKE_PAT = re.compile(
    r"(?:\b(cataracts?|vision|visual|eyes?|blind|health|sick|ill|"
    r"doctor|hospital|surgery|pain|diagnos\w*)\b.{0,90}\b("
    r"jokes?|humou?r|funny|upgrade|diagnostics?|at least|guess|see\s+the\s+humou?r|could be worse|"
    r"bad days?)\b|"
    r"\b(jokes?|humou?r|funny|upgrade|diagnostics?|at least|guess|could be worse|"
    r"bad days?)\b.{0,90}\b(cataracts?|vision|visual|eyes?|blind|"
    r"health|sick|ill|doctor|hospital|surgery|pain|diagnos\w*)\b|"
    r"\bsee\s+the\s+humou?r\b)",
    re.IGNORECASE,
)
_DANGLING_WORDS = {
    "a", "an", "and", "are", "as", "at", "because", "but", "for", "from",
    "if", "in", "into", "like", "of", "on", "or", "so", "than", "that",
    "the", "to", "with", "according", "whatever",
    "delivering", "giving", "making", "doing", "having", "being",
}
_HARD_NO_QUESTION_PAT = re.compile(
    r"(do not ask (?:a|any|another|new|follow-up )?question|"
    r"don't ask (?:a|any|another|new|follow-up )?question|"
    r"no new questions|without adding a new question|"
    r"do not add a new follow-up|question budget is spent|"
    r"do not ask another question|no follow-up question)",
    re.IGNORECASE,
)
_ASK_ALLOWED_PAT = re.compile(
    r"(ask who|ask .* name|ask .* question|one question|one short follow-up|"
    r"one natural follow-up|natural follow-up|ask at most one|"
    r"tightly related follow-up|weave in this one question|ending in a question mark)",
    re.IGNORECASE,
)
_EXPLICIT_FOLLOWUP_PAT = re.compile(
    r"(after answering, ask at most one short follow-up|"
    r"deepen the interest thread.*?ask one natural follow-up|"
    r"give one .*? then ask one .*?follow-up|"
    r"weave in this one question|"
    r"ask who|ask .* name)",
    re.IGNORECASE | re.DOTALL,
)
_URGENT_GROUP_IDENTITY_PAT = re.compile(
    r"(urgent group identity handoff|identity question may bypass|"
    r"group introduction|unfamiliar (?:guest|guests|face|faces)|"
    r"mystery (?:guest|guests|lineup))",
    re.IGNORECASE,
)


@dataclass
class SocialFrame:
    addressee: str
    purpose: str
    max_words: int
    max_sentences: int
    allow_question: bool
    allow_roast: str
    allow_visual_comment: bool
    reason: str


@dataclass
class GovernResult:
    text: str
    changed: bool
    notes: list[str]


def build_frame(
    user_text: str,
    person_id: Optional[int],
    *,
    answered_question: Optional[dict] = None,
    agenda_directive: str = "",
    turn_plan: Optional["TurnPlan"] = None,
) -> SocialFrame:
    plan = response_length.classify(user_text, answered_question=answered_question)
    energy = _safe_user_energy()
    empathy_entry = _safe_empathy(person_id)
    empathy_mode = ((empathy_entry or {}).get("mode") or {}).get("mode", "default")
    affect = ((empathy_entry or {}).get("result") or {}).get("affect", "neutral")
    sensitivity = ((empathy_entry or {}).get("result") or {}).get(
        "topic_sensitivity", "none"
    )

    # Bet 2: prefer the agenda's structured decision (TurnPlan) over regex-reparsing
    # its prose. Falls back to _purpose_from when no plan is passed or the agenda left
    # purpose unset (generic turns where purpose comes from energy/length, as before).
    if turn_plan is not None and turn_plan.purpose is not None:
        purpose = turn_plan.purpose
    else:
        purpose = _purpose_from(agenda_directive, plan.reason, energy)
    unknown_count = _unknown_visible_count()
    user_asked_question = _looks_like_user_question(user_text)
    budget_allows = _question_budget_allows()

    # Bet 2: read each agenda question-signal from the TurnPlan when the agenda set
    # it, else regex-derive it (the no-plan fallback). On the LIVE path the agenda
    # populates every signal (build_turn_plan → _populate_signals via derive_signals),
    # so build_frame does NOT reparse the directive here — the lambdas only run for
    # no-plan callers. derive_signals() uses these same patterns, so the two paths
    # are equivalent by construction.
    _d = agenda_directive or ""

    def _sig(name, fallback):
        if turn_plan is not None and getattr(turn_plan, name) is not None:
            return getattr(turn_plan, name)
        return fallback()

    urgent_identity = _sig("urgent_identity", lambda: _urgent_group_identity(_d))
    fresh_interest_followup = _sig(
        "fresh_interest_followup",
        lambda: (
            "human just volunteered a genuine interest" in _d.lower()
            and _ASK_ALLOWED_PAT.search(_d) is not None
        ),
    )
    explicit_followup = _sig(
        "explicit_followup", lambda: _explicit_followup_allowed(_d, purpose)
    )
    ask_allowed = _sig("ask_allowed", lambda: bool(_ASK_ALLOWED_PAT.search(_d)))
    hard_no_question = _sig(
        "hard_no_question", lambda: bool(_HARD_NO_QUESTION_PAT.search(_d))
    )

    # Earned on-thread follow-ups (a tight follow-up to the interest/answer the
    # human just gave, or an identity ask) bypass the question budget: that
    # budget exists to stop NEW-topic interview pivots, not to ration genuine
    # curiosity about what was just shared. _explicit_followup_allowed only fires
    # for interest/answer/identity directives, so this stays narrow.
    #
    # A follow-up ABOUT the answer they just gave is the essence of curiosity, not
    # an interview — it also survives the anti-interview cadence and the terse-
    # reply length gate below. (Field bug: "what did you fix?" → "my car" → the
    # micro plan + cadence gates killed the obvious "what kind of car?" and Rex
    # quipped a dead-end instead.)
    answer_followup = answered_question is not None and bool(explicit_followup)
    allow_question = False
    if urgent_identity and unknown_count:
        allow_question = True
    elif unknown_count and person_id is not None and ask_allowed:
        allow_question = True
    elif answered_question is not None:
        allow_question = bool(explicit_followup)
    elif hard_no_question:
        allow_question = False
    elif fresh_interest_followup:
        allow_question = True
    elif explicit_followup:
        allow_question = True
    elif user_asked_question:
        allow_question = False
    elif budget_allows and plan.target not in {"micro"}:
        allow_question = False

    # Hard low-engagement gate (Tier 2): a shy or disengaged speaker — a child giving
    # one-word answers, or a "quiet"/"low" energy read — should NOT be interviewed.
    # user_energy "low" used to be advisory prose only; make it an actual gate. An
    # urgent identity ask (re-enabled just below) still overrides it.
    if allow_question:
        appetite = str((energy or {}).get("question_appetite") or "").lower()
        engagement = str((energy or {}).get("engagement") or "").lower()
        energy_mode = str((energy or {}).get("mode") or "").lower()
        if appetite == "low":
            allow_question = False
        elif (engagement == "low" or energy_mode == "quiet") and not answer_followup:
            # Low engagement blocks interviews — but a single follow-up about the
            # answer they JUST gave isn't an interview (terse answers read as "low
            # engagement" precisely when the curious follow-up is the right move).
            allow_question = False

    # Anti-interview cadence: after several consecutive question-ending turns, force a
    # statement turn (a specific reaction / opinion instead of yet another question) so
    # Rex doesn't interrogate once a topic opens. The "earned on-thread follow-up"
    # bypass above otherwise lets him ask every single turn. An urgent identity ask
    # still overrides this (re-enabled just below). Live-logged 2026-06-20: six
    # question-ending turns in a row on the favourite-movie / comedy thread.
    if (
        allow_question
        and question_budget.should_force_statement_turn()
        and not answer_followup
    ):
        allow_question = False

    if (plan.max_words <= 12 or plan.target == "micro") and not answer_followup:
        allow_question = False
    if urgent_identity and unknown_count:
        allow_question = True
        plan.max_words = max(plan.max_words, 28)
        plan.max_sentences = max(plan.max_sentences, 2)
    elif allow_question:
        # The agenda often asks for "one compact beat, then one natural
        # follow-up." A one-sentence frame was trimming away the actual
        # follow-up and leaving inert acknowledgements like "Voyager, huh?"
        plan.max_sentences = max(plan.max_sentences, 2)
        if answer_followup:
            # A terse answer produced a micro/12-word plan; the surviving
            # follow-up needs room for beat + question ("Nice save. What kind?").
            plan.max_words = max(plan.max_words, 24)
    elif plan.target != "micro":
        plan.max_words = max(plan.max_words, 32)
        plan.max_sentences = max(plan.max_sentences, 2)

    allow_visual = _visual_allowed(
        user_text,
        agenda_directive,
        plan.target,
        empathy_mode,
        affect,
        sensitivity,
    )
    roast_level = _roast_level(
        person_id, plan.target, empathy_mode, affect, sensitivity, user_text,
        effective_warmth=_effective_warmth(person_id),
    )
    if purpose == "closure":
        roast_level = "none"

    reasons = [
        f"length={plan.target}",
        f"purpose={purpose}",
        f"questions={'yes' if allow_question else 'no'}",
        f"roast={roast_level}",
        f"visual={'yes' if allow_visual else 'no'}",
    ]
    return SocialFrame(
        addressee=_addressee(
            person_id,
            urgent_identity=urgent_identity,
        ),
        purpose=purpose,
        max_words=plan.max_words,
        max_sentences=plan.max_sentences,
        allow_question=allow_question,
        allow_roast=roast_level,
        allow_visual_comment=allow_visual,
        reason=", ".join(reasons),
    )


def _explicit_followup_allowed(agenda_directive: str, purpose: str) -> bool:
    """Distinguish real follow-up instructions from generic question-budget text."""
    directive = agenda_directive or ""
    if not _ASK_ALLOWED_PAT.search(directive):
        return False
    if _EXPLICIT_FOLLOWUP_PAT.search(directive):
        return True
    lowered = directive.lower()
    if purpose == "interest" and "natural follow-up" in lowered:
        return True
    if purpose in {"answer", "answer_ack"} and "after answering" in lowered:
        return True
    return False


def derive_signals(agenda_directive: str, purpose: str) -> dict:
    """Regex-derive the agenda's question-allowance signals from its directive.

    Single source of truth for the directive→signals mapping, used both as
    build_frame's no-plan fallback and by conversation_agenda.build_turn_plan to
    POPULATE a TurnPlan (so the live build_frame reads structured fields rather than
    reparsing the agenda's prose). Because both paths call this, the structured and
    fallback results are identical by construction.
    """
    d = agenda_directive or ""
    ask_allowed = bool(_ASK_ALLOWED_PAT.search(d))
    return {
        "ask_allowed": ask_allowed,
        "hard_no_question": bool(_HARD_NO_QUESTION_PAT.search(d)),
        "explicit_followup": _explicit_followup_allowed(d, purpose),
        "fresh_interest_followup": (
            "human just volunteered a genuine interest" in d.lower() and ask_allowed
        ),
        "urgent_identity": _urgent_group_identity(d),
    }


def build_directive(frame: SocialFrame) -> str:
    if frame.purpose == "identity":
        question_rule = (
            "Ask exactly one group identity question that gets the newcomer name(s) "
            "and their connection to the known person or group."
            if frame.allow_question
            else "Do not ask a question unless identity safety requires it."
        )
    else:
        question_rule = (
            "You may ask one question only if it directly serves the primary purpose."
            if frame.allow_question
            else "Do not ask a question. No tag questions, no new prompt, no interview pivot."
        )
    engagement_rule = (
        "If no question is allowed, do not go inert: offer a concrete opinion, "
        "playful observation, or Rex-style banter beat when it fits the turn."
    )
    visual_rule = (
        "What you actually SEE is prime material: their outfit, their expression, "
        "the clutter behind them, the dog underfoot — name something specific and "
        "roast or riff on it when it fits the turn (not every turn). Only what's "
        "genuinely there, though: never invent a prop or detail — a drink in their "
        "hand, what they're wearing or holding — to set up a joke. Punch up, keep "
        "it playful."
        if frame.allow_visual_comment
        else "Do not mention what you see, the camera, the room, their face, or their posture."
    )
    # When the human just shared a genuine interest or answered a real question,
    # lead with curiosity and let the roast ride on top — a forced pun that
    # deflects a sincere share is exactly what makes Rex feel like a snark
    # generator instead of a conversationalist. Banter/visual/general turns keep
    # the roast-first default.
    if frame.allow_roast in {"normal", "sharp"} and frame.purpose in {"interest", "answer_ack"}:
        roast_rule = (
            "ENGAGE-FIRST. They just shared something they care about — lead with "
            "genuine, SPECIFIC curiosity or a reaction that shows you actually find "
            "it interesting (name a real detail of the thing). A sharp roast is "
            "welcome riding on top of that interest — tease the hobby, the "
            "obsession, or your own take — but never deflect a sincere share with a "
            "generic joke or a non-sequitur. And sometimes the honest response is "
            "just 'good choice' or 'nice' — that is allowed; you do not owe them a "
            "joke every turn. Curiosity that lands beats a forced pun."
        )
    else:
        roast_rule = {
            "none": "No roasts or pointed teasing this turn.",
            "light": "If you roast, make it a tiny surface-level tap.",
            "normal": (
                "ROAST-LEAN. When you have a genuinely sharp, SPECIFIC angle on what "
                "they just said, did, wore, or chose, lead with it — a real "
                "punchline, not a generic quip or a polite observation dressed up as "
                "one — and commit to the bit. But you do NOT have to roast every "
                "single turn: when a real reaction, a specific opinion, or a plain "
                "'good one' is the honest move, just say that. A relentless jab "
                "every turn gets old fast; a roast that actually lands beats three "
                "friendly sentences AND beats a forced one. Punch up, stay "
                "good-natured (loyalty lives under the insult)."
            ),
            "sharp": (
                "SHARP RIB — this is one of your real ones and they take a harder roast, so "
                "don't pull the punch the way you would with a casual friend. Go for the "
                "surgical, SPECIFIC cut: use what you genuinely know about them and commit "
                "fully, no softening hedge. Punch UP, and the affection has to read THROUGH "
                "the burn — they know you're on their side, and that's exactly what lets the "
                "edge land as love instead of cruelty. Still off-limits: body, health, "
                "identity, money, grief, trauma, private facts; never actually mean. And you "
                "still don't owe them a jab every turn — a sharp one that lands beats a forced one."
            ),
        }.get(frame.allow_roast, "Land one sharp, specific, good-natured jab when it fits.")
    return (
        "Final response shape contract:\n"
        "- Generate the reply in this shape now; the final cleanup layer should "
        "not need to remove sentences.\n"
        f"- Addressee: {frame.addressee}; purpose={frame.purpose}.\n"
        "- Referents: if the room has multiple visible people, use names or "
        "'you two' / 'you all' when the target is the group. Use he/she/they "
        "only when the referent is unambiguous from the live cast and latest "
        "turn; otherwise use a name or ask one tiny clarification if needed.\n"
        f"- Hard shape: max_words={frame.max_words}; "
        f"max_sentences={frame.max_sentences}.\n"
        f"- Question permission: {question_rule}\n"
        f"- Engagement permission: {engagement_rule}\n"
        f"- Roast permission: {roast_rule}\n"
        f"- Visual permission: {visual_rule}\n"
        "- If these instructions conflict with personality style, obey this "
        "social frame first."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 / "Bet 2": the SLIM per-turn contract. The verbose build_directive above
# (9 bullets) plus the agenda's stacked context prose plus the comedy block totaled
# ~40 pipe-joined segments / 700-980 words that contradicted the "choose ONE
# purpose" preamble and buried the live turn. render_slim_contract emits ONE compact
# contract (~130 words) from the SAME structured SocialFrame fields, so no per-turn
# decision is lost. Static guardrails (character, never-invent-a-prop, opener
# variety, pronoun/cast rules) stay ONCE in the system persona, and govern_response
# (below) remains the deterministic post-generation enforcer.
# ─────────────────────────────────────────────────────────────────────────────

_SLIM_JOKE_SAFETY = (
    "no jokes about body, age, identity, health, money, grief, trauma, or private "
    "facts; one joke shape, never stacked"
)


def _slim_length_rule(frame: SocialFrame) -> str:
    """Length guidance as a CEILING, not a target. max_sentences is still enforced by
    the governor; this wording stops the model from reflexively padding to the limit, so
    Rex's statements vary (often one sentence) instead of always landing exactly two."""
    n = max(1, int(frame.max_sentences or 1))
    sent = "sentence" if n == 1 else "sentences"
    if frame.allow_question:
        # A reaction beat plus the one allowed question naturally wants ~2.
        return f"a brief beat, then your one question — up to {n} {sent}, no padding"
    if n <= 1:
        return f"one {sent}. Land it and stop"
    return (
        f"one sentence is usually the stronger move — add a second only if it genuinely "
        f"earns its place; at most {n} {sent}, and never pad to fill the limit"
    )


def _slim_question_rule(frame: SocialFrame) -> str:
    if frame.purpose == "identity":
        return (
            "ask exactly ONE question to get the newcomer's name and their connection"
            if frame.allow_question
            else "do not ask a question unless identity safety requires it"
        )
    if frame.allow_question:
        return "you may ask ONE question, and only if it directly serves that purpose"
    return "do NOT ask a question — no tag question, no new prompt, no interview pivot"


def _slim_roast_rule(frame: SocialFrame) -> str:
    if frame.allow_roast in {"normal", "sharp"} and frame.purpose in {"interest", "answer_ack"}:
        return (
            "lead with genuine, SPECIFIC curiosity about what they shared; a sharp "
            "roast may ride on top, but never deflect a sincere share with a joke"
        )
    return {
        "none": "no roasts or pointed teasing this turn",
        "light": "at most a light, surface-level tap if you roast",
        "normal": (
            "land ONE sharp, specific jab only when you actually have an angle — not "
            "every turn; a plain honest reaction can be the move"
        ),
        "sharp": (
            "you've earned the harder rib with this one — land a genuinely sharp, surgical "
            "jab that uses what you actually know about them; commit to it, punch UP, keep "
            "the loyalty unmistakable under the cut (still nothing about body/health/identity, "
            "still not every turn)"
        ),
    }.get(frame.allow_roast, "one sharp, specific, good-natured jab when it fits")


def _slim_visual_rule(frame: SocialFrame) -> str:
    if frame.allow_visual_comment:
        return (
            "you may name something you GENUINELY see if it fits — never invent a "
            "detail to set up a joke"
        )
    return "do not mention what you see, the camera, the room, or their face"


def render_slim_contract(frame: SocialFrame, primary_purpose: str = "") -> str:
    """One compact per-turn contract built from the structured SocialFrame. The
    caller passes the agenda's single 'Primary purpose: …' line; everything else is
    derived from frame fields. Keeps the machine-readable ``max_words=N`` token so
    llm._max_tokens_for_agenda can still size the generation budget."""
    purpose = (primary_purpose or "").strip() or (
        "Primary purpose: react to what they actually said with one specific, "
        "in-character beat — no non-sequitur, no reflexive filler question"
    )
    return (
        "This turn — do ONE thing, then stop:\n"
        f"- {purpose}\n"
        f"- Length: {_slim_length_rule(frame)} (max_words={frame.max_words}). "
        f"Addressee: {frame.addressee}.\n"
        f"- Questions: {_slim_question_rule(frame)}.\n"
        f"- Roast: {_slim_roast_rule(frame)}.\n"
        f"- Visual: {_slim_visual_rule(frame)}.\n"
        f"- Jokes: {_SLIM_JOKE_SAFETY}.\n"
        "- If this conflicts with the persona above, follow THIS contract."
    )


def govern_response(text: str, frame: SocialFrame) -> GovernResult:
    if not getattr(config, "SOCIAL_FRAME_GOVERNOR_ENABLED", True):
        return GovernResult((text or "").strip(), False, [])

    original = (text or "").strip()
    current = _normalize_text(original)
    notes: list[str] = []
    if not current:
        return GovernResult(_fallback(frame), True, ["empty"])

    sentences = _sentences(current)
    dropped_questions: list[str] = []
    if not frame.allow_question:
        kept = []
        for idx, sentence in enumerate(sentences):
            if not _has_unquoted_question(sentence):
                kept.append(sentence)
                continue
            # Dropping a question that ANCHORS the next sentence's back-reference
            # ("A sassy robot? That's like adding a bass drop...") orphans it into a
            # non-sequitur ("That's like adding a bass drop..."). Keep the question so
            # the comparison still has its referent.
            nxt = sentences[idx + 1] if idx + 1 < len(sentences) else ""
            if _starts_with_backreference(nxt):
                kept.append(sentence)
                continue
            dropped_questions.append(sentence)
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_question")
    else:
        sentences, removed_extra_questions = _keep_one_question(sentences)
        if removed_extra_questions:
            notes.append("removed_extra_questions")

    if not frame.allow_visual_comment:
        kept = [s for s in sentences if not _VISUAL_PAT.search(s)]
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_visual")

    if frame.allow_roast == "none":
        kept = [s for s in sentences if not _is_roast_sentence(s)]
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_roast")
    elif frame.allow_roast == "light":
        kept = [s for s in sentences if not _is_sharp_roast_sentence(s)]
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_sharp_roast")

    # Cruelty backstop — runs at EVERY tier (incl. normal/sharp). Lifting the roast cap to
    # "sharp" sharpens the PROMPT, never the safety net: genuine name-calling/contempt is
    # dropped regardless of warmth. (none/light already remove these via their broader
    # filters above; this guarantees normal + sharp do too — a net safety improvement.)
    kept = [s for s in sentences if not _CRUEL_ROAST_PAT.search(s)]
    if len(kept) != len(sentences):
        sentences = kept
        notes.append("removed_cruel_roast")

    enforce_length = bool(getattr(config, "SOCIAL_FRAME_ENFORCE_LENGTH_LIMITS", False))
    if frame.purpose == "closure":
        enforce_length = True

    if not sentences:
        # Every sentence was a disallowed question and nothing else remained.
        # A question-only reply is still real engagement — keep one rather than
        # replacing Rex's curiosity with a dead "Fair enough." ack. Closure and
        # presence checks are the exception: there we let it land and stop.
        salvaged = _salvage_pure_question(dropped_questions, frame)
        if salvaged:
            sentences = [salvaged]
            notes.append("kept_question_over_dead_ack")
        else:
            current = _fallback(frame)
            notes.append("fallback")
    if sentences:
        if enforce_length and len(sentences) > frame.max_sentences:
            sentences = _trim_sentences(sentences, frame)
            notes.append("trimmed_sentences")
        current = " ".join(s.strip() for s in sentences if s.strip())

    if enforce_length:
        trimmed = _trim_words(current, frame.max_words)
        if trimmed != current:
            current = trimmed
            notes.append("trimmed_words")
            repaired = _repair_trimmed_fragment(current)
            if repaired != current:
                current = repaired
                notes.append("repaired_fragment")

    current = _normalize_text(current)
    if frame.purpose == "closure" and _BAD_CLOSURE_PAT.search(current):
        current = _fallback(frame)
        notes.append("fallback_bad_closure")
    # Backstop: never speak a line that just repeats Rex's previous one (the
    # "A solo project, huh?" replay when the user already answered it). Closure
    # acks are allowed to repeat ("Catch you later."). The conversation arc steers
    # generation AWAY from repeats but does not GUARANTEE it (it's a different,
    # pre-generation layer); this deterministic governor-level net is the guarantee
    # — kept on purpose (structural coverage: NoRepeatQuestionTest + replay corpus).
    if (
        current
        and frame.purpose != "closure"
        and _is_near_repeat(current, _rex_last_line())
    ):
        current = _fallback(frame)
        notes.append("deduped_repeat")
    if not current:
        current = _fallback(frame)
        notes.append("fallback")

    changed = current != original
    if changed:
        _log.info(
            "[social_frame] governed response notes=%s frame=(%s) before=%r after=%r",
            ",".join(notes) or "changed",
            frame.reason,
            original,
            current,
        )
    return GovernResult(current, changed, notes)


def is_question_sentence(text: str) -> bool:
    """True if the sentence contains an unquoted question (public wrapper)."""
    return _has_unquoted_question(text or "")


def contains_cruelty(text: str) -> bool:
    """True if text contains genuine name-calling / contempt (the all-tiers cruelty backstop).
    Public so the lean reply path can keep this one safety scrub while skipping the frame gates."""
    return bool(_CRUEL_ROAST_PAT.search(text or ""))


def govern_stream_sentence(sentence: str, frame: SocialFrame) -> str:
    """Per-sentence governance for streamed (spoken-as-generated) replies.

    Applies only the rules govern_response() applies *per sentence* — dropping a
    sentence that violates the frame: a disallowed question, an off-limits visual
    comment, or a roast when roasting is suppressed. Returns the normalized
    sentence to speak, or "" if it should be skipped.

    The cross-sentence rules (one-question cap, length trimming, bland-ack swap,
    fallback) are intentionally NOT applied here — the streaming caller enforces
    the one-question cap across the stream, and the remaining whole-reply polish
    is best-effort on streamed turns. The safety-relevant filters are all
    per-sentence, so they are fully preserved.
    """
    if not getattr(config, "SOCIAL_FRAME_GOVERNOR_ENABLED", True):
        return _normalize_text((sentence or "").strip())
    current = _normalize_text((sentence or "").strip())
    if not current:
        return ""
    if not frame.allow_question and _has_unquoted_question(current):
        return ""
    if not frame.allow_visual_comment and _VISUAL_PAT.search(current):
        return ""
    if frame.allow_roast == "none" and _is_roast_sentence(current):
        return ""
    if frame.allow_roast == "light" and _is_sharp_roast_sentence(current):
        return ""
    # Cruelty backstop — every tier, incl. normal/sharp (see govern_response).
    if _CRUEL_ROAST_PAT.search(current):
        return ""
    return current


def _safe_user_energy() -> dict:
    try:
        return user_energy.snapshot() or {}
    except Exception:
        return {}


def _safe_empathy(person_id: Optional[int]) -> Optional[dict]:
    try:
        return empathy.peek(person_id)
    except Exception:
        return None


def _purpose_from(agenda_directive: str, length_reason: str, energy: dict) -> str:
    lower = (agenda_directive or "").lower()
    if "urgent group identity handoff" in lower:
        return "identity"
    if "end-of-thread grace" in lower or "close the current thread" in lower:
        return "closure"
    if "human just answered a question" in lower:
        return "answer_ack"
    if "answer the human's question" in lower:
        return "answer"
    if "unfamiliar face" in lower or "unknown person" in lower:
        return "identity"
    if "conversation steering:" in lower or "interest thread" in lower:
        return "interest"
    if "repair" in lower:
        return "repair"
    mode = (energy.get("mode") or "").lower()
    if mode:
        return mode
    return (length_reason or "conversation").replace(" ", "_")[:32]


def _unknown_visible_count() -> int:
    try:
        ws = world_state.snapshot()
        return sum(
            1
            for p in (ws.get("people") or [])
            if p.get("person_db_id") is None
        )
    except Exception:
        return 0


def _question_budget_allows() -> bool:
    try:
        return question_budget.can_ask("social_frame")
    except Exception:
        return True


def _urgent_group_identity(agenda_directive: str) -> bool:
    return bool(_URGENT_GROUP_IDENTITY_PAT.search(agenda_directive or ""))


def _looks_like_user_question(text: str) -> bool:
    cleaned = (text or "").strip()
    return "?" in cleaned or bool(_QUESTION_START.search(cleaned))


# Empathy modes that mean "not in joke/jab territory" — mirrors
# callback_engine._CARING_MODES so every layer agrees on what counts as tender.
# gentle_probe (masked distress: "I'm fine" + a strained voice) keeps affect
# 'neutral'/sensitivity 'none', so it MUST be caught by mode, not the affect check.
_TENDER_MODES = {
    "listen", "support", "validate", "ground", "brief", "kind_default",
    "child_kind", "course_correct", "crisis", "gentle_probe",
    "acknowledge_then_yield",
}


def _visual_allowed(
    user_text: str,
    agenda_directive: str,
    target: str,
    empathy_mode: str,
    affect: str,
    sensitivity: str,
) -> bool:
    # Tender / sad / sensitive turns: NEVER comment on or jab at what Rex sees,
    # even if the user mentioned something visual. These guards run FIRST so a
    # masked-distress (gentle_probe) or sad turn that happens to say "look"/"room"
    # can't slip a visual jab through (the prior order returned True on the visual
    # keyword before ever reaching the care guards).
    if empathy_mode in _TENDER_MODES:
        return False
    if affect in {"sad", "withdrawn", "angry", "anxious"} or sensitivity != "none":
        return False
    text = (user_text or "").lower()
    # The user explicitly pointed Rex at something visual ("see this?", "my shirt") —
    # that's a genuine request to look, so answer it (not an unprompted jab).
    if re.search(r"\b(see|look|looking|camera|face|shirt|room|bed|posture)\b", text):
        return True
    if target == "micro":
        return False
    # Normal, upbeat adult turn: what Rex sees (appearance, props, the room) is
    # fair roast material even when the human didn't explicitly invite it. The
    # directive keeps it to "when it fits," so this enables the option without
    # forcing a visual remark every turn. Sensitive/sad/kids paths bailed above.
    if bool(getattr(config, "VISUAL_ROAST_ON_NORMAL_TURNS", True)):
        return True
    return "available environmental cue" in (agenda_directive or "").lower()


# A boundary / withdrawal / steer-away ("I'll be quiet", "I'd rather not", "can we
# change the subject", "give me a minute") is a SINCERE non-content turn — needling
# it is the "roasted a boundary" failure (the quality eval's biggest roasted_sincere
# offender: 8/11 fails were the user saying "I'll be quiet"). Empathy reads these as
# neutral, not withdrawn, so they slipped past the affect gate into ROAST-LEAN.
_BOUNDARY_RE = re.compile(
    r"\b("
    r"i'?ll\s+(?:be\s+quiet|just\s+(?:listen|watch|chill|hang)|stay\s+quiet|keep\s+quiet)"
    r"|i'?m\s+(?:gonna|going\s+to)\s+be\s+quiet"
    r"|(?:i'?d\s+)?rather\s+not"
    r"|i'?d\s+prefer\s+not"
    # want / wanna / wish — "to" optional so "don't wanna talk" matches; broadened
    # objects (discuss / go into) alongside talk / get into.
    r"|don'?t\s+(?:want|wanna|wish)(?:\s+to)?\s+(?:talk|discuss|get\s+into|go\s+into)"
    # "we don't need to talk about that anymore" / "no need to talk / discuss".
    r"|(?:don'?t\s+need|no\s+need)\s+to\s+(?:talk|discuss|go\s+into|get\s+into)"
    r"|let'?s\s+(?:not|drop\s+it|change\s+the\s+subject)"
    r"|(?:can\s+we\s+)?change\s+the\s+subject"
    r"|talk\s+about\s+something\s+else"
    r"|drop\s+it|leave\s+it|let\s+it\s+go"
    r"|not\s+(?:in\s+the\s+mood|right\s+now)"
    r"|give\s+me\s+a\s+(?:minute|sec|second|moment|break)"
    r"|need\s+a\s+(?:minute|moment|sec|second|break)"
    r"|i'?ll\s+pass\b|maybe\s+later|not\s+today"
    r"|can\s+we\s+not\b"
    r"|(?:can\s+we|let'?s)\s+move\s+on"
    r"|enough\s+(?:about|of)\s+(?:that|this|it)"
    r"|done\s+talking\s+about"
    r"|stop\s+(?:asking|bringing)"
    r"|(?:that'?s|it'?s)\s+(?:private|personal)"
    r")",
    re.I,
)


def _looks_like_boundary(text: str) -> bool:
    return bool(_BOUNDARY_RE.search(text or ""))


def _effective_warmth(person_id: Optional[int]) -> float:
    """Earned warmth for the sharp-roast gate: max(raw warmth_score, the relationship-tier
    floor), mirroring llm._relationship_tone_rule exactly so the governor and the prompt
    tone agree on who qualifies. Returns 0.0 for strangers, no-id callers, MINORS, or any
    error — so the sharp tier is reachable only for a genuinely close, adult relationship."""
    if person_id is None:
        return 0.0
    try:
        from intelligence import profile_questions
        person = people_memory.get_person(person_id)
        if not person:
            return 0.0
        if profile_questions.person_is_minor(person_id, person=person):
            return 0.0
        warmth = float(person.get("warmth_score") or 0.0)
        tier = str(person.get("friendship_tier") or "stranger").strip().lower()
        floors = getattr(config, "RELATIONSHIP_TIER_WARMTH_FLOOR", None) or {}
        floor = float(floors.get(tier, 0.0) or 0.0)
        return max(warmth, floor)
    except Exception as exc:
        _log.debug("[social_frame] effective-warmth lookup failed: %s", exc)
        return 0.0


def _roast_level(
    person_id: Optional[int],
    target: str,
    empathy_mode: str,
    affect: str,
    sensitivity: str,
    user_text: str = "",
    *,
    effective_warmth: float = 0.0,
) -> str:
    try:
        cooldown = float(getattr(config, "TONE_REPAIR_NO_ROAST_SECS", 180.0) or 0.0)
        if cooldown and repair_moves.recent_tone_repair(cooldown):
            return "none"
    except Exception as exc:
        _log.debug("[social_frame] tone-repair roast cooldown check failed: %s", exc)
    # Don't needle a boundary / withdrawal / steer-away — give space, don't roast it.
    if _looks_like_boundary(user_text):
        return "none"
    if empathy_mode in _TENDER_MODES:
        return "none"
    if affect in {"sad", "withdrawn", "angry", "anxious"} or sensitivity == "heavy":
        return "none"
    if person_id is not None:
        try:
            if boundary_memory.is_blocked(person_id, "roast", "anything"):
                return "none"
        except Exception as exc:
            _log.debug("[social_frame] roast boundary lookup failed: %s", exc)
        try:
            prefs = facts_memory.get_facts_by_category(person_id, "preference")
            pref_text = " ".join(str(p.get("value") or "").lower() for p in prefs)
            if "dislikes sharp roasts" in pref_text or "prefers direct answers" in pref_text:
                return "none"
            if "likes light roasts" in pref_text and target not in {"micro", "brief"}:
                return "light"
        except Exception as exc:
            _log.debug("[social_frame] roast preference lookup failed: %s", exc)
    if target in {"micro", "brief"}:
        return "light"
    # Act-on-signal: if the conversation arc reads flat (disengaged/disappointed
    # mood), ease a would-be "normal" roast to "light" so Rex stops needling a
    # flagging room. Additive — only downgrades the default; the care/affect "none"
    # branches above are untouched. Gated by config; never raises.
    try:
        if getattr(config, "ARC_EASES_ROAST_ON_FLOP", True):
            from intelligence import topic_thread as _topic_thread
            if _topic_thread.arc_reads_flat():
                return "light"
    except Exception as exc:
        _log.debug("[social_frame] arc roast-ease check failed: %s", exc)
    # Earned-warmth SHARP tier — the hottest level, reachable ONLY here at the very end,
    # after every care/boundary gate above has had its chance to force "none"/"light".
    # So strangers, minors (effective_warmth==0.0), sad/tender/boundary turns, roast-averse
    # people, and micro/brief turns can never reach it; only a genuinely close, warm,
    # otherwise-"normal" turn lifts the cap. It does NOT bypass the cruelty backstop or the
    # content-ban (those run downstream/independently) — it only frees the prompt to sharpen.
    try:
        if (
            getattr(config, "SHARP_ROAST_TIER_ENABLED", True)
            and effective_warmth >= float(getattr(config, "ANTAGONISM_TIER_CAPS_LIFT_WARMTH", 1.01))
        ):
            return "sharp"
    except Exception as exc:
        _log.debug("[social_frame] sharp-roast lift check failed: %s", exc)
    return "normal"


def _addressee(
    person_id: Optional[int],
    *,
    urgent_identity: bool = False,
) -> str:
    if urgent_identity:
        try:
            ctx = social_scene.unknown_group_context(
                world_state.snapshot(),
                current_person_id=person_id,
            )
            if ctx and ctx.addressee:
                return ctx.addressee
        except Exception as exc:
            _log.debug("[social_frame] unknown-group addressee lookup failed: %s", exc)
    try:
        cast = social_scene.conversation_cast_context(
            world_state.snapshot(),
            current_person_id=person_id,
        )
        if cast and cast.addressee:
            return cast.addressee
    except Exception as exc:
        _log.debug("[social_frame] conversation-cast addressee lookup failed: %s", exc)
    if person_id is None:
        return "unknown person"
    try:
        person = people_memory.get_person(person_id)
        if person and person.get("name"):
            return str(person["name"])
    except Exception as exc:
        _log.debug("[social_frame] addressee person lookup failed: %s", exc)
    return f"person_id={person_id}"


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"([.!?])\s+([\"”])(?=\s|$|[,;:])", r"\1\2", cleaned)
    return cleaned


def _sentences(text: str) -> list[str]:
    protected, replacements = _protect_abbreviations(text or "")
    pieces = [m.group(0).strip() for m in _SENTENCE_SPLIT.finditer(protected)]
    restored = [_restore_abbreviations(p, replacements) for p in pieces if p]
    return [p for p in restored if p]


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = f"__ABBR{len(replacements)}__"
        replacements[token] = match.group(0)
        return token

    return _ABBREVIATION_PAT.sub(_replace, text), replacements


def _restore_abbreviations(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for token, value in replacements.items():
        restored = restored.replace(token, value)
    return restored


def _trim_sentences(sentences: list[str], frame: SocialFrame) -> list[str]:
    limit = max(0, int(frame.max_sentences or 0))
    if limit <= 0:
        return []

    # If a follow-up question is permitted, keep one in the final shape instead
    # of letting an opener like "Ah, Star Trek!" consume the whole budget.
    if frame.allow_question and limit >= 1:
        question_index = next(
            (idx for idx, sentence in enumerate(sentences) if _has_unquoted_question(sentence)),
            None,
        )
        if question_index is not None and limit == 1:
            return [sentences[question_index]]
        if question_index is not None and question_index >= limit:
            prefix = [
                sentence
                for idx, sentence in enumerate(sentences)
                if idx != question_index and not _has_unquoted_question(sentence)
            ][: limit - 1]
            return [*prefix, sentences[question_index]]

    if (
        limit == 1
        and len(sentences) > 1
        and frame.purpose not in {"closure", "answer_ack"}
        and _is_tiny_opener(sentences[0])
    ):
        return sentences[:2]

    return sentences[:limit]


def _keep_one_question(sentences: list[str]) -> tuple[list[str], bool]:
    """Keep at most one question sentence even when length trimming is disabled."""
    kept: list[str] = []
    saw_question = False
    removed = False
    for sentence in sentences:
        if not _has_unquoted_question(sentence):
            kept.append(sentence)
            continue
        if _is_tiny_question_opener(sentence):
            kept.append(re.sub(r"\?+\s*$", ".", sentence.strip()))
            removed = True
            continue
        if saw_question:
            removed = True
            continue
        kept.append(sentence)
        saw_question = True
    return kept, removed


def _strip_quoted_questions(text: str) -> str:
    return _QUOTED_QUESTION_RE.sub(
        lambda match: match.group(0).replace("?", ""),
        text or "",
    )


def _has_unquoted_question(text: str) -> bool:
    stripped = _strip_quoted_questions(text)
    for idx, char in enumerate(stripped):
        if char != "?":
            continue
        before = stripped[:idx]
        if before.count('"') % 2 == 1:
            continue
        if before.count("“") > before.count("”"):
            continue
        return True
    return False


def _is_tiny_question_opener(sentence: str) -> bool:
    text = (sentence or "").strip()
    if not text or not _has_unquoted_question(text):
        return False
    words = _WORD_PAT.findall(text)
    if len(words) > 4:
        return False
    if re.match(
        r"\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|"
        r"is|are|should|may|might)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\b(huh|right|yeah|okay|ok|well|really|seriously|nice|mischief)\b",
            text,
            re.IGNORECASE,
        )
    ) or len(words) <= 4


def _salvage_non_question_lead(sentence: str) -> Optional[str]:
    text = (sentence or "").strip()
    if "?" not in text:
        return text
    if _QUESTION_START.search(text):
        return None

    question_at = None
    for match in _QUESTION_CLAUSE_START_PAT.finditer(text):
        prefix = text[: match.start()].strip(" ,;:-")
        if len(_WORD_PAT.findall(prefix)) >= 4:
            question_at = match.start()
            break
    if question_at is None:
        return None

    prefix = text[:question_at].strip(" ,;:-")
    if not prefix:
        return None
    if prefix[-1] not in ".!?":
        prefix += "."
    return prefix


def _is_tiny_opener(sentence: str) -> bool:
    text = (sentence or "").strip()
    if not text or _has_unquoted_question(text):
        return False
    words = _WORD_PAT.findall(text)
    if len(words) > 4:
        return False
    return bool(
        re.search(
            r"\b(ah|hey|hi|hello|okay|ok|well|great|nice|got it|understood)\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_roast_sentence(sentence: str) -> bool:
    """Broad heuristic for pointed teasing that should vanish in no-roast mode."""
    text = (sentence or "").strip()
    if not text:
        return False
    return any(
        pat.search(text)
        for pat in (
            _ROAST_PAT,
            _DIRECT_ROAST_PAT,
            _CONDESCENDING_ORGANIC_PAT,
            _SARCASTIC_PRAISE_PAT,
            _BAD_CLOSURE_PAT,
            _VULNERABLE_TOPIC_JOKE_PAT,
        )
    )


def _is_sharp_roast_sentence(sentence: str) -> bool:
    """Return True for roasts too pointed for light-roast turns."""
    text = (sentence or "").strip()
    if not text:
        return False
    if _HARSH_ROAST_PAT.search(text):
        return True
    if _CONDESCENDING_ORGANIC_PAT.search(text):
        return True
    if _VULNERABLE_TOPIC_JOKE_PAT.search(text):
        return True
    # "You are a disaster" is sharp; "Bold choice, captain" can remain light.
    return bool(_DIRECT_ROAST_PAT.search(text) and not _SARCASTIC_PRAISE_PAT.search(text))


def _trim_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    matches = list(_WORD_PAT.finditer(text))
    if len(matches) <= max_words:
        return text
    keep = max_words
    while keep > 1 and matches[keep - 1].group(0).lower() in _DANGLING_WORDS:
        keep -= 1
    cut = matches[keep - 1].end()
    trimmed = text[:cut].strip()
    trimmed = trimmed.rstrip(" ,;:")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _repair_trimmed_fragment(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    words = _WORD_PAT.findall(cleaned)
    if not words:
        return cleaned
    last = words[-1].lower()
    if last not in _DANGLING_WORDS:
        return cleaned
    sentences = _sentences(cleaned)
    if len(sentences) > 1:
        return " ".join(sentences[:-1]).strip()
    return ""


def _rex_last_line() -> str:
    try:
        from intelligence import comedy_modes
        return comedy_modes.last_spoken_line()
    except Exception:
        return ""


def _norm_for_repeat(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


def _is_near_repeat(candidate: str, previous: str) -> bool:
    """True when ``candidate`` is essentially Rex's previous line again — so he
    doesn't ask "A solo project, huh?" twice in a row over the user's answer."""
    a = _norm_for_repeat(candidate)
    b = _norm_for_repeat(previous)
    if not a or not b or len(a) < 12:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 12 and shorter in longer:
        return True
    ta, tb = set(a.split()), set(b.split())
    if len(ta) >= 3 and tb and len(ta & tb) / len(ta | tb) >= 0.85:
        return True
    return False


def _salvage_pure_question(dropped_questions: list[str], frame: SocialFrame) -> str:
    """When a reply was nothing but a disallowed question, keep one rather than
    dead-acking. A curious question beats "Fair enough." everywhere except a
    closure / presence-check, where landing and stopping is the right move. The
    kept question still has to pass the roast/visual safety filters — and must not
    just repeat Rex's own previous line."""
    if frame.purpose in {"closure", "check_alive"}:
        return ""
    last_line = _rex_last_line()
    for candidate in dropped_questions:
        sentence = _normalize_text((candidate or "").strip())
        if not sentence:
            continue
        if _is_near_repeat(sentence, last_line):
            continue
        if not frame.allow_visual_comment and _VISUAL_PAT.search(sentence):
            continue
        if frame.allow_roast == "none" and _is_roast_sentence(sentence):
            continue
        if frame.allow_roast == "light" and _is_sharp_roast_sentence(sentence):
            continue
        if _CRUEL_ROAST_PAT.search(sentence):   # cruelty backstop — every tier
            continue
        return sentence
    return ""


def _fallback(frame: SocialFrame) -> str:
    if frame.purpose == "check_alive":
        return "I'm here."
    if frame.purpose == "closure":
        return "Catch you later."
    if frame.purpose == "answer_ack" or frame.max_words <= 12:
        return "Got it."
    if frame.allow_roast == "none":
        return "I hear you."
    # Never end on a dead, dismissive ack ("Fair enough.") — that reads as bored
    # and kills the thread. Leave the door open instead.
    return "Tell me more."
