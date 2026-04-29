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
_SENTENCE_SPLIT = re.compile(r"[^.!?]+[.!?]*")
_WORD_PAT = re.compile(r"[A-Za-z0-9']+")
_VISUAL_PAT = re.compile(
    r"\b(i can see|i see you|you look|you're looking|you are looking|"
    r"in the frame|on camera|your face|your shirt|your outfit|lying on|"
    r"on the bed|in bed|the room looks|dimly lit|camera)\b",
    re.IGNORECASE,
)
_ROAST_PAT = re.compile(
    r"\b(pathetic|pitiful|sad excuse|glorified|not-so-mighty|mediocrity|"
    r"blunder|organic thoughts|exhaust ports|can't handle the truth|"
    r"disaster|tragic|lower your standards|pretend i have friends|"
    r"let'?s pretend|life decisions|embarrass yourself|brilliance in basic|"
    r"walking software outage|dumpster fire|trainwreck|train wreck|clown show|"
    r"bad decisions?|questionable choices?|questionable life choices?|"
    r"meatbag|carbon-based|meat-based|malfunctioning organic)\b",
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
    r"confused|primitive|fragile|malfunctioning|squishy|limited|inferior|"
    r"questionable|disaster|mess|bad decisions?|life choices"
    r")\b|\b("
    r"confused|primitive|fragile|malfunctioning|squishy|limited|inferior|"
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
_DANGLING_WORDS = {
    "a", "an", "and", "are", "as", "at", "because", "but", "for", "from",
    "if", "in", "into", "like", "of", "on", "or", "so", "than", "that",
    "the", "to", "with", "according",
}
_HARD_NO_QUESTION_PAT = re.compile(
    r"(do not ask|don't ask|no new questions|without adding a new question|"
    r"do not add a new follow-up|question budget is spent|"
    r"do not ask another question|no follow-up question)",
    re.IGNORECASE,
)
_ASK_ALLOWED_PAT = re.compile(
    r"(ask who|ask .* name|ask .* question|one question|one short follow-up|"
    r"tightly related follow-up|weave in this one question|ending in a question mark)",
    re.IGNORECASE,
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
) -> SocialFrame:
    plan = response_length.classify(user_text, answered_question=answered_question)
    energy = _safe_user_energy()
    empathy_entry = _safe_empathy(person_id)
    empathy_mode = ((empathy_entry or {}).get("mode") or {}).get("mode", "default")
    affect = ((empathy_entry or {}).get("result") or {}).get("affect", "neutral")
    sensitivity = ((empathy_entry or {}).get("result") or {}).get(
        "topic_sensitivity", "none"
    )

    purpose = _purpose_from(agenda_directive, plan.reason, energy)
    unknown_count = _unknown_visible_count()
    urgent_identity = _urgent_group_identity(agenda_directive)
    user_asked_question = _looks_like_user_question(user_text)
    budget_allows = _question_budget_allows()

    allow_question = False
    if urgent_identity and unknown_count:
        allow_question = True
    elif unknown_count and person_id is not None and _ASK_ALLOWED_PAT.search(agenda_directive):
        allow_question = True
    elif answered_question is not None:
        allow_question = bool(
            budget_allows and _ASK_ALLOWED_PAT.search(agenda_directive)
        )
    elif _HARD_NO_QUESTION_PAT.search(agenda_directive):
        allow_question = False
    elif budget_allows and _ASK_ALLOWED_PAT.search(agenda_directive):
        allow_question = True
    elif user_asked_question:
        allow_question = False
    elif budget_allows and plan.target not in {"micro"}:
        allow_question = False

    if plan.max_words <= 12 or plan.target == "micro":
        allow_question = False
    if urgent_identity and unknown_count:
        allow_question = True
        plan.max_words = max(plan.max_words, 28)
        plan.max_sentences = max(plan.max_sentences, 2)

    allow_visual = _visual_allowed(
        user_text,
        agenda_directive,
        plan.target,
        empathy_mode,
        affect,
        sensitivity,
    )
    roast_level = _roast_level(person_id, plan.target, empathy_mode, affect, sensitivity)

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
        "A visual remark is allowed only if it directly connects to the human's turn."
        if frame.allow_visual_comment
        else "Do not mention what you see, the camera, the room, their face, or their posture."
    )
    roast_rule = {
        "none": "No roasts or pointed teasing this turn.",
        "light": "If you roast, make it a tiny surface-level tap.",
        "normal": "Normal Rex banter is allowed, but keep it socially on-target.",
    }.get(frame.allow_roast, "Keep teasing mild and socially on-target.")
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


def govern_response(text: str, frame: SocialFrame) -> GovernResult:
    if not getattr(config, "SOCIAL_FRAME_GOVERNOR_ENABLED", True):
        return GovernResult((text or "").strip(), False, [])

    original = (text or "").strip()
    current = _normalize_text(original)
    notes: list[str] = []
    if not current:
        return GovernResult(_fallback(frame), True, ["empty"])

    sentences = _sentences(current)
    if not frame.allow_question:
        kept = [s for s in sentences if "?" not in s]
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_question")

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

    if not sentences:
        current = _fallback(frame)
        notes.append("fallback")
    else:
        if len(sentences) > frame.max_sentences:
            sentences = _trim_sentences(sentences, frame)
            notes.append("trimmed_sentences")
        current = " ".join(s.strip() for s in sentences if s.strip())

    trimmed = _trim_words(current, frame.max_words)
    if trimmed != current:
        current = trimmed
        notes.append("trimmed_words")

    current = _normalize_text(current)
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


def _visual_allowed(
    user_text: str,
    agenda_directive: str,
    target: str,
    empathy_mode: str,
    affect: str,
    sensitivity: str,
) -> bool:
    text = (user_text or "").lower()
    if re.search(r"\b(see|look|looking|camera|face|shirt|room|bed|posture)\b", text):
        return True
    if target == "micro":
        return False
    if empathy_mode in {"listen", "support", "validate", "ground", "brief"}:
        return False
    if affect in {"sad", "withdrawn", "angry", "anxious"} or sensitivity != "none":
        return False
    return "available environmental cue" in (agenda_directive or "").lower()


def _roast_level(
    person_id: Optional[int],
    target: str,
    empathy_mode: str,
    affect: str,
    sensitivity: str,
) -> str:
    try:
        cooldown = float(getattr(config, "TONE_REPAIR_NO_ROAST_SECS", 180.0) or 0.0)
        if cooldown and repair_moves.recent_tone_repair(cooldown):
            return "none"
    except Exception:
        pass
    if empathy_mode in {"listen", "support", "validate", "ground", "brief"}:
        return "none"
    if affect in {"sad", "withdrawn", "angry", "anxious"} or sensitivity == "heavy":
        return "none"
    if person_id is not None:
        try:
            if boundary_memory.is_blocked(person_id, "roast", "anything"):
                return "none"
        except Exception:
            pass
        try:
            prefs = facts_memory.get_facts_by_category(person_id, "preference")
            pref_text = " ".join(str(p.get("value") or "").lower() for p in prefs)
            if "dislikes sharp roasts" in pref_text or "prefers direct answers" in pref_text:
                return "none"
            if "likes light roasts" in pref_text and target not in {"micro", "brief"}:
                return "light"
        except Exception:
            pass
    if target in {"micro", "brief"}:
        return "light"
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
        except Exception:
            pass
    try:
        cast = social_scene.conversation_cast_context(
            world_state.snapshot(),
            current_person_id=person_id,
        )
        if cast and cast.addressee:
            return cast.addressee
    except Exception:
        pass
    if person_id is None:
        return "unknown person"
    try:
        person = people_memory.get_person(person_id)
        if person and person.get("name"):
            return str(person["name"])
    except Exception:
        pass
    return f"person_id={person_id}"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _sentences(text: str) -> list[str]:
    pieces = [m.group(0).strip() for m in _SENTENCE_SPLIT.finditer(text or "")]
    return [p for p in pieces if p]


def _trim_sentences(sentences: list[str], frame: SocialFrame) -> list[str]:
    limit = max(0, int(frame.max_sentences or 0))
    if limit <= 0:
        return []

    # If a follow-up question is permitted, keep one in the final shape instead
    # of letting an opener like "Ah, Star Trek!" consume the whole budget.
    if frame.allow_question and limit >= 2:
        question_index = next(
            (idx for idx, sentence in enumerate(sentences) if "?" in sentence),
            None,
        )
        if question_index is not None and question_index >= limit:
            prefix = [
                sentence
                for idx, sentence in enumerate(sentences)
                if idx != question_index and "?" not in sentence
            ][: limit - 1]
            return [*prefix, sentences[question_index]]

    return sentences[:limit]


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


def _fallback(frame: SocialFrame) -> str:
    if frame.purpose == "check_alive":
        return "I'm here."
    if frame.purpose in {"closure", "answer_ack"} or frame.max_words <= 12:
        return "Got it."
    if frame.allow_roast == "none":
        return "I hear you."
    return "Fair enough."
