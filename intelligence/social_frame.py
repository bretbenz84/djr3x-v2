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
from intelligence import empathy, question_budget, response_length, user_energy
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
    r"disaster|tragic)\b",
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
    r"do not ask another|no unrelated.*question)",
    re.IGNORECASE,
)
_ASK_ALLOWED_PAT = re.compile(
    r"(ask who|ask .* name|ask .* question|one question|one short follow-up|"
    r"weave in this one question|ending in a question mark)",
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
    user_asked_question = _looks_like_user_question(user_text)
    budget_allows = _question_budget_allows()

    allow_question = False
    if unknown_count and person_id is not None and _ASK_ALLOWED_PAT.search(agenda_directive):
        allow_question = True
    elif answered_question is not None:
        allow_question = False
    elif user_asked_question:
        allow_question = False
    elif _HARD_NO_QUESTION_PAT.search(agenda_directive):
        allow_question = False
    elif budget_allows and _ASK_ALLOWED_PAT.search(agenda_directive):
        allow_question = True
    elif budget_allows and plan.target not in {"micro"}:
        allow_question = False

    if plan.max_words <= 12 or plan.target == "micro":
        allow_question = False

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
        addressee=_addressee(person_id),
        purpose=purpose,
        max_words=plan.max_words,
        max_sentences=plan.max_sentences,
        allow_question=allow_question,
        allow_roast=roast_level,
        allow_visual_comment=allow_visual,
        reason=", ".join(reasons),
    )


def build_directive(frame: SocialFrame) -> str:
    question_rule = (
        "You may ask one question only if it directly serves the primary purpose."
        if frame.allow_question
        else "Do not ask a question. No tag questions, no new prompt, no interview pivot."
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
        "Social frame governor:\n"
        f"- Addressee: {frame.addressee}; purpose={frame.purpose}.\n"
        f"- Hard shape: max_words={frame.max_words}; "
        f"max_sentences={frame.max_sentences}.\n"
        f"- Question permission: {question_rule}\n"
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
        kept = [s for s in sentences if not _ROAST_PAT.search(s)]
        if len(kept) != len(sentences):
            sentences = kept
            notes.append("removed_roast")

    if not sentences:
        current = _fallback(frame)
        notes.append("fallback")
    else:
        if len(sentences) > frame.max_sentences:
            sentences = sentences[:frame.max_sentences]
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
    if target in {"micro", "brief", "short"}:
        return "light"
    return "normal"


def _addressee(person_id: Optional[int]) -> str:
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
