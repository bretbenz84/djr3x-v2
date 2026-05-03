"""
intelligence/repair_moves.py - explicit conversational repair handling.

When the human says Rex misheard, misunderstood, pushed too hard, or landed a
line badly, this module turns that into a small repair move instead of letting
the normal roast/curiosity machinery treat it like ordinary banter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import threading
import time
from typing import Optional


_MISHEARD_PAT = re.compile(
    r"\b(you (misheard|heard wrong)|i (said|didn't say|did not say)|"
    r"that's not what i said|that is not what i said|not what i said|"
    r"that's not (his|her|their|my) name|that is not (his|her|their|my) name|"
    r"his name was|her name was|their name was|the name was|"
    r"wrong word|wrong name|transcribed wrong)\b",
    re.IGNORECASE,
)
_MISUNDERSTOOD_PAT = re.compile(
    r"\b(you (misunderstood|got it wrong|missed the point)|"
    r"that's not what i meant|that is not what i meant|not what i meant|"
    r"you'?re missing (the point|what i mean)|"
    r"no,? not that|no,? that's wrong|no,? that is wrong|incorrect)\b",
    re.IGNORECASE,
)
_FACTUAL_PAT = re.compile(
    r"\b(you made that up|you'?re making that up|that'?s not true|that is not true|"
    r"you invented that|don'?t assume|do not assume|that didn'?t happen|"
    r"that did not happen|where did you get that|you hallucinated)\b",
    re.IGNORECASE,
)
_TONE_PAT = re.compile(
    r"\b(that was (rude|mean|harsh|uncalled for|distasteful|not funny|too much)|"
    r"that wasn't funny|that wasnt funny|you were (rude|mean|harsh)|"
    r"don'?t roast|do not roast|stop roasting|not a joke|not (?:very )?funny|"
    r"too mean|you went too far)\b",
    re.IGNORECASE,
)
_PACING_PAT = re.compile(
    r"\b(too many questions|stop asking|so many questions|why are you asking|"
    r"this feels like an interview|not an interview|slow down|give me a second|"
    r"let me think)\b",
    re.IGNORECASE,
)
_INTERRUPT_PAT = re.compile(
    r"\b(you interrupted|you cut me off|you talked over me|let me finish|"
    r"i wasn't done|i was still talking)\b",
    re.IGNORECASE,
)
_WRONG_PERSON_PAT = re.compile(
    r"\b(wrong person|not me|that wasn'?t me|that was not me|"
    r"i didn'?t say that|i did not say that|"
    r"you mean (him|her|them|[A-Z][A-Za-z]+)|"
    r"you'?re talking to (him|her|them|the wrong person)|"
    r"that was (him|her|them|[A-Z][A-Za-z]+))\b",
    re.IGNORECASE,
)
_PRONOUN_PAT = re.compile(
    r"\b(wrong pronouns?|not (he|she|him|her)|"
    r"(?:i|they|he|she|[A-Z][A-Za-z]+)\s+(?:use|uses|go by|goes by)\s+"
    r"(?:he/him|she/her|they/them)|"
    r"(?:he/him|she/her|they/them)\s+pronouns?)\b",
    re.IGNORECASE,
)
_REPEAT_PAT = re.compile(
    r"\b(what did you say|say that again|repeat that|come again|"
    r"i didn'?t hear you|i did not hear you|what was that)\b",
    re.IGNORECASE,
)
_CLARIFY_PAT = re.compile(
    r"\b(what do you mean|what are you talking about|huh|i don'?t get it|"
    r"i do not get it|explain that|clarify)\b",
    re.IGNORECASE,
)
_BARE_NEGATION_PAT = re.compile(r"^\s*(no|nope|nah|wrong|incorrect)\s*[.!]?\s*$", re.I)
_CORRECTION_PAT = re.compile(
    r"\b(?:i said|it's|it is|his name was|her name was|their name was|"
    r"that's not his name,? it'?s|that's not her name,? it'?s|"
    r"that's not their name,? it'?s|that's not my name,? it'?s|"
    r"the name was|i meant)\s+(.+)$",
    re.IGNORECASE,
)
_NOT_X_Y_PAT = re.compile(
    r"\b(?:not|it wasn'?t|it was not|that wasn'?t|that was not)\s+(.+?),?\s+"
    r"(?:it'?s|it is|it was|i said|the correct(?:ion)? is)?\s*(.+)$",
    re.IGNORECASE,
)
_NO_COMMA_CORRECTION_PAT = re.compile(
    r"^\s*(?:no|nope|nah|wrong|incorrect)[,\s]+(.+)$",
    re.IGNORECASE,
)

_lock = threading.Lock()
_last_assistant_text: str = ""
_last_assistant_at: float = 0.0
_last_repair_at: float = 0.0
_last_tone_repair_at: float = 0.0

BETTER_LUCK_NEXT_TIME = "I'm sure we'll have better luck next time!"
_BETTER_LUCK_REPAIR_KINDS = {
    "misheard",
    "misunderstood",
    "wrong_person",
    "pronoun",
    "factual",
    "bare_negation",
}


@dataclass
class RepairMove:
    kind: str
    severity: str
    user_text: str
    correction: str = ""
    target: str = ""
    last_assistant_text: str = ""
    detected_at: float = 0.0


def clear() -> None:
    global _last_assistant_text, _last_assistant_at, _last_repair_at, _last_tone_repair_at
    with _lock:
        _last_assistant_text = ""
        _last_assistant_at = 0.0
        _last_repair_at = 0.0
        _last_tone_repair_at = 0.0


def note_assistant_turn(text: str) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    global _last_assistant_text, _last_assistant_at
    with _lock:
        _last_assistant_text = cleaned
        _last_assistant_at = time.monotonic()


def detect(user_text: str) -> Optional[dict]:
    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    kind = ""
    severity = "medium"

    if _INTERRUPT_PAT.search(cleaned):
        kind = "interruption"
        severity = "high"
    elif _TONE_PAT.search(cleaned):
        kind = "tone"
        severity = "high"
    elif _PACING_PAT.search(cleaned):
        kind = "pacing"
        severity = "medium"
    elif _WRONG_PERSON_PAT.search(cleaned):
        kind = "wrong_person"
        severity = "high"
    elif _PRONOUN_PAT.search(cleaned):
        kind = "pronoun"
        severity = "high"
    elif _REPEAT_PAT.search(cleaned):
        kind = "repeat"
        severity = "low"
    elif _CLARIFY_PAT.search(cleaned):
        kind = "clarify"
        severity = "low"
    elif _MISHEARD_PAT.search(cleaned):
        kind = "misheard"
        severity = "medium"
    elif _FACTUAL_PAT.search(cleaned):
        kind = "factual"
        severity = "high"
    elif _MISUNDERSTOOD_PAT.search(cleaned):
        kind = "misunderstood"
        severity = "medium"
    elif _BARE_NEGATION_PAT.match(cleaned):
        kind = "bare_negation"
        severity = "low"

    if not kind:
        return None

    # Avoid treating every ordinary "no" as a repair unless Rex just asked a
    # question or spoke recently enough that the negation is probably feedback.
    now = time.monotonic()
    with _lock:
        last_assistant = _last_assistant_text
        last_assistant_at = _last_assistant_at
        last_repair_at = _last_repair_at
    recent_assistant = last_assistant_at > 0.0 and (now - last_assistant_at) <= 120.0
    if kind in {"repeat", "clarify", "bare_negation"} and not recent_assistant:
        return None
    if kind == "bare_negation" and "?" not in last_assistant:
        return None
    if kind == "bare_negation" and now - last_repair_at < 10.0:
        return None

    correction = _extract_correction(cleaned)
    if kind in {"tone", "pacing", "interruption", "repeat", "clarify", "bare_negation"}:
        correction = ""
    if not correction and kind in {"misheard", "misunderstood", "wrong_person", "pronoun", "factual"}:
        # Preserve the useful part in common forms like "no, Tom Foster".
        no_match = _NO_COMMA_CORRECTION_PAT.match(cleaned)
        correction = (no_match.group(1).strip() if no_match else "")
        if correction.lower() == lowered or len(correction.split()) > 12:
            correction = ""

    move = RepairMove(
        kind=kind,
        severity=severity,
        user_text=cleaned,
        correction=correction.strip(" .!?\"'"),
        target=_extract_target(cleaned),
        last_assistant_text=last_assistant,
        detected_at=now,
    )
    return asdict(move)


def mark_handled(kind: str = "") -> None:
    global _last_repair_at, _last_tone_repair_at
    with _lock:
        now = time.monotonic()
        _last_repair_at = now
        if (kind or "").lower() == "tone":
            _last_tone_repair_at = now


def recent_tone_repair(max_age_secs: Optional[float] = None) -> bool:
    if max_age_secs is None:
        max_age_secs = 180.0
    with _lock:
        last = _last_tone_repair_at
    return last > 0.0 and (time.monotonic() - last) <= max_age_secs


def build_prompt(repair: dict) -> str:
    kind = repair.get("kind") or "repair"
    correction = repair.get("correction") or ""
    last_assistant = repair.get("last_assistant_text") or ""
    user_text = repair.get("user_text") or ""

    kind_rule = {
        "misheard": (
            "Rex likely misheard or used the wrong words. Own that briefly. "
            "If the human supplied the corrected words, use them exactly once."
        ),
        "misunderstood": (
            "Rex misunderstood the intent. Own the miss, restate the corrected "
            "understanding if possible, and do not defend the old answer."
        ),
        "tone": (
            "Rex's tone landed badly. Drop roasts completely. Give a concise, "
            "sincere repair and switch to warmer footing."
        ),
        "wrong_person": (
            "Rex attributed speech, identity, or intent to the wrong person. "
            "Correct the referent, do not argue, and do not keep addressing the "
            "wrong person."
        ),
        "pronoun": (
            "Rex used or implied the wrong pronouns. Accept the correction, use "
            "the corrected pronouns if supplied, and do not make the correction "
            "into a joke."
        ),
        "factual": (
            "Rex asserted something unsupported or false. Own the overreach, "
            "remove the invented detail, and continue from only what is known."
        ),
        "pacing": (
            "Rex is asking too much or moving too fast. Back off, give space, "
            "and do not ask a new question."
        ),
        "interruption": (
            "Rex interrupted or talked over the human. Apologize briefly and "
            "explicitly give the floor back."
        ),
        "repeat": (
            "The human did not hear Rex. Repeat or paraphrase Rex's last line "
            "briefly and more plainly. Do not add anything new."
        ),
        "clarify": (
            "The human needs Rex to clarify. Explain the previous line plainly, "
            "without adding a new topic or another question."
        ),
        "bare_negation": (
            "The human rejected the previous move. Acknowledge it and make a "
            "small clarifying repair."
        ),
    }.get(kind, "Make a concise conversational repair.")

    correction_clause = (
        f"\nCorrected detail supplied by the human: {correction!r}."
        if correction else ""
    )
    target = repair.get("target") or ""
    target_clause = f"\nLikely corrected referent/target: {target!r}." if target else ""
    last_clause = (
        f"\nRex's immediately previous line was: {last_assistant!r}."
        if last_assistant else ""
    )
    return (
        "The human is correcting or repairing the conversation with Rex.\n"
        f"Repair type: {kind}.\n"
        f"Rule: {kind_rule}\n"
        f"Human said: {user_text!r}."
        f"{correction_clause}"
        f"{target_clause}"
        f"{last_clause}\n\n"
        "Write ONE short in-character Rex reply. Requirements: acknowledge the "
        "miss without groveling, do not roast the human, do not add a new topic, "
        "do not punish the human for correcting you, and do not ask a question "
        "unless the repair cannot continue without a single clarification. If a "
        "correction was supplied, accept it. Do not begin with 'Rex:' or any "
        "speaker label. For misheard, misunderstood, wrong-person, pronoun, "
        "factual, or bare-negation repairs, include this exact recovery line: "
        f"{BETTER_LUCK_NEXT_TIME!r}"
    )


def fallback_response(repair: dict) -> str:
    kind = repair.get("kind") or "repair"
    correction = repair.get("correction") or ""
    if kind == "tone":
        return "Yeah, that landed wrong. I'll pull the claws in."
    if kind == "wrong_person":
        return "Got it. Wrong person, wrong circuit. I'll correct course."
    if kind == "pronoun":
        return "Got it. I'll use the right pronouns."
    if kind == "factual":
        return "You're right. I overreached there; I'll stick to what I actually know."
    if kind == "pacing":
        return "Fair. I'll stop interrogating the room and give you a second."
    if kind == "interruption":
        return "You're right. I cut in. Go ahead, I'm listening."
    if kind == "repeat":
        last = repair.get("last_assistant_text") or ""
        if last:
            return f"I said: {last}"
        return "I missed my own playback there. Nothing vital, mercifully."
    if kind == "clarify":
        return "Fair. I made that murkier than needed; let me simplify."
    if correction:
        return f"Got it. I heard that wrong: {correction}."
    return "Got it. I missed that one; let me reset."


def should_use_better_luck_line(repair: dict) -> bool:
    kind = repair.get("kind") or "repair"
    return kind in _BETTER_LUCK_REPAIR_KINDS


def add_better_luck_line(text: str) -> str:
    response = (text or "").strip()
    if not response:
        return BETTER_LUCK_NEXT_TIME
    if BETTER_LUCK_NEXT_TIME.lower() in response.lower():
        return response
    if response[-1] not in ".!?":
        response += "."
    return f"{response} {BETTER_LUCK_NEXT_TIME}"


def _extract_correction(text: str) -> str:
    match = _CORRECTION_PAT.search(text)
    if not match:
        match = _NOT_X_Y_PAT.search(text)
        if not match:
            return ""
        correction = match.group(2).strip()
    else:
        correction = match.group(1).strip()
    correction = re.sub(r"^(?:it'?s|it is|it was|should be)\s+", "", correction, flags=re.I)
    correction = re.sub(r"\s+", " ", correction)
    return correction[:160]


def _extract_target(text: str) -> str:
    patterns = [
        r"\byou mean\s+([^.!?]+)",
        r"\bthat was\s+([^.!?]+)",
        r"\byou'?re talking to\s+([^.!?]+)",
        r"\bnot me,?\s+([^.!?]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            target = re.sub(r"\s+", " ", match.group(1)).strip(" .!?\"'")
            if target:
                return target[:80]
    return ""
