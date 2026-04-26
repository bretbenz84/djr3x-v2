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
    r"no,? not that|no,? that's wrong|no,? that is wrong|incorrect)\b",
    re.IGNORECASE,
)
_TONE_PAT = re.compile(
    r"\b(that was (rude|mean|harsh|uncalled for|distasteful|not funny|too much)|"
    r"that wasn't funny|that wasnt funny|you were (rude|mean|harsh)|"
    r"don'?t roast|do not roast|stop roasting|not a joke|not funny)\b",
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
_BARE_NEGATION_PAT = re.compile(r"^\s*(no|nope|nah|wrong|incorrect)\s*[.!]?\s*$", re.I)
_CORRECTION_PAT = re.compile(
    r"\b(?:i said|it's|it is|his name was|her name was|their name was|"
    r"that's not his name,? it'?s|that's not her name,? it'?s|"
    r"that's not their name,? it'?s|that's not my name,? it'?s|"
    r"the name was|i meant)\s+(.+)$",
    re.IGNORECASE,
)

_lock = threading.Lock()
_last_assistant_text: str = ""
_last_repair_at: float = 0.0


@dataclass
class RepairMove:
    kind: str
    severity: str
    user_text: str
    correction: str = ""
    last_assistant_text: str = ""
    detected_at: float = 0.0


def clear() -> None:
    global _last_assistant_text, _last_repair_at
    with _lock:
        _last_assistant_text = ""
        _last_repair_at = 0.0


def note_assistant_turn(text: str) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return
    global _last_assistant_text
    with _lock:
        _last_assistant_text = cleaned


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
    elif _MISHEARD_PAT.search(cleaned):
        kind = "misheard"
        severity = "medium"
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
        last_repair_at = _last_repair_at
    if kind == "bare_negation" and "?" not in last_assistant:
        return None
    if kind == "bare_negation" and now - last_repair_at < 10.0:
        return None

    correction = _extract_correction(cleaned)
    if not correction and kind in {"misheard", "misunderstood"}:
        # Preserve the useful part in common forms like "no, Tom Foster".
        correction = re.sub(r"^\s*(no|nope|nah|wrong|incorrect)[,\s]+", "", cleaned, flags=re.I).strip()
        if correction.lower() == lowered or len(correction.split()) > 12:
            correction = ""

    move = RepairMove(
        kind=kind,
        severity=severity,
        user_text=cleaned,
        correction=correction.strip(" .!?\"'"),
        last_assistant_text=last_assistant,
        detected_at=now,
    )
    return asdict(move)


def mark_handled() -> None:
    global _last_repair_at
    with _lock:
        _last_repair_at = time.monotonic()


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
        "pacing": (
            "Rex is asking too much or moving too fast. Back off, give space, "
            "and do not ask a new question."
        ),
        "interruption": (
            "Rex interrupted or talked over the human. Apologize briefly and "
            "explicitly give the floor back."
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
        f"{last_clause}\n\n"
        "Write ONE short in-character Rex reply. Requirements: acknowledge the "
        "miss without groveling, do not roast the human, do not add a new topic, "
        "and do not ask a question unless the repair cannot continue without a "
        "single clarification. If a correction was supplied, accept it."
    )


def fallback_response(repair: dict) -> str:
    kind = repair.get("kind") or "repair"
    correction = repair.get("correction") or ""
    if kind == "tone":
        return "Yeah, that landed wrong. I'll pull the claws in."
    if kind == "pacing":
        return "Fair. I'll stop interrogating the room and give you a second."
    if kind == "interruption":
        return "You're right. I cut in. Go ahead, I'm listening."
    if correction:
        return f"Got it. I heard that wrong: {correction}."
    return "Got it. I missed that one; let me reset."


def _extract_correction(text: str) -> str:
    match = _CORRECTION_PAT.search(text)
    if not match:
        return ""
    correction = match.group(1).strip()
    correction = re.sub(r"\s+", " ", correction)
    return correction[:160]
