"""
intelligence/repair_moves.py - explicit conversational repair handling.

When the human says Rex misheard, misunderstood, pushed too hard, or landed a
line badly, this module turns that into a small repair move instead of letting
the normal roast/curiosity machinery treat it like ordinary banter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import re
import threading
import time
from typing import Optional


_MISHEARD_PAT = re.compile(
    r"\b(?:"
    r"you (?:misheard|heard wrong|heard me wrong)|you misheard me|"
    r"you (?:didn'?t|did not) (?:hear|catch|get) (?:me|that|it|what i said)|"
    r"i (?:didn'?t say|did not say)|"
    r"that'?s not what i said|that is not what i said|not what i said|"
    r"that'?s not (?:his|her|their|my) name|that is not (?:his|her|their|my) name|"
    r"(?:his|her|their|the) name was|"
    r"wrong word|wrong name|got my words wrong|transcribed (?:it )?wrong|mis-?transcribed"
    r")\b"
    # Bare "i said" means a correction ("no, I said blues") — but NOT when it's an
    # emphasis/reference lead-in ("like/as I said") or a recall question ("what I said",
    # already guarded in detect()). Exclude those so they don't fire a misheard repair.
    r"|(?<!like )(?<!as )(?<!what )(?<!that )\bi said\b",
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
    r"i wasn'?t done|i was still talking|"
    r"didn'?t (?:give me (?:any |enough )?time|let me) (?:to )?"
    r"(?:answer|respond|finish|reply|speak|talk|think)|"
    r"wouldn'?t let me (?:answer|respond|finish|reply|speak|talk))\b",
    re.IGNORECASE,
)
# The proper-name alternatives are matched CASE-SENSITIVELY via a scoped
# (?-i:...) group so IGNORECASE does not let [A-Z][a-z]+ match any lowercase
# word — that bug made "you mean my telescope" parse as a person correction and
# drove Rex to defend the bad guess ("Your telescope, not mine") instead of
# dropping it. An object/detail correction now falls through to grounding/factual.
_WRONG_PERSON_PAT = re.compile(
    r"\b(wrong person|not me|that wasn'?t me|that was not me|"
    r"you mean (him|her|them|(?-i:[A-Z][a-z'\-]+))|"
    r"you'?re talking to (him|her|them|the wrong person)|"
    r"that was (him|her|them|(?-i:[A-Z][a-z'\-]+)))\b",
    re.IGNORECASE,
)
# "Rex invented/assumed a detail" corrections + the natural "that makes no
# sense" rejection. Routed to the existing 'factual' frame (remove the invented
# detail, do not defend it). Gated on Rex having spoken recently so a bare "makes
# no sense" about a third-party topic is not swept up.
_GROUNDING_CORRECTION_PAT = re.compile(
    r"\b((that|this|it) (makes no sense|doesn'?t make (any |much )?sense|"
    r"made no sense|makes zero sense)|"
    r"none of (that|this) (makes sense|is right)|"
    # second-person confusion: the human says REX is the one not making sense
    # (field bug: "You're not saying anything that makes sense" got a roast
    # comeback instead of a humble reset).
    r"you'?re not making (any )?sense|"
    r"(you'?re )?not saying anything that makes sense|"
    r"nothing you('?re| are) saying makes (any )?sense|"
    r"you'?re (just )?making (that|this|it|stuff) up|"
    r"you (just )?(invented|assumed|made (?:that|this|it|the|a)? ?up|made up)\b|"
    r"i never said (that|this)|i didn'?t mention|where are you getting (that|this))\b",
    re.IGNORECASE,
)
# "(No,) I didn't say that" on its own disagrees with what Rex just CLAIMED — it
# is not an identity correction ("wrong person") or a transcription fix. Used to
# misfire as wrong_person → "faulty coordinate… better luck next time", derailing
# the conversation. Let a bare denial flow to normal conversational handling; a
# real correction ("I didn't say jazz, I said blues") carries more and still
# routes through the misheard/correction path below.
_BARE_CONTENT_DENIAL_PAT = re.compile(
    r"^\s*(?:no[,!.]?\s*|nope[,!.]?\s*|nah[,!.]?\s*)?"
    r"i\s+(?:didn'?t|did not)\s+say\s+that\b[.!]?\s*$",
    re.IGNORECASE,
)
# Asking Rex to RECALL or confirm he FOLLOWED what the user said ("do you recall what I
# said?", "did you (not) follow what I said?", "what did I say?") is a comprehension /
# recall request, NOT a misheard-correction. The bare "i said" alternative in _MISHEARD_PAT
# used to swallow these and fire a canned recovery line ("Consider it logged. Onward.")
# instead of letting Rex actually recall from the transcript. Hand them to normal
# conversation so the reply path (which has the recent transcript) can answer.
_RECALL_REQUEST_PAT = re.compile(
    r"\b(?:"
    r"(?:do|did|don'?t|can|could|would|are)\s+you\s+"
    r"(?:even\s+|not\s+|actually\s+|really\s+)*"
    r"(?:recall|remember|follow|catch|hear|get|register|understand|know|see)\s+"
    r"(?:what|when|why|how|that)\s+i\s+(?:said|asked|told|meant|answered)"
    r"|what\s+did\s+i\s+(?:say|ask|answer|just\s+say|tell\s+you)"
    r"|(?:repeat\s+back|recall|remind\s+me)\s+(?:of\s+)?(?:what\s+)?i\s+(?:said|asked|answered)"
    r")",
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

# A BARE restatement: the user simply repeats their real turn with an "I said / I meant …"
# lead-in and NO contrast ("I said I watch a lot of Netflix specials"). That is the user's
# actual turn after a mishear — it should be RESPONDED to, not echoed back as a "correction"
# (the "We'll get there — recalibrating. <your words>." field bug). A CONTRASTIVE correction
# ("I said blues, not jazz") is NOT bare and still flows through the repair-ack path.
_BARE_RESTATEMENT_LEAD_PAT = re.compile(
    r"^\s*(?:um|uh|well|okay|ok|so|yeah|yes)?,?\s*i (?:said|meant)\b",
    re.IGNORECASE,
)
_RESTATEMENT_CONTRAST_PAT = re.compile(
    r"\bnot\b|\bno\b|\bnope\b|\bwrong\b|\bincorrect\b|isn'?t|wasn'?t|aren'?t|"
    r"weren'?t|instead of|rather than|that'?s not|you (?:said|heard|got|thought)",
    re.IGNORECASE,
)


def is_bare_restatement(text: str) -> bool:
    """True when the user RE-STATES content with an 'I said/I meant …' lead-in and NO
    contrast — i.e. repeating their real turn after a mishear, which should be answered, not
    echoed. A contrastive correction ('I said blues, not jazz') returns False (repair path)."""
    t = (text or "").strip()
    if not t or not _BARE_RESTATEMENT_LEAD_PAT.search(t):
        return False
    return not _RESTATEMENT_CONTRAST_PAT.search(t)


_lock = threading.Lock()
_last_assistant_text: str = ""
_last_assistant_at: float = 0.0
_last_repair_at: float = 0.0
_last_tone_repair_at: float = 0.0

# Recovery tags appended to "I misheard you" / bare-negation repairs. A single
# fixed line got repeated on back-to-back repairs and a bystander noticed ("Is it
# just gonna keep asking the same question?"). Rotate with anti-repeat so two
# consecutive repairs never use the same tag.
_RECOVERY_LINES = [
    "I'm sure we'll have better luck next time!",
    "We'll get there — recalibrating.",
    "Noted. I'll route around that one.",
    "Fair enough — let me reset and try that again.",
    "Consider it logged. Onward.",
    "My circuits and I will do better on the next pass.",
]
BETTER_LUCK_NEXT_TIME = _RECOVERY_LINES[0]  # back-compat alias
_last_recovery_line: str = ""


def pick_recovery_line() -> str:
    """A recovery tag that differs from the one used on the previous repair."""
    global _last_recovery_line
    with _lock:
        choices = [
            line for line in _RECOVERY_LINES if line != _last_recovery_line
        ] or list(_RECOVERY_LINES)
        line = random.choice(choices)
        _last_recovery_line = line
        return line


# ── Misheard / misunderstood recovery (no correction supplied) ──────────────────
# When the human flags a mishearing but hasn't re-said it yet, Rex saves face with a
# quick self-deprecating "my circuits glitched" joke, then hands the floor back so they
# can repeat it. Both halves rotate with anti-repeat so back-to-back corrections never
# read identically (a bystander once noticed Rex repeating the same repair line).
_SAVE_FACE_LINES = [
    "Ah, my audio processor fumbled that one.",
    "Circuits crossed — that decoded as pure static.",
    "My transcription unit clearly skipped its calibration.",
    "Static in the receptors; that came through garbled.",
    "I'm sure we'll have better luck next time!",   # Star Tours sign-off, keep it
    "One of my logic boards took an unscheduled coffee break.",
    "My ears run on 90% guesswork and 10% optimism, apparently.",
    "Bad packet on my end — that arrived as nonsense.",
    "I misinterpret life forms sometimes; occupational hazard.",
]
_REPROMPT_LINES = [
    "OK, shoot — what'd you say again?",
    "Run that by me one more time?",
    "Go on, say it again — I'm listening properly now.",
    "Give it to me again?",
    "Once more, for the droid in the back?",
    "Hit me with it again and I'll get it right.",
    "Say that again?",
]
_last_save_face: str = ""
_last_reprompt: str = ""


def _pick_distinct(pool: list[str], last: str) -> str:
    choices = [s for s in pool if s != last] or list(pool)
    return random.choice(choices)


# Words that are part of the correction SIGNAL, not re-said content. A captured
# "correction" made up ENTIRELY of these ("I said", "you misunderstood me") means the
# human flagged a miss without supplying new words — so Rex should ask them to repeat,
# not "accept" a phantom correction.
_CORRECTION_SIGNAL_WORDS = {
    "i", "you", "me", "no", "nope", "nah", "not", "what", "that", "this", "it",
    "said", "say", "saying", "mean", "meant", "didn", "didnt", "don", "dont",
    "is", "was", "misunderstood", "misheard", "heard", "hear", "wrong",
    "the", "a", "an", "my", "your",
}


def correction_has_content(correction: Optional[str]) -> bool:
    """True if `correction` carries actual re-said content ('blues not jazz', 'Tom
    Foster'), not just an echo of the correction signal ('I said', 'you misunderstood
    me'). Lets the repair path tell 'here's the fix' from a bare 'you got it wrong'."""
    words = re.findall(r"[a-z']+", (correction or "").lower())
    return any(w.strip("'") not in _CORRECTION_SIGNAL_WORDS for w in words if w.strip("'"))


def misheard_recovery_response() -> str:
    """Two beats for an ASR/comprehension miss with NO correction supplied: a short
    save-face circuit-glitch joke + a varied invitation to repeat. Both rotate with
    anti-repeat. Used instead of the LLM 'own it and move on' path so Rex actually hands
    the floor back when the human hasn't re-said the thing yet."""
    global _last_save_face, _last_reprompt
    with _lock:
        joke = _pick_distinct(_SAVE_FACE_LINES, _last_save_face)
        _last_save_face = joke
        ask = _pick_distinct(_REPROMPT_LINES, _last_reprompt)
        _last_reprompt = ask
    return f"{joke} {ask}"


def _norm_apostrophes(s: str) -> str:
    """Fold curly/modifier apostrophes to a straight ' so a substring check survives the
    LLM rendering a recovery line with U+2019 while the constant uses U+0027. Without this,
    the dedup guard misses and add_better_luck_line() appends the recovery line a 2nd time
    (the 'We'll get there — recalibrating. … We'll get there — recalibrating.' field bug)."""
    return (s or "").lower().replace("’", "'").replace("ʼ", "'").replace("‘", "'")


def _contains_recovery_line(text: str) -> bool:
    low = _norm_apostrophes(text)
    return any(_norm_apostrophes(line) in low for line in _RECOVERY_LINES)


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
    requires_recent = False

    # A bare "I didn't say that" is a content disagreement, not a repair — hand
    # it to normal conversation so Rex can engage instead of derailing.
    if _BARE_CONTENT_DENIAL_PAT.match(cleaned):
        return None

    # "Do you recall what I said?" / "did you not follow what I said?" / "what did I say?"
    # are recall/comprehension requests, NOT corrections — let the reply path recall from
    # the transcript instead of firing a canned recovery line (see _RECALL_REQUEST_PAT).
    if _RECALL_REQUEST_PAT.search(cleaned):
        return None

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
    elif _GROUNDING_CORRECTION_PAT.search(cleaned):
        # "That makes no sense" / "you invented that" — Rex guessed/invented a
        # detail. Route to the factual frame (drop the invented thread). Gated on
        # Rex having spoken recently.
        kind = "factual"
        severity = "high"
        requires_recent = True
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
    if (kind in {"repeat", "clarify", "bare_negation"} or requires_recent) and not recent_assistant:
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
            "understanding if possible, and do not defend the old answer. Do NOT "
            "re-explain or justify how you arrived at the wrong read, and do NOT "
            "re-ask the question you built on it."
        ),
        "tone": (
            "Rex's tone landed badly. Drop roasts completely. Give a concise, "
            "sincere repair and switch to warmer footing."
        ),
        "wrong_person": (
            "Rex attributed speech, identity, or intent to the wrong person. "
            "Correct the referent, do not argue, and do not keep addressing the "
            "wrong person. Do NOT re-explain your reasoning or re-ask the question "
            "built on the wrong referent."
        ),
        "pronoun": (
            "Rex used or implied the wrong pronouns. Accept the correction, use "
            "the corrected pronouns if supplied, and do not make the correction "
            "into a joke."
        ),
        "factual": (
            "Rex asserted, guessed, or invented something the human is now "
            "correcting. Own the overreach in ONE short beat, remove the invented "
            "detail, and drop that thread entirely. Do NOT re-explain or defend "
            "how you arrived at the wrong detail, do NOT restate your reasoning, "
            "and do NOT re-ask the question you built on the invented detail. One "
            "acknowledgement, then move on from only what is actually known."
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
    # Pick the recovery tag once and stash it on the repair so the post-generation
    # add_better_luck_line() appends the SAME line the prompt asked for (no mismatch,
    # no double tag), while consecutive repairs still vary.
    recovery_line = repair.get("recovery_line") or pick_recovery_line()
    repair["recovery_line"] = recovery_line
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
        "unless the repair cannot continue without a single clarification. Never "
        "re-state or justify the reasoning behind the thing you got wrong — "
        "acknowledging the miss and dropping it is the entire move. If a "
        "correction was supplied, accept it. Do not begin with 'Rex:' or any "
        "speaker label. For misheard, misunderstood, wrong-person, pronoun, "
        "factual, or bare-negation repairs, include this exact recovery line: "
        f"{recovery_line!r}"
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


def add_better_luck_line(text: str, line: Optional[str] = None) -> str:
    line = line or pick_recovery_line()
    response = (text or "").strip()
    if not response:
        return line
    if _contains_recovery_line(response):
        return response
    if response[-1] not in ".!?":
        response += "."
    return f"{response} {line}"


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
