"""
intelligence/onboarding.py — first-meeting baseline-gathering for NEW people.

When Rex meets someone brand new, the normal question budget (3/90s) plus the
stranger depth-1 tier lock leave him barely able to ask anything, so he learns
nothing about the person he is actively talking to. This module backs a SCOPED,
stranger-only "onboarding" burst that runs right after enrollment: Rex walks a
short research-backed ladder of baseline questions (de-trapped "what do you do",
connection-to-the-room, hometown, a passion or two, one earned follow-up),
reacts to each answer with a brief retort, occasionally reveals a sliver about
himself, writes a real baseline to memory, and exits the moment momentum dies.

This module owns the *pure* pieces — eligibility, question selection, retort /
reveal / closer lines, answer sentiment + exit detection, the LLM depth
follow-up, value tidying, and the memory writes. The multi-turn flow STATE
itself lives in intelligence/interaction.py (_pending_onboarding), exactly like
the introduction and "tell me about someone" flows.

The burst rides the question-budget urgent bypass ("newcomer_baseline") so it
never loosens the deliberately-tight friend-protecting global cap; its own
ONBOARDING_MIN/MAX_QUESTIONS bound it instead. Master flag: ONBOARDING_ENABLED.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

import config
from intelligence import comedy_modes
from intelligence import profile_questions
from memory import facts as facts_memory
from memory import interests as interests_memory
from memory import people as people_memory
from memory import relationships as rel_memory

_log = logging.getLogger(__name__)

_client = None


def _openai_client():
    """Lazy OpenAI client for the (off-by-default) authored-question rephrase.
    The depth follow-up itself goes through llm.generate_curiosity_question."""
    global _client
    if _client is None:
        import apikeys
        from openai import OpenAI
        _client = OpenAI(api_key=apikeys.OPENAI_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Flags / eligibility
# ─────────────────────────────────────────────────────────────────────────────

def enabled() -> bool:
    return bool(getattr(config, "ONBOARDING_ENABLED", False))


def min_questions() -> int:
    return max(1, int(getattr(config, "ONBOARDING_MIN_QUESTIONS", 4)))


def max_questions() -> int:
    return max(min_questions(), int(getattr(config, "ONBOARDING_MAX_QUESTIONS", 8)))


def eligible(person_id: Optional[int], *, person: Optional[dict] = None) -> bool:
    """True when Rex should run the onboarding burst for this person.

    Brand-new only: low visit count, a near-empty profile, never a minor. The
    minor gate is shared with the normal profile-question path so a child is
    never interviewed here either.
    """
    if not enabled() or person_id is None:
        return False
    try:
        if person is None:
            person = people_memory.get_person(person_id)
        if profile_questions.person_is_minor(person_id, person=person):
            return False
        visit_count = int((person or {}).get("visit_count") or 0)
        if visit_count > int(getattr(config, "ONBOARDING_MAX_VISITS", 1)):
            return False
        if profile_questions.profile_fact_count(person_id) > int(
            getattr(config, "ONBOARDING_FACT_FLOOR", 3)
        ):
            return False
    except Exception as exc:
        _log.debug("[onboarding] eligibility check failed for person_id=%s: %s", person_id, exc)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Question selection
# ─────────────────────────────────────────────────────────────────────────────

def _known_fact_keys(person_id: int) -> set[str]:
    """Every fact KEY this person already has.

    Skip is by exact key (never by category): if Rex already knows their `job`
    he should not ask the job question, but a stray enrollment fact in the same
    category (gender/hair_color under 'identity'/'appearance') must NOT suppress
    the whole Tier-A baseline — and those keys never collide with question keys.
    """
    keys: set[str] = set()
    try:
        for fact in facts_memory.get_facts(person_id):
            key = str(fact.get("key") or "").strip()
            if key:
                keys.add(key)
    except Exception as exc:
        _log.debug("[onboarding] known-fact index failed: %s", exc)
    return keys


def next_question(
    person_id: int,
    *,
    asked_keys: Optional[set[str]] = None,
    last_answer: Optional[str] = None,
    last_question: Optional[str] = None,
    allow_depth: bool = True,
) -> Optional[dict]:
    """Return the next eligible baseline question as a resolved dict, or None.

    Walks ONBOARDING_QUESTION_POOL in tier order (A -> B -> C), skipping
    questions already asked this burst, already answered in a prior session,
    already covered by a known fact, or blocked by a boundary. Tier-C (earned
    depth) is offered only when allow_depth (real momentum). The text=None
    follow-up entry is generated against last_answer and skipped if that fails.
    """
    asked = set(asked_keys or set())
    try:
        answered = rel_memory.get_answered_question_keys(person_id)
    except Exception:
        answered = set()
    known = _known_fact_keys(person_id)
    skip = asked | set(answered) | known

    for entry in getattr(config, "ONBOARDING_QUESTION_POOL", []) or []:
        key = entry.get("key")
        if not key or key in skip:
            continue
        if entry.get("tier") == "C" and not allow_depth:
            continue
        try:
            if profile_questions.question_blocked_by_boundary(person_id, entry):
                continue
        except Exception:
            pass
        text = _resolve_text(entry, last_answer, last_question, person_id)
        if not text:
            continue
        resolved = dict(entry)
        resolved["text"] = text
        return resolved
    return None


def _resolve_text(
    entry: dict,
    last_answer: Optional[str],
    last_question: Optional[str],
    person_id: Optional[int],
) -> Optional[str]:
    text = entry.get("text")
    if text is None:
        # LLM-generated depth follow-up — needs a prior answer to dig into.
        if not (last_answer or "").strip():
            return None
        return generate_followup(last_answer, person_id=person_id, prev_question=last_question)
    if bool(getattr(config, "ONBOARDING_LLM_REPHRASE_ENABLED", False)):
        return _maybe_rephrase(str(text))
    return str(text)


# ─────────────────────────────────────────────────────────────────────────────
# LLM depth follow-up (local qwen sidecar; templated fallback)
# ─────────────────────────────────────────────────────────────────────────────

def generate_followup(
    prev_answer: Optional[str],
    *,
    person_id: Optional[int] = None,
    prev_question: Optional[str] = None,
) -> Optional[str]:
    """One short, curious follow-up question that digs into the last answer.

    Quality-critical and in-character, so it runs on the main OpenAI model via
    llm.generate_curiosity_question (the same brain as the rest of Rex's
    conversation, with built-in grief/heavy-topic restraint and known-interest
    awareness) — NOT the local qwen classifier sidecar. When the LLM follow-up
    is disabled (offline / tests) it falls back to a validated
    "how'd you get into <topic>?" template. Returns None when there is nothing
    safe/useful to ask, so selection falls through to an authored Tier-C question.
    """
    answer = (prev_answer or "").strip()
    if not answer:
        return None

    if bool(getattr(config, "ONBOARDING_LLM_FOLLOWUP_ENABLED", True)):
        try:
            from intelligence import llm

            out = llm.generate_curiosity_question(
                prev_question or "So, tell me about yourself.",
                answer,
                person_id=person_id,
            )
            # An empty return is deliberate (e.g. a heavy/sensitive answer) —
            # do NOT template over it; skip the depth probe entirely.
            return _first_question(out) or None
        except Exception as exc:
            _log.debug("[onboarding] OpenAI follow-up failed, using template: %s", exc)

    topic = _topic_from_answer(answer)
    # Only build the template when the topic reads like a thing you can "get into"
    # (a short noun-ish phrase). A vague answer ("it's going great") would make
    # "How'd you get into going great?" — better to skip and let selection fall
    # through to an authored Tier-C question.
    return f"How'd you get into {topic}?" if _looks_like_topic(topic) else None


def _maybe_rephrase(text: str) -> str:
    """Optional cosmetic rephrase of an authored question in Rex's voice
    (OpenAI, off by default). Falls back to the verbatim authored question."""
    base = (text or "").strip()
    if not base:
        return base
    try:
        resp = _openai_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are Rex, a witty droid. Output ONLY the rephrased question."},
                {"role": "user",
                 "content": f"Rephrase in a casual, witty droid voice, same meaning, "
                            f"one question, max 16 words: \"{base}\""},
            ],
            temperature=0.5,
            max_tokens=40,
        )
        rephrased = _first_question((resp.choices[0].message.content or ""))
        if rephrased:
            return rephrased
    except Exception as exc:
        _log.debug("[onboarding] rephrase failed, using verbatim: %s", exc)
    return base


def _first_question(text: str) -> str:
    """Pull a single clean question out of an LLM completion; '' if unusable."""
    cleaned = (text or "").strip().strip('"').strip()
    if not cleaned:
        return ""
    match = re.search(r"[^?\n]*\?", cleaned)
    candidate = (match.group(0) if match else cleaned).strip().strip('"').strip()
    # Drop any leading preamble sentence ("Sure! How long...?" -> "How long...?").
    candidate = re.split(r"(?<=[.!])\s+", candidate)[-1].strip().strip('"').strip()
    if not candidate:
        return ""
    if not candidate.endswith("?"):
        candidate += "?"
    # Reject runaway / multi-sentence output — fall back to the template instead.
    if len(candidate.split()) > 18:
        return ""
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Answer sentiment -> retort, and self-reveal / closer lines
# ─────────────────────────────────────────────────────────────────────────────

_WARM_PAT = re.compile(
    r"\b(?:because|honestly|truly|means? a lot|the thing is|i feel|it feels|"
    r"my (?:family|kids?|wife|husband|partner|mom|dad)|lost|passed away|grateful|"
    r"proud of|dream)\b",
    re.IGNORECASE,
)
_POSITIVE_PAT = re.compile(
    r"\b(?:love|loved|awesome|amazing|favou?rite|obsessed|can'?t get enough|"
    r"best|incredible|so much fun|passionate|my jam)\b|!",
    re.IGNORECASE,
)
_SURPRISE_PAT = re.compile(
    r"\b(?:actually|believe it or not|you'?d never guess|never|always|once|"
    r"twenty years|ten years|\d+\s+years|literally|funny enough)\b",
    re.IGNORECASE,
)


def classify_answer(text: str) -> str:
    """Map a user answer to a retort sentiment: warm | surprise | positive |
    flat | neutral. Warm wins (sincerity), then surprise, then positive."""
    cleaned = (text or "").strip()
    if is_soft_disengage(cleaned):
        return "flat"
    if _WARM_PAT.search(cleaned):
        return "warm"
    if _SURPRISE_PAT.search(cleaned):
        return "surprise"
    if _POSITIVE_PAT.search(cleaned):
        return "positive"
    return "neutral"


_RETORT_BANK_BY_SENTIMENT = {
    "warm": "onboarding_retort_warm",
    "surprise": "onboarding_retort_surprise",
    "positive": "onboarding_retort_positive",
    "neutral": "onboarding_retort_neutral",
    "flat": "onboarding_retort_neutral",
}


def retort_for(answer_text: str) -> str:
    """A short (2-5 word) in-character acknowledgment of the last answer."""
    sentiment = classify_answer(answer_text)
    bank = _RETORT_BANK_BY_SENTIMENT.get(sentiment, "onboarding_retort_neutral")
    line = comedy_modes.line_for(bank)
    return line or "Noted."


def reveal_line() -> str:
    lines = list(getattr(config, "ONBOARDING_REVEAL_LINES", []) or [])
    return random.choice(lines) if lines else ""


def closer_line(answered_count: int) -> str:
    lines = list(getattr(config, "ONBOARDING_CLOSERS", []) or [])
    if answered_count <= 0:
        return "Tough crowd. I'll fill in the blanks later."
    return random.choice(lines) if lines else "Filed. Good to meet you."


def backoff_line() -> str:
    lines = list(getattr(config, "ONBOARDING_BACKOFF_LINES", []) or [])
    return random.choice(lines) if lines else "Fair — I'll ease off the questions."


# ─────────────────────────────────────────────────────────────────────────────
# Exit / disengagement detection
# ─────────────────────────────────────────────────────────────────────────────

_DECLINE_PAT = re.compile(
    r"\b(?:rather not|don'?t want to|do not want to|stop asking|don'?t ask|"
    r"do not ask|no more questions|enough questions|quit asking|quit it|"
    r"change the subject|new subject|drop it|leave it|none of your business|"
    r"not telling you|stop interviewing|stop the quiz)\b",
    re.IGNORECASE,
)
_PIVOT_PAT = re.compile(
    r"\b(?:can|could|would|will)\s+you\b|"
    r"\b(?:play|put on|skip|pause|stop the music|what'?s the weather|what time|"
    r"set a timer|remind me|tell me a joke|let'?s play|start a game)\b",
    re.IGNORECASE,
)
_DUNNO_PAT = re.compile(
    r"^\s*(?:um+|uh+|hmm+|well)?[\s,]*"
    r"(?:i\s+(?:don'?t|do\s+not)\s+know|not\s+sure|dunno|no\s+idea|nothing|"
    r"can'?t\s+think|pass|skip|meh|whatever|i\s+guess)\b",
    re.IGNORECASE,
)
_SECOND_PERSON_PAT = re.compile(r"\byou\b|\byour\b", re.IGNORECASE)


def is_hard_decline(text: str) -> bool:
    """An explicit 'stop asking' / boundary-shaped turn — exit immediately."""
    return bool(_DECLINE_PAT.search(text or ""))


def is_pivot(text: str) -> bool:
    """A request/command to Rex, or a question turned back on Rex ('what about
    you?') — the burst should yield and release the turn to normal routing."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _PIVOT_PAT.search(cleaned):
        return True
    # A question aimed at Rex ("?" + a second-person reference) is a pivot:
    # Rex should just answer it, not plow into the next baseline question.
    if cleaned.endswith("?") and _SECOND_PERSON_PAT.search(cleaned):
        return True
    return False


def is_soft_disengage(text: str) -> bool:
    """Lukewarm/empty answer (short or 'I don't know') — does NOT abort before
    the MIN floor, but accrues toward the wind-down counter."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if _DUNNO_PAT.match(cleaned):
        return True
    words = re.findall(r"[A-Za-z']+", cleaned)
    return len(words) < 2


# ─────────────────────────────────────────────────────────────────────────────
# Answer -> memory
# ─────────────────────────────────────────────────────────────────────────────

_FILLER_PREFIX = re.compile(
    r"^(?:um+|uh+|hmm+|well|so|like|okay|ok|yeah|yep|oh|i mean|honestly|actually|"
    r"probably|i guess|i think|it'?s|it is|i'?m|i am|my|mostly|currently|"
    r"right now|these days|usually|kind of|kinda|sort of)[\s,]+",
    re.IGNORECASE,
)
_FACT_LEAD = re.compile(
    r"^(?:a|an|the|i work (?:as|at|in|for)|i'?m a|i'?m an|i am a|i am an)[\s,]+",
    re.IGNORECASE,
)
_INTEREST_LEAD = re.compile(
    r"^(?:i (?:like|love|enjoy|listen to|am into|'?m into)|into|mostly|a lot of|lots of)[\s,]+",
    re.IGNORECASE,
)
# A tidied value that still STARTS with one of these isn't a clean noun topic
# ("going great...", "it's complicated", "nothing really") — drop it rather than
# file junk. The structured person_qa row still records the full answer.
_BAD_VALUE_LEAD = {
    "it", "its", "i", "we", "they", "he", "she", "just", "really", "very",
    "kind", "sort", "not", "no", "nothing", "maybe", "probably", "going",
    "doing", "getting", "having", "idk", "dunno", "stuff", "things", "whatever",
}


def tidy_value(answer: str, store: str) -> str:
    """Heuristically reduce a spoken answer to a short, storable value.

    Strips leading filler / framing ('um, I'm a paramedic actually' -> 'paramedic
    actually'), light per-store lead-ins, and caps length. Best-effort: the clean
    structured record is the person_qa row; this enriches person_facts/interests
    for prompt injection. Returns '' for non-answers so nothing junk is stored.
    """
    text = (answer or "").strip().rstrip(" .!?,")
    if not text or _DUNNO_PAT.match(text):
        return ""
    # Keep only the first clause — an em-dash/semicolon usually introduces an
    # aside ("rock climbing — I'm obsessed" -> "rock climbing"). Commas are kept
    # so "Austin, Texas" survives.
    text = re.split(r"\s*[—–;]\s*", text)[0].strip()
    prev = None
    while prev != text:
        prev = text
        text = _FILLER_PREFIX.sub("", text).strip()
    lead = _INTEREST_LEAD if store == "interest" else _FACT_LEAD
    prev = None
    while prev != text:
        prev = text
        text = lead.sub("", text).strip()
    words = text.split()
    if len(words) > 10:
        text = " ".join(words[:10])
    result = text.strip(" ,.!?")
    if result and result.split()[0].lower().strip(".,!?'") in _BAD_VALUE_LEAD:
        return ""
    return result


def _topic_from_answer(answer: str) -> str:
    return tidy_value(answer, "interest")


def _looks_like_topic(topic: str) -> bool:
    """True when a phrase reads like a short noun topic you could 'get into'."""
    words = (topic or "").split()
    if not (1 <= len(words) <= 4):
        return False
    return words[0].lower().strip(".,!?'") not in _BAD_VALUE_LEAD


def note_question_asked(person_id: int, question: dict) -> None:
    """Record that Rex asked this onboarding question (pending answer)."""
    try:
        rel_memory.save_question_asked(
            int(person_id),
            str(question.get("key")),
            str(question.get("text") or question.get("key") or ""),
            int(question.get("depth", 1)),
        )
    except Exception as exc:
        _log.debug("[onboarding] save_question_asked failed: %s", exc)


def record_answer(person_id: int, question: dict, answer_text: str) -> None:
    """Attach the answer to the pending question (familiarity bump) and enrich
    person_facts / person_interests with a tidied value for prompt injection."""
    answer = (answer_text or "").strip()
    if not answer:
        return
    try:
        rel_memory.answer_latest_pending_question(int(person_id), answer)
    except Exception as exc:
        _log.debug("[onboarding] answer_latest_pending_question failed: %s", exc)

    value = tidy_value(answer, str(question.get("store") or ""))
    if not value:
        return
    store = str(question.get("store") or "")
    try:
        if store == "fact":
            facts_memory.add_fact(
                int(person_id),
                str(question.get("category") or "identity"),
                str(question.get("key")),
                value,
                "explicit",
            )
        elif store == "interest":
            interests_memory.upsert_interest(
                int(person_id),
                value,
                category=str(question.get("category") or "hobby"),
                interest_strength="medium",
                source="explicit",
            )
    except Exception as exc:
        _log.debug("[onboarding] enrich-memory write failed: %s", exc)


def decline_pending(person_id: int, reason: str = "") -> None:
    """Close the pending onboarding question without filing a factful answer
    (used when the person declines / sets a boundary)."""
    try:
        rel_memory.decline_latest_pending_question(int(person_id), reason)
    except Exception as exc:
        _log.debug("[onboarding] decline_latest_pending_question failed: %s", exc)
