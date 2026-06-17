"""
intelligence/topic_thread.py - in-session topic continuity + conversation arc.

This module tracks the "soft thread" of the current conversation: what the
conversation is roughly about, whether the user seems engaged or avoidant, and
whether Rex has a question hanging in the air. It is intentionally heuristic and
session-local; durable memories belong in memory/*.

It also owns the **conversation arc** (Bet 1): a short running summary of the
live conversation — topics covered, what landed vs flopped, the person's mood,
and open threads — maintained by a cheap local-LLM (Ollama) call and fed back
into the system prompt so Rex can see what he already asked/roasted (stop
repeating himself) and call back to an earlier thread. The arc is refreshed on a
coalesced BACKGROUND worker triggered from the user-turn path, so it never
touches the time-to-first-speech path; on any failure the previous summary is
retained. Gated by config.CONVERSATION_ARC_ENABLED and local_llm availability.
The arc shares this module's session lifecycle: clear() wipes it.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional

_log = logging.getLogger(__name__)


_AVOID_PAT = re.compile(
    r"\b(rather not|don'?t want to|do not want to|change (the )?subject|"
    r"talk about something else|drop it|leave it|not talk about|not now|"
    r"don'?t ask|do not ask|stop asking)\b",
    re.IGNORECASE,
)
_PLAYFUL_PAT = re.compile(r"\b(lol|haha|funny|joke|roast|kidding|teasing)\b", re.I)
_DEPTH_PAT = re.compile(r"\b(because|actually|honestly|i think|i feel|it was|we were)\b", re.I)
_QUESTION_START = re.compile(
    r"^\s*(who|what|when|where|why|how|can|could|would|will|do|does|did|is|are|am|should)\b",
    re.IGNORECASE,
)
_SHORT_CONFIRMATION_PAT = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|right|affirmative|sure|"
    r"no|nope|nah|negative)(?:[,.! ]|$)",
    re.IGNORECASE,
)
_POLAR_QUESTION_START = re.compile(
    r"^\s*(?:is|are|am|was|were|do|does|did|will|would|can|could|"
    r"should|have|has|had|didn'?t|don'?t|doesn'?t|isn'?t|aren'?t|"
    r"won'?t|wouldn'?t|can'?t|couldn'?t)\b",
    re.IGNORECASE,
)
_EXPLICIT_INTEREST_SWITCH_PAT = re.compile(
    r"\b("
    r"i (?:really )?(?:like|love|am into|enjoy)|"
    r"i'?d like to talk about|i would like to talk about|"
    r"let'?s talk about|my favorite|favorite kind of"
    r")\b",
    re.IGNORECASE,
)

_TOPIC_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("grief/loss", re.compile(r"\b(died|death|dead|passed|loss|grief|funeral)\b", re.I), "heavy"),
    ("health", re.compile(r"\b(sick|ill|hospital|health|doctor|pain|diagnos)\w*\b", re.I), "heavy"),
    ("work", re.compile(r"\b(work|job|office|boss|coworker|meeting|project)\b", re.I), "mild"),
    ("pets", re.compile(r"\b(dog|cat|pet|puppy|kitten)\b", re.I), "mild"),
    ("music", re.compile(r"\b(music|song|track|album|artist|band|dj|playlist)\b", re.I), "light"),
    ("family", re.compile(r"\b(mom|dad|parent|grandpa|grandma|kid|child|wife|husband|partner)\b", re.I), "mild"),
    ("visual detail", re.compile(r"\b(shirt|hat|jacket|screen|camera|room|desk|poster|light)\b", re.I), "light"),
    ("identity", re.compile(r"\b(name|who am i|who is|who'?s|recognize)\b", re.I), "light"),
    ("plans", re.compile(r"\b(plan|weekend|tomorrow|today|tonight|trip|event)\b", re.I), "light"),
]

_STOPWORDS = {
    "the", "and", "but", "for", "with", "that", "this", "you", "your",
    "about", "have", "just", "what", "when", "where", "yeah", "yes",
    "no", "not", "really", "pretty", "good", "okay", "like",
    # Conversational filler that must never become a topic label. Without these,
    # "things are going well" produced the garbage topic "things / are" and
    # "I like watching Apple TV..." produced "watching / apple".
    "things", "thing", "stuff", "are", "going", "well", "doing", "been",
    "got", "now", "today", "lot", "kind", "sort", "way", "fine", "great",
    "guess", "maybe", "sure", "alright", "nothing", "something", "anything",
    "watching", "trying", "personally", "currently", "tonight", "little",
    "here", "there", "actually", "still", "gonna", "wanna", "really",
}

# Bare interjections / vocal noise must never become a topic label. Whisper emits
# these for throat-clears and filler ("ahem", "uh", "hmm"); without this, a lone
# "ahem" became the pinned topic "ahem" and persisted across turns.
_INTERJECTIONS = {
    "ahem", "uh", "uhh", "um", "umm", "hmm", "hmmm", "huh", "uhuh",
    "ha", "haha", "hah", "heh", "mm", "mmm", "mhm", "er", "erm",
    "oh", "ooh", "ah", "aah", "ugh", "oof", "eh", "meh", "yo", "hey",
    "wow", "oops", "ow", "ouch", "nope", "yep", "yup", "nah",
}


@dataclass
class TopicThread:
    label: str
    emotional_weight: str
    user_stance: str
    summary: str
    started_at: float
    updated_at: float
    turn_count: int = 0
    unresolved_question: Optional[str] = None
    last_user_text: str = ""
    last_assistant_question: str = ""


_current: Optional[TopicThread] = None


def clear() -> None:
    global _current
    _current = None
    _clear_arc()


def snapshot() -> Optional[dict]:
    if _current is None:
        return None
    data = asdict(_current)
    # Expose the running arc summary too (consumers that read label /
    # unresolved_question are unaffected — this only adds a key).
    data["arc_summary"] = arc_summary()
    return data


def note_assistant_turn(text: str) -> None:
    """Remember Rex's latest question so the next user turn can answer it."""
    global _current
    cleaned = (text or "").strip()
    if not cleaned:
        return
    if _current is None:
        now = time.monotonic()
        _current = TopicThread(
            label="conversation",
            emotional_weight="light",
            user_stance="neutral",
            summary="conversation opened by Rex",
            started_at=now,
            updated_at=now,
        )
    if "?" in cleaned:
        question = _last_question_sentence(cleaned)
        _current.unresolved_question = question
        _current.last_assistant_question = question
    _current.updated_at = time.monotonic()


def note_user_turn(
    text: str,
    person_id: Optional[int] = None,
    *,
    answered_question: Optional[dict] = None,
) -> None:
    del person_id  # reserved for later person-specific topic preferences
    global _current

    cleaned = (text or "").strip()
    if not cleaned:
        return

    unresolved_before = _current.unresolved_question if _current is not None else ""
    answers_unresolved = _answers_unresolved_question(cleaned, unresolved_before)
    now = time.monotonic()
    label, weight = _classify_topic(cleaned)
    stance = _classify_stance(
        cleaned,
        answered_question=answered_question,
        answers_unresolved=answers_unresolved,
    )

    if _current is None or _should_start_new_thread(cleaned, label, stance):
        _current = TopicThread(
            label=label,
            emotional_weight=weight,
            user_stance=stance,
            summary=_summarize_text(cleaned, label),
            started_at=now,
            updated_at=now,
            turn_count=1,
            last_user_text=cleaned,
        )
    else:
        _current.label = _merged_label(_current.label, label)
        _current.emotional_weight = _heavier(_current.emotional_weight, weight)
        _current.user_stance = stance
        _current.summary = _summarize_text(cleaned, _current.label)
        _current.updated_at = now
        _current.turn_count += 1
        _current.last_user_text = cleaned

    # An answer to Rex's question IS the new topic — adopt it instead of staying
    # stuck on whatever the thread was labelled before (the old merge kept the
    # garbage "things / are" label even after the user said "astrophotography").
    if answered_question and label and label != "current exchange":
        _current.label = label
        _current.summary = _summarize_text(cleaned, label)

    if answered_question or answers_unresolved or stance in {"engaged", "avoidant"}:
        _current.unresolved_question = None

    # An exchange has progressed — refresh the running conversation arc in the
    # background (coalesced, off the speech path, no-op when disabled/unavailable).
    _trigger_arc_refresh()


def note_answered_question(answered_question: Optional[dict] = None) -> None:
    """Mark Rex's outstanding question as answered without adding another turn."""
    if _current is None:
        return
    _current.unresolved_question = None
    _current.user_stance = "engaged"
    _current.updated_at = time.monotonic()
    if answered_question:
        answer = (answered_question.get("answer_text") or "").strip()
        if answer:
            _current.summary = _summarize_text(answer, _current.label)


def build_directive() -> str:
    if _current is None:
        return ""
    age = time.monotonic() - _current.updated_at
    if age > 300:
        return ""

    lines = [
        "Topic thread: keep continuity with the current conversational thread.",
        f"Current topic: {_current.label}.",
        f"Thread summary: {_current.summary}.",
        f"User stance: {_current.user_stance}; emotional weight: {_current.emotional_weight}.",
    ]
    if _current.unresolved_question:
        lines.append(
            f"Rex's unresolved question: {_current.unresolved_question!r}. "
            "Treat the user's latest utterance as a likely answer if it fits; "
            "do not ask an unrelated new question in the same breath."
        )
    if _current.user_stance == "avoidant":
        lines.append(
            "The user is steering away from this topic. Briefly acknowledge the "
            "boundary and let the topic drop unless they reopen it."
        )
    elif _current.user_stance == "terse":
        lines.append(
            "The user gave a short/low-energy reply. Do not interrogate. Either "
            "leave space, make one gentle follow-up, or shift softly."
        )
    elif _current.user_stance == "playful":
        lines.append(
            "The user is playful. Banter is welcome, but keep the thread connected "
            "instead of random-topic hopping."
        )
    elif _current.user_stance == "engaged":
        lines.append(
            "The user is engaging with the thread. Continue or deepen this topic "
            "before introducing anything new."
        )
    if _current.emotional_weight == "heavy":
        lines.append(
            "This topic is emotionally heavy. Prioritize care, consent, and pacing; "
            "no roasts about the vulnerable subject."
        )
    return "\n".join(lines)


def _classify_topic(text: str) -> tuple[str, str]:
    for label, pat, weight in _TOPIC_PATTERNS:
        if pat.search(text):
            return label, weight
    keywords = _keywords(text)
    if keywords:
        return " / ".join(keywords[:2]), "light"
    return "current exchange", "light"


def _classify_stance(
    text: str,
    *,
    answered_question: Optional[dict],
    answers_unresolved: bool = False,
) -> str:
    if _AVOID_PAT.search(text):
        return "avoidant"
    if _PLAYFUL_PAT.search(text):
        return "playful"
    words = re.findall(r"[A-Za-z']+", text)
    if answered_question or answers_unresolved or len(words) >= 8 or _DEPTH_PAT.search(text):
        return "engaged"
    if len([w for w in words if len(w) > 2]) <= 2:
        return "terse"
    return "neutral"


def _should_start_new_thread(text: str, label: str, stance: str) -> bool:
    if _current is None:
        return True
    if stance == "avoidant":
        return False
    if label == _current.label or label == "current exchange":
        return False
    if _current.emotional_weight == "heavy" and label not in {"grief/loss", "health"}:
        return _looks_like_explicit_switch(text)
    if _current.turn_count <= 1:
        return False
    return _looks_like_explicit_switch(text) or len(text.split()) >= 6


def _looks_like_explicit_switch(text: str) -> bool:
    lowered = text.lower()
    return (
        "speaking of" in lowered
        or "by the way" in lowered
        or "anyway" in lowered
        or "new subject" in lowered
        or "let's talk about" in lowered
        or bool(_EXPLICIT_INTEREST_SWITCH_PAT.search(text))
        or bool(_QUESTION_START.search(text))
    )


def _merged_label(current: str, incoming: str) -> str:
    if incoming in {"current exchange", current}:
        return current
    if current == "conversation":
        return incoming
    return current


def _heavier(a: str, b: str) -> str:
    order = {"light": 0, "mild": 1, "heavy": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _summarize_text(text: str, label: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > 140:
        cleaned = cleaned[:137].rstrip() + "..."
    return f"{label}: {cleaned}"


def _keywords(text: str) -> list[str]:
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", text)
        if w.lower() not in _STOPWORDS and w.lower() not in _INTERJECTIONS
    ]
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def _last_question_sentence(text: str) -> str:
    parts = re.findall(r"[^?]*\?", text)
    if not parts:
        return text[-180:]
    return parts[-1].strip()[-180:]


def _answers_unresolved_question(text: str, question: Optional[str]) -> bool:
    cleaned = (text or "").strip()
    q = (question or "").strip()
    if not cleaned or not q or "?" in cleaned:
        return False
    if _AVOID_PAT.search(cleaned):
        return True
    words = re.findall(r"[A-Za-z']+", cleaned)
    if len(words) >= 8 or _DEPTH_PAT.search(cleaned):
        return True
    if _SHORT_CONFIRMATION_PAT.match(cleaned) and _is_polar_or_tag_question(q):
        return True
    return False


def _is_polar_or_tag_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if _POLAR_QUESTION_START.match(q):
        return True
    return bool(
        re.search(
            r"(?:,\s*)?(?:right|correct|yeah|yes|no|okay|ok|huh)\?\s*$",
            q,
            re.IGNORECASE,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation arc memory (Bet 1)
#
# A running, local-LLM-maintained summary of the live conversation. It is folded
# into THIS module (rather than a parallel module) so the one place that already
# tracks "what this conversation is about" also owns the richer memory and shares
# its session lifecycle. The summary lives as module-level state — NOT on the
# per-thread TopicThread dataclass — because it must survive topic switches
# within a session (note_user_turn replaces _current on a new thread).
#
# Flow: note_user_turn() -> _trigger_arc_refresh() marks dirty and ensures one
# background worker is summarizing. The worker re-derives the summary FRESH from
# the most recent window of the in-memory session transcript (memory/conversations.py)
# via a single local_llm call, and stores the result under _arc_lock. The prompt
# assembler reads the stored summary instantly via build_arc_directive(). Nothing
# here ever runs on the turn/speech path, and every failure path retains the
# previous summary.
#
# Backend (config.CONVERSATION_ARC_BACKEND): "openai" (default) summarizes with
# gpt-4o-mini via the existing OpenAI client for a rich 5-field schema (Topics /
# Shared / Mood / Used-up / Open threads); "local" uses the qwen2.5:1.5b
# sidecar with a 3-field factual-only schema. The cloud call is fine here because
# the refresh is off the speech path and Rex's replies already depend on OpenAI.
#
# Either backend summarizes FRESH from the transcript window — NOT an incremental
# rewrite. An earlier version fed the prior summary back and the local model echoed
# it verbatim, freezing the arc on turn 1. The affective fields are local-unsafe
# (the 1.5B called declined topics "landed" and reported Rex's mood as the user's),
# hence the reduced local schema.
# ─────────────────────────────────────────────────────────────────────────────

_arc_lock = threading.Lock()
_arc_summary: str = ""          # running summary text (read into the system prompt)
_arc_cursor: int = 0            # transcript length summarized through (new-material gate)
_arc_refreshing: bool = False   # a background worker is currently summarizing
_arc_dirty: bool = False        # new material arrived while a worker was running
_arc_thread: Optional[threading.Thread] = None  # most recent worker (tests join it)

_ARC_SYSTEM_PROMPT = (
    "You are the memory module for Rex, a witty sarcastic droid talking with a user. "
    "Compress the conversation into a compact memory Rex can reuse so he does not "
    "repeat himself and can follow up later. Summarize in your own words — never copy "
    "the dialogue back, never use quotation marks. Track the USER, not Rex."
)


def _clear_arc() -> None:
    global _arc_summary, _arc_cursor, _arc_dirty
    with _arc_lock:
        _arc_summary = ""
        _arc_cursor = 0
        _arc_dirty = False
        # An in-flight worker is left to finish; its post-generate commit guard
        # (cursor check) discards a summary computed from the old transcript.


def _under_test_runner() -> bool:
    """True when running under unittest/pytest (and not explicitly opted in).

    Keys off the ENTRY POINT — sys.argv[0] is 'python -m unittest' under the
    project's test command, and pytest sets PYTEST_CURRENT_TEST — rather than
    "'unittest' in sys.modules", so an incidental import of unittest by some
    dependency can NOT disable the arc on the robot (which runs `python main.py`,
    argv0='main.py'). DJR3X_ARC_TEST_OPT_IN forces the production path (used by
    live-validation harnesses).
    """
    if os.environ.get("DJR3X_ARC_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def _arc_enabled() -> bool:
    """Whether the arc may run RIGHT NOW.

    Fail-safe under a test runner: the refresh fires from deep inside
    note_user_turn and (with the openai backend) would make a real cloud call with
    the live API key in apikeys.py. So unless a test explicitly opts in
    (DJR3X_ARC_TEST_OPT_IN), the arc is inert under unittest/pytest. Unit tests that
    exercise the refresh mock `_arc_enabled` (or set the opt-in) directly.
    """
    if _under_test_runner():
        return False
    return _arc_backend_available()


def _arc_backend_available() -> bool:
    """The real gate: arc configured on AND the chosen backend usable."""
    try:
        import config
        if not bool(getattr(config, "CONVERSATION_ARC_ENABLED", True)):
            return False
        backend = str(getattr(config, "CONVERSATION_ARC_BACKEND", "openai")).lower()
        if backend == "local":
            from intelligence import local_llm
            return bool(local_llm.enabled())
        return True  # openai: assume usable; a failed call retains the prior summary
    except Exception:
        return False


def _arc_generate(prompt: str, *, max_tokens: int, timeout: float) -> str:
    """Dispatch the summary call to the configured backend. Raises on failure so
    the caller retains the previous summary."""
    import config
    backend = str(getattr(config, "CONVERSATION_ARC_BACKEND", "openai")).lower()
    if backend == "local":
        from intelligence import local_llm
        return local_llm.generate(
            prompt, system=_ARC_SYSTEM_PROMPT, temperature=0.0,
            max_tokens=max_tokens, timeout_secs=timeout,
        ).strip()
    from intelligence import llm
    return llm.summarize_conversation_arc(
        prompt, system=_ARC_SYSTEM_PROMPT, max_tokens=max_tokens, timeout_secs=timeout,
    ).strip()


def _arc_backend() -> str:
    try:
        import config
        return str(getattr(config, "CONVERSATION_ARC_BACKEND", "openai")).lower()
    except Exception:
        return "openai"


def arc_summary() -> str:
    """Return the current running summary text (instant read; never blocks)."""
    with _arc_lock:
        return _arc_summary


def _arc_field(summary: str, label: str) -> str:
    """Pull one labelled line's value out of the arc summary (e.g. 'Mood')."""
    m = re.search(rf"(?mi)^\s*{re.escape(label)}\s*:\s*(.+)$", summary or "")
    return m.group(1).strip() if m else ""


def arc_persistence_fields() -> Optional[tuple[str, str, str]]:
    """Derive (summary, emotion_tone, topics) from the current arc for cross-session
    persistence (`memory/conversations.save_conversation`), or None if no arc yet.
    `summary` is the arc flattened to one line; tone/topics fill the structured
    columns from the Mood/Topics lines."""
    summary = arc_summary().strip()
    if not summary:
        return None
    flat = re.sub(r"\s*\n\s*", " · ", summary).strip()
    return flat, (_arc_field(summary, "Mood") or "neutral"), _arc_field(summary, "Topics")


# Mood words that mean "this is falling flat" — used to ease off the roast. Kept
# conservative (explicit disengagement/negativity) so a neutral/positive arc, or an
# empty one, never trips it.
_ARC_FLAT_MOOD_RE = re.compile(
    r"\b(diseng|flat|bored|boring|disappoint|annoy|frustrat|evasive|withdrawn|"
    r"indifferen|irritat|unimpressed|unenthusi|low.?energy|reluctan|dismissiv|"
    r"checked.?out|uninterested|aloof|terse|curt|deflat|defensive|guarded|tired)",
    re.IGNORECASE,
)


def arc_reads_flat() -> bool:
    """True when the arc's Mood read says the conversation is falling flat.

    Drives the 'ease off the roast' behavior (`social_frame._roast_level`). Keyed
    on Mood (the reliable 'read the room' field) rather than Used up/flopped, which
    usually mentions some flop every turn. Empty/positive arc → False.
    """
    return bool(_ARC_FLAT_MOOD_RE.search(_arc_field(arc_summary(), "Mood")))


def build_arc_directive() -> str:
    """Prompt section exposing the running conversation summary.

    Empty when the arc is disabled or there is no summary yet. Injected into the
    system prompt (downstream of the social-frame governors), never into the
    agenda directive — keeping the free-text summary clear of the agenda's
    regex re-parsing in social_frame.
    """
    try:
        import config
        if not bool(getattr(config, "CONVERSATION_ARC_ENABLED", True)):
            return ""
    except Exception:
        pass
    summary = arc_summary().strip()
    if not summary:
        return ""
    return (
        "Conversation arc — your running memory of THIS conversation. Use it to AVOID "
        "repeating yourself: don't re-ask questions, and don't reuse any joke, premise, "
        "roast, or angle you've already used — the 'Used up' line lists what you've spent, "
        "so steer clear of it even reworded (a premise that landed is SPENT, not a cue to "
        "do it again). 'Topics' and 'Open threads' are LIGHT context, not a script: a "
        "casual one-off mention (a snack they're eating, an offhand remark) is NOT a thread "
        "to keep returning to — touch it once and move on. Looping the SAME topic into "
        "reply after reply (working pizza into every line because they mentioned it once) is "
        "exactly the repetitive feel to avoid; follow where THEY take the conversation. You "
        "MAY call back to an Open thread when it genuinely fits, but never force it, read "
        "these notes aloud, or recite them verbatim.\n" + summary
    )


def _trigger_arc_refresh() -> None:
    """Mark the arc dirty and ensure a single background worker is summarizing.

    Coalesces a burst of calls (multiple note_user_turn paths in one turn) into at
    most one in-flight worker. Returns immediately — the refresh runs on a daemon
    thread. No-op when disabled/unavailable or when there is no new material.
    """
    global _arc_dirty, _arc_refreshing, _arc_cursor, _arc_thread
    if not _arc_enabled():
        return
    try:
        from memory import conversations
        transcript_len = len(conversations.get_session_transcript())
    except Exception:
        return
    with _arc_lock:
        if _arc_cursor > transcript_len:
            _arc_cursor = 0  # transcript was reset/cleared underneath us
        if transcript_len <= _arc_cursor:
            return  # nothing new to fold
        _arc_dirty = True
        if _arc_refreshing:
            return  # the running worker will pick up the new material
        _arc_refreshing = True
    thread = threading.Thread(target=_arc_worker, name="arc-refresh", daemon=True)
    _arc_thread = thread
    thread.start()


def _arc_worker() -> None:
    """Background loop: summarize while there is fresh material, then stop."""
    global _arc_dirty, _arc_refreshing
    try:
        while True:
            with _arc_lock:
                if not _arc_dirty:
                    return
                _arc_dirty = False
            _arc_refresh_core()
    finally:
        with _arc_lock:
            _arc_refreshing = False


def _arc_refresh_core() -> bool:
    """Re-derive the running summary from the recent transcript window. Synchronous.

    Returns True iff a refresh ran and the summary was updated. Never raises — on
    any failure (disabled, no new material, local LLM down/slow, empty/garbage
    output) the previous summary is retained and this returns False.
    """
    global _arc_summary, _arc_cursor
    if not _arc_enabled():
        return False
    try:
        from memory import conversations
        transcript = conversations.get_session_transcript()
    except Exception:
        return False

    with _arc_lock:
        if _arc_cursor > len(transcript):
            _arc_cursor = 0  # transcript reset under us
        if len(transcript) <= _arc_cursor:
            return False  # nothing new since the last summary
        committed_cursor = _arc_cursor

    try:
        import config
        max_tokens = int(getattr(config, "CONVERSATION_ARC_MAX_TOKENS", 200))
        timeout = float(getattr(config, "CONVERSATION_ARC_TIMEOUT_SECS", 8.0))
        window = int(getattr(config, "CONVERSATION_ARC_CONTEXT_LINES", 12))
    except Exception:
        max_tokens, timeout, window = 200, 8.0, 12

    recent = transcript[-window:] if window > 0 else transcript
    # The cloud model handles the richer 5-field schema (mood, landed-vs-flopped);
    # the local 1.5B sidecar only gets the 3 factual fields it can do reliably.
    rich = _arc_backend() != "local"
    prompt = _build_arc_prompt(_render_transcript_lines(recent), rich=rich)

    started = time.monotonic()
    try:
        updated = _arc_generate(prompt, max_tokens=max_tokens, timeout=timeout)
    except Exception as exc:
        _log.debug("[arc] refresh skipped (%s backend unavailable): %s", _arc_backend(), exc)
        return False

    if not _arc_output_ok(updated):
        _log.debug("[arc] rejected low-quality summary: %r", updated[:120])
        return False
    updated = _sanitize_summary(updated)
    if not updated:
        return False

    new_len = len(transcript)
    with _arc_lock:
        # Commit only if no clear()/reset slipped in while we were generating.
        if _arc_cursor != committed_cursor:
            _log.debug("[arc] discarding stale summary (cursor moved)")
            return False
        _arc_summary = updated
        _arc_cursor = new_len
    preview = re.sub(r"\s*\n\s*", " | ", updated).strip()
    _log.info(
        "[arc] summary updated in %.2fs (window %d lines): %s",
        time.monotonic() - started, len(recent), preview,
    )
    return True


def _arc_output_ok(text: str) -> bool:
    """Reject empty output, a transcript echo, or a degenerate repetition loop.

    The 1.5B model occasionally (a) parrots the dialogue back (lines beginning with
    a "User:"/"Rex:" speaker prefix) or (b) runs away repeating one token
    ("motivation, motivation, ..."). Neither may be stored as "memory" — keeping
    the previous good summary is strictly better.
    """
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"(?mi)^[~\s>*\-]*(user|rex)\s*:", t):  # transcript echo
        return False
    words = re.findall(r"[a-z']+", t.lower())  # runaway single-token repetition
    if len(words) >= 12:
        most = Counter(words).most_common(1)[0][1]
        if most >= 8 and most / len(words) > 0.30:
            return False
    return True


def _sanitize_summary(text: str) -> str:
    """Tidy an accepted summary: strip markdown emphasis/bullets and dedup + cap the
    comma list on each labelled line (kills milder repetition the guard let pass)."""
    out: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.strip().lstrip("*#->• ").replace("**", "").strip()
        if not line:
            continue
        if ":" in line:
            label, _, rest = line.partition(":")
            items: list[str] = []
            seen: set[str] = set()
            for item in rest.split(","):
                item = item.strip().strip("*").strip()
                key = item.lower()
                if item and key not in seen:
                    seen.add(key)
                    items.append(item)
                if len(items) >= 6:
                    break
            line = f"{label.strip()}: {', '.join(items)}" if items else f"{label.strip()}: -"
        out.append(line)
    return "\n".join(out)


def _render_transcript_lines(lines: list[dict]) -> str:
    # Normalize speakers to roles ("User"/"Rex") so the person's NAME never leaks
    # into the summary (it was landing in "Topics:" as e.g. "Bret Benziger").
    out: list[str] = []
    for entry in lines:
        raw = str(entry.get("speaker") or "").strip()
        speaker = "Rex" if raw.lower() == "rex" else "User"
        text = re.sub(r"\s+", " ", str(entry.get("text") or "")).strip()
        if text:
            out.append(f"{speaker}: {text}")
    return "\n".join(out)


def _build_arc_prompt(transcript_rendered: str, *, rich: bool = True) -> str:
    # No prior summary is fed (that caused echo/freeze); the conversation comes
    # first, the rigid format last, and the prompt does NOT end with blank labels
    # (that turned it into a completion task and made the model echo the transcript).
    count = "five" if rich else "three"
    instructions = (
        f"Conversation to summarize (oldest to newest):\n{transcript_rendered}\n\n"
        f"Summarize the conversation so far as EXACTLY these {count} labelled lines "
        "and nothing else — no preamble, no dialogue, no paragraph. Each line is the "
        "label then a few words. Name the real subjects, never the speakers. Use '-' "
        "if empty.\n"
    )
    if rich:
        # Cloud model — the full "feels alive" schema (mood, landed-vs-flopped).
        return instructions + (
            "Topics: <subjects discussed>\n"
            "Shared: <DURABLE facts the user revealed about themselves — interests, "
            "plans, work, life events, relationships; NOT transient surroundings like the "
            "room/clutter/boxes, temperature, background noise, or weather>\n"
            "Mood: <the user's current mood and energy>\n"
            "Used up (do NOT reuse): <jokes, premises, roasts, or comedic angles Rex has "
            "ALREADY used this conversation, plus anything that fell flat or they dodged — "
            "all of it to AVOID next, not repeat>\n"
            "Open threads: <specific things Rex could follow up on later>"
        )
    # Local 1.5B sidecar — three factual fields only (it can't judge affect).
    return instructions + (
        "Topics: <subjects discussed>\n"
        "Shared: <facts the user revealed about themselves>\n"
        "Open threads: <specific things Rex could follow up on>"
    )
