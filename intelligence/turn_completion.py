"""
intelligence/turn_completion.py - hold and repair incomplete spoken turns.

Speech segmentation is intentionally fast, so a natural pause can split one
human sentence into two transcribed chunks. This module catches obvious
unfinished fragments, holds them briefly, and lets the next chunk complete the
turn before Rex responds.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import threading
import time
from typing import Optional

import numpy as np

import config

_log = logging.getLogger(__name__)


_WORD_PAT = re.compile(r"[A-Za-z0-9']+")
_ELLIPSIS_PAT = re.compile(r"(\.\.\.|…)\s*$")
_CANCEL_PAT = re.compile(
    r"\b(never mind|nevermind|forget it|scratch that|cancel that|ignore that)\b",
    re.IGNORECASE,
)
_TERMINAL_PUNCT_PAT = re.compile(r"[.!?]\s*$")
_TERMINAL_QUESTION_PAT = re.compile(r"\?\s*$")
# A follower that OPENS a brand-new wh/aux question ("What do you see?") must not
# be glued onto a held fragment — it is its own turn.
_WH_START_PAT = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whose|whom|"
    r"do|does|did|are|is|am|can|could|would|will|should)\b",
    re.IGNORECASE,
)
# Lowercase tokens that grammatically attach a follower to the prefix — a real
# continuation ("...to" + "the store", "...and then" + "we left"). A follower
# that begins with one of these is treated as a continuation, never distinct.
_CONTINUATION_OPENERS = frozenset({
    "and", "but", "so", "or", "because", "then", "to", "the", "a", "an",
    "with", "about", "for", "of", "at", "in", "on", "that", "which",
})
_COMPLETE_PREPOSITION_QUESTION_PAT = re.compile(
    r"^\s*(?:who|what|when|where|why|how|which|whose|whom|"
    r"can|could|would|will|do|does|did|is|are|am|should)\b"
    r".*\b(?:about|for|from|to|with|in|on|at|of|as|by|up|out|over|into|like)"
    r"\s*\??\s*$",
    re.IGNORECASE,
)
_COMPLETE_EMBEDDED_PREPOSITION_PAT = re.compile(
    r"\b(?:who|whom|what|which|where|whose|that)\b.*(?:"
    r"\b(?:refer(?:red|ring)?|talk(?:ed|ing)?|speak(?:s|ing|spoke|spoken)?|"
    r"ask(?:s|ed|ing)?|look(?:s|ed|ing)?|search(?:es|ed|ing)?|"
    r"wait(?:s|ed|ing)?|listen(?:s|ed|ing)?|respond(?:s|ed|ing)?|"
    r"repl(?:y|ies|ied|ying)|connect(?:s|ed|ing)?|belong(?:s|ed|ing)?|"
    r"relat(?:es|ed|ing)|come|came|coming|work(?:s|ed|ing)?|"
    r"live(?:s|d|ing)?|am|are|is|was|were|be|been|being)\b|"
    r"\b(?:i|you|we|they|it|that|this|who|what|where|there)'(?:m|re|s)\b"
    r")\s+(?:about|for|from|to|with|in|on|at|of|as|by|up|out|over|into|like)\s*$",
    re.IGNORECASE,
)

_INCOMPLETE_END_WORDS = {
    "about", "and", "because", "but", "for", "from",
    "if", "into", "or", "than", "the", "to",
    "unless", "until", "when", "where", "while", "who", "with",
    "without",
    # Common danglers that were missing — "well we're currently in" (the live
    # case where Rex interrupted a mid-sentence pause) ends with "in".
    "in", "on", "at", "of", "as", "by", "up", "out", "over",
    "a", "an", "my", "our",
    # NOTE: "before" and "after" are deliberately NOT here — they are far more
    # often sentence-final adverbs ("never done that before", "the morning after")
    # than danglers, and mlx_whisper rarely emits the terminal punctuation that
    # would otherwise save them. The genuinely-dangling forms ("before I left",
    # "after the show") are caught by _INCOMPLETE_END_PHRASES below. Live failure
    # 2026-06-17: "doing things I've never done before" -> "Finish the sentence?".
}
_INCOMPLETE_END_PHRASES = (
    "about to",
    "after i",
    "after the",
    "after we",
    "all i",
    "and then",
    "because i",
    "because it",
    "before i",
    "before the",
    "before we",
    "but i",
    "going to",
    "gonna",
    "got to",
    "had to",
    "have to",
    "i am",
    "i was",
    "i wanted to",
    "i want to",
    "i'm",
    "i'm going to",
    "if i",
    "if we",
    "it was",
    "let me",
    "looking forward to",
    "my point is",
    "need to",
    "needed to",
    "planning to",
    "so i",
    "supposed to",
    "the thing is",
    "to a",
    "to an",
    "to the",
    "there was",
    "trying to",
    "wait because",
    "wanted to",
    "want to",
    "when i",
    "when we",
)


@dataclass
class IncompleteSignal:
    reason: str
    prompt: str


@dataclass
class PendingFragment:
    text: str
    audio_array: object
    raw_best_id: Optional[int]
    raw_best_name: Optional[str]
    raw_best_score: float
    created_at: float
    hold_until: float
    reason: str
    prompt: str
    prompted_at: Optional[float] = None

    def to_log_dict(self) -> dict:
        return {
            "text": self.text,
            "raw_best_id": self.raw_best_id,
            "raw_best_name": self.raw_best_name,
            "raw_best_score": self.raw_best_score,
            "created_at": self.created_at,
            "hold_until": self.hold_until,
            "reason": self.reason,
            "prompt": self.prompt,
            "prompted_at": self.prompted_at,
        }


_lock = threading.Lock()
_pending: Optional[PendingFragment] = None


def enabled() -> bool:
    return bool(getattr(config, "INCOMPLETE_TURN_ENABLED", True))


def classify(text: str) -> Optional[IncompleteSignal]:
    """Return an incomplete-turn signal for obvious dangling fragments."""
    if not enabled():
        return None
    cleaned = _clean(text)
    if not cleaned:
        return None
    words = _words(cleaned)
    if not words:
        return None

    if _ELLIPSIS_PAT.search(text or ""):
        return IncompleteSignal("explicit ellipsis", _prompt_for(words, cleaned))

    # A transcribed full question can naturally end with a preposition:
    # "What are you talking about?", "Who are you with?", "Where are you from?"
    # Those are complete turns, not dangling fragments.
    if _TERMINAL_QUESTION_PAT.search(text or ""):
        return None

    # If ASR gave us strong final punctuation, trust it unless the sentence
    # still ends on an impossible cliffhanger like "going to."
    has_terminal_punct = bool(_TERMINAL_PUNCT_PAT.search(text or ""))
    lower = cleaned.lower().strip(" .!?")
    last = words[-1].lower()

    if len(words) < 3:
        return None

    if lower.endswith(_INCOMPLETE_END_PHRASES):
        phrase_tail = lower.rsplit(" ", 2)[-1]
        return IncompleteSignal(
            f"ends with phrase {phrase_tail!r}",
            _prompt_for(words, lower),
        )

    if _COMPLETE_PREPOSITION_QUESTION_PAT.match(cleaned):
        return None

    # Embedded wh-clauses often end with a preposition but are complete
    # answers: "I don't know who you're referring to", "that's where I'm from".
    if _COMPLETE_EMBEDDED_PREPOSITION_PAT.search(lower):
        return None

    if last in _INCOMPLETE_END_WORDS:
        if has_terminal_punct and last not in {"to", "because", "with", "about"}:
            return None
        return IncompleteSignal(
            f"ends with dangling word {last!r}",
            _prompt_for(words, lower),
        )

    return None


def _starts_with_continuation_opener(text: str) -> bool:
    m = re.match(r"\s*([A-Za-z']+)", text or "")
    return bool(m and m.group(1).lower() in _CONTINUATION_OPENERS)


# Verb-completion danglers: a prefix ending here grammatically DEMANDS the
# follower as its object ("I was going to" + "go to the store") — so even a
# punctuated follower should still chain. NOTE this excludes exclamation stems
# like "What the" (ends "the"), which is NOT a verb-completion dangler.
_VERB_DANGLER_ENDINGS = frozenset({"to", "gonna", "into", "onto", "wanna", "gotta"})


def _prefix_demands_continuation(prefix_text: str) -> bool:
    toks = re.findall(r"[a-z']+", (prefix_text or "").lower())
    return bool(toks and toks[-1] in _VERB_DANGLER_ENDINGS)


def _wh_word(text: str) -> str:
    m = re.match(r"\s*([A-Za-z]+)", text or "")
    w = (m.group(1).lower() if m else "")
    return w if w in {"who", "what", "when", "where", "why", "how", "which", "whose"} else ""


def _is_distinct_new_thought(prefix_text: str, follower_text: str) -> bool:
    """True when the follower is its OWN complete thought, not a continuation.

    Guards the merge so a held fragment ("What the...") is not glued onto an
    unrelated complete sentence or new question ("What do you see?"). A follower
    that is itself a dangling fragment, or that grammatically attaches to the
    prefix (opens with a connective, or completes a verb-dangler prefix), is NOT
    distinct and still chains.
    """
    follower = (follower_text or "").strip()
    if not follower:
        return False
    # The follower is itself incomplete -> it should chain, not stand alone.
    if classify(follower) is not None:
        return False
    # Opens with a connective/article -> grammatical continuation of the prefix.
    if _starts_with_continuation_opener(follower):
        return False
    words = _words(follower)
    if len(words) < 3:
        return False
    # (1) A new wh/aux QUESTION is its own turn (the logged "What do you see?"),
    # even when ASR drops the terminal '?'. A 4+ word wh/aux-led clause, or one
    # that RESTARTS with the same wh-word the held fragment began with ("What the"
    # -> "What do you see"), is a fresh question, not a short continuation.
    if _WH_START_PAT.match(follower):
        if (
            _TERMINAL_QUESTION_PAT.search(follower)
            or len(words) >= 4
            or (_wh_word(follower) and _wh_word(follower) == _wh_word(prefix_text))
        ):
            return True
    # (2) ASR marked it a full sentence AND the prefix doesn't grammatically
    # demand it as a verb object -> a distinct declarative.
    if _TERMINAL_PUNCT_PAT.search(follower) and not _prefix_demands_continuation(prefix_text):
        return True
    return False


def hold(
    *,
    text: str,
    audio_array: object,
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    raw_best_score: float,
    signal: IncompleteSignal,
) -> PendingFragment:
    """Store an incomplete fragment and return it."""
    global _pending
    hold_secs = float(getattr(config, "INCOMPLETE_TURN_HOLD_SECS", 4.0))
    now = time.monotonic()
    pending = PendingFragment(
        text=_strip_incomplete_punctuation(text),
        audio_array=audio_array,
        raw_best_id=raw_best_id,
        raw_best_name=raw_best_name,
        raw_best_score=float(raw_best_score or 0.0),
        created_at=now,
        hold_until=now + max(0.5, hold_secs),
        reason=signal.reason,
        prompt=signal.prompt,
    )
    with _lock:
        _pending = pending
    _log.info(
        "[turn_completion] holding incomplete fragment: %s",
        pending.to_log_dict(),
    )
    return pending


def consume_continuation(
    *,
    text: str,
    audio_array: object,
    raw_best_id: Optional[int],
    raw_best_name: Optional[str],
    raw_best_score: float,
) -> Optional[dict]:
    """
    If a pending incomplete fragment exists and this text can complete it,
    consume the fragment and return merged text/audio/speaker hints.
    """
    global _pending
    cleaned = _clean(text)
    if not cleaned:
        return None

    now = time.monotonic()
    with _lock:
        pending = _pending
        if pending is None:
            return None

        if _CANCEL_PAT.search(cleaned):
            _pending = None
            _log.info(
                "[turn_completion] pending fragment cancelled by user text=%r pending=%s",
                text,
                pending.to_log_dict(),
            )
            return {"action": "cancel", "text": text}

        if not _continuation_window_open(pending, now):
            _pending = None
            _log.info(
                "[turn_completion] pending fragment stale before continuation: %s",
                pending.to_log_dict(),
            )
            return None

        if bool(
            getattr(config, "INCOMPLETE_TURN_MERGE_REJECT_DISTINCT", True)
        ) and _is_distinct_new_thought(pending.text, cleaned):
            # The follower is its own complete utterance (e.g. a new question),
            # not a continuation of the held fragment. Drop the stale fragment
            # and let this turn be processed standalone instead of merging into
            # garble like "What the What do you see?".
            _pending = None
            _log.info(
                "[turn_completion] follower is a distinct new utterance, not "
                "merging: pending=%r follower=%r",
                pending.text,
                cleaned,
            )
            return None

        _pending = None

    merged_text = merge_text(pending.text, cleaned)
    merged_audio = merge_audio(pending.audio_array, audio_array)
    best_id = raw_best_id
    best_name = raw_best_name
    best_score = float(raw_best_score or 0.0)
    if pending.raw_best_score > best_score:
        best_id = pending.raw_best_id
        best_name = pending.raw_best_name
        best_score = pending.raw_best_score

    result = {
        "action": "merge",
        "text": merged_text,
        "audio_array": merged_audio,
        "raw_best_id": best_id,
        "raw_best_name": best_name,
        "raw_best_score": best_score,
        "pending_text": pending.text,
        "continuation_text": cleaned,
        "was_prompted": pending.prompted_at is not None,
    }
    _log.info(
        "[turn_completion] merged incomplete fragment: %r + %r -> %r",
        pending.text,
        cleaned,
        merged_text,
    )
    return result


def mark_prompt_due() -> Optional[PendingFragment]:
    """
    Mark and return the pending fragment if its silent hold expired and Rex
    should ask a tiny completion repair. The fragment remains pending so the
    user's answer can still merge into the original thought.
    """
    global _pending
    now = time.monotonic()
    with _lock:
        if _pending is None or _pending.prompted_at is not None:
            return None
        if now < _pending.hold_until:
            return None
        _pending.prompted_at = now
        _log.info(
            "[turn_completion] incomplete hold expired, prompting: %s",
            _pending.to_log_dict(),
        )
        return _pending


def clear_stale_prompted() -> Optional[PendingFragment]:
    """Clear a prompted fragment after its answer window expires."""
    global _pending
    now = time.monotonic()
    reply_window = float(
        getattr(config, "INCOMPLETE_TURN_PROMPT_REPLY_WINDOW_SECS", 10.0)
    )
    with _lock:
        if _pending is None or _pending.prompted_at is None:
            return None
        if now - _pending.prompted_at <= max(1.0, reply_window):
            return None
        stale = _pending
        _pending = None
    _log.info("[turn_completion] prompted fragment expired: %s", stale.to_log_dict())
    return stale


def pending_snapshot() -> Optional[dict]:
    with _lock:
        if _pending is None:
            return None
        return _pending.to_log_dict()


def clear() -> None:
    global _pending
    with _lock:
        _pending = None


def merge_text(prefix: str, suffix: str) -> str:
    first = _strip_incomplete_punctuation(prefix)
    second = _clean(suffix)
    if not first:
        return second
    if not second:
        return first
    second = _drop_boundary_overlap(first, second)
    if not second:
        return first
    return f"{first} {second}".strip()


def merge_audio(first: object, second: object) -> object:
    try:
        if first is None:
            return second
        if second is None:
            return first
        silence_len = int(float(getattr(config, "AUDIO_SAMPLE_RATE", 16000)) * 0.08)
        silence = np.zeros(silence_len, dtype=np.float32)
        return np.concatenate([
            np.asarray(first, dtype=np.float32),
            silence,
            np.asarray(second, dtype=np.float32),
        ])
    except Exception:
        return second if second is not None else first


def _continuation_window_open(pending: PendingFragment, now: float) -> bool:
    if pending.prompted_at is not None:
        reply_window = float(
            getattr(config, "INCOMPLETE_TURN_PROMPT_REPLY_WINDOW_SECS", 10.0)
        )
        return now - pending.prompted_at <= max(1.0, reply_window)
    # If the human starts speaking just as the hold expires, prefer merging
    # over interrupting them with the repair prompt.
    return now <= pending.hold_until + 0.75


def _prompt_for(words: list[str], text: str) -> str:
    lower = text.lower().strip(" .!?")
    last = words[-1].lower() if words else ""
    if lower.endswith(("going to", "gonna", "planning to")):
        return "Going where?"
    if lower.endswith(("want to", "wanted to", "trying to", "need to", "have to", "supposed to", "about to")):
        return "Going to do what?"
    if last == "because":
        return "Because why?"
    if last == "with":
        return "With who?"
    if last == "about":
        return "About what?"
    if last == "for":
        return "For what?"
    if last == "the" or lower.endswith(("to a", "to an", "to the")):
        return "The what?"
    if last in {"and", "but", "so", "then"}:
        return "And then?"
    return "You left me hanging. Finish the sentence?"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _words(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_PAT.finditer(text or "")]


def _drop_boundary_overlap(prefix: str, suffix: str) -> str:
    prefix_words = _words(prefix)
    suffix_words = _words(suffix)
    max_overlap = min(4, len(prefix_words), len(suffix_words))
    for n in range(max_overlap, 0, -1):
        if [w.lower() for w in prefix_words[-n:]] != [
            w.lower() for w in suffix_words[:n]
        ]:
            continue
        match_end = 0
        count = 0
        for match in _WORD_PAT.finditer(suffix):
            count += 1
            match_end = match.end()
            if count >= n:
                break
        trimmed = suffix[match_end:].lstrip(" ,;:")
        _log.debug(
            "[turn_completion] dropped %d-word boundary overlap while merging: %r",
            n,
            suffix[:match_end],
        )
        return trimmed
    return suffix


def _strip_incomplete_punctuation(text: str) -> str:
    cleaned = _clean(text)
    cleaned = re.sub(r"(\.\.\.|…)\s*$", "", cleaned).strip()
    return cleaned.rstrip(" ,;:")
