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

_INCOMPLETE_END_WORDS = {
    "about", "after", "and", "because", "before", "but", "for", "from",
    "if", "into", "or", "than", "to",
    "unless", "until", "when", "where", "while", "who", "with",
    "without",
}
_INCOMPLETE_END_PHRASES = (
    "about to",
    "after i",
    "all i",
    "and then",
    "because i",
    "because it",
    "before i",
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
    "planning to",
    "so i",
    "supposed to",
    "the thing is",
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

    if last in _INCOMPLETE_END_WORDS:
        if has_terminal_punct and last not in {"to", "because", "with", "about"}:
            return None
        return IncompleteSignal(
            f"ends with dangling word {last!r}",
            _prompt_for(words, lower),
        )

    return None


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
    if last in {"and", "but", "so", "then"}:
        return "And then?"
    return "You left me hanging. Finish the sentence?"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _words(text: str) -> list[str]:
    return [m.group(0) for m in _WORD_PAT.finditer(text or "")]


def _strip_incomplete_punctuation(text: str) -> str:
    cleaned = _clean(text)
    cleaned = re.sub(r"(\.\.\.|…)\s*$", "", cleaned).strip()
    return cleaned.rstrip(" ,;:")
