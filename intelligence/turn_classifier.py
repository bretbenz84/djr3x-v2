"""
intelligence/turn_classifier.py — one cheap structured read of the current user turn.

Bet 3: a single local-LLM (qwen2.5:1.5b) call returning a TurnClass
{topic, engagement, intent, sentiment, wants_pivot, addressee}, meant to retire the
regex zoo (user_energy._classify, conversation_steering._looks_disengaged,
topic_thread._classify_topic, parts of dialogue_act) with one fuzzy LLM judgment.

Unlike the conversation arc (background, off the speech path), this runs ON the
turn's critical path — callers want it BEFORE building the reply — so it is built
to be fast (tiny output, low token cap) and to DEGRADE GRACEFULLY: on disabled /
unavailable / slow / malformed output, classify() returns None and callers keep
their existing deterministic heuristics. Gated by
config.CONVERSATION_TURN_CLASSIFIER_ENABLED (default False until latency is
validated on the robot). Labelled-line output (NOT JSON) — the same shape that
proved reliable for the arc on the 1.5B model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Optional

_log = logging.getLogger(__name__)

_ENGAGEMENT = {"engaged", "neutral", "low"}
_INTENT = {
    "answer", "question", "command", "share", "banter", "correction",
    "smalltalk", "other",
}
_SENTIMENT = {"positive", "neutral", "negative"}
_ADDRESSEE = {"rex", "other", "group", "unclear"}

_SYSTEM = (
    "You label ONE user turn from a conversation with Rex, a sarcastic DJ droid. "
    "Reply with ONLY the six labelled lines requested, each using a lowercase value "
    "from its allowed set. No preamble, no commentary, no extra lines."
)


@dataclass
class TurnClass:
    topic: str          # short subject label, "" if none
    engagement: str     # engaged | neutral | low
    intent: str         # answer | question | command | share | banter | correction | smalltalk | other
    sentiment: str      # positive | neutral | negative
    wants_pivot: bool   # the user wants to change the subject
    addressee: str      # rex | other | group | unclear
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def enabled() -> bool:
    """True only when the classifier is configured on AND the local LLM is available."""
    try:
        import config
        if not bool(getattr(config, "CONVERSATION_TURN_CLASSIFIER_ENABLED", False)):
            return False
        from intelligence import local_llm
        return bool(local_llm.enabled())
    except Exception:
        return False


def classify(text: str, *, rex_last_line: str = "") -> Optional[TurnClass]:
    """Structured read of the user turn, or None if disabled/unavailable/malformed.

    Never raises. Callers MUST treat None as "no signal" and fall back to their
    existing deterministic heuristics.
    """
    cleaned = (text or "").strip()
    if not cleaned or not enabled():
        return None
    try:
        import config
        max_tokens = int(getattr(config, "CONVERSATION_TURN_CLASSIFIER_MAX_TOKENS", 64))
        timeout = float(getattr(config, "CONVERSATION_TURN_CLASSIFIER_TIMEOUT_SECS", 1.5))
    except Exception:
        max_tokens, timeout = 64, 1.5

    try:
        from intelligence import local_llm
        raw = local_llm.generate(
            _build_prompt(cleaned, rex_last_line),
            system=_SYSTEM,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_secs=timeout,
        )
    except Exception as exc:
        _log.debug("[turn_classifier] skipped (local LLM unavailable): %s", exc)
        return None
    return parse(raw)


def _build_prompt(text: str, rex_last_line: str) -> str:
    ctx = f'Rex just said: "{rex_last_line.strip()}"\n' if rex_last_line.strip() else ""
    return (
        f'{ctx}User said: "{text}"\n\n'
        "Output EXACTLY these six lines, nothing else:\n"
        "Topic: <2-4 word subject, or - if none>\n"
        "Engagement: engaged | neutral | low\n"
        "Intent: answer | question | command | share | banter | correction | other\n"
        "Sentiment: positive | neutral | negative\n"
        "Pivot: yes | no\n"
        "Addressee: rex | other | group | unclear"
    )


def _field(raw: str, label: str) -> str:
    m = re.search(rf"(?mi)^\s*{label}\s*:\s*(.+)$", raw or "")
    return m.group(1).strip() if m else ""


def _pick(value: str, allowed: set[str], default: str) -> str:
    """First allowed lowercase token in the value (handles 'engaged (high)' etc.)."""
    for tok in re.findall(r"[a-z_]+", (value or "").lower()):
        if tok in allowed:
            return tok
    return default


def parse(raw: str) -> Optional[TurnClass]:
    """Parse the labelled-line model output into a validated TurnClass, or None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    # Guard against a malformed / echoed response: require at least one of the
    # categorical lines to be present before trusting the (defaulted) rest.
    if not (_field(raw, "Engagement") or _field(raw, "Intent") or _field(raw, "Sentiment")):
        return None
    topic = _field(raw, "Topic")
    topic = "" if topic.lower() in {"", "-", "none", "n/a", "unknown"} else re.sub(r"\s+", " ", topic)[:60]
    pivot_raw = _field(raw, "Pivot").lower()
    return TurnClass(
        topic=topic,
        engagement=_pick(_field(raw, "Engagement"), _ENGAGEMENT, "neutral"),
        intent=_pick(_field(raw, "Intent"), _INTENT, "other"),
        sentiment=_pick(_field(raw, "Sentiment"), _SENTIMENT, "neutral"),
        wants_pivot=("yes" in pivot_raw or "true" in pivot_raw),
        addressee=_pick(_field(raw, "Addressee"), _ADDRESSEE, "unclear"),
        raw=raw,
    )
