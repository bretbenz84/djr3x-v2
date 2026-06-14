"""
Premise-level anti-repeat for Rex's MAIN conversational replies (llm.stream).

The streaming reply path only ever guarded against *verbatim* repeats (the spoken
cooldown / last-line check) and leaned on the conversation arc to remind the model not
to "reuse jokes." That misses the real failure mode seen in the field: Rex landing the
SAME comedic premise three times in one chat — "nature reminds you modern convenience
is overrated" / "nature reminds us who's really in charge" / "nature reminds us who's
boss" — paraphrases that share almost no surface tokens, so a verbatim guard can't see
them, while the arc actively flagged "nature humor lands" and fed it back as encouragement.

This module tracks the salient *content* of Rex's recent lines (not their exact words)
and renders a short prompt directive telling him which premises/angles he has already
spent this conversation, so the model brings a fresh take instead of circling one bit.
Pure string heuristics — no LLM call, no embeddings — so it's free on the speech path.
"""

from __future__ import annotations

from collections import deque
import re
from typing import Optional

import config


# Per-line salient-keyword sets for Rex's most recent finalized lines (newest last).
_RECENT: deque[set[str]] = deque(maxlen=8)


# Words that carry no premise signal: function words, hedges, and Rex/DJ filler that
# shows up in nearly every line. Kept lowercase; compared after light stemming.
_STOPWORDS = frozenset(
    """
    the a an and or but so if then than that this these those there here
    is are was were be been being am do does did doing done have has had having
    will would shall should can could may might must to of in on at by for with
    from into onto about over under as it its it's they them their you your yours
    i me my mine we us our ours he him his she her hers who whom whose what which
    when where why how not no yes well oh ah uh um hmm just like really very even
    still yet always never ever maybe perhaps thing things stuff kind sort lot lots
    bit got get getting gonna wanna sure okay ok right kinda guess know think mean
    one two some any all most more less much many few every each other another
    out up down off back away around again now today tonight here's there's let
    going make makes made take takes taken give gives say says said see sees seen
    little big good bad nice great huh doesn't don't didn't isn't aren't wasn't
    doesn didn isn aren wasn weren wouldn couldn shouldn hasn haven hadn mustn
    needn won't wouldn't couldn't shouldn't hasn't haven't hadn't wouldnt
    """.split()
)

# Light, conservative stemmer so "remind" / "reminding" / "reminds" collapse to one
# token (the field repeat hinged on exactly that). Order matters: longest first.
_SUFFIXES = ("ingly", "ing", "edly", "ed", "es", "ly", "s")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _keywords(text: str) -> set[str]:
    """Salient content lemmas from a line — the words that carry its premise."""
    out: set[str] = set()
    for raw in re.findall(r"[A-Za-z']+", (text or "").lower()):
        # Take the root before any contraction/possessive ("who's" -> "who",
        # "rex's" -> "rex") so apostrophes never leak into a keyword.
        w = raw.split("'")[0]
        if len(w) < 4 or w in _STOPWORDS:
            continue
        stem = _stem(w)
        if len(stem) >= 3 and stem not in _STOPWORDS:
            out.add(stem)
    return out


def note_line(text: str) -> None:
    """Record the salient keywords of a finalized Rex line for premise anti-repeat."""
    kws = _keywords(text)
    if kws:
        _RECENT.append(kws)


def _avoid_keywords() -> list[str]:
    """Premises Rex is circling: keywords from his last line, plus any recurring across
    the recent window. Ordered most-recent-first, capped for a tight prompt line."""
    if not _RECENT:
        return []
    lines = list(_RECENT)
    last = lines[-1]

    # Frequency across the window catches the slow drift (turn 3 -> turn 8); the last
    # line's own keywords catch the back-to-back repeat (turn 8 -> turn 9).
    freq: dict[str, int] = {}
    for kw_set in lines:
        for kw in kw_set:
            freq[kw] = freq.get(kw, 0) + 1
    recurring = {kw for kw, n in freq.items() if n >= 2}

    avoid = set(last) | recurring
    if not avoid:
        return []

    cap = int(getattr(config, "PREMISE_ANTIREPEAT_MAX_KEYWORDS", 8))
    # Rank: recurring first (strongest signal), then by recency in the last line.
    ranked = sorted(
        avoid,
        key=lambda kw: (kw not in recurring, -freq.get(kw, 0), kw),
    )
    return ranked[:cap]


def build_avoid_directive() -> str:
    """Prompt section naming the premises Rex has already spent, or "" if none yet."""
    if not bool(getattr(config, "PREMISE_ANTIREPEAT_ENABLED", True)):
        return ""
    if len(_RECENT) < int(getattr(config, "PREMISE_ANTIREPEAT_MIN_LINES", 2)):
        return ""
    kws = _avoid_keywords()
    if not kws:
        return ""
    return (
        "Premise anti-repeat — you have ALREADY built bits/takes around these this "
        "conversation: " + ", ".join(kws) + ". Do NOT reuse the same premise, joke, or "
        "angle (even reworded). If the topic genuinely continues, come at it from a "
        "different direction — a new observation, a question you actually have, or a "
        "different target — never the same punchline twice."
    )


def recent_keywords() -> list[str]:
    """Test/inspection helper — the current avoid set."""
    return _avoid_keywords()


def clear() -> None:
    """Reset at conversation/session boundaries (mirrors topic_thread.clear())."""
    _RECENT.clear()
