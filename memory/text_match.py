"""
memory/text_match.py — light, shared text overlap for topic-relevant recall.

The topic-relevance signal used across the memory stores (facts, interests, episodes)
is "how many words of what the person JUST said does this memory mention?". Previously
each store re-tokenized WITHOUT stemming while the topic tokens were also unstemmed, so
"dogs" never matched a fact about a "dog" and "coding" missed "code". This module gives
ONE stemmer + overlap used on BOTH sides, so the comparison is symmetric.

Deliberately tiny and dependency-free — a real stemmer/embeddings would be the semantic
upgrade, but symmetric light stemming closes most of the brittle-keyword gap for free.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Longest-first so "running" → "run", not "runn". Mirrors premise_memory/_stem intent.
_SUFFIXES = ("ingly", "ing", "edly", "ied", "ies", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """Collapse common inflections to a shared root (conservative; min root length 3)."""
    w = (word or "").lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"        # stories → story, hobbies → hobby
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def stems(text: str) -> set[str]:
    """Stemmed content tokens (length ≥ 3 after stemming) from arbitrary text."""
    out: set[str] = set()
    for raw in _TOKEN_RE.findall((text or "").lower()):
        s = stem(raw)
        if len(s) >= 3:
            out.add(s)
    return out


def overlap_count(text: str, topic_tokens) -> int:
    """How many live-topic words this text mentions, comparing STEMMED forms on both
    sides. `topic_tokens` may be raw (unstemmed) tokens — they're stemmed here, so callers
    don't have to pre-stem. 0 when no tokens / no match."""
    if not topic_tokens:
        return 0
    topic = {stem(t) for t in topic_tokens if t}
    return len(stems(text) & topic)


def strong_overlap(topic_tokens, text, *, min_shared: int = 2, distinctive_len: int = 6) -> bool:
    """A CONSERVATIVE match: the text shares ≥ min_shared stems with the topic tokens,
    OR a single stem long enough to be distinctive on its own ("intern" ties an utterance
    to the stored intern-training plan; "new" alone must not tie it to anything).

    This is the write/inject threshold for statement-time recall — where a false positive
    injects a wrong "you already know this" or resolves the wrong plan — as opposed to
    overlap_count's permissive any-overlap ranking, which only reorders candidates."""
    if not topic_tokens:
        return False
    topic = {stem(t) for t in topic_tokens if t}
    shared = stems(text) & topic
    if len(shared) >= max(1, int(min_shared)):
        return True
    return any(len(s) >= int(distinctive_len) for s in shared)
