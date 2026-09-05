"""
memory/retrieval.py — unified cross-silo memory ranking for prompt injection.

Each store used to be fetched and capped on its OWN axis (12 facts, 8 interests), so a
person with 25 strong facts and 2 weak interests still burned 8 interest slots while
good facts were cut — and nothing bounded the TOTAL. This layer pulls candidates from
the silos, scores them on ONE axis (base × type-weight + topic relevance), and packs to
a single global budget. Rendering in _build_person_context is unchanged — only the
SELECTION (which items, how many of each) is now unified and bounded.

Relevance is pluggable (`set_relevance_backend`): the default is stemmed keyword overlap
(memory.text_match); the semantic layer registers an embedding-cosine backend here so
"make recall useful" can graduate from keywords to meaning WITHOUT touching this scorer
or the prompt. Failure-safe: any backend error falls back to keyword overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    kind: str        # 'fact' | 'interest'
    score: float
    overlap: float   # topic-relevance signal (keyword count or scaled cosine)
    row: dict


# ── Pluggable relevance backend ─────────────────────────────────────────────────
# A backend is fn(topic_tokens, text, cap) -> float in [0, cap]. Default: keyword overlap.
_relevance_backend = None
_semantic_checked = False


def set_relevance_backend(fn) -> None:
    """Register a relevance scorer (e.g. the semantic embedding layer). None resets to
    the keyword default."""
    global _relevance_backend, _semantic_checked
    _relevance_backend = fn
    _semantic_checked = fn is not None  # explicit set wins; None re-arms auto-install


def _ensure_backend() -> None:
    """Lazily install the semantic backend the first time, IFF it's enabled. Keeps the
    semantic module (and its embedding deps) unimported when the feature is off."""
    global _semantic_checked
    if _semantic_checked or _relevance_backend is not None:
        return
    _semantic_checked = True
    try:
        import config
        if getattr(config, "MEMORY_SEMANTIC_RECALL_ENABLED", False):
            from memory import semantic
            set_relevance_backend(semantic.relevance)
            _log.info("[retrieval] semantic embedding relevance enabled")
    except Exception as exc:
        _log.debug("semantic backend install skipped: %s", exc)


def _relevance(topic_tokens, text: str, cap: int) -> float:
    if not topic_tokens:
        return 0.0
    _ensure_backend()
    if _relevance_backend is not None:
        try:
            return float(_relevance_backend(topic_tokens, text, cap))
        except Exception as exc:
            _log.debug("relevance backend failed; using keyword overlap: %s", exc)
    from memory import text_match
    return float(min(text_match.overlap_count(text, topic_tokens), cap))


# ── Per-silo base scores (0..~1) ────────────────────────────────────────────────

_STRENGTH_RANK = {"high": 1.0, "medium": 0.66, "low": 0.33}


def _fact_base(fact: dict) -> float:
    from memory import facts as facts_db
    return float(facts_db.score_fact_for_prompt(fact))


def _interest_base(interest: dict) -> float:
    rank = _STRENGTH_RANK.get((interest.get("interest_strength") or "low"), 0.33)
    try:
        conf = float(interest.get("confidence") or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    return 0.6 * rank + 0.4 * conf


def _fact_text(fact: dict) -> str:
    return " ".join(str(fact.get(k) or "") for k in ("key", "value", "category"))


def _interest_text(interest: dict) -> str:
    return " ".join(str(interest.get(k) or "") for k in ("name", "category", "notes"))


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _inline_budget():
    """One total wall-clock budget for the semantic backend's inline embedding
    during this retrieval (memory.semantic.turn_budget); a no-op context when the
    semantic layer is off or unavailable."""
    import contextlib
    try:
        import config
        if not getattr(config, "MEMORY_SEMANTIC_RECALL_ENABLED", False):
            return contextlib.nullcontext()
        from memory import semantic
        return semantic.turn_budget()
    except Exception:
        return contextlib.nullcontext()


def retrieve_person_memory(
    person_id: int,
    *,
    topic_tokens=None,
    mute_terms=None,
    budget: int | None = None,
    fact_pool: int = 30,
    interest_pool: int = 24,
) -> dict:
    """Return a globally-budgeted, cross-silo selection of a person's facts + interests.

    Pulls a generous candidate pool from each silo (already Tier-C filtered: ephemeral /
    expired-provisional / boundary-muted facts are gone), scores every candidate on one
    axis, and keeps the top `budget` across BOTH silos. Returns
    {"facts": [...], "interests": [...]} preserving each silo's internal score order, so
    _build_person_context renders them exactly as before — only the counts are unified."""
    from memory import facts as facts_db, interests as interests_db

    boost = float(_cfg("MEMORY_TOPIC_RELEVANCE_BOOST", 0.5))
    cap = int(_cfg("MEMORY_TOPIC_RELEVANCE_MAX_MATCHES", 3))
    fw = float(_cfg("MEMORY_RETRIEVAL_FACT_WEIGHT", 1.0))
    iw = float(_cfg("MEMORY_RETRIEVAL_INTEREST_WEIGHT", 0.85))
    if budget is None:
        budget = int(_cfg("MEMORY_PROMPT_BUDGET_ITEMS", 16))

    facts_list = facts_db.get_prompt_worthy_facts(
        person_id, limit=fact_pool, topic_tokens=topic_tokens, mute_terms=mute_terms
    )
    interests_list = interests_db.get_interests_for_prompt(
        person_id, limit=interest_pool, topic_tokens=topic_tokens
    )

    items: list[MemoryItem] = []
    with _inline_budget():
        for f in facts_list:
            rel = _relevance(topic_tokens, _fact_text(f), cap)
            items.append(MemoryItem("fact", _fact_base(f) * fw + boost * rel, rel, f))
        for it in interests_list:
            rel = _relevance(topic_tokens, _interest_text(it), cap)
            items.append(MemoryItem("interest", _interest_base(it) * iw + boost * rel, rel, it))

    items.sort(key=lambda m: m.score, reverse=True)
    budget = max(0, int(budget))
    chosen = items[:budget]

    # Fact-quota floor (field 2026-08-01: interests score a flat ~0.85 while facts
    # carry age penalties, so 15 of 16 slots went to interests and Rex "forgot" the
    # person's favorite movie, job, hometown, and pet mid-conversation). Guarantee
    # the top-N facts a seat: evict the LOWEST-scored chosen interests to make room.
    # Topic-RELEVANT interests are never evicted — the floor exists to stop
    # GENERIC interests crowding facts out, not to override live relevance.
    min_facts = int(_cfg("MEMORY_RETRIEVAL_MIN_FACTS", 6))
    fact_items = [m for m in items if m.kind == "fact"]
    want = min(min_facts, len(fact_items), budget)
    have = sum(1 for m in chosen if m.kind == "fact")
    if have < want:
        chosen_set = {id(m) for m in chosen}
        missing = [m for m in fact_items if id(m) not in chosen_set][: want - have]
        evictable = [m for m in chosen if m.kind == "interest" and m.overlap <= 0.0]
        evict = {id(m) for m in evictable[len(evictable) - len(missing):]}
        missing = missing[: len(evict)]
        chosen = [m for m in chosen if id(m) not in evict] + missing
        chosen.sort(key=lambda m: m.score, reverse=True)

    return {
        "facts": [m.row for m in chosen if m.kind == "fact"],
        "interests": [m.row for m in chosen if m.kind == "interest"],
    }
