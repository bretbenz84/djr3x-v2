"""
episodic_recall.py — Rex's first-person episodic memory, read back into behavior.

PHASE 2 (the read side; episodes.py is the write side / "diary"). Where episodes.py
ONLY captures, this module ranks, clusters, and shapes those rows into prompt-ready
material so Rex can reference his own past — "since I was last on…" continuity and
"things I remember doing with you" callbacks.

Scope (deliberately narrow, see config notes):
  • conversation_summary is EXCLUDED — people.db owns per-person conversation recall
    (the "Last conversation" block + nostalgia hook). rex.db recall is the
    EXPERIENTIAL layer: things Rex did/saw + cross-session continuity.
  • scenes are noisy and repetitive → CLUSTERED to a single ambient "vibe", never
    surfaced individually.

Everything is gated by `config.EPISODIC_RECALL_ENABLED` (a SEPARATE switch from
capture) and inherits rex_db's test-runner write/read suppression. Every public
function is failure-safe (swallow + log; recall must never crash the robot) and
returns an empty/None result when disabled.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from memory import episodes, rex_db

_log = logging.getLogger(__name__)

# Boilerplate prefixes the scene captioner prepends; stripped before clustering.
_SCENE_PREFIX_RE = re.compile(
    r"^\s*(?:when i powered up,? i saw|i looked around(?: the room)?)\s*[:\-]?\s*",
    re.IGNORECASE,
)
# Low-signal words to ignore when distilling a scene "vibe".
_SCENE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "is", "are",
    "was", "were", "this", "that", "these", "those", "it", "its", "room", "scene",
    "space", "area", "shows", "show", "features", "featuring", "appears", "various",
    "some", "there", "here", "view", "image", "depicts", "depicting", "showing",
    "looks", "look", "seen", "visible", "no", "people", "person", "man", "woman",
    "sitting", "sits", "seated", "front", "while", "into", "from", "for", "by",
})


def _enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "EPISODIC_RECALL_ENABLED", False))
    except Exception:
        return False


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def _now() -> datetime:
    return datetime.now()


def _parse_ts(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _current_session() -> Optional[str]:
    """The current run's session id, so recall can skip episodes from THIS run (Rex
    shouldn't 'remember' what he saw two seconds ago as if it were a past visit)."""
    try:
        return episodes._session()
    except Exception:
        return None


def _kind_weight(kind: str) -> float:
    weights = _cfg("EPISODIC_RECALL_KIND_WEIGHTS", {}) or {}
    default = float(_cfg("EPISODIC_RECALL_DEFAULT_KIND_WEIGHT", 0.5))
    try:
        return float(weights.get(kind, default))
    except Exception:
        return default


def _recency_factor(created_at, now: datetime) -> float:
    ts = _parse_ts(created_at)
    if ts is None:
        return 0.5
    halflife = float(_cfg("EPISODIC_RECALL_RECENCY_HALFLIFE_DAYS", 5.0)) or 5.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / halflife)


def _score(row, now: datetime) -> float:
    try:
        salience = float(row["salience"])
    except Exception:
        salience = 0.5
    return salience * _recency_factor(row["created_at"], now) * _kind_weight(row["kind"])


def _topic_overlap(row, topic_tokens) -> int:
    """How many live-topic words this episode's summary/detail mention (stemmed both
    sides). The relevance signal that lifts an on-topic callback over a fresher but
    unrelated one."""
    if not topic_tokens:
        return 0
    try:
        from memory import text_match
        text = f"{row['summary'] or ''} {row['detail'] or ''}"
        return text_match.overlap_count(text, topic_tokens)
    except Exception:
        return 0


def _topic_bonus(row, topic_tokens) -> float:
    if not topic_tokens:
        return 0.0
    boost = float(_cfg("EPISODIC_RECALL_TOPIC_BOOST", 0.3))
    cap = int(_cfg("MEMORY_TOPIC_RELEVANCE_MAX_MATCHES", 3))
    return boost * min(_topic_overlap(row, topic_tokens), cap)


def _norm_summary(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".")


# ── Reads ───────────────────────────────────────────────────────────────────────

def _fetch_window(lookback_days: int, exclude_session: Optional[str]) -> list:
    cutoff = (_now() - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d %H:%M:%S")
    if exclude_session:
        return rex_db.fetchall(
            "SELECT * FROM rex_episodes WHERE created_at >= ? AND "
            "(session_id IS NULL OR session_id != ?) ORDER BY created_at DESC",
            (cutoff, exclude_session),
        )
    return rex_db.fetchall(
        "SELECT * FROM rex_episodes WHERE created_at >= ? ORDER BY created_at DESC",
        (cutoff,),
    )


# ── Scene clustering ──────────────────────────────────────────────────────────────

def _clean_scene(summary: str) -> str:
    return _SCENE_PREFIX_RE.sub("", summary or "").strip()


def cluster_scenes(scene_rows: list, *, top_k: int = 4) -> Optional[str]:
    """Collapse repeated scene captions into ONE ambient 'vibe' line, data-driven
    from the most frequent significant words. Returns None when there's nothing."""
    cleaned = [_clean_scene(r["summary"]) for r in scene_rows if (r["summary"] or "").strip()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None
    counts: Counter = Counter()
    for text in cleaned:
        words = {
            w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in _SCENE_STOPWORDS
        }
        counts.update(words)
    top = [w for w, _ in counts.most_common(top_k)]
    if not top:
        # Fall back to the most recent caption if no salient keywords surfaced.
        return cleaned[0]
    descriptor = ", ".join(top)
    if len(cleaned) >= 2:
        return f"the space keeps reading as: {descriptor}"
    return f"the space read as: {descriptor}"


# ── Ranking + dedupe ──────────────────────────────────────────────────────────────

def rank_episodes(rows: list, *, now: Optional[datetime] = None, topic_tokens=None) -> list:
    """Sort experiential rows by score (desc). Excludes scenes (clustered separately)
    and zero-weight kinds (e.g. conversation_summary). When `topic_tokens` is given, an
    episode that connects to the live topic is lifted above a fresher unrelated one."""
    now = now or _now()
    scored = []
    for r in rows:
        kind = r["kind"]
        if kind == "scene":
            continue
        if _kind_weight(kind) <= 0.0:
            continue
        scored.append((_score(r, now) + _topic_bonus(r, topic_tokens), r))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [r for _, r in scored]


def _dedupe(rows: list, *, limit: int) -> list:
    """Collapse repeats (same kind + person + near-identical summary) — e.g. three
    'I wished Bret a happy birthday' rows become one."""
    seen: set = set()
    out: list = []
    for r in rows:
        key = (r["kind"], r["person_id"], _norm_summary(r["summary"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out


# ── Public recall shapes ──────────────────────────────────────────────────────────

def session_recap(
    *, lookback_days: Optional[int] = None, max_highlights: int = 2,
    exclude_session: Optional[str] = None, exclude_sensitive: bool = True,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Cross-session "since I was last on" digest: a clustered scene vibe plus the top
    couple of experiential episodes. Returns raw first-person material for the LLM to
    rephrase, or None when disabled / nothing worth recalling.

    `exclude_sensitive` (default True): sensitive kinds (emotional check-ins, boundaries)
    are kept OUT of this out-loud musing so they can't be overheard."""
    if not _enabled():
        return None
    try:
        now = now or _now()
        lookback = lookback_days if lookback_days is not None else int(_cfg("EPISODIC_RECALL_LOOKBACK_DAYS", 14))
        exclude = exclude_session if exclude_session is not None else _current_session()
        rows = _fetch_window(lookback, exclude)
        if not rows:
            return None

        scenes = [r for r in rows if r["kind"] == "scene"]
        vibe = cluster_scenes(scenes)
        candidates = rows
        if exclude_sensitive:
            sensitive = set(_cfg("EPISODIC_RECALL_SENSITIVE_KINDS", ()) or ())
            candidates = [r for r in rows if r["kind"] not in sensitive]
        highlights = _dedupe(rank_episodes(candidates, now=now), limit=max_highlights)

        parts: list[str] = []
        if vibe:
            parts.append(vibe)
        for r in highlights:
            s = (r["summary"] or "").strip()
            if s:
                parts.append(s.rstrip("."))
        if not parts:
            return None
        return "; ".join(parts) + "."
    except Exception as exc:
        _log.debug("session_recap failed: %s", exc)
        return None


def recent_conversation_topics(
    person_id: Optional[int], *, limit: int = 4, lookback_days: Optional[int] = None,
    exclude_current_session: bool = True,
) -> list[str]:
    """What Rex and this person ALREADY talked about in recent PRIOR runs — the newest
    conversation_summary episodes (current run excluded), de-duped, newest first. This is the
    "don't re-open the same thing every boot" signal (distinct from person_episodes, which is
    nostalgia callbacks). Returns [] when disabled / no person / nothing stored."""
    if not _enabled() or not isinstance(person_id, int):
        return []
    try:
        lookback = lookback_days if lookback_days is not None else int(_cfg("EPISODIC_RECALL_LOOKBACK_DAYS", 14))
        cutoff = (_now() - timedelta(days=int(lookback))).strftime("%Y-%m-%d %H:%M:%S")
        exclude = _current_session() if exclude_current_session else None
        if exclude:
            rows = rex_db.fetchall(
                "SELECT summary FROM rex_episodes WHERE person_id = ? AND kind = 'conversation_summary' "
                "AND created_at >= ? AND (session_id IS NULL OR session_id != ?) "
                "ORDER BY created_at DESC LIMIT ?",
                (person_id, cutoff, exclude, max(1, int(limit)) * 4),
            )
        else:
            rows = rex_db.fetchall(
                "SELECT summary FROM rex_episodes WHERE person_id = ? AND kind = 'conversation_summary' "
                "AND created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (person_id, cutoff, max(1, int(limit)) * 4),
            )
        out: list[str] = []
        seen: set[str] = set()
        for r in rows or []:
            s = (r["summary"] or "").strip().rstrip(".")
            if not s:
                continue
            key = _norm_summary(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= max(1, int(limit)):
                break
        return out
    except Exception as exc:
        _log.debug("recent_conversation_topics failed: %s", exc)
        return []


def person_episodes(
    person_id: Optional[int], *, limit: int = 3, lookback_days: Optional[int] = None,
    exclude_sensitive: bool = False, now: Optional[datetime] = None, topic_tokens=None,
) -> list[str]:
    """Ranked experiential callbacks for ONE person ("I made you laugh", "we played
    trivia") — first-person summary strings. Excludes scenes, person_seen (low value),
    and conversation_summary. With `exclude_sensitive`, also drops sensitive kinds
    (emotional check-ins, boundaries). When `topic_tokens` is given, callbacks that
    connect to the live topic rank first (so Rex recalls the FITTING moment, not a random
    one) — falling back to recency/salience when nothing connects. Returns [] when
    disabled / no person."""
    if not _enabled() or not isinstance(person_id, int):
        return []
    try:
        now = now or _now()
        lookback = lookback_days if lookback_days is not None else int(_cfg("EPISODIC_RECALL_LOOKBACK_DAYS", 14))
        cutoff = (now - timedelta(days=int(lookback))).strftime("%Y-%m-%d %H:%M:%S")
        rows = rex_db.fetchall(
            "SELECT * FROM rex_episodes WHERE person_id = ? AND created_at >= ? "
            "ORDER BY created_at DESC",
            (person_id, cutoff),
        )
        skip = {"person_seen"}
        if exclude_sensitive:
            skip |= set(_cfg("EPISODIC_RECALL_SENSITIVE_KINDS", ()) or ())
        ranked = rank_episodes(
            [r for r in rows if r["kind"] not in skip], now=now, topic_tokens=topic_tokens
        )
        deduped = _dedupe(ranked, limit=limit)
        return [(r["summary"] or "").strip().rstrip(".") for r in deduped if (r["summary"] or "").strip()]
    except Exception as exc:
        _log.debug("person_episodes failed: %s", exc)
        return []


# ── Maintenance ───────────────────────────────────────────────────────────────────

def prune() -> int:
    """Cap stored scene episodes at EPISODIC_RECALL_SCENE_RETENTION (newest kept);
    older scenes are deleted. Non-scene episodes are never pruned here. Returns the
    number of rows deleted (0 when gated/failed). Safe to call at shutdown."""
    if rex_db.writes_suppressed():
        return 0
    retention = int(_cfg("EPISODIC_RECALL_SCENE_RETENTION", 40))
    if retention < 0:
        return 0
    try:
        before = rex_db.fetchone("SELECT COUNT(*) AS n FROM rex_episodes WHERE kind='scene'")
        n_before = int(before["n"]) if before else 0
        if n_before <= retention:
            return 0
        rex_db.execute(
            "DELETE FROM rex_episodes WHERE kind='scene' AND id NOT IN "
            "(SELECT id FROM rex_episodes WHERE kind='scene' ORDER BY created_at DESC LIMIT ?)",
            (retention,),
        )
        return max(0, n_before - retention)
    except Exception as exc:
        _log.debug("prune failed: %s", exc)
        return 0
