"""
memory/recall.py — query-time memory recall for the reply path.

Field failure (2026-08-01 22:38 session): Bret quizzed Rex on things the DBs had
held for weeks — favorite movie, job, hometown, dog, camping, "what movie did I
watch last night" (a literal rex_episodes row from the night before) — and Rex
denied knowing ANY of it. Root cause was retrieval, not storage: the lean reply
prompt carried only the top-4 facts + top-4 interests by STATIC score (junk
relationship edges won those slots), no topic awareness, and rex_episodes were
never read at question time at all (episodes only fed greetings/callbacks).

This module is the fix's core:

  * stems-of-the-utterance topic tokens for every reply turn, so the fact/interest
    ranking that already supports `topic_tokens` finally receives them;
  * `is_memory_question()` — detects "do you remember / what's my / did I tell
    you / what do you know about me" shapes;
  * `search_episodes()` — query-time episodic recall over rex_episodes (topic
    overlap required, salience+recency ranked, dated);
  * `memory_question_lines()` — the RICH recall block injected only on a memory
    question: identity facts as key:value, interests WITH their notes, direct
    Q&A answers, relationship edges, and matching diary episodes with dates —
    plus the instruction to answer FROM it and to admit a genuine blank honestly.

Fail-safe by construction: every public function returns an empty value on any
error so a broken DB can never break a reply.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

_log = logging.getLogger(__name__)


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


# ── Topic tokens for the current utterance ──────────────────────────────────────

# Question-machinery and filler stems that carry no topic signal — without this,
# "What movie did I WATCH LAST NIGHT?" matched every episode containing "did"
# (three 'I did an impersonation' rows outranked the actual movie memory).
_TOKEN_STOPSTEMS = {
    "did", "does", "was", "were", "are", "have", "has", "had", "will", "would",
    "could", "should", "can", "cant", "dont", "didnt", "you", "your", "yours",
    "the", "and", "for", "that", "this", "with", "what", "when", "where", "who",
    "whos", "how", "why", "which", "tell", "know", "say", "said", "told",
    "mention", "ever", "not", "about", "them", "they", "then", "than", "just",
    "like", "get", "got", "going", "gonna", "last", "night", "today", "yesterday",
    "tomorrow", "thing", "some", "any", "all", "one", "little", "bit", "also",
    "else", "okay", "yeah", "well", "really", "very", "much", "more",
}


def utterance_tokens(text: str) -> set:
    """Stemmed CONTENT tokens of what the person JUST said — the topic signal the
    per-silo rankers already accept, minus question-machinery words that would make
    everything match everything. Empty set on any failure."""
    try:
        from memory import text_match
        return {
            t for t in text_match.stems(text or "")
            if t not in _TOKEN_STOPSTEMS and text_match.stem(t) not in _TOKEN_STOPSTEMS
        }
    except Exception:
        return set()


# ── Memory-question detection ───────────────────────────────────────────────────
# Deliberately generous: a false positive only means a slightly bigger prompt for
# one turn; a false negative is the permanent-amnesia failure being fixed.
_MEMORY_QUESTION_RES = (
    re.compile(r"\bdo you (?:remember|recall|know)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is| are| was| were)? my\b", re.IGNORECASE),
    re.compile(r"\bwho(?:'s| is| was)? my\b", re.IGNORECASE),
    re.compile(r"\bwhen did i\b", re.IGNORECASE),
    re.compile(r"\bwhere did i\b", re.IGNORECASE),
    re.compile(r"\bwhat did i (?:say|tell|watch|do|mention)\b", re.IGNORECASE),
    re.compile(r"\bdid i (?:tell|mention|say|ever)\b", re.IGNORECASE),
    re.compile(r"\bhave i (?:ever\s+)?(?:told|mentioned|said)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:else\s+)?do you know about\b", re.IGNORECASE),
    re.compile(r"\btell me (?:.{0,20}\b)?about (?:me|myself|my)\b", re.IGNORECASE),
    re.compile(r"\bwhat movie did i\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:do i do|i do) for\b", re.IGNORECASE),
    re.compile(r"\bwhere do i (?:live|work|come from)\b", re.IGNORECASE),
    re.compile(r"\byou (?:have no|don't have any) memory\b", re.IGNORECASE),
    re.compile(r"\bwhat are my\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) my favorite\b", re.IGNORECASE),
)


def is_memory_question(text: str) -> bool:
    """True when the utterance is asking Rex what he remembers/knows about the
    speaker — the cue to inject the rich recall block."""
    t = " ".join(str(text or "").split())
    if not t:
        return False
    return any(rx.search(t) for rx in _MEMORY_QUESTION_RES)


# ── Query-time episodic recall ──────────────────────────────────────────────────

def search_episodes(
    topic_tokens,
    person_id: Optional[int] = None,
    limit: int = 4,
    lookback_days: Optional[int] = None,
) -> list[dict]:
    """rex_episodes rows that MENTION the live topic, best-first. Requires at least
    one stemmed-token overlap (no topic tokens → nothing; this is query-time recall,
    not ambient reminiscing). Excludes scene captions. [] on any failure."""
    if not topic_tokens:
        return []
    try:
        from datetime import datetime, timedelta, timezone
        from memory import rex_db, text_match
        days = int(lookback_days or _cfg("RECALL_EPISODE_LOOKBACK_DAYS", 120))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows = rex_db.fetchall(
            "SELECT * FROM rex_episodes WHERE created_at >= ? AND kind != 'scene' "
            "ORDER BY created_at DESC LIMIT 400",
            (cutoff,),
        )
        scored: list[tuple[float, dict]] = []
        for r in rows:
            row = dict(r)
            text = f"{row.get('summary') or ''} {row.get('detail') or ''}"
            overlap = text_match.overlap_count(text, topic_tokens)
            if overlap <= 0:
                continue
            # A person-scoped hit outranks a general one; salience breaks ties.
            person_bonus = 0.5 if (
                person_id is not None and row.get("person_id") == person_id
            ) else 0.0
            try:
                salience = float(row.get("salience") or 0.5)
            except (TypeError, ValueError):
                salience = 0.5
            scored.append((overlap + person_bonus + 0.25 * salience, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Collapse near-identical repeats (three 'I did an impersonation of Bret'
        # rows are ONE memory).
        seen: set = set()
        out: list[dict] = []
        for _, row in scored:
            key = re.sub(r"\s+", " ", str(row.get("summary") or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= max(0, int(limit)):
                break
        return out
    except Exception as exc:
        _log.debug("[recall] episode search failed: %s", exc)
        return []


def _episode_line(row: dict) -> str:
    date = str(row.get("created_at") or "")[:10]
    summary = " ".join(str(row.get("summary") or "").split())
    return f"[{date}] {summary}" if date else summary


# ── The rich recall block ───────────────────────────────────────────────────────

def _fact_pairs(person_id: int, topic_tokens, limit: int) -> list[str]:
    from memory import facts as facts_db
    out = []
    for f in facts_db.get_prompt_worthy_facts(
        person_id, limit=limit, topic_tokens=topic_tokens
    ):
        key = str(f.get("key") or "").replace("_", " ").strip()
        value = str(f.get("value") or "").strip()
        if key and value:
            hedge = " (inferred — hedge it)" if f.get("source") == "inferred" else ""
            out.append(f"{key}: {value}{hedge}")
    return out


def _interest_pairs(person_id: int, topic_tokens, limit: int) -> list[str]:
    from memory import interests as interests_db
    out = []
    for it in interests_db.get_interests_for_prompt(
        person_id, limit=limit, topic_tokens=topic_tokens
    ):
        name = str(it.get("name") or "").strip()
        if not name:
            continue
        notes = " ".join(str(it.get("notes") or "").split())
        out.append(f"{name} — {notes}" if notes else name)
    return out


def _qa_pairs(person_id: int, limit: int) -> list[str]:
    from memory import relationships as rel_db
    out = []
    for row in rel_db.get_qa_history(person_id)[-limit:]:
        row = dict(row)
        topic = str(row.get("question_key") or "").replace("_", " ").strip()
        answer = " ".join(str(row.get("answer_text") or "").split())
        if topic and answer:
            out.append(f"their answer about {topic}: \"{answer}\"")
    return out


def _relationship_lines(person_id: int) -> list[str]:
    from memory import social
    out = []
    for edge in social.get_outbound(person_id):
        label = str(edge.get("relationship") or "").strip()
        name = str(edge.get("to_name") or "").strip()
        if label and name:
            out.append(f"their {label}: {name}")
    return out


def memory_question_lines(person_id: Optional[int], utterance: str) -> list[str]:
    """The rich recall block for a direct memory question — [] when it isn't one,
    when there's no known person, or when recall is disabled. Each element is one
    system-prompt line (lean_brain renders them as '- ' bullets)."""
    if person_id is None or not bool(_cfg("RECALL_RICH_ENABLED", True)):
        return []
    if not is_memory_question(utterance):
        return []
    tokens = utterance_tokens(utterance)
    lines: list[str] = []
    try:
        facts = _fact_pairs(int(person_id), tokens,
                            int(_cfg("RECALL_RICH_FACT_LIMIT", 14)))
    except Exception as exc:
        _log.debug("[recall] facts failed: %s", exc)
        facts = []
    try:
        interests = _interest_pairs(int(person_id), tokens,
                                    int(_cfg("RECALL_RICH_INTEREST_LIMIT", 10)))
    except Exception as exc:
        _log.debug("[recall] interests failed: %s", exc)
        interests = []
    try:
        qa = _qa_pairs(int(person_id), int(_cfg("RECALL_RICH_QA_LIMIT", 8)))
    except Exception as exc:
        _log.debug("[recall] qa failed: %s", exc)
        qa = []
    try:
        rels = _relationship_lines(int(person_id))
    except Exception as exc:
        _log.debug("[recall] relationships failed: %s", exc)
        rels = []
    episodes = [
        _episode_line(r)
        for r in search_episodes(tokens, person_id=int(person_id),
                                 limit=int(_cfg("RECALL_EPISODE_LIMIT", 4)))
    ]
    if not (facts or interests or qa or rels or episodes):
        return []
    lines.append(
        "MEMORY QUESTION: they are asking what you REMEMBER. Everything below is "
        "your real memory of them — answer richly and specifically FROM it, in your "
        "own voice (never recite it as a list or mention databases/records). If the "
        "specific thing they asked about is genuinely NOT below, say so honestly and "
        "briefly — never invent a memory."
    )
    if facts:
        lines.append("Facts you know: " + "; ".join(facts) + ".")
    if rels:
        lines.append("People in their life: " + "; ".join(rels) + ".")
    if interests:
        lines.append("Their interests (with what you know about each): "
                     + " | ".join(interests) + ".")
    if qa:
        lines.append("Things they've told you directly: " + " | ".join(qa) + ".")
    if episodes:
        lines.append(
            "From your diary — dated things that actually happened (today is "
            "relative to these dates): " + " | ".join(episodes) + "."
        )
    return lines
