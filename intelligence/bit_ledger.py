"""
intelligence/bit_ledger.py — persistent per-person cooldown for comedy bits.

Rex talks to the same person every day, and session-level anti-repeat can't see
yesterday: the haircut observation ran on Jul 31 AND Aug 2, "I made you" was
re-roasted twice the next afternoon, and the hydration bit played on both ends
of the weekend. Every offender came through the lean impulse path, so that's
where this ledger records and enforces.

Mechanics:
  * record()        — after a lean impulse is SPOKEN, store its topic signature
                      (quoted phrases + content words) in rex.db.
  * is_repeat()     — True when a freshly generated line re-runs a bit used with
                      this person within BIT_LEDGER_COOLDOWN_DAYS: a quoted
                      phrase matches, 2+ content words are shared with one prior
                      bit, or a single DISTINCTIVE word (7+ chars, e.g.
                      "hydration") recurs.
  * recent_topics() — short angle strings for the lean prompt's exclusion list,
                      so generation steers away instead of being vetoed after.

Follow-up-shaped cues (event follow-ups, open threads, celebrations …) are
exempted by the CALLER — a "how did the interview go?" is attentiveness, not a
bit. Fail-safe throughout: any error reads as "not a repeat" / "no exclusions".
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import config
from memory import rex_db

_log = logging.getLogger(__name__)

_QUOTED_PAT = re.compile(r"[\"“‘']([^\"“”‘’']{2,60})[\"”’']")
_TOKEN_PAT = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    # articles / pronouns / auxiliaries / droid-tic filler that would make every
    # pair of Rex lines "overlap"
    "the", "and", "for", "that", "this", "with", "you", "your", "yours", "our",
    "are", "was", "were", "has", "have", "had", "not", "but", "its", "it's",
    "his", "her", "their", "them", "they", "she", "him", "who", "what", "when",
    "where", "why", "how", "did", "does", "doing", "done", "just", "still",
    "like", "into", "out", "off", "over", "than", "then", "too", "very",
    "some", "one", "two", "all", "any", "either", "both", "most", "more",
    "about", "after", "before", "again", "there", "here", "which", "while",
    "because", "being", "gonna", "going", "got", "get", "gets", "let", "lets",
    "let's", "i'm", "i've", "i'll", "you're", "you've", "don't", "doesn't",
    "didn't", "can't", "won't", "isn't", "aren't", "wasn't", "that's", "it'd",
    "kind", "sort", "thing", "things", "little", "big", "good", "bad", "nice",
    "right", "yeah", "okay", "hey", "say", "said", "says", "tell", "know",
    "make", "makes", "made", "way", "lot", "bit", "actually", "really",
    "whole", "every", "even", "much", "many",
    # Rex's own tic vocabulary — shared by half his lines, useless as topic signal
    "rex", "droid", "circuits", "photoreceptors", "processing", "systems",
    "nominal", "organic", "organics", "human", "humans",
}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _cutoff_iso(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _cooldown_days() -> float:
    return float(getattr(config, "BIT_LEDGER_COOLDOWN_DAYS", 5.0))


def content_tokens(text: str) -> set[str]:
    """Content words of a Rex line, for the repeat check.

    Inline delivery tags are stripped FIRST (field 2026-08-05: five lull lines
    dropped in a row as "repeats a recent bit", three of them wrongly). A line
    carrying "[curious]" tokenized `curious` as CONTENT — and at 7 characters that
    clears BIT_LEDGER_DISTINCTIVE_LEN, whose rule is "one shared distinctive word
    = same bit". So a single stored bit containing the word "curious" silently
    blocked EVERY future line tagged [curious], regardless of subject: a news
    offer about an AWS outage and a question about the best thing they'd eaten
    were both refused as repeats of each other. Tags describe how a line is
    SPOKEN, never what it is about, so they must never reach the comparison.
    """
    try:
        from utils.audio_tags import strip_audio_tags
        text = strip_audio_tags(text or "")
    except Exception:
        text = re.sub(r"\[[^\]]{1,32}\]", " ", str(text or ""))
    return {
        t.strip("'")
        for t in _TOKEN_PAT.findall((text or "").lower())
        if len(t.strip("'")) >= 3 and t.strip("'") not in _STOPWORDS
    }


def quoted_phrases(text: str) -> list[str]:
    """Quoted spans, normalized — a bit that QUOTES something ("I made you") is
    keyed on the quote itself, since the surrounding joke re-words every time."""
    out = []
    for m in _QUOTED_PAT.findall(text or ""):
        norm = re.sub(r"\s+", " ", m).strip().lower()
        if norm and len(norm.split()) <= 8:
            out.append(norm)
    return out


def _topic_of(text: str, quoted: list[str], tokens: set[str]) -> str:
    """A short human-readable angle key for prompt exclusion lists."""
    if quoted:
        return " / ".join(f"'{q}'" for q in quoted[:2])
    ordered = [
        t for t in _TOKEN_PAT.findall((text or "").lower()) if t in tokens
    ]
    seen: list[str] = []
    for t in ordered:
        if t not in seen:
            seen.append(t)
        if len(seen) >= 4:
            break
    return " ".join(seen)


def record(person_id: Optional[int], text: str, source: str = "") -> None:
    """File a SPOKEN bit under this person. Prunes expired rows on the way."""
    if not bool(getattr(config, "BIT_LEDGER_ENABLED", True)):
        return
    line = (text or "").strip()
    if person_id is None or not line:
        return
    try:
        quoted = quoted_phrases(line)
        tokens = content_tokens(line)
        if not quoted and len(tokens) < 2:
            return  # nothing distinctive enough to key on
        topic = _topic_of(line, quoted, tokens)
        rex_db.execute(
            "DELETE FROM bit_ledger WHERE spoken_at < ?",
            (_cutoff_iso(_cooldown_days() * 2),),
        )
        rex_db.execute(
            "INSERT INTO bit_ledger (person_id, topic, quoted, tokens, source, spoken_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                int(person_id),
                topic[:120],
                json.dumps(quoted, ensure_ascii=False),
                json.dumps(sorted(tokens), ensure_ascii=False),
                (source or "")[:40],
                _now_iso(),
            ),
        )
    except Exception as exc:
        _log.debug("[bit_ledger] record failed: %s", exc)


def _rows_within_cooldown(person_id: int) -> list:
    return rex_db.fetchall(
        "SELECT topic, quoted, tokens FROM bit_ledger "
        "WHERE person_id = ? AND spoken_at >= ? ORDER BY spoken_at DESC LIMIT 60",
        (int(person_id), _cutoff_iso(_cooldown_days())),
    )


def is_repeat(person_id: Optional[int], text: str) -> bool:
    """True when `text` re-runs a bit used with this person inside the cooldown."""
    if not bool(getattr(config, "BIT_LEDGER_ENABLED", True)):
        return False
    line = (text or "").strip()
    if person_id is None or not line:
        return False
    try:
        quoted = set(quoted_phrases(line))
        tokens = content_tokens(line)
        min_overlap = int(getattr(config, "BIT_LEDGER_MIN_OVERLAP", 2))
        distinctive_len = int(getattr(config, "BIT_LEDGER_DISTINCTIVE_LEN", 7))
        for row in _rows_within_cooldown(int(person_id)):
            try:
                prior_quoted = set(json.loads(row["quoted"] or "[]"))
                prior_tokens = set(json.loads(row["tokens"] or "[]"))
            except Exception:
                continue
            if quoted & prior_quoted:
                return True
            overlap = tokens & prior_tokens
            if len(overlap) >= min_overlap:
                return True
            if any(len(t) >= distinctive_len for t in overlap):
                return True
    except Exception as exc:
        _log.debug("[bit_ledger] is_repeat failed: %s", exc)
    return False


def recent_topics(person_id: Optional[int], limit: int = 6) -> list[str]:
    """Distinct recent angle keys, newest first — for the lean prompt's
    'already used, find something new' exclusion list."""
    if person_id is None or not bool(getattr(config, "BIT_LEDGER_ENABLED", True)):
        return []
    try:
        out: list[str] = []
        for row in _rows_within_cooldown(int(person_id)):
            topic = str(row["topic"] or "").strip()
            if topic and topic not in out:
                out.append(topic)
            if len(out) >= int(limit):
                break
        return out
    except Exception as exc:
        _log.debug("[bit_ledger] recent_topics failed: %s", exc)
        return []
