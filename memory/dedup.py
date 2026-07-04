"""
memory/dedup.py — fuzzy duplicate detection + consolidation for the content stores.

Interests and events used to dedup on an EXACT lower(name) match, so every LLM
paraphrase forked a new row ("R3X droid" / "building an R3X droid" / "Droid
Development"; "camping trip" x4). This module provides:

  * normalized token matching (`interest_match` / `event_match`) used at WRITE time so
    a near-duplicate updates the existing row instead of forking, and
  * a one-time `consolidate_all()` pass (run once via PRAGMA user_version in
    database.verify_schema) that collapses the duplicates already in the DB.

Matching is deliberately CONSERVATIVE — it merges obvious paraphrases (equal token
sets, multi-token containment, very-high SequenceMatcher) but leaves genuinely
distinct interests apart. Over-merging two real interests is worse than leaving a
stray duplicate, so the thresholds lean strict.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from memory import database as db

_log = logging.getLogger(__name__)

# Filler words that carry no interest/event identity. Dropping them lets
# "building an R3X droid" reduce to the same token set as "R3X droid".
_STOP = {
    "a", "an", "the", "my", "your", "our", "his", "her", "their", "theirs",
    "of", "for", "and", "to", "into", "about", "with", "on", "in", "at",
    "building", "build", "making", "make", "doing", "do", "some", "stuff",
    "thing", "things", "trip", "trips",
}

# An interest name that LEADS with one of these object/subject pronouns is almost
# certainly a mis-parsed conversational fragment ("him sassy", "them again"), not a
# real interest. Possessives (my/your/his/her) are deliberately EXCLUDED — they often
# precede a real noun ("my robots") that is a salvageable interest.
_JUNK_LEAD = {"him", "them", "me", "it", "they", "us"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(name: str) -> list[str]:
    """Lowercase alnum tokens with light singular stemming, stopwords removed."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall((name or "").lower()):
        tok = raw[:-1] if (raw.endswith("s") and len(raw) > 3) else raw
        if tok and tok not in _STOP:
            out.append(tok)
    return out


def _token_set(name: str) -> frozenset[str]:
    return frozenset(_tokens(name))


def _norm_str(name: str) -> str:
    return " ".join(sorted(_tokens(name)))


def looks_like_junk_interest(name: str) -> bool:
    """True for clearly mis-parsed interest fragments (pronoun-led, or no real token)."""
    raw = (name or "").strip().lower()
    if not raw:
        return True
    first = (_TOKEN_RE.findall(raw) or [""])[0]
    if first in _JUNK_LEAD:
        return True
    # Nothing left after stopword removal → not a nameable interest.
    return not _tokens(name)


def names_match(a: str, b: str, *, ratio: float = 0.88) -> bool:
    """Conservative fuzzy match between two interest/event names."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        # All-stopword names: fall back to exact normalized comparison.
        return (a or "").strip().lower() == (b or "").strip().lower()
    if sa == sb:
        return True
    # Multi-token containment: "r3x droid" ⊂ "r3x droid encoder build".
    if len(sa) >= 2 and len(sb) >= 2 and (sa <= sb or sb <= sa):
        return True
    return SequenceMatcher(None, _norm_str(a), _norm_str(b)).ratio() >= ratio


def interest_match(name: str, rows: list[dict]) -> Optional[dict]:
    """Return the existing interest row a new `name` should fold into, or None."""
    for row in rows:
        if names_match(name, row.get("name") or ""):
            return row
    return None


def event_match(name: str, rows: list[dict]) -> Optional[dict]:
    """Return the existing event row a new `name` should fold into, or None."""
    for row in rows:
        if names_match(name, row.get("event_name") or ""):
            return row
    return None


# ── One-time consolidation of pre-existing duplicates ───────────────────────────

_STRENGTH_RANK = {"low": 1, "medium": 2, "high": 3}


def _cluster(rows: list[dict], name_key: str) -> list[list[dict]]:
    """Greedily group rows whose names match into clusters."""
    clusters: list[list[dict]] = []
    for row in rows:
        name = row.get(name_key) or ""
        placed = False
        for cluster in clusters:
            if names_match(name, cluster[0].get(name_key) or ""):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])
    return clusters


def consolidate_person_interests(person_id: int) -> int:
    """Collapse duplicate interests for one person. Returns rows removed."""
    rows = [dict(r) for r in db.fetchall(
        "SELECT * FROM person_interests WHERE person_id = ? ORDER BY id", (int(person_id),)
    )]
    removed = 0
    for cluster in _cluster(rows, "name"):
        if len(cluster) < 2:
            continue
        # Survivor: strongest, then most confident, then most recent.
        survivor = max(cluster, key=lambda r: (
            _STRENGTH_RANK.get((r.get("interest_strength") or "low"), 1),
            float(r.get("confidence") or 0.0),
            str(r.get("last_mentioned_at") or r.get("first_mentioned_at") or ""),
        ))
        # Display name: shortest (fewest tokens) member reads most canonical.
        display = min(
            (r.get("name") or "" for r in cluster),
            key=lambda n: (len(_tokens(n)) or 99, len(n)),
        ) or survivor.get("name")
        strength = max(
            (r.get("interest_strength") or "low" for r in cluster),
            key=lambda s: _STRENGTH_RANK.get(s, 1),
        )
        confidence = max(float(r.get("confidence") or 0.0) for r in cluster)
        notes = "; ".join(sorted({
            (r.get("notes") or "").strip() for r in cluster if (r.get("notes") or "").strip()
        }))[:500]
        first_seen = min(
            (str(r.get("first_mentioned_at") or "") for r in cluster if r.get("first_mentioned_at")),
            default=survivor.get("first_mentioned_at"),
        )
        last_seen = max(
            (str(r.get("last_mentioned_at") or "") for r in cluster if r.get("last_mentioned_at")),
            default=survivor.get("last_mentioned_at"),
        )
        db.execute(
            """UPDATE person_interests
               SET name = ?, interest_strength = ?, confidence = ?,
                   notes = COALESCE(NULLIF(?, ''), notes),
                   first_mentioned_at = ?, last_mentioned_at = ?
               WHERE id = ?""",
            (display, strength, confidence, notes, first_seen, last_seen, int(survivor["id"])),
        )
        for row in cluster:
            if int(row["id"]) != int(survivor["id"]):
                db.execute("DELETE FROM person_interests WHERE id = ?", (int(row["id"]),))
                removed += 1
    return removed


def consolidate_person_events(person_id: int) -> int:
    """Collapse duplicate OPEN events for one person. Returns rows removed."""
    rows = [dict(r) for r in db.fetchall(
        """SELECT * FROM person_events
           WHERE person_id = ?
             AND followed_up = FALSE
             AND COALESCE(status, 'planned') = 'planned'
           ORDER BY id""",
        (int(person_id),),
    )]
    removed = 0
    for cluster in _cluster(rows, "event_name"):
        if len(cluster) < 2:
            continue
        # Survivor: prefer a dated event, then the earliest mentioned.
        survivor = min(cluster, key=lambda r: (
            0 if r.get("event_date") else 1,
            str(r.get("mentioned_at") or ""),
        ))
        notes = "; ".join(sorted({
            (r.get("event_notes") or "").strip() for r in cluster if (r.get("event_notes") or "").strip()
        }))[:500]
        date = next((r.get("event_date") for r in cluster if r.get("event_date")), survivor.get("event_date"))
        db.execute(
            """UPDATE person_events
               SET event_date = COALESCE(?, event_date),
                   event_notes = COALESCE(NULLIF(?, ''), event_notes)
               WHERE id = ?""",
            (date, notes, int(survivor["id"])),
        )
        for row in cluster:
            if int(row["id"]) != int(survivor["id"]):
                db.execute("DELETE FROM person_events WHERE id = ?", (int(row["id"]),))
                removed += 1
    return removed


def consolidate_all() -> dict:
    """Run the duplicate-collapse pass across every person. Returns counts removed."""
    people = db.fetchall("SELECT id FROM people")
    interests_removed = 0
    events_removed = 0
    for row in people:
        pid = int(row["id"])
        try:
            interests_removed += consolidate_person_interests(pid)
            events_removed += consolidate_person_events(pid)
        except Exception as exc:
            _log.debug("consolidate person_id=%s failed: %s", pid, exc)
    return {"interests_removed": interests_removed, "events_removed": events_removed}


def purge_low_quality() -> dict:
    """One-time: delete already-stored facts/interests the fact_quality gate now
    rejects, and blank junk interest notes. Idempotent (a clean DB deletes nothing).

    Runs WITHOUT a source utterance, so it cannot see the negation class (a bare
    'Coney Island' hometown, a fabricated pet 'Max') — those age out via their low
    inferred confidence + fast decay, or are removed by hand. It DOES clear the
    tautologies, first-person fragments, fiction scenes, and verbatim-question
    values/notes that carry a lexical signal."""
    from memory import fact_quality
    facts_removed = interests_removed = notes_cleaned = 0
    for r in [dict(x) for x in db.fetchall("SELECT * FROM person_facts")]:
        if fact_quality.reject_fact(r.get("category", "") or "", r.get("key", "") or "",
                                    r.get("value", "") or ""):
            db.execute("DELETE FROM person_facts WHERE id = ?", (int(r["id"]),))
            facts_removed += 1
    for r in [dict(x) for x in db.fetchall("SELECT * FROM person_interests")]:
        notes = r.get("notes", "") or ""
        if fact_quality.reject_interest(r.get("name", "") or "", notes):
            db.execute("DELETE FROM person_interests WHERE id = ?", (int(r["id"]),))
            interests_removed += 1
            continue
        cleaned = fact_quality.clean_interest_note(notes)
        if cleaned != notes.strip():
            db.execute("UPDATE person_interests SET notes = ? WHERE id = ?",
                       (cleaned, int(r["id"])))
            notes_cleaned += 1
    return {"facts_removed": facts_removed, "interests_removed": interests_removed,
            "notes_cleaned": notes_cleaned}
