"""
memory/room_model.py — a persistent, per-object model of Rex's room (rex.db room_objects).

The local COCO stream (vision.scene.detect_objects_local → world_state.objects) is live but
stateless. This records which objects Rex has actually seen over time, so:
  * object-grounded curiosity can prefer what's NEW to the room (not the daily fixtures), and
  * Rex can notice a genuinely new object ("wait — when did that get here?") across sessions.

ONE row PER LABEL (sighting_count + first/last_seen), keyed on label — NOT (label, position) —
because the head moves and shifts an object's coarse position frame to frame; a chair is one
chair wherever it lands. Gated + test-suppressed exactly like memory.episodes: never creates or
writes a real rex.db under the test runner or when capture is disabled. Every call is fail-safe
(the room model is nice-to-have; it must never crash a turn). Screens/devices/people/animals are
filtered upstream by the detector and never reach this table.
"""

from __future__ import annotations

import logging
from datetime import datetime

import config
from memory import rex_db

_log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _enabled() -> bool:
    return bool(getattr(config, "EPISODIC_MEMORY_ENABLED", True)) and bool(
        getattr(config, "ROOM_MODEL_ENABLED", True)
    )


def _suppressed() -> bool:
    # No writes when capture is disabled OR the test runner would touch a real rex.db.
    return (not _enabled()) or rex_db.writes_suppressed()


def _clean_label(value) -> str:
    """Normalize a label for the per-label key. Exploration feeds OPEN-VOCAB
    names ("a half-disassembled droid arm"), not just COCO classes — collapse
    whitespace, drop leading articles, and truncate so free text can't bloat
    the key space."""
    s = " ".join(str(value or "").strip().lower().split())
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
            break
    return s[:60]


def record_objects(objects) -> None:
    """Upsert the currently-visible objects into the room baseline: bump last_seen +
    sighting_count for labels Rex has seen before, insert new ones with first_seen=now.
    De-duped per label per call (the same label in two positions counts once)."""
    if _suppressed() or not objects:
        return
    rex_db.ensure_schema()  # idempotent; creates room_objects on first use / older rex.db
    now = _now_iso()
    seen: set[str] = set()
    # RARITY-gated question queue (curiosity Phase 1): a label Rex has NEVER
    # logged before becomes an "ask about this" candidate — but only once the
    # room baseline exists (a fresh install must not queue questions about every
    # fixture it meets on day one; same guard the change-remark uses).
    prior = label_sightings([o.get("label") for o in objects if isinstance(o, dict)])
    min_baseline = int(getattr(config, "ROOM_CHANGE_MIN_BASELINE", 4))
    baseline_ok = established_count(min_baseline) >= min_baseline and _room_age_ok()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = _clean_label(obj.get("label"))
        if not label or label in seen:
            continue
        seen.add(label)
        bucket = str(obj.get("position") or "").strip() or "unknown"
        brand_new = baseline_ok and prior.get(label, 0) == 0
        try:
            rex_db.execute(
                "INSERT INTO room_objects "
                "(label, location_bucket, first_seen, last_seen, sighting_count, ask_status) "
                "VALUES (?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(label) DO UPDATE SET "
                "  location_bucket = excluded.location_bucket, "
                "  last_seen = excluded.last_seen, "
                "  sighting_count = sighting_count + 1",
                (label, bucket, now, now, "pending" if brand_new else None),
            )
        except Exception as exc:
            _log.debug("room_model.record_objects failed for %r: %s", label, exc)


def _room_age_ok() -> bool:
    """True once the room model is old enough to trust novelty. During the first
    day(s) of a fresh install, ordinary furniture trickles in as "never seen" —
    the fifth chair Rex meets is a fixture, not news. Age is measured from the
    OLDEST first_seen in the table."""
    days = float(getattr(config, "ROOM_QUESTION_MIN_ROOM_AGE_DAYS", 1.0))
    if days <= 0:
        return True
    try:
        row = rex_db.fetchone("SELECT MIN(first_seen) AS oldest FROM room_objects")
        oldest = str(row["oldest"] or "") if row else ""
        if not oldest:
            return False
        age = datetime.now() - datetime.strptime(oldest, "%Y-%m-%d %H:%M:%S")
        return age.total_seconds() >= days * 86400.0
    except Exception:
        return False


def label_sightings(labels) -> dict:
    """Map each given label -> its recorded sighting_count (0 when Rex has never logged it),
    in one query — so a caller can tell a brand-new object (low/zero count) from a fixture."""
    wanted = {_clean_label(x) for x in (labels or [])}
    wanted.discard("")
    if not wanted:
        return {}
    try:
        rows = rex_db.fetchall("SELECT label, sighting_count FROM room_objects")
    except Exception:
        return {}
    counts = {str(r["label"]).strip().lower(): int(r["sighting_count"] or 0) for r in rows}
    return {lbl: counts.get(lbl, 0) for lbl in wanted}


def established_count(min_sightings: int) -> int:
    """How many distinct labels are established fixtures (sighting_count >= min_sightings) —
    the baseline size. Change-detection consults this so a fresh install (no baseline yet)
    doesn't flood: Rex must KNOW the room before a new object can stand out."""
    try:
        row = rex_db.fetchone(
            "SELECT COUNT(*) AS n FROM room_objects WHERE sighting_count >= ?",
            (int(min_sightings),),
        )
        return int(row["n"]) if row else 0
    except Exception:
        return 0


# ── Learn-by-asking (curiosity Phase 1) ─────────────────────────────────────────
# The question queue lives ON the object rows (ask_status): 'pending' objects are
# what Rex hasn't asked a human about yet; answers land back here as human_name
# with a corroboration count. intelligence/room_questions.py owns the asking.

def pending_question(min_sightings: int = 2):
    """The best object to ask about: pending, CONFIRMED (>= min_sightings — a
    one-frame misread must not become a question), most recently seen first.
    Returns the row dict or None."""
    try:
        row = rex_db.fetchone(
            "SELECT * FROM room_objects WHERE ask_status = 'pending' "
            "AND sighting_count >= ? ORDER BY last_seen DESC LIMIT 1",
            (int(min_sightings),),
        )
        return dict(row) if row else None
    except Exception:
        return None


def note_question_asked(label: str) -> None:
    if _suppressed():
        return
    rex_db.execute(
        "UPDATE room_objects SET ask_status = 'asked', last_asked_at = ? WHERE label = ?",
        (_now_iso(), _clean_label(label)),
    )


def dismiss_question(label: str) -> None:
    """Close the question without an answer (person declined / changed subject)."""
    if _suppressed():
        return
    rex_db.execute(
        "UPDATE room_objects SET ask_status = 'dismissed' WHERE label = ?",
        (_clean_label(label),),
    )


def record_answer(label: str, name: str, note: str = "") -> bool:
    """Write a human-supplied identity for an object, with corroboration counting
    (the memory-poisoning defense): a matching repeat bumps name_confidence; a
    CONTRADICTING answer only replaces a single-source name (confidence <= 1) —
    a twice-confirmed name outranks one joker."""
    if _suppressed():
        return False
    label = _clean_label(label)
    name = str(name or "").strip()[:120]
    if not label or not name:
        return False
    try:
        row = rex_db.fetchone("SELECT human_name, name_confidence FROM room_objects WHERE label = ?", (label,))
        if row is None:
            return False
        current = str(row["human_name"] or "").strip()
        confidence = int(row["name_confidence"] or 0)
        if current and current.lower() == name.lower():
            confidence += 1                     # corroborated
        elif current and confidence > 1:
            _log.info("room_model: kept %r (conf %d) over new claim %r for %s",
                      current, confidence, name, label)
            rex_db.execute(
                "UPDATE room_objects SET ask_status = 'answered' WHERE label = ?", (label,))
            return False
        else:
            current, confidence = name, 1       # first claim, or replacing single-source
        rex_db.execute(
            "UPDATE room_objects SET human_name = ?, human_note = ?, "
            "name_confidence = ?, ask_status = 'answered' WHERE label = ?",
            (current, str(note or "").strip()[:400] or None, confidence, label),
        )
        return True
    except Exception as exc:
        _log.debug("room_model.record_answer failed: %s", exc)
        return False


def human_label(label: str):
    """The human-given name for a detector label ("potted plant" -> "the sourdough
    starter"), or None. Single-source (confidence 1) names are returned too — the
    CALLER decides whether to hedge ("someone told me it's ...")."""
    try:
        row = rex_db.fetchone(
            "SELECT human_name, name_confidence FROM room_objects WHERE label = ?",
            (_clean_label(label),),
        )
        if row and row["human_name"]:
            return {"name": str(row["human_name"]), "confidence": int(row["name_confidence"] or 0)}
    except Exception:
        pass
    return None
