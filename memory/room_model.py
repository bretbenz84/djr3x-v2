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
    return str(value or "").strip().lower()


def record_objects(objects) -> None:
    """Upsert the currently-visible objects into the room baseline: bump last_seen +
    sighting_count for labels Rex has seen before, insert new ones with first_seen=now.
    De-duped per label per call (the same label in two positions counts once)."""
    if _suppressed() or not objects:
        return
    rex_db.ensure_schema()  # idempotent; creates room_objects on first use / older rex.db
    now = _now_iso()
    seen: set[str] = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = _clean_label(obj.get("label"))
        if not label or label in seen:
            continue
        seen.add(label)
        bucket = str(obj.get("position") or "").strip() or "unknown"
        try:
            rex_db.execute(
                "INSERT INTO room_objects "
                "(label, location_bucket, first_seen, last_seen, sighting_count) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(label) DO UPDATE SET "
                "  location_bucket = excluded.location_bucket, "
                "  last_seen = excluded.last_seen, "
                "  sighting_count = sighting_count + 1",
                (label, bucket, now, now),
            )
        except Exception as exc:
            _log.debug("room_model.record_objects failed for %r: %s", label, exc)


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
