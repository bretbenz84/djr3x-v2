"""
memory/people.py — Person identity, biometric lookup, and relationship management.

Metric note:
  - Face matching uses Euclidean distance (dlib standard, lower = better match).
  - Voice matching uses cosine similarity (Resemblyzer standard, higher = better match).
  These are intentionally different — using the same metric for both is a common bug.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from memory import database as db

_log = logging.getLogger(__name__)

# Tables that hold per-person data (excludes personality_settings, which is global).
_PERSON_TABLES = [
    "biometrics",
    "person_facts",
    "person_qa",
    "conversations",
    "person_events",
]

# person_relationships uses from_person_id/to_person_id rather than person_id,
# so it can't share the simple _PERSON_TABLES delete path.
_RELATIONSHIP_TABLE = "person_relationships"

# Tier order used for antagonism cap comparisons.
_TIER_ORDER = ["stranger", "acquaintance", "friend", "close_friend", "best_friend"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_blob(encoding: np.ndarray) -> bytes:
    # Store as float32 — sufficient precision, half the size of float64.
    return encoding.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _compute_tier(familiarity: float, antagonism: float) -> str:
    """Derive friendship_tier from familiarity score, then apply any antagonism cap."""
    tier = "stranger"
    for name, (low, high) in config.FAMILIARITY_TIERS.items():
        if low <= familiarity < high:
            tier = name
            break

    # ANTAGONISM_TIER_CAPS is already highest-threshold-first; first match wins.
    for threshold, cap in sorted(config.ANTAGONISM_TIER_CAPS, reverse=True):
        if antagonism >= threshold:
            if _TIER_ORDER.index(tier) > _TIER_ORDER.index(cap):
                tier = cap
            break

    return tier


# ─────────────────────────────────────────────────────────────────────────────
# Biometric lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_by_face(encoding: np.ndarray) -> Optional[dict]:
    """
    Return the best-matching person record for a 128-dim dlib face encoding, or None.

    Uses Euclidean distance. Match is accepted only if distance is strictly below
    FACE_RECOGNITION_DISTANCE_THRESHOLD (default 0.6 — the dlib standard).
    """
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'face'"
    )
    best_id, best_dist = None, float("inf")
    for row in rows:
        stored = _from_blob(bytes(row["encoding"]))
        if stored.shape != encoding.shape:
            _log.warning("face encoding shape mismatch: stored %s vs query %s", stored.shape, encoding.shape)
            continue
        dist = float(np.linalg.norm(stored - encoding.astype(np.float32)))
        if dist < best_dist:
            best_dist = dist
            best_id = row["person_id"]

    if best_id is not None and best_dist < config.FACE_RECOGNITION_DISTANCE_THRESHOLD:
        return get_person(best_id)
    return None


def find_by_voice(embedding: np.ndarray) -> Optional[dict]:
    """
    Return the best-matching person record for a Resemblyzer voice embedding, or None.

    Uses cosine similarity. Match is accepted only if similarity is at or above
    SPEAKER_ID_SIMILARITY_THRESHOLD (default 0.75).
    """
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'voice'"
    )
    query = embedding.astype(np.float32)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    best_id, best_sim = None, -1.0
    for row in rows:
        stored = _from_blob(bytes(row["encoding"]))
        if stored.shape != query.shape:
            _log.warning("voice embedding shape mismatch: stored %s vs query %s", stored.shape, query.shape)
            continue
        stored_norm = stored / (np.linalg.norm(stored) + 1e-10)
        sim = float(np.dot(stored_norm, query_norm))
        if sim > best_sim:
            best_sim = sim
            best_id = row["person_id"]

    if best_id is not None and best_sim >= config.SPEAKER_ID_SIMILARITY_THRESHOLD:
        return get_person(best_id)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Person CRUD
# ─────────────────────────────────────────────────────────────────────────────

def enroll_person(name: str) -> Optional[int]:
    """Insert a new person row with defaults and return the new person_id."""
    now = _now()
    return db.execute(
        """
        INSERT INTO people
            (name, first_seen, last_seen, visit_count,
             familiarity_score, friendship_tier,
             warmth_score, antagonism_score, playfulness_score,
             curiosity_score, trust_score, net_relationship_score)
        VALUES (?, ?, ?, 0, 0.0, 'stranger', 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
        """,
        (name, now, now),
    )


def add_biometric(person_id: int, type: str, encoding: np.ndarray) -> Optional[int]:
    """Store a face or voice encoding as a BLOB. type must be 'face' or 'voice'."""
    return db.execute(
        "INSERT INTO biometrics (person_id, type, encoding, created_at) VALUES (?, ?, ?, ?)",
        (person_id, type, _to_blob(encoding), _now()),
    )


def get_person(person_id: int) -> Optional[dict]:
    """Return the full people row as a plain dict, or None if not found."""
    row = db.fetchone("SELECT * FROM people WHERE id = ?", (person_id,))
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Visit & familiarity tracking
# ─────────────────────────────────────────────────────────────────────────────

def update_visit(person_id: int) -> None:
    """
    Increment visit_count, update last_seen, apply the return-visit familiarity increment.

    days_known is derived at runtime (today − first_seen) and is not stored.
    """
    db.execute(
        "UPDATE people SET visit_count = visit_count + 1, last_seen = ? WHERE id = ?",
        (_now(), person_id),
    )
    update_familiarity(person_id, config.FAMILIARITY_INCREMENTS["return_visit"])


def update_familiarity(person_id: int, increment: float) -> None:
    """Add increment to familiarity_score (clamped to 1.0) and recalculate friendship_tier."""
    row = db.fetchone(
        "SELECT familiarity_score, antagonism_score FROM people WHERE id = ?",
        (person_id,),
    )
    if row is None:
        return
    new_score = min(1.0, row["familiarity_score"] + increment)
    new_tier = _compute_tier(new_score, row["antagonism_score"])
    db.execute(
        "UPDATE people SET familiarity_score = ?, friendship_tier = ? WHERE id = ?",
        (new_score, new_tier, person_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relationship scoring
# ─────────────────────────────────────────────────────────────────────────────

def update_relationship_scores(person_id: int, **kwargs: float) -> None:
    """
    Apply deltas to any combination of warmth, antagonism, playfulness, curiosity, trust.

    Each dimension is clamped to 0.0–1.0 after the delta is applied.
    net_relationship_score = (warmth − antagonism), clamped to −1.0–1.0.
    friendship_tier is re-evaluated whenever antagonism changes.
    """
    _valid = {"warmth", "antagonism", "playfulness", "curiosity", "trust"}
    deltas = {k: v for k, v in kwargs.items() if k in _valid}
    if not deltas:
        return

    row = db.fetchone(
        """SELECT warmth_score, antagonism_score, playfulness_score,
                  curiosity_score, trust_score, familiarity_score
           FROM people WHERE id = ?""",
        (person_id,),
    )
    if row is None:
        return

    def _apply(field: str, current: float) -> float:
        return min(1.0, max(0.0, current + deltas.get(field, 0.0)))

    warmth      = _apply("warmth",      row["warmth_score"])
    antagonism  = _apply("antagonism",  row["antagonism_score"])
    playfulness = _apply("playfulness", row["playfulness_score"])
    curiosity   = _apply("curiosity",   row["curiosity_score"])
    trust       = _apply("trust",       row["trust_score"])

    net = min(1.0, max(-1.0, warmth - antagonism))
    new_tier = _compute_tier(row["familiarity_score"], antagonism)

    db.execute(
        """UPDATE people
           SET warmth_score = ?, antagonism_score = ?, playfulness_score = ?,
               curiosity_score = ?, trust_score = ?, net_relationship_score = ?,
               friendship_tier = ?
           WHERE id = ?""",
        (warmth, antagonism, playfulness, curiosity, trust, net, new_tier, person_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory wipe
# ─────────────────────────────────────────────────────────────────────────────

def delete_person(person_id: int) -> None:
    """Delete all rows for a person across every person-related table."""
    for table in _PERSON_TABLES:
        db.execute(f"DELETE FROM {table} WHERE person_id = ?", (person_id,))
    db.execute(
        f"DELETE FROM {_RELATIONSHIP_TABLE} WHERE from_person_id = ? OR to_person_id = ?",
        (person_id, person_id),
    )
    db.execute("DELETE FROM people WHERE id = ?", (person_id,))


def delete_all_people() -> None:
    """
    Remove all rows from every person-related table.

    personality_settings is global (not per-person) and is left untouched.
    The database schema and empty tables remain intact.
    """
    for table in _PERSON_TABLES:
        db.execute(f"DELETE FROM {table}")
    db.execute(f"DELETE FROM {_RELATIONSHIP_TABLE}")
    db.execute("DELETE FROM people")
