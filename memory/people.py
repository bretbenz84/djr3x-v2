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
from memory.name_validation import (
    names_are_similar,
    normalize_person_name,
    normalized_name_key,
)

_log = logging.getLogger(__name__)

# Tables that hold per-person data (excludes personality_settings, which is global).
_PERSON_TABLES = [
    "biometrics",
    "person_facts",
    "person_qa",
    "conversations",
    "person_events",
    "person_aliases",
    "person_emotional_events",
    "person_conversation_boundaries",
    "person_preferences",
    "person_interests",
    "person_disposition_stats",
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


def _normalize_name(value: str) -> str:
    return normalized_name_key(value)


def _clean_display_name(value: str) -> Optional[str]:
    return normalize_person_name(value, allow_single=True)


def _person_aliases_available() -> bool:
    row = db.fetchone(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'person_aliases' LIMIT 1"
    )
    return bool(row)


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

    Uses Euclidean distance. A person's MULTIPLE stored encodings are aggregated to
    that person's CLOSEST one (so extra reference photos help recall instead of
    competing as separate candidates). The winner is accepted only if its distance is
    below FACE_RECOGNITION_DISTANCE_THRESHOLD (default 0.6) AND it beats the next-closest
    DIFFERENT person by at least FACE_RECOGNITION_MARGIN — otherwise the frame is treated
    as ambiguous (returns None) so the identity does not flip between two confusable
    faces (e.g. family members whose encodings both fall under 0.6 of the live face).
    """
    query = encoding.astype(np.float32)
    rows = db.fetchall(
        "SELECT person_id, encoding FROM biometrics WHERE type = 'face'"
    )
    per_person_best: dict[int, float] = {}
    for row in rows:
        stored = _from_blob(bytes(row["encoding"]))
        if stored.shape != query.shape:
            _log.warning("face encoding shape mismatch: stored %s vs query %s", stored.shape, query.shape)
            continue
        dist = float(np.linalg.norm(stored - query))
        pid = row["person_id"]
        if dist < per_person_best.get(pid, float("inf")):
            per_person_best[pid] = dist

    if not per_person_best:
        return None

    ranked = sorted(per_person_best.items(), key=lambda kv: kv[1])  # (person_id, dist) asc
    best_id, best_dist = ranked[0]
    second_dist = ranked[1][1] if len(ranked) > 1 else float("inf")

    if best_dist >= config.FACE_RECOGNITION_DISTANCE_THRESHOLD:
        return None
    margin = float(getattr(config, "FACE_RECOGNITION_MARGIN", 0.0) or 0.0)
    if (second_dist - best_dist) < margin:
        _log.info(
            "face match ambiguous: best=id%s d=%.3f vs next d=%.3f (margin %.3f < %.2f) — no match",
            best_id, best_dist, second_dist, second_dist - best_dist, margin,
        )
        return None
    return get_person(best_id)


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
    clean = _clean_display_name(name)
    if not clean:
        _log.warning("enroll_person rejected non-name candidate: %r", name)
        return None
    now = _now()
    new_id = db.execute(
        """
        INSERT INTO people
            (name, first_seen, last_seen, visit_count,
             familiarity_score, friendship_tier,
             warmth_score, antagonism_score, playfulness_score,
             curiosity_score, trust_score, net_relationship_score)
        VALUES (?, ?, ?, 0, 0.0, 'stranger', 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
        """,
        (clean, now, now),
    )
    # Log a first-person "I met <name>" episode to Rex's diary (rex.db). Gated +
    # failure-safe inside episodes; never let a diary hiccup break enrollment.
    if isinstance(new_id, int):
        try:
            from memory import episodes
            episodes.record_person_enrolled(new_id, clean)
        except Exception as exc:
            _log.debug("[people] episodic person_enrolled capture failed: %s", exc)
    return new_id


def find_person_by_name(name: str) -> Optional[dict]:
    """
    Return the best existing person row for this spoken/stored name.

    Full names require an exact normalized match. A one-token name reuses an
    existing person only when exactly one stored person has that first token.
    This prevents duplicate rows like "Jeff Benziger" while still avoiding wild
    first-name collisions.
    """
    norm = _normalize_name(name)
    if not norm:
        return None
    query_tokens = norm.split()
    if _person_aliases_available():
        alias_row = db.fetchone(
            """
            SELECT p.*,
                   0 AS face_count,
                   0 AS voice_count
            FROM person_aliases a
            JOIN people p ON p.id = a.person_id
            WHERE a.alias_norm = ?
            LIMIT 1
            """,
            (norm,),
        )
        if alias_row:
            return dict(alias_row)

    rows = db.fetchall(
        """
        SELECT p.*,
               SUM(CASE WHEN b.type = 'face' THEN 1 ELSE 0 END) AS face_count,
               SUM(CASE WHEN b.type = 'voice' THEN 1 ELSE 0 END) AS voice_count
        FROM people p
        LEFT JOIN biometrics b ON b.person_id = p.id
        GROUP BY p.id
        """,
    )
    exact: list[dict] = []
    first_name: list[dict] = []
    for row in rows:
        person = dict(row)
        stored_norm = _normalize_name(person.get("name") or "")
        if not stored_norm:
            continue
        stored_tokens = stored_norm.split()
        if stored_norm == norm:
            exact.append(person)
        elif len(query_tokens) == 1 and stored_tokens and stored_tokens[0] == query_tokens[0]:
            first_name.append(person)

    def _score(person: dict) -> tuple[int, int, float, int]:
        face_count = int(person.get("face_count") or 0)
        voice_count = int(person.get("voice_count") or 0)
        visit_count = int(person.get("visit_count") or 0)
        familiarity = float(person.get("familiarity_score") or 0.0)
        return (face_count + voice_count, visit_count, familiarity, -int(person["id"]))

    if exact:
        return max(exact, key=_score)
    if len(first_name) == 1:
        return first_name[0]
    return None


def _person_score(person: dict) -> tuple[int, int, float, int]:
    face_count = int(person.get("face_count") or 0)
    voice_count = int(person.get("voice_count") or 0)
    visit_count = int(person.get("visit_count") or 0)
    familiarity = float(person.get("familiarity_score") or 0.0)
    return (face_count + voice_count, visit_count, familiarity, -int(person["id"]))


def find_potential_person_match(name: str) -> Optional[dict]:
    """Return a likely existing person for enrollment confirmation, if any."""
    clean = _clean_display_name(name)
    norm = _normalize_name(clean or "")
    if not clean or not norm:
        return None
    query_tokens = norm.split()

    alias = find_person_by_name(clean)
    if alias and _normalize_name(alias.get("name") or "") == norm:
        return {"match_type": "exact", "person": alias, "candidate_name": clean}

    if _person_aliases_available():
        alias_row = db.fetchone(
            """
            SELECT p.*
            FROM person_aliases a
            JOIN people p ON p.id = a.person_id
            WHERE a.alias_norm = ?
            LIMIT 1
            """,
            (norm,),
        )
        if alias_row:
            return {"match_type": "alias", "person": dict(alias_row), "candidate_name": clean}

    rows = db.fetchall(
        """
        SELECT p.*,
               SUM(CASE WHEN b.type = 'face' THEN 1 ELSE 0 END) AS face_count,
               SUM(CASE WHEN b.type = 'voice' THEN 1 ELSE 0 END) AS voice_count
        FROM people p
        LEFT JOIN biometrics b ON b.person_id = p.id
        GROUP BY p.id
        """,
    )
    candidates = [dict(row) for row in rows]
    if len(query_tokens) == 1:
        first_matches = [
            person
            for person in candidates
            if len(_normalize_name(person.get("name") or "").split()) >= 2
            and _normalize_name(person.get("name") or "").split()[0] == query_tokens[0]
        ]
        if len(first_matches) == 1:
            return {
                "match_type": "first_name",
                "person": first_matches[0],
                "candidate_name": clean,
            }

    fuzzy = [
        person
        for person in candidates
        if names_are_similar(clean, person.get("name") or "")
    ]
    if len(fuzzy) == 1:
        return {"match_type": "fuzzy", "person": fuzzy[0], "candidate_name": clean}
    return None


def add_alias(person_id: int, alias: str, source: str = "user_confirmed") -> bool:
    """Attach an unambiguous spoken alias to an existing person."""
    clean = _clean_display_name(alias)
    alias_norm = _normalize_name(clean or "")
    if not clean or not alias_norm:
        return False
    person = get_person(int(person_id))
    if person is None:
        return False
    if _normalize_name(person.get("name") or "") == alias_norm:
        return True
    if not _person_aliases_available():
        return False

    existing = db.fetchone(
        "SELECT person_id FROM person_aliases WHERE alias_norm = ?",
        (alias_norm,),
    )
    if existing and int(existing["person_id"]) != int(person_id):
        _log.info(
            "add_alias blocked: alias %r already belongs to person_id=%s",
            clean,
            existing["person_id"],
        )
        return False
    now = _now()
    db.execute(
        """
        INSERT INTO person_aliases (person_id, alias, alias_norm, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(alias_norm) DO UPDATE SET
            alias = excluded.alias,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        (int(person_id), clean, alias_norm, source, now, now),
    )
    return True


def list_person_names() -> list[str]:
    """Return non-blank stored display names for person-memory routing."""
    rows = db.fetchall("SELECT name FROM people WHERE name IS NOT NULL AND TRIM(name) != ''")
    return [str(row["name"]) for row in rows if str(row["name"]).strip()]


def find_or_create_person(name: str) -> tuple[Optional[int], bool]:
    """Return (person_id, created). Reuses an existing row when the name is clear."""
    clean = _clean_display_name(name)
    if not clean:
        return None, False

    match = find_potential_person_match(clean)
    if match:
        match_type = str(match.get("match_type") or "")
        person = match.get("person") or {}
        if match_type in {"exact", "alias", "first_name"}:
            person_id = int(person["id"])
            if match_type == "first_name":
                add_alias(person_id, clean, source="first_name_match")
            return person_id, False
        if match_type == "fuzzy":
            _log.info(
                "find_or_create_person refused fuzzy duplicate candidate=%r existing=%r",
                clean,
                person.get("name"),
            )
            return None, False

    return enroll_person(clean), True


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


def rename_person(person_id: int, name: str) -> bool:
    """
    Update a person's display name.

    Returns False when the name is blank, the person row is missing, or the new
    name already belongs to another person row. Biometrics remain tied to the
    same person_id, so correcting a bad spoken name preserves face/voice memory.
    """
    clean = _clean_display_name(name) or ""
    if not clean:
        return False

    current = get_person(person_id)
    if current is None:
        return False

    existing = find_person_by_name(clean)
    if existing and int(existing["id"]) != int(person_id):
        _log.warning(
            "rename_person blocked: target name %r already belongs to person_id=%s",
            clean,
            existing["id"],
        )
        return False

    old_name = str(current.get("name") or "").strip()
    db.execute(
        "UPDATE people SET name = ?, last_seen = ? WHERE id = ?",
        (clean, _now(), person_id),
    )
    if old_name and _normalize_name(old_name) != _normalize_name(clean):
        add_alias(int(person_id), old_name, source="previous_name")
    add_alias(int(person_id), clean, source="canonical_name")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Visit & familiarity tracking
# ─────────────────────────────────────────────────────────────────────────────

def update_visit(person_id: int) -> None:
    """
    Increment visit_count, update last_seen, apply the return-visit familiarity increment.

    A genuine RETURN (someone Rex has met before coming back) also earns a small
    warmth + trust bump — friendship should reflect showing up again over time, not
    only explicit praise. The first-ever sighting (prior_visits == 0) gets only the
    familiarity increment, since there's no relationship to "return" to yet.

    days_known is derived at runtime (today − first_seen) and is not stored.
    """
    row = db.fetchone("SELECT visit_count FROM people WHERE id = ?", (person_id,))
    prior_visits = int(row["visit_count"]) if row and row["visit_count"] is not None else 0
    db.execute(
        "UPDATE people SET visit_count = visit_count + 1, last_seen = ? WHERE id = ?",
        (_now(), person_id),
    )
    update_familiarity(person_id, config.FAMILIARITY_INCREMENTS["return_visit"])
    if prior_visits >= 1:
        apply_relationship_increment(person_id, "consistent_return_visit")  # trust
        apply_relationship_increment(person_id, "return_visit_warmth")      # warmth


def record_milestone_greeted(person_id: int, milestone: int) -> None:
    """Remember the highest visit milestone Rex has announced for this person, so
    a milestone greeting ("your 5th visit") fires once instead of every startup.

    Monotonic: never lowers the stored value, so an out-of-order call can't
    re-arm an already-acknowledged milestone."""
    try:
        pid = int(person_id)
        n = int(milestone)
    except (TypeError, ValueError):
        return
    db.execute(
        "UPDATE people SET last_milestone_greeted = MAX(COALESCE(last_milestone_greeted, 0), ?) "
        "WHERE id = ?",
        (n, pid),
    )


def _local_date() -> str:
    """Today's calendar day in LOCAL time as 'YYYY-MM-DD'.

    Used for "same day" greeting tallies — the user experiences "again today" in
    their local day, not UTC. Self-contained: only ever compared against itself.
    """
    return datetime.now().date().isoformat()


def record_greeting(person_id: int) -> None:
    """Track durable Rex-initiated greetings for future self-memory answers.

    Also maintains a per-local-day tally (`greetings_today` / `greetings_today_date`)
    so the startup greeting can tell when the same person has summoned Rex more
    than once in a day and do "oh, it's you again" repeat-visit banter.
    """
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return
    today = _local_date()
    db.execute(
        """
        UPDATE people
           SET lifetime_greeting_count = COALESCE(lifetime_greeting_count, 0) + 1,
               last_greeted_at = ?,
               greetings_today = CASE
                   WHEN greetings_today_date = ? THEN COALESCE(greetings_today, 0) + 1
                   ELSE 1
               END,
               greetings_today_date = ?
         WHERE id = ?
        """,
        (_now(), today, today, pid),
    )


def greetings_today_count(person_id: int) -> int:
    """How many times Rex has already greeted this person so far TODAY (local day).

    Returns 0 if the stored tally is from a previous day (stale) or unknown. Read
    this BEFORE recording the current greeting to know how many prior greetings
    happened today: 0 → first time today, >=1 → a same-day repeat visit.
    """
    try:
        pid = int(person_id)
    except (TypeError, ValueError):
        return 0
    row = db.fetchone(
        "SELECT greetings_today, greetings_today_date FROM people WHERE id = ?",
        (pid,),
    )
    if row is None or row["greetings_today_date"] != _local_date():
        return 0
    return int(row["greetings_today"] or 0)


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


def apply_relationship_increment(person_id: Optional[int], kind: str) -> None:
    """Apply a named relationship delta from ``config.RELATIONSHIP_INCREMENTS``.

    Keeps that table as the single source of truth for per-interaction
    relationship adjustments instead of scattering magic numbers across call
    sites. Unknown kinds and a missing ``person_id`` are no-ops.
    """
    if person_id is None:
        return
    try:
        dimension, delta = config.RELATIONSHIP_INCREMENTS[kind]
    except (KeyError, TypeError, ValueError):
        _log.debug("unknown relationship increment kind: %s", kind)
        return
    update_relationship_scores(person_id, **{dimension: float(delta)})


# ─────────────────────────────────────────────────────────────────────────────
# Memory wipe
# ─────────────────────────────────────────────────────────────────────────────

def has_face_biometric(person_id: int) -> bool:
    """Return True if this person has at least one stored face biometric."""
    if person_id is None:
        return False
    row = db.fetchone(
        "SELECT 1 FROM biometrics WHERE person_id = ? AND type = 'face' LIMIT 1",
        (person_id,),
    )
    return row is not None


def has_voice_biometric(person_id: int) -> bool:
    """Return True if this person has at least one stored voice biometric."""
    if person_id is None:
        return False
    row = db.fetchone(
        "SELECT 1 FROM biometrics WHERE person_id = ? AND type = 'voice' LIMIT 1",
        (person_id,),
    )
    return row is not None


def count_biometrics(person_id: int, type_: str) -> int:
    """Return the number of biometric rows of a given type stored for a person."""
    if person_id is None:
        return 0
    row = db.fetchone(
        "SELECT COUNT(*) AS n FROM biometrics WHERE person_id = ? AND type = ?",
        (person_id, type_),
    )
    return int(row["n"]) if row else 0


def delete_person(person_id: int) -> None:
    """Delete all rows for a person across every person-related table."""
    with db.connection() as conn:
        for table in _PERSON_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE person_id = ?", (person_id,))
        conn.execute(
            f"DELETE FROM {_RELATIONSHIP_TABLE} WHERE from_person_id = ? OR to_person_id = ?",
            (person_id, person_id),
        )
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))


def merge_person(survivor_id: int, victim_id: int) -> bool:
    """Merge ``victim_id`` into ``survivor_id`` and delete the victim row.

    Re-points every per-person row (biometrics/voiceprints, facts, interests,
    preferences, QA, events, aliases, …) from the victim onto the survivor, so the
    survivor inherits the victim's voice/face encodings and history, then removes
    the now-empty victim person. Use when two rows turn out to be the SAME human
    (e.g. a mis-registered duplicate that captured the person's voiceprint).

    UNIQUE-constrained tables (person_aliases.alias_norm,
    person_conversation_boundaries) use UPDATE OR IGNORE so a victim row that would
    collide with an existing survivor row is dropped rather than raising; any such
    leftover victim rows are then deleted. ``biometrics`` has no unique index, so the
    voice and face encodings always migrate intact (the survivor starts matching the
    victim's voice). Returns False if either id is missing or the ids are equal.
    """
    if survivor_id is None or victim_id is None:
        return False
    survivor_id = int(survivor_id)
    victim_id = int(victim_id)
    if survivor_id == victim_id:
        return False
    with db.connection() as conn:
        have = {
            int(r[0])
            for r in conn.execute(
                "SELECT id FROM people WHERE id IN (?, ?)", (survivor_id, victim_id)
            ).fetchall()
        }
        if survivor_id not in have or victim_id not in have:
            _log.warning(
                "[identity] merge_person aborted — missing row(s) survivor=%s victim=%s",
                survivor_id,
                victim_id,
            )
            return False
        for table in _PERSON_TABLES:
            conn.execute(
                f"UPDATE OR IGNORE {table} SET person_id = ? WHERE person_id = ?",
                (survivor_id, victim_id),
            )
            conn.execute(f"DELETE FROM {table} WHERE person_id = ?", (victim_id,))
        rel_cols = {
            str(r[1])
            for r in conn.execute(
                f"PRAGMA table_info({_RELATIONSHIP_TABLE})"
            ).fetchall()
        }
        for col in ("from_person_id", "to_person_id", "described_by"):
            if col in rel_cols:
                conn.execute(
                    f"UPDATE OR IGNORE {_RELATIONSHIP_TABLE} "
                    f"SET {col} = ? WHERE {col} = ?",
                    (survivor_id, victim_id),
                )
        conn.execute(
            f"DELETE FROM {_RELATIONSHIP_TABLE} "
            "WHERE from_person_id = ? OR to_person_id = ?",
            (victim_id, victim_id),
        )
        conn.execute("DELETE FROM people WHERE id = ?", (victim_id,))
    _log.info(
        "[identity] merged person_id=%s into survivor_id=%s", victim_id, survivor_id
    )
    return True


def delete_all_people() -> None:
    """
    Remove all rows from every person-related table.

    personality_settings is global (not per-person) and is left untouched.
    The database schema and empty tables remain intact.
    """
    with db.connection() as conn:
        for table in _PERSON_TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.execute(f"DELETE FROM {_RELATIONSHIP_TABLE}")
        conn.execute("DELETE FROM people")
