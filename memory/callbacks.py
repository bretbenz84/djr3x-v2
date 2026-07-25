"""
memory/callbacks.py — Callback-humor material (person_callback_material table).

Persistent store behind intelligence/callback_engine.py: a small per-person pool
of light, self-volunteered "fun fact" premises (passions, hobbies, quirky
admissions, strong trivial opinions, self-descriptions) that Rex can resurface
later for comedic payoff.

This module is deliberately dumb storage. All firing judgment (tone gates,
relevance, pacing) lives in the engine; all sensitivity classification happens
at write time in the banker. Three invariants are enforced HERE so no caller
can bypass them:

  • The selector pool (`active_pool`) only ever returns sensitivity='safe',
    unretired rows. 'guarded' and 'excluded' rows exist for audit/idempotence
    (so re-extraction doesn't re-litigate them) and are never surfaced.
  • Retire, don't delete: boundaries, forget requests, and pool overflow set
    retired_at (mirroring emotional_events' mute-don't-delete), so a premise
    can't come back by being re-banked — bank() never resurrects retired rows
    or upgrades a row's sensitivity toward 'safe'.
  • Re-banking an existing (person_id, topic_slug) refreshes the premise text
    but keeps use_count/last_used_at — re-mentioning a topic doesn't reset its
    cooldown.

Writes are gated by config.CALLBACK_BANK_ENABLED; reads by the same flag OR
CALLBACK_HUMOR_ENABLED (so an already-built pool keeps firing if capture is
later switched off). Everything is failure-safe: storage errors log and return
empty/None — callback humor must never break a turn.
"""

import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)

SENSITIVITY_SAFE = "safe"
SENSITIVITY_GUARDED = "guarded"
SENSITIVITY_EXCLUDED = "excluded"
_SENSITIVITIES = (SENSITIVITY_SAFE, SENSITIVITY_GUARDED, SENSITIVITY_EXCLUDED)
# Conservative ordering: a row's sensitivity may only ever move DOWN this list
# index-wise (toward excluded), never back up toward safe.
_SENSITIVITY_RANK = {SENSITIVITY_SAFE: 0, SENSITIVITY_GUARDED: 1, SENSITIVITY_EXCLUDED: 2}

CATEGORIES = {
    "passion",
    "hobby",
    "project",
    "quirk",
    "opinion",
    "self_description",
    "running_bit",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return cleaned[:40]


def _write_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "CALLBACK_BANK_ENABLED", False))
    except Exception:
        return False


def _read_enabled() -> bool:
    try:
        import config
        return bool(getattr(config, "CALLBACK_BANK_ENABLED", False)) or bool(
            getattr(config, "CALLBACK_HUMOR_ENABLED", False)
        )
    except Exception:
        return False


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


# ── Writes ────────────────────────────────────────────────────────────────────

def bank(
    person_id: int,
    premise: str,
    *,
    category: str,
    topic: str,
    sensitivity: str,
    source_quote: str = "",
    volunteered_playfully: bool = False,
    session_id: Optional[str] = None,
    source_fact_id: Optional[int] = None,
    source: str = "explicit",
) -> Optional[int]:
    """Insert or refresh one callback premise. Returns the row id, or None when
    gated/invalid/failed.

    Upsert key is (person_id, topic_slug). On update: premise/category/quote
    refresh, usage history is KEPT, retirement is never undone, and sensitivity
    only moves toward more conservative (a later 'guarded' read downgrades a
    'safe' row, but a later 'safe' read never upgrades a 'guarded' one).

    `source` records HOW Rex came to believe this. It used to be hardcoded
    'explicit' for every row, which meant a guess from the object detector was
    stored with the same standing as something the person said out loud — and the
    field showed exactly what that produces: "has Mysterious black object in their
    space" and "has Bret Benziger in their space", the latter filing the owner
    himself as a piece of his own furniture (2026-07-25).
    """
    source = (source or "explicit").strip().lower() or "explicit"
    if not _write_enabled():
        return None
    if not isinstance(person_id, int):
        return None
    premise = re.sub(r"\s+", " ", (premise or "").strip())
    topic_slug = _slug(topic)
    if not premise or not topic_slug:
        return None
    sensitivity = (sensitivity or "").strip().lower()
    if sensitivity not in _SENSITIVITIES:
        sensitivity = SENSITIVITY_GUARDED
    category = (category or "").strip().lower()
    if category not in CATEGORIES:
        category = "quirk"
    now = _now()
    try:
        # Single-statement upsert: two overlapping post-response background
        # threads banking the same topic must not be able to interleave a
        # SELECT-then-INSERT such that a sensitivity DEMOTION is lost. The
        # instr() trick ranks 'safe' < 'guarded' < 'excluded' by position, so
        # the merge keeps whichever classification is more conservative.
        # Usage history (use_count/last_used_at) and retirement are untouched.
        db.execute(
            "INSERT INTO person_callback_material "
            "(person_id, premise, category, topic_slug, sensitivity, source, "
            " source_quote, source_fact_id, volunteered_playfully, session_id, "
            " created_at, updated_at, use_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0) "
            "ON CONFLICT(person_id, topic_slug) DO UPDATE SET "
            "  premise = excluded.premise, "
            "  category = excluded.category, "
            "  source_quote = excluded.source_quote, "
            "  volunteered_playfully = MAX(person_callback_material.volunteered_playfully, "
            "                              excluded.volunteered_playfully), "
            "  session_id = excluded.session_id, "
            "  updated_at = excluded.updated_at, "
            "  sensitivity = CASE WHEN "
            "      instr('safe|guarded|excluded', person_callback_material.sensitivity) >= "
            "      instr('safe|guarded|excluded', excluded.sensitivity) "
            "    THEN person_callback_material.sensitivity "
            "    ELSE excluded.sensitivity END",
            (
                person_id, premise, category, topic_slug, sensitivity, source,
                (source_quote or "")[:300], source_fact_id,
                1 if volunteered_playfully else 0, session_id, now, now,
            ),
        )
        row = db.fetchone(
            "SELECT id FROM person_callback_material "
            "WHERE person_id = ? AND topic_slug = ?",
            (person_id, topic_slug),
        )
        if row is None:
            return None
        _prune_overflow(person_id)
        return int(row["id"])
    except Exception as exc:
        _log.debug("[callbacks] bank failed: %s", exc)
        return None


def mark_used(callback_id: int) -> None:
    """Record that a callback actually FIRED (spoke) on this premise."""
    try:
        db.execute(
            "UPDATE person_callback_material "
            "SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
            (_now(), int(callback_id)),
        )
    except Exception as exc:
        _log.debug("[callbacks] mark_used failed: %s", exc)


def retire(callback_id: int, reason: str) -> None:
    try:
        db.execute(
            "UPDATE person_callback_material SET retired_at = ?, retired_reason = ? "
            "WHERE id = ? AND retired_at IS NULL",
            (_now(), (reason or "")[:120], int(callback_id)),
        )
    except Exception as exc:
        _log.debug("[callbacks] retire failed: %s", exc)


def retire_matching_topic(person_id: int, topic: str, reason: str) -> int:
    """Retire active premises whose topic overlaps `topic` (boundary hook:
    'stop bringing that up' must stop banked callbacks on the topic, not just
    live remarks). Uses boundaries' own overlap matcher so 'my job' and 'work'
    cluster the same way they do for consent checks. Returns rows retired."""
    if not isinstance(person_id, int) or not (topic or "").strip():
        return 0
    try:
        from memory.boundaries import _topics_overlap
    except Exception:
        return 0
    count = 0
    try:
        for row in db.fetchall(
            "SELECT id, topic_slug, premise FROM person_callback_material "
            "WHERE person_id = ? AND retired_at IS NULL",
            (person_id,),
        ):
            row_topic = (row["topic_slug"] or "").replace("_", " ")
            if _topics_overlap(topic, row_topic) or _topics_overlap(
                topic, row["premise"] or ""
            ):
                retire(int(row["id"]), reason)
                count += 1
    except Exception as exc:
        _log.debug("[callbacks] retire_matching_topic failed: %s", exc)
    if count:
        _log.info(
            "[callbacks] retired %d premise(s) for person %s on topic %r (%s)",
            count, person_id, topic, reason,
        )
    return count


def _prune_overflow(person_id: int) -> None:
    """Keep the SAFE pool small (roast material is a curated pool, not an
    archive): beyond CALLBACK_BANK_MAX_PER_PERSON active safe rows, retire the
    least valuable (lowest use_count, then longest untouched)."""
    try:
        import config
        cap = int(getattr(config, "CALLBACK_BANK_MAX_PER_PERSON", 12))
    except Exception:
        cap = 12
    if cap <= 0:
        return
    try:
        rows = db.fetchall(
            "SELECT id FROM person_callback_material "
            "WHERE person_id = ? AND sensitivity = 'safe' AND retired_at IS NULL "
            "ORDER BY use_count DESC, "
            "COALESCE(last_used_at, updated_at, created_at) DESC",
            (person_id,),
        )
        for row in rows[cap:]:
            retire(int(row["id"]), "pool_overflow")
    except Exception as exc:
        _log.debug("[callbacks] prune failed: %s", exc)


# ── Reads ─────────────────────────────────────────────────────────────────────

def active_pool(person_id: Optional[int]) -> list[dict]:
    """The ONLY selector-facing read: safe, unretired premises for one person,
    newest first. Hard-filters sensitivity here so no caller can surface
    guarded/excluded material by accident."""
    if not _read_enabled() or not isinstance(person_id, int):
        return []
    try:
        rows = db.fetchall(
            "SELECT * FROM person_callback_material "
            "WHERE person_id = ? AND sensitivity = 'safe' AND retired_at IS NULL "
            "ORDER BY updated_at DESC",
            (person_id,),
        )
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        _log.debug("[callbacks] active_pool failed: %s", exc)
        return []


def get_all(person_id: Optional[int]) -> list[dict]:
    """Every row regardless of sensitivity/retirement — debugging and tests."""
    if not isinstance(person_id, int):
        return []
    return [
        _row_to_dict(r)
        for r in db.fetchall(
            "SELECT * FROM person_callback_material WHERE person_id = ? ORDER BY id",
            (person_id,),
        )
    ]


def has_topic(person_id: int, topic: str) -> bool:
    """True when ANY row (including guarded/excluded/retired) already covers this
    topic — lets the banker skip re-classifying material it has already judged."""
    if not isinstance(person_id, int):
        return False
    topic_slug = _slug(topic)
    if not topic_slug:
        return False
    row = db.fetchone(
        "SELECT id FROM person_callback_material WHERE person_id = ? AND topic_slug = ?",
        (person_id, topic_slug),
    )
    return row is not None


def is_running_bit(row: dict) -> bool:
    """A premise that has LANDED enough to become a recurring "running bit" — and not yet
    aged out. A promoted bit escapes the reuse-suppression (no use-decay, no cross-session
    lockout) so a gag that genuinely recurs comes back instead of fading; the recurrence
    IS the joke (silent — never numbered aloud). Promotion is computed from use_count (no
    schema change), so it's EARNED by real recurrence, and it ages back out at RETIRE_AT.
    Gated by RUNNING_BIT_ENABLED."""
    try:
        import config
        if not bool(getattr(config, "RUNNING_BIT_ENABLED", True)):
            return False
        promote = int(getattr(config, "RUNNING_BIT_PROMOTE_AT", 3))
        retire = int(getattr(config, "RUNNING_BIT_RETIRE_AT", 8))
    except Exception:
        return False
    try:
        uses = int(row.get("use_count") or 0)
    except Exception:
        return False
    return promote <= uses < retire


def off_cooldown(row: dict, *, now: Optional[datetime] = None) -> bool:
    """Whether a premise's cross-session reuse cooldown has elapsed. A promoted running
    bit uses the much shorter RUNNING_BIT_REUSE_COOLDOWN_DAYS so it can recur instead of
    being locked out for a week."""
    try:
        import config
        if is_running_bit(row):
            days = float(getattr(config, "RUNNING_BIT_REUSE_COOLDOWN_DAYS", 0.0))
        else:
            days = float(getattr(config, "CALLBACK_REUSE_COOLDOWN_DAYS", 7.0))
    except Exception:
        days = 7.0
    last = _parse_ts(row.get("last_used_at"))
    if last is None:
        return True
    now = now or datetime.now(timezone.utc)
    return (now - last).total_seconds() >= days * 86400.0


def freshness_factor(row: dict) -> float:
    """Decaying-reuse weight: halves every CALLBACK_USE_DECAY_HALFLIFE_USES
    uses, so a well-worn bit gradually steps back without ever being banned.
    A promoted running bit is EXEMPT — it keeps full weight so it recurs instead of
    fading (until it ages out at RETIRE_AT and the normal decay resumes)."""
    if is_running_bit(row):
        try:
            import config
            return max(0.0, float(getattr(config, "RUNNING_BIT_FRESHNESS", 1.0)))
        except Exception:
            return 1.0
    try:
        import config
        halflife = float(getattr(config, "CALLBACK_USE_DECAY_HALFLIFE_USES", 3.0))
    except Exception:
        halflife = 3.0
    if halflife <= 0:
        return 1.0
    try:
        uses = max(0, int(row.get("use_count") or 0))
    except Exception:
        uses = 0
    return 0.5 ** (uses / halflife)
