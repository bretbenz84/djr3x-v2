"""
memory/database.py — SQLite connection layer for people.db.

Schema creation belongs to setup_assets.py. This module only connects and queries.
"""

import logging
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# config.py lives at the project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from config import DB_PATH  # noqa: E402

_DB_FILE: Path = _PROJECT_ROOT / DB_PATH

_EXPECTED_TABLES = frozenset({
    "people",
    "person_aliases",
    "biometrics",
    "person_facts",
    "person_qa",
    "conversations",
    "person_events",
    "personality_settings",
    "person_relationships",
    "person_emotional_events",
    "person_conversation_boundaries",
    "person_preferences",
    "person_interests",
    "person_disposition_stats",
    "person_callback_material",
    "voice_signatures",
})

# Inline migrations for schema additions introduced after initial deploy.
# Idempotent: CREATE TABLE IF NOT EXISTS is safe on both new and old DBs.
_MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS person_relationships (
        id              INTEGER PRIMARY KEY,
        from_person_id  INTEGER REFERENCES people(id),
        to_person_id    INTEGER REFERENCES people(id),
        relationship    TEXT,
        described_by    INTEGER REFERENCES people(id),
        created_at      DATETIME,
        updated_at      DATETIME,
        UNIQUE(from_person_id, to_person_id, relationship)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rel_from ON person_relationships(from_person_id)",
    "CREATE INDEX IF NOT EXISTS idx_rel_to   ON person_relationships(to_person_id)",
    """
    CREATE TABLE IF NOT EXISTS person_emotional_events (
        id                       INTEGER PRIMARY KEY,
        person_id                INTEGER REFERENCES people(id),
        category                 TEXT,
        valence                  REAL,
        description              TEXT,
        loss_subject             TEXT,
        loss_subject_kind        TEXT,
        loss_subject_name        TEXT,
        mentioned_at             DATETIME,
        last_acknowledged_at     DATETIME,
        checkins_muted_at        DATETIME,
        checkins_muted_reason    TEXT,
        sensitivity_decay_days   INTEGER,
        person_invited_topic     INTEGER DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emoevent_person ON person_emotional_events(person_id)",
    """
    CREATE TABLE IF NOT EXISTS person_conversation_boundaries (
        id              INTEGER PRIMARY KEY,
        person_id       INTEGER REFERENCES people(id),
        behavior        TEXT,
        topic           TEXT,
        description     TEXT,
        source_text     TEXT,
        active          INTEGER DEFAULT 1,
        created_at      DATETIME,
        updated_at      DATETIME,
        UNIQUE(person_id, behavior, topic)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_boundary_person ON person_conversation_boundaries(person_id)",
    """
    CREATE TABLE IF NOT EXISTS person_preferences (
        id                  INTEGER PRIMARY KEY,
        person_id           INTEGER REFERENCES people(id),
        domain              TEXT,
        preference_type     TEXT,
        key                 TEXT,
        value               TEXT,
        confidence          REAL DEFAULT 1.0,
        importance          REAL DEFAULT 0.5,
        source              TEXT,
        created_at          DATETIME,
        updated_at          DATETIME,
        last_used_at        DATETIME,
        ask_cooldown_until  DATETIME,
        UNIQUE(person_id, domain, preference_type, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_pref_person ON person_preferences(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_pref_lookup ON person_preferences(person_id, domain, key)",
    """
    CREATE TABLE IF NOT EXISTS person_interests (
        id                      INTEGER PRIMARY KEY,
        person_id               INTEGER REFERENCES people(id),
        name                    TEXT,
        category                TEXT,
        interest_strength       TEXT,
        confidence              REAL DEFAULT 1.0,
        source                  TEXT,
        first_mentioned_at      DATETIME,
        last_mentioned_at       DATETIME,
        last_asked_about_at     DATETIME,
        ask_cooldown_until      DATETIME,
        notes                   TEXT,
        associated_people       TEXT,
        associated_stories      TEXT,
        UNIQUE(person_id, name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_interest_person ON person_interests(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_interest_lookup ON person_interests(person_id, name)",
    """
    CREATE TABLE IF NOT EXISTS person_disposition_stats (
        person_id               INTEGER PRIMARY KEY REFERENCES people(id),
        total_samples           INTEGER DEFAULT 0,
        smile_samples           INTEGER DEFAULT 0,
        frown_samples           INTEGER DEFAULT 0,
        neutral_samples         INTEGER DEFAULT 0,
        surprise_samples        INTEGER DEFAULT 0,
        brow_furrow_samples     INTEGER DEFAULT 0,
        other_samples           INTEGER DEFAULT 0,
        smile_score             REAL DEFAULT 0.0,
        frown_score             REAL DEFAULT 0.0,
        neutral_score           REAL DEFAULT 0.0,
        surprise_score          REAL DEFAULT 0.0,
        brow_furrow_score       REAL DEFAULT 0.0,
        dominant_expression     TEXT,
        disposition_label       TEXT,
        confidence              REAL DEFAULT 0.0,
        first_observed_at       DATETIME,
        last_observed_at        DATETIME,
        last_mentioned_at       DATETIME
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_disposition_label ON person_disposition_stats(disposition_label)",
    """
    CREATE TABLE IF NOT EXISTS person_aliases (
        id          INTEGER PRIMARY KEY,
        person_id   INTEGER REFERENCES people(id),
        alias       TEXT NOT NULL,
        alias_norm  TEXT NOT NULL UNIQUE,
        source      TEXT,
        created_at  DATETIME,
        updated_at  DATETIME
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alias_person ON person_aliases(person_id)",
    # Callback humor: banked per-person "fun fact" premises (memory/callbacks.py).
    """
    CREATE TABLE IF NOT EXISTS person_callback_material (
        id              INTEGER PRIMARY KEY,
        person_id       INTEGER REFERENCES people(id),
        premise         TEXT,
        category        TEXT,
        topic_slug      TEXT,
        sensitivity     TEXT DEFAULT 'guarded',
        source          TEXT,
        source_quote    TEXT,
        source_fact_id  INTEGER,
        volunteered_playfully INTEGER DEFAULT 0,
        session_id      TEXT,
        created_at      DATETIME,
        updated_at      DATETIME,
        last_used_at    DATETIME,
        use_count       INTEGER DEFAULT 0,
        retired_at      DATETIME,
        retired_reason  TEXT,
        UNIQUE(person_id, topic_slug)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_callback_person ON person_callback_material(person_id)",
    # Voice-primary identity: cross-session memory for recurring UNKNOWN voices.
    # A signature is one persisted voice embedding Rex has heard but has no name
    # for yet. person_id stays NULL until the voice is named, at which point the
    # signature is linked + its samples become a real voice biometric. This gives
    # "I've heard your voice before" continuity across sessions without creating a
    # nameless person row (memory/voice_signatures.py).
    """
    CREATE TABLE IF NOT EXISTS voice_signatures (
        id            INTEGER PRIMARY KEY,
        embedding     BLOB NOT NULL,
        turns         INTEGER DEFAULT 1,
        person_id     INTEGER REFERENCES people(id),
        label         TEXT,
        created_at    DATETIME,
        last_seen_at  DATETIME
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_voice_sig_person ON voice_signatures(person_id)",
    # Cross-session dedupe for PROACTIVE topic asks (e.g. "got Juneteenth plans?"), so a
    # date-bound question Rex already raised in a PRIOR run isn't repeated. topic_key is
    # caller-defined (e.g. "holiday_plans:2026-06-19" — the date makes next year a fresh
    # key). Kept separate from person_qa so it never pollutes the pending-question logic.
    """
    CREATE TABLE IF NOT EXISTS proactive_topics_asked (
        person_id   INTEGER NOT NULL,
        topic_key   TEXT NOT NULL,
        asked_at    DATETIME,
        answered    INTEGER DEFAULT 0,
        PRIMARY KEY (person_id, topic_key)
    )
    """,
]


def _safe_exec(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> None:
    """Run one migration statement, isolating its failure. A single bad ALTER/UPDATE
    must NOT abort the rest of the migration (the old behavior left a half-migrated DB
    that still passed table-only verification, then surfaced as swallowed query errors)."""
    try:
        conn.execute(sql, params)
    except Exception as exc:
        _log.warning("migration statement skipped: %s | %s", sql.strip().split("\n")[0][:80], exc)


def _run_migrations() -> None:
    try:
        with connection() as conn:
            for stmt in _MIGRATIONS:
                _safe_exec(conn, stmt)
            _ensure_column(
                conn,
                "person_emotional_events",
                "checkins_muted_at",
                "DATETIME",
            )
            _ensure_column(
                conn,
                "person_emotional_events",
                "checkins_muted_reason",
                "TEXT",
            )
            for column in ("loss_subject", "loss_subject_kind", "loss_subject_name"):
                _ensure_column(
                    conn,
                    "person_emotional_events",
                    column,
                    "TEXT",
                )
            _ensure_column(
                conn,
                "person_facts",
                "last_confirmed_at",
                "DATETIME",
            )
            _ensure_column(
                conn,
                "person_facts",
                "evidence_count",
                "INTEGER DEFAULT 1",
            )
            for column, definition in (
                ("lifetime_greeting_count", "INTEGER DEFAULT 0"),
                ("last_greeted_at", "DATETIME"),
                # Per-local-day greeting counter so Rex can do "oh, it's you again"
                # repeat-visit banter when summoned multiple times in one day.
                ("greetings_today", "INTEGER DEFAULT 0"),
                ("greetings_today_date", "TEXT"),
                # Highest visit milestone Rex has already announced, so a
                # milestone greeting ("your 5th visit") fires once, not every boot.
                ("last_milestone_greeted", "INTEGER DEFAULT 0"),
            ):
                _ensure_column(
                    conn,
                    "people",
                    column,
                    definition,
                )
            for column, definition in (
                ("importance", "REAL DEFAULT 0.5"),
                ("decay_rate", "TEXT DEFAULT 'normal'"),
                ("last_used_at", "DATETIME"),
                ("stale_after_days", "INTEGER"),
                ("corrected_at", "DATETIME"),
                # "Tell me about someone" pre-briefings: gossip/fact label,
                # mean↔kind score, and which person told Rex the detail.
                ("fact_kind", "TEXT DEFAULT 'fact'"),
                ("kindness", "REAL"),
                ("told_by", "INTEGER"),
            ):
                _ensure_column(
                    conn,
                    "person_facts",
                    column,
                    definition,
                )
            for column, definition in (
                ("status", "TEXT DEFAULT 'planned'"),
                ("canceled_at", "DATETIME"),
                ("updated_at", "DATETIME"),
                # When Rex last SPOKE an anticipation for this event — distinct
                # from mentioned_at (when the human mentioned it). The
                # anticipation cooldown keys on this, so a never-anticipated
                # event can't be throttled by its own mention (field 2026-07-18:
                # the river float mentioned at 1 AM was still inside the 20h
                # cooldown at 9 PM, so Rex never brought it up).
                ("anticipated_at", "DATETIME"),
            ):
                _ensure_column(
                    conn,
                    "person_events",
                    column,
                    definition,
                )
            # WHEN an emotional event itself happened ('recent' / 'historical'
            # / 'unknown'), distinct from mentioned_at (when it was disclosed).
            # Legacy rows default to 'unknown', which excludes them from
            # greeting/check-in queries — an undated old loss must not be
            # treated as fresh grief.
            _ensure_column(
                conn,
                "person_emotional_events",
                "recency",
                "TEXT DEFAULT 'unknown'",
            )
            _safe_exec(
                conn,
                """UPDATE person_events
                   SET status = 'planned'
                   WHERE status IS NULL OR status = ''""",
            )
            _safe_exec(
                conn,
                """UPDATE person_events
                   SET updated_at = COALESCE(updated_at, follow_up_at, mentioned_at)
                   WHERE updated_at IS NULL""",
            )
            _safe_exec(
                conn,
                """UPDATE person_facts
                   SET last_confirmed_at = COALESCE(last_confirmed_at, updated_at, created_at)
                   WHERE last_confirmed_at IS NULL""",
            )
            _safe_exec(
                conn,
                """UPDATE person_facts
                   SET evidence_count = 1
                   WHERE evidence_count IS NULL OR evidence_count < 1""",
            )
            _safe_exec(
                conn,
                """UPDATE person_facts
                   SET importance = 0.5
                   WHERE importance IS NULL""",
            )
            _safe_exec(
                conn,
                """UPDATE person_facts
                   SET decay_rate = 'normal'
                   WHERE decay_rate IS NULL OR decay_rate = ''""",
            )
            _sweep_orphans(conn)
    except Exception as exc:
        _log.warning("schema migration skipped: %s", exc)


# Child tables keyed by a plain person_id column. NULL person_id is legitimate for
# voice_signatures (an as-yet-unnamed voice) and is left alone — only rows pointing at
# a person id that no longer exists are swept.
_ORPHAN_PERSON_TABLES = (
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
    "person_callback_material",
    "voice_signatures",
    "proactive_topics_asked",
)


def _sweep_orphans(conn: sqlite3.Connection) -> None:
    """Delete child rows that reference a person id no longer in ``people``.

    Foreign keys are not enforced on this DB (cross-person ``told_by`` / ``described_by``
    columns would need ON DELETE SET NULL via a table rebuild first), so incomplete
    historical deletes leave orphaned facts/interests that still feed prompts. This is a
    cheap, self-healing sweep run on every migration pass; it never touches NULL refs."""
    for table in _ORPHAN_PERSON_TABLES:
        try:
            conn.execute(
                f"DELETE FROM {table} "
                "WHERE person_id IS NOT NULL "
                "AND person_id NOT IN (SELECT id FROM people)"
            )
        except Exception as exc:
            _log.debug("orphan sweep skipped for %s: %s", table, exc)
    for col in ("from_person_id", "to_person_id"):
        try:
            conn.execute(
                f"DELETE FROM person_relationships "
                f"WHERE {col} IS NOT NULL "
                f"AND {col} NOT IN (SELECT id FROM people)"
            )
        except Exception as exc:
            _log.debug("orphan sweep skipped for person_relationships.%s: %s", col, exc)


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    try:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except Exception as exc:
        _log.warning("add column %s.%s skipped: %s", table, column, exc)


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection, yield it, commit on clean exit or roll back on exception, then close."""
    conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    # WAL allows one writer; the vision/disposition threads write concurrently. Without a
    # busy timeout a second writer gets SQLITE_BUSY immediately, which the execute()
    # wrappers swallow as a silently-dropped write. Give contended writes a retry window.
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db() -> sqlite3.Connection:
    """Return an open connection with row_factory=Row. Caller is responsible for closing it."""
    conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def fetchone(query: str, params: tuple = ()) -> sqlite3.Row | None:
    """Execute a SELECT and return the first matching row, or None on no match or error."""
    try:
        with connection() as conn:
            return conn.execute(query, params).fetchone()
    except Exception as exc:
        _log.error("fetchone failed | query=%s | params=%s | %s", query, params, exc)
        return None


def fetchall(query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """Execute a SELECT and return all matching rows, or an empty list on error."""
    try:
        with connection() as conn:
            return conn.execute(query, params).fetchall()
    except Exception as exc:
        _log.error("fetchall failed | query=%s | params=%s | %s", query, params, exc)
        return []


def execute(query: str, params: tuple = ()) -> int | None:
    """Execute an INSERT/UPDATE/DELETE. Returns lastrowid for INSERT, None on error."""
    try:
        with connection() as conn:
            cur = conn.execute(query, params)
            return cur.lastrowid
    except Exception as exc:
        _log.error("execute failed | query=%s | params=%s | %s", query, params, exc)
        return None


def executemany(query: str, params_seq: list[tuple]) -> int | None:
    """Execute a batch statement. Returns total rowcount, or None on error."""
    try:
        with connection() as conn:
            cur = conn.executemany(query, params_seq)
            return cur.rowcount
    except Exception as exc:
        _log.error(
            "executemany failed | query=%s | count=%d | %s",
            query, len(params_seq), exc,
        )
        return None


def verify_schema() -> None:
    """Raise RuntimeError if people.db is missing or any expected table is absent.

    Runs inline migrations first so older DBs transparently gain new tables.
    """
    _run_migrations()
    try:
        with connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open people.db at {_DB_FILE}. "
            "Run setup_assets.py to create and initialize it."
        ) from exc

    found = {row["name"] for row in rows}
    missing = _EXPECTED_TABLES - found
    if missing:
        raise RuntimeError(
            f"people.db is missing tables: {sorted(missing)}. "
            "Run setup_assets.py to initialize the schema."
        )
    _run_one_time_data_migrations()


# Bump when a new one-time data migration is added below; PRAGMA user_version gates it
# so the pass runs once per DB, not on every boot.
_DATA_MIGRATION_VERSION = 2


def _run_one_time_data_migrations() -> None:
    """Run one-time data cleanups (not schema) gated by PRAGMA user_version.

    v1: collapse the duplicate/fragmented interests and events that accumulated under
    the old exact-string dedup (e.g. 'R3X droid' / 'building an R3X droid', 'camping
    trip' x4).
    v2: purge already-stored garbage the new fact_quality gate rejects — tautologies
    ('dad'->'dad'), first-person fragments, fiction scenes, verbatim-question values/
    notes — that landed before the extraction gate existed.
    Idempotent, but user_version keeps it from re-scanning every person on every launch."""
    try:
        with connection() as conn:
            current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except Exception as exc:
        _log.debug("user_version read failed; skipping data migrations: %s", exc)
        return
    if current >= _DATA_MIGRATION_VERSION:
        return
    try:
        from memory import dedup
        summary = dedup.consolidate_all()
        _log.info("[data_migration] v%d duplicate consolidation: %s",
                  _DATA_MIGRATION_VERSION, summary)
        purge = dedup.purge_low_quality()
        _log.info("[data_migration] v%d low-quality purge: %s",
                  _DATA_MIGRATION_VERSION, purge)
    except Exception as exc:
        _log.warning("[data_migration] duplicate consolidation failed: %s", exc)
        return
    try:
        with connection() as conn:
            conn.execute(f"PRAGMA user_version = {_DATA_MIGRATION_VERSION}")
    except Exception as exc:
        _log.debug("user_version write failed: %s", exc)
