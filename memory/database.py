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
    "biometrics",
    "person_facts",
    "person_qa",
    "conversations",
    "person_events",
    "personality_settings",
    "person_relationships",
    "person_emotional_events",
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
        mentioned_at             DATETIME,
        last_acknowledged_at     DATETIME,
        checkins_muted_at        DATETIME,
        checkins_muted_reason    TEXT,
        sensitivity_decay_days   INTEGER,
        person_invited_topic     INTEGER DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emoevent_person ON person_emotional_events(person_id)",
]


def _run_migrations() -> None:
    try:
        with connection() as conn:
            for stmt in _MIGRATIONS:
                conn.execute(stmt)
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
    except Exception as exc:
        _log.warning("schema migration skipped: %s", exc)


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection, yield it, commit on clean exit or roll back on exception, then close."""
    conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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
