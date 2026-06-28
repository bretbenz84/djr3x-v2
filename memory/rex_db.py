"""
rex_db.py — connection layer for Rex's OWN episodic-memory database (rex.db).

A SEPARATE SQLite database from people.db (memory/database.py). people.db is what
Rex knows ABOUT people; rex.db is what's happened to HIM — a first-person, timestamped
log of his experiences (people seen, scenes observed, things he did, session
summaries). Kept separate on purpose: different lifecycle (you can wipe/prune Rex's
"diary" without touching the people knowledge base), and the conceptual split is clean.

Mirrors database.py's public API (execute / fetchall / fetchone / connection /
ensure_schema) so callers feel familiar, with two deliberate differences:

  1. The DB PATH is read at CALL TIME from config.REX_DB_PATH (NOT bound at import like
     database._DB_FILE), so tests can patch the path to a temp file.
  2. A TEST-RUNNER GATE (`writes_suppressed()`) — under `python -m unittest` / pytest on
     the DEFAULT path, all writes no-op, so the suite never creates or populates a real
     rex.db. A test that points REX_DB_PATH at a temp file opts back IN to real I/O.

Schema lives here (idempotent CREATE TABLE IF NOT EXISTS) AND in setup_assets.py (for
fresh systems). Every public function swallows + logs exceptions (never raises) — the
robot must never crash because its diary hiccuped. Person ids are SOFT references to
people.db (a different DB → no hard foreign key); a name snapshot is stored alongside.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REL = "assets/memory/rex.db"
_lock = threading.Lock()

# Idempotent schema — mirrored in setup_assets.py for fresh installs.
SCHEMA = """
CREATE TABLE IF NOT EXISTS rex_episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,                 -- ISO 'YYYY-MM-DD HH:MM:SS' local time
    kind        TEXT    NOT NULL,                 -- e.g. person_seen | made_laugh | animal | scene | conversation_summary | person_enrolled | game_played | visit_departure | boundary | celebrity | emotional_checkin | birthday_wish | milestone | celebration | reunion  (canonical set: config.EPISODIC_RECALL_KIND_WEIGHTS)
    summary     TEXT    NOT NULL,                 -- first-person, human-readable: "I saw a dog", "I made Bret laugh"
    person_id   INTEGER,                          -- SOFT ref to people.id (people.db, different DB → no FK)
    person_name TEXT,                             -- name snapshot at capture time
    detail      TEXT,                             -- optional JSON blob of structured fields
    salience    REAL    NOT NULL DEFAULT 0.5,     -- 0..1, how memorable (for Phase-2 ranking)
    session_id  TEXT                              -- groups episodes from one process run
);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_created ON rex_episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_kind    ON rex_episodes(kind);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_person  ON rex_episodes(person_id);

-- Persistent per-object room model (memory/room_model.py), fed by the local COCO
-- stream. One row PER LABEL (robust to the head moving the object's position around);
-- sighting_count + first/last_seen give object PERMANENCE so curiosity prefers what's
-- new and Rex can notice 'wait, that's new' across sessions. Screens/devices/people/
-- animals are filtered upstream and never reach this table.
CREATE TABLE IF NOT EXISTS room_objects (
    label           TEXT    PRIMARY KEY,           -- COCO class, lowercased
    location_bucket TEXT,                           -- most-recent coarse position ("foreground left")
    first_seen      TEXT    NOT NULL,               -- ISO 'YYYY-MM-DD HH:MM:SS' local
    last_seen       TEXT    NOT NULL,
    sighting_count  INTEGER NOT NULL DEFAULT 1
);
"""


def _default_db_path() -> Path:
    return _PROJECT_ROOT / _DEFAULT_REL


def db_path() -> Path:
    """The rex.db path, read at CALL time from config (so tests can patch it)."""
    try:
        import config
        p = getattr(config, "REX_DB_PATH", None)
        if p:
            p = Path(p)
            return p if p.is_absolute() else _PROJECT_ROOT / p
    except Exception:
        pass
    return _default_db_path()


def _under_test_runner() -> bool:
    """True under unittest/pytest, keyed on the ENTRY POINT (sys.argv[0] / PYTEST_
    CURRENT_TEST) — NOT 'unittest' in sys.modules — so an incidental import can't
    disable real logging on the robot (which runs `python main.py`)."""
    if os.environ.get("DJR3X_EPISODIC_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def writes_suppressed() -> bool:
    """Suppress real-DB writes under the test runner on the DEFAULT path, so the suite
    never creates/populates a real rex.db. A test that points REX_DB_PATH at a temp file
    (non-default path) opts back in — its writes go to the temp DB."""
    return _under_test_runner() and db_path() == _default_db_path()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema() -> None:
    """Create rex.db + the rex_episodes table if absent (idempotent). Never raises."""
    if writes_suppressed():  # never create the real file under the test runner
        return
    try:
        with connection() as conn:
            conn.executescript(SCHEMA)
    except Exception as exc:
        _log.debug("rex_db ensure_schema failed: %s", exc)


def execute(query: str, params: tuple = ()) -> Optional[int]:
    """INSERT/UPDATE/DELETE on rex.db. Returns lastrowid (INSERT) or None on error."""
    if writes_suppressed():
        return None
    try:
        with _lock, connection() as conn:
            cur = conn.execute(query, params)
            return cur.lastrowid
    except Exception as exc:
        _log.debug("rex_db execute failed: %s", exc)
        return None


def fetchall(query: str, params: tuple = ()) -> list:
    # READS are gated too: connection() would mkdir+connect (creating the real file)
    # before any per-query check, so a read on the default path under the test runner
    # must short-circuit here — otherwise the suite would create a real rex.db.
    if writes_suppressed():
        return []
    try:
        with connection() as conn:
            return list(conn.execute(query, params).fetchall())
    except Exception as exc:
        _log.debug("rex_db fetchall failed: %s", exc)
        return []


def fetchone(query: str, params: tuple = ()):
    if writes_suppressed():
        return None
    try:
        with connection() as conn:
            return conn.execute(query, params).fetchone()
    except Exception as exc:
        _log.debug("rex_db fetchone failed: %s", exc)
        return None
