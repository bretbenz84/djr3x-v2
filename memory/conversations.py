"""
memory/conversations.py — Session conversation summaries and in-memory transcript buffer.

The transcript buffer (add_to_transcript / get_session_transcript / clear_transcript)
is module-level in-memory state and is never persisted to the database.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory import database as db

_log = logging.getLogger(__name__)

# In-memory session transcript — cleared between sessions, never written to DB
_transcript: list[dict] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Persisted conversation summaries
# ─────────────────────────────────────────────────────────────────────────────

def save_conversation(
    person_id: int,
    summary: str,
    emotion_tone: str,
    topics: str,
) -> None:
    """Insert a session summary. topics is a comma-separated string of topic tags."""
    db.execute(
        """INSERT INTO conversations (person_id, session_date, summary, emotion_tone, topics)
           VALUES (?, ?, ?, ?, ?)""",
        (person_id, _now(), summary, emotion_tone, topics),
    )


def get_last_conversation(person_id: int) -> Optional[dict]:
    """Return the most recent conversation summary for a person, or None."""
    row = db.fetchone(
        """SELECT * FROM conversations
           WHERE person_id = ?
           ORDER BY session_date DESC
           LIMIT 1""",
        (person_id,),
    )
    return dict(row) if row else None


def get_conversation_history(person_id: int, limit: int = 5) -> list[dict]:
    """Return the N most recent conversation summaries for a person, newest first."""
    rows = db.fetchall(
        """SELECT * FROM conversations
           WHERE person_id = ?
           ORDER BY session_date DESC
           LIMIT ?""",
        (person_id, limit),
    )
    return [dict(r) for r in rows]


def delete_conversations(person_id: int) -> None:
    """Remove all conversation records for a person."""
    db.execute("DELETE FROM conversations WHERE person_id = ?", (person_id,))


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session transcript buffer
# ─────────────────────────────────────────────────────────────────────────────

def add_to_transcript(speaker: str, text: str) -> None:
    """Append a speaker/text entry to the in-memory session transcript."""
    _transcript.append({"speaker": speaker, "text": text})


def get_session_transcript() -> list[dict]:
    """Return a copy of the current in-memory session transcript."""
    return list(_transcript)


def clear_transcript() -> None:
    """Clear the in-memory session transcript buffer."""
    _transcript.clear()
