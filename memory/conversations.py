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

_REX_SPEAKERS = {"rex", "dj-r3x", "djr3x"}


def add_to_transcript(speaker: str, text: str, *, learnable: bool = True) -> None:
    """Append a speaker/text entry to the in-memory session transcript.

    ``learnable`` marks whether this turn may feed session-end memory extraction.
    Human turns are recorded as learnable by default, BEFORE the turn's routing has
    decided whether learning should be suppressed (commands, games, corrections,
    general-knowledge Q&A, minors). When a turn turns out to be non-learnable, the
    interaction layer flips this flag via ``mark_last_human_turn_unlearnable`` so the
    session-end consolidation pass honors the same suppression the per-turn extractor
    already does — otherwise a suppressed turn ("China", a misheard command) would be
    re-extracted as a permanent fact at teardown.
    """
    _transcript.append({"speaker": speaker, "text": text, "learnable": bool(learnable)})


def mark_last_human_turn_unlearnable() -> bool:
    """Flag the most recent NON-Rex transcript turn as not-learnable. Returns True if
    one was found and flipped. Targets the current exchange's human turn (only one is
    recorded per exchange; Rex's own lines in between are skipped)."""
    for entry in reversed(_transcript):
        if str(entry.get("speaker") or "").lower() not in _REX_SPEAKERS:
            entry["learnable"] = False
            return True
    return False


def get_session_transcript() -> list[dict]:
    """Return a copy of the current in-memory session transcript."""
    return list(_transcript)


def clear_transcript() -> None:
    """Clear the in-memory session transcript buffer."""
    _transcript.clear()
