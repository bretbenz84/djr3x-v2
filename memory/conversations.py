"""
memory/conversations.py — Session conversation summaries and in-memory transcript buffer.

The transcript buffer (add_to_transcript / get_session_transcript / clear_transcript)
is module-level in-memory state and is never persisted to the database.
"""

import itertools
import logging
import sys
import time
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
# Monotonic per-process turn IDs for transcript entries (Lean Brain phase 0). NOT
# reset by clear_transcript(): a consumer that remembers "covered through turn N"
# (the conversation arc, generation IDs) must never see an old ID come back after
# a session reset — length-based cursors cannot tell a reset from a race.
_turn_seq = itertools.count(1)


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
    _transcript.append({
        "speaker": speaker,
        "text": text,
        "learnable": bool(learnable),
        # Correlation keys (Lean Brain phase 0): a per-process monotonic turn id
        # and a wall-clock stamp. Readers that copy entries into their own dicts
        # (the lean transcript builder) may drop them; nothing should compare an
        # entry against a literal dict.
        "turn_id": next(_turn_seq),
        "ts": time.time(),
    })
    _log_turn(speaker, text)


def last_turn_id() -> int:
    """The turn_id of the most recent transcript entry, or 0 when empty."""
    if not _transcript:
        return 0
    try:
        return int(_transcript[-1].get("turn_id") or 0)
    except (TypeError, ValueError):
        return 0


def mark_last_human_turn_unlearnable() -> bool:
    """Flag the most recent NON-Rex transcript turn as not-learnable. Returns True if
    one was found and flipped. Targets the current exchange's human turn (only one is
    recorded per exchange; Rex's own lines in between are skipped)."""
    for entry in reversed(_transcript):
        if str(entry.get("speaker") or "").lower() not in _REX_SPEAKERS:
            entry["learnable"] = False
            return True
    return False


def relabel_prior_turn(old_speaker: str, new_speaker: str, *, skip_text: str = "") -> bool:
    """Move the ATTRIBUTION of the most recent transcript turn recorded under
    ``old_speaker`` to ``new_speaker`` — the "that was JT speaking" correction. The
    words stay; the speaker label changes, so session-end extraction credits the right
    person. ``skip_text`` excludes the correction utterance itself (it is usually the
    newest matching entry). The relabeled turn is also marked non-learnable for the
    ORIGINAL person's sake; whether it is learnable for the new speaker is a later
    turn's problem (conservative: don't extract from a disputed line at all)."""
    old_l = (old_speaker or "").strip().lower()
    skip = (skip_text or "").strip()
    for entry in reversed(_transcript):
        speaker = str(entry.get("speaker") or "").strip()
        if speaker.lower() != old_l:
            continue
        if skip and str(entry.get("text") or "").strip() == skip:
            continue
        entry["speaker"] = new_speaker
        entry["learnable"] = False
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Persisted per-turn conversation log (owner idea 2026-08-01): every spoken turn
# is also written through to the conversation_log table, so date-targeted recall
# ("what did we talk about on July 12?" / "earlier today?") can read the actual
# words back later. The in-memory buffer above stays authoritative for the live
# session; this is the durable copy. Fail-safe: a DB hiccup never breaks a turn.
# ─────────────────────────────────────────────────────────────────────────────

_session_id: Optional[str] = None
_person_id_cache: dict = {}


def _log_turn(speaker: str, text: str) -> None:
    global _session_id
    try:
        import config
        if not bool(getattr(config, "CONVERSATION_LOG_ENABLED", True)):
            return
        speaker_s = str(speaker or "").strip()
        text_s = str(text or "").strip()
        if not speaker_s or not text_s:
            return
        local = datetime.now()
        if _session_id is None:
            _session_id = local.strftime("session-%Y-%m-%d-%H-%M-%S")
        person_id = None
        if speaker_s.lower() not in _REX_SPEAKERS \
                and not speaker_s.lower().startswith("unknown_voice"):
            if speaker_s in _person_id_cache:
                person_id = _person_id_cache[speaker_s]
            else:
                try:
                    from memory import people
                    row = people.find_person_by_name(speaker_s)
                    person_id = int(row["id"]) if row else None
                except Exception:
                    person_id = None
                _person_id_cache[speaker_s] = person_id
        db.execute(
            """INSERT OR IGNORE INTO conversation_log
               (ts, day, session_id, speaker, person_id, text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (local.strftime("%Y-%m-%d %H:%M:%S"), local.strftime("%Y-%m-%d"),
             _session_id, speaker_s, person_id, text_s),
        )
    except Exception as exc:
        _log.debug("conversation_log write skipped: %s", exc)


def get_logged_turns(
    day_start: str, day_end: Optional[str] = None, *, limit: int = 400
) -> list[dict]:
    """Turns whose LOCAL calendar day is in [day_start, day_end] (inclusive,
    'YYYY-MM-DD'; day_end defaults to day_start), oldest first."""
    try:
        rows = db.fetchall(
            """SELECT * FROM conversation_log
               WHERE day >= ? AND day <= ?
               ORDER BY ts ASC LIMIT ?""",
            (day_start, day_end or day_start, int(limit)),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        _log.debug("conversation_log read failed: %s", exc)
        return []


def last_logged_day_before(day: str) -> Optional[str]:
    """The most recent day strictly before `day` that has logged turns, or None."""
    try:
        row = db.fetchone(
            "SELECT MAX(day) AS d FROM conversation_log WHERE day < ?", (day,)
        )
        return row["d"] if row and row["d"] else None
    except Exception as exc:
        _log.debug("conversation_log day lookup failed: %s", exc)
        return None


def get_session_transcript() -> list[dict]:
    """Return a copy of the current in-memory session transcript."""
    return list(_transcript)


def clear_transcript() -> None:
    """Clear the in-memory session transcript buffer."""
    _transcript.clear()
