"""
Plain-text conversational log at <project_root>/logs/conversation.log.

Each line is:
    YYYY-MM-DD HH:MM:SS | HEARD | <Speaker>: <text>
    YYYY-MM-DD HH:MM:SS | REX   | <text>

Call log_heard() when speech is transcribed, log_rex() when Rex speaks.
Thread-safe; appends only; creates the file on first write.
"""

import threading
from datetime import datetime
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "logs" / "conversation.log"
_lock = threading.Lock()


def _write(line: str) -> None:
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_heard(speaker: str | None, text: str) -> None:
    """Log a transcribed utterance. speaker is a name or None for unknown."""
    label = speaker.strip() if speaker and speaker.strip() else "Guest"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(f"{ts} | HEARD | {label}: {text}")


def log_rex(text: str) -> None:
    """Log something Rex said."""
    if not text or not text.strip():
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(f"{ts} | REX   | {text.strip()}")
