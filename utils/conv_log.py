"""
Plain-text conversational log at <project_root>/logs/conversation.log.

Each line is:
    YYYY-MM-DD HH:MM:SS | HEARD | <Speaker>: <text>
    YYYY-MM-DD HH:MM:SS | REX   | <text>

Call log_heard() when speech is transcribed, log_rex() when Rex speaks.
Thread-safe; appends only; creates the file on first write.
"""

import threading
import time
from datetime import datetime
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "logs" / "conversation.log"
_lock = threading.Lock()
_last_rex_norm: str = ""
_last_rex_at: float = 0.0
# Central TTS logging writes when playback starts; legacy call sites often log
# again after blocking speech returns. Keep this long enough to cover a normal
# generated line plus TTS/API/playback latency without suppressing intentional
# repeats later in the conversation.
_REX_DEDUPE_WINDOW_SECS = 30.0


def _write(line: str) -> None:
    with _lock:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def log_heard(speaker: str | None, text: str) -> None:
    """Log a transcribed utterance. speaker is a name or None for unknown."""
    label = speaker.strip() if speaker and speaker.strip() else "Guest"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(f"{ts} | HEARD | {label}: {text}")


def log_rex(text: str) -> None:
    """Log something Rex said."""
    global _last_rex_norm, _last_rex_at
    if not text or not text.strip():
        return
    norm = _normalize(text)
    now = time.monotonic()
    with _lock:
        if norm and norm == _last_rex_norm and (now - _last_rex_at) <= _REX_DEDUPE_WINDOW_SECS:
            return
        _last_rex_norm = norm
        _last_rex_at = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{ts} | REX   | {text.strip()}\n")


def clear_dedupe_state() -> None:
    """Test/debug hook."""
    global _last_rex_norm, _last_rex_at
    with _lock:
        _last_rex_norm = ""
        _last_rex_at = 0.0
