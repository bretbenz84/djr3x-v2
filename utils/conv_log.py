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

import config

_LOG_PATH = Path(__file__).parent.parent / "logs" / "conversation.log"
_lock = threading.Lock()
_last_rex_norm: str = ""
_last_rex_at: float = 0.0
# Central TTS logging writes when playback starts; legacy call sites often log
# again after blocking speech returns. Keep this long enough to cover a normal
# generated line plus TTS/API/playback latency without suppressing intentional
# repeats later in the conversation.
_REX_DEDUPE_WINDOW_SECS = 30.0


def _max_lines() -> int:
    if getattr(config, "DEBUG_MODE", False):
        return int(getattr(config, "CONVERSATION_LOG_DEBUG_MAX_LINES", 120) or 0)
    return int(getattr(config, "CONVERSATION_LOG_MAX_LINES", 400) or 0)


def _trim_locked() -> None:
    max_lines = _max_lines()
    if max_lines <= 0 or not _LOG_PATH.exists():
        return
    lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) <= max_lines:
        return
    kept = lines[-max_lines:]
    _LOG_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _append_locked(line: str) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    _trim_locked()


def _write(line: str) -> None:
    with _lock:
        _append_locked(line)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _mirror_to_gui(speaker: str, text: str, kind: str) -> None:
    if not bool(getattr(config, "GUI_ENABLED", False)):
        return
    try:
        from gui.state_bridge import gui_bridge
        gui_bridge.add_conversation_line(speaker, text, kind=kind)
    except Exception:
        pass


def log_heard(speaker: str | None, text: str) -> None:
    """Log a transcribed utterance. speaker is a name or None for unknown."""
    label = speaker.strip() if speaker and speaker.strip() else "Unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write(f"{ts} | HEARD | {label}: {text}")
    _mirror_to_gui(label if label != "Unknown" else "Unknown speaker", text, "user")


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
        _append_locked(f"{ts} | REX   | {text.strip()}")
    _mirror_to_gui("Rex", text.strip(), "rex")


def log_system(text: str) -> None:
    """Log an important system message to the GUI conversation panel."""
    if not text or not text.strip():
        return
    _mirror_to_gui("System", text.strip(), "system")


def clear_dedupe_state() -> None:
    """Test/debug hook."""
    global _last_rex_norm, _last_rex_at
    with _lock:
        _last_rex_norm = ""
        _last_rex_at = 0.0
