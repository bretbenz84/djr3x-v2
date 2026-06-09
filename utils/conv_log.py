"""
Plain-text conversational log at <project_root>/logs/conversation.log.

Each line is:
    YYYY-MM-DD HH:MM:SS | HEARD | <Speaker>: <text>
    YYYY-MM-DD HH:MM:SS | REX   | <text>

Call log_heard() when speech is transcribed, log_rex() when Rex speaks.
Thread-safe; appends only; creates the file on first write.
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import config

def _conversation_log_path() -> Path:
    """conversation.log normally; conversation-<run stamp>.log in DEBUG_MODE so each
    run keeps its own transcript, matching the per-run djr3x-<stamp>.log files."""
    base = Path(__file__).parent.parent / "logs"
    if getattr(config, "DEBUG_MODE", False):
        try:
            from utils.logging import run_stamp
            return base / f"conversation-{run_stamp()}.log"
        except Exception:
            pass
    return base / "conversation.log"


_LOG_PATH = _conversation_log_path()
# The real on-disk log, captured before any test patches _LOG_PATH. Writes to THIS
# path are suppressed under the test runner so `unittest discover` never clobbers a
# live run's conversation.log (the suite's conversation-flow tests call log_rex/
# log_heard; without this they overwrite/trim the real transcript). A test that
# patches _LOG_PATH to a temp file is exempt — it's exercising the writer on purpose.
_DEFAULT_LOG_PATH = _LOG_PATH
_lock = threading.Lock()


def _under_test_runner() -> bool:
    """True when running under unittest/pytest, keyed on the ENTRY POINT (sys.argv[0]
    / PYTEST_CURRENT_TEST) rather than 'unittest' in sys.modules — so an incidental
    import can't disable real logging on the robot (which runs `python main.py`)."""
    if os.environ.get("DJR3X_CONV_LOG_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def _writes_suppressed() -> bool:
    return _under_test_runner() and _LOG_PATH == _DEFAULT_LOG_PATH
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
    # Single on-disk write chokepoint (both _write and log_rex route here). Suppress
    # writes to the DEFAULT real log under the test runner so `unittest discover`
    # never clobbers a live run's conversation.log; a test that patched _LOG_PATH to
    # a temp file is exempt and writes normally.
    if _writes_suppressed():
        return
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
