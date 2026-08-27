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


def _strip_audio_tags(text: str) -> str:
    """Belt-and-braces: Rex-line text may carry v3 [audio tags] (delivery shaping for
    ElevenLabs — authored seam lines or LLM-emitted). They must never reach the
    transcript or GUI, so every Rex write seam scrubs them here."""
    try:
        from utils.audio_tags import strip_audio_tags
        return strip_audio_tags(text)
    except Exception:
        return text


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
    # A human turn between two identical Rex lines proves they are two real
    # answers, not the enqueue-then-blocking-return double-write log_rex's
    # window guards against. Field 2026-08-26: Jeopardy re-asked "Pick a dollar
    # value too..." twice more at 20:19:34 and 20:19:45 — both audible, both
    # erased from the transcript, which read as Rex ignoring the player.
    clear_dedupe_state()
    _write(f"{ts} | HEARD | {label}: {text}")
    _mirror_to_gui(label if label != "Unknown" else "Unknown speaker", text, "user")


def log_rex(text: str, *, to_gui: bool = True) -> None:
    """Log something Rex said.

    `to_gui=False` writes the on-disk transcript (and updates the dedupe window)
    but skips the GUI conversation panel — used by the streaming reply path, which
    has already filled the GUI bubble sentence-by-sentence via log_rex_stream() and
    would otherwise re-add the whole reply as a duplicate line."""
    global _last_rex_norm, _last_rex_at
    text = _strip_audio_tags(text)
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
    # Feed the line to the ASR context-biasing window: the user's reply usually
    # re-uses entities Rex just named ("Lake Folsom" → "like falsum" field bug
    # 2026-08-02). Lazy import + best-effort so logging never breaks on it.
    try:
        from audio import transcription
        transcription.note_rex_line(text)
    except Exception:
        pass
    if to_gui:
        _mirror_to_gui("Rex", text.strip(), "rex")


def claim_rex_line(text: str) -> None:
    """Mark `text` as already written, without writing it again.

    For a handler that speaks SEVERAL lines and returns only one of them. The
    caller logs whatever it gets back, which lands after everything the handler
    already logged — so the impersonation bit wrote intro, then outro, then the
    parody it returned, and the GUI showed the punchline after the bow (field
    2026-08-04). The handler logs the parody at the moment it is spoken and
    claims it here; the caller's later write then dedupes away.

    Only the immediately-previous line is compared, so this has to be called
    AFTER the handler's last write, not before it.
    """
    global _last_rex_norm, _last_rex_at
    text = _strip_audio_tags(text)
    if not text or not text.strip():
        return
    with _lock:
        _last_rex_norm = _normalize(text)
        _last_rex_at = time.monotonic()


def log_rex_stream(text: str) -> None:
    """Stream one freshly-generated reply sentence to the GUI conversation panel.

    GUI-only (no on-disk write, no dedupe): the sentences grow a single Rex bubble
    in place so the reply text appears in the dashboard the moment it is generated,
    reading along with the TTS, instead of after playback finishes. The on-disk
    transcript is written once at the end via log_rex(..., to_gui=False)."""
    text = _strip_audio_tags(text).strip()
    if not text or not bool(getattr(config, "GUI_ENABLED", False)):
        return
    try:
        from gui.state_bridge import gui_bridge
        gui_bridge.append_rex_stream(text)
    except Exception:
        pass


def finish_rex_stream(full_text: str | None = None) -> None:
    """Close out the streamed Rex bubble (see log_rex_stream). Clears the streaming
    marker so the next reply starts fresh; `full_text`, when given, normalizes the
    bubble to the canonical reply text."""
    if not bool(getattr(config, "GUI_ENABLED", False)):
        return
    if full_text:
        full_text = _strip_audio_tags(full_text)
    try:
        from gui.state_bridge import gui_bridge
        gui_bridge.finish_rex_stream(full_text)
    except Exception:
        pass


def log_system(text: str) -> None:
    """Log an important system message to the GUI conversation panel."""
    if not text or not text.strip():
        return
    _mirror_to_gui("System", text.strip(), "system")


def clear_dedupe_state() -> None:
    """Forget the last Rex line for dedupe purposes. Called by log_heard (a human
    turn separates two identical Rex lines into two real events) and by tests."""
    global _last_rex_norm, _last_rex_at
    with _lock:
        _last_rex_norm = ""
        _last_rex_at = 0.0
