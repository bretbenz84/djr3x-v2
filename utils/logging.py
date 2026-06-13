"""
Centralized logging setup for DJ-R3X.

Call setup_logging() once at startup (from main.py) before any other modules log.
All other modules should get their logger with:

    from utils.logging import get_logger
    log = get_logger(__name__)
"""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

import config

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "djr3x.log"
_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5

_RUN_STAMP: "str | None" = None
_ACTIVE_LOG_PATH: "Path | None" = None
_GUI_LOG_HANDLER: "logging.Handler | None" = None


def run_stamp() -> str:
    """The per-run timestamp (YYYY-MM-DD-HH-MM-SS), computed ONCE on first use and
    cached for the life of the process. Shared by every per-run log file (djr3x +
    conversation) so a single run's files carry the same stamp."""
    global _RUN_STAMP
    if _RUN_STAMP is None:
        _RUN_STAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return _RUN_STAMP


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with console + file handlers.

    config.DEBUG_MODE=True:  this run gets its OWN timestamped file,
        logs/djr3x-<YYYY-MM-DD-HH-MM-SS>.log, so runs accumulate as distinct per-run
        logs (nothing is cleared or overwritten).
    config.DEBUG_MODE=False: one shared logs/djr3x.log that size-rotates
        (10 MB x 5 backups) — the steady-state, bounded-size behavior.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    global _ACTIVE_LOG_PATH
    file_handler: logging.Handler
    if getattr(config, "DEBUG_MODE", False):
        run_log_file = _LOG_DIR / f"djr3x-{run_stamp()}.log"
        file_handler = logging.FileHandler(run_log_file, encoding="utf-8")
        _ACTIVE_LOG_PATH = run_log_file
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        _ACTIVE_LOG_PATH = _LOG_FILE
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)


def active_log_path() -> Path:
    """The file this run's app log is being written to (set by setup_logging).

    DEBUG_MODE=True → logs/djr3x-<run stamp>.log; otherwise the shared
    logs/djr3x.log. Falls back to the shared path if setup_logging hasn't run."""
    return _ACTIVE_LOG_PATH or _LOG_FILE


class _CallbackLogHandler(logging.Handler):
    """Forwards formatted records to a plain callable (e.g. the GUI bridge).

    The callback must be thread-safe and non-blocking — records arrive on
    whatever thread emitted them. emit() never raises and never logs, so a
    broken sink can't recurse into logging or take the app down."""

    def __init__(self, callback) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._callback(self.format(record))
        except Exception:
            pass


def install_gui_log_handler(callback, level: int = logging.INFO) -> None:
    """Mirror root-logger records (same format as the log file) into `callback`.

    Idempotent: a second install replaces the previous handler."""
    global _GUI_LOG_HANDLER
    root = logging.getLogger()
    if _GUI_LOG_HANDLER is not None:
        root.removeHandler(_GUI_LOG_HANDLER)
    handler = _CallbackLogHandler(callback)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.setLevel(level)
    root.addHandler(handler)
    _GUI_LOG_HANDLER = handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
