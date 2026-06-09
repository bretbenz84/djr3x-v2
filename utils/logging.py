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

    file_handler: logging.Handler
    if getattr(config, "DEBUG_MODE", False):
        run_log_file = _LOG_DIR / f"djr3x-{run_stamp()}.log"
        file_handler = logging.FileHandler(run_log_file, encoding="utf-8")
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
