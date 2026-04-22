"""
Centralized logging setup for DJ-R3X.

Call setup_logging() once at startup (from main.py) before any other modules log.
All other modules should get their logger with:

    from utils.logging import get_logger
    log = get_logger(__name__)
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "djr3x.log"
_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with console and rotating file handlers."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

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
