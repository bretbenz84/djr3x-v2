"""
Single entry point for all project configuration.

Loads apikeys.py (API credentials) and .env (hardware device paths).
config.py can still be imported directly anywhere — it's plain Python with no side effects.

Raises RuntimeError at import time if any required API key contains a placeholder value.
Logs a warning and sets the corresponding *_ENABLED flag to False for missing hardware config.

Usage:
    from utils.config_loader import OPENAI_API_KEY, ELEVENLABS_API_KEY
    from utils.config_loader import CAMERA_INDEX, MAESTRO_PORT
    from utils.config_loader import CAMERA_ENABLED, SERVOS_ENABLED
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

_log = logging.getLogger(__name__)

# ── API Keys (from apikeys.py) ─────────────────────────────────────────────────

try:
    import apikeys as _apikeys
except ImportError:
    raise RuntimeError(
        "apikeys.py not found. Copy apikeys.example.py to apikeys.py and fill in real credentials."
    )

_PLACEHOLDER_SUFFIXES = ("...",)
_PLACEHOLDER_PREFIXES = ("your-", "your_")


def _is_placeholder(value: str) -> bool:
    if not value:
        return True
    v = value.lower()
    return any(v.endswith(s) for s in _PLACEHOLDER_SUFFIXES) or any(
        v.startswith(p) for p in _PLACEHOLDER_PREFIXES
    )


OPENAI_API_KEY: str = getattr(_apikeys, "OPENAI_API_KEY", "")
ELEVENLABS_API_KEY: str = getattr(_apikeys, "ELEVENLABS_API_KEY", "")

_bad = [
    name
    for name, val in (
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
    )
    if _is_placeholder(val)
]
if _bad:
    raise RuntimeError(
        f"Required API key(s) still contain placeholder values in apikeys.py: {', '.join(_bad)}\n"
        "Edit apikeys.py and replace them with real credentials before starting."
    )

# ── Hardware Config (from .env) ────────────────────────────────────────────────


def _require_port(env_key: str, label: str) -> "str | None":
    val = os.getenv(env_key, "").strip()
    if not val:
        _log.warning("Hardware config missing: %s not set — %s disabled.", env_key, label)
        return None
    return val


def _require_camera_index(env_key: str) -> "int | None":
    val = os.getenv(env_key, "").strip()
    if not val:
        _log.warning("Hardware config missing: %s not set — camera disabled.", env_key)
        return None
    try:
        return int(val)
    except ValueError:
        _log.warning(
            "Hardware config invalid: %s=%r is not an integer — camera disabled.", env_key, val
        )
        return None


CAMERA_INDEX: "int | None" = _require_camera_index("CAMERA_INDEX")
MAESTRO_PORT: "str | None" = _require_port("MAESTRO_PORT", "servo controller")
ARDUINO_HEAD_PORT: "str | None" = _require_port("ARDUINO_HEAD_PORT", "head LEDs")
ARDUINO_CHEST_PORT: "str | None" = _require_port("ARDUINO_CHEST_PORT", "chest LEDs")

CAMERA_ENABLED: bool = CAMERA_INDEX is not None
SERVOS_ENABLED: bool = MAESTRO_PORT is not None
HEAD_LEDS_ENABLED: bool = ARDUINO_HEAD_PORT is not None
CHEST_LEDS_ENABLED: bool = ARDUINO_CHEST_PORT is not None
