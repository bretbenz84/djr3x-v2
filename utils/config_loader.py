"""
Single entry point for all project configuration.

Loads apikeys.py (API credentials) and .env (hardware device paths).
config.py also reads .env directly for build-specific servo limit overrides.
config.py can still be imported directly anywhere — it's plain Python with no side effects.

Raises RuntimeError at import time if any required API key contains a placeholder value.
Logs a warning and sets the corresponding *_ENABLED flag to False for missing hardware config.

Usage:
    from utils.config_loader import OPENAI_API_KEY, ELEVENLABS_API_KEY
    from utils.config_loader import CAMERA_INDEX, CAMERA_DEVICE_NAME, MAESTRO_PORT
    from utils.config_loader import CAMERA_ENABLED, CAMERA_SELECTION_DESCRIPTION, SERVOS_ENABLED
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


def _require_int_env(env_key: str, label: str) -> "int | None":
    val = os.getenv(env_key, "").strip()
    if not val:
        _log.warning("Hardware config missing: %s not set — %s disabled.", env_key, label)
        return None
    try:
        return int(val)
    except ValueError:
        _log.warning(
            "Hardware config invalid: %s=%r is not an integer — %s disabled.", env_key, val, label
        )
        return None


def _optional_env(env_key: str) -> "str | None":
    val = os.getenv(env_key, "").strip()
    return val or None


def _load_camera_config() -> "tuple[int | None, str | None]":
    raw_index = os.getenv("CAMERA_INDEX", "").strip()
    device_name = _optional_env("CAMERA_DEVICE_NAME")

    if not raw_index and not device_name:
        _log.warning(
            "Hardware config missing: neither CAMERA_INDEX nor CAMERA_DEVICE_NAME set — camera disabled."
        )
        return None, None

    camera_index = None
    if raw_index:
        try:
            camera_index = int(raw_index)
        except ValueError:
            _log.warning(
                "Hardware config invalid: CAMERA_INDEX=%r is not an integer — ignoring index.",
                raw_index,
            )

    return camera_index, device_name


def _load_audio_config() -> "tuple[int | None, str | None, str]":
    raw_index = os.getenv("AUDIO_DEVICE_INDEX", "").strip()
    device_name = _optional_env("AUDIO_DEVICE_NAME")

    if not raw_index and not device_name:
        _log.warning(
            "Hardware config missing: neither AUDIO_DEVICE_NAME nor AUDIO_DEVICE_INDEX set — microphone disabled."
        )
        return None, None, "disabled"

    fallback_index = None
    if raw_index:
        try:
            fallback_index = int(raw_index)
        except ValueError:
            _log.warning(
                "Hardware config invalid: AUDIO_DEVICE_INDEX=%r is not an integer — ignoring index.",
                raw_index,
            )

    if device_name:
        resolved = _resolve_sounddevice_input_name(device_name)
        if resolved is not None:
            return resolved, device_name, f'device name match "{device_name}" → index {resolved}'
        if fallback_index is None:
            return None, device_name, f'device name match "{device_name}" not found'
        _log.warning(
            "Audio device name %r was not found; falling back to AUDIO_DEVICE_INDEX=%d.",
            device_name,
            fallback_index,
        )
        return fallback_index, device_name, f'fallback index {fallback_index} (name "{device_name}" not found)'

    if fallback_index is not None:
        return fallback_index, None, f"index {fallback_index}"
    return None, None, "disabled"


def _resolve_sounddevice_input_name(device_name: str) -> "int | None":
    wanted = device_name.strip().lower()
    if not wanted:
        return None
    try:
        import sounddevice as sd
    except Exception as exc:
        _log.warning(
            "Hardware config invalid: AUDIO_DEVICE_NAME=%r but sounddevice is unavailable (%s) — microphone disabled.",
            device_name,
            exc,
        )
        return None

    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        _log.warning(
            "Could not query audio devices while resolving AUDIO_DEVICE_NAME=%r: %s",
            device_name,
            exc,
        )
        return None

    input_devices: list[tuple[int, str]] = []
    for idx, info in enumerate(devices):
        try:
            max_input_channels = int(info.get("max_input_channels", 0))
        except Exception:
            max_input_channels = 0
        if max_input_channels <= 0:
            continue
        name = str(info.get("name") or "").strip()
        if name:
            input_devices.append((idx, name))

    exact = [(idx, name) for idx, name in input_devices if name.lower() == wanted]
    if exact:
        if len(exact) > 1:
            _log.warning(
                "AUDIO_DEVICE_NAME=%r matched multiple input devices exactly; using index %d (%s).",
                device_name,
                exact[0][0],
                exact[0][1],
            )
        return exact[0][0]

    contains = [(idx, name) for idx, name in input_devices if wanted in name.lower()]
    if len(contains) == 1:
        return contains[0][0]
    if len(contains) > 1:
        matches = ", ".join(f"{idx}:{name}" for idx, name in contains)
        _log.warning(
            "AUDIO_DEVICE_NAME=%r matched multiple input devices (%s); use a more specific name.",
            device_name,
            matches,
        )
        return None

    available = ", ".join(f"{idx}:{name}" for idx, name in input_devices) or "no input devices"
    _log.warning(
        "AUDIO_DEVICE_NAME=%r did not match any input device. Available inputs: %s",
        device_name,
        available,
    )
    return None


CAMERA_INDEX, CAMERA_DEVICE_NAME = _load_camera_config()
AUDIO_DEVICE_INDEX, AUDIO_DEVICE_NAME, AUDIO_SELECTION_DESCRIPTION = _load_audio_config()
MAESTRO_PORT: "str | None" = _require_port("MAESTRO_PORT", "servo controller")
ARDUINO_HEAD_PORT: "str | None" = _require_port("ARDUINO_HEAD_PORT", "head LEDs")
ARDUINO_CHEST_PORT: "str | None" = _require_port("ARDUINO_CHEST_PORT", "chest LEDs")

if CAMERA_DEVICE_NAME:
    CAMERA_SELECTION_DESCRIPTION = f'device name match "{CAMERA_DEVICE_NAME}"'
elif CAMERA_INDEX is not None:
    CAMERA_SELECTION_DESCRIPTION = f"index {CAMERA_INDEX}"
else:
    CAMERA_SELECTION_DESCRIPTION = "disabled"

CAMERA_ENABLED: bool = CAMERA_INDEX is not None or CAMERA_DEVICE_NAME is not None
AUDIO_ENABLED: bool = AUDIO_DEVICE_INDEX is not None
SERVOS_ENABLED: bool = MAESTRO_PORT is not None
HEAD_LEDS_ENABLED: bool = ARDUINO_HEAD_PORT is not None
CHEST_LEDS_ENABLED: bool = ARDUINO_CHEST_PORT is not None
