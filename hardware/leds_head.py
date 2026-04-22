"""
Head Arduino (Uno) LED controller — 82 WS2812B NeoPixels.

Pixels 0–1: eyes (RGB).
Pixels 2–81: mouth trapezoid PCB (physically GRB, but the Arduino handles the swap
in its EMOTION_COLORS table — Python sends plain RGB unchanged).

All operations are no-ops (with a debug log) when HEAD_LEDS_ENABLED is False.
"""

import logging
import threading

import serial

import config
from utils.config_loader import ARDUINO_HEAD_PORT, HEAD_LEDS_ENABLED

_log = logging.getLogger(__name__)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()


# ── Connection ─────────────────────────────────────────────────────────────────

def connect() -> bool:
    global _ser
    if not HEAD_LEDS_ENABLED:
        _log.debug("HEAD_LEDS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(ARDUINO_HEAD_PORT, config.HEAD_ARDUINO_BAUD, timeout=1)
        _log.info("Head Arduino connected on %s at %d baud", ARDUINO_HEAD_PORT, config.HEAD_ARDUINO_BAUD)
        return True
    except serial.SerialException as exc:
        _log.error("Failed to open head Arduino port %s: %s", ARDUINO_HEAD_PORT, exc)
        _ser = None
        return False


def disconnect() -> None:
    global _ser
    with _lock:
        if _ser and _ser.is_open:
            _ser.close()
        _ser = None


# ── Transport ──────────────────────────────────────────────────────────────────

def send_command(cmd: str) -> None:
    """Send a newline-terminated command string to the head Arduino."""
    if not HEAD_LEDS_ENABLED:
        _log.debug("send_command no-op: HEAD_LEDS_ENABLED=False (cmd=%r)", cmd)
        return
    with _lock:
        if _ser is None or not _ser.is_open:
            _log.warning("Head Arduino not connected — dropping command %r", cmd)
            return
        _ser.write((cmd + "\n").encode())


# ── Command API ────────────────────────────────────────────────────────────────

def speak(emotion: str) -> None:
    """Start mouth speak animation for the given emotion. Eyes stay unchanged."""
    send_command(f"SPEAK:{emotion}")


def speak_level(brightness: int) -> None:
    """Set mouth brightness directly (0–255). Used to drive LEDs from audio level."""
    brightness = max(0, min(255, brightness))
    send_command(f"SPEAK_LEVEL:{brightness}")


def speak_stop() -> None:
    """Stop the mouth speak animation and return to idle pattern."""
    send_command("SPEAK_STOP")


def idle() -> None:
    """Enter idle LED pattern (slow breathing pulse)."""
    send_command("IDLE")


def active() -> None:
    """Enter active LED pattern (brighter, more energetic)."""
    send_command("ACTIVE")


def set_eye_color(r: int, g: int, b: int) -> None:
    """
    Set eye pixels 0–1 to an RGB color.
    Eyes are standard RGB LEDs — values are passed through unchanged.
    """
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    send_command(f"EYE:{r},{g},{b}")


def set_eye_emotion(emotion: str) -> None:
    """Convenience wrapper: looks up emotion in config.EYE_COLORS and sets eye color."""
    color = config.EYE_COLORS.get(emotion, config.EYE_COLORS["neutral"])
    set_eye_color(*color)


def off() -> None:
    """Turn all head LEDs off immediately."""
    send_command("OFF")


def sleep() -> None:
    """Enter sleep LED state (eyes off, mouth dim or off)."""
    send_command("SLEEP")
