"""
Chest Arduino (Nano) LED controller — 98 WS2811 LEDs.

FastLED is configured with COLOR_ORDER GRB on the Arduino side — no byte-order
compensation needed here. Python sends commands as plain strings.

All operations are no-ops (with a debug log) when CHEST_LEDS_ENABLED is False.
"""

import logging
import threading

import serial

import config
from utils.config_loader import ARDUINO_CHEST_PORT, CHEST_LEDS_ENABLED

_log = logging.getLogger(__name__)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()


# ── Connection ─────────────────────────────────────────────────────────────────

def connect() -> bool:
    global _ser
    if not CHEST_LEDS_ENABLED:
        _log.debug("CHEST_LEDS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD, timeout=1)
        _log.info("Chest Arduino connected on %s at %d baud", ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD)
        return True
    except serial.SerialException as exc:
        _log.error("Failed to open chest Arduino port %s: %s", ARDUINO_CHEST_PORT, exc)
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
    """Send a newline-terminated command string to the chest Arduino."""
    if not CHEST_LEDS_ENABLED:
        _log.debug("send_command no-op: CHEST_LEDS_ENABLED=False (cmd=%r)", cmd)
        return
    with _lock:
        if _ser is None or not _ser.is_open:
            _log.warning("Chest Arduino not connected — dropping command %r", cmd)
            return
        _ser.write((cmd + "\n").encode())


# ── Command API ────────────────────────────────────────────────────────────────

def startup() -> None:
    """Play startup light sequence."""
    send_command("STARTUP")


def idle() -> None:
    """Enter idle LED pattern (default: RandomBlocks2)."""
    send_command("IDLE")


def active() -> None:
    """Enter active LED pattern."""
    send_command("ACTIVE")


def speak(emotion: str) -> None:
    """
    Enter speak pattern for the given emotion.
    Emotion colors: excited=red, sad=blue, angry=rapid flash, happy=confetti.
    """
    send_command(f"SPEAK:{emotion}")


def sleep() -> None:
    """Enter sleep LED state."""
    send_command("SLEEP")


def off() -> None:
    """Turn all chest LEDs off immediately."""
    send_command("OFF")


def next_pattern() -> None:
    """Cycle to the next built-in LED pattern."""
    send_command("NEXT")
