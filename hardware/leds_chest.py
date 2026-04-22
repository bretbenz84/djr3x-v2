"""
Chest Arduino (Nano) LED controller — 98 WS2811 LEDs.

FastLED is configured with COLOR_ORDER GRB on the Arduino side — no byte-order
compensation needed here. Python sends commands as plain strings.

All operations are no-ops (with a debug log) when CHEST_LEDS_ENABLED is False.
"""

import logging
import threading
import time

import serial

import config
from utils.config_loader import ARDUINO_CHEST_PORT, CHEST_LEDS_ENABLED

_log = logging.getLogger(__name__)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_DROP_REPORT_INTERVAL_SECS = 5.0
_dropped_counts: dict[str, int] = {}
_drop_window_started_at = 0.0
_next_drop_report_at = 0.0


def _cmd_family(cmd: str) -> str:
    return (cmd.split(":", 1)[0].strip().upper() or "UNKNOWN")


def _report_drops_if_due(now: float) -> None:
    global _dropped_counts, _drop_window_started_at, _next_drop_report_at
    if not _dropped_counts or now < _next_drop_report_at:
        return
    total = sum(_dropped_counts.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(_dropped_counts.items()))
    elapsed = now - _drop_window_started_at
    _log.warning(
        "Chest Arduino not connected — dropped %d command(s) in %.1fs (%s). "
        "Suppressing per-command logs; summary repeats every %.0fs while disconnected.",
        total,
        elapsed,
        breakdown,
        _DROP_REPORT_INTERVAL_SECS,
    )
    _dropped_counts = {}
    _drop_window_started_at = now
    _next_drop_report_at = now + _DROP_REPORT_INTERVAL_SECS


def _record_drop(cmd: str) -> None:
    global _drop_window_started_at, _next_drop_report_at
    now = time.monotonic()
    if not _dropped_counts:
        _drop_window_started_at = now
        _next_drop_report_at = now  # report first drop immediately
    family = _cmd_family(cmd)
    _dropped_counts[family] = _dropped_counts.get(family, 0) + 1
    _report_drops_if_due(now)


def _flush_drop_summary(reason: str) -> None:
    """Emit one final drop summary (if pending) and clear counters."""
    global _dropped_counts, _drop_window_started_at, _next_drop_report_at
    if not _dropped_counts:
        return
    now = time.monotonic()
    total = sum(_dropped_counts.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(_dropped_counts.items()))
    elapsed = now - _drop_window_started_at
    _log.info(
        "Chest Arduino %s — %d command(s) were dropped over %.1fs (%s).",
        reason,
        total,
        elapsed,
        breakdown,
    )
    _dropped_counts = {}
    _drop_window_started_at = 0.0
    _next_drop_report_at = 0.0


# ── Connection ─────────────────────────────────────────────────────────────────

def connect() -> bool:
    global _ser
    if not CHEST_LEDS_ENABLED:
        _log.debug("CHEST_LEDS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD, timeout=1)
        _log.info("Chest Arduino connected on %s at %d baud", ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD)
        _flush_drop_summary("reconnected")
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
            _record_drop(cmd)
            return
        _flush_drop_summary("is online")
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
