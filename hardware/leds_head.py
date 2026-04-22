"""
Head Arduino (Uno) LED controller — 82 WS2812B NeoPixels.

Pixels 0–1: eyes (RGB).
Pixels 2–81: mouth trapezoid PCB (physically GRB, but the Arduino handles the swap
in its EMOTION_COLORS table — Python sends plain RGB unchanged).

All operations are no-ops (with a debug log) when HEAD_LEDS_ENABLED is False.
"""

import logging
import threading
import time

import serial

import config
from utils.config_loader import ARDUINO_HEAD_PORT, HEAD_LEDS_ENABLED

_log = logging.getLogger(__name__)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_DROP_REPORT_INTERVAL_SECS = 5.0
_dropped_counts: dict[str, int] = {}
_drop_window_started_at = 0.0
_next_drop_report_at = 0.0
_speech_drop_notified = False


def _cmd_family(cmd: str) -> str:
    return (cmd.split(":", 1)[0].strip().upper() or "UNKNOWN")


def _is_speech_led_command(family: str) -> bool:
    return family in {"SPEAK", "SPEAK_LEVEL", "SPEAK_STOP"}


def _report_drops_if_due(now: float) -> None:
    global _dropped_counts, _drop_window_started_at, _next_drop_report_at
    if not _dropped_counts or now < _next_drop_report_at:
        return
    total = sum(_dropped_counts.values())
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(_dropped_counts.items()))
    elapsed = now - _drop_window_started_at
    _log.warning(
        "Head Arduino not connected — dropped %d command(s) in %.1fs (%s). "
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
        "Head Arduino %s — %d command(s) were dropped over %.1fs (%s).",
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
    global _ser, _speech_drop_notified
    if not HEAD_LEDS_ENABLED:
        _log.debug("HEAD_LEDS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(ARDUINO_HEAD_PORT, config.HEAD_ARDUINO_BAUD, timeout=1)
        _log.info("Head Arduino connected on %s at %d baud", ARDUINO_HEAD_PORT, config.HEAD_ARDUINO_BAUD)
        _flush_drop_summary("reconnected")
        _speech_drop_notified = False
        return True
    except serial.SerialException as exc:
        _log.error("Failed to open head Arduino port %s: %s", ARDUINO_HEAD_PORT, exc)
        _ser = None
        return False


def disconnect() -> None:
    global _ser, _speech_drop_notified
    with _lock:
        if _ser and _ser.is_open:
            _ser.close()
        _ser = None
        _speech_drop_notified = False


# ── Transport ──────────────────────────────────────────────────────────────────

def send_command(cmd: str) -> None:
    """Send a newline-terminated command string to the head Arduino."""
    global _speech_drop_notified
    if not HEAD_LEDS_ENABLED:
        _log.debug("send_command no-op: HEAD_LEDS_ENABLED=False (cmd=%r)", cmd)
        return
    family = _cmd_family(cmd)
    with _lock:
        if _ser is None or not _ser.is_open:
            if _is_speech_led_command(family):
                if not _speech_drop_notified:
                    _log.warning(
                        "Head Arduino not connected — ignoring mouth LED updates for this speech routine."
                    )
                    _speech_drop_notified = True
                if family == "SPEAK_STOP":
                    _speech_drop_notified = False
                return
            _record_drop(cmd)
            return
        if family == "SPEAK_STOP":
            _speech_drop_notified = False
        _flush_drop_summary("is online")
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
