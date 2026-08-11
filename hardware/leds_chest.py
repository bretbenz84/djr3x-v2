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


def _mirror_gui_chest_led_state(cmd: str) -> None:
    """Best-effort mode-level mirror for the GUI avatar's chest panels.

    Runs BEFORE the enabled/connected checks: the avatar shows intended state
    even on a dev Mac with no chest Arduino (mirrors leds_head's GUI bridge).
    NEXT is a pattern cycle within the current mode, so it changes nothing here;
    COMPLIMENT is a one-shot flash overlay that keeps the underlying mode.
    """
    try:
        from gui.state_bridge import gui_bridge

        family = _cmd_family(cmd)
        parts = [p.strip() for p in cmd.split(":")]
        if family in {"STARTUP", "IDLE", "ACTIVE", "SLEEP", "OFF", "FADEOFF"}:
            gui_bridge.update_chest_led_state(mode=family.lower())
        elif family == "SPEAK":
            emotion = parts[1].lower() if len(parts) > 1 and parts[1] else "neutral"
            gui_bridge.update_chest_led_state(mode="speak", emotion=emotion)
        elif family == "CHARGE":
            soc = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            charging = len(parts) > 2 and parts[2] == "1"
            gui_bridge.update_chest_led_state(mode="charge", soc=soc, charging=charging)
        elif family == "COMPLIMENT":
            gui_bridge.update_chest_led_state(flash=True)
    except Exception:
        pass


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
    """Open the chest Arduino, retrying briefly like servos.py and motion.py.

    The retries matter at startup: the menu bar LED console (and, while Rex is
    off, the battery meter painting the charge gauge) hold this board and only
    let go on their ~1 Hz single-instance-lock poll. main.py now waits for that
    release explicitly (utils/port_handoff.py), so these attempts are the
    backstop, not the plan.
    """
    global _ser
    if not CHEST_LEDS_ENABLED:
        _log.debug("CHEST_LEDS_ENABLED=False — skipping connect")
        return False

    attempts = max(1, int(getattr(config, "CHEST_ARDUINO_CONNECT_RETRY_ATTEMPTS", 3)))
    delay = max(0.0, float(getattr(config, "CHEST_ARDUINO_CONNECT_RETRY_DELAY_SECS", 0.5)))

    for attempt in range(1, attempts + 1):
        try:
            _ser = serial.Serial(ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD, timeout=1)
        except serial.SerialException as exc:
            _ser = None
            _log.log(
                logging.ERROR if attempt == attempts else logging.WARNING,
                "Failed to open chest Arduino port %s (attempt %d/%d): %s",
                ARDUINO_CHEST_PORT, attempt, attempts, exc,
            )
            if attempt < attempts and delay:
                time.sleep(delay)
            continue

        # Opening the port toggles DTR on CH340 adapters, resetting the Arduino.
        # Wait for boot to complete before sending any commands.
        time.sleep(2.0)
        _ser.reset_input_buffer()
        _log.info(
            "Chest Arduino connected on %s at %d baud (attempt %d/%d)",
            ARDUINO_CHEST_PORT, config.CHEST_ARDUINO_BAUD, attempt, attempts,
        )
        _flush_drop_summary("reconnected")
        return True
    return False


def disconnect() -> None:
    global _ser
    with _lock:
        if _ser and _ser.is_open:
            _ser.close()
        _ser = None


def connected() -> bool:
    """True when the chest-LED Arduino serial link is open (live status)."""
    ser = _ser
    return ser is not None and bool(getattr(ser, "is_open", False))


# ── Transport ──────────────────────────────────────────────────────────────────

def send_command(cmd: str) -> None:
    """Send a newline-terminated command string to the chest Arduino."""
    _mirror_gui_chest_led_state(cmd)
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
    Emotion patterns: excited=racing gold/red pops, sad=slow blue sighs,
    angry=red alert, happy=bouncing gold + confetti.
    """
    send_command(f"SPEAK:{emotion}")


def sleep() -> None:
    """Enter sleep LED state."""
    send_command("SLEEP")


def charge_status(soc: int, charging: bool) -> None:
    """Show the off-state contiguous 24-LED battery meter."""
    send_command(f"CHARGE:{max(0, min(100, int(soc)))}:{1 if charging else 0}")


def off() -> None:
    """Turn all chest LEDs off immediately."""
    send_command("OFF")


def fade_off() -> None:
    """Smoothly fade the chest LEDs to black instead of an instant off — a lifelike
    power-down for shutdown. The firmware freezes the current frame and ramps
    brightness to 0 over ~4s autonomously, so this returns immediately."""
    send_command("FADEOFF")


def next_pattern() -> None:
    """Cycle to the next built-in LED pattern."""
    send_command("NEXT")


def compliment_flash() -> None:
    """Celebratory white<->blue flash — the chest reaction to a compliment (the
    positive mirror of the red angry-flash insults get). One-shot: the firmware
    self-terminates the flash after a couple of seconds (or until the next command),
    so it can't get stuck on."""
    send_command("COMPLIMENT")
