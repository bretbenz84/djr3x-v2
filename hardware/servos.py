"""
Pololu Maestro Mini 18 servo controller.

All operations are no-ops (with a debug log) when SERVOS_ENABLED is False.
Channel numbers, limits, and neutral positions come from config.py.
"""

import logging
import math
import random
import struct
import threading
import time

import serial

import config
from utils.config_loader import MAESTRO_PORT, SERVOS_ENABLED

_log = logging.getLogger(__name__)

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_CHANNEL_TO_NAME = {
    cfg["ch"]: name
    for name, cfg in config.SERVO_CHANNELS.items()
}

# breathing_thread stop event — set by shutdown()
_stop_breathing = threading.Event()
_breathing_emotion = "neutral"
_breathing_lock = threading.Lock()

# ── Channel index lookups ──────────────────────────────────────────────────────

def _channel_cfg(channel: int) -> "dict | None":
    for cfg in config.SERVO_CHANNELS.values():
        if cfg["ch"] == channel:
            return cfg
    return None


def _clamp(channel: int, position: int) -> int:
    cfg = _channel_cfg(channel)
    if cfg is None:
        return position
    return max(cfg["min"], min(cfg["max"], position))


def _derive_body_state(positions: dict) -> str:
    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]
    tilt_cfg = config.SERVO_CHANNELS["headtilt"]

    neck = positions.get("neck", neck_cfg["neutral"])
    lift = positions.get("headlift", lift_cfg["neutral"])
    tilt = positions.get("headtilt", tilt_cfg["neutral"])

    neck_dead = 450
    lift_dead = 450
    tilt_dead = 250
    if neck <= neck_cfg["neutral"] - neck_dead:
        return "looking_left"
    if neck >= neck_cfg["neutral"] + neck_dead:
        return "looking_right"
    # Headtilt is inverted: lower value = looking up, higher value = looking down.
    if lift >= lift_cfg["neutral"] + lift_dead or tilt <= tilt_cfg["neutral"] - tilt_dead:
        return "looking_up"
    if lift <= lift_cfg["neutral"] - lift_dead or tilt >= tilt_cfg["neutral"] + tilt_dead:
        return "looking_down"
    return "neutral"


def _record_servo_positions(channel_dict: "dict[int, int]") -> None:
    """Mirror commanded servo positions into WorldState proprioception."""
    updates = {
        _CHANNEL_TO_NAME[ch]: _clamp(ch, int(pos))
        for ch, pos in channel_dict.items()
        if ch in _CHANNEL_TO_NAME
    }
    if not updates:
        return
    try:
        from world_state import world_state

        self_state = world_state.get("self_state")
        positions = dict(self_state.get("servo_positions") or {})
        positions.update(updates)
        self_state["servo_positions"] = positions
        self_state["body_state"] = _derive_body_state(positions)
        world_state.update("self_state", self_state)
    except Exception as exc:
        _log.debug("servo proprioception update failed: %s", exc)


# ── Serial connection ──────────────────────────────────────────────────────────

def connect() -> bool:
    global _ser
    if not SERVOS_ENABLED:
        _log.debug("SERVOS_ENABLED=False — skipping connect")
        return False
    try:
        _ser = serial.Serial(MAESTRO_PORT, config.SERVO_BAUD, timeout=0.1)
        _log.info("Maestro connected on %s at %d baud", MAESTRO_PORT, config.SERVO_BAUD)
        return True
    except serial.SerialException as exc:
        _log.error("Failed to open Maestro port %s: %s", MAESTRO_PORT, exc)
        _ser = None
        return False


def disconnect() -> None:
    global _ser
    _stop_breathing.set()
    with _lock:
        if _ser and _ser.is_open:
            _ser.close()
        _ser = None


# ── Core command primitives ────────────────────────────────────────────────────

def _send_set_target(channel: int, position: int) -> None:
    """Send Maestro compact protocol Set Target command (0x84)."""
    if _ser is None or not _ser.is_open:
        return
    low  = position & 0x7F
    high = (position >> 7) & 0x7F
    cmd  = bytes([0x84, channel, low, high])
    _ser.write(cmd)


def set_servo(channel: int, position: int) -> None:
    """Move channel to position (quarter-microseconds), clamped to channel limits."""
    if not SERVOS_ENABLED:
        _log.debug("set_servo no-op: SERVOS_ENABLED=False (ch=%d pos=%d)", channel, position)
        return
    position = _clamp(channel, position)
    with _lock:
        _send_set_target(channel, position)
    _record_servo_positions({channel: position})


def get_servo(channel: int) -> "int | None":
    """
    Read actual servo position from Maestro (proprioception).
    Returns position in quarter-microseconds, or None on failure.
    """
    if not SERVOS_ENABLED:
        _log.debug("get_servo no-op: SERVOS_ENABLED=False (ch=%d)", channel)
        return None
    with _lock:
        if _ser is None or not _ser.is_open:
            return None
        _ser.write(bytes([0x90, channel]))
        data = _ser.read(2)
        if len(data) < 2:
            _log.warning("get_servo: short read for ch=%d", channel)
            return None
        return struct.unpack("<H", data)[0]


def set_servos(channel_dict: "dict[int, int]") -> None:
    """Set multiple channels in one pass. channel_dict maps channel int → position."""
    if not SERVOS_ENABLED:
        _log.debug("set_servos no-op: SERVOS_ENABLED=False")
        return
    with _lock:
        for channel, position in channel_dict.items():
            _send_set_target(channel, _clamp(channel, position))
    _record_servo_positions(channel_dict)


# ── High-level behaviours ──────────────────────────────────────────────────────

def neutral(step_us: int = 40, step_delay: float = 0.02) -> None:
    """
    Move all channels smoothly to their neutral positions.
    Reads current positions first, then interpolates to neutral.
    """
    if not SERVOS_ENABLED:
        _log.debug("neutral() no-op: SERVOS_ENABLED=False")
        return

    targets = {name: cfg["neutral"] for name, cfg in config.SERVO_CHANNELS.items()}

    # Read current positions
    current: dict[str, int] = {}
    for name, cfg in config.SERVO_CHANNELS.items():
        pos = get_servo(cfg["ch"])
        current[name] = pos if pos is not None else cfg["neutral"]

    # Step toward neutral
    done = False
    while not done:
        done = True
        moves: dict[int, int] = {}
        for name, cfg in config.SERVO_CHANNELS.items():
            cur  = current[name]
            tgt  = targets[name]
            diff = tgt - cur
            if diff == 0:
                continue
            done     = False
            step     = min(step_us, abs(diff)) * (1 if diff > 0 else -1)
            new_pos  = cur + step
            current[name] = new_pos
            moves[cfg["ch"]] = new_pos
        if moves:
            with _lock:
                for ch, pos in moves.items():
                    _send_set_target(ch, pos)
            time.sleep(step_delay)
    _record_servo_positions({cfg["ch"]: cfg["neutral"] for cfg in config.SERVO_CHANNELS.values()})


def set_breathing_emotion(emotion: str) -> None:
    """Update the emotion state that controls breathing speed."""
    global _breathing_emotion
    with _breathing_lock:
        _breathing_emotion = emotion


def breathing_thread() -> None:
    """
    Background thread: slow sinusoidal oscillation on the headlift servo.
    Amplitude and period come from config.py. Stops cleanly when _stop_breathing is set.
    Call this as a daemon thread from main.py.
    """
    if not SERVOS_ENABLED:
        _log.debug("breathing_thread no-op: SERVOS_ENABLED=False")
        return

    _log.info("Breathing thread started")
    headlift_cfg = config.SERVO_CHANNELS["headlift"]
    channel      = headlift_cfg["ch"]
    neutral_pos  = headlift_cfg["neutral"]
    amplitude    = config.BREATHING_AMPLITUDE_QUS

    tick = 0.05  # seconds between position updates

    while not _stop_breathing.is_set():
        with _breathing_lock:
            emotion = _breathing_emotion

        if emotion == "excited":
            period = config.BREATHING_PERIOD_EXCITED
        elif emotion == "sad":
            period = config.BREATHING_PERIOD_SAD
        else:
            period = config.BREATHING_PERIOD_SECS

        t   = time.monotonic()
        pos = int(neutral_pos + amplitude * math.sin(2 * math.pi * t / period))
        pos = _clamp(channel, pos)

        with _lock:
            _send_set_target(channel, pos)

        _stop_breathing.wait(tick)

    _log.info("Breathing thread stopped")


def idle_animation() -> None:
    """
    One cycle of random small movements on neck and headlift channels.
    Intended to be called periodically from the consciousness loop during IDLE.
    """
    if not SERVOS_ENABLED:
        _log.debug("idle_animation no-op: SERVOS_ENABLED=False")
        return

    neck_cfg  = config.SERVO_CHANNELS["neck"]
    lift_cfg  = config.SERVO_CHANNELS["headlift"]

    # Small random offsets from neutral (±200 quarter-microseconds)
    neck_offset = random.randint(-200, 200)
    lift_offset = random.randint(-150, 150)

    neck_pos = _clamp(neck_cfg["ch"], neck_cfg["neutral"] + neck_offset)
    lift_pos = _clamp(lift_cfg["ch"], lift_cfg["neutral"] + lift_offset)

    set_servos({neck_cfg["ch"]: neck_pos, lift_cfg["ch"]: lift_pos})
    time.sleep(random.uniform(0.8, 2.0))
    set_servos({neck_cfg["ch"]: neck_cfg["neutral"], lift_cfg["ch"]: lift_cfg["neutral"]})


def move_to(targets: "dict[int, int]", step_us: int = 40, step_delay: float = 0.02) -> None:
    """Smoothly interpolate specific channels to target positions (quarter-microseconds)."""
    if not SERVOS_ENABLED:
        _log.debug("move_to no-op: SERVOS_ENABLED=False")
        return

    current: dict[int, int] = {}
    for ch, tgt in targets.items():
        pos = get_servo(ch)
        current[ch] = pos if pos is not None else tgt

    done = False
    while not done:
        done = True
        moves: dict[int, int] = {}
        for ch, tgt in targets.items():
            cur = current[ch]
            diff = tgt - cur
            if diff == 0:
                continue
            done = False
            step = min(step_us, abs(diff)) * (1 if diff > 0 else -1)
            new_pos = cur + step
            current[ch] = new_pos
            moves[ch] = new_pos
        if moves:
            with _lock:
                for ch, pos in moves.items():
                    _send_set_target(ch, _clamp(ch, pos))
            time.sleep(step_delay)
    _record_servo_positions(targets)


def stop_breathing() -> None:
    """Signal the breathing thread to stop. Returns immediately; thread exits within ~50 ms."""
    _stop_breathing.set()


def shutdown() -> None:
    """Stop breathing thread and cleanly disconnect. Call before process exit."""
    _stop_breathing.set()
    disconnect()
