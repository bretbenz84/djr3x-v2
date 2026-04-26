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
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException, OSError)

_CMD_SET_TARGET = 0x84
_CMD_SET_SPEED = 0x87
_CMD_SET_ACCEL = 0x89
_CMD_GET_POSITION = 0x90

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_CHANNEL_TO_NAME = {
    cfg["ch"]: name
    for name, cfg in config.SERVO_CHANNELS.items()
}
_ALL_CHANNELS = sorted(_CHANNEL_TO_NAME)
_commanded_positions: dict[int, int] = {
    cfg["ch"]: cfg["neutral"]
    for cfg in config.SERVO_CHANNELS.values()
}
_last_reconnect_attempt_at = 0.0

# breathing_thread stop event — set by shutdown()
_stop_breathing = threading.Event()
_breathing_emotion = "neutral"
_breathing_lock = threading.Lock()

# Set while a scripted arm gesture or speech-reactive gesture owns arm channels.
_arm_idle_pause = threading.Event()

# Speech-reactive servo state.
_speech_active = threading.Event()
_speech_baseline: dict[int, int] = {}
_face_tracking_baseline: dict[int, int] = {}
_last_speech_move_at = 0.0
_speech_hand_counter = 0
_speech_elbow_target: int | None = None
_speech_elbow_direction = 1
_next_speech_elbow_at = 0.0

# ── Channel index lookups ──────────────────────────────────────────────────────

def _channel_cfg(channel: int) -> "dict | None":
    for cfg in config.SERVO_CHANNELS.values():
        if cfg["ch"] == channel:
            return cfg
    return None


def _channel(name: str) -> int:
    return int(config.SERVO_CHANNELS[name]["ch"])


def _clamp(channel: int, position: int) -> int:
    cfg = _channel_cfg(channel)
    if cfg is None:
        return position
    return max(cfg["min"], min(cfg["max"], position))


def _encode(cmd: int, channel: int, value: int) -> bytes:
    """Encode a Pololu compact protocol command."""
    return bytes([cmd, channel, value & 0x7F, (value >> 7) & 0x7F])


def _get_config_int(name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _get_config_float(name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _default_head_pose() -> dict[int, int]:
    return {
        _channel("neck"): config.SERVO_CHANNELS["neck"]["neutral"],
        _channel("headlift"): config.SERVO_CHANNELS["headlift"]["neutral"],
        _channel("headtilt"): config.SERVO_CHANNELS["headtilt"]["neutral"],
    }


_face_tracking_baseline.update(_default_head_pose())


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

def _open_serial_with_retries(
    *,
    log_errors: bool = True,
    attempts: int | None = None,
    delay: float | None = None,
) -> "serial.Serial | None":
    attempts = max(1, attempts if attempts is not None else _get_config_int("SERVO_CONNECT_RETRY_ATTEMPTS", 3))
    delay = max(0.0, delay if delay is not None else _get_config_float("SERVO_CONNECT_RETRY_DELAY_SECS", 1.0))
    timeout = max(0.01, _get_config_float("SERVO_SERIAL_TIMEOUT_SECS", 0.1))

    for attempt in range(1, attempts + 1):
        try:
            ser = serial.Serial(MAESTRO_PORT, config.SERVO_BAUD, timeout=timeout)
            _log.info(
                "Maestro connected on %s at %d baud (attempt %d/%d)",
                MAESTRO_PORT, config.SERVO_BAUD, attempt, attempts,
            )
            startup_delay = max(0.0, _get_config_float("SERVO_CONNECT_STARTUP_DELAY_SECS", 0.2))
            if startup_delay:
                time.sleep(startup_delay)
            return ser
        except _SERIAL_ERRORS as exc:
            level = logging.ERROR if attempt == attempts else logging.WARNING
            if log_errors:
                _log.log(
                    level,
                    "Failed to open Maestro port %s (attempt %d/%d): %s",
                    MAESTRO_PORT, attempt, attempts, exc,
                )
            if attempt < attempts and delay:
                time.sleep(delay)
    return None


def _close_serial_locked() -> None:
    global _ser
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
    _ser = None


def _send_command_locked(raw: bytes) -> bool:
    """Write a Maestro command, attempting one throttled reconnect on failure."""
    global _ser, _last_reconnect_attempt_at

    if _ser is None or not getattr(_ser, "is_open", False):
        now = time.monotonic()
        cooldown = max(0.0, _get_config_float("SERVO_RECONNECT_COOLDOWN_SECS", 5.0))
        if now - _last_reconnect_attempt_at < cooldown:
            return False
        _last_reconnect_attempt_at = now
        _ser = _open_serial_with_retries(
            log_errors=False,
            attempts=_get_config_int("SERVO_RUNTIME_RECONNECT_ATTEMPTS", 1),
            delay=_get_config_float("SERVO_RUNTIME_RECONNECT_DELAY_SECS", 0.0),
        )
        if _ser is None:
            return False

    try:
        _ser.write(raw)
        return True
    except _SERIAL_ERRORS as exc:
        _log.warning("Maestro write failed — attempting reconnect: %s", exc)
        _close_serial_locked()
        _last_reconnect_attempt_at = time.monotonic()
        _ser = _open_serial_with_retries(
            log_errors=False,
            attempts=_get_config_int("SERVO_RUNTIME_RECONNECT_ATTEMPTS", 1),
            delay=_get_config_float("SERVO_RUNTIME_RECONNECT_DELAY_SECS", 0.0),
        )
        if _ser is None:
            return False
        try:
            _ser.write(raw)
            return True
        except _SERIAL_ERRORS as retry_exc:
            _log.warning("Maestro write failed after reconnect: %s", retry_exc)
            _close_serial_locked()
            return False


def _apply_startup_motion_profile_locked() -> None:
    default_speed = _get_config_int("SERVO_DEFAULT_SPEED", 40)
    default_accel = _get_config_int("SERVO_DEFAULT_ACCELERATION", 8)
    for channel in _ALL_CHANNELS:
        cfg = _channel_cfg(channel) or {}
        acceleration = int(cfg.get("acceleration", default_accel))
        _send_command_locked(_encode(_CMD_SET_ACCEL, channel, max(0, acceleration)))
    for channel in _ALL_CHANNELS:
        _send_command_locked(_encode(_CMD_SET_SPEED, channel, max(0, default_speed)))


def connect() -> bool:
    global _ser, _last_reconnect_attempt_at
    if not SERVOS_ENABLED:
        _log.debug("SERVOS_ENABLED=False — skipping connect")
        return False
    with _lock:
        _ser = _open_serial_with_retries()
        if _ser is None:
            _last_reconnect_attempt_at = time.monotonic()
            return False
        if bool(getattr(config, "SERVO_APPLY_STARTUP_MOTION_PROFILE", True)):
            _apply_startup_motion_profile_locked()
        return True


def disconnect() -> None:
    global _ser
    _stop_breathing.set()
    with _lock:
        _close_serial_locked()


# ── Core command primitives ────────────────────────────────────────────────────

def _send_set_target(channel: int, position: int) -> None:
    """Send Maestro compact protocol Set Target command (0x84)."""
    _send_command_locked(_encode(_CMD_SET_TARGET, channel, position))


def _send_set_speed(channel: int, speed: int) -> None:
    """Send Maestro compact protocol Set Speed command (0x87)."""
    _send_command_locked(_encode(_CMD_SET_SPEED, channel, max(0, int(speed))))


def _send_set_acceleration(channel: int, acceleration: int) -> None:
    """Send Maestro compact protocol Set Acceleration command (0x89)."""
    _send_command_locked(_encode(_CMD_SET_ACCEL, channel, max(0, int(acceleration))))


def _remember_positions(channel_dict: "dict[int, int]") -> None:
    for channel, position in channel_dict.items():
        if channel in _CHANNEL_TO_NAME:
            _commanded_positions[channel] = _clamp(channel, int(position))


def set_servo(channel: int, position: int) -> None:
    """Move channel to position (quarter-microseconds), clamped to channel limits."""
    if not SERVOS_ENABLED:
        _log.debug("set_servo no-op: SERVOS_ENABLED=False (ch=%d pos=%d)", channel, position)
        return
    position = _clamp(channel, position)
    with _lock:
        _send_set_target(channel, position)
        _remember_positions({channel: position})
    _record_servo_positions({channel: position})


def set_speed(channel: int, speed: int) -> None:
    """Set the Maestro move speed for one channel."""
    if not SERVOS_ENABLED:
        _log.debug("set_speed no-op: SERVOS_ENABLED=False (ch=%d speed=%d)", channel, speed)
        return
    with _lock:
        _send_set_speed(channel, speed)


def set_acceleration(channel: int, acceleration: int) -> None:
    """Set the Maestro acceleration for one channel."""
    if not SERVOS_ENABLED:
        _log.debug(
            "set_acceleration no-op: SERVOS_ENABLED=False (ch=%d accel=%d)",
            channel, acceleration,
        )
        return
    with _lock:
        _send_set_acceleration(channel, acceleration)


def set_motion_profile(
    channels: "list[int] | tuple[int, ...] | None" = None,
    *,
    speed: int | None = None,
    acceleration: int | None = None,
) -> None:
    """Set speed and/or acceleration for multiple channels."""
    if not SERVOS_ENABLED:
        _log.debug("set_motion_profile no-op: SERVOS_ENABLED=False")
        return
    selected = list(channels or _ALL_CHANNELS)
    with _lock:
        for channel in selected:
            if acceleration is not None:
                _send_set_acceleration(channel, acceleration)
            if speed is not None:
                _send_set_speed(channel, speed)


def get_servo(channel: int) -> "int | None":
    """
    Read actual servo position from Maestro (proprioception).
    Returns position in quarter-microseconds, or None on failure.
    """
    if not SERVOS_ENABLED:
        _log.debug("get_servo no-op: SERVOS_ENABLED=False (ch=%d)", channel)
        return None
    with _lock:
        if not _send_command_locked(bytes([_CMD_GET_POSITION, channel])):
            return None
        if _ser is None or not getattr(_ser, "is_open", False):
            return None
        try:
            data = _ser.read(2)
        except _SERIAL_ERRORS as exc:
            _log.warning("get_servo: read failed for ch=%d: %s", channel, exc)
            _close_serial_locked()
            return None
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
        _remember_positions(channel_dict)
    _record_servo_positions(channel_dict)


def pause_arm_idle() -> None:
    """Prevent idle arm wander from fighting a speech or scripted arm gesture."""
    _arm_idle_pause.set()


def resume_arm_idle() -> None:
    """Allow idle arm wander to use the arm channels again."""
    _arm_idle_pause.clear()


def arm_idle_paused() -> bool:
    return _arm_idle_pause.is_set()


def speech_motion_active() -> bool:
    return _speech_active.is_set()


def _baseline_position(channel: int) -> int:
    cfg = _channel_cfg(channel)
    if cfg is None:
        return _commanded_positions.get(channel, 6000)
    return _clamp(
        channel,
        _commanded_positions.get(
            channel,
            _face_tracking_baseline.get(channel, cfg["neutral"]),
        ),
    )


def set_face_tracking_baseline(
    *,
    neck: int | None = None,
    lift: int | None = None,
    tilt: int | None = None,
) -> None:
    """
    Store the last face-tracking head pose so speech gestures wobble around it
    instead of recentering away from the person Rex is addressing.
    """
    updates: dict[int, int] = {}
    mapping = {
        _channel("neck"): neck,
        _channel("headlift"): lift,
        _channel("headtilt"): tilt,
    }
    for channel, value in mapping.items():
        if value is not None:
            updates[channel] = _clamp(channel, int(value))
    if not updates:
        return
    with _lock:
        _face_tracking_baseline.update(updates)
        _commanded_positions.update(updates)
        if _speech_active.is_set():
            _speech_baseline.update(updates)


def begin_speech_motion(emotion: str = "neutral") -> None:
    """Capture the current gaze/pose and prepare speech-reactive servo motion."""
    global _last_speech_move_at, _speech_hand_counter
    global _speech_elbow_target, _speech_elbow_direction, _next_speech_elbow_at

    pause_arm_idle()
    with _lock:
        _speech_baseline.clear()
        for channel in (_channel("neck"), _channel("headlift"), _channel("headtilt")):
            _speech_baseline[channel] = _baseline_position(channel)
        _last_speech_move_at = 0.0
        _speech_hand_counter = 0
        _speech_elbow_target = None
        _speech_elbow_direction = 1
        _next_speech_elbow_at = 0.0
    _speech_active.set()

    if SERVOS_ENABLED:
        set_motion_profile(
            config.HEAD_CHANNELS,
            speed=_get_config_int("SERVO_SPEECH_HEAD_SPEED", 45),
            acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
        )
        set_motion_profile(
            config.ARM_CHANNELS,
            speed=_get_config_int("SERVO_SPEECH_ARM_SPEED", 35),
            acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
        )


def end_speech_motion() -> None:
    """Return speech-owned channels toward their baseline and release arms."""
    _speech_active.clear()
    try:
        if SERVOS_ENABLED:
            set_motion_profile(
                config.HEAD_CHANNELS + config.ARM_CHANNELS,
                speed=_get_config_int("SERVO_DEFAULT_SPEED", 40),
                acceleration=_get_config_int("SERVO_DEFAULT_ACCELERATION", 8),
            )
            baseline = dict(_speech_baseline) if _speech_baseline else _default_head_pose()
            baseline[_channel("visor")] = config.SERVO_CHANNELS["visor"]["neutral"]
            baseline[_channel("elbow")] = config.SERVO_CHANNELS["elbow"]["neutral"]
            baseline[_channel("hand")] = config.SERVO_CHANNELS["hand"]["neutral"]
            baseline[_channel("heroarm")] = config.SERVO_CHANNELS["heroarm"]["neutral"]
            set_servos(baseline)
    finally:
        resume_arm_idle()


def speech_reactive_move(intensity: float) -> None:
    """
    Move head, visor, and expressive arm channels from a 0-1 speech intensity.

    This is intentionally throttled below the mouth-LED update rate so the
    Maestro receives natural-looking emphasis beats instead of servo chatter.
    """
    global _last_speech_move_at, _speech_hand_counter
    global _speech_elbow_target, _speech_elbow_direction, _next_speech_elbow_at

    if not SERVOS_ENABLED or not _speech_active.is_set():
        return

    now = time.monotonic()
    interval = max(0.04, _get_config_float("SERVO_SPEECH_UPDATE_INTERVAL_SECS", 0.12))
    if now - _last_speech_move_at < interval:
        return
    _last_speech_move_at = now

    intensity = max(0.0, min(1.0, float(intensity)))
    arm_intensity = min(1.0, intensity * _get_config_float("SERVO_SPEECH_ARM_INTENSITY_MULT", 1.8))

    neck_ch = _channel("neck")
    lift_ch = _channel("headlift")
    tilt_ch = _channel("headtilt")
    visor_ch = _channel("visor")
    elbow_ch = _channel("elbow")
    hand_ch = _channel("hand")
    hero_ch = _channel("heroarm")

    with _lock:
        base_neck = _clamp(neck_ch, _speech_baseline.get(neck_ch, _baseline_position(neck_ch)))
        base_lift = _clamp(lift_ch, _speech_baseline.get(lift_ch, _baseline_position(lift_ch)))
        base_tilt = _clamp(tilt_ch, _speech_baseline.get(tilt_ch, _baseline_position(tilt_ch)))

    neck_wobble = int(_get_config_int("SERVO_SPEECH_NECK_WOBBLE_QUS", 260) * (0.35 + intensity))
    lift_wobble = int(_get_config_int("SERVO_SPEECH_LIFT_WOBBLE_QUS", 160) * (0.35 + intensity))
    tilt_wobble = int(_get_config_int("SERVO_SPEECH_TILT_WOBBLE_QUS", 120) * (0.35 + intensity))

    targets: dict[int, int] = {
        neck_ch: _clamp(neck_ch, base_neck + random.randint(-neck_wobble, neck_wobble)),
        lift_ch: _clamp(lift_ch, base_lift + random.randint(-lift_wobble, lift_wobble)),
        tilt_ch: _clamp(tilt_ch, base_tilt + random.randint(-tilt_wobble, tilt_wobble)),
    }

    visor_cfg = config.SERVO_CHANNELS["visor"]
    visor_open_floor = max(visor_cfg["neutral"], int(visor_cfg["min"] + (visor_cfg["max"] - visor_cfg["min"]) * 0.55))
    visor_wave = 0.5 + 0.5 * math.sin(now * 8.0)
    visor_swing = int((visor_cfg["max"] - visor_open_floor) * (0.35 + 0.40 * intensity))
    targets[visor_ch] = _clamp(
        visor_ch,
        int(visor_open_floor + visor_wave * visor_swing) + random.randint(-45, 45),
    )

    elbow_lo, elbow_hi = config.SERVO_CHANNELS["elbow"]["min"], config.SERVO_CHANNELS["elbow"]["max"]
    if _speech_elbow_target is None or now >= _next_speech_elbow_at:
        span = elbow_hi - elbow_lo
        center = int(elbow_lo + span * 0.55)
        amplitude = int(span * (0.10 + 0.12 * arm_intensity))
        _speech_elbow_target = _clamp(
            elbow_ch,
            center + _speech_elbow_direction * amplitude + random.randint(-25, 25),
        )
        _speech_elbow_direction *= -1
        _next_speech_elbow_at = now + random.uniform(
            _get_config_float("SERVO_SPEECH_ELBOW_INTERVAL_MIN_SECS", 0.35),
            _get_config_float("SERVO_SPEECH_ELBOW_INTERVAL_MAX_SECS", 0.75),
        )
    targets[elbow_ch] = _speech_elbow_target

    _speech_hand_counter += 1
    hand_divisor = max(1, _get_config_int("SERVO_SPEECH_HAND_DIVISOR", 3))
    if _speech_hand_counter % hand_divisor == 0:
        hand_cfg = config.SERVO_CHANNELS["hand"]
        center = hand_cfg["neutral"]
        amplitude = int((hand_cfg["max"] - hand_cfg["min"]) * (0.08 + 0.12 * arm_intensity))
        direction = -1 if (_speech_hand_counter // hand_divisor) % 2 == 0 else 1
        targets[hand_ch] = _clamp(hand_ch, center + direction * amplitude)

    hero_cfg = config.SERVO_CHANNELS["heroarm"]
    hero_swing = int((hero_cfg["max"] - hero_cfg["min"]) * (0.10 + 0.18 * arm_intensity))
    targets[hero_ch] = _clamp(
        hero_ch,
        hero_cfg["neutral"] + random.randint(-hero_swing, hero_swing),
    )

    set_servos(targets)


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
                _remember_positions(moves)
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
                _remember_positions(moves)
            time.sleep(step_delay)
    _record_servo_positions(targets)


def stop_breathing() -> None:
    """Signal the breathing thread to stop. Returns immediately; thread exits within ~50 ms."""
    _stop_breathing.set()


def shutdown() -> None:
    """Stop breathing thread and cleanly disconnect. Call before process exit."""
    _stop_breathing.set()
    disconnect()
