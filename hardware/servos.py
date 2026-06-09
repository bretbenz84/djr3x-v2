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
_manual_override = threading.Event()

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
# Pokerarm sways back and forth while speaking on a slower cadence than the hero arm.
_speech_poker_target: int | None = None
_speech_poker_direction = -1
_next_speech_poker_at = 0.0
_speech_emotion_frame: dict = {}

# Listening-motion state: gentle "I'm processing you" body language that runs
# from speech onset through transcription/LLM/TTS so Rex isn't frozen while he
# thinks. Subtler than speech motion. Breathing + face tracking yield to it.
_listening_active = threading.Event()
_listening_thread: "threading.Thread | None" = None
_listening_lock = threading.Lock()

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


def _motion_float(motion: dict, key: str, default: float) -> float:
    try:
        return float((motion or {}).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _resolve_speech_emotion_frame(emotion) -> dict:
    if isinstance(emotion, dict):
        return dict(emotion)
    try:
        as_dict = getattr(emotion, "as_dict", None)
        if callable(as_dict):
            return dict(as_dict())
    except Exception:
        pass
    try:
        from intelligence import emotion_orchestrator
        return emotion_orchestrator.frame_for_speech(str(emotion or "neutral")).as_dict()
    except Exception:
        return {
            "affect": str(emotion or "neutral"),
            "led_style": str(emotion or "neutral"),
            "motion_style": "neutral",
            "speech_motion": {},
        }


def _scaled_profile_value(base: int, mult: float) -> int:
    try:
        value = int(round(float(base) * float(mult)))
    except (TypeError, ValueError):
        value = base
    return max(1, min(255, value))


def _gui_servo_sim_enabled() -> bool:
    return bool(
        getattr(config, "GUI_ENABLED", False)
        and getattr(config, "GUI_SERVO_SIM_ENABLED", True)
    )


def _automatic_motion_allowed() -> bool:
    try:
        import state as state_module
        from state import State
        return state_module.get_state() not in (State.SLEEP, State.SHUTDOWN)
    except Exception:
        return True


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
    try:
        from gui.state_bridge import gui_bridge
        for name, value in updates.items():
            gui_bridge.update_servo_position(name, value)
    except Exception:
        pass


def _record_manual_override_state(enabled: bool) -> None:
    try:
        from world_state import world_state

        self_state = world_state.get("self_state")
        self_state["manual_servo_override"] = bool(enabled)
        world_state.update("self_state", self_state)
    except Exception as exc:
        _log.debug("manual servo override world_state update failed: %s", exc)
    try:
        from gui.state_bridge import gui_bridge

        gui_bridge.update_servo_override(bool(enabled))
    except Exception:
        pass


def _program_servo_updates_blocked() -> bool:
    return _manual_override.is_set()


def manual_override_enabled() -> bool:
    """Return True when GUI manual servo override owns servo targets."""
    return _manual_override.is_set()


def set_manual_override_enabled(enabled: bool) -> None:
    """Freeze programmatic servo target updates so GUI sliders can drive servos."""
    enabled = bool(enabled)
    was_enabled = _manual_override.is_set()
    if enabled:
        _manual_override.set()
    else:
        _manual_override.clear()
    if was_enabled != enabled:
        _record_manual_override_state(enabled)
        _log.info("Manual servo override %s", "enabled" if enabled else "disabled")


def set_manual_servo(channel: int, position: int) -> bool:
    """
    Direct GUI-owned servo control.

    Programmatic setters no-op while manual override is enabled, but this path
    intentionally bypasses that gate and mirrors the resulting pose to the GUI
    avatar / world state immediately.
    """
    if not _manual_override.is_set():
        return False
    channel = int(channel)
    position = _clamp(channel, int(position))
    with _lock:
        if SERVOS_ENABLED:
            _send_set_target(channel, position)
        _remember_positions({channel: position})
        if channel in {
            _channel("neck"),
            _channel("headlift"),
            _channel("headtilt"),
        }:
            _face_tracking_baseline[channel] = position
    _record_servo_positions({channel: position})
    return True


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
    _send_command_locked(_encode(_CMD_SET_TARGET, channel, _clamp(channel, int(position))))


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
    if _program_servo_updates_blocked():
        return
    position = _clamp(channel, position)
    if not SERVOS_ENABLED:
        _log.debug("set_servo no-op: SERVOS_ENABLED=False (ch=%d pos=%d)", channel, position)
        if _gui_servo_sim_enabled():
            with _lock:
                _remember_positions({channel: position})
            _record_servo_positions({channel: position})
        return
    with _lock:
        _send_set_target(channel, position)
        _remember_positions({channel: position})
    _record_servo_positions({channel: position})


def set_speed(channel: int, speed: int) -> None:
    """Set the Maestro move speed for one channel."""
    if _program_servo_updates_blocked():
        return
    if not SERVOS_ENABLED:
        _log.debug("set_speed no-op: SERVOS_ENABLED=False (ch=%d speed=%d)", channel, speed)
        return
    with _lock:
        _send_set_speed(channel, speed)


def set_acceleration(channel: int, acceleration: int) -> None:
    """Set the Maestro acceleration for one channel."""
    if _program_servo_updates_blocked():
        return
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
    if _program_servo_updates_blocked():
        return
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
    if _program_servo_updates_blocked():
        return
    channel_dict = {ch: _clamp(ch, int(pos)) for ch, pos in channel_dict.items()}
    if not SERVOS_ENABLED:
        _log.debug("set_servos no-op: SERVOS_ENABLED=False")
        if _gui_servo_sim_enabled():
            with _lock:
                _remember_positions(channel_dict)
            _record_servo_positions(channel_dict)
        return
    with _lock:
        for channel, position in channel_dict.items():
            _send_set_target(channel, position)
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
    if _program_servo_updates_blocked():
        return
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


def get_face_tracking_baseline() -> dict[int, int]:
    """Return the head pose that speech/breathing should orbit around."""
    with _lock:
        return dict(_face_tracking_baseline)


def reset_face_tracking_baseline() -> None:
    """Reset gaze baseline to the configured neutral head pose."""
    with _lock:
        _face_tracking_baseline.update(_default_head_pose())


def begin_speech_motion(emotion: str = "neutral") -> None:
    """Capture the current gaze/pose and prepare speech-reactive servo motion."""
    global _last_speech_move_at, _speech_hand_counter
    global _speech_elbow_target, _speech_elbow_direction, _next_speech_elbow_at
    global _speech_poker_target, _speech_poker_direction, _next_speech_poker_at
    global _speech_emotion_frame

    if _program_servo_updates_blocked():
        return

    # Rex is about to talk — hand the body off from listening motion to speech.
    stop_listening_motion()

    frame = _resolve_speech_emotion_frame(emotion)
    motion = frame.get("speech_motion") or {}

    pause_arm_idle()
    with _lock:
        _speech_emotion_frame = frame
        _speech_baseline.clear()
        for channel in (_channel("neck"), _channel("headlift"), _channel("headtilt")):
            baseline = _baseline_position(channel)
            if channel == _channel("headlift"):
                baseline += int(_motion_float(motion, "lift_bias_qus", 0.0))
            elif channel == _channel("headtilt"):
                baseline += int(_motion_float(motion, "tilt_bias_qus", 0.0))
            _speech_baseline[channel] = _clamp(channel, baseline)
        _last_speech_move_at = 0.0
        _speech_hand_counter = 0
        _speech_elbow_target = None
        _speech_elbow_direction = 1
        _next_speech_elbow_at = 0.0
        _speech_poker_target = None
        _speech_poker_direction = -1
        _next_speech_poker_at = 0.0
    _speech_active.set()
    set_breathing_emotion(str(frame.get("led_style") or frame.get("affect") or "neutral"))

    if SERVOS_ENABLED:
        head_speed = _scaled_profile_value(
            _get_config_int("SERVO_SPEECH_HEAD_SPEED", 45),
            _motion_float(motion, "head_speed_mult", 1.0),
        )
        arm_speed = _scaled_profile_value(
            _get_config_int("SERVO_SPEECH_ARM_SPEED", 35),
            _motion_float(motion, "arm_speed_mult", 1.0),
        )
        set_motion_profile(
            config.HEAD_CHANNELS,
            speed=head_speed,
            acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
        )
        set_motion_profile(
            config.ARM_CHANNELS,
            speed=arm_speed,
            acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
        )


def end_speech_motion() -> None:
    """Return speech-owned channels toward their baseline and release arms."""
    global _speech_emotion_frame
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
            baseline[_channel("pokerarm")] = config.SERVO_CHANNELS["pokerarm"]["neutral"]
            baseline[_channel("heroarm")] = config.SERVO_CHANNELS["heroarm"]["neutral"]
            set_servos(baseline)
        elif _gui_servo_sim_enabled():
            baseline = dict(_speech_baseline) if _speech_baseline else _default_head_pose()
            baseline[_channel("visor")] = config.SERVO_CHANNELS["visor"]["neutral"]
            baseline[_channel("elbow")] = config.SERVO_CHANNELS["elbow"]["neutral"]
            baseline[_channel("hand")] = config.SERVO_CHANNELS["hand"]["neutral"]
            baseline[_channel("pokerarm")] = config.SERVO_CHANNELS["pokerarm"]["neutral"]
            baseline[_channel("heroarm")] = config.SERVO_CHANNELS["heroarm"]["neutral"]
            set_servos(baseline)
    finally:
        _speech_emotion_frame = {}
        set_breathing_emotion("neutral")
        resume_arm_idle()


def speech_reactive_move(intensity: float) -> None:
    """
    Move head, visor, and expressive arm channels from a 0-1 speech intensity.

    This is intentionally throttled below the mouth-LED update rate so the
    Maestro receives natural-looking emphasis beats instead of servo chatter.
    """
    global _last_speech_move_at, _speech_hand_counter
    global _speech_elbow_target, _speech_elbow_direction, _next_speech_elbow_at
    global _speech_poker_target, _speech_poker_direction, _next_speech_poker_at

    if _program_servo_updates_blocked():
        return
    if not _speech_active.is_set():
        return
    if not SERVOS_ENABLED and not _gui_servo_sim_enabled():
        return

    now = time.monotonic()
    with _lock:
        frame = dict(_speech_emotion_frame)
    motion = frame.get("speech_motion") if isinstance(frame.get("speech_motion"), dict) else {}
    interval = max(
        0.035,
        _get_config_float("SERVO_SPEECH_UPDATE_INTERVAL_SECS", 0.12)
        * _motion_float(motion, "interval_scale", 1.0),
    )
    if now - _last_speech_move_at < interval:
        return
    _last_speech_move_at = now

    intensity = max(0.0, min(1.0, float(intensity)))
    frame_intensity = _motion_float(frame, "intensity", 0.35)
    expression_gain = 0.70 + 0.45 * max(0.0, min(1.0, frame_intensity))
    expressive_intensity = min(1.0, intensity * expression_gain)
    arm_intensity = min(
        1.0,
        intensity
        * _get_config_float("SERVO_SPEECH_ARM_INTENSITY_MULT", 1.8)
        * _motion_float(motion, "arm_intensity_mult", 1.0),
    )

    neck_ch = _channel("neck")
    lift_ch = _channel("headlift")
    tilt_ch = _channel("headtilt")
    visor_ch = _channel("visor")
    elbow_ch = _channel("elbow")
    hand_ch = _channel("hand")
    poker_ch = _channel("pokerarm")
    hero_ch = _channel("heroarm")

    with _lock:
        base_neck = _clamp(neck_ch, _speech_baseline.get(neck_ch, _baseline_position(neck_ch)))
        base_lift = _clamp(lift_ch, _speech_baseline.get(lift_ch, _baseline_position(lift_ch)))
        base_tilt = _clamp(tilt_ch, _speech_baseline.get(tilt_ch, _baseline_position(tilt_ch)))

    neck_wobble = int(
        _get_config_int("SERVO_SPEECH_NECK_WOBBLE_QUS", 260)
        * _motion_float(motion, "head_wobble_mult", 1.0)
        * (0.35 + expressive_intensity)
    )
    lift_wobble = int(
        _get_config_int("SERVO_SPEECH_LIFT_WOBBLE_QUS", 160)
        * _motion_float(motion, "lift_wobble_mult", 1.0)
        * (0.35 + expressive_intensity)
    )
    tilt_wobble = int(
        _get_config_int("SERVO_SPEECH_TILT_WOBBLE_QUS", 120)
        * _motion_float(motion, "tilt_wobble_mult", 1.0)
        * (0.35 + expressive_intensity)
    )

    targets: dict[int, int] = {
        neck_ch: _clamp(neck_ch, base_neck + random.randint(-neck_wobble, neck_wobble)),
        lift_ch: _clamp(lift_ch, base_lift + random.randint(-lift_wobble, lift_wobble)),
        tilt_ch: _clamp(tilt_ch, base_tilt + random.randint(-tilt_wobble, tilt_wobble)),
    }

    visor_cfg = config.SERVO_CHANNELS["visor"]
    visor_floor_frac = max(0.0, min(1.0, _motion_float(motion, "visor_open_floor_frac", 0.55)))
    visor_open_floor = max(
        visor_cfg["neutral"],
        int(visor_cfg["min"] + (visor_cfg["max"] - visor_cfg["min"]) * visor_floor_frac),
    )
    visor_wave = 0.5 + 0.5 * math.sin(now * 8.0)
    visor_swing = int(
        (visor_cfg["max"] - visor_open_floor)
        * (0.35 + 0.40 * expressive_intensity)
        * _motion_float(motion, "visor_swing_mult", 1.0)
    )
    targets[visor_ch] = _clamp(
        visor_ch,
        int(visor_open_floor + visor_wave * visor_swing) + random.randint(-45, 45),
    )

    elbow_lo, elbow_hi = config.SERVO_CHANNELS["elbow"]["min"], config.SERVO_CHANNELS["elbow"]["max"]
    if _speech_elbow_target is None or now >= _next_speech_elbow_at:
        span = elbow_hi - elbow_lo
        center = int(elbow_lo + span * 0.55)
        amplitude = int(
            span
            * (0.10 + 0.12 * arm_intensity)
            * _motion_float(motion, "elbow_amp_mult", 1.0)
        )
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

    # Pokerarm: a slow, deliberate back-and-forth sway — a slower cadence than the
    # per-frame hero arm, and far livelier (more frequent + wider) than the idle arm
    # wander, so it reads as "talking with the arm" without the hero arm's pace.
    poker_cfg = config.SERVO_CHANNELS["pokerarm"]
    if _speech_poker_target is None or now >= _next_speech_poker_at:
        span = poker_cfg["max"] - poker_cfg["min"]
        center = poker_cfg["neutral"]
        amplitude = int(
            span
            * (0.14 + 0.16 * arm_intensity)
            * _motion_float(motion, "poker_amp_mult", 1.0)
        )
        _speech_poker_target = _clamp(
            poker_ch,
            center + _speech_poker_direction * amplitude + random.randint(-30, 30),
        )
        _speech_poker_direction *= -1
        _next_speech_poker_at = now + random.uniform(
            _get_config_float("SERVO_SPEECH_POKER_INTERVAL_MIN_SECS", 0.9),
            _get_config_float("SERVO_SPEECH_POKER_INTERVAL_MAX_SECS", 1.7),
        )
    targets[poker_ch] = _speech_poker_target

    _speech_hand_counter += 1
    hand_divisor = max(1, _get_config_int("SERVO_SPEECH_HAND_DIVISOR", 3))
    if _speech_hand_counter % hand_divisor == 0:
        hand_cfg = config.SERVO_CHANNELS["hand"]
        center = hand_cfg["neutral"]
        amplitude = int(
            (hand_cfg["max"] - hand_cfg["min"])
            * (0.08 + 0.12 * arm_intensity)
            * _motion_float(motion, "hand_amp_mult", 1.0)
        )
        direction = -1 if (_speech_hand_counter // hand_divisor) % 2 == 0 else 1
        targets[hand_ch] = _clamp(hand_ch, center + direction * amplitude)

    hero_cfg = config.SERVO_CHANNELS["heroarm"]
    hero_swing = int(
        (hero_cfg["max"] - hero_cfg["min"])
        * (0.10 + 0.18 * arm_intensity)
        * _motion_float(motion, "hero_swing_mult", 1.0)
    )
    targets[hero_ch] = _clamp(
        hero_ch,
        hero_cfg["neutral"] + random.randint(-hero_swing, hero_swing),
    )

    set_servos(targets)


# ── Listening motion (subtle "I'm thinking about what you said" feedback) ───────
# Runs from the moment VAD/Whisper hears the user through transcription → LLM →
# TTS, so Rex shows he's listening instead of freezing. Deliberately gentler and
# slower than speech motion: small head nods orbiting the current gaze, a slow
# visor flutter, and occasional small arm/hand shifts. Breathing and face
# tracking yield while it's active (see breathing_thread and
# consciousness._step_face_tracking), and it yields the instant Rex speaks.

def listening_motion_active() -> bool:
    return _listening_active.is_set()


def _listening_targets(beat: int) -> dict[int, int]:
    """One beat of gentle listening pose targets, orbiting the current gaze.

    Pure-ish helper (reads config + the face-tracking baseline) so it can be unit
    tested without hardware. Head nods/visor every beat; arms on a slower cadence.
    """
    neck_ch = _channel("neck")
    lift_ch = _channel("headlift")
    tilt_ch = _channel("headtilt")
    visor_ch = _channel("visor")
    elbow_ch = _channel("elbow")
    hand_ch = _channel("hand")
    hero_ch = _channel("heroarm")

    with _lock:
        base = dict(_face_tracking_baseline)
    base_neck = _clamp(neck_ch, base.get(neck_ch, _baseline_position(neck_ch)))
    base_lift = _clamp(lift_ch, base.get(lift_ch, _baseline_position(lift_ch)))
    base_tilt = _clamp(tilt_ch, base.get(tilt_ch, _baseline_position(tilt_ch)))

    targets: dict[int, int] = {}

    # Head: gentle downward "mhm" nod every few beats, easing back to the tracked
    # gaze in between. headlift lower = head down; headtilt is inverted (higher =
    # looking down), so a nod biases lift DOWN and tilt slightly DOWN.
    nod_every = max(1, _get_config_int("SERVO_LISTENING_NOD_EVERY_BEATS", 2))
    if beat % nod_every == 0:
        lift_amp = _get_config_int("SERVO_LISTENING_LIFT_NOD_QUS", 240)
        tilt_amp = _get_config_int("SERVO_LISTENING_TILT_QUS", 80)
        neck_amp = _get_config_int("SERVO_LISTENING_NECK_QUS", 110)
        targets[lift_ch] = _clamp(lift_ch, base_lift - random.randint(int(lift_amp * 0.3), lift_amp))
        targets[tilt_ch] = _clamp(tilt_ch, base_tilt + random.randint(0, tilt_amp))
        targets[neck_ch] = _clamp(neck_ch, base_neck + random.randint(-neck_amp, neck_amp))
    else:
        targets[lift_ch] = base_lift
        targets[tilt_ch] = base_tilt
        targets[neck_ch] = base_neck

    # Visor: slow, shallow flutter around a slightly-open resting position.
    visor_cfg = config.SERVO_CHANNELS["visor"]
    visor_swing = _get_config_int("SERVO_LISTENING_VISOR_QUS", 220)
    wave = 0.5 + 0.5 * math.sin(beat * 0.9)
    targets[visor_ch] = _clamp(
        visor_ch,
        int(visor_cfg["neutral"] - visor_swing * 0.4 + wave * visor_swing) + random.randint(-30, 30),
    )

    # Arms: small, occasional shifts so the hand/arm look alive, not twitchy.
    arm_every = max(1, _get_config_int("SERVO_LISTENING_ARM_EVERY_BEATS", 2))
    if beat % arm_every == 0:
        elbow_cfg = config.SERVO_CHANNELS["elbow"]
        hand_cfg = config.SERVO_CHANNELS["hand"]
        hero_cfg = config.SERVO_CHANNELS["heroarm"]
        elbow_amp = _get_config_int("SERVO_LISTENING_ELBOW_QUS", 110)
        hand_amp = _get_config_int("SERVO_LISTENING_HAND_QUS", 380)
        hero_amp = _get_config_int("SERVO_LISTENING_HERO_QUS", 300)
        targets[elbow_ch] = _clamp(elbow_ch, elbow_cfg["neutral"] + random.randint(-elbow_amp, elbow_amp))
        targets[hand_ch] = _clamp(hand_ch, hand_cfg["neutral"] + random.randint(-hand_amp, hand_amp))
        targets[hero_ch] = _clamp(hero_ch, hero_cfg["neutral"] + random.randint(-hero_amp, hero_amp))

    return targets


def _listening_loop() -> None:
    """Background loop: emit gentle listening beats until stop_listening_motion()."""
    beat = 0
    started = time.monotonic()
    # Safety net: never let a missed stop strand listening motion (which would
    # also keep face tracking yielded). Auto-stop after a generous ceiling.
    max_secs = _get_config_float("SERVO_LISTENING_MAX_SECS", 20.0)
    while _listening_active.is_set() and not _stop_breathing.is_set():
        if time.monotonic() - started >= max_secs:
            _log.debug("listening motion hit max duration — auto-stopping.")
            stop_listening_motion()
            return
        # Yield to real speech and to blocked/asleep states; re-check shortly.
        if (
            _program_servo_updates_blocked()
            or not _automatic_motion_allowed()
            or _speech_active.is_set()
        ):
            time.sleep(0.1)
            continue
        beat += 1
        try:
            set_servos(_listening_targets(beat))
        except Exception as exc:
            _log.debug("listening beat failed: %s", exc)
        lo = _get_config_float("SERVO_LISTENING_BEAT_MIN_SECS", 0.45)
        hi = _get_config_float("SERVO_LISTENING_BEAT_MAX_SECS", 0.85)
        time.sleep(random.uniform(min(lo, hi), max(lo, hi)))


def start_listening_motion() -> None:
    """Begin gentle listening motion (idempotent). No-op without servos/sim, while
    Rex is already speaking, or if disabled via config."""
    global _listening_thread
    if not bool(getattr(config, "SERVO_LISTENING_MOTION_ENABLED", True)):
        return
    if _program_servo_updates_blocked() or _speech_active.is_set():
        return
    if not SERVOS_ENABLED and not _gui_servo_sim_enabled():
        return
    with _listening_lock:
        if _listening_active.is_set():
            return
        _listening_active.set()
        pause_arm_idle()
        if SERVOS_ENABLED:
            set_motion_profile(
                config.HEAD_CHANNELS + config.ARM_CHANNELS,
                speed=_get_config_int("SERVO_LISTENING_SPEED", 22),
                acceleration=_get_config_int("SERVO_LISTENING_ACCELERATION", 6),
            )
        if _listening_thread is None or not _listening_thread.is_alive():
            _listening_thread = threading.Thread(
                target=_listening_loop, daemon=True, name="listening-motion"
            )
            _listening_thread.start()


def stop_listening_motion() -> None:
    """Stop listening motion: restore the default motion profile, ease the visor
    back to rest, and hand the arms back to idle wander. Breathing and face
    tracking resume on their own once the flag clears. Idempotent."""
    with _listening_lock:
        if not _listening_active.is_set():
            return
        _listening_active.clear()
    try:
        if SERVOS_ENABLED:
            set_motion_profile(
                config.HEAD_CHANNELS + config.ARM_CHANNELS,
                speed=_get_config_int("SERVO_DEFAULT_SPEED", 40),
                acceleration=_get_config_int("SERVO_DEFAULT_ACCELERATION", 8),
            )
        if SERVOS_ENABLED or _gui_servo_sim_enabled():
            # Settle the visor (nothing else owns it); arms return via idle wander.
            set_servo(_channel("visor"), config.SERVO_CHANNELS["visor"]["neutral"])
    finally:
        resume_arm_idle()


# ── High-level behaviours ──────────────────────────────────────────────────────

def neutral(step_us: int = 40, step_delay: float = 0.02) -> None:
    """
    Move all channels smoothly to their neutral positions.
    Reads current positions first, then interpolates to neutral.
    """
    if _program_servo_updates_blocked():
        return
    if not SERVOS_ENABLED:
        _log.debug("neutral() no-op: SERVOS_ENABLED=False")
        if _gui_servo_sim_enabled():
            targets = {
                cfg["ch"]: _clamp(cfg["ch"], cfg["neutral"])
                for cfg in config.SERVO_CHANNELS.values()
            }
            with _lock:
                _remember_positions(targets)
                _face_tracking_baseline.update(_default_head_pose())
            _record_servo_positions(targets)
        return

    targets = {
        name: _clamp(cfg["ch"], cfg["neutral"])
        for name, cfg in config.SERVO_CHANNELS.items()
    }

    # Read current positions
    current: dict[str, int] = {}
    for name, cfg in config.SERVO_CHANNELS.items():
        pos = get_servo(cfg["ch"])
        current[name] = _clamp(cfg["ch"], pos if pos is not None else targets[name])

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
                    _send_set_target(ch, _clamp(ch, pos))
                _remember_positions(moves)
            time.sleep(step_delay)
    _record_servo_positions({
        cfg["ch"]: _clamp(cfg["ch"], cfg["neutral"])
        for cfg in config.SERVO_CHANNELS.values()
    })
    reset_face_tracking_baseline()


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
    if not SERVOS_ENABLED and not _gui_servo_sim_enabled():
        _log.debug("breathing_thread no-op: SERVOS_ENABLED=False")
        return

    _log.info("Breathing thread started")
    headlift_cfg = config.SERVO_CHANNELS["headlift"]
    channel      = headlift_cfg["ch"]
    amplitude    = config.BREATHING_AMPLITUDE_QUS

    tick = 0.05  # seconds between position updates

    while not _stop_breathing.is_set():
        if _program_servo_updates_blocked() or not _automatic_motion_allowed():
            _stop_breathing.wait(tick)
            continue
        # Listening motion owns the head-lift nod while active — don't fight it.
        if _listening_active.is_set():
            _stop_breathing.wait(tick)
            continue

        with _breathing_lock:
            emotion = _breathing_emotion

        if emotion == "excited":
            period = config.BREATHING_PERIOD_EXCITED
        elif emotion == "sad":
            period = config.BREATHING_PERIOD_SAD
        else:
            period = config.BREATHING_PERIOD_SECS

        with _lock:
            baseline_pos = _face_tracking_baseline.get(channel, headlift_cfg["neutral"])

        t   = time.monotonic()
        pos = int(baseline_pos + amplitude * math.sin(2 * math.pi * t / period))
        pos = _clamp(channel, pos)

        if SERVOS_ENABLED:
            with _lock:
                _send_set_target(channel, pos)
        elif _gui_servo_sim_enabled():
            with _lock:
                _remember_positions({channel: pos})
            _record_servo_positions({channel: pos})

        _stop_breathing.wait(tick)

    _log.info("Breathing thread stopped")


def idle_animation() -> None:
    """
    One cycle of random small movements on neck and headlift channels.
    Intended to be called periodically from the consciousness loop during IDLE.
    """
    if _program_servo_updates_blocked():
        return
    if not SERVOS_ENABLED:
        _log.debug("idle_animation no-op: SERVOS_ENABLED=False")
        if not _gui_servo_sim_enabled():
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


def move_to(
    targets: "dict[int, int]",
    step_us: int = 40,
    step_delay: float = 0.02,
    start: "dict[int, int] | None" = None,
) -> None:
    """Smoothly interpolate specific channels to target positions (quarter-microseconds).

    The interpolation start point is read from the Maestro (proprioception) so the
    sweep begins wherever the servo actually is. Pass ``start`` to override that read
    with a known pose for specific channels — use it when the caller already knows the
    current position and the proprioception read is unreliable (e.g. the first move
    right after a fresh serial connect, where a failed/garbage read would otherwise
    collapse the sweep into a jump).
    """
    if _program_servo_updates_blocked():
        return
    targets = {ch: _clamp(ch, int(tgt)) for ch, tgt in targets.items()}
    if not SERVOS_ENABLED:
        _log.debug("move_to no-op: SERVOS_ENABLED=False")
        if _gui_servo_sim_enabled():
            with _lock:
                _remember_positions(targets)
            _record_servo_positions(targets)
        return

    current: dict[int, int] = {}
    for ch, tgt in targets.items():
        if start is not None and ch in start:
            pos = int(start[ch])
        else:
            pos = get_servo(ch)
        current[ch] = _clamp(ch, pos if pos is not None else tgt)

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
