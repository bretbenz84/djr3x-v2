"""
Pololu Maestro Mini 18 servo controller.

All operations are no-ops (with a debug log) when SERVOS_ENABLED is False.
Channel numbers, limits, and neutral positions come from config.py.

October 10 2026 rework (see hardware/head_pose.py): the head rides on TWO servos
(channels 2 and 11) that must always be commanded together as a pitch and a tilt,
the poker arm has a coupled elbow/wrist pair (12/13) with a measured clearance
ceiling that is moved in stages, and the visor runs inverted. Every public write
in this module refuses a lone head or poker channel, and a head pair that no
pitch/tilt explains, before anything reaches the wire.
"""

import logging
import math
import random
import struct
import threading
import time
from contextlib import contextmanager

import serial

import config
from hardware import head_pose, throttle_motion
from utils.config_loader import MAESTRO_PORT, SERVOS_ENABLED

_log = logging.getLogger(__name__)
_SERIAL_ERRORS = (serial.SerialException, serial.SerialTimeoutException, OSError)

_CMD_SET_TARGET = 0x84
_CMD_SET_SPEED = 0x87
_CMD_SET_ACCEL = 0x89
_CMD_GET_POSITION = 0x90

HEAD_PAIR = head_pose.HEAD_PAIR
POKER_PAIR = head_pose.POKER_PAIR

_ser: "serial.Serial | None" = None
_lock = threading.Lock()
_CHANNEL_TO_NAME = {
    cfg["ch"]: name
    for name, cfg in config.SERVO_CHANNELS.items()
}
_ALL_CHANNELS = sorted(_CHANNEL_TO_NAME)
_commanded_positions: dict[int, int] = {
    # A channel with an unpowered "rest" (the elbow, the head pair, the poker
    # elbow/wrist) starts there, not at neutral: with the robot powered off that
    # is physically where it is.
    cfg["ch"]: cfg.get("rest", cfg["neutral"])
    for cfg in config.SERVO_CHANNELS.values()
}
# The staged poker elbow/wrist move runs on its own thread when a caller cannot
# wait; one at a time, and a newer request supersedes an older one.
_poker_lock = threading.Lock()
_poker_generation = 0
_last_reconnect_attempt_at = 0.0
_manual_override = threading.Event()
_shutdown_latch = threading.Event()   # power-down pose reached — servos frozen
_sleep_latch = threading.Event()      # sleep pose reached — frozen until wake

# The microphone rotates with this arm. Enforce the lease at the wire and
# commanded-position layers, so direct animation writes cannot bypass it.
_voice_hold_until = 0.0
_voice_hold_ready = 0.0
_voice_hold_timer = None
_voice_hold_lock = threading.RLock()
_photo_hold = None  # (expiry, head targets); published atomically, bounded on failure


@contextmanager
def hold_head_for_photo(duration: float = 2.0):
    """Hold actual gaze briefly; suppress competing animation/tracking targets.

    Never center the head or restore an old pose after capture. Normal writers
    resume after the lease. No serial work in dev mode.
    """
    global _photo_hold
    if not SERVOS_ENABLED:
        yield
        return
    if _program_servo_updates_blocked():
        raise RuntimeError("Head unavailable for photo capture")
    if _photo_hold and time.monotonic() < _photo_hold[0]:
        raise RuntimeError("Photo capture already holds the head")
    targets = {}
    for name in ("neck", "headlift", "head_left", "head_right"):
        ch = config.SERVO_CHANNELS[name]["ch"]
        position = get_servo(ch)
        if position is None or position <= 0:
            raise RuntimeError("Cannot confirm head position for photo capture")
        targets[ch] = position
    lease = (time.monotonic() + min(3.0, max(.1, duration)), targets)
    _photo_hold = lease
    try:
        set_servos(targets)
        yield
    finally:
        if _photo_hold is lease:
            _photo_hold = None


def voice_enrollment_hold_active() -> bool:
    return time.monotonic() < _voice_hold_until


def release_voice_enrollment_hold() -> None:
    global _voice_hold_until, _voice_hold_ready, _voice_hold_timer
    with _voice_hold_lock:
        _voice_hold_until = _voice_hold_ready = 0.0
        if _voice_hold_timer:
            _voice_hold_timer.cancel()
            _voice_hold_timer = None


def hold_voice_enrollment_mic(duration: float) -> float:
    """Center the hero arm once; return earliest safe capture time. Bounded lease."""
    global _voice_hold_until, _voice_hold_ready, _voice_hold_timer
    with _voice_hold_lock:
        now = time.monotonic()
        if not SERVOS_ENABLED:
            return now
        if _program_servo_updates_blocked():
            raise RuntimeError('servo control unavailable for voice enrollment')
        if voice_enrollment_hold_active():
            return _voice_hold_ready
        _voice_hold_until = now + max(1., min(60., duration))
        _voice_hold_ready = now + float(getattr(config, 'VOICE_LEARNING_SERVO_SETTLE_SECS', 1.5))
        cfg = config.SERVO_CHANNELS['heroarm']
        set_servo(cfg['ch'], (cfg['min'] + cfg['max']) // 2)
        deadline = _voice_hold_until
        def expire():
            with _voice_hold_lock:
                if _voice_hold_until == deadline:
                    release_voice_enrollment_hold()
        _voice_hold_timer = threading.Timer(max(0., deadline - time.monotonic()), expire)
        _voice_hold_timer.daemon = True
        _voice_hold_timer.start()
        return _voice_hold_ready


def _voice_hold_position(channel, position):
    # Shared by public setters, direct animation wire writes and proprioception.
    lease = _photo_hold
    if (lease and time.monotonic() < lease[0] and not _program_servo_updates_blocked()
            and channel in lease[1]):
        return lease[1][channel]
    cfg = config.SERVO_CHANNELS['heroarm']
    if channel == cfg['ch'] and voice_enrollment_hold_active():
        return (cfg['min'] + cfg['max']) // 2
    return position

# breathing_thread stop event — set by shutdown()
_stop_breathing = threading.Event()
_breathing_emotion = "neutral"
_breathing_lock = threading.Lock()

# Set while a scripted arm gesture or speech-reactive gesture owns arm channels.
_arm_idle_pause = threading.Event()

# Set while a SCRIPTED arm gesture (e.g. the wave-back) owns the arm channels. When set,
# the speech-reactive "talking with the hands" motion yields the ARM to the gesture (head
# and visor keep talking) — otherwise the per-frame talking motion overrides the wave and
# you see no wave. Distinct from _arm_idle_pause (which is set during ALL speech to stand
# down the idle wander, so it can't be used to mean "a gesture owns the arm").
_scripted_arm_gesture = threading.Event()


def begin_arm_gesture() -> None:
    """Mark that a scripted arm gesture owns the arm; speech motion will leave the arm
    channels alone (head/visor keep talking) until end_arm_gesture()."""
    _scripted_arm_gesture.set()


def end_arm_gesture() -> None:
    _scripted_arm_gesture.clear()


def arm_gesture_active() -> bool:
    return _scripted_arm_gesture.is_set()


# The head's equivalent of the above: set while a scripted HEAD gesture owns the
# neck/lift/tilt/visor. Speech motion and the listening loop both yield those channels
# while it's set, which is what lets a gesture HOLD a pose — without it the 0.12 s
# talking/listening ticks overwrite the pose before anyone sees it. Face tracking has
# its own suspend (consciousness.suspend_face_tracking), which such a gesture should
# call for the same reason.
_scripted_head_gesture = threading.Event()
_scripted_head_channels: "set[int]" = set()


def begin_head_gesture(channels: "list[int] | tuple[int, ...] | None" = None) -> None:
    """Mark that a scripted head gesture owns some head channels; speech and listening
    motion leave exactly those alone until end_head_gesture().

    Pass the channels the gesture actually holds — a gesture that only tilts has no
    business freezing the neck, and the channels it does NOT name keep their normal
    talking motion and their normal motion profile.
    """
    global _scripted_head_channels
    selected = config.HEAD_CHANNELS if channels is None else channels
    owned = {int(ch) for ch in selected}
    # The two head servos are one unit: holding one holds the other, or the next
    # talking tick would write a lone channel and be refused.
    if owned.intersection(HEAD_PAIR):
        owned.update(HEAD_PAIR)
    _scripted_head_channels = owned
    _scripted_head_gesture.set()


def end_head_gesture() -> None:
    global _scripted_head_channels
    _scripted_head_gesture.clear()
    _scripted_head_channels = set()


def head_gesture_active() -> bool:
    return _scripted_head_gesture.is_set()


def head_gesture_channels() -> "set[int]":
    """Head channels a scripted gesture currently owns (empty when none does)."""
    if not _scripted_head_gesture.is_set():
        return set()
    return set(_scripted_head_channels)


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
# Queeny mode gives the wrist (and the elbow's ramp) their own motion profile for the
# duration of a line; True while that override is on the Maestro so end_speech_motion
# knows to hand the channels back to the normal speech profile.
_pride_arm_profile = False

# Listening-motion state: gentle "I'm processing you" body language that runs
# from speech onset through transcription/LLM/TTS so Rex isn't frozen while he
# thinks. Subtler than speech motion. Breathing + face tracking yield to it.
_listening_active = threading.Event()
_listening_thread: "threading.Thread | None" = None
_listening_lock = threading.Lock()

# ── Channel index lookups ──────────────────────────────────────────────────────


def _require_independent_channels(channels) -> None:
    """Throttle targets need a coupled planner; generic writes must not bypass it."""
    reserved = {cfg["ch"] for cfg in config.THROTTLE_SERVO_CHANNELS.values()}
    if reserved.intersection(channels):
        raise ValueError("Throttle joints require the dedicated coupled motion controller")


def _apply_throttle_startup_locked() -> None:
    """Opt-in shoulder-only commissioning, with elbow/wrist physically unplugged.

    This is a commanded pulse, not shaft feedback or a collision-safe move.
    Repeat on explicit connect; transport retries do not issue extra poses.
    """
    if config.THROTTLE_ARM_ENABLED or not config.THROTTLE_SHOULDER_ENABLED:
        return
    cfg = config.THROTTLE_SERVO_CHANNELS["throttle_shoulder"]
    target = config.THROTTLE_SHOULDER_STARTUP
    if target is None or not cfg["min"] <= target <= cfg["max"]:
        raise RuntimeError("A valid explicit throttle shoulder startup target is required")
    throttle_motion.invalidate_park()
    for cmd, value in ((_CMD_SET_ACCEL, cfg["acceleration"]),
                       (_CMD_SET_SPEED, cfg["speed"]),
                       (_CMD_SET_TARGET, target)):
        if not _send_command_locked(_encode(cmd, cfg["ch"], value)):
            raise RuntimeError("Could not apply throttle shoulder startup settings")
    _log.info("Throttle shoulder commissioning target: %.2f us", target / 4)


def _channel_cfg(channel: int) -> "dict | None":
    for cfg in config.SERVO_CHANNELS.values():
        if cfg["ch"] == channel:
            return cfg
    return None


def _channel(name: str) -> int:
    return int(config.SERVO_CHANNELS[name]["ch"])


def _channel_neutral(channel: int) -> int:
    cfg = _channel_cfg(channel)
    return int(cfg["neutral"]) if cfg else 6000


def _clamp(channel: int, position: int) -> int:
    cfg = _channel_cfg(channel)
    if cfg is None:
        return position
    return max(cfg["min"], min(cfg["max"], position))


def _validate_pairs(targets: "dict[int, int]") -> None:
    """Refuse a lone head or poker channel, a head pair outside the linkage's
    envelope, or a poker pair above its clearance ceiling. Raises ValueError."""
    head_pose.validate_head_pair(targets)
    head_pose.validate_poker_pair(targets)


def _complete_poker_pair(targets: "dict[int, int]") -> "dict[int, int]":
    """A lone poker elbow or wrist takes its partner from the current command."""
    present = set(POKER_PAIR).intersection(targets)
    if present and present != set(POKER_PAIR):
        with _lock:
            for ch in POKER_PAIR:
                targets.setdefault(ch, int(_commanded_positions.get(ch, _channel_neutral(ch))))
    return targets


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


def _pride_flair_active() -> bool:
    """Is queeny mode on? Then the talking motion goes flamboyant (config
    PRIDE_SPEECH_*). Lazy, guarded import — the hardware layer must never hard-depend
    on intelligence/, and a pride module that fails to import just means no flair."""
    try:
        from intelligence import pride
        return bool(pride.is_active())
    except Exception:
        return False


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
    """Neck and lift at neutral, the head pair level and straight."""
    pose = {
        _channel("neck"): config.SERVO_CHANNELS["neck"]["neutral"],
        _channel("headlift"): config.SERVO_CHANNELS["headlift"]["neutral"],
    }
    pose.update(head_pose.head_pulses(0.0, 0.0))
    return pose


_face_tracking_baseline.update(_default_head_pose())


def _head_pair_pose(positions: "dict") -> "tuple[float, float]":
    """(pitch, tilt) of a name-keyed or channel-keyed position mapping; level and
    straight when the pair is not fully known."""
    pair = {}
    for ch in HEAD_PAIR:
        value = positions.get(ch, positions.get(_CHANNEL_TO_NAME.get(ch)))
        if value is None:
            return 0.0, 0.0
        pair[ch] = int(value)
    try:
        return head_pose.head_pose(pair)
    except Exception:
        return 0.0, 0.0


def commanded_positions() -> "dict[int, int]":
    """A copy of the last pulse commanded (or read back at connect) per channel."""
    with _lock:
        return dict(_commanded_positions)


def commanded_head_pose() -> "tuple[float, float]":
    """The (pitch, tilt) the head pair was last commanded to."""
    with _lock:
        pair = {ch: _commanded_positions.get(ch) for ch in HEAD_PAIR}
    return _head_pair_pose(pair)


def _derive_body_state(positions: dict) -> str:
    neck_cfg = config.SERVO_CHANNELS["neck"]
    lift_cfg = config.SERVO_CHANNELS["headlift"]

    neck = positions.get("neck", neck_cfg["neutral"])
    lift = positions.get("headlift", lift_cfg["neutral"])
    pitch, _tilt = _head_pair_pose(positions)

    neck_dead = 450
    lift_dead = 450
    pitch_dead = 0.25
    if neck <= neck_cfg["neutral"] - neck_dead:
        return "looking_left"
    if neck >= neck_cfg["neutral"] + neck_dead:
        return "looking_right"
    if lift >= lift_cfg["neutral"] + lift_dead or pitch >= pitch_dead:
        return "looking_up"
    if lift <= lift_cfg["neutral"] - lift_dead or pitch <= -pitch_dead:
        return "looking_down"
    return "neutral"


def _record_servo_positions(channel_dict: "dict[int, int]") -> None:
    """Mirror commanded servo positions into WorldState proprioception.

    Besides the raw pulses by channel name, publishes ``self_state["head_pose"]``
    (pitch and tilt of the two-servo head) so behaviour code never has to read the
    two head channels and fit them itself."""
    channel_dict = {ch: _voice_hold_position(ch, pos) for ch, pos in channel_dict.items()}
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
        pitch, tilt = _head_pair_pose(positions)
        self_state["head_pose"] = {"pitch": round(pitch, 3), "tilt": round(tilt, 3)}
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
    return (
        _manual_override.is_set()
        or _shutdown_latch.is_set()
        or _sleep_latch.is_set()
    )


def throttle_motion_blocked(*, parking=False) -> bool:
    """Head sleep/shutdown latches permit only the throttle's bounded park path.

    Manual ownership always wins. move_throttle_pose separately restricts parking
    targets to TUCK/PARK, so this cannot admit idle or speech after the head rests.
    """
    return _manual_override.is_set() if parking else _program_servo_updates_blocked()


def latch_sleep_pose() -> None:
    """Freeze ordinary servo writes; an already-requested throttle park can finish.

    Same race as the shutdown latch, on the sleep path (field 2026-08-13: the
    "Going dark" clip's end_speech_motion drove the visor back to neutral around
    the sleep glide and nothing ever re-closed it). Unlike the shutdown latch
    this one is released — by animations.wake(), or by animations.shutdown()
    when Rex is powered down while asleep."""
    _sleep_latch.set()
    _log.info("Servo sleep latch set — sleep pose held until wake.")


def release_sleep_latch() -> None:
    if _sleep_latch.is_set():
        _sleep_latch.clear()
        _log.info("Servo sleep latch released.")


def latch_shutdown_pose() -> None:
    """Make the head/body rest pose final; only the throttle park path may finish.
    Ordinary writes no-op until process exit. The shutdown droop runs at the TOP of _shutdown()
    (theatrics first) while vision/consciousness threads are still being torn
    down — a periodic scene capture racing that window drove the visor to max +
    the neck to center right after the droop finished, and its own "restore"
    write then died against the closing port (field bug 2026-07-16). Called by
    animations.shutdown() once the rest pose has been commanded."""
    _shutdown_latch.set()
    _log.info("Servo shutdown latch set — rest pose is final.")


def manual_override_enabled() -> bool:
    """Return True when GUI manual servo override owns servo targets."""
    return _manual_override.is_set()


def set_manual_override_enabled(enabled: bool) -> None:
    """Freeze programmatic servo target updates so GUI sliders can drive servos."""
    enabled = bool(enabled)
    if enabled:
        from intelligence.voice_learning import mic_moved
        mic_moved("manual_servo_control")
        release_voice_enrollment_hold()
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
    _require_independent_channels([channel])
    targets = {channel: _clamp(channel, int(position))}
    # The GUI drives one slider at a time, but the head servos and the poker
    # elbow/wrist only ever move as pairs: fill the partner from where it is and
    # refuse a pair the linkage cannot take, rather than driving one servo alone.
    for pair in (HEAD_PAIR, POKER_PAIR):
        if channel in pair:
            with _lock:
                for other in pair:
                    targets.setdefault(other, int(_commanded_positions.get(other, _channel_neutral(other))))
    try:
        _validate_pairs(targets)
    except ValueError as exc:
        _log.warning("manual servo move refused: %s", exc)
        return False
    with _lock:
        if SERVOS_ENABLED:
            for ch, pos in targets.items():
                _send_set_target(ch, pos)
        _remember_positions(targets)
        for ch, pos in targets.items():
            if ch in (_channel("neck"), _channel("headlift"), *HEAD_PAIR):
                _face_tracking_baseline[ch] = pos
    _record_servo_positions(targets)
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


def _read_position_locked(channel: int) -> "int | None":
    """GET POSITION under the caller's lock; None when the link cannot answer."""
    if not _send_command_locked(bytes([_CMD_GET_POSITION, channel])):
        return None
    if _ser is None or not getattr(_ser, "is_open", False):
        return None
    try:
        data = _ser.read(2)
    except Exception:
        return None
    if not data or len(data) < 2:
        return None
    return struct.unpack("<H", data)[0]


def _assert_startup_rest_pose_locked() -> dict[int, int]:
    """Command gravity's own pose on weight-loaded channels that are OFF, first thing.

    With the robot fully powered off every servo goes limp: the hero arm falls to
    its elbow "rest", the head hangs fully down on its two servos, and the poker
    arm stands at its park. The Maestro's speed/acceleration limits ramp the pulse
    it SENDS, not the shaft it can't see: after a cold boot the channel is off (a
    position read returns 0), so whatever target is commanded first is a step
    change the servo chases at full torque, from wherever the joint actually is.
    That is the violent elbow jerk on startup (field 2026-08-22) and the poker arm
    dropping at the first October 10 start (an assumed elbow-down rest).

    So the first thing we ever say to an off channel is "stay where you are."
    Every later move then ramps from a position the Maestro and the physical joint
    agree on. A channel that still reads a live pulse (the program restarted with
    the servos powered) keeps that reading instead — commanding its rest would be
    the very jump this avoids. Returns the pose commanded, for the caller to mirror
    into proprioception outside the lock.
    """
    pose: dict[int, int] = {}
    live: dict[int, int] = {}
    for cfg in config.SERVO_CHANNELS.values():
        if "rest" not in cfg:
            continue
        channel = int(cfg["ch"])
        reading = _read_position_locked(channel)
        if reading:
            live[channel] = _clamp(channel, int(reading))
        else:
            pose[channel] = _clamp(channel, int(cfg["rest"]))
    for channel, position in pose.items():
        _send_set_target(channel, position)
    _remember_positions(pose)
    _remember_positions(live)
    if pose:
        _log.info(
            "Startup rest pose asserted (off channels): %s",
            ", ".join(f"{_CHANNEL_TO_NAME.get(ch, ch)}={pos}" for ch, pos in sorted(pose.items())),
        )
    if live:
        _log.info(
            "Startup kept live readings (powered channels): %s",
            ", ".join(f"{_CHANNEL_TO_NAME.get(ch, ch)}={pos}" for ch, pos in sorted(live.items())),
        )
    pose.update(live)
    return pose


def connect() -> bool:
    global _ser, _last_reconnect_attempt_at
    if not SERVOS_ENABLED:
        _log.debug("SERVOS_ENABLED=False — skipping connect")
        return False
    rest_pose: dict[int, int] = {}
    with _lock:
        _ser = _open_serial_with_retries()
        if _ser is None:
            _last_reconnect_attempt_at = time.monotonic()
            return False
        if bool(getattr(config, "SERVO_APPLY_STARTUP_MOTION_PROFILE", True)):
            _apply_startup_motion_profile_locked()
        try:
            _apply_throttle_startup_locked()
        except RuntimeError:
            _close_serial_locked()
            raise
        if bool(getattr(config, "SERVO_ASSERT_STARTUP_REST_POSE", True)):
            rest_pose = _assert_startup_rest_pose_locked()
    if rest_pose:
        _record_servo_positions(rest_pose)
    return True


def disconnect() -> None:
    global _ser
    from sequences import throttle_arm
    throttle_arm.stop()
    _stop_breathing.set()
    with _lock:
        _close_serial_locked()


def connected() -> bool:
    """True when the Maestro serial link is currently open (live status)."""
    ser = _ser
    return ser is not None and bool(getattr(ser, "is_open", False))


# ── Core command primitives ────────────────────────────────────────────────────


def throttle_connection():
    """Lease the current transport; a reconnect invalidates throttle animation."""
    with _lock:
        if not SERVOS_ENABLED or not connected():
            raise RuntimeError('Throttle requires an open Maestro connection')
        return _ser


def _throttle_write_locked(connection, packet):
    # Deliberately no reconnect/replay: other channels may recover their link,
    # but this arm must not resume motion on a new, unverified controller session.
    if connection is not _ser or not connected() or not SERVOS_ENABLED:
        raise RuntimeError('Throttle Maestro connection changed or closed')
    try:
        if _ser.write(packet) != len(packet):
            raise OSError('Incomplete throttle command')
    except _SERIAL_ERRORS:
        _close_serial_locked()
        raise


def _read_throttle_locked(connection):
    result = {}
    for ch in throttle_motion.CHANNELS:
        _throttle_write_locked(connection, bytes([_CMD_GET_POSITION, ch]))
        try:
            data = _ser.read(2)
            if len(data) != 2:
                raise OSError('Incomplete throttle position reply')
        except _SERIAL_ERRORS:
            _close_serial_locked()
            raise
        result[ch] = struct.unpack('<H', data)[0]
    return result


def read_throttle_pose(connection):
    """Read current output pulses under the shared serial lock (not shaft feedback)."""
    with _lock:
        return _read_throttle_locked(connection)


def _throttle_packet(pose):
    packet = bytearray([0x9F, 3, 8])
    for ch in throttle_motion.CHANNELS:
        packet.extend((pose[ch] & 127, pose[ch] >> 7))
    return packet


def move_throttle_pose(connection, pose, *, speed_caps, accel_caps, duration,
                       cancel=None, cold_start=False, parking=False,
                       coordinated=False):
    """Validate the full travel box and send all three targets atomically.

    The dedicated worker owns targets; speech/idle callbacks never write them.
    Profiles, fresh pulse reads, and the batch are serialized with hero/head I/O.
    """
    limits = {cfg['ch']: cfg for cfg in config.THROTTLE_SERVO_CHANNELS.values()}
    with _lock:
        if parking and pose not in (throttle_motion.TUCK, throttle_motion.PARK,
                                    throttle_motion.FULL_DOWN_RAISED,
                                    throttle_motion.RETRACT_RAISED):
            raise ValueError('Only verified retraction targets may finish after a head latch')
        if (not config.THROTTLE_ARM_ENABLED or throttle_motion_blocked(parking=parking)
                or (cancel is not None and cancel.is_set())):
            raise InterruptedError('Throttle motion is disabled or canceled')
        throttle_motion.validate_pose(pose, limits)
        current = _read_throttle_locked(connection)
        if cold_start:
            if (any(current.values()) or pose != throttle_motion.PARK
                    or not throttle_motion.cold_start_park_known()):
                raise ValueError('Cold throttle startup requires a previously completed park')
            current = dict(throttle_motion.PARK)
        else:
            throttle_motion.validate_pose(current, limits)
        if not throttle_motion.clearance_box(current, pose):
            raise ValueError('Throttle transition outside the measured clearance model')
        pace = dict(config.THROTTLE_PACE)
        if coordinated:
            # High-five joints share proportional progress instead of the
            # independent pace used for general expressive animation.
            pace = dict.fromkeys(pace, min(pace.values()))
        profile = throttle_motion.profiles(current, pose, limits, speed_caps, accel_caps,
                                           duration, pace=pace)
        throttle_motion.invalidate_park()
        for ch, (speed, accel) in profile.items():
            _throttle_write_locked(connection, _encode(_CMD_SET_ACCEL, ch, accel))
            _throttle_write_locked(connection, _encode(_CMD_SET_SPEED, ch, speed))
        if throttle_motion_blocked(parking=parking) or (cancel is not None and cancel.is_set()):
            raise InterruptedError('Throttle motion canceled before target write')
        _throttle_write_locked(connection, _throttle_packet(pose))


def hold_throttle_pose(connection):
    """Cancel an in-flight target at its current pulse, without reconnecting."""
    limits = {cfg['ch']: cfg for cfg in config.THROTTLE_SERVO_CHANNELS.values()}
    with _lock:
        pose = _read_throttle_locked(connection)
        throttle_motion.validate_pose(pose, limits)
        throttle_motion.invalidate_park()
        _throttle_write_locked(connection, _throttle_packet(pose))


def _send_set_target(channel: int, position: int) -> None:
    """Send Maestro compact protocol Set Target command (0x84)."""
    _require_independent_channels([channel])
    _send_command_locked(_encode(_CMD_SET_TARGET, channel, _clamp(channel, int(_voice_hold_position(channel, position)))))


def _send_set_speed(channel: int, speed: int) -> None:
    """Send Maestro compact protocol Set Speed command (0x87)."""
    _require_independent_channels([channel])
    _send_command_locked(_encode(_CMD_SET_SPEED, channel, max(0, int(speed))))


def _send_set_acceleration(channel: int, acceleration: int) -> None:
    """Send Maestro compact protocol Set Acceleration command (0x89)."""
    _require_independent_channels([channel])
    _send_command_locked(_encode(_CMD_SET_ACCEL, channel, max(0, int(acceleration))))


def _remember_positions(channel_dict: "dict[int, int]") -> None:
    for channel, position in channel_dict.items():
        if channel in _CHANNEL_TO_NAME:
            _commanded_positions[channel] = _clamp(channel, int(_voice_hold_position(channel, position)))


def set_servo(channel: int, position: int) -> None:
    """Move channel to position (quarter-microseconds), clamped to channel limits.

    The head servos (2/11) and the poker elbow/wrist (12/13) never move alone: use
    set_servos with both, set_head_pose, or move_poker. A lone poker channel is
    completed from its partner's current command; a lone head channel is refused."""
    if _program_servo_updates_blocked():
        return
    _require_independent_channels([channel])
    if channel in HEAD_PAIR or channel in POKER_PAIR:
        set_servos({channel: int(position)})
        return
    position = _clamp(channel, _voice_hold_position(channel, position))
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


def set_servos(channel_dict: "dict[int, int]", *, parking: bool = False) -> None:
    """Set multiple channels in one pass. channel_dict maps channel int → position.

    Head channels 2 and 11 must both be present and form a pair the linkage can take
    (ValueError otherwise). The poker elbow/wrist are completed from their current
    command when one is missing, checked against the clearance ceiling, and routed
    through the staged move (one joint at a time) on a background thread; the other
    channels are sent at once. ``parking`` lets the shutdown/sleep park finish after
    the pose latch, like the throttle's own park path."""
    if _program_servo_updates_blocked() and not (parking and not _manual_override.is_set()):
        return
    _require_independent_channels(channel_dict)
    channel_dict = _complete_poker_pair({int(ch): int(pos) for ch, pos in channel_dict.items()})
    _validate_pairs(channel_dict)
    poker = {ch: channel_dict.pop(ch) for ch in POKER_PAIR if ch in channel_dict}
    if poker:
        move_poker(poker[head_pose.POKER_ELBOW_CH], poker[head_pose.POKER_WRIST_CH],
                   block=False, parking=parking)
    if not channel_dict:
        return
    channel_dict = {ch: _clamp(ch, int(_voice_hold_position(ch, pos))) for ch, pos in channel_dict.items()}
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


def set_head_pose(
    pitch: "float | None" = None,
    tilt: "float | None" = None,
    *,
    lift: "int | None" = None,
    neck: "int | None" = None,
) -> dict[int, int]:
    """Command the two-servo head as a pitch (-1 down .. 1 up) and a tilt (-1 cocked to
    Rex's left .. 1 right), keeping whichever axis is left out where it is; optional
    lift/neck pulses ride in the same write. Returns the targets sent."""
    pitch0, tilt0 = commanded_head_pose()
    targets = head_pose.head_pulses(pitch0 if pitch is None else pitch, tilt0 if tilt is None else tilt)
    if lift is not None:
        targets[_channel("headlift")] = int(lift)
    if neck is not None:
        targets[_channel("neck")] = int(neck)
    set_servos(targets)
    return targets


def _poker_stage_seconds(stage: "dict[int, int]", current: "dict[int, int]", speed: int) -> float:
    """Time for the Maestro to glide one stage: speed units are 100 q-us per second."""
    travel = max(abs(int(stage[ch]) - int(current.get(ch, stage[ch]))) for ch in POKER_PAIR)
    return travel / (max(1, int(speed)) * 100.0) + 0.15


def move_poker(
    elbow: int,
    wrist: int,
    *,
    speed: int = 20,
    acceleration: int = 5,
    block: bool = True,
    parking: bool = False,
) -> bool:
    """Move the poker elbow (12) and wrist (13) together to a clear pose, in stages.

    The elbow can rise fully only while the wrist is at or below 1564 us; curled further
    in, the wrist meets the arm and the elbow's ceiling falls. The target pair is checked
    against the measured ceiling before anything moves (ValueError when it collides), and
    the path is staged one joint at a time — elbow down to the lower of the current and
    target elbow, then the wrist, then the elbow up — so the ceiling is never crossed on
    the way. With ``block`` the caller waits for the glide; otherwise the stages run on a
    background thread and a newer request supersedes an older one. Returns True when the
    final stage was sent."""
    global _poker_generation
    target = {head_pose.POKER_ELBOW_CH: _clamp(head_pose.POKER_ELBOW_CH, int(elbow)),
              head_pose.POKER_WRIST_CH: _clamp(head_pose.POKER_WRIST_CH, int(wrist))}
    head_pose.validate_poker_pair(target)
    if _program_servo_updates_blocked() and not (parking and not _manual_override.is_set()):
        return False
    with _lock:
        start = {ch: int(_commanded_positions.get(ch, _channel_neutral(ch))) for ch in POKER_PAIR}
    if not head_pose.poker_clear(start):
        # A reading the ceiling does not explain (a board restart, a manual pose):
        # drop the elbow first, from where the wrist says it may be.
        start[head_pose.POKER_ELBOW_CH] = min(start[head_pose.POKER_ELBOW_CH],
                                              head_pose.poker_elbow_max(start[head_pose.POKER_WRIST_CH]))
    route = head_pose.poker_route(start, target)
    if not SERVOS_ENABLED:
        _log.debug("move_poker no-op: SERVOS_ENABLED=False (%s stages to %s)", len(route), target)
        if _gui_servo_sim_enabled():
            with _lock:
                _remember_positions(target)
            _record_servo_positions(target)
        return True
    with _poker_lock:
        _poker_generation += 1
        generation = _poker_generation

    def run() -> bool:
        current = dict(start)
        for stage in route:
            with _poker_lock:
                if generation != _poker_generation:
                    return False       # superseded by a newer poker move
            if _program_servo_updates_blocked() and not (parking and not _manual_override.is_set()):
                return False
            with _lock:
                for ch in POKER_PAIR:
                    _send_set_acceleration(ch, acceleration)
                    _send_set_speed(ch, speed)
                for ch, pos in stage.items():
                    _send_set_target(ch, pos)
                _remember_positions(stage)
            _record_servo_positions(stage)
            wait = _poker_stage_seconds(stage, current, speed)
            current = dict(stage)
            if stage is not route[-1]:
                time.sleep(wait)
        return True

    if block:
        return run()
    threading.Thread(target=run, daemon=True, name="poker-arm-stages").start()
    return True


def wait_for_arrival(targets: "dict[int, int]", timeout: float = 3.0, tolerance: int = 8) -> str:
    """Poll the board until every channel reads within ``tolerance`` of its target, or
    ``timeout`` passes. Returns "" on arrival, else the channels still short with their
    pulses (``ch3@6400->6976``), which is also logged.

    A Maestro that silently clamps a target outside the range stored in its own settings
    looks exactly like a servo that never arrives; naming the channel is how the owner
    tells the two apart (first live shutdown on the two-servo head timed out with a
    bare error and nothing parked)."""
    if not SERVOS_ENABLED or not targets:
        return ""
    deadline = time.monotonic() + max(0.0, float(timeout))
    short = ""
    while True:
        current: dict[int, int] = {}
        for ch in targets:
            reading = get_servo(ch)
            if reading is not None:
                current[ch] = int(reading)
        short = head_pose.describe_short(targets, current, tolerance)
        if not short or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if short:
        _log.warning("Servo arrival timeout after %.1fs: %s", timeout, short)
    return short


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


def _baseline_position(channel: int, *, ignore_commanded: bool = False) -> int:
    """Where speech/breathing motion should orbit this channel.

    Normally the last commanded position — that's the gaze Rex is holding. Pass
    ignore_commanded while a scripted head gesture owns the channel: its held pose is
    a gesture, not a gaze, and capturing it as the speech baseline would leave Rex
    talking the rest of the line from wherever the gesture parked him.
    """
    cfg = _channel_cfg(channel)
    if cfg is None:
        return _commanded_positions.get(channel, 6000)
    fallback = _face_tracking_baseline.get(channel, cfg["neutral"])
    if ignore_commanded:
        return _clamp(channel, fallback)
    return _clamp(channel, _commanded_positions.get(channel, fallback))


def set_face_tracking_baseline(
    *,
    neck: int | None = None,
    lift: int | None = None,
    pitch: float | None = None,
    tilt: float | None = None,
    head: "dict[int, int] | None" = None,
) -> None:
    """
    Store the last face-tracking head pose so speech gestures wobble around it
    instead of recentering away from the person Rex is addressing.

    The head pair is given as ``pitch``/``tilt`` fractions (either alone keeps the
    other axis where the baseline has it) or as a ready pulse pair in ``head``.
    """
    if _program_servo_updates_blocked():
        return
    updates: dict[int, int] = {}
    mapping = {
        _channel("neck"): neck,
        _channel("headlift"): lift,
    }
    for channel, value in mapping.items():
        if value is not None:
            updates[channel] = _clamp(channel, int(value))
    if head is not None:
        pair = {ch: int(head[ch]) for ch in HEAD_PAIR}
        head_pose.validate_head_pair(pair)
        updates.update(pair)
    elif pitch is not None or tilt is not None:
        for name, value in (("pitch", pitch), ("tilt", tilt)):
            if value is not None and abs(float(value)) > 1.5:
                raise ValueError(f"head {name} is a fraction in -1..1, not quarter-microseconds: {value}")
        with _lock:
            current = {ch: _face_tracking_baseline.get(ch, _commanded_positions.get(ch)) for ch in HEAD_PAIR}
        pitch0, tilt0 = _head_pair_pose(current)
        updates.update(head_pose.head_pulses(pitch0 if pitch is None else pitch,
                                             tilt0 if tilt is None else tilt))
    if not updates:
        return
    with _lock:
        _face_tracking_baseline.update(updates)
        _commanded_positions.update(updates)
        if _speech_active.is_set():
            _speech_baseline.update(updates)


def reset_face_tracking_baseline() -> None:
    """Reset gaze baseline to the configured neutral head pose."""
    with _lock:
        _face_tracking_baseline.update(_default_head_pose())


def begin_speech_motion(emotion: str = "neutral") -> None:
    """Capture the current gaze/pose and prepare speech-reactive servo motion."""
    global _last_speech_move_at, _speech_hand_counter
    global _speech_elbow_target, _speech_elbow_direction, _next_speech_elbow_at
    global _speech_poker_target, _speech_poker_direction, _next_speech_poker_at
    global _speech_emotion_frame, _pride_arm_profile

    if _program_servo_updates_blocked():
        return

    # Rex is about to talk — hand the body off from listening motion to speech.
    stop_listening_motion()

    frame = _resolve_speech_emotion_frame(emotion)
    motion = frame.get("speech_motion") or {}

    # A scripted head gesture mid-hold must not become the baseline Rex talks around.
    head_owned = head_gesture_channels()

    pause_arm_idle()
    with _lock:
        _speech_emotion_frame = frame
        _speech_baseline.clear()
        for channel in (_channel("neck"), _channel("headlift")):
            baseline = _baseline_position(channel, ignore_commanded=channel in head_owned)
            if channel == _channel("headlift"):
                baseline += int(_motion_float(motion, "lift_bias_qus", 0.0))
            _speech_baseline[channel] = _clamp(channel, baseline)
        # The head pair's anchor is a pitch and a tilt (the gaze the head holds), with
        # the emotion's chin bias — still written in the old tilt servo's units, so
        # convert — added to the pitch. The sway orbits this anchor.
        pair = {ch: _baseline_position(ch, ignore_commanded=ch in head_owned) for ch in HEAD_PAIR}
        pitch0, tilt0 = _head_pair_pose(pair)
        pitch0 += head_pose.pitch_from_legacy_delta(_motion_float(motion, "tilt_bias_qus", 0.0))
        _speech_baseline.update(head_pose.head_pulses(pitch0, tilt0))
        _last_speech_move_at = 0.0
        _speech_hand_counter = 0
        _speech_elbow_target = None
        _speech_elbow_direction = 1
        _next_speech_elbow_at = 0.0
        _speech_poker_target = None
        _speech_poker_direction = -1
        _next_speech_poker_at = 0.0
    _speech_active.set()
    from sequences import throttle_arm
    throttle_arm.speech_start(frame)
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
        # Same rule as the arm below, per channel: a scripted gesture set its own
        # profile for the channels it holds and speech is yielding those anyway. The
        # rest of the head still gets the speech profile — otherwise a line that starts
        # mid-gesture is delivered at whatever slow profile happened to be loaded.
        free_head = [ch for ch in config.HEAD_CHANNELS if ch not in head_owned and ch not in HEAD_PAIR]
        if free_head:
            set_motion_profile(
                free_head,
                speed=head_speed,
                acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
            )
        if not set(HEAD_PAIR).intersection(head_owned):
            # The two head servos sway slowly together: their own gentle profile.
            set_motion_profile(
                list(HEAD_PAIR),
                speed=_get_config_int("SERVO_SPEECH_HEAD_PAIR_SPEED", 20),
                acceleration=_get_config_int("SERVO_SPEECH_HEAD_PAIR_ACCELERATION", 6),
            )
        # Don't reprofile the arm if a scripted gesture (wave-back) owns it — it set its
        # own fast speed for the wave and the speech motion is yielding the arm anyway.
        if not _scripted_arm_gesture.is_set():
            set_motion_profile(
                config.ARM_CHANNELS,
                speed=arm_speed,
                acceleration=_get_config_int("SERVO_SPEECH_ACCELERATION", 8),
            )
            # Queeny mode: the wrist can't wave at the speech profile's slew (it covers
            # under a tenth of its travel per beat), so give it — and the elbow's ramp —
            # their own profile for this line. end_speech_motion hands them back.
            _pride_arm_profile = _pride_flair_active()
            if _pride_arm_profile:
                set_motion_profile(
                    [_channel("hand")],
                    speed=_get_config_int("PRIDE_SPEECH_HAND_SPEED", 120),
                    acceleration=_get_config_int("PRIDE_SPEECH_HAND_ACCEL", 40),
                )
                set_motion_profile(
                    [_channel("elbow")],
                    acceleration=_get_config_int("PRIDE_SPEECH_ELBOW_ACCEL", 25),
                )


def end_speech_motion() -> None:
    """Return speech-owned channels toward their baseline and release arms."""
    global _speech_emotion_frame, _pride_arm_profile
    _speech_active.clear()
    from sequences import throttle_arm
    throttle_arm.speech_stop()
    # A sleep/shutdown ack clip ends INSIDE the transition — the state has already
    # flipped (or is about to latch) by the time this fires from the audio-end
    # callback. Restoring the "awake" baseline here re-opens the visor mid-droop
    # (field 2026-08-13: the "Going dark" clip's release raced the sleep glide and
    # the visor never fully closed). Keep the bookkeeping, skip the pose.
    if not _automatic_motion_allowed():
        # A profile is not a pose — handing the wrist/elbow back costs no motion, and
        # leaving queeny mode's fast wrist armed would make whatever moves the arm next
        # (after a sleep, say) snap.
        if _pride_arm_profile:
            _pride_arm_profile = False
            set_motion_profile(
                config.ARM_CHANNELS,
                speed=_get_config_int("SERVO_DEFAULT_SPEED", 40),
                acceleration=_get_config_int("SERVO_DEFAULT_ACCELERATION", 8),
            )
        _speech_emotion_frame = {}
        set_breathing_emotion("neutral")
        return
    # If a scripted arm gesture (wave-back) owns the arm, don't yank the arm back to neutral
    # when the line finishes mid-gesture — leave the arm channels to the gesture; only the
    # head/visor return to baseline.
    arm_owned = _scripted_arm_gesture.is_set()
    # Likewise for a scripted HEAD gesture still holding its pose: let it finish and
    # restore the channels it holds itself, rather than yanking them back mid-gesture.
    head_owned = head_gesture_channels()

    def _baseline_pose() -> dict[int, int]:
        baseline = dict(_speech_baseline) if _speech_baseline else _default_head_pose()
        baseline[_channel("visor")] = config.SERVO_CHANNELS["visor"]["neutral"]
        for channel in head_owned:
            baseline.pop(channel, None)
        if not arm_owned:
            baseline[_channel("elbow")] = config.SERVO_CHANNELS["elbow"]["neutral"]
            baseline[_channel("hand")] = config.SERVO_CHANNELS["hand"]["neutral"]
            baseline[_channel("pokerarm")] = config.SERVO_CHANNELS["pokerarm"]["neutral"]
            baseline[_channel("heroarm")] = config.SERVO_CHANNELS["heroarm"]["neutral"]
        return baseline

    try:
        if SERVOS_ENABLED:
            # Don't reset the arm channels' motion profile while a gesture owns them
            # (it set its own fast speed for the wave).
            profile_channels = [ch for ch in config.HEAD_CHANNELS if ch not in head_owned]
            if not arm_owned:
                profile_channels += list(config.ARM_CHANNELS)
            set_motion_profile(
                profile_channels,
                speed=_get_config_int("SERVO_DEFAULT_SPEED", 40),
                acceleration=_get_config_int("SERVO_DEFAULT_ACCELERATION", 8),
            )
            set_servos(_baseline_pose())
        elif _gui_servo_sim_enabled():
            set_servos(_baseline_pose())
    finally:
        # The reprofile above already put the arm back on the default speed/accel
        # (it covers ARM_CHANNELS whenever a scripted gesture doesn't own them); if a
        # gesture DOES own them it set its own profile, so either way the override is
        # no longer ours.
        _pride_arm_profile = False
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

    from sequences import throttle_arm
    throttle_arm.speech_level(intensity)
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
    # Queeny mode talks with its hands: a wider wrist swing on a quicker beat and a
    # much bigger elbow. Read once per update so a mode that expires mid-line simply
    # settles back on the next beat.
    pride_flair = _pride_flair_active()
    pride_hand_mult = _get_config_float("PRIDE_SPEECH_HAND_AMP_MULT", 1.7) if pride_flair else 1.0
    pride_elbow_mult = (
        _get_config_float("PRIDE_SPEECH_ELBOW_AMP_MULT", 1.9) if pride_flair else 1.0
    )
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
    visor_ch = _channel("visor")
    elbow_ch = _channel("elbow")
    hand_ch = _channel("hand")
    poker_ch = _channel("pokerarm")
    hero_ch = _channel("heroarm")

    with _lock:
        base_neck = _clamp(neck_ch, _speech_baseline.get(neck_ch, _baseline_position(neck_ch)))
        base_lift = _clamp(lift_ch, _speech_baseline.get(lift_ch, _baseline_position(lift_ch)))
        base_pair = {ch: _speech_baseline.get(ch, _baseline_position(ch)) for ch in HEAD_PAIR}
    base_pitch, base_tilt = _head_pair_pose(base_pair)

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

    # The two head servos sway slowly TOGETHER around the anchor: a small nod with a
    # slight cock to each side, on one smooth sine (never a fresh random jump — the
    # whole head rides the linkage, and reversals there read as shaking). The neck
    # and lift are sturdier and keep the lively random wobble.
    period = max(0.5, _get_config_float("SERVO_SPEECH_SWAY_PERIOD_SECS", 2.4))
    sway = math.sin(math.tau * now / period)
    sway_scale = _motion_float(motion, "tilt_wobble_mult", 1.0) * (0.35 + expressive_intensity) / 1.35
    pitch = base_pitch + _get_config_float("SERVO_SPEECH_PITCH_SWAY", 0.06) * sway * sway_scale
    tilt = base_tilt + _get_config_float("SERVO_SPEECH_TILT_SWAY", 0.10) * sway * sway_scale
    targets: dict[int, int] = {
        neck_ch: _clamp(neck_ch, base_neck + random.randint(-neck_wobble, neck_wobble)),
        lift_ch: _clamp(lift_ch, base_lift + random.randint(-lift_wobble, lift_wobble)),
    }
    targets.update(head_pose.head_pulses(pitch, tilt))

    # Visor (inverted since October 10: LOW pulses are UP). While talking it lives in
    # the speaking band between the fully-expressive "up" edge and the camera-clear
    # "down" edge; an emotion's open fraction says how far up its resting point sits
    # and the wave opens it further from there. Clamp AFTER the jitter too, so a
    # random offset can never cross the camera boundary.
    up_edge, down_edge = (int(v) for v in getattr(config, "VISOR_SPEECH_RANGE_QUS", head_pose.VISOR_SPEECH))
    down_edge = min(down_edge, _get_config_int("VISOR_CAMERA_CLEAR_MAX_QUS", head_pose.VISOR_CAMERA_CLEAR_MAX))
    down_edge = min(down_edge, int(_motion_float(motion, "visor_down_qus", down_edge)))
    open_frac = max(0.0, min(1.0, _motion_float(motion, "visor_open_floor_frac", 0.55)))
    rest_point = down_edge - int((down_edge - up_edge) * open_frac)
    visor_wave = 0.5 + 0.5 * math.sin(now * 8.0)
    visor_swing = int(
        (rest_point - up_edge)
        * (0.35 + 0.40 * expressive_intensity)
        * _motion_float(motion, "visor_swing_mult", 1.0)
    )
    targets[visor_ch] = _clamp(
        visor_ch,
        max(up_edge, min(down_edge, int(rest_point - visor_wave * visor_swing) + random.randint(-45, 45))),
    )

    elbow_lo, elbow_hi = config.SERVO_CHANNELS["elbow"]["min"], config.SERVO_CHANNELS["elbow"]["max"]
    if _speech_elbow_target is None or now >= _next_speech_elbow_at:
        span = elbow_hi - elbow_lo
        center = int(elbow_lo + span * 0.55)
        amplitude = int(
            span
            * (0.10 + 0.12 * arm_intensity)
            * _motion_float(motion, "elbow_amp_mult", 1.0)
            * pride_elbow_mult
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
    hand_divisor = max(
        1,
        _get_config_int("PRIDE_SPEECH_HAND_DIVISOR", 2)
        if pride_flair
        else _get_config_int("SERVO_SPEECH_HAND_DIVISOR", 3),
    )
    if _speech_hand_counter % hand_divisor == 0:
        hand_cfg = config.SERVO_CHANNELS["hand"]
        center = hand_cfg["neutral"]
        amplitude = int(
            (hand_cfg["max"] - hand_cfg["min"])
            * (0.08 + 0.12 * arm_intensity)
            * _motion_float(motion, "hand_amp_mult", 1.0)
            * pride_hand_mult
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

    # If a scripted arm gesture (e.g. the wave-back) owns the arm, leave the arm channels
    # to it — keep only the head/visor talking motion. Otherwise the per-frame talking
    # motion overrides the wave and you see no wave. Same for a scripted head gesture
    # holding a pose: without this the next talking tick, 0.12 s later, erases it.
    if _scripted_arm_gesture.is_set():
        for ch in config.ARM_CHANNELS:
            targets.pop(ch, None)
    for ch in head_gesture_channels():
        targets.pop(ch, None)
    if not targets:
        return

    set_servos(targets)


def listening_pitch_nod() -> float:
    """How far (pitch fraction) the listening nod dips the head pair."""
    return _get_config_float("SERVO_LISTENING_PITCH_NOD", 0.07)


# ── Listening motion (subtle "I'm thinking about what you said" feedback) ───────
# Runs from the moment VAD/Whisper hears the user through transcription → LLM →
# TTS, so Rex shows he's listening instead of freezing. Deliberately gentler and
# slower than speech motion: small head nods orbiting the current gaze, a slow
# visor flutter, and occasional small arm/hand shifts. Breathing and face
# tracking yield while it's active (see breathing_thread and
# consciousness._step_face_tracking), and it yields the instant Rex speaks.

def listening_motion_active() -> bool:
    return _listening_active.is_set()


# Per-listening-session motion plan: random phases/periods for the slow sine drifts and
# the schedule of the next smooth nod. Seeded by start_listening_motion() so every
# session sways a little differently without any per-tick randomness (per-tick random
# jumps are exactly what read as "stuttering").
_listening_session: dict = {}


def _begin_listening_session(now: float) -> None:
    def _period(base: float) -> float:
        return base * random.uniform(0.8, 1.2)

    _listening_session.clear()
    _listening_session.update({
        "t0": now,
        "phase_neck": random.uniform(0.0, math.tau),
        "phase_lift": random.uniform(0.0, math.tau),
        "phase_visor": random.uniform(0.0, math.tau),
        "phase_elbow": random.uniform(0.0, math.tau),
        "phase_hand": random.uniform(0.0, math.tau),
        "phase_hero": random.uniform(0.0, math.tau),
        "period_neck": _period(_get_config_float("SERVO_LISTENING_NECK_PERIOD_SECS", 7.0)),
        "period_lift": _period(_get_config_float("SERVO_LISTENING_LIFT_PERIOD_SECS", 5.2)),
        "period_visor": _period(_get_config_float("SERVO_LISTENING_VISOR_PERIOD_SECS", 4.5)),
        "period_elbow": _period(6.5),
        "period_hand": _period(8.5),
        "period_hero": _period(7.5),
        "next_nod_at": now + random.uniform(
            _get_config_float("SERVO_LISTENING_NOD_MIN_GAP_SECS", 2.5),
            _get_config_float("SERVO_LISTENING_NOD_MAX_GAP_SECS", 6.0),
        ),
        "nod_started_at": None,
    })


def _listening_nod_progress(now: float) -> float:
    """0..1 progress through the current smooth nod, or 0.0 when not nodding.

    Advances the session's nod schedule as a side effect: starts a nod when its
    time arrives and books the next one when it completes.
    """
    s = _listening_session
    if not s:
        return 0.0
    dur = max(0.2, _get_config_float("SERVO_LISTENING_NOD_SECS", 1.4))
    started = s.get("nod_started_at")
    if started is None:
        if now >= float(s.get("next_nod_at") or 0.0):
            s["nod_started_at"] = now
            return 0.0
        return 0.0
    u = (now - float(started)) / dur
    if u >= 1.0:
        s["nod_started_at"] = None
        s["next_nod_at"] = now + random.uniform(
            _get_config_float("SERVO_LISTENING_NOD_MIN_GAP_SECS", 2.5),
            _get_config_float("SERVO_LISTENING_NOD_MAX_GAP_SECS", 6.0),
        )
        return 0.0
    return max(0.0, min(1.0, u))


def _listening_targets(now: float) -> dict[int, int]:
    """One tick of smooth listening pose targets, orbiting the current gaze.

    Continuous, eased motion: every channel follows a slow sine drift (per-session
    random phase/period), and the head periodically dips through a smooth
    raised-cosine "mhm" nod. No per-tick randomness — the old random micro-jumps
    every beat read as stuttering, not attentiveness. Pure-ish helper (reads config,
    the face-tracking baseline, and the session plan) so it unit-tests without
    hardware.
    """
    s = _listening_session
    if not s:
        _begin_listening_session(now)
        s = _listening_session
    t = now - float(s.get("t0") or now)

    neck_ch = _channel("neck")
    lift_ch = _channel("headlift")
    visor_ch = _channel("visor")
    elbow_ch = _channel("elbow")
    hand_ch = _channel("hand")
    hero_ch = _channel("heroarm")

    with _lock:
        base = dict(_face_tracking_baseline)
    base_neck = _clamp(neck_ch, base.get(neck_ch, _baseline_position(neck_ch)))
    base_lift = _clamp(lift_ch, base.get(lift_ch, _baseline_position(lift_ch)))
    base_pitch, base_tilt = _head_pair_pose({ch: base.get(ch, _baseline_position(ch)) for ch in HEAD_PAIR})

    def _sway(amp: int, period_key: str, phase_key: str) -> float:
        period = max(0.5, float(s.get(period_key) or 1.0))
        return amp * math.sin(math.tau * t / period + float(s.get(phase_key) or 0.0))

    # Smooth "mhm" nod: a raised-cosine dip (sin² easing — zero velocity at both
    # ends). The lift drops and the two head servos pitch the head slightly down
    # together, keeping whatever cock the gaze has, then ease back.
    nod = math.sin(math.pi * _listening_nod_progress(now)) ** 2
    lift_nod = _get_config_int("SERVO_LISTENING_LIFT_NOD_QUS", 240)
    neck_amp = _get_config_int("SERVO_LISTENING_NECK_QUS", 110)
    lift_sway = _get_config_int("SERVO_LISTENING_LIFT_SWAY_QUS", 90)

    targets: dict[int, int] = {
        neck_ch: _clamp(neck_ch, int(base_neck + _sway(neck_amp, "period_neck", "phase_neck"))),
        lift_ch: _clamp(
            lift_ch,
            int(base_lift + _sway(lift_sway, "period_lift", "phase_lift") - nod * lift_nod),
        ),
    }
    targets.update(head_pose.head_pulses(base_pitch - nod * listening_pitch_nod(), base_tilt))

    # Visor: slow, shallow flutter around the awake resting position, inside the
    # speaking band (inverted servo: lower pulses are more open).
    visor_cfg = config.SERVO_CHANNELS["visor"]
    visor_swing = _get_config_int("SERVO_LISTENING_VISOR_QUS", 220)
    up_edge, down_edge = (int(v) for v in getattr(config, "VISOR_SPEECH_RANGE_QUS", head_pose.VISOR_SPEECH))
    flutter = int(
        visor_cfg["neutral"]
        + visor_swing * 0.4
        - visor_swing * (0.5 + 0.5 * math.sin(math.tau * t / max(0.5, float(s.get("period_visor") or 4.5)) + float(s.get("phase_visor") or 0.0)))
    )
    targets[visor_ch] = _clamp(visor_ch, max(up_edge, min(down_edge, flutter)))

    # Arms: slow continuous sways on their own (longer) periods — alive, never twitchy.
    elbow_cfg = config.SERVO_CHANNELS["elbow"]
    hand_cfg = config.SERVO_CHANNELS["hand"]
    hero_cfg = config.SERVO_CHANNELS["heroarm"]
    elbow_amp = _get_config_int("SERVO_LISTENING_ELBOW_QUS", 110)
    hand_amp = _get_config_int("SERVO_LISTENING_HAND_QUS", 380)
    hero_amp = _get_config_int("SERVO_LISTENING_HERO_QUS", 300)
    targets[elbow_ch] = _clamp(elbow_ch, int(elbow_cfg["neutral"] + _sway(elbow_amp, "period_elbow", "phase_elbow")))
    targets[hand_ch] = _clamp(hand_ch, int(hand_cfg["neutral"] + _sway(hand_amp, "period_hand", "phase_hand")))
    targets[hero_ch] = _clamp(hero_ch, int(hero_cfg["neutral"] + _sway(hero_amp, "period_hero", "phase_hero")))

    return targets


def _listening_loop() -> None:
    """Background loop: stream smooth listening targets until stop_listening_motion()."""
    started = time.monotonic()
    _begin_listening_session(started)
    # Safety net: never let a missed stop strand listening motion (which would
    # also keep face tracking yielded). Auto-stop after a generous ceiling.
    max_secs = _get_config_float("SERVO_LISTENING_MAX_SECS", 20.0)
    tick = max(0.05, _get_config_float("SERVO_LISTENING_TICK_SECS", 0.12))
    while _listening_active.is_set() and not _stop_breathing.is_set():
        now = time.monotonic()
        if now - started >= max_secs:
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
        try:
            targets = _listening_targets(now)
            # Yield whatever a scripted gesture is holding — this loop ticks every
            # 0.12 s, so without it a gesture's pose is erased before anyone sees it
            # (the arm hole was there for the wave-back too, not just the head).
            for ch in head_gesture_channels():
                targets.pop(ch, None)
            if _scripted_arm_gesture.is_set():
                for ch in config.ARM_CHANNELS:
                    targets.pop(ch, None)
            if not targets:
                time.sleep(tick)
                continue
            set_servos(targets)
        except Exception as exc:
            _log.debug("listening tick failed: %s", exc)
        time.sleep(tick)


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

    # One interpolated glide for every channel (move_to keeps the two head servos
    # arriving together and stages the poker elbow/wrist through clearance).
    move_to(
        {int(cfg["ch"]): _clamp(int(cfg["ch"]), int(cfg["neutral"])) for cfg in config.SERVO_CHANNELS.values()},
        step_us=step_us,
        step_delay=step_delay,
    )
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


_headlift_hum_boot_at = time.monotonic()   # process-start reference for the startup mute


def _maybe_headlift_hum(targets: "dict[int, int]", current: "dict[int, int]") -> None:
    """Fire the head-lift motion hum for a SUSTAINED, larger-travel lift sweep.

    Only ``move_to`` sweeps reach here — face-tracking / gaze micro-steps use the
    direct ``set_servo(s)`` path and never hum. Gated on: total sweep travel >=
    SOUND_EFFECTS_HEADLIFT_MIN_TRAVEL_QUS, normal operation only (IDLE/ACTIVE — the
    shutdown droop and sleep poses stay silent), and a startup mute window (the boot
    sound already owns that register). Direction: higher qus = head up
    (see _derive_body_state). Best-effort; never raises, never blocks."""
    try:
        ch = _channel("headlift")
        if ch not in targets or ch not in current:
            return
        if not bool(getattr(config, "SOUND_EFFECTS_HEADLIFT_ENABLED", True)):
            return
        delta = int(targets[ch]) - int(current[ch])
        min_travel = int(getattr(config, "SOUND_EFFECTS_HEADLIFT_MIN_TRAVEL_QUS", 1200))
        if abs(delta) < max(1, min_travel):
            return
        mute = float(getattr(config, "SOUND_EFFECTS_HEADLIFT_STARTUP_MUTE_SECS", 20.0))
        if (time.monotonic() - _headlift_hum_boot_at) < mute:
            return
        from state import State, get_state
        if get_state() not in (State.IDLE, State.ACTIVE):
            return
        from audio import sound_effects
        sound_effects.play("headlift_up" if delta > 0 else "headlift_down")
    except Exception:
        pass


def move_to(
    targets: "dict[int, int]",
    step_us: "int | dict[int, int]" = 40,
    step_delay: float = 0.02,
    start: "dict[int, int] | None" = None,
    *,
    parking: bool = False,
) -> None:
    """Smoothly interpolate specific channels to target positions (quarter-microseconds).

    Head channels 2 and 11 must both be present (and form a pair the linkage can
    take); they are stepped so they arrive together. A poker elbow/wrist target is
    completed from its partner's current command and routed through the staged
    clearance move instead of interpolating. ``parking`` lets the shutdown/sleep park
    stream after the pose latch, like the throttle's own park path.

    The interpolation start point is read from the Maestro (proprioception) so the
    sweep begins wherever the servo actually is. Pass ``start`` to override that read
    with a known pose for specific channels — use it when the caller already knows the
    current position and the proprioception read is unreliable (e.g. the first move
    right after a fresh serial connect, where a failed/garbage read would otherwise
    collapse the sweep into a jump). ``step_us`` may be a single step for every
    channel or a per-channel mapping, which lets mechanically different linkages
    move together without forcing the stickier one to creep.
    """
    if _program_servo_updates_blocked() and not (parking and not _manual_override.is_set()):
        return
    _require_independent_channels(targets)
    targets = _complete_poker_pair({int(ch): int(tgt) for ch, tgt in targets.items()})
    targets = {ch: _clamp(ch, int(tgt)) for ch, tgt in targets.items()}
    _validate_pairs(targets)
    # The poker elbow/wrist never interpolate freely: a straight line between two
    # clear poses can cross the clearance ceiling, so they take the staged move on
    # their own thread while the other channels glide here.
    poker = {ch: targets.pop(ch) for ch in POKER_PAIR if ch in targets}
    if poker:
        move_poker(poker[head_pose.POKER_ELBOW_CH], poker[head_pose.POKER_WRIST_CH],
                   speed=12, acceleration=3, block=False, parking=parking)
    if not targets:
        return
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

    _maybe_headlift_hum(targets, current)

    # The two head servos must ARRIVE together: equal per-step increments would let
    # the shorter travel finish first and park the pair, mid-glide, in a corner no
    # pitch/tilt explains. Scale each head servo's step so both take the same
    # number of steps (a straight line in pulse space between two valid pairs stays
    # well inside the envelope).
    pair_steps: dict[int, float] = {}
    if set(HEAD_PAIR).issubset(targets):
        travel = {ch: abs(targets[ch] - current[ch]) for ch in HEAD_PAIR}
        longest = max(travel.values())
        if longest:
            base_step = max(1, int((step_us.get(HEAD_PAIR[0], 40) if isinstance(step_us, dict) else step_us)))
            steps_needed = max(1, math.ceil(longest / base_step))
            pair_steps = {ch: max(1.0, travel[ch] / steps_needed) for ch in HEAD_PAIR}
    exact = {ch: float(pos) for ch, pos in current.items()}

    done = False
    while not done:
        # Same mid-flight abort as neutral(): once the shutdown latch (or the
        # manual override) engages, an in-progress interpolation must stop
        # streaming rather than finish marching to its stale target.
        if _program_servo_updates_blocked() and not (parking and not _manual_override.is_set()):
            return
        done = True
        moves: dict[int, int] = {}
        for ch, tgt in targets.items():
            cur = current[ch]
            diff = tgt - cur
            if diff == 0:
                continue
            done = False
            if ch in pair_steps:
                channel_step = pair_steps[ch]
                exact[ch] += min(channel_step, abs(tgt - exact[ch])) * (1 if diff > 0 else -1)
                new_pos = int(round(exact[ch]))
                if abs(tgt - exact[ch]) < 0.5:
                    new_pos = tgt
            else:
                channel_step = step_us.get(ch, 40) if isinstance(step_us, dict) else step_us
                channel_step = max(1, int(channel_step))
                step = min(channel_step, abs(diff)) * (1 if diff > 0 else -1)
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
    from intelligence.voice_learning import mic_moved
    mic_moved("servo_shutdown")
    release_voice_enrollment_hold()
    _stop_breathing.set()
    disconnect()
