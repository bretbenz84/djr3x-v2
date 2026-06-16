"""
High-level motion control — the API the rest of the app calls to drive the base.

Wraps the hardware.motion transport with: the Mac-side heartbeat, speed-cap
clamping, the autonomous-motion policy gate (suppressed while INTERACTION_PAUSED
or while a gamepad owns the base), and human-friendly verbs (turn_left,
move_forward, come_here, stop). Units follow docs/motion_protocol.md §4:
+linear = forward, +angle/+deg = LEFT/CCW.

Disabled cleanly when motion isn't connected — every command becomes a no-op that
returns None, so callers never need to special-case "no base attached".
"""

import logging
import math
import threading
import time

import config
from hardware import motion

_log = logging.getLogger(__name__)

_heartbeat_thread: "threading.Thread | None" = None
_stop = threading.Event()


def _get_float(name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _get_int(name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _clampf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def ramp_toward(current: float, target: float, accel_step: float, decel_step: float) -> float:
    """Slew `current` one step toward `target`, capped per call by `accel_step` when
    the speed magnitude is growing and `decel_step` when it is shrinking toward zero.

    Lets manual teleop (the GUI joystick) ramp up gently but brake faster — without the
    abrupt stop that can topple a tall base. Asymmetric by design: pick decel_step >
    accel_step for a quick-but-smooth release. A reversal decelerates through zero
    first (uses decel_step until the sign flips). A non-positive step jumps straight to
    the target. Call repeatedly at a fixed cadence; step = rate * dt."""
    if target == current:
        return target
    speeding_up = abs(target) > abs(current) and (current == 0.0 or (target > 0.0) == (current > 0.0))
    step = accel_step if speeding_up else decel_step
    if step <= 0.0:
        return target
    delta = target - current
    if delta > step:
        return current + step
    if delta < -step:
        return current - step
    return target


# ── Lifecycle ───────────────────────────────────────────────────────────────────

def connect(port: "str | None" = None) -> bool:
    """Open the link, push runtime config to the ESP32, and start the heartbeat.
    Returns True only on a clean handshake."""
    if not motion.connect(port):
        return False
    try:
        _push_config()
    except Exception:
        _log.debug("motion config push failed", exc_info=True)
    _start_heartbeat()
    return True


def disconnect() -> None:
    _stop.set()
    thread = _heartbeat_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1.0)
    # Best-effort: leave the base stopped before dropping the link.
    try:
        if motion.connected():
            motion.send({"cmd": "stop"})
    except Exception:
        pass
    motion.disconnect()


def available() -> bool:
    """True when a base is connected and speaking the protocol. Callers gate
    motion intents on this so behavior is unchanged when no base is attached."""
    return motion.connected()


def _start_heartbeat() -> None:
    global _heartbeat_thread
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return
    _stop.clear()
    _heartbeat_thread = threading.Thread(target=_heartbeat_loop, name="motion-heartbeat", daemon=True)
    _heartbeat_thread.start()


def _heartbeat_loop() -> None:
    """Link manager: ping while connected; auto-reconnect (throttled) after a drop
    so an unplug/replug of the ESP32 heals on its own without restarting Rex."""
    period = max(0.02, _get_int("MOTION_HEARTBEAT_MS", 150) / 1000.0)
    reconnect_interval = max(0.5, _get_float("MOTION_RECONNECT_INTERVAL_SECS", 2.0))
    last_reconnect = 0.0
    while not _stop.is_set():
        if motion.connected():
            motion.ping()
        else:
            now = time.monotonic()
            if now - last_reconnect >= reconnect_interval:
                last_reconnect = now
                try:
                    if motion.reconnect():
                        _log.info("Motion base reconnected")
                        try:
                            _push_config()
                        except Exception:
                            _log.debug("motion config push after reconnect failed", exc_info=True)
                except Exception:
                    _log.debug("motion reconnect attempt failed", exc_info=True)
        _stop.wait(period)


_TUNING_KEYS = (
    ("kp", "MOTION_WHEEL_KP"),
    ("ki", "MOTION_WHEEL_KI"),
    ("kd", "MOTION_WHEEL_KD"),
    ("counts_per_meter", "MOTION_COUNTS_PER_METER"),
    ("track_width_m", "MOTION_TRACK_WIDTH_M"),
)


def _push_config() -> None:
    """Send the Mac's caps/zones/timing to the ESP32 (it clamps to its hard caps).
    max_ang is converted deg/s -> rad/s for the wire. The drive-tuning keys (PID gains
    + calibration geometry) are added ONLY when explicitly set in config.py/.env — when
    None the firmware keeps its calib.h boot defaults, so a (re)connect never clobbers a
    bench-tuned value with a placeholder. See docs/motion_protocol.md §10."""
    cfg = {
        "cmd": "config",
        "max_lin": _get_float("MOTION_MAX_LINEAR_MS", 0.25),
        "max_ang": math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 60.0)),
        "slow_zone_m": _get_float("MOTION_SLOW_ZONE_M", 0.60),
        "stop_zone_m": _get_float("MOTION_STOP_ZONE_M", 0.25),
        "come_stop_at_m": _get_float("MOTION_COME_STOP_AT_M", 0.60),
        "default_turn_deg": _get_float("MOTION_DEFAULT_TURN_DEG", 90.0),
        "default_turn_rate": _get_float("MOTION_DEFAULT_TURN_RATE", 40.0),
        "watchdog_ms": _get_int("MOTION_WATCHDOG_MS", 500),
        "drive_expiry_ms": _get_int("MOTION_DRIVE_EXPIRY_MS", 300),
        "manual_idle_return_secs": _get_int("MOTION_MANUAL_IDLE_RETURN_SECS", 4),
        "manual_autoreturn": bool(getattr(config, "MOTION_MANUAL_AUTORETURN", False)),
    }
    for wire_key, cfg_key in _TUNING_KEYS:
        val = getattr(config, cfg_key, None)
        if val is not None:
            cfg[wire_key] = float(val)
    motion.send(cfg)


# ── Policy gate ─────────────────────────────────────────────────────────────────

def _autonomous_allowed() -> "str | None":
    """Return None if autonomous motion may run, else a reason string."""
    if not motion.connected():
        return "not_connected"
    if bool(getattr(config, "INTERACTION_PAUSED", False)):
        return "interaction_paused"
    if motion.owner() == "manual":
        return "manual_override"
    return None


# ── Commands ────────────────────────────────────────────────────────────────────
# turn/move/come/drive are autonomous (gated). stop/estop/clear always pass while
# connected — you must always be able to halt the base.

def turn(deg: float, rate: "float | None" = None) -> "int | None":
    """Spin in place by `deg` (+ = left/CCW). Closed loop on the ESP32."""
    reason = _autonomous_allowed()
    if reason:
        _log.debug("motion turn suppressed: %s", reason)
        return None
    max_rate = _get_float("MOTION_MAX_ANGULAR_DEG_S", 60.0)
    rate = _get_float("MOTION_DEFAULT_TURN_RATE", 40.0) if rate is None else rate
    rate = _clampf(abs(rate), 1.0, max_rate)
    deg = _clampf(deg, -360.0, 360.0)
    return motion.send({"cmd": "turn", "deg": deg, "rate": rate})


def move(dist: float, speed: "float | None" = None) -> "int | None":
    """Drive straight `dist` metres (+ = forward, - = back). ToF-gated."""
    reason = _autonomous_allowed()
    if reason:
        _log.debug("motion move suppressed: %s", reason)
        return None
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.25)
    speed = max_lin if speed is None else speed
    speed = _clampf(abs(speed), 0.0, max_lin)
    dist = _clampf(dist, -10.0, 10.0)
    return motion.send({"cmd": "move", "dist": dist, "speed": speed})


def come(heading: float = 0.0, stop_at: "float | None" = None) -> "int | None":
    """Turn toward `heading` (deg, + = left), then advance to `stop_at` m from the
    nearest forward obstacle."""
    reason = _autonomous_allowed()
    if reason:
        _log.debug("motion come suppressed: %s", reason)
        return None
    if "come" not in motion.caps():
        _log.debug("motion come unsupported by firmware")
        return None
    stop_at = _get_float("MOTION_COME_STOP_AT_M", 0.60) if stop_at is None else stop_at
    return motion.send({
        "cmd": "come",
        "heading": _clampf(heading, -180.0, 180.0),
        "stop_at": _clampf(stop_at, 0.05, 5.0),
    })


def drive(lin: float, ang: float) -> "int | None":
    """Continuous velocity (m/s, rad/s). Expires after the drive deadman unless
    refreshed — for teleop-style control, call repeatedly."""
    reason = _autonomous_allowed()
    if reason:
        _log.debug("motion drive suppressed: %s", reason)
        return None
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.25)
    max_ang = math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 60.0))
    return motion.send({
        "cmd": "drive",
        "lin": _clampf(lin, -max_lin, max_lin),
        "ang": _clampf(ang, -max_ang, max_ang),
    })


def drive_manual(lin: float, ang: float) -> "int | None":
    """Operator-console teleop (e.g. the GUI joystick). Like drive() but bypasses
    the INTERACTION_PAUSED gate — the operator is explicitly in control. Still
    clamps to caps and requires a connected base; the firmware ignores it anyway
    while a gamepad owns the base."""
    if not motion.connected():
        return None
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.25)
    max_ang = math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 60.0))
    return motion.send({
        "cmd": "drive",
        "lin": _clampf(lin, -max_lin, max_lin),
        "ang": _clampf(ang, -max_ang, max_ang),
    })


def stop() -> "int | None":
    """Controlled stop. Always honored while connected (bypasses the gate)."""
    if not motion.connected():
        return None
    return motion.send({"cmd": "stop"})


def estop() -> "int | None":
    """Hard disable until clear(). Always honored while connected."""
    if not motion.connected():
        return None
    return motion.send({"cmd": "estop"})


def clear() -> "int | None":
    if not motion.connected():
        return None
    return motion.send({"cmd": "clear"})


# ── Voice-friendly verbs (defaults from config) ──────────────────────────────────

def turn_left(deg: "float | None" = None) -> "int | None":
    return turn(_get_float("MOTION_DEFAULT_TURN_DEG", 90.0) if deg is None else abs(deg))


def turn_right(deg: "float | None" = None) -> "int | None":
    return turn(-(_get_float("MOTION_DEFAULT_TURN_DEG", 90.0) if deg is None else abs(deg)))


def move_forward(dist: "float | None" = None) -> "int | None":
    return move(_get_float("MOTION_DEFAULT_MOVE_DIST_M", 0.30) if dist is None else abs(dist))


def move_back(dist: "float | None" = None) -> "int | None":
    return move(-(_get_float("MOTION_DEFAULT_MOVE_DIST_M", 0.30) if dist is None else abs(dist)))


def come_here() -> "int | None":
    return come(0.0)


# ── Status (for GUI / telemetry / logging) ───────────────────────────────────────

def status() -> "dict | None":
    return motion.telemetry()


def is_moving() -> bool:
    return motion.state() == "moving"
