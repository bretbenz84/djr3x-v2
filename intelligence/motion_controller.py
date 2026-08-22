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
import state as state_module
from hardware import motion
from state import State

_log = logging.getLogger(__name__)

_heartbeat_thread: "threading.Thread | None" = None
_stop = threading.Event()

# Voice "arc": a brief simultaneous curve (forward/back + left/right). The heartbeat
# thread re-sends the `drive` setpoint (faster than the firmware deadman) until _arc_until,
# then stops — so it auto-stops and needs no extra thread. Guarded by _arc_lock.
_arc_lock = threading.Lock()
_arc_active = False
_arc_lin = 0.0
_arc_ang = 0.0
_arc_until = 0.0

# Calibrated-compass post-turn verifier. Firmware already closes turns on IMU yaw;
# this is a slower absolute-heading check after motor current settles. Epochs prevent
# a delayed correction from superseding a newer human/autonomy command.
_turn_verify_lock = threading.Lock()
_turn_verify_epoch = 0
_pending_turn_verify: dict[int, dict] = {}
_turn_verify_unavailable_logged = False


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
    global _charging_last_true_at
    _charging_last_true_at = 0.0        # fresh base — no stale sticky-charging memory
    try:
        _push_config()
    except Exception:
        _log.debug("motion config push failed", exc_info=True)
    # Forward gamepad action-button events (the 8BitDo Pro 2 buttons motion doesn't
    # use) to the soundboard / animation dispatch — Rex reacts to a button press.
    try:
        motion.set_callbacks(on_event=_on_motion_event, on_done=_on_motion_done)
    except Exception:
        _log.debug("motion event callback wiring failed", exc_info=True)
    _start_heartbeat()
    return True


# ── Sound-effect accents for real drive motion ────────────────────────────────────

_last_come_seq: "int | None" = None
# Outcome of the most recent `come`, so the come-here errand can tell "I arrived"
# from "something stepped in front of me". ToF cannot make that call — a dog
# standing 0.5 m away looks exactly like having reached someone.
_last_come_result: "str | None" = None


def last_come_result() -> "tuple[int | None, str | None]":
    """(seq, result) of the latest `come`; result is None while it is still running."""
    return _last_come_seq, _last_come_result


# A VOICE-COMMANDED move ships a spoken confirmation ("Spinning around.") whose
# cached audio reaches the speaker ~3 ms after queueing, so the drive sound lost the
# race for the output gate and was dropped on nearly every command — while
# autonomous moves, which say nothing, kept theirs (owner 2026-07-24: "when you
# command him to move, he does not play the sound effects"). The interaction layer
# stamps this before issuing a commanded move; _fx then plays that clip in OVERLAY
# mode (own output stream) so it rides under the confirmation instead of losing to it.
_user_commanded_motion_at = 0.0


def note_user_commanded_motion() -> None:
    """Mark the motion about to be issued as coming from an explicit voice command."""
    global _user_commanded_motion_at
    _user_commanded_motion_at = time.monotonic()


def _user_commanded_fx() -> bool:
    if _user_commanded_motion_at <= 0.0:
        return False
    window = float(getattr(config, "MOTION_COMMANDED_FX_WINDOW_SECS", 20.0))
    return (time.monotonic() - _user_commanded_motion_at) <= window


def _fx_gain() -> float:
    """Volume scale for drive accents. Autonomous motion is FREQUENT now (idle
    wander, radar orient, edge-in, object step...), so its motor sounds duck to
    a fraction of a commanded move's level (owner 2026-08-19: about half) —
    present, but ambient. A voice-commanded move keeps full volume: there the
    sound is confirmation, not texture."""
    if _user_commanded_fx():
        return 1.0
    try:
        return max(0.0, min(1.0, float(getattr(
            config, "MOTION_AUTONOMOUS_FX_GAIN", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def _fx(key: str) -> None:
    """Fire a drive sound effect (audio/sound_effects). Best-effort, never raises,
    never blocks — the effect layer owns cooldowns/enable flags/preemption."""
    try:
        from audio import sound_effects
        sound_effects.play(key, overlay=_user_commanded_fx(), gain=_fx_gain())
    except Exception:
        pass


# The drive clips are ~4 s but a real leg is longer — 12 feet at the exploring speed
# is ~9 s — so the whir used to stop while the wheels were still turning (owner
# 2026-07-24). For a finite move/turn the effect now LOOPS for the duration of the
# command and is cut the moment the base reports done.
_drive_loop_lock = threading.Lock()
_drive_loop: dict = {}          # seq -> LoopHandle


def _fx_drive_loop_start(key: str, seq: "int | None") -> None:
    if seq is None or int(seq) <= 0:
        return
    if not bool(getattr(config, "SOUND_EFFECTS_DRIVE_LOOP_ENABLED", True)):
        _fx(key)                # one-shot fallback keeps the old behavior
        return
    try:
        from audio import sound_effects
        handle = sound_effects.start_loop(
            key,
            mode="overlay" if _user_commanded_fx() else "gated",
            gap_secs=float(getattr(config, "SOUND_EFFECTS_DRIVE_LOOP_GAP_SECS", 0.1)),
            # Safety cap: a dropped `done` frame must never leave the whir droning.
            max_secs=float(getattr(config, "SOUND_EFFECTS_DRIVE_LOOP_MAX_SECS", 30.0)),
            gain=_fx_gain(),        # autonomous moves whir at half volume
        )
    except Exception:
        return
    if handle is None:
        return
    with _drive_loop_lock:
        _drive_loop[int(seq)] = handle


def _fx_drive_loop_stop(seq: "int | None") -> None:
    """Stop the whir for a finished command (and reap any stale handles)."""
    with _drive_loop_lock:
        handle = _drive_loop.pop(int(seq), None) if seq is not None else None
        stale = [s for s, h in _drive_loop.items() if not h.running]
        for s in stale:
            _drive_loop.pop(s, None)
    if handle is None:
        return
    try:
        from audio import sound_effects
        sound_effects.stop_loop(handle, join_timeout=0.2)
    except Exception:
        pass


def _fx_drive_loop_stop_all() -> None:
    with _drive_loop_lock:
        handles = list(_drive_loop.values())
        _drive_loop.clear()
    try:
        from audio import sound_effects
        for h in handles:
            sound_effects.stop_loop(h, join_timeout=0.2)
    except Exception:
        pass


# User-commanded seqs that should SPEAK when the firmware cuts them on an obstacle.
# Silence was the field failure (2026-07-23): "move forward 5 feet" stopped at ~2 ft
# on a zone block and Rex said nothing, reading as "he ignores my commands". Only
# explicit voice-command moves register here — autonomous/exploration legs stay quiet.
_announce_blocked_lock = threading.Lock()
_announce_blocked_seqs: dict[int, float] = {}   # seq -> registered-at (pruned by age)
_announce_blocked_last_spoken = 0.0


def announce_if_blocked(seq: "int | None") -> None:
    """Register a user-commanded motion seq: if it later completes as 'blocked',
    Rex says a short line so the human knows the move was cut, not ignored."""
    if seq is None or int(seq) <= 0:
        return
    now = time.monotonic()
    with _announce_blocked_lock:
        _announce_blocked_seqs[int(seq)] = now
        # Prune anything stale (done frame missed / superseded).
        for k in [k for k, t in _announce_blocked_seqs.items() if now - t > 120.0]:
            _announce_blocked_seqs.pop(k, None)


def _maybe_announce_blocked(msg: dict) -> None:
    global _announce_blocked_last_spoken
    try:
        seq = int(msg.get("seq"))
    except (TypeError, ValueError):
        return
    with _announce_blocked_lock:
        if _announce_blocked_seqs.pop(seq, None) is None:
            return
        now = time.monotonic()
        cooldown = float(getattr(config, "MOTION_BLOCKED_ANNOUNCE_COOLDOWN_SECS", 10.0))
        if (now - _announce_blocked_last_spoken) < cooldown:
            return
        _announce_blocked_last_spoken = now
    try:
        from audio import speech_queue
        speech_queue.enqueue(
            str(getattr(config, "MOTION_BLOCKED_ANNOUNCE_LINE",
                        "Something's in my way — that's as far as I get.")),
            emotion="neutral",
            priority=1,
            tag="motion_blocked",
        )
    except Exception as exc:
        _log.debug("blocked announce failed: %s", exc)


# Swing escape: a turn whose sweep is blocked first steps forward (if the front
# is open), then re-runs on that move's done. {"seq", "deg", "rate"} or None.
_swing_escape: "dict | None" = None
_swing_lock = threading.Lock()   # the move's done can land before send() returns


def _front_room_m() -> "float | None":
    tele = motion.telemetry()
    tof = tele.get("tof_mm") if isinstance(tele, dict) else None
    if not isinstance(tof, dict):
        return None
    best = None
    for k in ("fl", "fr"):
        try:
            mm = float(tof.get(k))
        except (TypeError, ValueError):
            continue
        if mm >= 0 and (best is None or mm < best):
            best = mm
    return None if best is None else best / 1000.0


def _try_swing_escape(deg: float, rate: float) -> "int | None":
    """The swing is blocked: earn the room by stepping forward, then turn on
    arrival. Returns the move's seq, or None when there is no room ahead."""
    global _swing_escape
    if not bool(getattr(config, "MOTION_SWING_ESCAPE_ENABLED", True)):
        return None
    room = _front_room_m()
    need = _get_float("MOTION_SWING_ESCAPE_CLEARANCE_M", 1.0)
    if room is None or room < need:
        _log.info("[swing] no escape forward (front %.2f m < %.2f m)",
                  room if room is not None else -1.0, need)
        return None
    step = _get_float("MOTION_SWING_ESCAPE_STEP_M", 0.60)
    with _swing_lock:
        seq = move(step)
        if seq is None:
            return None
        _swing_escape = {"seq": seq, "deg": deg, "rate": rate}
    _log.info("[swing] %+.0f° turn blocked behind — stepping %.2f m forward first (seq %d)",
              deg, step, seq)
    return seq


def _finish_swing_escape(msg: dict) -> None:
    global _swing_escape
    with _swing_lock:
        pend = _swing_escape
        if pend is None or msg.get("seq") != pend["seq"]:
            return
        _swing_escape = None
    if str(msg.get("result") or "") not in ("completed", "blocked"):
        _log.info("[swing] escape step %s — turn dropped", msg.get("result"))
        return
    # Re-check from the new spot; a second step is never chained. The done
    # beats the next telemetry frame, so let the ring report the new spot first.
    def _go():
        time.sleep(_get_float("MOTION_SWING_ESCAPE_SETTLE_SECS", 0.3))
        turn(pend["deg"], pend["rate"], _escaped=True)
    threading.Thread(target=_go, daemon=True, name="swing-escape-turn").start()


def _on_motion_done(msg: dict) -> None:
    """Reader-thread callback for command completions: the come-here arrival chirp
    and the "whoa, blocked" accent when the base stops a command on an obstacle."""
    try:
        result = str((msg or {}).get("result") or "")
        _finish_swing_escape(msg or {})
        # The wheels have stopped — cut the looping whir first, so the arrival /
        # blocked accent lands in silence instead of on top of a drive sound.
        try:
            _fx_drive_loop_stop(msg.get("seq") if msg else None)
        except Exception:
            pass
        global _last_come_result
        if _last_come_seq is not None and msg.get("seq") == _last_come_seq:
            _last_come_result = result or None
        if result == "blocked":
            _fx("slow_down")
            _maybe_announce_blocked(msg or {})
        elif result == "completed" and _last_come_seq is not None \
                and msg.get("seq") == _last_come_seq:
            _fx("arrived")
        _handle_turn_verification_done(msg or {})
    except Exception:
        pass


def _invalidate_turn_verification() -> int:
    global _turn_verify_epoch
    with _turn_verify_lock:
        _turn_verify_epoch += 1
        _pending_turn_verify.clear()
        return _turn_verify_epoch


def _calibrated_compass_yaw() -> "float | None":
    if not bool(getattr(config, "MOTION_COMPASS_TURN_VERIFY_ENABLED", True)):
        return None
    try:
        from hardware import compass
        return compass.get_service_yaw(require_calibrated=True)
    except Exception:
        return None


def _remember_turn_verification(
    seq: int,
    *,
    desired_deg: float,
    rate: float,
    start_yaw: "float | None",
    epoch: int,
    attempt: int,
) -> None:
    if start_yaw is None:
        # Silent for a whole session of overshooting turns (field 2026-08-11:
        # every 90° turn landed ~115° and NOTHING in the log said the compass
        # check never armed). Say so once, loudly, so a dead mag / missing
        # calibration is visible in the log instead of inferred from absence.
        global _turn_verify_unavailable_logged
        if (not _turn_verify_unavailable_logged
                and bool(getattr(config, "MOTION_COMPASS_TURN_VERIFY_ENABLED", True))):
            _turn_verify_unavailable_logged = True
            _log.warning(
                "[motion] compass turn verification unavailable (no calibrated "
                "fused yaw) — finite turns run open-loop this session; check mag "
                "telemetry health and compass calibration")
        return
    if abs(desired_deg) > 170.0:
        return  # shortest-angle comparison is ambiguous at/above a half turn
    with _turn_verify_lock:
        _pending_turn_verify[int(seq)] = {
            "desired_deg": float(desired_deg),
            "rate": float(rate),
            "start_yaw": float(start_yaw),
            "epoch": int(epoch),
            "attempt": int(attempt),
        }


def _handle_turn_verification_done(msg: dict) -> None:
    try:
        seq = int(msg.get("seq"))
    except (TypeError, ValueError):
        return
    with _turn_verify_lock:
        record = _pending_turn_verify.pop(seq, None)
    if record is None or str(msg.get("result") or "") != "completed":
        return
    threading.Thread(
        target=_verify_completed_turn,
        args=(record,),
        daemon=True,
        name="compass-turn-verify",
    ).start()


def _verify_completed_turn(record: dict) -> None:
    """After current settles, compare physical turn with calibrated fused heading."""
    settle = _get_float("MOTION_COMPASS_TURN_SETTLE_SECS", 0.8)
    if _stop.wait(max(0.0, settle)):
        return
    with _turn_verify_lock:
        if int(record["epoch"]) != _turn_verify_epoch:
            return
    if not motion.connected() or motion.owner() == "manual" or charging():
        return
    end_yaw = _calibrated_compass_yaw()
    if end_yaw is None:
        return
    try:
        from hardware.compass import ang_diff
        actual = ang_diff(float(end_yaw), float(record["start_yaw"]))
    except Exception:
        return
    desired = float(record["desired_deg"])
    error = desired - actual
    while error > 180.0:
        error -= 360.0
    while error <= -180.0:
        error += 360.0
    tolerance = _get_float("MOTION_COMPASS_TURN_TOLERANCE_DEG", 4.0)
    if abs(error) <= tolerance:
        _log.info(
            "[motion] compass verified turn: requested=%+.1f actual=%+.1f error=%+.1f deg",
            desired, actual, error,
        )
        return
    attempt = int(record.get("attempt", 0))
    max_attempts = _get_int("MOTION_COMPASS_TURN_MAX_CORRECTIONS", 1)
    max_correction = _get_float("MOTION_COMPASS_TURN_MAX_CORRECTION_DEG", 30.0)
    if attempt >= max_attempts or abs(error) > max_correction:
        _log.warning(
            "[motion] compass turn mismatch not auto-corrected: requested=%+.1f "
            "actual=%+.1f error=%+.1f deg attempt=%d",
            desired, actual, error, attempt,
        )
        return
    with _turn_verify_lock:
        if int(record["epoch"]) != _turn_verify_epoch:
            return
    _log.warning(
        "[motion] compass correcting turn: requested=%+.1f actual=%+.1f error=%+.1f deg",
        desired, actual, error,
    )
    turn(error, rate=min(abs(float(record["rate"])), 25.0), _verify_attempt=attempt + 1)


# ── Gamepad action buttons → sound clips / servo animations ──────────────────────
# The pad pairs to the ESP32, so its non-drive buttons arrive as `event:"button"`
# telemetry (firmware/djr3x_motion/gamepad.cpp). Map each to a clip and/or animation
# via config.MOTION_GAMEPAD_BUTTON_ACTIONS — data-driven, no code change to remap.

def _on_motion_event(msg: dict) -> None:
    """Reader-thread callback for firmware `event` messages. `button` and `gamepad`
    events are handled here; everything else (estop/comms/zone_block/...) is consumed
    elsewhere."""
    try:
        if not isinstance(msg, dict):
            return
        if msg.get("event") == "gamepad":
            # Firmware pad connect/disconnect events — surfaced at INFO so pairing
            # problems are diagnosable from djr3x.log (added 2026-07-21).
            _log.info("[gamepad] pad %s", msg.get("state"))
            return
        if msg.get("event") != "button":
            return
        btn = str(msg.get("btn") or "").strip().lower()
        if not btn:
            return
        actions = getattr(config, "MOTION_GAMEPAD_BUTTON_ACTIONS", {}) or {}
        action = actions.get(btn)
        if not isinstance(action, dict) or not action:
            _log.debug("[gamepad] unmapped button: %r", btn)
            return
        _dispatch_button_action(btn, action)
    except Exception:
        _log.debug("[gamepad] button event handler failed", exc_info=True)


def _dispatch_button_action(btn: str, action: dict) -> None:
    """Run a button's mapped action: a servo animation and/or a sound clip. Each leg
    is best-effort and isolated so a missing clip never blocks the animation."""
    anim = str(action.get("animation") or "").strip()
    clip = str(action.get("clip") or "").strip()
    _log.info(
        "[gamepad] button %s -> %s", btn,
        {k: v for k, v in (("animation", anim), ("clip", clip)) if v},
    )
    if anim:
        try:
            from sequences import animations
            animations.play_body_beat(anim)
        except Exception:
            _log.debug("[gamepad] animation %r failed", anim, exc_info=True)
    if clip:
        try:
            from audio import soundboard
            soundboard.play(clip)
        except Exception:
            _log.debug("[gamepad] clip %r failed", clip, exc_info=True)


def disconnect() -> None:
    _invalidate_turn_verification()
    _cancel_arc()          # kill any in-flight arc before the heartbeat thread stops
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
            _heartbeat_tick()
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


def _heartbeat_tick() -> None:
    """One connected-loop step: refresh an in-flight arc's `drive` (then stop it when it
    expires or motion is no longer allowed), otherwise just `ping`. Both keep the firmware
    watchdog fed (any valid Mac line resets it). The arc decision + send run UNDER _arc_lock
    so a concurrent stop()/_cancel_arc can't be overridden by a stale `drive` (the small
    JSON send is fast; lock order is always _arc_lock -> motion locks, so no deadlock)."""
    global _arc_active
    # Obstacle sensing is checked every tick, not just when something asks to move:
    # this is what puts a matrix dying at 3am in the log at 3am, and what cuts a leg
    # that is already running closed-loop on the ESP32 (the policy gate can only
    # refuse NEW commands). Deliberately OUTSIDE _arc_lock — stop() reaches for it
    # via _cancel_arc(), and it also ends any in-flight arc for us.
    if _tof_should_cut_inflight():
        stop()
        return
    with _arc_lock:
        if _arc_active:
            if time.monotonic() < _arc_until and _autonomous_allowed() is None:
                motion.send({"cmd": "drive", "lin": _arc_lin, "ang": _arc_ang})  # < deadman
            else:                            # expired, or interrupted (paused / gamepad took over)
                _arc_active = False
                motion.send({"cmd": "stop"})
            return
    motion.ping()


def _cancel_swing_escape() -> None:
    global _swing_escape
    with _swing_lock:
        _swing_escape = None


def _cancel_arc() -> None:
    """Drop any in-flight arc (a new command / stop supersedes it). Does NOT send a
    stop itself — the caller's own command (or stop) does that."""
    global _arc_active
    with _arc_lock:
        _arc_active = False


_TUNING_KEYS = (
    ("kp", "MOTION_WHEEL_KP"),
    ("ki", "MOTION_WHEEL_KI"),
    ("kd", "MOTION_WHEEL_KD"),
    ("kff", "MOTION_WHEEL_KFF"),
    ("min_duty", "MOTION_WHEEL_MIN_DUTY"),
    ("breakaway_duty", "MOTION_WHEEL_BREAKAWAY_DUTY"),
    ("accel_lin", "MOTION_ACCEL_LINEAR_MS2"),
    ("accel_ang", "MOTION_ACCEL_ANGULAR_RAD_S2"),
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
        "max_lin": _get_float("MOTION_MAX_LINEAR_MS", 0.40),
        "max_ang": math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 85.0)),
        "slow_zone_m": _get_float("MOTION_SLOW_ZONE_M", 0.60),
        "stop_zone_m": _get_float("MOTION_STOP_ZONE_M", 0.30),
        "come_stop_at_m": _get_float("MOTION_COME_STOP_AT_M", 0.60),
        "default_turn_deg": _get_float("MOTION_DEFAULT_TURN_DEG", 90.0),
        "default_turn_rate": _get_float("MOTION_DEFAULT_TURN_RATE", 75.0),
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


# ── Obstacle-sensing health gate ────────────────────────────────────────────────
# Field 2026-08-07..08-11: the front 8x8 matrix ToF died electrically and Rex spent
# four days driving into walls and low objects. No code was broken — safety.cpp
# fails OPEN on a -1 reading by documented choice, the radial ring stayed alive, so
# nothing downstream ever saw a reason to stop. But the matrix is what covers the
# near floor and anything short enough to duck under the ring's ±22.5° beams, so
# "the ring is answering" was never the same as "he can see where he's going".
#
# Autonomy now requires the obstacle sensing to be PRESENT, not merely un-alarmed.
# The asymmetry is deliberate: losing it blocks on the very next check, regaining
# it has to hold for MOTION_TOF_RECOVERY_SECS first, so a sensor flapping in and
# out cannot ratchet him across the room one frame at a time.
#
# Never gated: stop/estop/clear (you must always be able to halt him) and operator
# teleop — drive_manual and the gamepad — because a human at the controls IS the
# obstacle sensing. This blocks autonomy, not the robot.

_tof_lock = threading.Lock()
_tof_healthy_since = 0.0
_tof_fault_since = 0.0
_tof_warned = False
_tof_block_reason: "str | None" = "tof_startup"
_tof_cut_for_reason: "str | None" = None
_tof_announced_at = 0.0


def _tof_sensing_fault() -> "str | None":
    """Raw read: is the obstacle sensing THERE? None = present and fresh."""
    if not bool(getattr(config, "MOTION_REQUIRE_TOF_FOR_AUTONOMY", True)):
        return None
    if bool(getattr(config, "MOTION_TOF_MATRIX_REQUIRED", True)):
        # No tofmx frame inside the staleness window means absent, no-ACK,
        # I2C-wedged, or in a read-error streak. All four are "blind to the floor".
        if motion.tof_matrix() is None:
            return "tof_matrix_down"
    alive, total = motion.radial_tof_alive()
    if total and alive == 0:
        return "tof_ring_down"
    return None


def tof_block_reason() -> "str | None":
    """The gate's answer, with recovery hysteresis. None = sensing is trusted.

    Also owns the transition logging, and the heartbeat calls it every tick — so
    a matrix that dies at 3am is in the log then, not whenever something next
    tries to move.
    """
    global _tof_healthy_since, _tof_block_reason, _tof_fault_since, _tof_warned
    fault = _tof_sensing_fault()
    now = time.monotonic()
    settle = max(0.0, float(getattr(config, "MOTION_TOF_RECOVERY_SECS", 3.0)))
    with _tof_lock:
        if fault is not None:
            _tof_healthy_since = 0.0
            if _tof_fault_since == 0.0 or fault != _tof_block_reason:
                _tof_fault_since = now
                _tof_warned = False
            _tof_block_reason = fault
            # Block from the very first observation, but don't cry wolf about it:
            # the matrix takes ~6 s to initialise after a base reboot, and on every
            # normal startup the heartbeat's first tick beats the first tofmx frame.
            # A WARNING on each launch would train everyone to ignore the one that
            # matters. Blocking is immediate; the alarm waits out the same window.
            if not _tof_warned and (now - _tof_fault_since) >= settle:
                _tof_warned = True
                _log.warning(
                    "Obstacle sensing lost (%s) for %.1fs — autonomous motion "
                    "blocked. stop/estop and operator teleop still work.",
                    fault, now - _tof_fault_since,
                )
            return fault
        _tof_fault_since = 0.0
        if _tof_healthy_since == 0.0:
            _tof_healthy_since = now
        healthy_for = now - _tof_healthy_since
        if _tof_block_reason is not None and healthy_for < settle:
            return _tof_block_reason
        if _tof_block_reason is not None:
            _log.info(
                "Obstacle sensing healthy for %.1fs — autonomous motion enabled "
                "(previous state: %s).", healthy_for, _tof_block_reason,
            )
            _tof_block_reason = None
            _tof_warned = False
        return None


def _tof_should_cut_inflight() -> bool:
    """True once per outage when sensing dies with an autonomous leg still running.

    The gate above only refuses NEW commands; a move/turn/come already accepted is
    closed-loop on the ESP32 and would drive the rest of it blind.
    """
    global _tof_cut_for_reason
    reason = tof_block_reason()
    if reason is None:
        _tof_cut_for_reason = None
        return False
    if _tof_cut_for_reason == reason:
        return False                                  # already cut for this outage
    if motion.owner() != "auto" or motion.state() != "moving":
        return False
    _tof_cut_for_reason = reason
    _log.warning("Obstacle sensing lost mid-move (%s) — stopping the base.", reason)
    return True


def _suppressed(verb: str, reason: str) -> None:
    """Log a refused autonomous command, and when a human asked for it out loud,
    let Rex say WHY. A silent no-op reads as "he ignores my commands" (field
    2026-07-23 — the same lesson that produced announce_if_blocked)."""
    global _tof_announced_at
    _log.debug("motion %s suppressed: %s", verb, reason)
    if not reason.startswith(("tof_", "swing_")) or not _user_commanded_fx():
        return                       # autonomous legs stay quiet; they retry constantly
    now = time.monotonic()
    cooldown = float(getattr(config, "MOTION_TOF_BLOCKED_ANNOUNCE_COOLDOWN_SECS", 30.0))
    with _tof_lock:
        if (now - _tof_announced_at) < cooldown:
            return
        _tof_announced_at = now
    if reason.startswith("swing_"):
        line = str(getattr(config, "MOTION_SWING_BLOCKED_LINE",
                           "Can't swing that way — I'd clip something behind me."))
        tag = "motion_swing_blocked"
    else:
        line = str(getattr(config, "MOTION_TOF_BLOCKED_LINE",
                           "My depth sensor is down, sweetheart. I don't drive blind."))
        tag = "motion_tof_blocked"
    try:
        from audio import speech_queue
        speech_queue.enqueue(line, emotion="neutral", priority=1, tag=tag)
    except Exception as exc:
        _log.debug("tof-blocked announce failed: %s", exc)


# ── Policy gate ─────────────────────────────────────────────────────────────────

def _autonomous_allowed() -> "str | None":
    """Return None if autonomous motion may run, else a reason string."""
    if not motion.connected():
        return "not_connected"
    if state_module.get_state() in (State.SLEEP, State.SHUTDOWN):
        return "robot_asleep"
    if charging():
        return "charging"
    if bool(getattr(config, "INTERACTION_PAUSED", False)):
        return "interaction_paused"
    if motion.owner() == "manual":
        return "manual_override"
    # Last, so the everyday reasons above keep their clearer message and Rex
    # doesn't announce a dead sensor while he is merely asleep or on the cord.
    return tof_block_reason()


def _swing_gate(verb: str, deg: float) -> "tuple[float, str | None]":
    """Swing check for an autonomous spin of `deg` (+ = left). Returns the angle
    to actually send (possibly shrunk) and a refusal reason, or None to go.

    The base pivots about its rear axle and carries arms that reach well past
    the ring, so a spin near an obstacle — especially one BEHIND him — sweeps
    the body into it while the firmware reflex, which only gates linear travel,
    sees nothing wrong (the bookshelf hand-loss incidents, 2026-08). See
    intelligence/motion_swing.py."""
    from intelligence import motion_swing
    tele = motion.telemetry()
    tof = tele.get("tof_mm") if isinstance(tele, dict) else None
    send_deg, reason = motion_swing.check_turn(deg, tof)
    if reason:
        _suppressed(verb, reason)
    return send_deg, reason


# ── Commands ────────────────────────────────────────────────────────────────────
# turn/move/come/drive are autonomous (gated). stop/estop/clear always pass while
# connected — you must always be able to halt the base.

def turn(
    deg: float,
    rate: "float | None" = None,
    *,
    _verify_attempt: int = 0,
    _escaped: bool = False,
) -> "int | None":
    """Spin in place by `deg` (+ = left/CCW). Closed loop on the ESP32.

    If the swing would sweep the body/arms into something (usually behind him),
    he first steps forward to earn the room and turns on arrival; `_escaped`
    marks that second attempt so it can't step again."""
    reason = _autonomous_allowed()
    if reason:
        _suppressed("turn", reason)
        return None
    max_rate = _get_float("MOTION_MAX_ANGULAR_DEG_S", 85.0)
    rate = _get_float("MOTION_DEFAULT_TURN_RATE", 75.0) if rate is None else rate
    rate = _clampf(abs(rate), 1.0, max_rate)
    deg = _clampf(deg, -360.0, 360.0)
    from intelligence import motion_swing
    tele = motion.telemetry()
    send_deg, reason = motion_swing.check_turn(
        deg, tele.get("tof_mm") if isinstance(tele, dict) else None)
    spin_floor = _get_float("MOTION_SPIN_ALL_OR_NOTHING_DEG", 270.0)
    if (not reason and spin_floor > 0.0 and abs(deg) >= spin_floor
            and abs(send_deg) < abs(deg) - 1.0):
        # A near-full spin is ALL OR NOTHING. check_turn's shrink is the right
        # degradation for a 90 — turning most of the way toward what you asked for
        # is still useful — and the wrong one for "do a 360": the point of a spin is
        # ending where you started, so a 360 quietly delivered as 147 reports
        # `completed` at a heading nobody asked for, and anything sequenced behind it
        # (a route's next leg) then drives off that wrong heading. Fall into the same
        # arm a hard block takes: step forward to earn the room if the front is open
        # (694a975), else refuse and let the caller say so.
        _log.info("[motion] %+.0f° spin refused rather than shrunk to %+.0f°",
                  deg, send_deg)
        reason = "swing_blocked"
    if reason:
        if not _escaped:
            seq = _try_swing_escape(deg, rate)
            if seq is not None:
                return seq
        _suppressed("turn", reason)
        return None
    deg = send_deg
    start_yaw = _calibrated_compass_yaw()
    epoch = _invalidate_turn_verification()
    _cancel_arc()
    seq = motion.send({"cmd": "turn", "deg": deg, "rate": rate})
    if seq is not None:
        _remember_turn_verification(
            seq,
            desired_deg=deg,
            rate=rate,
            start_yaw=start_yaw,
            epoch=epoch,
            attempt=_verify_attempt,
        )
        _fx_drive_loop_start("motion_turn", seq)
    return seq


def move(dist: float, speed: "float | None" = None) -> "int | None":
    """Drive straight `dist` metres (+ = forward, - = back). ToF-gated."""
    reason = _autonomous_allowed()
    if reason:
        _suppressed("move", reason)
        return None
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.40)
    speed = max_lin if speed is None else speed
    speed = _clampf(abs(speed), 0.0, max_lin)
    dist = _clampf(dist, -10.0, 10.0)
    _invalidate_turn_verification()
    _cancel_arc()
    seq = motion.send({"cmd": "move", "dist": dist, "speed": speed})
    if seq is not None:
        _fx_drive_loop_start("motion_move", seq)
    return seq


def compass_turn_delta(target_deg: float) -> "float | None":
    """Signed RELATIVE turn (+ = left/CCW, the turn() convention) that would face the
    chassis at true compass heading ``target_deg``. None when no trustworthy heading
    exists (compass disabled / uncalibrated / telemetry down). Compass headings grow
    CLOCKWISE (N=0, E=90) while turn() is CCW-positive, hence the sign flip."""
    try:
        from hardware import compass
        yaw = compass.get_service_yaw(require_calibrated=True)
    except Exception:
        yaw = None
    if yaw is None:
        return None
    cw = ((float(target_deg) - float(yaw) + 180.0) % 360.0) - 180.0   # clockwise amount
    return -cw


def turn_to_compass(target_deg: float) -> "int | None":
    """Rotate to face true compass heading ``target_deg`` ("turn north").

    Returns the command seq, 0 when already facing it within
    COMPASS_TURN_DEADBAND_DEG (nothing sent — treat as success), or None when the
    heading is unavailable or the turn was suppressed/refused."""
    rel = compass_turn_delta(target_deg)
    if rel is None:
        _log.info("motion compass turn unavailable (no calibrated heading)")
        return None
    deadband = _get_float("COMPASS_TURN_DEADBAND_DEG", 6.0)
    if abs(rel) <= deadband:
        _log.info("motion compass turn: already facing %.0f° (off by %.1f°)", target_deg, rel)
        return 0
    _log.info("motion compass turn: target=%.0f° -> relative %+.1f°", target_deg, rel)
    return turn(rel)


def come(heading: float = 0.0, stop_at: "float | None" = None,
         speed: "float | None" = None) -> "int | None":
    """Turn toward `heading` (deg, + = left), then advance to `stop_at` m from the
    nearest forward obstacle. ``speed`` (m/s) sets the advance pace — None keeps
    the firmware default (max_lin); older firmware ignores the field entirely."""
    reason = _autonomous_allowed()
    if reason:
        _suppressed("come", reason)
        return None
    if "come" not in motion.caps():
        _log.debug("motion come unsupported by firmware")
        return None
    heading = _clampf(heading, -180.0, 180.0)
    if heading:
        # The firmware spins to `heading` before advancing — that spin sweeps the
        # body just like turn(). Don't shrink it (he'd walk off at the wrong
        # bearing); refuse the whole come if the swing is blocked.
        _, reason = _swing_gate("come", heading)
        if reason:
            return None
    stop_at = _get_float("MOTION_COME_STOP_AT_M", 0.60) if stop_at is None else stop_at
    _invalidate_turn_verification()
    _cancel_arc()
    payload = {
        "cmd": "come",
        "heading": heading,
        "stop_at": _clampf(stop_at, 0.05, 5.0),
    }
    if speed is not None:
        payload["speed"] = _clampf(abs(speed), 0.0,
                                   _get_float("MOTION_MAX_LINEAR_MS", 0.40))
    seq = motion.send(payload)
    if seq is not None:
        global _last_come_seq, _last_come_result
        _last_come_seq = seq
        _last_come_result = None      # in flight
        _fx_drive_loop_start("motion_move", seq)
    return seq


def arc(lin: float, ang: float, duration_s: "float | None" = None) -> "int | None":
    """Drive a brief simultaneous curve (m/s, rad/s) for `duration_s`, then auto-stop.
    Fire-and-forget: the heartbeat thread refreshes the `drive` setpoint and stops it when
    it expires (so it can't run away). Used for voice "move forward and to your right".
    Gated like the other autonomous commands."""
    reason = _autonomous_allowed()
    if reason:
        _suppressed("arc", reason)
        return None
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.40)
    max_ang = math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 85.0))
    dur = _get_float("MOTION_ARC_DURATION_SECS", 1.6) if duration_s is None else float(duration_s)
    ang = _clampf(ang, -max_ang, max_ang)
    yaw_deg = math.degrees(ang) * max(0.2, dur)
    if yaw_deg:
        allowed, reason = _swing_gate("arc", yaw_deg)
        if reason:
            return None
        if abs(allowed) < abs(yaw_deg):      # keep the curve, shorten the swing
            ang *= allowed / yaw_deg
    _invalidate_turn_verification()
    global _arc_active, _arc_lin, _arc_ang, _arc_until
    with _arc_lock:
        _arc_lin = _clampf(lin, -max_lin, max_lin)
        _arc_ang = _clampf(ang, -max_ang, max_ang)
        _arc_until = time.monotonic() + max(0.2, dur)
        _arc_active = True
    _fx("motion_move")
    return 1   # "issued" — the heartbeat drives + auto-stops it; no per-tick seq


def drive(lin: float, ang: float) -> "int | None":
    """Continuous velocity (m/s, rad/s). Expires after the drive deadman unless
    refreshed — for teleop-style control, call repeatedly."""
    reason = _autonomous_allowed()
    if reason:
        _suppressed("drive", reason)
        return None
    _invalidate_turn_verification()
    _cancel_arc()
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.40)
    max_ang = math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 85.0))
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
    if charging():
        _log.debug("manual drive suppressed: charging")
        return None
    _invalidate_turn_verification()
    _cancel_arc()
    max_lin = _get_float("MOTION_MAX_LINEAR_MS", 0.40)
    max_ang = math.radians(_get_float("MOTION_MAX_ANGULAR_DEG_S", 85.0))
    return motion.send({
        "cmd": "drive",
        "lin": _clampf(lin, -max_lin, max_lin),
        "ang": _clampf(ang, -max_ang, max_ang),
    })


def stop() -> "int | None":
    """Controlled stop. Always honored while connected (bypasses the gate)."""
    _invalidate_turn_verification()
    _cancel_swing_escape()
    _fx_drive_loop_stop_all()   # a stop means silence now, not at the clip's end
    if not motion.connected():
        return None
    _cancel_arc()
    return motion.send({"cmd": "stop"})


def estop() -> "int | None":
    """Hard disable until clear(). Always honored while connected."""
    _invalidate_turn_verification()
    if not motion.connected():
        return None
    _cancel_arc()
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


def arc_move(forward: bool = True, left: bool = True, small: bool = False) -> "int | None":
    """Voice "move forward and to your right" -> a gentle simultaneous curve (config
    magnitudes), shorter when "a little". forward/left are the signs (REP-103: +ang = left)."""
    lin = _get_float("MOTION_ARC_LIN_MS", 0.15) * (1.0 if forward else -1.0)
    ang = math.radians(_get_float("MOTION_ARC_ANG_DEG_S", 35.0)) * (1.0 if left else -1.0)
    dur = (_get_float("MOTION_ARC_SMALL_DURATION_SECS", 1.0) if small
           else _get_float("MOTION_ARC_DURATION_SECS", 1.6))
    return arc(lin, ang, dur)


def move_forward(dist: "float | None" = None) -> "int | None":
    return move(_get_float("MOTION_DEFAULT_MOVE_DIST_M", 0.30) if dist is None else abs(dist))


def move_back(dist: "float | None" = None) -> "int | None":
    return move(-(_get_float("MOTION_DEFAULT_MOVE_DIST_M", 0.30) if dist is None else abs(dist)))


def come_here() -> "int | None":
    return come(0.0)


# ── Status (for GUI / telemetry / logging) ───────────────────────────────────────

def status() -> "dict | None":
    return motion.telemetry()


_charging_last_true_at = 0.0   # monotonic ts of the last positive charging reading
_charge_asserted_off_at = 0.0  # monotonic ts of the operator's last "you're unplugged"


def charging() -> bool:
    """Whether drive must remain locked because the charger is attached.

    Prefer the firmware's latch. As a second safety layer, charger voltage itself
    locks the base: this build reads about 14.2 V plugged in versus roughly 13.4 V
    at a full unplugged pack. This also covers old firmware that dropped
    ``charging`` when charge current tapered near zero.

    STICKY RELEASE (field 2026-07-23): a servo current spike sags the pack voltage
    under the ~160 mΩ junction, briefly flapping BOTH the firmware flag and the
    voltage test to "unplugged" — which was letting the wheels wake up (and back-off
    reflex fire) while the cable was still attached. So once charging is seen, stay
    locked for MOTION_CHARGING_RELEASE_GRACE_SECS after the LAST positive reading; a
    genuine unplug is sustained and releases after the grace, a flap is not.
    """
    global _charging_last_true_at, _charge_asserted_off_at
    snapshot = motion.telemetry() or {}
    now = time.monotonic()
    fw_flag = bool(snapshot.get("charging"))
    if fw_flag:
        # The firmware SEES the charger again — any standing operator unplug
        # assertion is stale, drop it so the voltage backstop is re-armed.
        _charge_asserted_off_at = 0.0
    raw = fw_flag
    if not raw:
        # Voltage backstop — but the operator's explicit "you're unplugged"
        # outranks it: a freshly-topped pack's surface charge floats above the
        # lockout voltage for many minutes after a genuine unplug, and holding
        # the wheels through that window is exactly the wait the assertion
        # exists to skip. The mute lasts until the firmware sees the charger
        # again (above) or the mute window expires (surface charge is long
        # decayed by then, so the backstop means something again).
        mute = _get_float("MOTION_CHARGE_ASSERT_VOLTAGE_MUTE_SECS", 1800.0)
        asserted_off = (_charge_asserted_off_at > 0.0
                        and (now - _charge_asserted_off_at) < mute)
        if not asserted_off:
            try:
                raw = float(snapshot.get("batt_mv")) >= _get_float(
                    "MOTION_CHARGER_VOLTAGE_LOCKOUT_MV", 14000.0)
            except (TypeError, ValueError):
                raw = False
    if raw:
        _charging_last_true_at = now
        return True
    grace = _get_float("MOTION_CHARGING_RELEASE_GRACE_SECS", 20.0)
    return _charging_last_true_at > 0.0 and (now - _charging_last_true_at) < grace


def charge_assert(on: bool) -> "int | None":
    """Relay the operator's spoken word about the charge cable to the firmware
    ("chg_assert"). The firmware owns the sanity check (an "off" is refused
    while charge current measurably flows); here we just mirror the assertion
    into the host-side sticky timestamp so `charging()` agrees immediately
    instead of waiting out the release grace."""
    global _charging_last_true_at, _charge_asserted_off_at
    if not motion.connected():
        return None
    seq = motion.send({"cmd": "chg_assert", "on": bool(on)})
    if seq is not None:
        if on:
            _charging_last_true_at = time.monotonic()
            _charge_asserted_off_at = 0.0
        else:
            _charging_last_true_at = 0.0
            _charge_asserted_off_at = time.monotonic()
    return seq


def is_moving() -> bool:
    return motion.state() == "moving"
