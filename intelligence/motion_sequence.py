"""Ordered execution for deterministic multi-clause drive commands.

The parser lives in action_router; this module owns only sequencing. Every finite
turn/move waits for the ESP32 ``done`` event and a settled base before the next step.
Blocked, failed, timed-out, cancelled, or superseded steps abort the remainder.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import config
from hardware import motion
from intelligence import motion_controller
from intelligence.action_router import ActionDecision

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_cancel: Optional[threading.Event] = None
_thread: Optional[threading.Thread] = None


def active() -> bool:
    with _lock:
        return bool(_thread is not None and _thread.is_alive() and _cancel is not None)


def cancel(reason: str = "cancelled", *, stop_base: bool = False) -> None:
    with _lock:
        event = _cancel
    if event is not None:
        event.set()
        _log.info("[motion_sequence] %s", reason)
    if stop_base:
        try:
            motion_controller.stop()
        except Exception:
            pass


def spin_clearance_reason(decisions: list[ActionDecision]) -> Optional[str]:
    """Refusal reason when a step in this route is a near-full spin with no room.

    ADVISORY, not a second gate. motion_controller.turn() runs the authoritative
    swing check on every send (intelligence/motion_swing.py — the bookshelf
    hand-loss incidents); this only looks ahead so the refusal can be SPOKEN before
    the route commits, instead of the first leg silently returning None and the
    human hearing nothing. It never changes an angle: dual-correcting one error is
    how a smooth motion becomes a stutter.

    It exists because check_turn SHRINKS rather than refuses whenever the allowed
    angle clears MOTION_SWING_MIN_TURN_DEG, and a shrink is the right degradation
    for a 90 and the wrong one for a spin: a 360 delivered as 147 reports
    "completed" at a heading nobody asked for, and the route's next leg then drives
    off that heading. Returns None when the ring has no data — unknown sensing is
    not a refusal anywhere else in the stack either.
    """
    if not bool(getattr(config, "MOTION_ROUTE_SPIN_CHECK_ENABLED", True)):
        return None
    floor = float(getattr(config, "MOTION_ROUTE_SPIN_CHECK_DEG", 270.0))
    spins = [d for d in decisions or []
             if d.action == "motion.turn" and abs(float((d.args or {}).get("deg") or 0.0)) >= floor]
    if not spins:
        return None
    try:
        from intelligence import motion_swing
        tele = motion.telemetry()
        tof = tele.get("tof_mm") if isinstance(tele, dict) else None
        if not isinstance(tof, dict):
            return None
        for decision in spins:
            args = decision.args or {}
            want = abs(float(args.get("deg") or 0.0))
            signed = want if str(args.get("direction") or "left").lower() != "right" else -want
            allowed, limiter = motion_swing.allowed_turn_deg(signed, tof)
            if limiter is not None and abs(allowed) < want - 1.0:
                _log.info("[motion_sequence] full spin refused — %s caps it at %.0f°",
                          limiter, abs(allowed))
                return "spin_no_elbow_room"
    except Exception as exc:
        _log.debug("[motion_sequence] spin clearance check skipped: %s", exc)
    return None


def start(
    decisions: list[ActionDecision],
    *,
    on_issued: Optional[Callable[[ActionDecision], None]] = None,
) -> bool:
    """Start a bounded sequence in the background; supersede any prior sequence."""
    if not decisions or not motion_controller.available():
        return False
    max_steps = max(2, int(getattr(config, "MOTION_SEQUENCE_MAX_STEPS", 8)))
    if len(decisions) < 2 or len(decisions) > max_steps:
        return False
    if any(d.action not in {"motion.turn", "motion.move", "motion.arc"} for d in decisions):
        return False

    cancel("superseded by a new motion sequence", stop_base=True)
    # Sequences only carry explicit voice routes — stand realign down so it can't
    # rotate the body off the user's chosen heading between (or after) the steps.
    try:
        from intelligence import motion_agency
        motion_agency.note_user_motion()
    except Exception:
        pass
    # Every leg of a spoken route is user-commanded: keep the drive sounds in
    # overlay mode so the "On it — 2 moves." confirmation can't mute them.
    try:
        motion_controller.note_user_commanded_motion()
    except Exception:
        pass
    event = threading.Event()
    copied = [
        ActionDecision(
            action=d.action,
            confidence=d.confidence,
            args=dict(d.args or {}),
            reason=d.reason,
        )
        for d in decisions
    ]
    # The FIRST leg is issued here, on the caller's thread, so a refusal is known
    # before the caller confirms anything. Issued in the background, the swing
    # check refused the opening turn ("Can't swing that way") and the caller,
    # told only that a thread had started, still said "On it — 2 moves" (field
    # 2026-09-02 23:04:34). A refused first leg is no sequence at all.
    first = _issue(copied[0])
    if first[0] is None:
        _log.info("[motion_sequence] not started — first step %s args=%s was refused",
                  copied[0].action, copied[0].args)
        return False
    thread = threading.Thread(
        target=_run,
        args=(copied, event, on_issued, first),
        name="motion-sequence",
        daemon=True,
    )
    global _cancel, _thread
    with _lock:
        _cancel = event
        _thread = thread
    thread.start()
    return True


def _run(
    decisions: list[ActionDecision],
    event: threading.Event,
    on_issued: Optional[Callable[[ActionDecision], None]],
    first: Optional[tuple[Optional[int], float]] = None,
) -> None:
    """`first` is the already-issued (seq, arc_duration) of step 1 when start()
    issued it synchronously; later steps are issued here."""
    result = "completed"
    try:
        for index, decision in enumerate(decisions, 1):
            if event.is_set():
                result = "cancelled"
                break
            if index == 1 and first is not None:
                seq, arc_duration = first
            else:
                seq, arc_duration = _issue(decision)
            if seq is None:
                result = "suppressed"
                break
            if decision.action == "motion.move" and seq:
                # Voice-commanded sequence leg: speak up if the firmware cuts it on
                # an obstacle (silence read as ignoring the command, field 2026-07-23).
                try:
                    motion_controller.announce_if_blocked(int(seq))
                except Exception:
                    pass
            if on_issued is not None:
                try:
                    on_issued(decision)
                except Exception:
                    pass
            _log.info(
                "[motion_sequence] step %d/%d issued: %s args=%s",
                index, len(decisions), decision.action, decision.args,
            )
            if decision.action == "motion.arc":
                if event.wait(arc_duration):
                    result = "cancelled"
                    break
            elif seq == 0:
                # No-op step (compass turn already facing the heading) — nothing was
                # sent, so there is no done frame to wait for. Advance immediately.
                pass
            else:
                done = motion.wait_done(int(seq), timeout=_step_timeout(decision))
                if event.is_set():
                    result = "cancelled"
                    break
                done_result = str((done or {}).get("result") or "timeout").lower()
                if done_result != "completed":
                    result = done_result
                    break
            if not _wait_until_settled(event):
                result = "cancelled" if event.is_set() else "not_settled"
                break
    except Exception as exc:
        result = "error"
        _log.warning("[motion_sequence] runner failed: %s", exc)
    finally:
        if result != "completed":
            try:
                motion_controller.stop()
            except Exception:
                pass
        _log.info("[motion_sequence] ended: %s", result)
        global _cancel, _thread
        with _lock:
            if _cancel is event:
                _cancel = None
                _thread = None


def _issue(decision: ActionDecision) -> tuple[Optional[int], float]:
    args = decision.args or {}
    if decision.action == "motion.turn":
        if args.get("compass"):
            # Compass-relative step ("turn north then move forward"): the relative
            # angle is computed from the live heading AT ISSUE TIME. seq 0 = already
            # facing it — report a tiny settle so the runner advances to the next step.
            seq = motion_controller.turn_to_compass(float(args.get("compass_deg") or 0.0))
            if seq == 0:
                return 0, 0.1
            return seq, 0.0
        direction = str(args.get("direction") or "left").lower()
        deg = float(args.get("deg") or getattr(config, "MOTION_DEFAULT_TURN_DEG", 90.0))
        # `rate` arrives only from a motion.route step the speaker asked to take
        # slowly (action_router.route_tool_to_decisions). The voice-friendly
        # turn_left/turn_right wrappers take no rate, so a paced step goes to the
        # signed primitive instead — same command, same clamps, one extra kwarg.
        rate = _pace_value(args, "rate")
        if rate is not None:
            signed = deg if direction in ("left", "around") else -deg
            return motion_controller.turn(signed, rate=rate,
                                          allow_reverse=(direction == "around")), 0.0
        if direction == "around":
            # A heading goal: the other way round is an acceptable alternative.
            return motion_controller.turn(deg, allow_reverse=True), 0.0
        if direction == "right":
            return motion_controller.turn_right(deg), 0.0
        return motion_controller.turn_left(deg), 0.0
    if decision.action == "motion.move":
        direction = str(args.get("direction") or "forward").lower()
        dist = float(args.get("dist_m") or getattr(config, "MOTION_DEFAULT_MOVE_DIST_M", 0.30))
        speed = _pace_value(args, "speed")
        if speed is not None:
            return motion_controller.move(-dist if direction == "back" else dist,
                                          speed=speed), 0.0
        if direction == "back":
            return motion_controller.move_back(dist), 0.0
        return motion_controller.move_forward(dist), 0.0
    if decision.action == "motion.arc":
        forward = str(args.get("lin_dir") or "forward").lower() != "back"
        left = str(args.get("ang_dir") or "left").lower() == "left"
        small = bool(args.get("small"))
        seq = motion_controller.arc_move(forward, left, small=small)
        duration = float(getattr(
            config,
            "MOTION_ARC_SMALL_DURATION_SECS" if small else "MOTION_ARC_DURATION_SECS",
            1.0 if small else 1.6,
        ))
        return seq, duration
    return None, 0.0


def _pace_value(args: dict, key: str) -> Optional[float]:
    """A positive per-step rate/speed override, or None. Never raises on junk."""
    try:
        value = float(args.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _step_timeout(decision: ActionDecision) -> float:
    args = decision.args or {}
    # The timeout must be computed from the rate this step is ACTUALLY driven at.
    # A half-speed leg measured against the full-speed default times out mid-move,
    # and a timed-out step aborts the whole remaining route (_run breaks on any
    # non-"completed" done result) — so "drive that slowly" would have read as
    # "drive the first leg and give up".
    if decision.action == "motion.turn":
        amount = abs(float(args.get("deg") or getattr(config, "MOTION_DEFAULT_TURN_DEG", 90.0)))
        rate = _pace_value(args, "rate") or float(
            getattr(config, "MOTION_DEFAULT_TURN_RATE", 75.0))
        rate = max(1.0, rate)
        return min(120.0, max(8.0, amount / rate + 8.0))
    amount = abs(float(args.get("dist_m") or getattr(config, "MOTION_DEFAULT_MOVE_DIST_M", 0.30)))
    speed = _pace_value(args, "speed") or float(
        getattr(config, "MOTION_MAX_LINEAR_MS", 0.40))
    speed = max(0.03, speed)
    return min(180.0, max(8.0, amount / speed + 10.0))


def _wait_until_settled(event: threading.Event) -> bool:
    deadline = time.monotonic() + float(getattr(config, "MOTION_SEQUENCE_SETTLE_TIMEOUT_SECS", 4.0))
    while time.monotonic() < deadline:
        if event.is_set():
            return False
        if motion.state() == "idle":
            return True
        event.wait(0.04)
    return False
