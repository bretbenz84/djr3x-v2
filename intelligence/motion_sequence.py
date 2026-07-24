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
    thread = threading.Thread(
        target=_run,
        args=(copied, event, on_issued),
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
) -> None:
    result = "completed"
    try:
        for index, decision in enumerate(decisions, 1):
            if event.is_set():
                result = "cancelled"
                break
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
        if direction == "around":
            return motion_controller.turn(deg), 0.0
        if direction == "right":
            return motion_controller.turn_right(deg), 0.0
        return motion_controller.turn_left(deg), 0.0
    if decision.action == "motion.move":
        direction = str(args.get("direction") or "forward").lower()
        dist = float(args.get("dist_m") or getattr(config, "MOTION_DEFAULT_MOVE_DIST_M", 0.30))
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


def _step_timeout(decision: ActionDecision) -> float:
    args = decision.args or {}
    if decision.action == "motion.turn":
        amount = abs(float(args.get("deg") or getattr(config, "MOTION_DEFAULT_TURN_DEG", 90.0)))
        rate = max(1.0, float(getattr(config, "MOTION_DEFAULT_TURN_RATE", 75.0)))
        return min(120.0, max(8.0, amount / rate + 8.0))
    amount = abs(float(args.get("dist_m") or getattr(config, "MOTION_DEFAULT_MOVE_DIST_M", 0.30)))
    speed = max(0.03, float(getattr(config, "MOTION_MAX_LINEAR_MS", 0.40)))
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
