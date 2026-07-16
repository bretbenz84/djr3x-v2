"""
intelligence/motion_agency.py — autonomous base motion (owner spec 2026-07-06).

Two behaviors, evaluated once per consciousness tick (~1 Hz):

REALIGN — turn the base to face the person the head is tracking. The neck servo is
the signal: face-tracking keeps the FACE centered in frame, so frame error goes to
zero even when the body points the wrong way — but the neck's offset from neutral is
exactly the body's misalignment. When the neck sits past MOTION_FACE_NECK_FRACTION of
its half-span for MOTION_FACE_CONFIRM_TICKS consecutive ticks, the base turns by a
proportional chunk and face-tracking naturally re-centers the neck as it comes around.
Iterative small corrections + a cooldown, never one exact spin (no oscillation).

APPROACH — when the tracked person stays at "public" distance (vision/proxemics:
face width < 30% of frame) for MOTION_APPROACH_CONFIRM_TICKS ticks AND the base is
already roughly facing them, issue `come`: the firmware turns to heading 0 and
advances until the nearest FORWARD ToF obstacle is MOTION_COME_STOP_AT_M away — the
person's own body is the stop target, and anything in between (furniture, wall) stops
the base the same way. No cliff sensing needed or used (owner: never upstairs).

Safety layering (all independent of this module):
  - firmware reflex: Z_STOP zone forces ST_BLOCKED regardless of host commands
  - drive deadman + comms watchdog on the ESP32
  - motion_controller._autonomous_allowed(): manual gamepad owner wins, paused
    interaction blocks, disconnected blocks
This module only DECIDES; it never streams velocities (turn/come are closed-loop
firmware commands), acts only from motion state "idle", one action per tick.

Kill switches: AUTONOMOUS_MOTION_ENABLED master; MOTION_FACE_PERSON_ENABLED /
MOTION_APPROACH_ENABLED per behavior.
"""

import logging
import time
from typing import Optional

import config
from hardware import motion
from intelligence import motion_controller

_log = logging.getLogger(__name__)

# Per-behavior confirmation counters + cooldown stamps (reset by _reset()).
_state = {
    "neck_hits": 0,
    "far_hits": 0,
    "last_turn_at": 0.0,
    "last_approach_at": 0.0,
}


def _flag(name: str, default: bool = True) -> bool:
    return bool(getattr(config, name, default))


def _num(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def _reset(*counters: str) -> None:
    for key in counters:
        _state[key] = 0


def neck_offset_fraction() -> Optional[float]:
    """Neck offset from neutral as a signed fraction of the half-span.

    + = head panned toward Rex's RIGHT (larger frame x — the face-reveal lateral
    convention; qus above neutral from the tracking logs). None when the neck
    position or channel config is unavailable (e.g. servo-less dev Mac).
    """
    try:
        from world_state import world_state
        positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
        neck = positions.get("neck")
        cfg = config.SERVO_CHANNELS["neck"]
        neutral = float(cfg["neutral"])
        half_span = max(1.0, min(float(cfg["max"]) - neutral, neutral - float(cfg["min"])))
        if neck is None:
            return None
        return (float(neck) - neutral) / half_span
    except Exception:
        return None


def _tracked_person(snapshot: dict) -> Optional[dict]:
    """The world_state person entry the head is currently locked onto, or None."""
    try:
        from world_state import world_state
        tracking = (world_state.get("self_state") or {}).get("face_tracking") or {}
        if not (tracking.get("locked") and tracking.get("visible")):
            return None
        lock_key = str(tracking.get("lock_key") or "")
        slot = lock_key.split(":", 1)[1] if ":" in lock_key else lock_key
        for person in snapshot.get("people") or []:
            if isinstance(person, dict) and person.get("id") == slot:
                return person
        return None
    except Exception:
        return None


def _turn_degrees_for(frac: float) -> float:
    """Base turn (deg, + = left/CCW per the wire protocol) that reduces a neck
    offset fraction. Neck toward Rex's right (+frac) needs a RIGHT (CW, negative)
    base turn; MOTION_FACE_TURN_INVERT flips if field testing disagrees."""
    max_deg = _num("MOTION_FACE_TURN_MAX_DEG", 60.0)
    min_deg = _num("MOTION_FACE_TURN_MIN_DEG", 10.0)
    deg = -frac * max_deg
    if _flag("MOTION_FACE_TURN_INVERT", False):
        deg = -deg
    if abs(deg) < min_deg:
        deg = min_deg if deg >= 0 else -min_deg
    return max(-max_deg, min(max_deg, deg))


def step(snapshot: dict, profile) -> None:
    """One autonomy tick. Call from the consciousness loop; never raises."""
    try:
        _step_inner(snapshot, profile)
    except Exception as exc:
        _log.debug("motion agency step error: %s", exc)


def _step_inner(snapshot: dict, profile) -> None:
    if not _flag("AUTONOMOUS_MOTION_ENABLED", True):
        return
    # A room-exploration session OWNS the base while it wanders — realign/approach
    # must not interleave a maneuver between its legs.
    try:
        from intelligence import exploration
        if exploration.active():
            _reset("neck_hits", "far_hits")
            return
    except Exception:
        pass
    if not motion_controller.available():
        _reset("neck_hits", "far_hits")
        return
    # Never start a maneuver while the human is mid-sentence (motor noise into the
    # mic during THEIR turn) or the base is already doing something / blocked.
    if getattr(profile, "user_mid_sentence", False):
        _reset("neck_hits", "far_hits")
        return
    if motion.state() != "idle":
        return

    person = _tracked_person(snapshot)
    if person is None:
        _reset("neck_hits", "far_hits")
        return

    now = time.monotonic()
    frac = neck_offset_fraction()

    # ── REALIGN: rotate the base under the head ──────────────────────────────
    if _flag("MOTION_FACE_PERSON_ENABLED", True) and frac is not None:
        threshold = _num("MOTION_FACE_NECK_FRACTION", 0.30)
        if abs(frac) >= threshold:
            _state["neck_hits"] += 1
        else:
            _state["neck_hits"] = 0
        confirm = int(_num("MOTION_FACE_CONFIRM_TICKS", 2))
        cooldown = _num("MOTION_FACE_TURN_COOLDOWN_SECS", 8.0)
        if (_state["neck_hits"] >= confirm
                and (now - _state["last_turn_at"]) >= cooldown):
            deg = _turn_degrees_for(frac)
            seq = motion_controller.turn(deg)
            if seq is not None:
                _log.info(
                    "[motion_agency] realign: neck %.0f%% off-center -> base turn %+.0f deg "
                    "(person=%s)",
                    frac * 100.0, deg, person.get("person_db_id") or person.get("id"),
                )
                _state["last_turn_at"] = now
            _reset("neck_hits")
            return  # one maneuver per tick

    # ── APPROACH: close distance to a far person ──────────────────────────────
    if not _flag("MOTION_APPROACH_ENABLED", True):
        return
    # Critical battery: stop VOLUNTEERING drives (voice-commanded motion still
    # obeys — the pack's BMS is the hard protection; this is Rex pacing himself).
    try:
        from intelligence import battery_awareness
        if battery_awareness.battery_critical():
            _reset("far_hits")
            return
    except Exception:
        pass
    # A whole-base approach is a big proactive act — respect the same social gates
    # as unsolicited speech, plus require an active turn NOT being processed.
    if getattr(profile, "suppress_proactive", False) or getattr(profile, "interaction_busy", False):
        _reset("far_hits")
        return
    centered = _num("MOTION_APPROACH_CENTERED_FRACTION", 0.18)
    facing_them = frac is None or abs(frac) < centered
    if person.get("distance_zone") == "public" and facing_them:
        _state["far_hits"] += 1
    else:
        _state["far_hits"] = 0
    confirm = int(_num("MOTION_APPROACH_CONFIRM_TICKS", 4))
    cooldown = _num("MOTION_APPROACH_COOLDOWN_SECS", 120.0)
    if (_state["far_hits"] >= confirm
            and (now - _state["last_approach_at"]) >= cooldown):
        seq = motion_controller.come(0.0)  # firmware stops MOTION_COME_STOP_AT_M short
        if seq is not None:
            _log.info(
                "[motion_agency] approach: person %s at public distance -> come "
                "(stop_at=%.2fm, ToF-guarded)",
                person.get("person_db_id") or person.get("id"),
                _num("MOTION_COME_STOP_AT_M", 0.60),
            )
            _state["last_approach_at"] = now
        _reset("far_hits")
