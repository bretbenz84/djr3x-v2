"""Swing check: will a spin-in-place sweep the body (or an arm) into something?

Field 2026-08 (more than once): Rex stood about a foot from a bookshelf, was told
to "turn left", and his LEFT HAND swept back into the shelf and fell off. Nothing
in the stack saw it coming:

- The firmware reflex (safety.cpp) only gates LINEAR travel. A pure spin has
  lin = 0, so "turning away from a block is always free" — even when the thing
  you are close to is BEHIND you and the turn sweeps the body into it.
- The base does not spin about its own centre. The drive axle sits
  MOTION_AXLE_AFT_OF_CENTER_M (~9 in) aft of the ring centre (drive wheels rear,
  omni casters front), and a differential spin rotates about the axle midpoint.
  So the front of the ring orbits ~0.5 m out while the rear edge barely moves,
  and the arms on the left (front-left and back-left, ~1.5 ft proud of the
  ring) orbit further still. A CCW (left) turn carries every LEFT-side part
  REARWARD — exactly into a shelf that is behind-left.

This module is the "swing check" the sensing roadmap (docs/motion_sensing_roadmap.md
§2) called for: before any autonomous spin, project the radial ToF ring's returns
into the pivot (axle) frame, sweep every body extent through the requested angle,
and find the largest angle that keeps every extent clear of every return by
MOTION_SWING_MARGIN_M. The caller then turns that far, or not at all.

Frames: REP-103 body frame, x forward, y left, +angle = CCW/left. Sensor bearings
are quoted about the RING CENTRE (docs/motion_system.md §6.2); everything here is
re-expressed about the AXLE MIDPOINT, the pivot, at (-d, 0) from the ring centre.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import config

_log = logging.getLogger(__name__)

# Radial ring bearings about the ring centre (docs §6.2) — the tof_mm keys.
_RING_SENSORS = (
    ("fl", 22.5), ("fr", -22.5), ("rl", 157.5), ("rr", -157.5),
    ("lf", 67.5), ("lb", 112.5), ("rf", -67.5), ("rb", -112.5),
)


def _num(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def _wrap_deg(a: float) -> float:
    """Wrap to (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a > 180.0:
        a -= 360.0
    elif a <= -180.0:
        a += 360.0
    return a


def _ring_to_pivot(bearing_deg: float, radius_m: float, d: float) -> "tuple[float, float]":
    """A point given in ring-centre polar coords -> (radius, bearing_deg) about the
    pivot, which sits `d` metres behind the ring centre."""
    b = math.radians(bearing_deg)
    x = d + radius_m * math.cos(b)
    y = radius_m * math.sin(b)
    return math.hypot(x, y), math.degrees(math.atan2(y, x))


def body_extents() -> "list[tuple[str, float, float]]":
    """(label, radius_m, bearing_deg) of every body point that can hit something,
    about the PIVOT. The ring is sampled every 45°; the arms come from config."""
    d = _num("MOTION_AXLE_AFT_OF_CENTER_M", 0.23)
    ring_r = _num("MOTION_RING_RADIUS_M", 0.27)
    out: list[tuple[str, float, float]] = []
    for deg in range(0, 360, 45):
        r, b = _ring_to_pivot(float(deg), ring_r, d)
        out.append((f"ring@{deg}", r, b))
    for item in getattr(config, "MOTION_BODY_EXTENTS", ()) or ():
        try:
            label, bearing, radius = item
            r, b = _ring_to_pivot(float(bearing), float(radius), d)
            out.append((str(label), r, b))
        except (TypeError, ValueError):
            _log.warning("[swing] bad MOTION_BODY_EXTENTS entry ignored: %r", item)
    return out


def obstacles_from_tof(tof_mm: dict) -> "list[tuple[str, float, float]]":
    """Every VALID radial return as (sensor, radius_m, bearing_deg) about the pivot.
    A negative reading is the -1 error sentinel and is skipped — no information,
    not "clear". Room-max returns (~4000 mm) are real far readings and harmless."""
    d = _num("MOTION_AXLE_AFT_OF_CENTER_M", 0.23)
    ring_r = _num("MOTION_RING_RADIUS_M", 0.27)
    out: list[tuple[str, float, float]] = []
    for key, bearing in _RING_SENSORS:
        try:
            mm = float(tof_mm.get(key))
        except (TypeError, ValueError):
            continue
        if mm < 0.0:
            continue
        r, b = _ring_to_pivot(bearing, ring_r + mm / 1000.0, d)
        out.append((key, r, b))
    return out


def allowed_turn_deg(turn_deg: float, tof_mm: Optional[dict]) -> "tuple[float, Optional[str]]":
    """Largest |angle| (same sign as `turn_deg`, ≤ |turn_deg|) the body can spin
    without an extent sweeping into a ToF return. Returns (allowed_deg, limiter)
    where limiter names the extent/sensor pair that capped it (None = unlimited).

    Unknown sensing (no telemetry) returns the request untouched: the presence
    gate in motion_controller already refuses autonomy when the ring is dead, and
    a single errored sensor must not make "turn left" refuse forever.
    """
    if not turn_deg:
        return 0.0, None
    if not isinstance(tof_mm, dict):
        return turn_deg, None
    sign = 1.0 if turn_deg > 0 else -1.0
    want = abs(float(turn_deg))
    margin = max(0.0, _num("MOTION_SWING_MARGIN_M", 0.10))
    pad = max(0.0, _num("MOTION_SWING_ANGULAR_PAD_DEG", 20.0))

    limit = want
    limiter: Optional[str] = None
    for ext_label, ext_r, ext_b in body_extents():
        reach = ext_r + margin
        for key, obs_r, obs_b in obstacles_from_tof(tof_mm):
            if obs_r > reach:
                continue                      # this extent never reaches that far out
            # Angular distance from the extent to the return, measured in the
            # direction of the turn: 0..360.
            delta = (sign * (obs_b - ext_b)) % 360.0
            room = delta - pad
            if room < 0.0:
                # Inside the pad: either already brushing it (delta ≈ 0) — stop —
                # or just past it in the turn direction (delta ≈ 360), in which
                # case the turn moves AWAY and the full circle minus pad applies.
                room = 0.0 if delta <= 180.0 else (delta - pad) % 360.0
            if room < limit:
                limit = room
                limiter = f"{ext_label}->{key}@{obs_r:.2f}m"
    return sign * max(0.0, limit), limiter


def check_turn(turn_deg: float, tof_mm: Optional[dict]) -> "tuple[float, Optional[str]]":
    """Policy wrapper: (deg_to_send, reason). reason None = go; "swing_blocked" =
    refuse (allowed angle under MOTION_SWING_MIN_TURN_DEG). A shrunk turn is a go
    with a smaller angle — the caller should log it."""
    if not bool(getattr(config, "MOTION_SWING_CHECK_ENABLED", True)):
        return turn_deg, None
    allowed, limiter = allowed_turn_deg(turn_deg, tof_mm)
    if limiter is None:
        return turn_deg, None
    min_turn = _num("MOTION_SWING_MIN_TURN_DEG", 10.0)
    if abs(allowed) < min_turn:
        _log.info("[swing] %+.0f° turn refused — %s would sweep into it (room %.0f°)",
                  turn_deg, limiter, abs(allowed))
        return 0.0, "swing_blocked"
    _log.info("[swing] %+.0f° turn shrunk to %+.0f° — %s in the swing path",
              turn_deg, allowed, limiter)
    return allowed, None
