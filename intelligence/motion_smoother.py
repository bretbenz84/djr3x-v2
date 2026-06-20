"""Per-axis velocity/acceleration-limited target follower.

This is the "no instantaneous jumps" smoother from the gaze spec (deliverable 3).
It turns a stream of *target* poses into physically plausible *commanded* poses by
clamping how fast each axis may move (velocity limit) and how fast that velocity may
change (acceleration limit).

Where it is used
----------------
* **Offline sim / demo / tests** — the canonical actuation path for
  :class:`intelligence.head_interface.SimHead`, so a scripted conversation produces
  realistic motion you can eyeball before touching hardware.
* **NOT the live robot.** On the real droid the gaze engine composes with the
  existing 12.5 Hz face-tracking loop in ``consciousness._step_face_tracking``,
  which already enforces its own per-axis slew caps (``*_MAX_STEP_QUS``),
  exponential smoothing and rail-damping. Stacking a second smoother there would
  double-filter and fight the single servo writer. This module exists so the
  *behavioural timing* (saccade-fast vs posture-slow) can be validated in sim.

Design notes tied to the spec rules
-----------------------------------
* Saccade-like gaze shifts should be FAST but still velocity-clamped — callers pass
  a high ``max_vel``.
* Posture moves (the POLE / head-height axis) should be SLOW — callers pass a low
  ``max_vel``.
* A watchdog rejects NaN / non-finite targets and holds the last safe value.

The module is pure Python with no hardware, numpy, or project-config dependency, so
it imports and runs anywhere (including under the bare test interpreter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class AxisLimiter:
    """A single 1-D velocity + acceleration limited follower.

    Integrates toward ``target`` each :meth:`step`, never exceeding ``max_vel`` and
    never changing velocity faster than ``max_accel``. Holds its last safe position
    if a non-finite target/dt is supplied (watchdog).
    """

    position: float = 0.0
    velocity: float = 0.0
    #: Hard travel limits; targets are clamped into ``[min_limit, max_limit]``.
    min_limit: float = -math.inf
    max_limit: float = math.inf
    #: 0/None disables that limit (move is then purely step-clamped by the caller).
    max_vel: float = math.inf
    max_accel: float = math.inf

    def reset(self, position: float) -> None:
        if _finite(position):
            self.position = float(position)
        self.velocity = 0.0

    def _clamp_target(self, target: float) -> float:
        return max(self.min_limit, min(self.max_limit, float(target)))

    def step(self, target: float, dt: float) -> float:
        """Advance one tick toward ``target`` over ``dt`` seconds; return new position."""
        # Watchdog: a bad target or dt must not move (or NaN-poison) the axis.
        if not _finite(target) or not _finite(dt) or dt <= 0.0:
            self.velocity = 0.0
            return self.position

        target = self._clamp_target(target)
        max_vel = self.max_vel if (self.max_vel and self.max_vel > 0) else math.inf
        max_accel = self.max_accel if (self.max_accel and self.max_accel > 0) else math.inf

        # Desired velocity to *exactly* reach the target this tick, then cap it to the
        # axis speed limit (sign-preserving).
        desired_vel = (target - self.position) / dt
        if desired_vel > max_vel:
            desired_vel = max_vel
        elif desired_vel < -max_vel:
            desired_vel = -max_vel

        # Acceleration limit: cap how much the velocity may change this tick.
        if math.isfinite(max_accel):
            dv = desired_vel - self.velocity
            dv_cap = max_accel * dt
            if dv > dv_cap:
                desired_vel = self.velocity + dv_cap
            elif dv < -dv_cap:
                desired_vel = self.velocity - dv_cap

        new_pos = self.position + desired_vel * dt

        # Don't overshoot the target due to acceleration ramp granularity.
        if (desired_vel > 0 and new_pos > target) or (desired_vel < 0 and new_pos < target):
            new_pos = target
            desired_vel = (new_pos - self.position) / dt if dt > 0 else 0.0

        self.position = self._clamp_target(new_pos)
        self.velocity = desired_vel
        return self.position

    def at_target(self, target: float, tol: float = 1e-3) -> bool:
        return _finite(target) and abs(self.position - float(target)) <= tol


@dataclass
class MotionSmoother:
    """A 3-axis follower (yaw / pitch / pole) wrapping three :class:`AxisLimiter`.

    Velocity/accel limits are per-axis so the POLE (posture) axis can be tuned slow
    while YAW/PITCH (gaze) stay saccade-fast. Limits may be overridden per
    :meth:`step` call (e.g. a fast saccade vs a slow settle).
    """

    yaw: AxisLimiter = field(default_factory=AxisLimiter)
    pitch: AxisLimiter = field(default_factory=AxisLimiter)
    pole: AxisLimiter = field(default_factory=AxisLimiter)

    @classmethod
    def from_limits(
        cls,
        *,
        yaw_limit: tuple[float, float],
        pitch_limit: tuple[float, float],
        pole_limit: tuple[float, float],
        yaw_max_vel: float,
        pitch_max_vel: float,
        pole_max_vel: float,
        yaw_max_accel: float = math.inf,
        pitch_max_accel: float = math.inf,
        pole_max_accel: float = math.inf,
        start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "MotionSmoother":
        return cls(
            yaw=AxisLimiter(
                position=start[0], min_limit=yaw_limit[0], max_limit=yaw_limit[1],
                max_vel=yaw_max_vel, max_accel=yaw_max_accel,
            ),
            pitch=AxisLimiter(
                position=start[1], min_limit=pitch_limit[0], max_limit=pitch_limit[1],
                max_vel=pitch_max_vel, max_accel=pitch_max_accel,
            ),
            pole=AxisLimiter(
                position=start[2], min_limit=pole_limit[0], max_limit=pole_limit[1],
                max_vel=pole_max_vel, max_accel=pole_max_accel,
            ),
        )

    def reset(self, yaw: float, pitch: float, pole: float) -> None:
        self.yaw.reset(yaw)
        self.pitch.reset(pitch)
        self.pole.reset(pole)

    def step(
        self,
        target_yaw: float,
        target_pitch: float,
        target_pole: float,
        dt: float,
        *,
        yaw_max_vel: float | None = None,
        pitch_max_vel: float | None = None,
        pole_max_vel: float | None = None,
    ) -> tuple[float, float, float]:
        """Advance all three axes one tick; return ``(yaw, pitch, pole)``.

        Optional ``*_max_vel`` overrides let a caller request a fast saccade for this
        move without permanently changing the axis limit.
        """
        if yaw_max_vel is not None:
            self.yaw.max_vel = yaw_max_vel
        if pitch_max_vel is not None:
            self.pitch.max_vel = pitch_max_vel
        if pole_max_vel is not None:
            self.pole.max_vel = pole_max_vel
        return (
            self.yaw.step(target_yaw, dt),
            self.pitch.step(target_pitch, dt),
            self.pole.step(target_pole, dt),
        )

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.yaw.position, self.pitch.position, self.pole.position)
