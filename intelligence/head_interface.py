"""Injectable head-actuator interface for the 3-DOF gaze controller.

The gaze *brain* (:mod:`intelligence.gaze_engine`) is pure and never touches
hardware. Actuation is abstracted behind :class:`HeadInterface` so the same
behaviour can drive a software sim (offline tuning / tests) or, in principle, the
servos directly.

Three implementations / paths
-----------------------------
* :class:`SimHead` — a no-op/sim actuator backed by
  :class:`intelligence.motion_smoother.MotionSmoother`. Records a pose history so a
  scripted conversation can be eyeballed before wiring hardware. Used by
  ``tools/gaze_demo_sim.py`` and the unit tests.
* :class:`RealHead` — a thin adapter that maps abstract ``yaw_deg / pitch_deg /
  pole_mm`` onto the project's Pololu-Maestro driver (:mod:`hardware.servos`).
  Useful for direct/manual actuation and bench bring-up.
* **The live robot does NOT use this interface.** On the droid, the gaze engine
  composes with the existing 12.5 Hz closed-loop face-tracking in
  ``consciousness._step_face_tracking`` (camera-on-head visual servo, single servo
  writer). ``RealHead`` is an *open-loop* actuator and must not be run concurrently
  with that loop, or two writers will fight the one serial lock.

DOF axis convention (matches the spec)
--------------------------------------
* ``yaw_deg``   — +left / -right (REP-103-ish; maps to the ``neck`` channel)
* ``pitch_deg`` — +up / -down, 0 = level (maps to the inverted ``headtilt`` channel)
* ``pole_mm``   — head height; + = raised/lean-in, 0 = settled (maps to ``headlift``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from intelligence.motion_smoother import MotionSmoother

_log = logging.getLogger(__name__)


@dataclass
class HeadPose:
    """Abstract head pose in human-readable units."""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    pole_mm: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.yaw_deg, self.pitch_deg, self.pole_mm)


@runtime_checkable
class HeadInterface(Protocol):
    """The clean injectable head actuator the gaze controller targets."""

    def set_yaw(self, deg: float, max_vel_dps: float) -> None: ...
    def set_pitch(self, deg: float, max_vel_dps: float) -> None: ...
    def set_pole(self, mm: float, max_vel_mms: float) -> None: ...
    def get_pose(self) -> HeadPose: ...


@dataclass
class SimHead:
    """Software head for offline testing — velocity-limited, records history.

    ``set_*`` store a target + requested max velocity; :meth:`tick` advances the
    :class:`MotionSmoother` by ``dt`` so :meth:`get_pose` reflects realistic motion.
    This lets ``demo_sim`` print a believable per-tick pose/velocity trace.
    """

    smoother: MotionSmoother = field(default_factory=MotionSmoother)
    record: bool = True
    history: list[HeadPose] = field(default_factory=list)
    _t_yaw: float = 0.0
    _t_pitch: float = 0.0
    _t_pole: float = 0.0
    _v_yaw: Optional[float] = None
    _v_pitch: Optional[float] = None
    _v_pole: Optional[float] = None

    def __post_init__(self) -> None:
        # Start targets at the current smoother position so a fresh SimHead holds
        # its pose until commanded.
        self._t_yaw, self._t_pitch, self._t_pole = self.smoother.position

    def set_yaw(self, deg: float, max_vel_dps: float) -> None:
        self._t_yaw = float(deg)
        self._v_yaw = float(max_vel_dps) if max_vel_dps else None

    def set_pitch(self, deg: float, max_vel_dps: float) -> None:
        self._t_pitch = float(deg)
        self._v_pitch = float(max_vel_dps) if max_vel_dps else None

    def set_pole(self, mm: float, max_vel_mms: float) -> None:
        self._t_pole = float(mm)
        self._v_pole = float(max_vel_mms) if max_vel_mms else None

    def tick(self, dt: float) -> HeadPose:
        """Advance the simulated head ``dt`` seconds toward the last commanded targets."""
        yaw, pitch, pole = self.smoother.step(
            self._t_yaw, self._t_pitch, self._t_pole, dt,
            yaw_max_vel=self._v_yaw, pitch_max_vel=self._v_pitch, pole_max_vel=self._v_pole,
        )
        pose = HeadPose(yaw, pitch, pole)
        if self.record:
            self.history.append(pose)
        return pose

    def get_pose(self) -> HeadPose:
        yaw, pitch, pole = self.smoother.position
        return HeadPose(yaw, pitch, pole)


class RealHead:
    """Open-loop adapter mapping abstract pose onto :mod:`hardware.servos`.

    NOTE: for direct / bench actuation only. The live conversational robot drives
    the head through ``consciousness._step_face_tracking`` (closed-loop). Do not run
    a :class:`RealHead` and that loop at the same time.
    """

    def __init__(self, gaze_config=None) -> None:
        # Lazy: avoid importing config/servos at module import so SimHead-only users
        # (tests, demo) never pull in hardware.
        from intelligence.gaze_engine import GazeConfig

        self._cfg = gaze_config or GazeConfig.from_config()

    # --- abstract -> Maestro quarter-microseconds -----------------------------
    def _profile_and_set(self, name: str, qus: int, max_vel_native: int) -> None:
        try:
            from hardware import servos  # lazy: hardware optional
            import config

            ch = int(config.SERVO_CHANNELS[name]["ch"])
            # TODO(hardware): Maestro 8-bit speed only caps the slew rate; the gaze
            # engine's per-move velocity is expressed in deg/s|mm/s. Translate via the
            # axis scale so a "saccade" gets a high speed and "posture" a low one.
            servos.set_motion_profile([ch], speed=int(max_vel_native), acceleration=8)
            servos.set_servo(ch, int(qus))
        except Exception as exc:  # pragma: no cover - hardware/serial edge
            _log.debug("RealHead set %s failed: %s", name, exc)

    def set_yaw(self, deg: float, max_vel_dps: float) -> None:
        self._profile_and_set("neck", self._cfg.yaw_deg_to_neck_qus(deg), 140)

    def set_pitch(self, deg: float, max_vel_dps: float) -> None:
        self._profile_and_set("headtilt", self._cfg.pitch_deg_to_tilt_qus(deg), 120)

    def set_pole(self, mm: float, max_vel_mms: float) -> None:
        # Posture axis: deliberately slow.
        self._profile_and_set("headlift", self._cfg.pole_mm_to_lift_qus(mm), 35)

    def get_pose(self) -> HeadPose:
        try:
            from world_state import world_state

            positions = (world_state.get("self_state") or {}).get("servo_positions") or {}
            neck = int(positions.get("neck", self._cfg.neck_neutral))
            tilt = int(positions.get("headtilt", self._cfg.tilt_neutral))
            lift = int(positions.get("headlift", self._cfg.lift_neutral))
        except Exception:
            neck, tilt, lift = self._cfg.neck_neutral, self._cfg.tilt_neutral, self._cfg.lift_neutral
        return HeadPose(
            yaw_deg=self._cfg.neck_qus_to_yaw_deg(neck),
            pitch_deg=self._cfg.tilt_qus_to_pitch_deg(tilt),
            pole_mm=self._cfg.lift_qus_to_pole_mm(lift),
        )
