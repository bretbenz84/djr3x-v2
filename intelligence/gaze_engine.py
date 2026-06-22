"""Human-like head-gaze controller for DJ-R3X (3-DOF static-face droid).

R3X has no articulating eyes, so "eye contact" is simulated entirely through head
pose. This module is the pure, hardware-free *brain*: an explicit state machine plus
a stochastic two-state (ON-target / OFF-target) duty-cycle gaze generator. It emits a
:class:`GazeDecision` each tick; actuation happens elsewhere:

* **Live robot** — ``consciousness._step_face_tracking`` consumes the decision and
  composes it with the existing 12.5 Hz closed-loop face-tracking (ON-target = let the
  camera-on-head visual servo centre the active speaker; OFF-target = suspend centring
  and drive a relative aversion pose through the one servo writer).
* **Offline** — ``tools/gaze_demo_sim.py`` and the tests drive an abstract
  :class:`intelligence.head_interface.SimHead` from the same decisions.

Behavioural rules encoded (see the per-method docstrings):

* **50 / 70 rule** — P(ON-target) ≈ 0.50 while R3X is SPEAKING, ≈ 0.70 while
  LISTENING; ~0.85 for the first few seconds of an opening, with a POLE lean-in.
* **Markov dwell** — ON/OFF dwell times are sampled per segment (never a fixed
  schedule) so gaze never looks metronomic. ON ~ N(μ(duty), σ) clipped to
  [1.0, 5.0]; OFF ~ N(1.2, 0.5) clipped to [0.4, 2.5].
* **Anti-stare hard cap** — never hold ON-target > 5.0 s (a sustained stare reads as
  a threat); a break is forced.
* **Aversion direction** — people look away to the SIDE or DOWN, essentially never
  up (an up-stare reads as awkward / spacey), so every look-away is yaw-to-the-side
  and/or a downward pitch: a low-load break glances to the side ~level; "thinking" /
  planning a complex reply looks down-and-aside; just-heard material is absorbed with
  a brief down-glance. Pitch on an aversion is hard-clamped to ≤ 0 (never up).
* **Complexity-scaled pre-turn aversion** — just before R3X speaks he looks away,
  longer + further DOWN-and-aside for a complex reply, shorter + to-the-side for a
  simple one (pitch is hard-clamped ≤ 0, so a pre-turn aversion never looks up).
* **Turn-yield return** — at the end of his turn he returns ON-target to hand the
  floor back.
* **Multi-person include-sweep** — while speaking he occasionally sweeps to a
  non-active listener to "include" them, then returns to the active partner.

**Live-actuation caveat (current robot):** the SPEAKING-state duty-cycle (the ~0.50
ON-target rule) and the multi-person include-sweep are exercised by the offline sim and
the tests, but do NOT actuate on the live robot today — ``consciousness._step_face_tracking``
suspends gaze aversions while R3X is speaking, and the live adapter never populates
listener bearings. The PREP_TURN pre-turn look-away and the LISTENING duty-cycle ARE live.

The module imports the project ``config`` for tunables + servo geometry but performs
no I/O and no hardware access, so it is safe to import and unit-test anywhere.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from intelligence.head_interface import HeadPose

_log = logging.getLogger(__name__)


def _cfg(name: str, default):
    try:
        import config as _project_config
        return getattr(_project_config, name, default)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ─────────────────────────────────────────────────────────────────────────────
# States
# ─────────────────────────────────────────────────────────────────────────────


class GazeState(str, Enum):
    """Explicit gaze states.

    Transition triggers (see :meth:`GazeEngine._transition`):

    * IDLE      — no active conversation; engine stands down (existing idle-wander /
                  rest behaviour owns the head).
    * OPENING   — conversation just became active; high ON-duty + POLE lean-in for a
                  few seconds, then -> LISTENING.
    * LISTENING — partner has the floor (or a conversational lull); ON-duty ~0.70.
    * PREP_TURN — armed by ``note_about_to_speak``; complexity-scaled OFF-target
                  aversion just before speech onset, then -> SPEAKING.
    * SPEAKING  — R3X is talking; ON-duty ~0.50 + occasional include-sweep.
    * YIELDING  — speech just ended; return ON-target to signal "your turn", then
                  -> LISTENING.
    * CLOSING   — conversation went idle; end on an OFF-target with POLE lowered,
                  then -> IDLE.
    """

    IDLE = "idle"
    OPENING = "opening"
    LISTENING = "listening"
    PREP_TURN = "prep_turn"
    SPEAKING = "speaking"
    YIELDING = "yielding"
    CLOSING = "closing"


# Aversion "kinds" — the *reason* for a look-away, which picks the pose direction.
# People look away to the SIDE or DOWN, never up, so every kind is yaw-aside / pitch-down.
KIND_NONE = "none"
KIND_SIDE = "side"                 # low-load break: yaw to the side, level-to-slightly-down
KIND_THINKING = "thinking"         # planning a complex reply: look DOWN and aside
KIND_INTERNALIZING = "internalizing"  # absorbing what was heard: brief PITCH DOWN
KIND_INCLUDE_SWEEP = "include_sweep"  # sweep to a listener to include them
KIND_CLOSE = "close"               # disengage: OFF + POLE lowered

# Back-compat alias (the old "visualizing/up" kind is now a down-to-think pose).
KIND_VISUALIZING = KIND_THINKING


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GazeConfig:
    """All gaze tunables + DOF geometry in one place (spec deliverable 4).

    Defaults mirror ``config.GAZE_*`` (the live source of truth, overridable via
    ``user_config`` / ``.env``); :meth:`from_config` builds an instance from them.
    Everything is tunable without touching logic.
    """

    # --- duty cycles (P(ON-target)) -------------------------------------------
    duty_speaking: float = 0.50
    duty_listening: float = 0.70
    duty_opening: float = 0.85
    duty_closing: float = 0.30

    # --- dwell distributions (seconds) ----------------------------------------
    on_dwell_sd: float = 0.8
    on_dwell_min: float = 1.0
    on_hard_cap: float = 5.0          # never hold ON-target longer than this
    off_dwell_mean: float = 1.2
    off_dwell_sd: float = 0.5
    off_dwell_min: float = 0.4
    off_dwell_max: float = 2.5

    # --- phase durations (seconds) --------------------------------------------
    opening_secs: float = 3.0
    closing_secs: float = 2.5
    yield_secs: float = 0.5
    internalize_min_secs: float = 0.4
    internalize_max_secs: float = 0.9   # keep down-glances < 1.0 s (not submissive)
    include_sweep_secs: float = 1.5

    # --- pre-turn aversion (complexity-scaled) --------------------------------
    pre_aversion_min_secs: float = 0.4
    pre_aversion_max_secs: float = 1.4
    pre_aversion_visualize_threshold: float = 0.5   # complexity above this -> PITCH UP

    # --- aversion offset ranges (degrees / mm); aversion pitch is DOWN-only ----
    side_yaw_min_deg: float = 15.0
    side_yaw_max_deg: float = 25.0
    side_pitch_down_max_deg: float = 5.0   # a side break may dip slightly, never look up
    think_pitch_min_deg: float = 8.0       # "look down to think" (applied downward)
    think_pitch_max_deg: float = 16.0
    think_yaw_min_deg: float = 5.0         # small sideways component of a think glance
    think_yaw_max_deg: float = 14.0
    internalize_pitch_min_deg: float = 8.0
    internalize_pitch_max_deg: float = 15.0
    on_target_jitter_deg: float = 2.5     # small noise so ON-target is never robotic

    # --- engagement (POLE) by phase (mm) --------------------------------------
    pole_rest_mm: float = 20.0
    pole_lean_in_mm: float = 45.0
    pole_settle_mm: float = 5.0

    # --- multi-person ----------------------------------------------------------
    include_sweep_prob: float = 0.20      # of SPEAKING ON-segments
    orient_glance_secs: float = 0.6       # new-person orienting glance

    # --- conversation activity thresholds (seconds) ---------------------------
    close_after_idle_secs: float = 12.0

    # --- velocities for sim / open-loop actuation -----------------------------
    saccade_vel_dps: float = 320.0        # fast (but clamped) gaze shift
    smooth_vel_dps: float = 90.0          # gentle on-target settle
    pole_vel_mms: float = 25.0            # posture moves are SLOW

    # --- DOF limits (degrees / mm) --------------------------------------------
    yaw_limit_deg: float = 70.0
    pitch_up_limit_deg: float = 25.0
    pitch_down_limit_deg: float = 20.0
    pole_min_mm: float = 0.0
    pole_max_mm: float = 60.0
    pole_gain_qus_per_mm: float = 22.0    # headlift qus per mm of POLE travel

    # --- servo geometry (quarter-microseconds), filled from SERVO_CHANNELS -----
    neck_min: int = 1984
    neck_max: int = 9984
    neck_neutral: int = 6000
    tilt_min: int = 3904
    tilt_max: int = 5504
    tilt_neutral: int = 4320
    lift_min: int = 1984
    lift_max: int = 7744
    lift_neutral: int = 6000

    @classmethod
    def from_config(cls) -> "GazeConfig":
        inst = cls(
            duty_speaking=float(_cfg("GAZE_DUTY_SPEAKING", cls.duty_speaking)),
            duty_listening=float(_cfg("GAZE_DUTY_LISTENING", cls.duty_listening)),
            duty_opening=float(_cfg("GAZE_DUTY_OPENING", cls.duty_opening)),
            duty_closing=float(_cfg("GAZE_DUTY_CLOSING", cls.duty_closing)),
            on_dwell_sd=float(_cfg("GAZE_ON_DWELL_SD", cls.on_dwell_sd)),
            on_dwell_min=float(_cfg("GAZE_ON_DWELL_MIN", cls.on_dwell_min)),
            on_hard_cap=float(_cfg("GAZE_ON_HARD_CAP_SECS", cls.on_hard_cap)),
            off_dwell_mean=float(_cfg("GAZE_OFF_DWELL_MEAN", cls.off_dwell_mean)),
            off_dwell_sd=float(_cfg("GAZE_OFF_DWELL_SD", cls.off_dwell_sd)),
            off_dwell_min=float(_cfg("GAZE_OFF_DWELL_MIN", cls.off_dwell_min)),
            off_dwell_max=float(_cfg("GAZE_OFF_DWELL_MAX", cls.off_dwell_max)),
            opening_secs=float(_cfg("GAZE_OPENING_SECS", cls.opening_secs)),
            closing_secs=float(_cfg("GAZE_CLOSING_SECS", cls.closing_secs)),
            yield_secs=float(_cfg("GAZE_YIELD_SECS", cls.yield_secs)),
            internalize_min_secs=float(_cfg("GAZE_INTERNALIZE_MIN_SECS", cls.internalize_min_secs)),
            internalize_max_secs=float(_cfg("GAZE_INTERNALIZE_MAX_SECS", cls.internalize_max_secs)),
            include_sweep_secs=float(_cfg("GAZE_INCLUDE_SWEEP_SECS", cls.include_sweep_secs)),
            pre_aversion_min_secs=float(_cfg("GAZE_PRE_AVERSION_MIN_SECS", cls.pre_aversion_min_secs)),
            pre_aversion_max_secs=float(_cfg("GAZE_PRE_AVERSION_MAX_SECS", cls.pre_aversion_max_secs)),
            pre_aversion_visualize_threshold=float(
                _cfg("GAZE_PRE_AVERSION_VISUALIZE_THRESHOLD", cls.pre_aversion_visualize_threshold)
            ),
            side_yaw_min_deg=float(_cfg("GAZE_SIDE_YAW_MIN_DEG", cls.side_yaw_min_deg)),
            side_yaw_max_deg=float(_cfg("GAZE_SIDE_YAW_MAX_DEG", cls.side_yaw_max_deg)),
            side_pitch_down_max_deg=float(_cfg("GAZE_SIDE_PITCH_DOWN_MAX_DEG", cls.side_pitch_down_max_deg)),
            think_pitch_min_deg=float(_cfg("GAZE_THINK_PITCH_MIN_DEG", cls.think_pitch_min_deg)),
            think_pitch_max_deg=float(_cfg("GAZE_THINK_PITCH_MAX_DEG", cls.think_pitch_max_deg)),
            think_yaw_min_deg=float(_cfg("GAZE_THINK_YAW_MIN_DEG", cls.think_yaw_min_deg)),
            think_yaw_max_deg=float(_cfg("GAZE_THINK_YAW_MAX_DEG", cls.think_yaw_max_deg)),
            internalize_pitch_min_deg=float(_cfg("GAZE_INTERNALIZE_PITCH_MIN_DEG", cls.internalize_pitch_min_deg)),
            internalize_pitch_max_deg=float(_cfg("GAZE_INTERNALIZE_PITCH_MAX_DEG", cls.internalize_pitch_max_deg)),
            on_target_jitter_deg=float(_cfg("GAZE_ON_TARGET_JITTER_DEG", cls.on_target_jitter_deg)),
            pole_rest_mm=float(_cfg("GAZE_POLE_REST_MM", cls.pole_rest_mm)),
            pole_lean_in_mm=float(_cfg("GAZE_POLE_LEAN_IN_MM", cls.pole_lean_in_mm)),
            pole_settle_mm=float(_cfg("GAZE_POLE_SETTLE_MM", cls.pole_settle_mm)),
            include_sweep_prob=float(_cfg("GAZE_INCLUDE_SWEEP_PROB", cls.include_sweep_prob)),
            orient_glance_secs=float(_cfg("GAZE_ORIENT_GLANCE_SECS", cls.orient_glance_secs)),
            close_after_idle_secs=float(_cfg("GAZE_CLOSE_AFTER_IDLE_SECS", cls.close_after_idle_secs)),
            saccade_vel_dps=float(_cfg("GAZE_SACCADE_VEL_DPS", cls.saccade_vel_dps)),
            smooth_vel_dps=float(_cfg("GAZE_SMOOTH_VEL_DPS", cls.smooth_vel_dps)),
            pole_vel_mms=float(_cfg("GAZE_POLE_VEL_MMS", cls.pole_vel_mms)),
            yaw_limit_deg=float(_cfg("GAZE_YAW_LIMIT_DEG", cls.yaw_limit_deg)),
            pitch_up_limit_deg=float(_cfg("GAZE_PITCH_UP_LIMIT_DEG", cls.pitch_up_limit_deg)),
            pitch_down_limit_deg=float(_cfg("GAZE_PITCH_DOWN_LIMIT_DEG", cls.pitch_down_limit_deg)),
            pole_min_mm=float(_cfg("GAZE_POLE_MIN_MM", cls.pole_min_mm)),
            pole_max_mm=float(_cfg("GAZE_POLE_MAX_MM", cls.pole_max_mm)),
            pole_gain_qus_per_mm=float(_cfg("GAZE_POLE_GAIN_QUS_PER_MM", cls.pole_gain_qus_per_mm)),
        )
        # Pull live servo geometry so the deg/mm<->qus mapping matches the build.
        try:
            chans = _cfg("SERVO_CHANNELS", None)
            if isinstance(chans, dict):
                inst.neck_min = int(chans["neck"]["min"]); inst.neck_max = int(chans["neck"]["max"])
                inst.neck_neutral = int(chans["neck"]["neutral"])
                inst.tilt_min = int(chans["headtilt"]["min"]); inst.tilt_max = int(chans["headtilt"]["max"])
                inst.tilt_neutral = int(chans["headtilt"]["neutral"])
                inst.lift_min = int(chans["headlift"]["min"]); inst.lift_max = int(chans["headlift"]["max"])
                inst.lift_neutral = int(chans["headlift"]["neutral"])
        except Exception:
            pass
        return inst

    # --- duty helper ----------------------------------------------------------
    def duty_for(self, state: GazeState) -> float:
        return {
            GazeState.SPEAKING: self.duty_speaking,
            GazeState.LISTENING: self.duty_listening,
            GazeState.OPENING: self.duty_opening,
            GazeState.CLOSING: self.duty_closing,
        }.get(state, self.duty_listening)

    def pole_for(self, state: GazeState) -> float:
        if state == GazeState.OPENING:
            return self.pole_lean_in_mm
        if state == GazeState.CLOSING:
            return self.pole_settle_mm
        return self.pole_rest_mm

    # --- abstract (deg/mm) <-> Maestro quarter-microseconds -------------------
    # YAW: linear about neck neutral. Half-span maps to the +/- yaw limit.
    def yaw_deg_to_neck_qus(self, deg: float) -> int:
        per_deg = ((self.neck_max - self.neck_min) / 2.0) / max(1e-6, self.yaw_limit_deg)
        return int(round(_clamp(self.neck_neutral + deg * per_deg, self.neck_min, self.neck_max)))

    def neck_qus_to_yaw_deg(self, qus: float) -> float:
        per_deg = ((self.neck_max - self.neck_min) / 2.0) / max(1e-6, self.yaw_limit_deg)
        return (qus - self.neck_neutral) / max(1e-6, per_deg)

    # PITCH: headtilt is INVERTED (up = lower qus) and asymmetric (more down travel).
    def pitch_deg_to_tilt_qus(self, deg: float) -> int:
        if deg >= 0:  # up
            per_deg = (self.tilt_neutral - self.tilt_min) / max(1e-6, self.pitch_up_limit_deg)
            qus = self.tilt_neutral - deg * per_deg
        else:  # down
            per_deg = (self.tilt_max - self.tilt_neutral) / max(1e-6, self.pitch_down_limit_deg)
            qus = self.tilt_neutral + (-deg) * per_deg
        return int(round(_clamp(qus, self.tilt_min, self.tilt_max)))

    def tilt_qus_to_pitch_deg(self, qus: float) -> float:
        if qus <= self.tilt_neutral:  # up
            per_deg = (self.tilt_neutral - self.tilt_min) / max(1e-6, self.pitch_up_limit_deg)
            return (self.tilt_neutral - qus) / max(1e-6, per_deg)
        per_deg = (self.tilt_max - self.tilt_neutral) / max(1e-6, self.pitch_down_limit_deg)
        return -(qus - self.tilt_neutral) / max(1e-6, per_deg)

    # POLE: head height as a headlift offset about neutral, anchored at rest mm.
    def pole_mm_to_lift_qus(self, mm: float) -> int:
        qus = self.lift_neutral + (mm - self.pole_rest_mm) * self.pole_gain_qus_per_mm
        return int(round(_clamp(qus, self.lift_min, self.lift_max)))

    def lift_qus_to_pole_mm(self, qus: float) -> float:
        return self.pole_rest_mm + (qus - self.lift_neutral) / max(1e-6, self.pole_gain_qus_per_mm)

    def pole_bias_qus(self, mm: float) -> int:
        """Headlift *delta* (qus) for a POLE engagement of ``mm`` relative to rest —
        used by the live closed-loop adapter to bias the gaze baseline."""
        return int(round((mm - self.pole_rest_mm) * self.pole_gain_qus_per_mm))


# ─────────────────────────────────────────────────────────────────────────────
# Inputs / output
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GazeInputs:
    """Per-tick world snapshot fed to the engine (built by the live adapter or sim)."""

    now: float
    speaking: bool = False            # R3X is speaking (TTS playing)
    listening: bool = False           # R3X is in the think/processing wait after the user
    conversation_active: bool = False
    conversation_idle_secs: float = 0.0
    num_people: int = 1
    active_speaker_id: Optional[int] = None
    # Abstract partner bearing for sim / logging (the live closed loop ignores the
    # absolute yaw — ON-target there means "centre the face"). Degrees.
    partner_bearing: tuple[float, float] = (0.0, 0.0)
    # [(person_id, yaw_deg)] of non-active listeners, for include-sweeps.
    listener_bearings: list[tuple[Optional[int], float]] = field(default_factory=list)
    new_person: bool = False          # someone just entered frame
    suppressed: bool = False          # manual override / stand-down


@dataclass
class GazeDecision:
    active: bool                       # False => engine stood down; adapter does nothing
    state: GazeState
    mode: str                         # "on_target" | "off_target"
    kind: str                         # one of the KIND_* constants
    pose: HeadPose                    # abstract absolute target (sim / logging)
    yaw_offset_deg: float             # OFF-target deviation from on-target (live: rel. neck)
    pitch_offset_deg: float           # OFF-target pitch (live: absolute about level)
    pole_mm: float                    # engagement / POLE height
    velocity: str                     # "saccade" | "smooth" | "posture"
    center_on: Optional[int]          # person to centre / sweep toward
    reason: str
    segment_id: int

    @property
    def drive(self) -> bool:
        """True when the live adapter should actively command the head (an aversion or
        an include-sweep). ON-target falls through to the existing centring loop."""
        return self.active and (self.mode == "off_target" or self.kind == KIND_INCLUDE_SWEEP)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────


class GazeEngine:
    """Stateful gaze brain. Call :meth:`step` at a fixed rate (e.g. 12.5–50 Hz)."""

    def __init__(self, config: Optional[GazeConfig] = None, seed: Optional[int] = None):
        self.cfg = config or GazeConfig.from_config()
        self.rng = random.Random(seed)
        self._log_hook: Optional[Callable[[dict], None]] = None
        self.reset()

    # --- lifecycle ------------------------------------------------------------
    def reset(self) -> None:
        self.state = GazeState.IDLE
        self._seg_kind = "on"
        self._seg_started = 0.0
        self._seg_until = 0.0
        self._seg_id = 0
        self._seg_sweep_target: Optional[int] = None
        self._on_started = 0.0
        self._prev_speaking = False
        self._prev_listening = False
        self._prev_conv_active = False
        self._prep_active = False
        self._prep_until = 0.0
        self._prep_complexity = 0.0
        self._pending_prep: Optional[float] = None
        self._yield_until = 0.0
        self._internalize_until = 0.0
        self._opening_until = 0.0
        self._closing_until = 0.0
        self._closed = True
        self._last_off_yaw_sign = 0
        self._last_kind = KIND_NONE
        self._last_logged_seg = -1
        self._last_state_logged: Optional[GazeState] = None
        self._last_log_key: Optional[tuple] = None
        # cached aversion pose for the current OFF segment (so it doesn't jitter every tick)
        self._off_yaw = 0.0
        self._off_pitch = 0.0

    def seed(self, value: int) -> None:
        self.rng.seed(value)

    def set_log_hook(self, hook: Optional[Callable[[dict], None]]) -> None:
        self._log_hook = hook

    # --- external events ------------------------------------------------------
    def note_about_to_speak(self, complexity: float) -> None:
        """Arm the complexity-scaled pre-turn aversion (call just before TTS onset).

        ``complexity`` in [0, 1] maps to an aversion duration in
        [pre_aversion_min, pre_aversion_max]; above the visualize threshold it becomes a
        PITCH-UP "looking up to think" pose, otherwise a short to-the-side glance.
        """
        self._pending_prep = _clamp(float(complexity), 0.0, 1.0)

    # --- sampling -------------------------------------------------------------
    def _sample_on(self, duty: float) -> float:
        duty = _clamp(duty, 0.05, 0.95)
        mean = self.cfg.off_dwell_mean * duty / (1.0 - duty)
        val = self.rng.gauss(mean, self.cfg.on_dwell_sd)
        return _clamp(val, self.cfg.on_dwell_min, self.cfg.on_hard_cap)

    def _sample_off(self) -> float:
        val = self.rng.gauss(self.cfg.off_dwell_mean, self.cfg.off_dwell_sd)
        return _clamp(val, self.cfg.off_dwell_min, self.cfg.off_dwell_max)

    def _begin_segment(self, now: float, kind: str, duty: float) -> None:
        self._seg_kind = kind
        self._seg_started = now
        self._seg_sweep_target = None
        self._seg_id += 1
        if kind == "on":
            self._on_started = now
            self._seg_until = now + self._sample_on(duty)
        else:
            self._seg_until = now + self._sample_off()
            self._choose_off_offsets()

    def _advance_segment(self, now: float, state: GazeState, inputs: GazeInputs) -> None:
        duty = self.cfg.duty_for(state)
        expired = now >= self._seg_until
        stared = self._seg_kind == "on" and (now - self._on_started) >= self.cfg.on_hard_cap
        if expired or stared:
            new_kind = "off" if self._seg_kind == "on" else "on"
            self._begin_segment(now, new_kind, duty)
            # On a fresh SPEAKING ON-segment, maybe make it an include-sweep.
            if (
                new_kind == "on"
                and state == GazeState.SPEAKING
                and inputs.num_people > 1
                and inputs.listener_bearings
                and self.rng.random() < self.cfg.include_sweep_prob
            ):
                pid, yaw = self.rng.choice(inputs.listener_bearings)
                self._seg_sweep_target = pid
                self._off_yaw = yaw
                self._seg_until = now + self.cfg.include_sweep_secs

    def _choose_off_offsets(self, kind: str = KIND_SIDE) -> None:
        """Pick the OFF-target offsets once per segment, jittered + non-repeating.

        Aversion pitch is DOWN-only (people look away to the side or down, never up — an
        up-stare reads as awkward), so every branch produces pitch <= 0 and the result
        is hard-clamped to <= 0 as a backstop."""
        # Alternate yaw side from the last aversion so we never snap to the same pose twice.
        sign = -1 if self._last_off_yaw_sign >= 0 else 1
        if kind == KIND_THINKING:
            # "Look down to think" — a downward glance with a small sideways component.
            yaw = sign * self.rng.uniform(self.cfg.think_yaw_min_deg, self.cfg.think_yaw_max_deg)
            pitch = -self.rng.uniform(self.cfg.think_pitch_min_deg, self.cfg.think_pitch_max_deg)
        elif kind == KIND_INTERNALIZING:
            yaw = sign * self.rng.uniform(0.0, self.cfg.think_yaw_min_deg)
            pitch = -self.rng.uniform(self.cfg.internalize_pitch_min_deg, self.cfg.internalize_pitch_max_deg)
        else:  # KIND_SIDE / default low-load break: mostly horizontal, maybe a slight dip
            yaw = sign * self.rng.uniform(self.cfg.side_yaw_min_deg, self.cfg.side_yaw_max_deg)
            pitch = -self.rng.uniform(0.0, self.cfg.side_pitch_down_max_deg)
        self._off_yaw = yaw
        self._off_pitch = min(0.0, pitch)  # backstop: an aversion never looks up
        self._last_off_yaw_sign = sign

    # --- transitions ----------------------------------------------------------
    def _enter_prep_turn(self, now: float, complexity: float) -> None:
        self._prep_active = True
        self._prep_complexity = complexity
        dur = self.cfg.pre_aversion_min_secs + complexity * (
            self.cfg.pre_aversion_max_secs - self.cfg.pre_aversion_min_secs
        )
        self._prep_until = now + dur
        kind = KIND_THINKING if complexity >= self.cfg.pre_aversion_visualize_threshold else KIND_SIDE
        self._choose_off_offsets(kind)
        self._prep_kind = kind

    def _transition(self, now: float, inp: GazeInputs) -> None:
        speaking, listening, conv = inp.speaking, inp.listening, inp.conversation_active

        # consume a pending pre-turn arm (only if not already speaking)
        if self._pending_prep is not None:
            if not speaking:
                self._enter_prep_turn(now, self._pending_prep)
            self._pending_prep = None

        # 1. PREP_TURN holds until speech starts or it elapses.
        if self._prep_active:
            if speaking:
                self._prep_active = False
            elif now < self._prep_until:
                self.state = GazeState.PREP_TURN
                return
            else:
                self._prep_active = False

        # 2. SPEAKING.
        if speaking:
            if self.state != GazeState.SPEAKING:
                self.state = GazeState.SPEAKING
                self._begin_segment(now, "on", self.cfg.duty_for(GazeState.SPEAKING))
            return

        # 3. Just stopped speaking -> YIELDING (hand the floor back, ON-target).
        if self._prev_speaking and not speaking:
            self.state = GazeState.YIELDING
            self._yield_until = now + self.cfg.yield_secs
            return
        if self.state == GazeState.YIELDING and now < self._yield_until:
            return

        # 4. Think/processing wait after the user -> brief internalizing, then LISTENING.
        if listening:
            if not self._prev_listening:
                self._internalize_until = now + self.rng.uniform(
                    self.cfg.internalize_min_secs, self.cfg.internalize_max_secs
                )
                self._choose_off_offsets(KIND_INTERNALIZING)
            if self.state != GazeState.LISTENING:
                self.state = GazeState.LISTENING
                self._begin_segment(now, "on", self.cfg.duty_for(GazeState.LISTENING))
            return

        # 5. Conversation active: OPENING (fresh) then LISTENING.
        if conv:
            if not self._prev_conv_active:
                self._opening_until = now + self.cfg.opening_secs
                self._closed = False
            if now < self._opening_until:
                if self.state != GazeState.OPENING:
                    self.state = GazeState.OPENING
                    self._begin_segment(now, "on", self.cfg.duty_for(GazeState.OPENING))
                return
            if self.state not in (GazeState.LISTENING,):
                self.state = GazeState.LISTENING
                self._begin_segment(now, "on", self.cfg.duty_for(GazeState.LISTENING))
            return

        # 6. Conversation went inactive -> CLOSING once, then IDLE.
        if self._prev_conv_active and not conv and not self._closed:
            self.state = GazeState.CLOSING
            self._closing_until = now + self.cfg.closing_secs
            self._closed = True
            return
        if self.state == GazeState.CLOSING and now < self._closing_until:
            return

        self.state = GazeState.IDLE

    # --- decision -------------------------------------------------------------
    def step(self, inputs: GazeInputs) -> GazeDecision:
        """Advance the state machine and return the gaze decision for this tick."""
        now = inputs.now
        if inputs.suppressed:
            self.state = GazeState.IDLE
            self._prev_speaking = inputs.speaking
            self._prev_listening = inputs.listening
            self._prev_conv_active = inputs.conversation_active
            return self._inactive(now)

        self._transition(now, inputs)

        if self.state == GazeState.IDLE:
            decision = self._inactive(now)
        else:
            decision = self._decide(now, inputs)

        self._prev_speaking = inputs.speaking
        self._prev_listening = inputs.listening
        self._prev_conv_active = inputs.conversation_active
        self._maybe_log(decision, now)
        return decision

    def _inactive(self, now: float) -> GazeDecision:
        return GazeDecision(
            active=False, state=GazeState.IDLE, mode="on_target", kind=KIND_NONE,
            pose=HeadPose(0.0, 0.0, self.cfg.pole_rest_mm),
            yaw_offset_deg=0.0, pitch_offset_deg=0.0, pole_mm=self.cfg.pole_rest_mm,
            velocity="smooth", center_on=None, reason="idle / stood down",
            segment_id=self._seg_id,
        )

    def _on_target_pose(self, inp: GazeInputs, pole_mm: float) -> HeadPose:
        by, bp = inp.partner_bearing
        jit = self.cfg.on_target_jitter_deg
        return HeadPose(
            yaw_deg=by + self.rng.uniform(-jit, jit),
            pitch_deg=bp + self.rng.uniform(-jit, jit),
            pole_mm=pole_mm,
        )

    def _decide(self, now: float, inp: GazeInputs) -> GazeDecision:
        state = self.state
        by, bp = inp.partner_bearing

        # PREP_TURN — complexity-scaled pre-turn aversion.
        if state == GazeState.PREP_TURN:
            kind = getattr(self, "_prep_kind", KIND_SIDE)
            pose = HeadPose(by + self._off_yaw, self._off_pitch, self.cfg.pole_rest_mm)
            reason = (
                f"pre-turn aversion ({'down-to-think' if kind == KIND_THINKING else 'to-the-side'}, "
                f"complexity={self._prep_complexity:.2f})"
            )
            return self._mk(state, "off_target", kind, pose, self._off_yaw, self._off_pitch,
                            self.cfg.pole_rest_mm, "saccade", None, reason)

        # YIELDING — return on-target to hand the floor back.
        if state == GazeState.YIELDING:
            pose = self._on_target_pose(inp, self.cfg.pole_rest_mm)
            return self._mk(state, "on_target", KIND_NONE, pose, 0.0, 0.0,
                            self.cfg.pole_rest_mm, "smooth", inp.active_speaker_id,
                            "turn-yield: returning gaze to hand over the floor")

        # CLOSING — disengage: OFF-target + POLE lowered.
        if state == GazeState.CLOSING:
            if self._last_kind != KIND_CLOSE:
                self._choose_off_offsets(KIND_SIDE)
            pose = HeadPose(by + self._off_yaw, self._off_pitch, self.cfg.pole_settle_mm)
            return self._mk(state, "off_target", KIND_CLOSE, pose, self._off_yaw, self._off_pitch,
                            self.cfg.pole_settle_mm, "smooth", None,
                            "closing: disengaging, settling the pole")

        # LISTENING internalizing window (brief down-glance right after the user).
        if state == GazeState.LISTENING and now < self._internalize_until:
            pose = HeadPose(by + self._off_yaw, self._off_pitch, self.cfg.pole_rest_mm)
            return self._mk(state, "off_target", KIND_INTERNALIZING, pose, self._off_yaw,
                            self._off_pitch, self.cfg.pole_rest_mm, "saccade", None,
                            "internalizing: brief down-glance to absorb what was said")

        # Duty states: OPENING / LISTENING / SPEAKING.
        self._advance_segment(now, state, inp)
        pole = self.cfg.pole_for(state)
        if self._seg_kind == "on":
            if self._seg_sweep_target is not None:
                # include-sweep: look at a non-active listener, then return next segment.
                pose = HeadPose(self._off_yaw, 0.0, pole)
                return self._mk(state, "on_target", KIND_INCLUDE_SWEEP, pose, self._off_yaw, 0.0,
                                pole, "saccade", self._seg_sweep_target,
                                "include-sweep: glancing at a listener to include them")
            pose = self._on_target_pose(inp, pole)
            return self._mk(state, "on_target", KIND_NONE, pose, 0.0, 0.0, pole, "smooth",
                            inp.active_speaker_id, f"on-target ({state.value}, duty)")
        # OFF segment — a low-load break (yaw to the side, level-to-slightly-down).
        kind = self._last_kind if self._last_kind in (KIND_SIDE, KIND_THINKING) else KIND_SIDE
        pose = HeadPose(by + self._off_yaw, self._off_pitch, pole)
        return self._mk(state, "off_target", KIND_SIDE, pose, self._off_yaw, self._off_pitch,
                        pole, "saccade", None, f"off-target break ({state.value})")

    def _mk(self, state, mode, kind, pose, yoff, poff, pole, vel, center, reason) -> GazeDecision:
        self._last_kind = kind
        return GazeDecision(
            active=True, state=state, mode=mode, kind=kind, pose=pose,
            yaw_offset_deg=yoff, pitch_offset_deg=poff, pole_mm=pole,
            velocity=vel, center_on=center, reason=reason, segment_id=self._seg_id,
        )

    # --- logging --------------------------------------------------------------
    def _maybe_log(self, d: GazeDecision, now: float) -> None:
        """Emit (timestamp, state, target_pose, reason) once per segment / state / kind
        change (so brief beats like the internalizing down-glance are visible)."""
        key = (d.segment_id, d.state, d.kind)
        if key == self._last_log_key:
            return
        self._last_log_key = key
        self._last_logged_seg = d.segment_id
        self._last_state_logged = d.state
        record = {
            "t": round(now, 3),
            "state": d.state.value,
            "mode": d.mode,
            "kind": d.kind,
            "pose": (round(d.pose.yaw_deg, 1), round(d.pose.pitch_deg, 1), round(d.pose.pole_mm, 1)),
            "reason": d.reason,
        }
        if self._log_hook is not None:
            try:
                self._log_hook(record)
            except Exception:
                pass
        else:
            _log.debug("[gaze] %s", record)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton + helpers (used by the live integration)
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[GazeEngine] = None


def get_engine() -> GazeEngine:
    global _engine
    if _engine is None:
        _engine = GazeEngine()
    return _engine


def enabled() -> bool:
    return bool(_cfg("GAZE_ENGINE_ENABLED", True))


def under_test_runner() -> bool:
    """Inert under unittest/pytest unless explicitly opted in. The live actuation
    (consciousness._maybe_drive_gaze) is stateful and would otherwise perturb the
    face-tracking unit tests via the module singleton (the callback/rex_pov idiom).
    The pure engine + sim are exercised directly by the gaze tests regardless."""
    if os.environ.get("DJR3X_GAZE_TEST_OPT_IN"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    argv0 = (sys.argv[0] if sys.argv else "").lower()
    return "unittest" in argv0 or "pytest" in argv0 or "py.test" in argv0


def reset() -> None:
    """Reset the live engine's runtime state (e.g. on session end)."""
    if _engine is not None:
        _engine.reset()


def step(inputs: GazeInputs) -> GazeDecision:
    return get_engine().step(inputs)


def note_about_to_speak(text: Optional[str] = None, complexity: Optional[float] = None) -> None:
    """Arm the complexity-scaled pre-turn aversion. If ``complexity`` is omitted it is
    estimated from ``text`` via a word-count proxy (spec fallback)."""
    if not enabled():
        return
    if complexity is None:
        complexity = complexity_from_text(text or "")
    try:
        get_engine().note_about_to_speak(complexity)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("[gaze] note_about_to_speak failed: %s", exc)


def complexity_from_text(text: str) -> float:
    """Token-count proxy for reply complexity in [0, 1] (used when no LLM tag is
    available). ~6 words -> simple (0.0); ~40+ words -> complex (1.0)."""
    words = len((text or "").split())
    lo, hi = 6.0, 40.0
    return _clamp((words - lo) / (hi - lo), 0.0, 1.0)
