"""Two-servo head, poker elbow/wrist and inverted-visor geometry (October 10, 2026 rework).

Pure arithmetic over Pololu quarter-microseconds: no serial, no threads, no state. The
calibrated numbers are the owner's Pololu Maestro Control Center readings (microseconds
x 4); the model that joins them is the one the V3 program verified against all seven
head readings. Keep this module importable from tests and tools without the hardware.

THE HEAD. The single head-tilt servo is gone. Two servos in the head face each other,
each with a linkage onto one centre post: ``head_left`` (channel 2) and ``head_right``
(channel 11). Driven against each other they PITCH the head up or down; driven the same
way they shift the post sideways and COCK (tilt) the head to one side. Every head command
must set both channels, so callers think in two fractions and let :func:`head_pulses`
produce the pair:

* ``pitch``: -1 fully down, 0 level, +1 fully up (fractions of the measured travel).
* ``tilt``:  -1 cocked to Rex's own left, 0 straight, +1 to his right.

Degrees are still V2's old estimate (about 20 down, 25 up): unverified on this linkage.
Confirmed on the robot: ``tilt < 0`` cocks the head to Rex's own left.

THE POKER ARM gained an elbow (12) and a wrist (13) beside the old swing on channel 6.
The elbow can rise fully only while the wrist is at or below 1564 us; curled further in
the wrist meets the arm and the elbow's ceiling falls. Pairs above the measured ceiling
are refused, and every move between two clear poses is staged one joint at a time
(:func:`poker_route`) so the ceiling is never crossed on the way.

THE VISOR (3) now runs inverted: low pulses raise it, high pulses lower it. Its neutral open
point is 1433 us and surprise 1136 us (owner, 2026-10-11); the other visor poses derive from
the neutral in config. The pulse where it starts to hide the camera is still unread.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import config

# ── Channels ──────────────────────────────────────────────────────────────────
HEAD_LEFT_CH = 2
HEAD_RIGHT_CH = 11
HEAD_PAIR = (HEAD_LEFT_CH, HEAD_RIGHT_CH)        # Always commanded together.
POKER_ELBOW_CH = 12
POKER_WRIST_CH = 13
POKER_PAIR = (POKER_ELBOW_CH, POKER_WRIST_CH)    # Always commanded together (clearance).
VISOR_CH = 3

# ── Two-servo head: console readings October 10 (left, right) ─────────────────
HEAD_LEVEL = (5832, 6480)        # 1458 / 1620 us
HEAD_FULL_UP = (8128, 5248)      # 2032 / 1312 us: the left servo rises as the right falls.
HEAD_FULL_DOWN = (4052, 7616)    # 1013 / 1904 us
# A full side tilt moves both servos the same way by about this much in total (the four
# measured tilted poses moved 828-1492, each with one servo already at its limit). What
# one servo cannot take within its limits, the other takes.
TILT_SHIFT = 1100
HEAD_ENVELOPE = 600              # A pair further than this from any pitch/tilt is refused.
PITCH_DOWN_DEG, PITCH_UP_DEG = 20.0, 25.0

# The old single tilt servo (3904-5504, neutral 4320, low = up) is gone, but many tuning
# knobs are still written in its quarter-microseconds: 1184 of travel covered the full
# "down", 416 the full "up". These convert such a knob into a pitch fraction so the knob
# keeps meaning what it meant.
LEGACY_TILT_DOWN_QUS = 1184
LEGACY_TILT_UP_QUS = 416


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _limits(channel: int) -> tuple[int, int]:
    for cfg in config.SERVO_CHANNELS.values():
        if int(cfg["ch"]) == channel:
            return int(cfg["min"]), int(cfg["max"])
    raise KeyError(f"channel {channel} is not in SERVO_CHANNELS")


def head_limits() -> dict[int, tuple[int, int]]:
    return {ch: _limits(ch) for ch in HEAD_PAIR}


def _pitch_pulses(pitch: float) -> tuple[float, float]:
    left, right = HEAD_LEVEL
    if pitch >= 0:
        return (left + pitch * (HEAD_FULL_UP[0] - left), right + pitch * (HEAD_FULL_UP[1] - right))
    return (left - pitch * (HEAD_FULL_DOWN[0] - left), right - pitch * (HEAD_FULL_DOWN[1] - right))


def head_pulses(pitch: float, tilt: float = 0.0) -> dict[int, int]:
    """Pulses for the two head servos from a pitch and a tilt (see the module docstring).

    Pitch moves the servos against each other along the measured level/up/down line (the
    travel is asymmetric, hence two branches). Tilting right lowers both pulses by a total
    of |tilt| x TILT_SHIFT, each servo taking half within its limits and the other taking
    whatever one cannot; tilting left raises both. This reproduces all seven console
    readings exactly."""
    pitch = _clamp(float(pitch), -1.0, 1.0)
    tilt = _clamp(float(tilt), -1.0, 1.0)
    left, right = _pitch_pulses(pitch)
    limits = head_limits()
    sign = -1 if tilt > 0 else 1
    shift = abs(tilt) * TILT_SHIFT
    room = [
        (limits[HEAD_LEFT_CH][1] - left) if sign > 0 else (left - limits[HEAD_LEFT_CH][0]),
        (limits[HEAD_RIGHT_CH][1] - right) if sign > 0 else (right - limits[HEAD_RIGHT_CH][0]),
    ]
    take = [min(shift / 2, room[0]), min(shift / 2, room[1])]
    take[0] += min(shift - sum(take), room[0] - take[0])
    take[1] += min(shift - sum(take), room[1] - take[1])
    return {
        HEAD_LEFT_CH: int(round(left + sign * take[0])),
        HEAD_RIGHT_CH: int(round(right + sign * take[1])),
    }


def _fit_head(pulses: Mapping[int, int]) -> tuple[float, float, int]:
    """(pitch, tilt, pulse error) that best explain a pair of head pulses."""
    left, right = int(pulses[HEAD_LEFT_CH]), int(pulses[HEAD_RIGHT_CH])

    def score(pitch: float) -> tuple[int, float, float]:
        base = _pitch_pulses(pitch)
        tilt = _clamp(-((left - base[0]) + (right - base[1])) / TILT_SHIFT, -1.0, 1.0)
        got = head_pulses(pitch, tilt)
        return abs(got[HEAD_LEFT_CH] - left) + abs(got[HEAD_RIGHT_CH] - right), pitch, tilt

    best = min(score(i / 100.0) for i in range(-100, 101))
    best = min(best, *(score(_clamp(best[1] + i / 1000.0, -1.0, 1.0)) for i in range(-10, 11)))
    return best[1], best[2], best[0]


def head_pose(pulses: Mapping[int, int]) -> tuple[float, float]:
    """The (pitch, tilt) a pair of head pulses stands for.

    Where one servo sits at its limit the decomposition is ambiguous (fully down tilted
    right also reads as a lesser pitch with more tilt); the pose itself is still exact,
    only the reading of it is not."""
    pitch, tilt, _ = _fit_head(pulses)
    return pitch, tilt


def head_fit_error(pulses: Mapping[int, int]) -> int:
    return _fit_head(pulses)[2]


def head_pair_complete(targets: Iterable[int]) -> bool:
    """False when exactly one of the two head channels is named."""
    present = set(HEAD_PAIR).intersection(int(ch) for ch in targets)
    return not present or present == set(HEAD_PAIR)


def validate_head_pair(targets: Mapping[int, int]) -> None:
    """Refuse a write that sets one head channel without the other, or a pair that no
    pitch/tilt within range explains to within HEAD_ENVELOPE quarter-microseconds. This
    keeps the linkage out of corners it never saw, such as both servos at their highest."""
    if not head_pair_complete(targets):
        raise ValueError("IncompleteHeadPose: the two head servos (2 and 11) must be set together")
    if set(HEAD_PAIR).issubset(int(ch) for ch in targets):
        limits = head_limits()
        for ch in HEAD_PAIR:
            low, high = limits[ch]
            if not low <= int(targets[ch]) <= high:
                raise ValueError(f"HeadPulseOutsideLimits: ch{ch}={targets[ch]} not in {low}-{high}")
        error = head_fit_error(targets)
        if error > HEAD_ENVELOPE:
            raise ValueError(
                f"HeadPairOutsideEnvelope: ch2={targets[HEAD_LEFT_CH]} ch11={targets[HEAD_RIGHT_CH]} "
                f"is {error} q-us from any pitch/tilt"
            )


def pitch_from_degrees(degrees: float) -> float:
    """Pitch fraction for degrees above level (negative looks down)."""
    return _clamp(degrees / (PITCH_UP_DEG if degrees >= 0 else PITCH_DOWN_DEG), -1.0, 1.0)


def degrees_from_pitch(pitch: float) -> float:
    return pitch * (PITCH_UP_DEG if pitch >= 0 else PITCH_DOWN_DEG)


def pitch_from_legacy_delta(delta_qus: float) -> float:
    """A head-tilt offset in the OLD servo's quarter-microseconds (positive = down) as a
    pitch fraction (positive = up)."""
    delta = float(delta_qus)
    if delta >= 0:
        return -_clamp(delta / LEGACY_TILT_DOWN_QUS, 0.0, 1.0)
    return _clamp(-delta / LEGACY_TILT_UP_QUS, 0.0, 1.0)


def legacy_delta_from_pitch(pitch: float) -> int:
    """Inverse of :func:`pitch_from_legacy_delta` (positive = down)."""
    pitch = _clamp(float(pitch), -1.0, 1.0)
    if pitch <= 0:
        return int(round(-pitch * LEGACY_TILT_DOWN_QUS))
    return int(round(-pitch * LEGACY_TILT_UP_QUS))


def head_targets(spec: Mapping[str, float], current: Mapping[int, int]) -> dict[int, int]:
    """Pulses for a head pose given as lift/pitch/tilt, keeping whatever it leaves out."""
    pitch, tilt = head_pose(current)
    targets = head_pulses(spec.get("pitch", pitch), spec.get("tilt", tilt))
    if "lift" in spec:
        targets[1] = int(spec["lift"])
    return targets


# A head move inside these pulse deltas (about 5 degrees) leaves the camera view
# effectively unchanged: it does not restart scene settling or arrival tracking.
HEAD_SMALL = {0: 400, 1: 180, HEAD_LEFT_CH: 450, HEAD_RIGHT_CH: 300}
PITCH_SMALL, TILT_SMALL = 0.12, 0.12


def head_reframes(current: Mapping[int, int], targets: Mapping[int, int]) -> bool:
    """Whether a head move is big enough to count as a new camera view. Both head
    servos count."""
    return any(
        ch in current and abs(int(current[ch]) - int(value)) > HEAD_SMALL[ch]
        for ch, value in targets.items() if ch in HEAD_SMALL
    )


# ── Named head steps (requests and Rex's own plans) ───────────────────────────
NECK_CENTER = 5472
NECK_LOOK_RIGHT = 8000                           # The 33-degree side pose.
NECK_LOOK_LEFT = 2 * NECK_CENTER - NECK_LOOK_RIGHT  # 2944, mirrored about the centre.
LOOK_PITCH, LOOK_TILT = 0.8, 0.8
HEAD_POSES: dict[str, dict[str, float]] = {
    "look_down": {"lift": 3200, "pitch": -LOOK_PITCH},
    "look_up": {"lift": 4200, "pitch": LOOK_PITCH},
    "look_level": {"lift": 3600, "pitch": 0.0, "tilt": 0.0},
    "tilt_left": {"tilt": -LOOK_TILT},
    "tilt_right": {"tilt": LOOK_TILT},
    "tilt_straight": {"tilt": 0.0},
}


def neck_look_target(side: str, current_neck: int, further: bool = True) -> int:
    """look_left / look_right from the current neck: from anywhere nearer the centre the
    33-degree pose; already there or beyond on that side, a person asking again wants more
    (the neck's limit), while Rex's own repeated step moves nothing.

    First live run October 10: Rex was following the owner at the neck's left limit when
    asked to "look left", and the fixed 33-degree pose swung his head back to the right."""
    sign = -1 if side == "left" else 1
    pose = NECK_LOOK_LEFT if sign < 0 else NECK_LOOK_RIGHT
    low, high = _limits(0)
    neck = int(current_neck)
    if sign * (neck - pose) >= -2:           # Already at the pose or beyond it on that side.
        return (high if sign > 0 else low) if further else neck
    return pose


# ── Head expressions on the two-servo head ────────────────────────────────────
# Owner, October 10 evening: "dogs cock their heads left and right when they're perplexed";
# use the new freedom for more expression. Each is a finite run of (pitch change, tilt
# change, dwell seconds) from where the head is, then back to it exactly; "subtle" halves
# the amounts. 'side' expressions cock away from any lean already there.
HEAD_EXPRESSIONS: dict[str, dict] = {
    "nod": {"moves": ((-0.3, 0.0, 0.25), (0.1, 0.0, 0.25)), "speed": 35, "acceleration": 6},
    "tilt": {"moves": ((0.0, 0.6, 0.5),), "speed": 20, "acceleration": 6, "side": True},
    "perplexed": {"moves": ((-0.08, 0.6, 0.7), (-0.08, -0.6, 0.7)), "speed": 25, "acceleration": 6,
                  "side": True},
    "wiggle": {"moves": ((0.0, 0.35, 0.12), (0.0, -0.35, 0.12), (0.0, 0.35, 0.12)), "speed": 50,
               "acceleration": 10},
    "sympathetic": {"moves": ((-0.15, 0.45, 1.2),), "speed": 12, "acceleration": 3, "side": True},
}
# Which feeling fits which, for the conversation model (it chooses; events are never
# mapped to an expression, and most lines should carry none).
HEAD_EXPRESSION_GUIDE: dict[str, str] = {
    "nod": "a small agreeing dip: yes, go on, I follow",
    "tilt": "one cock of the head and back: a question, mild interest",
    "perplexed": "the dog's double head-cock, one side then the other: something does not add up, "
                 "a word not caught",
    "wiggle": "three quick little cocks side to side: playful, pleased, teasing",
    "sympathetic": "one slow cock held with the head a touch down: sympathy, tenderness, bad news "
                   "received gently",
}


def expression_steps(name: str, current: Mapping[int, int], *, subtle: bool = False) -> list[dict]:
    """The pulse pairs an expression visits from ``current``, then the exact return.

    Each step is ``{"targets": {2: .., 11: ..}, "dwell": s, "speed": n, "acceleration": n}``.
    A sided expression cocks away from any lean the head already has."""
    if name not in HEAD_EXPRESSIONS:
        raise ValueError(f"UnknownHeadExpression: {name}")
    spec = HEAD_EXPRESSIONS[name]
    scale = 0.5 if subtle else 1.0
    pitch0, tilt0 = head_pose(current)
    side = (-1.0 if tilt0 > 0.2 else 1.0) if spec.get("side") else 1.0
    steps = [
        {
            "targets": head_pulses(pitch0 + dp * scale, tilt0 + side * dt * scale),
            "dwell": float(dwell),
            "speed": int(spec["speed"]),
            "acceleration": int(spec.get("acceleration", 6)),
        }
        for dp, dt, dwell in spec["moves"]
    ]
    steps.append({
        "targets": {ch: int(current[ch]) for ch in HEAD_PAIR},
        "dwell": 0.0,
        "speed": int(spec["speed"]),
        "acceleration": int(spec.get("acceleration", 6)),
    })
    return steps


# ── Poker elbow (12) and wrist (13): console readings October 10 ──────────────
POKER_ELBOW_DOWN, POKER_ELBOW_UP = 3456, 7680                 # 864 / 1920 us
POKER_WRIST_DOWN, POKER_WRIST_STRAIGHT, POKER_WRIST_IN = 1984, 5760, 9944   # 496 / 1440 / 2486 us
# (wrist, highest elbow allowed) measured pairs; linear between them. The elbow rises
# fully only while the wrist is at or below 1564 us.
POKER_ELBOW_CEILING = ((6256, 7680), (7576, 7228), (8544, 6836), (9984, 6388))
POKER_PARK = {POKER_ELBOW_CH: POKER_ELBOW_UP, POKER_WRIST_CH: 6256}   # 1920 / 1564 us: where the arm
                                                                       # stands when switched on.
POKER_UP = {POKER_ELBOW_CH: 6000, POKER_WRIST_CH: 3768}        # 45 degrees up, hand level (1500 / 942 us).
POKER_OUT = {POKER_ELBOW_CH: 4228, POKER_WRIST_CH: 5556}       # Straight out (1057 / 1389 us).
POKER_DOWN = {POKER_ELBOW_CH: POKER_ELBOW_DOWN, POKER_WRIST_CH: POKER_WRIST_STRAIGHT}
POKER_IDLE = {POKER_ELBOW_CH: (5200, 6800), POKER_WRIST_CH: (2900, 4700)}   # Around the awake pose.
POKER_POSES = {"poker_park": POKER_PARK, "poker_up": POKER_UP, "poker_out": POKER_OUT,
               "poker_down": POKER_DOWN}


def poker_elbow_max(wrist: int) -> int:
    wrist = int(wrist)
    if wrist <= POKER_ELBOW_CEILING[0][0]:
        return POKER_ELBOW_UP
    for (w0, e0), (w1, e1) in zip(POKER_ELBOW_CEILING, POKER_ELBOW_CEILING[1:]):
        if wrist <= w1:
            return int(round(e0 + (e1 - e0) * (wrist - w0) / (w1 - w0)))
    return POKER_ELBOW_CEILING[-1][1]


def poker_clear(pose: Mapping[int, int]) -> bool:
    return int(pose[POKER_ELBOW_CH]) <= poker_elbow_max(pose[POKER_WRIST_CH]) + 2


def poker_pair_complete(targets: Iterable[int]) -> bool:
    present = set(POKER_PAIR).intersection(int(ch) for ch in targets)
    return not present or present == set(POKER_PAIR)


def validate_poker_pair(targets: Mapping[int, int]) -> None:
    """Refuse a lone elbow or wrist, a pulse outside the limits, or a pair above the
    measured ceiling, before anything moves."""
    if not poker_pair_complete(targets):
        raise ValueError("IncompletePokerPose: the poker elbow (12) and wrist (13) must be set together")
    if set(POKER_PAIR).issubset(int(ch) for ch in targets):
        for ch in POKER_PAIR:
            low, high = _limits(ch)
            if not low <= int(targets[ch]) <= high:
                raise ValueError(f"PokerPulseOutsideLimits: ch{ch}={targets[ch]} not in {low}-{high}")
        if not poker_clear(targets):
            raise ValueError(
                f"PokerElbowWristCollision: elbow {targets[POKER_ELBOW_CH]} is above the "
                f"{poker_elbow_max(targets[POKER_WRIST_CH])} ceiling for wrist {targets[POKER_WRIST_CH]}"
            )


def poker_route(start: Mapping[int, int], target: Mapping[int, int]) -> list[dict[int, int]]:
    """Stages between two clear poker poses that stay clear throughout: the elbow drops to
    the lower of the two first, then the wrist moves, then the elbow rises. Each stage moves
    one joint, so the ceiling (which only falls as the wrist curls in) is never crossed."""
    low = min(int(start[POKER_ELBOW_CH]), int(target[POKER_ELBOW_CH]))
    route: list[dict[int, int]] = []
    previous = {ch: int(start[ch]) for ch in POKER_PAIR}
    for stage in (
        {POKER_ELBOW_CH: low, POKER_WRIST_CH: int(start[POKER_WRIST_CH])},
        {POKER_ELBOW_CH: low, POKER_WRIST_CH: int(target[POKER_WRIST_CH])},
        {ch: int(target[ch]) for ch in POKER_PAIR},
    ):
        if stage != previous:
            route.append(stage)
            previous = stage
    return route or [dict(previous)]


# ── Visor (3), inverted: low pulses up, high pulses down ──────────────────────
# Read from config, where they derive from the owner's neutral open point (1433 us) and
# surprise position (1136 us); see config.VISOR_NEUTRAL_QUS. The camera-clear edge is
# still a guess until the camera-hiding pulse is read.
VISOR_UP = int(config.VISOR_UP_QUS)                       # Awake, open: startup target and idle centre.
VISOR_DOWN = int(config.VISOR_DOWN_QUS)                   # Down / park.
VISOR_SPEECH = tuple(int(v) for v in config.VISOR_SPEECH_RANGE_QUS)   # Expressive range while speaking.
VISOR_IDLE = tuple(int(v) for v in config.SERVO_IDLE_VISOR_RANGE_QUS)  # Idle drift; half-blinks dip to VISOR_SPEECH[1].
VISOR_CAMERA_CLEAR_MAX = int(config.VISOR_CAMERA_CLEAR_MAX_QUS)        # Lowest the visor may sit while the camera sees.
VISOR_WIDE = int(config.VISOR_SURPRISE_QUS)               # Thrown fully up: surprise (1136 us).


def visor_frac(pulse: int) -> float:
    """0 = fully down, 1 = fully up, over the configured travel."""
    low, high = _limits(VISOR_CH)
    if high <= low:
        return 0.5
    return _clamp((high - int(pulse)) / (high - low), 0.0, 1.0)


def visor_pulse(frac: float) -> int:
    low, high = _limits(VISOR_CH)
    return int(round(high - _clamp(float(frac), 0.0, 1.0) * (high - low)))


# ── Model-facing joint percentages ────────────────────────────────────────────
# name -> (channel or head axis, value at 0 %, value at 100 %). The head is pitch (0 down,
# 50 level, 100 up) and tilt (0 left, 50 straight, 100 right), never a raw servo; the
# visor is 0 down, 100 up.
def joint_ranges() -> dict[str, tuple]:
    ranges: dict[str, tuple] = {
        "neck": (0, *_limits(0)),
        "head_lift": (1, *_limits(1)),
        "head_pitch": ("pitch", -1.0, 1.0),
        "head_tilt": ("tilt", -1.0, 1.0),
        "visor": (VISOR_CH, _limits(VISOR_CH)[1], _limits(VISOR_CH)[0]),
        "hero_elbow": (4, *_limits(4)),
        "hero_wrist": (5, *_limits(5)),
        "poker_arm": (6, *_limits(6)),
        "hero_arm": (7, *_limits(7)),
        "poker_elbow": (POKER_ELBOW_CH, POKER_ELBOW_DOWN, POKER_ELBOW_UP),
        "poker_wrist": (POKER_WRIST_CH, POKER_WRIST_DOWN, POKER_WRIST_IN),
    }
    return ranges


def joint_percents(commanded: Mapping[int, int]) -> dict[str, int]:
    """Where each joint is, as percentages, with the head as pitch/tilt."""
    head = head_pose(commanded) if set(HEAD_PAIR).issubset(commanded) else None
    result: dict[str, int] = {}
    for name, (channel, low, high) in joint_ranges().items():
        if isinstance(channel, str):
            if head is None:
                continue
            value = head[0] if channel == "pitch" else head[1]
        elif channel in commanded:
            value = commanded[channel]
        else:
            continue
        result[name] = int(round(100 * (float(value) - low) / (high - low)))
    return result


def describe_short(targets: Mapping[int, int], current: Mapping[int, int], tolerance: int = 8) -> str:
    """Channels still short of their targets, as ``ch3@6400->6976``, for arrival timeouts.

    The Maestro silently clamps a target outside the range stored in its own settings,
    which looks exactly like a servo that never arrives; naming the channel and its pulse is
    how the owner tells the two apart."""
    short = []
    for ch, target in sorted(targets.items()):
        now = current.get(ch)
        if now is None or abs(int(now) - int(target)) > tolerance:
            short.append(f"ch{ch}@{'?' if now is None else int(now)}->{int(target)}")
    return " ".join(short)


__all__ = [name for name in globals() if not name.startswith("_") and name not in ("math", "Iterable", "Mapping", "config")]
