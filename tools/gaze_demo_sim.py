#!/usr/bin/env python3
"""Offline gaze-controller sim — scripts a short 2-person conversation and prints the
gaze/pose/reason log so the timing can be eyeballed before wiring hardware.

Runs the pure :mod:`intelligence.gaze_engine` brain against a software
:class:`intelligence.head_interface.SimHead` (velocity-limited via
:mod:`intelligence.motion_smoother`) — no robot, no audio, no servos.

The scripted flow is the cantina "sir or madame?" greeting:

    R3X greets an arriving traveler (P1), asks whether to address them as
    "sir or madame?", listens to the answer, gives a longer (complex) reply, then
    the conversation lulls and closes. A second person (P2) stands off to the side so
    you can see the occasional include-sweep while R3X is speaking.

Usage::

    ./venv/bin/python tools/gaze_demo_sim.py
    ./venv/bin/python tools/gaze_demo_sim.py --rate 50 --seed 7 --ticks
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.gaze_engine import GazeConfig, GazeEngine, GazeInputs, GazeState  # noqa: E402
from intelligence.head_interface import SimHead  # noqa: E402
from intelligence.motion_smoother import MotionSmoother  # noqa: E402


# Partner bearings (degrees): P1 is the active traveler, P2 a bystander to the left.
P1_ID, P1_YAW = 1, 12.0
P2_ID, P2_YAW = 2, -32.0


# Scripted phases: (start_s, end_s, speaking, listening, conversation_active, label).
# `listening` here means R3X's think/processing wait (not "the user is talking").
SCRIPT = [
    (0.0, 0.6, False, False, True, "P1 walks up — conversation opens"),
    (0.6, 3.2, True, False, True, "R3X: \"Well, a fresh face! Welcome in.\""),
    (3.2, 6.2, False, False, True, "P1 replies"),
    (6.2, 6.7, False, True, True, "R3X processes (think wait)"),
    (6.7, 9.4, True, False, True, "R3X: \"Do I call you sir... or madame?\""),
    (9.4, 12.4, False, False, True, "P1 answers"),
    (12.4, 13.1, False, True, True, "R3X processes a longer reply"),
    (13.1, 17.6, True, False, True, "R3X: long, complex reply"),
    (17.6, 23.0, False, False, True, "lull — both quiet, R3X still engaged"),
    (23.0, 26.0, False, False, False, "P1 drifts off — conversation closes"),
]

# Instants at which R3X is "about to speak", with the reply's complexity in [0,1].
# (Armed a beat before each speaking phase, matching tts.speak -> note_about_to_speak.)
SPEAK_ARMS = [
    (0.45, 0.10, "greeting (simple)"),
    (6.55, 0.45, "sir-or-madame question (medium)"),
    (12.95, 0.95, "long reply (complex)"),
]


def _phase_at(t: float):
    for start, end, speaking, listening, conv, label in SCRIPT:
        if start <= t < end:
            return speaking, listening, conv, label
    return False, False, False, "idle"


def _velocity_dps(decision, cfg: GazeConfig) -> float:
    return cfg.saccade_vel_dps if decision.velocity == "saccade" else cfg.smooth_vel_dps


def run_demo(seed: int = 0, rate_hz: float = 50.0, duration_s: float = 26.5):
    """Run the scripted conversation; return the per-segment gaze log (list of dicts)."""
    cfg = GazeConfig.from_config()
    engine = GazeEngine(config=cfg, seed=seed)

    log: list[dict] = []
    engine.set_log_hook(log.append)

    smoother = MotionSmoother.from_limits(
        yaw_limit=(-cfg.yaw_limit_deg, cfg.yaw_limit_deg),
        pitch_limit=(-cfg.pitch_down_limit_deg, cfg.pitch_up_limit_deg),
        pole_limit=(cfg.pole_min_mm, cfg.pole_max_mm),
        yaw_max_vel=cfg.saccade_vel_dps,
        pitch_max_vel=cfg.saccade_vel_dps,
        pole_max_vel=cfg.pole_vel_mms,
        start=(P1_YAW, 0.0, cfg.pole_rest_mm),
    )
    head = SimHead(smoother=smoother)

    dt = 1.0 / rate_hz
    armed = [False] * len(SPEAK_ARMS)
    ticks: list[tuple] = []
    on_count = off_count = 0

    steps = int(duration_s * rate_hz)
    for i in range(steps):
        t = i * dt
        speaking, listening, conv, _label = _phase_at(t)

        for k, (arm_t, complexity, _why) in enumerate(SPEAK_ARMS):
            if not armed[k] and t >= arm_t:
                engine.note_about_to_speak(complexity)
                armed[k] = True

        inputs = GazeInputs(
            now=t,
            speaking=speaking,
            listening=listening,
            conversation_active=conv,
            conversation_idle_secs=0.0 if conv else (t - 23.0),
            num_people=2,
            active_speaker_id=P1_ID,
            partner_bearing=(P1_YAW, 0.0),
            listener_bearings=[(P2_ID, P2_YAW)],
        )
        decision = engine.step(inputs)

        vel = _velocity_dps(decision, cfg)
        head.set_yaw(decision.pose.yaw_deg, vel)
        head.set_pitch(decision.pose.pitch_deg, vel)
        head.set_pole(decision.pose.pole_mm, cfg.pole_vel_mms)
        pose = head.tick(dt)

        if decision.active and decision.state in (GazeState.SPEAKING, GazeState.LISTENING):
            if decision.mode == "on_target":
                on_count += 1
            else:
                off_count += 1
        ticks.append((t, decision, pose))

    duty = on_count / max(1, on_count + off_count)
    return {"log": log, "ticks": ticks, "duty_on_during_turn": duty}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=50.0, help="update rate (Hz)")
    ap.add_argument("--ticks", action="store_true", help="also print a coarse per-0.5s pose trace")
    args = ap.parse_args()

    result = run_demo(seed=args.seed, rate_hz=args.rate)

    print("=" * 78)
    print("DJ-R3X gaze sim — 'sir or madame?' greeting (2 people)")
    print("ON-target = looking at the partner;  OFF-target = a deliberate look-away")
    print("pose = (yaw°, pitch°, pole mm)   +yaw=left  +pitch=up  +pole=lean-in")
    print("=" * 78)
    print(f"{'t(s)':>6}  {'STATE':<10} {'MODE':<11} {'KIND':<13} {'POSE':<20} REASON")
    print("-" * 78)
    for rec in result["log"]:
        y, p, pole = rec["pose"]
        pose_s = f"({y:>5.1f},{p:>5.1f},{pole:>4.1f})"
        print(
            f"{rec['t']:>6.2f}  {rec['state']:<10} {rec['mode']:<11} "
            f"{rec['kind']:<13} {pose_s:<20} {rec['reason']}"
        )
    print("-" * 78)
    print(
        f"Measured ON-target duty during speaking+listening turns: "
        f"{result['duty_on_during_turn']:.2f} "
        f"(spec: ~0.50 speaking / ~0.70 listening, blended here)"
    )

    if args.ticks:
        print("\nCoarse pose trace (every 0.5 s):")
        last = -1.0
        for t, decision, pose in result["ticks"]:
            if t - last >= 0.5 - 1e-9:
                last = t
                print(
                    f"  t={t:>5.2f}  {decision.state.value:<10} "
                    f"yaw={pose.yaw_deg:>6.1f}  pitch={pose.pitch_deg:>5.1f}  pole={pose.pole_mm:>4.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
