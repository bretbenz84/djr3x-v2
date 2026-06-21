"""Tests for the human-like gaze controller (intelligence/gaze_engine.py + the
motion smoother, head interface, and the sim demo).

Covers the spec's behavioural guarantees: seeded determinism, the 50/70 duty cycle,
the anti-stare hard cap, complexity-scaled pre-turn aversion, pitch-direction
aversion semantics, the state machine transitions, the deg/mm<->qus mapping + clamp,
the velocity-limited smoother + watchdog, and a headless run of the demo.
"""

import importlib.util
import math
import os
import unittest

from intelligence.gaze_engine import (
    GazeConfig,
    GazeEngine,
    GazeInputs,
    GazeState,
    KIND_INTERNALIZING,
    KIND_THINKING,
    complexity_from_text,
)
from intelligence.head_interface import HeadPose, SimHead
from intelligence.motion_smoother import AxisLimiter, MotionSmoother


def _run(engine, *, seconds, dt=0.1, speaking=False, listening=False, conv=True,
         num_people=1, warmup=0.0, partner=(0.0, 0.0)):
    """Drive the engine for `seconds` in a fixed role; collect decisions after warmup."""
    out = []
    steps = int(seconds / dt)
    for i in range(steps):
        t = i * dt
        d = engine.step(GazeInputs(
            now=t, speaking=speaking, listening=listening, conversation_active=conv,
            num_people=num_people, partner_bearing=partner,
        ))
        if t >= warmup:
            out.append((t, d))
    return out


class DutyCycleTest(unittest.TestCase):
    def _steady_duty(self, *, speaking, listening, seconds=4000.0, warmup=6.0, num_people=1, seed=3):
        eng = GazeEngine(seed=seed)
        on = off = 0
        for t, d in _run(eng, seconds=seconds, speaking=speaking, listening=listening,
                         conv=True, num_people=num_people, warmup=warmup):
            if not d.active or d.state not in (GazeState.SPEAKING, GazeState.LISTENING):
                continue
            if d.kind in (KIND_INTERNALIZING, "include_sweep"):
                continue
            if d.mode == "on_target":
                on += 1
            else:
                off += 1
        self.assertGreater(on + off, 1000, "not enough steady samples")
        return on / (on + off)

    def test_listening_duty_near_070(self):
        duty = self._steady_duty(speaking=False, listening=False)  # -> LISTENING
        self.assertAlmostEqual(duty, 0.70, delta=0.08)

    def test_speaking_duty_near_050(self):
        duty = self._steady_duty(speaking=True, listening=False)   # -> SPEAKING
        self.assertAlmostEqual(duty, 0.50, delta=0.08)

    def test_listening_duty_higher_than_speaking(self):
        listen = self._steady_duty(speaking=False, listening=False)
        speak = self._steady_duty(speaking=True, listening=False)
        self.assertGreater(listen, speak)


class AntiStareTest(unittest.TestCase):
    def test_never_holds_on_target_past_hard_cap(self):
        cfg = GazeConfig.from_config()
        eng = GazeEngine(config=cfg, seed=11)
        dt = 0.08
        run_len = 0.0
        max_on = 0.0
        for t, d in _run(eng, seconds=3000.0, dt=dt, speaking=False, listening=False, warmup=6.0):
            if d.active and d.state == GazeState.LISTENING and d.mode == "on_target" \
                    and d.kind not in (KIND_INTERNALIZING,):
                run_len += dt
                max_on = max(max_on, run_len)
            else:
                run_len = 0.0
        # Allow a couple of ticks of slop beyond the cap.
        self.assertLessEqual(max_on, cfg.on_hard_cap + 3 * dt)
        self.assertGreater(max_on, 1.0, "expected some sustained on-target dwell")


class DeterminismTest(unittest.TestCase):
    def _sequence(self, seed):
        eng = GazeEngine(seed=seed)
        seq = []
        dt = 0.1
        for i in range(600):
            t = i * dt
            speaking = 8.0 <= t < 14.0
            listening = 6.0 <= t < 8.0
            conv = t < 40.0
            if abs(t - 5.9) < 1e-9:
                eng.note_about_to_speak(0.8)
            d = eng.step(GazeInputs(now=t, speaking=speaking, listening=listening,
                                    conversation_active=conv, num_people=2,
                                    listener_bearings=[(2, -30.0)], partner_bearing=(10.0, 0.0)))
            seq.append((d.state.value, d.mode, d.kind,
                        round(d.pose.yaw_deg, 3), round(d.pose.pitch_deg, 3),
                        round(d.pose.pole_mm, 3), d.segment_id))
        return seq

    def test_same_seed_is_reproducible(self):
        self.assertEqual(self._sequence(42), self._sequence(42))

    def test_different_seed_differs(self):
        self.assertNotEqual(self._sequence(1), self._sequence(2))


class PreTurnAversionTest(unittest.TestCase):
    def _prep_span_and_first(self, complexity, seed=0):
        eng = GazeEngine(seed=seed)
        dt = 0.05
        # establish LISTENING for a few seconds
        for i in range(int(5.0 / dt)):
            eng.step(GazeInputs(now=i * dt, conversation_active=True))
        t0 = 5.0
        eng.note_about_to_speak(complexity)
        span = 0.0
        first = None
        prep_seen = False
        for i in range(int(3.0 / dt)):
            t = t0 + i * dt
            d = eng.step(GazeInputs(now=t, conversation_active=True))  # not yet speaking
            if d.state == GazeState.PREP_TURN:
                prep_seen = True
                span += dt
                if first is None:
                    first = d
            elif prep_seen:
                break
        return span, first

    def test_duration_scales_with_complexity(self):
        cfg = GazeConfig.from_config()
        simple, _ = self._prep_span_and_first(0.0)
        complex_, _ = self._prep_span_and_first(1.0)
        self.assertLess(simple, complex_)
        self.assertAlmostEqual(simple, cfg.pre_aversion_min_secs, delta=0.12)
        self.assertAlmostEqual(complex_, cfg.pre_aversion_max_secs, delta=0.15)

    def test_complex_prep_looks_down_to_think(self):
        _, first = self._prep_span_and_first(1.0)
        self.assertIsNotNone(first)
        self.assertEqual(first.mode, "off_target")
        self.assertEqual(first.kind, KIND_THINKING)
        self.assertLess(first.pitch_offset_deg, 0.0)  # pitch DOWN, never up

    def test_simple_prep_is_to_the_side_not_down_to_think(self):
        _, first = self._prep_span_and_first(0.0)
        self.assertIsNotNone(first)
        self.assertNotEqual(first.kind, KIND_THINKING)
        self.assertLessEqual(first.pitch_offset_deg, 0.0)  # never up


class PitchSemanticsTest(unittest.TestCase):
    def test_internalizing_glances_down_after_listening(self):
        eng = GazeEngine(seed=5)
        dt = 0.05
        # a few seconds with the user holding the floor (LISTENING, not the think wait)
        for i in range(int(4.0 / dt)):
            eng.step(GazeInputs(now=i * dt, conversation_active=True, listening=False))
        # now R3X enters the think/processing wait -> internalizing down-glance
        d = eng.step(GazeInputs(now=4.0, conversation_active=True, listening=True))
        self.assertEqual(d.kind, KIND_INTERNALIZING)
        self.assertEqual(d.mode, "off_target")
        self.assertLess(d.pitch_offset_deg, 0.0)  # pitch DOWN

    def test_aversions_never_look_up(self):
        # Across a long mixed conversation (speaking, listening, complex pre-turns),
        # EVERY look-away must be level-or-down — Rex never looks up to avert.
        eng = GazeEngine(seed=9)
        dt = 0.1
        worst_up = -999.0
        for i in range(40000):
            t = i * dt
            speaking = (int(t) % 12) in (4, 5, 6, 7)   # talk in bursts
            if abs((t % 12) - 3.9) < 1e-9:
                eng.note_about_to_speak(0.95)          # complex -> "down to think"
            d = eng.step(GazeInputs(now=t, speaking=speaking, conversation_active=True,
                                    partner_bearing=(8.0, 0.0)))
            if d.active and d.mode == "off_target":
                worst_up = max(worst_up, d.pitch_offset_deg)
        self.assertLessEqual(worst_up, 0.0, f"an aversion looked up by {worst_up:.1f} deg")


class StateMachineTest(unittest.TestCase):
    def test_idle_when_no_conversation(self):
        eng = GazeEngine(seed=0)
        d = eng.step(GazeInputs(now=0.0, conversation_active=False))
        self.assertFalse(d.active)
        self.assertEqual(d.state, GazeState.IDLE)

    def test_opening_then_listening(self):
        eng = GazeEngine(seed=0)
        d0 = eng.step(GazeInputs(now=0.0, conversation_active=True))
        self.assertEqual(d0.state, GazeState.OPENING)
        # after the opening window it settles into LISTENING
        cfg = eng.cfg
        d1 = eng.step(GazeInputs(now=cfg.opening_secs + 0.5, conversation_active=True))
        self.assertEqual(d1.state, GazeState.LISTENING)

    def test_speaking_then_yielding(self):
        eng = GazeEngine(seed=0)
        eng.step(GazeInputs(now=0.0, conversation_active=True, speaking=True))
        d_speak = eng.step(GazeInputs(now=0.1, conversation_active=True, speaking=True))
        self.assertEqual(d_speak.state, GazeState.SPEAKING)
        d_yield = eng.step(GazeInputs(now=0.2, conversation_active=True, speaking=False))
        self.assertEqual(d_yield.state, GazeState.YIELDING)
        self.assertEqual(d_yield.mode, "on_target")  # returns gaze to hand over

    def test_closing_when_conversation_goes_idle(self):
        eng = GazeEngine(seed=0)
        eng.step(GazeInputs(now=0.0, conversation_active=True))
        eng.step(GazeInputs(now=5.0, conversation_active=True))
        d = eng.step(GazeInputs(now=6.0, conversation_active=False))
        self.assertEqual(d.state, GazeState.CLOSING)
        self.assertLess(d.pole_mm, eng.cfg.pole_rest_mm)  # pole lowered / disengage

    def test_suppressed_stands_down(self):
        eng = GazeEngine(seed=0)
        d = eng.step(GazeInputs(now=0.0, conversation_active=True, suppressed=True))
        self.assertFalse(d.active)
        self.assertFalse(d.drive)


class MappingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = GazeConfig.from_config()

    def test_yaw_mapping_and_clamp(self):
        c = self.cfg
        self.assertEqual(c.yaw_deg_to_neck_qus(0), c.neck_neutral)
        self.assertEqual(c.yaw_deg_to_neck_qus(999), c.neck_max)   # clamped
        self.assertEqual(c.yaw_deg_to_neck_qus(-999), c.neck_min)  # clamped
        self.assertAlmostEqual(c.neck_qus_to_yaw_deg(c.yaw_deg_to_neck_qus(30)), 30, delta=0.5)

    def test_pitch_is_inverted_and_asymmetric(self):
        c = self.cfg
        self.assertEqual(c.pitch_deg_to_tilt_qus(0), c.tilt_neutral)
        # UP (+) maps toward the LOW (min) qus end (inverted channel)
        self.assertLess(c.pitch_deg_to_tilt_qus(20), c.tilt_neutral)
        self.assertEqual(c.pitch_deg_to_tilt_qus(c.pitch_up_limit_deg), c.tilt_min)
        # DOWN (-) maps toward the HIGH (max) qus end
        self.assertGreater(c.pitch_deg_to_tilt_qus(-15), c.tilt_neutral)
        self.assertEqual(c.pitch_deg_to_tilt_qus(-c.pitch_down_limit_deg), c.tilt_max)

    def test_pole_mapping(self):
        c = self.cfg
        self.assertEqual(c.pole_mm_to_lift_qus(c.pole_rest_mm), c.lift_neutral)
        self.assertGreater(c.pole_mm_to_lift_qus(c.pole_max_mm), c.lift_neutral)  # lean-in raises
        self.assertEqual(c.pole_bias_qus(c.pole_rest_mm), 0)
        self.assertGreater(c.pole_bias_qus(c.pole_lean_in_mm), 0)

    def test_complexity_proxy(self):
        self.assertEqual(complexity_from_text("Nope."), 0.0)
        self.assertEqual(complexity_from_text(" ".join(["w"] * 60)), 1.0)
        mid = complexity_from_text(" ".join(["w"] * 23))
        self.assertTrue(0.0 < mid < 1.0)


class MotionSmootherTest(unittest.TestCase):
    def test_velocity_limit_respected(self):
        ax = AxisLimiter(position=0.0, min_limit=-100, max_limit=100, max_vel=10.0)
        # 1 second of dt=0.1 at vel<=10 can move at most ~10 units toward 100
        pos = 0.0
        for _ in range(10):
            pos = ax.step(100.0, 0.1)
        self.assertLessEqual(pos, 10.0 + 1e-6)

    def test_clamps_to_limits(self):
        ax = AxisLimiter(position=0.0, min_limit=-5, max_limit=5, max_vel=1000)
        self.assertEqual(ax.step(1000.0, 1.0), 5.0)
        self.assertEqual(ax.step(-1000.0, 1.0), -5.0)

    def test_watchdog_holds_on_nan(self):
        ax = AxisLimiter(position=3.0, min_limit=-5, max_limit=5, max_vel=10)
        self.assertEqual(ax.step(float("nan"), 0.1), 3.0)
        self.assertEqual(ax.step(2.0, float("inf")), 3.0)
        self.assertEqual(ax.step(2.0, 0.0), 3.0)

    def test_3axis_smoother_moves_toward_target(self):
        sm = MotionSmoother.from_limits(
            yaw_limit=(-70, 70), pitch_limit=(-20, 25), pole_limit=(0, 60),
            yaw_max_vel=300, pitch_max_vel=300, pole_max_vel=25, start=(0, 0, 20),
        )
        for _ in range(200):
            y, p, pole = sm.step(40, 10, 50, 0.02)
        self.assertAlmostEqual(y, 40, delta=1.0)
        self.assertAlmostEqual(p, 10, delta=1.0)
        self.assertAlmostEqual(pole, 50, delta=1.0)


class SimHeadTest(unittest.TestCase):
    def test_simhead_follows_commands(self):
        sm = MotionSmoother.from_limits(
            yaw_limit=(-70, 70), pitch_limit=(-20, 25), pole_limit=(0, 60),
            yaw_max_vel=300, pitch_max_vel=300, pole_max_vel=40, start=(0, 0, 20),
        )
        head = SimHead(smoother=sm)
        head.set_yaw(30, 300)
        head.set_pitch(15, 300)
        head.set_pole(45, 40)
        for _ in range(200):
            head.tick(0.02)
        pose = head.get_pose()
        self.assertIsInstance(pose, HeadPose)
        self.assertAlmostEqual(pose.yaw_deg, 30, delta=1.5)
        self.assertAlmostEqual(pose.pitch_deg, 15, delta=1.5)
        self.assertAlmostEqual(pose.pole_mm, 45, delta=1.5)


class DemoSimTest(unittest.TestCase):
    def _load_demo(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "gaze_demo_sim.py")
        spec = importlib.util.spec_from_file_location("gaze_demo_sim", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_demo_runs_headless(self):
        mod = self._load_demo()
        result = mod.run_demo(seed=1)
        self.assertTrue(result["log"], "demo produced no gaze log")
        states = {rec["state"] for rec in result["log"]}
        kinds = {rec["kind"] for rec in result["log"]}
        # The scripted flow should exercise the headline behaviours.
        self.assertIn("opening", states)
        self.assertIn("speaking", states)
        self.assertIn("closing", states)
        self.assertIn(KIND_THINKING, kinds)  # the complex reply looks down to think
        self.assertTrue(0.0 <= result["duty_on_during_turn"] <= 1.0)


class LiveActuationTest(unittest.TestCase):
    """The consciousness-side actuation (deg/mm offsets -> slew-limited servo writes).

    Inert under the test runner by default (so it never perturbs the face-tracking
    suite); opt in with DJR3X_GAZE_TEST_OPT_IN to drive it here.
    """

    def setUp(self):
        from unittest import mock as _mock
        self.mock = _mock
        import intelligence.consciousness as c
        self.c = c
        c._gaze_release()  # reset drive phase + velocity ramp state
        c.world_state.update(
            "self_state", {"servo_positions": {"neck": 6000, "headlift": 6000, "headtilt": 4320}}
        )

    def _mock_servo(self):
        m = self.mock.MagicMock()
        m.get_face_tracking_baseline.return_value = {}
        return m

    def test_maybe_drive_gaze_inert_under_test_runner(self):
        # Default (no opt-in): the gate returns False without touching the head.
        servo = self._mock_servo()
        self.assertFalse(self.c._maybe_drive_gaze(servo, 100.0, False))
        servo.set_servos.assert_not_called()

    def test_drive_gaze_aversion_maps_offsets_to_servo_writes(self):
        c = self.c
        servo = self._mock_servo()
        dec = c.gaze_engine.GazeDecision(
            active=True, state=c.gaze_engine.GazeState.LISTENING, mode="off_target",
            kind=c.gaze_engine.KIND_THINKING, pose=HeadPose(20.0, -12.0, 45.0),
            yaw_offset_deg=20.0,     # +yaw = left
            pitch_offset_deg=-12.0,  # DOWN (aversions never look up)
            pole_mm=45.0,            # lean-in (above rest)
            velocity="saccade", center_on=None, reason="t", segment_id=1,
        )
        anchor = (6000, 6000, 4320)
        # Step a handful of ticks so the ramp builds past the inclusion threshold.
        for _ in range(5):
            c._drive_gaze_aversion(servo, 100.0, dec, anchor)

        servo.set_servos.assert_called()
        updates = servo.set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        tilt_ch = c.config.SERVO_CHANNELS["headtilt"]["ch"]
        # +yaw (left) raises the neck qus above the anchor.
        self.assertGreater(updates[neck_ch], 6000)
        # DOWN pitch drives headtilt toward its HIGH (max) end (inverted channel).
        self.assertGreater(updates[tilt_ch], 4320)
        # lean-in raises headlift above the anchor.
        self.assertGreater(updates[lift_ch], 6000)
        servo.set_face_tracking_baseline.assert_called()

    def test_aversion_motion_ramps_in_softly(self):
        # The look-away accelerates from rest: per-tick neck velocity grows, then caps —
        # never a constant-speed snap (and never exceeds the velocity cap).
        c = self.c
        c._gaze_release()
        cap = c.config.GAZE_AVERSION_NECK_MAX_STEP_QUS
        vels = []
        cur = 6000
        target = 6000 + 4 * cap  # far away so it wants to cruise at the cap
        for _ in range(4):
            nxt, v = c._gaze_ramped_step("neck", cur, target, cap, cap / c.config.GAZE_AVERSION_RAMP_TICKS)
            vels.append(abs(v))
            cur = nxt
        self.assertLess(vels[0], vels[-1])          # accelerating (soft ease-in)
        self.assertTrue(all(v <= cap + 1e-6 for v in vels))  # never exceeds the cap


class FaceJumpGuardTest(unittest.TestCase):
    """The hardened jump guard: a transient identity-matched ghost (e.g. a phantom box
    high in frame above a seated person) must persist before the head chases it."""

    def setUp(self):
        import intelligence.consciousness as c
        self.c = c

    def test_identified_moderate_jump_followed_immediately(self):
        c = self.c
        last = {"key": "db:1", "cx": 960, "cy": 800, "at": 100.0}
        # ~380px move: beyond MAX_JUMP_FRAC (330px) but within the identified instant
        # ceiling (~485px) — a real sit/lean, follow at once.
        accept, _last, _pend = c._evaluate_face_jump(
            960, 420, "db:1", 100.1, 1920, 1080, last, None, identified=True, live_tracked=False,
        )
        self.assertTrue(accept)

    def test_identified_extreme_transient_ghost_rejected_then_confirmed(self):
        c = self.c
        # Replicates the field log: locked low (cy 862), a phantom box appears high
        # (cy 390) — a ~600px leap, past the identified instant ceiling.
        last = {"key": "db:1", "cx": 1230, "cy": 862, "at": 100.0}
        accept, last2, pend = c._evaluate_face_jump(
            850, 390, "db:1", 100.08, 1920, 1080, last, None, identified=True, live_tracked=False,
        )
        self.assertFalse(accept)  # transient ghost is NOT chased up
        # If it genuinely persists past the (shorter) identified confirm window, accept.
        accept2, _l3, _p2 = c._evaluate_face_jump(
            850, 390, "db:1", 100.08 + 0.3, 1920, 1080, last2, pend,
            identified=True, live_tracked=False,
        )
        self.assertTrue(accept2)


if __name__ == "__main__":
    unittest.main()
