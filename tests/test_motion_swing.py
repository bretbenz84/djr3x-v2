"""Swing check: a spin must not sweep the body or an arm into a ToF return.

Field 2026-08: a foot from a bookshelf, "turn left", and the left hand swept back
into the shelf and fell off. The base pivots about its REAR axle, so a left turn
carries the left-side arms rearward. These tests pin the geometry model and the
motion_controller wiring (turn / come / arc) against a fake ESP32 whose ToF ring
can be posed per test.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import motion_controller as mc
from intelligence import motion_swing as ms
from tests.test_motion import FakeESP32Serial, _MotionTestBase

OPEN = {"fl": 4000, "fr": 4000, "rl": 4000, "rr": 4000,
        "lf": 1500, "lb": 1500, "rf": 1500, "rb": 1500}


class _PosedRing(FakeESP32Serial):
    """Fake base whose radial ring reports whatever the test poses."""
    tof = dict(OPEN)

    def _telemetry(self):
        t = super()._telemetry()
        t["tof_mm"] = dict(self.tof)
        return t


class ModelTest(unittest.TestCase):
    def test_open_room_leaves_every_turn_alone(self):
        for deg in (90, -90, 180, -180, 360):
            self.assertEqual(ms.check_turn(deg, OPEN), (deg, None))

    def test_shelf_behind_left_refuses_a_left_turn(self):
        # The incident: ~a foot behind on the left. A CCW turn carries the
        # back-left arm rearward, straight into it.
        shelf = dict(OPEN, rl=300)
        deg, reason = ms.check_turn(90, shelf)
        self.assertEqual(reason, "swing_blocked")
        self.assertEqual(deg, 0.0)

    def test_shelf_behind_left_still_allows_the_right_turn(self):
        # CW carries the left arms FORWARD, away from it; the right side of the
        # ring swings rearward but the shelf is on the left.
        shelf = dict(OPEN, rl=300)
        self.assertEqual(ms.check_turn(-90, shelf), (-90, None))

    def test_rear_obstacle_shrinks_a_half_turn_it_cannot_finish(self):
        # Something close behind-right: the front of the ring orbits ~0.5 m out
        # about the axle and would come round into it before 180°.
        deg, reason = ms.check_turn(180, dict(OPEN, rr=300))
        self.assertIsNone(reason)
        self.assertGreaterEqual(deg, config.MOTION_SWING_MIN_TURN_DEG)
        self.assertLess(deg, 180)

    def test_sign_is_preserved_when_shrunk(self):
        deg, reason = ms.check_turn(-180, dict(OPEN, rl=300))
        self.assertIsNone(reason)
        self.assertLess(deg, 0)
        self.assertGreater(deg, -180)

    def test_errored_sensor_is_no_information_not_clear(self):
        # -1 is the wire's error sentinel: skipped, so an otherwise open ring
        # still permits the turn (the presence gate covers a whole dead ring).
        self.assertEqual(ms.check_turn(90, dict(OPEN, rl=-1)), (90, None))
        # ...but a real return on a neighbouring sensor still counts: something
        # 25 cm off the left-back quarter is reached by the front-left arm
        # partway round, so the turn is cut short.
        deg, reason = ms.check_turn(90, dict(OPEN, rl=-1, lb=250))
        self.assertIsNone(reason)
        self.assertLess(deg, 60)

    def test_no_telemetry_passes_through(self):
        self.assertEqual(ms.check_turn(90, None), (90, None))

    def test_disabled_flag_is_a_bypass(self):
        with mock.patch.object(config, "MOTION_SWING_CHECK_ENABLED", False):
            self.assertEqual(ms.check_turn(90, dict(OPEN, rl=300)), (90, None))

    def test_arm_extents_come_from_config(self):
        # Without the arms, a shelf behind-left only limits the ring — a 90° left
        # turn is allowed; with them it is refused. The arms ARE the difference.
        with mock.patch.object(config, "MOTION_BODY_EXTENTS", ()):
            deg, reason = ms.check_turn(90, dict(OPEN, rl=300))
        self.assertIsNone(reason)
        self.assertGreaterEqual(deg, 45)


class ControllerWiringTest(_MotionTestBase):
    def _connect_posed(self, tof):
        self.fake = _PosedRing()
        self.fake.tof = dict(tof)
        from hardware import motion
        motion.serial.Serial = lambda *a, **k: self.fake
        ok = mc.connect(port="FAKE")
        time.sleep(0.1)
        return ok

    def test_turn_left_refused_with_shelf_behind_left(self):
        self._connect_posed(dict(OPEN, rl=300))
        self.assertIsNone(mc.turn_left())
        self.assertIsNone(self._last("turn"))
        # The other way is fine.
        self.assertIsNotNone(mc.turn_right())
        self.assertEqual(self._last("turn")["deg"], -config.MOTION_DEFAULT_TURN_DEG)

    def test_shrunk_turn_sends_the_shrunk_angle(self):
        self._connect_posed(dict(OPEN, rr=300))
        self.assertIsNotNone(mc.turn(180))
        sent = self._last("turn")["deg"]
        self.assertGreater(sent, 0)
        self.assertLess(sent, 180)

    def test_come_heading_refused_when_its_spin_is_blocked(self):
        self._connect_posed(dict(OPEN, rl=300))
        self.assertIsNone(mc.come(heading=90.0))
        self.assertIsNone(self._last("come"))
        # A straight-ahead come has no spin to check.
        self.assertIsNotNone(mc.come(heading=0.0))
        self.assertIsNotNone(self._last("come"))

    def test_arc_swing_is_shortened_not_the_curve(self):
        self._connect_posed(dict(OPEN, rl=300))
        # Curving left near a shelf behind-left: refused outright.
        self.assertIsNone(mc.arc_move(forward=True, left=True))
        # Curving right is allowed.
        self.assertIsNotNone(mc.arc_move(forward=True, left=False))
        mc._cancel_arc()

    def test_refused_voice_turn_is_spoken(self):
        self._connect_posed(dict(OPEN, rl=300))
        mc.note_user_commanded_motion()
        with mock.patch("audio.speech_queue.enqueue") as enq:
            self.assertIsNone(mc.turn_left())
        self.assertTrue(enq.called)
        self.assertEqual(enq.call_args.kwargs.get("tag"), "motion_swing_blocked")


if __name__ == "__main__":
    unittest.main()
