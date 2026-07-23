"""
Cardinal-direction voice commands ("turn north", "go east two feet") against the
calibrated compass. Covers the classifier patterns (incl. figurative "go south"
safety), the compass->relative-turn math and its sign convention, the deadband
sentinel, and heading-unavailable behavior. The compass service is mocked — no
hardware, no serial.
"""

import unittest
from unittest import mock

import config
from intelligence import action_router as ar
from intelligence import motion_controller as mc


class CompassClassifierTest(unittest.TestCase):
    def _turn(self, text):
        d = ar.classify_explicit_motion(text)
        self.assertIsNotNone(d, text)
        self.assertEqual(d.action, "motion.turn", text)
        return d.args

    def _move(self, text):
        d = ar.classify_explicit_motion(text)
        self.assertIsNotNone(d, text)
        self.assertEqual(d.action, "motion.move", text)
        return d.args

    def test_turn_forms(self):
        for text in ("turn north", "face north", "point north", "rotate north",
                     "turn to the north", "face towards north", "turn due north"):
            args = self._turn(text)
            self.assertEqual(args["compass"], "north", text)
            self.assertEqual(args["compass_deg"], 0.0, text)

    def test_all_cardinals_and_diagonals(self):
        for text, card, deg in [
            ("turn east", "east", 90.0),
            ("face south", "south", 180.0),
            ("turn west", "west", 270.0),
            ("turn northeast", "northeast", 45.0),
            ("face north-west", "northwest", 315.0),
            ("turn south east", "southeast", 135.0),
        ]:
            args = self._turn(text)
            self.assertEqual(args["compass"], card, text)
            self.assertEqual(args["compass_deg"], deg, text)

    def test_go_forms_with_distance(self):
        args = self._move("go north two feet")
        self.assertEqual(args["compass"], "north")
        self.assertAlmostEqual(args["dist_m"], 2 * 0.3048, places=4)
        args = self._move("drive east")
        self.assertEqual(args["compass"], "east")
        self.assertNotIn("dist_m", args)
        args = self._move("head west 1 meter")
        self.assertEqual(args["compass"], "west")
        self.assertAlmostEqual(args["dist_m"], 1.0)

    def test_figurative_go_south_is_conversation(self):
        for text in ("this could go south", "it might go south fast",
                     "things will go south", "everything went south yesterday"):
            d = ar.classify_explicit_motion(text)
            if d is not None:
                self.assertNotIn("compass", d.args or {}, text)

    def test_plain_turns_unaffected(self):
        d = ar.classify_explicit_motion("turn right 45 degrees")
        self.assertEqual(d.args, {"direction": "right", "deg": 45.0})

    def test_compass_inside_sequences(self):
        seq = ar.classify_explicit_motion_sequence("turn north then move forward two feet")
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(seq[0].args["compass"], "north")


class CompassTurnMathTest(unittest.TestCase):
    def _delta(self, yaw, target):
        with mock.patch("hardware.compass.get_service_yaw", return_value=yaw):
            return mc.compass_turn_delta(target)

    def test_sign_convention(self):
        # Facing east (90), want north (0): turn LEFT/CCW 90 -> +90 (turn() is CCW+).
        self.assertAlmostEqual(self._delta(90.0, 0.0), 90.0)
        # Facing north, want east: turn right 90 -> -90.
        self.assertAlmostEqual(self._delta(0.0, 90.0), -90.0)

    def test_wraparound_takes_short_way(self):
        self.assertAlmostEqual(self._delta(350.0, 10.0), -20.0)   # cross north to the right
        self.assertAlmostEqual(self._delta(10.0, 350.0), 20.0)    # cross north to the left
        self.assertAlmostEqual(abs(self._delta(0.0, 180.0)), 180.0)

    def test_unavailable_heading(self):
        self.assertIsNone(self._delta(None, 0.0))

    def test_turn_to_compass_deadband_sentinel(self):
        with mock.patch("hardware.compass.get_service_yaw", return_value=3.0), \
                mock.patch.object(config, "COMPASS_TURN_DEADBAND_DEG", 6.0, create=True):
            self.assertEqual(mc.turn_to_compass(0.0), 0)     # within deadband: no command

    def test_turn_to_compass_issues_relative_turn(self):
        with mock.patch("hardware.compass.get_service_yaw", return_value=90.0), \
                mock.patch.object(mc, "turn", return_value=7) as turn:
            self.assertEqual(mc.turn_to_compass(0.0), 7)
        self.assertAlmostEqual(turn.call_args.args[0], 90.0)

    def test_turn_to_compass_unavailable_returns_none(self):
        with mock.patch("hardware.compass.get_service_yaw", return_value=None), \
                mock.patch.object(mc, "turn") as turn:
            self.assertIsNone(mc.turn_to_compass(0.0))
        turn.assert_not_called()


class CompassSequenceStepTest(unittest.TestCase):
    def test_issue_computes_relative_at_run_time(self):
        from intelligence import motion_sequence as ms
        d = ar.ActionDecision(action="motion.turn", confidence=0.95,
                              args={"compass": "north", "compass_deg": 0.0}, reason="t")
        with mock.patch.object(mc, "turn_to_compass", return_value=5) as ttc:
            seq, dur = ms._issue(d)
        self.assertEqual(seq, 5)
        ttc.assert_called_once_with(0.0)

    def test_issue_already_facing_advances(self):
        from intelligence import motion_sequence as ms
        d = ar.ActionDecision(action="motion.turn", confidence=0.95,
                              args={"compass": "north", "compass_deg": 0.0}, reason="t")
        with mock.patch.object(mc, "turn_to_compass", return_value=0):
            seq, _ = ms._issue(d)
        self.assertEqual(seq, 0)   # _run treats 0 as "no-op step, advance immediately"


if __name__ == "__main__":
    unittest.main()
