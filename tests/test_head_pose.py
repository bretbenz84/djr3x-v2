"""The two-servo head, poker clearance and inverted visor geometry (October 10 2026).

Pure arithmetic against the owner's Maestro Control Center readings; nothing here
touches hardware. The seven head readings must round-trip exactly, the linkage's
unseen corners must be refused, and every poker route must stay under the measured
ceiling at each stage.
"""

import unittest

from hardware import head_pose as hp


# The owner's console readings (microseconds), left / right.
READINGS_US = {
    "level": (1458, 1620),
    "level_down": (1013, 1904),
    "level_up": (2032, 1312),
    "down_right": (1008, 1702),
    "down_left": (1350, 1849),
    "up_left": (2032, 1587),
    "up_right": (1659, 1312),
}


def _pair(us):
    return {hp.HEAD_LEFT_CH: us[0] * 4, hp.HEAD_RIGHT_CH: us[1] * 4}


class HeadModelTest(unittest.TestCase):
    def test_every_owner_reading_round_trips_exactly(self):
        for name, us in READINGS_US.items():
            pulses = _pair(us)
            self.assertEqual(hp.head_fit_error(pulses), 0, name)
            pitch, tilt = hp.head_pose(pulses)
            self.assertEqual(hp.head_pulses(pitch, tilt), pulses, name)

    def test_named_extremes(self):
        self.assertEqual(hp.head_pulses(0, 0), _pair(READINGS_US["level"]))
        self.assertEqual(hp.head_pulses(1, 0), _pair(READINGS_US["level_up"]))
        self.assertEqual(hp.head_pulses(-1, 0), {2: 4052, 11: 7616})

    def test_pitch_moves_the_servos_against_each_other(self):
        level = hp.head_pulses(0, 0)
        up = hp.head_pulses(0.5, 0)
        down = hp.head_pulses(-0.5, 0)
        self.assertGreater(up[2], level[2]); self.assertLess(up[11], level[11])
        self.assertLess(down[2], level[2]); self.assertGreater(down[11], level[11])

    def test_tilt_moves_both_servos_the_same_way(self):
        level = hp.head_pulses(0, 0)
        right = hp.head_pulses(0, 0.5)     # tilting right lowers both pulses
        left = hp.head_pulses(0, -0.5)
        self.assertLess(right[2], level[2]); self.assertLess(right[11], level[11])
        self.assertGreater(left[2], level[2]); self.assertGreater(left[11], level[11])
        # The full shift is TILT_SHIFT split evenly while both have room.
        self.assertEqual((level[2] - right[2]) + (level[11] - right[11]), hp.TILT_SHIFT // 2)

    def test_tilt_overflow_goes_to_the_other_servo(self):
        # Fully up, the right servo is at its low limit: a right tilt must land on the left.
        up_right = hp.head_pulses(1, 1)
        self.assertEqual(up_right[11], 5248)
        self.assertEqual(up_right[2], 8128 - hp.TILT_SHIFT)
        for ch, (low, high) in hp.head_limits().items():
            for pitch in (-1, -0.5, 0, 0.5, 1):
                for tilt in (-1, -0.5, 0, 0.5, 1):
                    value = hp.head_pulses(pitch, tilt)[ch]
                    self.assertTrue(low <= value <= high, (pitch, tilt, ch, value))

    def test_inverse_is_close_everywhere_inside_the_envelope(self):
        for i in range(-10, 11):
            for j in range(-10, 11):
                pitch, tilt = i / 10, j / 10
                pulses = hp.head_pulses(pitch, tilt)
                self.assertLessEqual(hp.head_fit_error(pulses), 2, (pitch, tilt))

    def test_refuses_one_channel_without_the_other(self):
        with self.assertRaisesRegex(ValueError, "IncompleteHeadPose"):
            hp.validate_head_pair({2: 5832})
        with self.assertRaisesRegex(ValueError, "IncompleteHeadPose"):
            hp.validate_head_pair({11: 6480, 0: 5472})
        self.assertTrue(hp.head_pair_complete({0: 5472, 1: 3600}))

    def test_refuses_corners_the_linkage_never_saw(self):
        with self.assertRaisesRegex(ValueError, "HeadPairOutsideEnvelope"):
            hp.validate_head_pair({2: 8128, 11: 7616})     # both at their highest
        with self.assertRaisesRegex(ValueError, "HeadPairOutsideEnvelope"):
            hp.validate_head_pair({2: 4032, 11: 5248})     # both at their lowest
        with self.assertRaisesRegex(ValueError, "OutsideLimits"):
            hp.validate_head_pair({2: 3000, 11: 6480})
        hp.validate_head_pair(hp.head_pulses(0.3, -0.7))   # a real pose passes

    def test_confirmed_direction_negative_tilt_is_rex_left(self):
        # Owner, October 10 evening: tilt_left cocks the head to Rex's own left. The
        # owner's "tilted left" readings decode to a negative tilt.
        for name in ("down_left", "up_left"):
            self.assertLess(hp.head_pose(_pair(READINGS_US[name]))[1], 0, name)
        for name in ("down_right", "up_right"):
            self.assertGreater(hp.head_pose(_pair(READINGS_US[name]))[1], 0, name)

    def test_degrees_and_legacy_units(self):
        self.assertAlmostEqual(hp.pitch_from_degrees(25), 1.0)
        self.assertAlmostEqual(hp.pitch_from_degrees(-10), -0.5)
        self.assertAlmostEqual(hp.degrees_from_pitch(-1), -20)
        self.assertAlmostEqual(hp.pitch_from_legacy_delta(1184), -1.0)   # old qus down
        self.assertAlmostEqual(hp.pitch_from_legacy_delta(-416), 1.0)    # old qus up
        self.assertEqual(hp.legacy_delta_from_pitch(-0.5), 592)

    def test_head_targets_keeps_what_the_spec_leaves_out(self):
        current = {**hp.head_pulses(-0.4, 0.5), 1: 3600}
        targets = hp.head_targets(hp.HEAD_POSES["look_down"], current)
        pitch, tilt = hp.head_pose(targets)
        self.assertAlmostEqual(pitch, -0.8, places=1)
        self.assertAlmostEqual(tilt, 0.5, places=1)          # the cock survives
        self.assertEqual(targets[1], 3200)
        targets = hp.head_targets(hp.HEAD_POSES["tilt_left"], current)
        pitch, tilt = hp.head_pose(targets)
        self.assertAlmostEqual(pitch, -0.4, places=1)         # the pitch survives
        self.assertAlmostEqual(tilt, -0.8, places=1)
        self.assertNotIn(1, targets)

    def test_reframe_counts_both_head_servos(self):
        current = hp.head_pulses(0, 0)
        self.assertFalse(hp.head_reframes(current, {2: current[2] + 400, 11: current[11] + 200}))
        self.assertTrue(hp.head_reframes(current, {2: current[2] + 500, 11: current[11]}))
        self.assertTrue(hp.head_reframes(current, {2: current[2], 11: current[11] - 350}))


class RelativeLookTest(unittest.TestCase):
    def test_side_look_goes_to_the_33_degree_pose_from_near_centre(self):
        self.assertEqual(hp.neck_look_target("left", 5472), hp.NECK_LOOK_LEFT)
        self.assertEqual(hp.neck_look_target("right", 5472), hp.NECK_LOOK_RIGHT)
        self.assertEqual(hp.neck_look_target("left", 7000), hp.NECK_LOOK_LEFT)

    def test_already_there_a_person_goes_on_to_the_limit_and_rex_stays(self):
        # First live run: following the owner at the neck's left limit, "look left"
        # swung the head back to the right. Never again.
        self.assertEqual(hp.neck_look_target("left", 1984), 1984)
        self.assertEqual(hp.neck_look_target("left", 2500, further=True), 1984)
        self.assertEqual(hp.neck_look_target("left", 2500, further=False), 2500)
        self.assertEqual(hp.neck_look_target("right", 8500, further=True), 8960)
        self.assertEqual(hp.neck_look_target("right", 8500, further=False), 8500)


class ExpressionTest(unittest.TestCase):
    def test_every_expression_returns_exactly_home(self):
        current = hp.head_pulses(-0.2, 0.1)
        for name in hp.HEAD_EXPRESSIONS:
            steps = hp.expression_steps(name, current)
            self.assertEqual(steps[-1]["targets"], current, name)
            for step in steps:
                hp.validate_head_pair(step["targets"])

    def test_sided_expressions_cock_away_from_an_existing_lean(self):
        leaning_right = hp.head_pulses(0, 0.5)
        steps = hp.expression_steps("tilt", leaning_right)
        self.assertLess(hp.head_pose(steps[0]["targets"])[1], 0.5)   # cocks back toward the left
        straight = hp.head_pulses(0, 0)
        steps = hp.expression_steps("tilt", straight)
        self.assertAlmostEqual(hp.head_pose(steps[0]["targets"])[1], 0.6, places=1)

    def test_perplexed_is_the_double_cock_slightly_down(self):
        steps = hp.expression_steps("perplexed", hp.head_pulses(0, 0))
        first, second = (hp.head_pose(s["targets"]) for s in steps[:2])
        self.assertAlmostEqual(first[1], 0.6, places=1)
        self.assertAlmostEqual(second[1], -0.6, places=1)
        self.assertLess(first[0], 0); self.assertLess(second[0], 0)
        self.assertEqual([s["dwell"] for s in steps[:2]], [0.7, 0.7])

    def test_subtle_halves_the_amounts(self):
        full = hp.head_pose(hp.expression_steps("sympathetic", hp.head_pulses(0, 0))[0]["targets"])
        half = hp.head_pose(hp.expression_steps("sympathetic", hp.head_pulses(0, 0), subtle=True)[0]["targets"])
        self.assertAlmostEqual(half[1], full[1] / 2, places=1)
        self.assertAlmostEqual(half[0], full[0] / 2, places=1)

    def test_nod_dips_then_lifts(self):
        steps = hp.expression_steps("nod", hp.head_pulses(0, 0))
        self.assertAlmostEqual(hp.head_pose(steps[0]["targets"])[0], -0.3, places=1)
        self.assertAlmostEqual(hp.head_pose(steps[1]["targets"])[0], 0.1, places=1)

    def test_guide_covers_every_expression(self):
        self.assertEqual(set(hp.HEAD_EXPRESSION_GUIDE), set(hp.HEAD_EXPRESSIONS))


class PokerClearanceTest(unittest.TestCase):
    def test_measured_ceiling_pairs(self):
        for wrist_us, elbow_us in ((1564, 1920), (1894, 1807), (2136, 1709), (2496, 1597)):
            self.assertEqual(hp.poker_elbow_max(wrist_us * 4), elbow_us * 4, wrist_us)
        self.assertEqual(hp.poker_elbow_max(1984), hp.POKER_ELBOW_UP)     # wrist down: full rise
        self.assertTrue(7228 < hp.poker_elbow_max(7000) < 7680)         # linear between

    def test_named_poses_are_clear(self):
        for name, pose in hp.POKER_POSES.items():
            hp.validate_poker_pair(pose)

    def test_refuses_a_lone_joint_and_a_collision(self):
        with self.assertRaisesRegex(ValueError, "IncompletePokerPose"):
            hp.validate_poker_pair({12: 6000})
        with self.assertRaisesRegex(ValueError, "PokerElbowWristCollision"):
            hp.validate_poker_pair({12: 7680, 13: 9000})
        with self.assertRaisesRegex(ValueError, "OutsideLimits"):
            hp.validate_poker_pair({12: 9000, 13: 5760})

    def test_route_moves_one_joint_at_a_time_and_never_crosses_the_ceiling(self):
        poses = list(hp.POKER_POSES.values()) + [{12: 6388, 13: 9944}, {12: 3456, 13: 1984}]
        for start in poses:
            for target in poses:
                route = hp.poker_route(start, target)
                previous = dict(start)
                for stage in route:
                    moved = [ch for ch in hp.POKER_PAIR if stage[ch] != previous[ch]]
                    self.assertLessEqual(len(moved), 1, (start, target, route))
                    self.assertTrue(hp.poker_clear(stage), (start, target, stage))
                    previous = stage
                self.assertEqual(route[-1], {ch: target[ch] for ch in hp.POKER_PAIR})

    def test_park_to_awake_goes_wrist_first_then_elbow_down(self):
        # From the park (elbow up, wrist 1564) to the 45-degree pose: the elbow drops
        # to the lower of the two first, then the wrist.
        route = hp.poker_route(hp.POKER_PARK, hp.POKER_UP)
        self.assertEqual(route, [{12: 6000, 13: 6256}, {12: 6000, 13: 3768}])
        # And back: wrist first, then the elbow up (shutdown).
        route = hp.poker_route(hp.POKER_UP, hp.POKER_PARK)
        self.assertEqual(route, [{12: 6000, 13: 6256}, {12: 7680, 13: 6256}])

    def test_no_move_is_one_stage(self):
        self.assertEqual(hp.poker_route(hp.POKER_UP, hp.POKER_UP), [dict(hp.POKER_UP)])


class VisorAndJointsTest(unittest.TestCase):
    def test_visor_fraction_is_inverted(self):
        self.assertAlmostEqual(hp.visor_frac(6976), 0.0)   # down
        self.assertAlmostEqual(hp.visor_frac(4544), 1.0)   # up
        self.assertEqual(hp.visor_pulse(1.0), 4544)
        self.assertLess(hp.VISOR_UP, hp.VISOR_DOWN)
        self.assertLessEqual(hp.VISOR_SPEECH[1], hp.VISOR_CAMERA_CLEAR_MAX)

    def test_joint_percents_use_pitch_and_tilt_not_servos(self):
        pose = {**hp.head_pulses(0.5, -0.5), 0: 5472, 1: 3600, 3: 5120, 12: 6000, 13: 3768}
        percents = hp.joint_percents(pose)
        self.assertEqual(percents["head_pitch"], 75)
        self.assertEqual(percents["head_tilt"], 25)
        self.assertEqual(percents["visor"], 76)          # 0 down, 100 up
        self.assertNotIn("head_left", percents)
        self.assertNotIn("head_right", percents)

    def test_describe_short_names_the_channel_and_pulse(self):
        self.assertEqual(hp.describe_short({3: 6976, 2: 4052}, {3: 6400, 2: 4052}), "ch3@6400->6976")
        self.assertEqual(hp.describe_short({3: 6976}, {}), "ch3@?->6976")
        self.assertEqual(hp.describe_short({3: 6976}, {3: 6970}), "")


if __name__ == "__main__":
    unittest.main()
