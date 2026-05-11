import unittest
from unittest import mock


class IdleArmWanderTests(unittest.TestCase):
    def test_idle_hero_arm_target_crosses_center_from_back_pose(self):
        from sequences import animations

        target = animations._idle_arm_target(
            7,
            animations.HEROARM_NEUTRAL,
            animations.HEROARM_BACK,
            (1600, 1600),
            animations._IDLE_HEROARM_MIN_TRAVEL_QUS,
        )

        self.assertEqual(target, animations.HEROARM_NEUTRAL - 1600)
        self.assertGreaterEqual(
            abs(target - animations.HEROARM_BACK),
            animations._IDLE_HEROARM_MIN_TRAVEL_QUS,
        )

    def test_idle_arm_wander_targets_use_visible_hero_swing(self):
        from sequences import animations

        with (
            mock.patch.object(
                animations,
                "_current_body_pose",
                return_value={7: animations.HEROARM_BACK, 6: animations.POKERARM_NEUTRAL},
            ),
            mock.patch.object(animations.random, "choice", return_value=1),
            mock.patch.object(animations.random, "randint", side_effect=[1600, 900]),
        ):
            targets = animations._idle_arm_wander_targets()

        self.assertEqual(targets[7], animations.HEROARM_NEUTRAL - 1600)
        self.assertEqual(targets[6], animations.POKERARM_NEUTRAL + 900)


if __name__ == "__main__":
    unittest.main()
