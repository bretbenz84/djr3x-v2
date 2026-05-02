import unittest
from unittest import mock


class BodyBeatAnimationTest(unittest.TestCase):
    def test_offended_recoil_uses_inverted_upward_headtilt(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.random, "choice", return_value=1),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle"),
            mock.patch.object(animations.servos, "resume_arm_idle"),
        ):
            self.assertTrue(animations.play_body_beat("insult", async_=False))

        first_move = moves[0]
        self.assertEqual(first_move[1], animations.HEADLIFT_HIGH)
        self.assertEqual(first_move[2], animations.HEADTILT_UP)
        self.assertLess(first_move[2], animations.HEADTILT_NEUTRAL)
        self.assertEqual(first_move[3], animations.VISOR_OPEN)

    def test_named_body_beats_are_registered(self):
        from sequences import animations

        self.assertEqual(
            set(animations.body_beat_names()),
            {
                "dramatic_visor_peek",
                "offended_recoil",
                "proud_dj_pose",
                "suspicious_glance",
                "thinking_tilt",
                "tiny_victory_dance",
            },
        )


if __name__ == "__main__":
    unittest.main()
