import unittest
from unittest import mock


class BodyBeatAnimationTest(unittest.TestCase):
    def test_neck_center_uses_configured_range_midpoint(self):
        from sequences import animations

        neck_cfg = animations.config.SERVO_CHANNELS["neck"]
        expected_midpoint = (int(neck_cfg["min"]) + int(neck_cfg["max"])) // 2

        self.assertEqual(animations.NECK_CENTER, expected_midpoint)

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
                "agreement_nod",
                "anger_flash",
                "disagreement_shake",
                "disbelief_stare",
                "dramatic_visor_peek",
                "disgust_recoil",
                "giddy_wiggle",
                "happy_bounce",
                "offended_recoil",
                "proud_dj_pose",
                "sad_droop",
                "surprise_pop",
                "suspicious_glance",
                "thinking_tilt",
                "tiny_victory_dance",
            },
        )

    def test_surprise_pop_opens_visor_like_raised_eyebrows(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
        }

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
        ):
            self.assertTrue(animations.play_body_beat("surprise", async_=False))

        first_move = moves[0]
        self.assertEqual(first_move[1], animations.HEADLIFT_HIGH)
        self.assertEqual(first_move[2], animations.HEADTILT_UP)
        self.assertEqual(first_move[3], animations.VISOR_OPEN)

    def test_wake_word_ack_wave_moves_hand_and_elbow_together(self):
        from sequences import animations

        moves = []
        snapshot = {
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle") as pause,
            mock.patch.object(animations.servos, "resume_arm_idle") as resume,
        ):
            self.assertTrue(animations.wake_word_ack_wave(count=2, async_=False))

        pause.assert_called_once()
        resume.assert_called_once()
        self.assertEqual(
            moves[0],
            {
                7: animations.HEROARM_FORWARD,
                4: animations.ELBOW_NEUTRAL,
                5: animations.HAND_NEUTRAL,
            },
        )
        self.assertEqual(moves[1], {4: animations.ELBOW_UP, 5: animations.HAND_RIGHT})
        self.assertEqual(moves[2], {4: animations.ELBOW_DOWN, 5: animations.HAND_LEFT})
        self.assertEqual(moves[3], {4: animations.ELBOW_UP, 5: animations.HAND_RIGHT})
        self.assertEqual(moves[4], {4: animations.ELBOW_DOWN, 5: animations.HAND_LEFT})
        self.assertEqual(moves[-1], snapshot)

    def test_sleep_animation_uses_shutdown_rest_pose(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations.leds_chest, "sleep") as chest_sleep,
            mock.patch.object(animations.leds_head, "sleep") as head_sleep,
            mock.patch.object(animations.servos, "pause_arm_idle") as pause_arm,
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.time, "sleep", return_value=None),
        ):
            animations.sleep()

        chest_sleep.assert_called_once()
        head_sleep.assert_called_once()
        pause_arm.assert_called_once()
        self.assertEqual(moves[0], {3: animations.VISOR_CLOSED})
        self.assertEqual(
            moves[1],
            {
                0: animations.NECK_CENTER,
                1: animations.HEADLIFT_FLOOR,
                2: animations.HEADTILT_DOWN,
                4: animations.ELBOW_NEUTRAL,
                5: animations.HAND_NEUTRAL,
                6: animations.POKERARM_NEUTRAL,
                7: animations.HEROARM_NEUTRAL,
            },
        )

    def test_shutdown_animation_centers_neck_at_configured_midpoint(self):
        from sequences import animations

        moves = []
        neck_cfg = animations.config.SERVO_CHANNELS["neck"]
        expected_midpoint = (int(neck_cfg["min"]) + int(neck_cfg["max"])) // 2

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations.servos, "stop_breathing") as stop_breathing,
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.leds_head, "off") as head_off,
            mock.patch.object(animations.leds_chest, "off") as chest_off,
            mock.patch.object(animations.time, "sleep", return_value=None),
        ):
            animations.shutdown()

        stop_breathing.assert_called_once()
        # Visor, neck, head-lift and head-tilt all droop together in ONE move_to
        # so the droid powers down in a single motion (not visor→tilt→lift).
        self.assertEqual(len(moves), 1)
        self.assertEqual(
            moves[0],
            {
                3: animations.VISOR_CLOSED,
                0: expected_midpoint,
                1: animations.HEADLIFT_FLOOR,
                2: animations.HEADTILT_DOWN,
            },
        )
        head_off.assert_called_once()
        chest_off.assert_called_once()

    def test_wake_animation_restores_active_pose_and_arm_idle(self):
        from sequences import animations

        with (
            mock.patch.object(animations.leds_chest, "active") as chest_active,
            mock.patch.object(animations.leds_head, "active") as head_active,
            mock.patch.object(animations.leds_head, "set_eye_color") as eye_color,
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "resume_arm_idle") as resume_arm,
        ):
            animations.wake()

        chest_active.assert_called_once()
        move_to.assert_called_once_with(
            {
                1: animations.HEADLIFT_NEUTRAL,
                2: animations.HEADTILT_NEUTRAL,
                3: animations.VISOR_HALF,
            },
            step_us=35,
            step_delay=0.02,
        )
        head_active.assert_called_once()
        eye_color.assert_called_once_with(255, 200, 0)
        resume_arm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
