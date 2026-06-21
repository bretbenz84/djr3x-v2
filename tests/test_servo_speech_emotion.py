import unittest
from unittest import mock


class ServoSpeechEmotionTests(unittest.TestCase):
    def test_begin_speech_motion_uses_emotion_frame_for_pose_and_breathing(self):
        from hardware import servos

        lift_ch = servos._channel("headlift")
        neutral_lift = servos.config.SERVO_CHANNELS["headlift"]["neutral"]
        try:
            servos._manual_override.clear()
            with servos._lock:
                servos._commanded_positions[lift_ch] = neutral_lift
                servos._face_tracking_baseline.pop(lift_ch, None)
            with (
                mock.patch.object(servos, "SERVOS_ENABLED", False),
                mock.patch.object(servos, "set_breathing_emotion") as breathing,
            ):
                servos.begin_speech_motion("sad")

                self.assertEqual(servos._speech_emotion_frame["affect"], "sad")
                self.assertEqual(servos._speech_emotion_frame["motion_style"], "subdued")
                self.assertLess(servos._speech_baseline[lift_ch], neutral_lift)
                breathing.assert_called_with("sad")
        finally:
            servos.end_speech_motion()

    def test_speech_reactive_move_yields_arm_to_scripted_gesture(self):
        """While a scripted arm gesture (wave-back) owns the arm, the talking-with-the-hands
        motion must leave the arm channels alone (head/visor keep talking) — otherwise the
        per-frame talking motion overrides the wave and no wave is seen."""
        from hardware import servos

        captured = []
        try:
            servos._manual_override.clear()
            servos._speech_active.set()
            servos._speech_emotion_frame = {}
            servos._speech_baseline = {}
            with (
                mock.patch.object(servos, "SERVOS_ENABLED", True),
                mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
                mock.patch.object(servos, "set_servos", side_effect=lambda t: captured.append(dict(t))),
            ):
                # Normal talking motion drives the arm.
                servos.end_arm_gesture()
                servos._last_speech_move_at = 0.0
                servos.speech_reactive_move(0.8)
                self.assertTrue(any(ch in captured[-1] for ch in servos.config.ARM_CHANNELS))

                # With a scripted arm gesture active, the arm channels are left to it …
                servos.begin_arm_gesture()
                servos._last_speech_move_at = 0.0
                servos.speech_reactive_move(0.8)
                self.assertFalse(any(ch in captured[-1] for ch in servos.config.ARM_CHANNELS))
                # … but the head keeps talking.
                self.assertIn(servos._channel("neck"), captured[-1])
        finally:
            servos.end_arm_gesture()
            servos._speech_active.clear()
            servos._speech_emotion_frame = {}


if __name__ == "__main__":
    unittest.main()
