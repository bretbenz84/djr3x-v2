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


if __name__ == "__main__":
    unittest.main()
