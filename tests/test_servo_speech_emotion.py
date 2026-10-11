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


class SpeechVisorBandTests(unittest.TestCase):
    """The talking visor lives inside VISOR_SPEECH_RANGE_QUS (October 10: the servo is
    INVERTED, low pulses are up). An emotion's open fraction sets how far up its
    resting point sits between the camera-clear down edge (5520) and the fully
    expressive up edge (4880); the wave opens it further from there. A squint goes
    DOWN to the camera-clear edge and never past it.
    """

    def _visor_target(self, servos, speech_motion, intensity=0.5, jitter=-45):
        """One talking frame's visor target for a given emotion motion profile.

        The per-frame jitter is pinned to one extreme to check the clamp; the visor
        wave still rides the wall clock, so the target lands somewhere in the band.
        """
        captured = []
        try:
            servos._manual_override.clear()
            servos._speech_active.set()
            servos._speech_emotion_frame = {"speech_motion": speech_motion}
            servos._speech_baseline = {}
            servos._last_speech_move_at = 0.0
            with (
                mock.patch.object(servos, "SERVOS_ENABLED", True),
                mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
                mock.patch.object(servos.random, "randint", return_value=jitter),
                mock.patch.object(servos, "set_servos", side_effect=lambda t: captured.append(dict(t))),
            ):
                servos.speech_reactive_move(intensity)
            return captured[-1][servos._channel("visor")]
        finally:
            servos._speech_active.clear()
            servos._speech_emotion_frame = {}
            servos._speech_baseline = {}

    def test_a_subdued_emotion_sits_low_in_the_band(self):
        import config
        from hardware import servos

        up_edge, down_edge = config.VISOR_SPEECH_RANGE_QUS
        self.assertEqual((up_edge, down_edge), (4880, 5520))
        neutral = int(config.SERVO_CHANNELS["visor"]["neutral"])
        # A brooding emotion (frac 0.35, narrow swing) sits LOWER than the awake visor
        # (a higher pulse on the inverted servo) but never past the camera-clear edge.
        target = self._visor_target(
            servos, {"visor_open_floor_frac": 0.35, "visor_swing_mult": 0.30}
        )
        self.assertLessEqual(target, down_edge)
        self.assertGreater(target, neutral)

    def test_a_wide_eyed_emotion_rides_high_in_the_band(self):
        import config
        from hardware import servos

        from intelligence import emotion_orchestrator
        target = self._visor_target(servos, emotion_orchestrator.frame_for_emotion("shocked").speech_motion)
        up_edge, _ = config.VISOR_SPEECH_RANGE_QUS
        self.assertGreaterEqual(target, up_edge)
        # frac 0.92 opens it well above the awake position (a LOWER pulse).
        self.assertLess(target, int(config.SERVO_CHANNELS["visor"]["neutral"]))

    def test_angry_squint_reaches_the_camera_clear_edge_and_not_past_it(self):
        import config
        from hardware import servos
        from intelligence import emotion_orchestrator
        motion = emotion_orchestrator.frame_for_emotion("angry").speech_motion
        with mock.patch.object(servos.math, "sin", return_value=-1):
            target = self._visor_target(servos, motion, jitter=45)
        self.assertEqual(target, int(config.VISOR_CAMERA_CLEAR_MAX_QUS))
        self.assertEqual(target, 5520)

    def test_a_capture_lifts_the_visor_above_the_speaking_band(self):
        import config
        from vision import camera

        visor_cfg = config.SERVO_CHANNELS["visor"]
        target = camera._visor_capture_target(visor_cfg)
        # A picture must never be taken through the expressive band (inverted: a
        # LOWER pulse is further up).
        self.assertEqual(target, 1200 * 4)
        self.assertLess(target, int(config.VISOR_CAMERA_CLEAR_MAX_QUS))
        self.assertLess(target, int(config.VISOR_SPEECH_RANGE_QUS[0]))
        self.assertEqual(int(config.VISOR_CAMERA_CLEAR_MAX_QUS), 5520)   # 1380 us, mirrored


if __name__ == "__main__":
    unittest.main()
