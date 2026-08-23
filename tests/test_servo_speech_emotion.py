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


class SpeechVisorFloorTests(unittest.TestCase):
    """The talking visor may dip to VISOR_SPEECH_FLOOR_QUS (1500 us).

    Its floor used to be the visor's own neutral (1640 us), which sat above six of
    the ten emotion visor_open_floor_frac values and flattened them all to a single
    opening — the visor barely moved while Rex talked.
    """

    def _visor_target(self, servos, speech_motion, intensity=0.5):
        """One talking frame's visor target for a given emotion motion profile.

        The per-frame jitter is pinned to 0 so only the floor/swing math shows; the
        visor wave still rides the wall clock, so the target lands somewhere in
        [floor, floor + swing] — which is what these tests bound.
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
                mock.patch.object(servos.random, "randint", return_value=0),
                mock.patch.object(servos, "set_servos", side_effect=lambda t: captured.append(dict(t))),
            ):
                servos.speech_reactive_move(intensity)
            return captured[-1][servos._channel("visor")]
        finally:
            servos._speech_active.clear()
            servos._speech_emotion_frame = {}
            servos._speech_baseline = {}

    def test_a_subdued_emotion_reaches_the_speech_floor(self):
        import config
        from hardware import servos

        floor = int(config.VISOR_SPEECH_FLOOR_QUS)
        self.assertEqual(floor, 6000)                                   # 1500 us
        self.assertLess(floor, int(config.SERVO_CHANNELS["visor"]["neutral"]))

        neutral = int(config.SERVO_CHANNELS["visor"]["neutral"])
        # A brooding emotion (frac 0.35, narrow swing) rides the floor: the whole
        # band now sits BELOW the old neutral floor, which is the point.
        target = self._visor_target(
            servos, {"visor_open_floor_frac": 0.35, "visor_swing_mult": 0.30}
        )
        self.assertGreaterEqual(target, floor)
        self.assertLess(target, neutral - 200)

    def test_a_wide_eyed_emotion_still_floors_well_above_it(self):
        import config
        from hardware import servos

        target = self._visor_target(servos, {"visor_open_floor_frac": 0.92})
        # frac 0.92 lands above the speech floor, so the emotion's own frac wins and
        # the visor stays open — the floor never drags an expressive emotion down.
        self.assertGreater(target, int(config.SERVO_CHANNELS["visor"]["neutral"]))

    def test_a_capture_opens_the_visor_above_the_camera_clear_floor(self):
        import config
        from vision import camera

        visor_cfg = config.SERVO_CHANNELS["visor"]
        target = camera._visor_capture_target(visor_cfg)
        # A picture must never be taken through the expressive floor.
        self.assertGreaterEqual(target, int(config.VISOR_CAMERA_CLEAR_FLOOR_QUS))
        self.assertGreater(target, int(config.VISOR_SPEECH_FLOOR_QUS))
        self.assertEqual(int(config.VISOR_CAMERA_CLEAR_FLOOR_QUS), 6600)   # 1650 us


if __name__ == "__main__":
    unittest.main()
