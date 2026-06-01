import unittest
from unittest import mock


class ListeningTargetsTests(unittest.TestCase):
    """Gentle listening pose targets must stay in range and orbit the gaze."""

    def setUp(self):
        from hardware import servos
        self.servos = servos
        # Known gaze baseline (neutral head pose) so target math is predictable.
        self._neck = servos._channel("neck")
        self._lift = servos._channel("headlift")
        self._tilt = servos._channel("headtilt")
        self._visor = servos._channel("visor")
        self._elbow = servos._channel("elbow")
        self._hand = servos._channel("hand")
        self._hero = servos._channel("heroarm")
        self._baseline = {
            self._neck: servos.config.SERVO_CHANNELS["neck"]["neutral"],
            self._lift: servos.config.SERVO_CHANNELS["headlift"]["neutral"],
            self._tilt: servos.config.SERVO_CHANNELS["headtilt"]["neutral"],
        }

    def _patch_baseline(self):
        return mock.patch.object(self.servos, "_face_tracking_baseline", dict(self._baseline))

    def test_targets_stay_within_channel_limits(self):
        with self._patch_baseline():
            for beat in range(1, 9):
                targets = self.servos._listening_targets(beat)
                for ch, pos in targets.items():
                    name = self.servos._CHANNEL_TO_NAME[ch]
                    cfg = self.servos.config.SERVO_CHANNELS[name]
                    self.assertGreaterEqual(pos, cfg["min"], f"{name} below min on beat {beat}")
                    self.assertLessEqual(pos, cfg["max"], f"{name} above max on beat {beat}")

    def test_head_and_visor_present_every_beat(self):
        with self._patch_baseline():
            for beat in range(1, 6):
                t = self.servos._listening_targets(beat)
                for ch in (self._neck, self._lift, self._tilt, self._visor):
                    self.assertIn(ch, t)

    def test_non_nod_beat_returns_head_to_gaze_baseline(self):
        # nod_every defaults to 2, so an odd beat eases back to baseline (no random).
        with self._patch_baseline():
            t = self.servos._listening_targets(1)
        self.assertEqual(t[self._lift], self._baseline[self._lift])
        self.assertEqual(t[self._neck], self._baseline[self._neck])
        self.assertEqual(t[self._tilt], self._baseline[self._tilt])
        # Arms move on a 2-beat cadence → not on beat 1.
        self.assertNotIn(self._elbow, t)

    def test_nod_beat_dips_head_lift_down(self):
        # On a nod beat the head lift biases DOWN (lower qus) relative to gaze.
        with self._patch_baseline(), mock.patch.object(
            self.servos.random, "randint", return_value=150
        ):
            t = self.servos._listening_targets(2)
        self.assertLess(t[self._lift], self._baseline[self._lift])
        # Arms shift on the 2-beat cadence.
        self.assertIn(self._hand, t)


class ListeningLifecycleTests(unittest.TestCase):
    def setUp(self):
        from hardware import servos
        self.servos = servos
        servos.stop_listening_motion()  # clean slate
        servos.resume_arm_idle()

    def tearDown(self):
        self.servos.stop_listening_motion()
        self.servos.resume_arm_idle()

    def test_disabled_flag_is_a_noop(self):
        with (
            mock.patch.object(self.servos.config, "SERVO_LISTENING_MOTION_ENABLED", False),
            mock.patch.object(self.servos, "_gui_servo_sim_enabled", return_value=True),
        ):
            self.servos.start_listening_motion()
        self.assertFalse(self.servos.listening_motion_active())

    def test_does_not_start_while_speaking(self):
        self.servos._speech_active.set()
        try:
            with mock.patch.object(self.servos, "_gui_servo_sim_enabled", return_value=True):
                self.servos.start_listening_motion()
            self.assertFalse(self.servos.listening_motion_active())
        finally:
            self.servos._speech_active.clear()

    def test_start_sets_flag_and_pauses_arm_idle_then_stop_clears(self):
        # Patch the worker thread so no real motion loop runs during the test.
        with (
            mock.patch.object(self.servos, "_gui_servo_sim_enabled", return_value=True),
            mock.patch.object(self.servos.threading, "Thread") as Thread,
        ):
            Thread.return_value = mock.Mock()
            self.servos.start_listening_motion()
            self.assertTrue(self.servos.listening_motion_active())
            self.assertTrue(self.servos.arm_idle_paused())
            Thread.return_value.start.assert_called_once()

        self.servos.stop_listening_motion()
        self.assertFalse(self.servos.listening_motion_active())
        self.assertFalse(self.servos.arm_idle_paused())

    def test_begin_speech_motion_hands_off_from_listening(self):
        # Listening active, then Rex starts speaking → listening must yield.
        self.servos._listening_active.set()
        try:
            self.servos.begin_speech_motion("neutral")
            self.assertFalse(self.servos.listening_motion_active())
        finally:
            self.servos.end_speech_motion()
            self.servos._listening_active.clear()


if __name__ == "__main__":
    unittest.main()
