import unittest
from unittest import mock


class ListeningTargetsTests(unittest.TestCase):
    """Smooth listening pose targets: in range, orbiting the gaze, and CONTINUOUS —
    consecutive ticks may only differ by small deltas (the anti-stutter property)."""

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
        servos._listening_session.clear()
        self.addCleanup(servos._listening_session.clear)

    def _patch_baseline(self):
        return mock.patch.object(self.servos, "_face_tracking_baseline", dict(self._baseline))

    def test_targets_stay_within_channel_limits(self):
        with self._patch_baseline():
            self.servos._begin_listening_session(100.0)
            for i in range(0, 200):
                targets = self.servos._listening_targets(100.0 + i * 0.12)
                for ch, pos in targets.items():
                    name = self.servos._CHANNEL_TO_NAME[ch]
                    cfg = self.servos.config.SERVO_CHANNELS[name]
                    self.assertGreaterEqual(pos, cfg["min"], f"{name} below min at tick {i}")
                    self.assertLessEqual(pos, cfg["max"], f"{name} above max at tick {i}")

    def test_all_channels_present_every_tick(self):
        with self._patch_baseline():
            self.servos._begin_listening_session(50.0)
            for i in range(5):
                t = self.servos._listening_targets(50.0 + i * 0.12)
                for ch in (self._neck, self._lift, self._tilt, self._visor,
                           self._elbow, self._hand, self._hero):
                    self.assertIn(ch, t)

    def test_consecutive_ticks_move_smoothly(self):
        # The anti-stutter property: at the streaming tick rate, no channel may jump
        # more than a small glide step between consecutive targets.
        with self._patch_baseline():
            self.servos._begin_listening_session(200.0)
            prev = None
            for i in range(0, 120):
                t = self.servos._listening_targets(200.0 + i * 0.12)
                if prev is not None:
                    for ch, pos in t.items():
                        name = self.servos._CHANNEL_TO_NAME[ch]
                        self.assertLessEqual(
                            abs(pos - prev[ch]), 90,
                            f"{name} jumped {abs(pos - prev[ch])} qus between ticks",
                        )
                prev = t

    def test_nod_dips_head_lift_down_and_tilts_down(self):
        # Force a nod mid-flight: at the nod's midpoint the lift is well below the
        # gaze baseline and the (inverted) tilt biases toward looking down.
        with self._patch_baseline():
            self.servos._begin_listening_session(300.0)
            s = self.servos._listening_session
            s["nod_started_at"] = 300.0
            nod_mid = 300.0 + float(self.servos.config.SERVO_LISTENING_NOD_SECS) / 2.0
            t = self.servos._listening_targets(nod_mid)
        self.assertLess(t[self._lift], self._baseline[self._lift] - 50)
        self.assertGreater(t[self._tilt], self._baseline[self._tilt])

    def test_nod_schedule_advances(self):
        # A completed nod clears its start and books the next one in the future.
        with self._patch_baseline():
            self.servos._begin_listening_session(400.0)
            s = self.servos._listening_session
            s["nod_started_at"] = 400.0
            done_at = 400.0 + float(self.servos.config.SERVO_LISTENING_NOD_SECS) + 0.1
            self.servos._listening_targets(done_at)
        self.assertIsNone(s["nod_started_at"])
        self.assertGreater(s["next_nod_at"], done_at)


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
