"""Autonomous motion must not run while the obstacle sensing is dead.

Field 2026-08-07..08-11: the front 8x8 matrix ToF failed electrically and Rex
drove into walls for four days. safety.cpp fails OPEN on -1 by design and the
radial ring stayed alive, so nothing stopped him. These tests pin the host-side
block, including the parts that are easy to regress: stop/estop and operator
teleop must stay usable, and recovery must not be instant.
"""

import unittest
from unittest import mock

import config
from intelligence import motion_controller as mc


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, secs):
        self.now += secs


class TofGateTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        # Fresh gate state per test — these are module globals by design (one base).
        mc._tof_healthy_since = 0.0
        mc._tof_fault_since = 0.0
        mc._tof_warned = False
        mc._tof_block_reason = "tof_startup"
        mc._tof_cut_for_reason = None
        mc._tof_announced_at = 0.0
        mc._user_commanded_motion_at = 0.0
        self._patches = [
            mock.patch.object(mc.time, "monotonic", self.clock),
            mock.patch.object(mc, "charging", return_value=False),
            mock.patch.object(mc.motion, "connected", return_value=True),
            mock.patch.object(mc.motion, "owner", return_value="auto"),
            mock.patch.object(mc.motion, "state", return_value="idle"),
            mock.patch.object(mc.motion, "tof_matrix", return_value={"g": "x"}),
            mock.patch.object(mc.motion, "radial_tof_alive", return_value=(8, 8)),
            mock.patch.object(mc.state_module, "get_state", return_value=mc.State.ACTIVE),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _settle_healthy(self):
        """Walk past the recovery window so the gate is open."""
        mc.tof_block_reason()
        self.clock.advance(config.MOTION_TOF_RECOVERY_SECS + 0.1)
        self.assertIsNone(mc.tof_block_reason())

    # ── the block itself ────────────────────────────────────────────────────────

    def test_dead_matrix_blocks_autonomous_motion(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None      # the Aug 7 failure exactly
        self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")
        self.assertEqual(mc._autonomous_allowed(), "tof_matrix_down")

    def test_dead_radial_ring_blocks_autonomous_motion(self):
        self._settle_healthy()
        mc.motion.radial_tof_alive.return_value = (0, 8)
        self.assertEqual(mc.tof_block_reason(), "tof_ring_down")

    def test_empty_room_is_not_a_dead_ring(self):
        """An empty room reads CLEAR, not -1 — that must not look like blindness."""
        self._settle_healthy()
        mc.motion.radial_tof_alive.return_value = (8, 8)
        self.assertIsNone(mc.tof_block_reason())

    def test_no_telemetry_yet_does_not_fake_a_dead_ring(self):
        self._settle_healthy()
        mc.motion.radial_tof_alive.return_value = (0, 0)   # nothing reported yet
        self.assertIsNone(mc.tof_block_reason())

    def test_move_and_turn_and_come_all_refuse_while_blind(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(mc.motion, "send") as send:
            self.assertIsNone(mc.move(1.0))
            self.assertIsNone(mc.turn(90.0))
            self.assertIsNone(mc.come())
            self.assertIsNone(mc.drive(0.2, 0.0))
            self.assertIsNone(mc.arc(0.15, 0.3))
        send.assert_not_called()

    def test_blind_start_is_blocked_before_any_frame_arrives(self):
        mc.motion.tof_matrix.return_value = None
        self.assertEqual(mc._autonomous_allowed(), "tof_matrix_down")

    # ── what must NOT be gated ──────────────────────────────────────────────────

    def test_stop_and_estop_still_work_while_blind(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(mc.motion, "send", return_value=7) as send:
            self.assertEqual(mc.stop(), 7)
            self.assertEqual(mc.estop(), 7)
            self.assertEqual(mc.clear(), 7)
        self.assertEqual(
            [c.args[0]["cmd"] for c in send.call_args_list], ["stop", "estop", "clear"]
        )

    def test_operator_teleop_still_works_while_blind(self):
        """A human at the controls IS the obstacle sensing."""
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(mc.motion, "send", return_value=3) as send:
            self.assertEqual(mc.drive_manual(0.2, 0.0), 3)
        send.assert_called_once()

    def test_gamepad_ownership_keeps_its_own_reason(self):
        self._settle_healthy()
        mc.motion.owner.return_value = "manual"
        mc.motion.tof_matrix.return_value = None
        self.assertEqual(mc._autonomous_allowed(), "manual_override")

    def test_master_switch_disables_the_gate(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(config, "MOTION_REQUIRE_TOF_FOR_AUTONOMY", False):
            self.assertIsNone(mc.tof_block_reason())

    def test_matrix_can_be_dropped_from_the_bar_for_a_base_without_one(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(config, "MOTION_TOF_MATRIX_REQUIRED", False):
            self.assertIsNone(mc.tof_block_reason())

    # ── recovery hysteresis ─────────────────────────────────────────────────────

    def test_recovery_is_not_instant(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")
        mc.motion.tof_matrix.return_value = {"g": "x"}          # sensor is back
        self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")   # not yet trusted
        self.clock.advance(config.MOTION_TOF_RECOVERY_SECS - 0.1)
        self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")
        self.clock.advance(0.2)
        self.assertIsNone(mc.tof_block_reason())

    def test_a_flapping_sensor_never_opens_the_gate(self):
        """One good frame between dropouts must not ratchet him forward."""
        self._settle_healthy()
        for _ in range(20):
            mc.motion.tof_matrix.return_value = None
            self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")
            self.clock.advance(1.0)
            mc.motion.tof_matrix.return_value = {"g": "x"}
            self.assertEqual(mc.tof_block_reason(), "tof_matrix_down")
            self.clock.advance(1.0)

    # ── the alarm must not cry wolf ─────────────────────────────────────────────

    def test_a_normal_startup_logs_no_warning(self):
        """The heartbeat's first tick beats the first tofmx frame every launch —
        a WARNING there would train everyone to ignore the one that matters."""
        mc.motion.tof_matrix.return_value = None
        with self.assertLogs(mc._log, level="WARNING") as caught:
            mc.tof_block_reason()                      # blocked from the first look
            mc.motion.tof_matrix.return_value = {"g": "x"}   # frame lands 0.2s later
            self.clock.advance(0.2)
            mc.tof_block_reason()
            mc._log.warning("sentinel")                # assertLogs needs one record
        self.assertEqual([r.message for r in caught.records], ["sentinel"])

    def test_a_sensor_that_stays_dead_does_warn(self):
        mc.motion.tof_matrix.return_value = None
        mc.tof_block_reason()
        self.clock.advance(config.MOTION_TOF_RECOVERY_SECS + 0.1)
        with self.assertLogs(mc._log, level="WARNING") as caught:
            mc.tof_block_reason()
        self.assertIn("Obstacle sensing lost", caught.records[0].message)

    def test_the_warning_fires_once_per_outage(self):
        mc.motion.tof_matrix.return_value = None
        mc.tof_block_reason()
        self.clock.advance(config.MOTION_TOF_RECOVERY_SECS + 0.1)
        with self.assertLogs(mc._log, level="WARNING") as caught:
            for _ in range(10):
                self.clock.advance(1.0)
                mc.tof_block_reason()
            mc._log.warning("sentinel")
        self.assertEqual(len(caught.records), 2)       # the real one + the sentinel

    # ── cutting a leg already in flight ─────────────────────────────────────────

    def test_losing_sensing_mid_move_stops_the_base(self):
        self._settle_healthy()
        mc.motion.state.return_value = "moving"
        mc.motion.tof_matrix.return_value = None
        self.assertTrue(mc._tof_should_cut_inflight())

    def test_the_mid_move_cut_fires_once_per_outage(self):
        self._settle_healthy()
        mc.motion.state.return_value = "moving"
        mc.motion.tof_matrix.return_value = None
        self.assertTrue(mc._tof_should_cut_inflight())
        self.assertFalse(mc._tof_should_cut_inflight())
        self.assertFalse(mc._tof_should_cut_inflight())

    def test_no_cut_when_the_base_is_not_moving(self):
        self._settle_healthy()
        mc.motion.state.return_value = "idle"
        mc.motion.tof_matrix.return_value = None
        self.assertFalse(mc._tof_should_cut_inflight())

    def test_no_cut_while_a_human_is_driving(self):
        self._settle_healthy()
        mc.motion.state.return_value = "moving"
        mc.motion.owner.return_value = "manual"
        mc.motion.tof_matrix.return_value = None
        self.assertFalse(mc._tof_should_cut_inflight())

    def test_heartbeat_stops_the_base_when_sensing_dies_mid_leg(self):
        self._settle_healthy()
        mc.motion.state.return_value = "moving"
        mc.motion.tof_matrix.return_value = None
        with mock.patch.object(mc, "stop") as stop, mock.patch.object(mc.motion, "ping") as ping:
            mc._heartbeat_tick()
        stop.assert_called_once()
        ping.assert_not_called()

    # ── telling the human ───────────────────────────────────────────────────────

    def test_a_refused_voice_command_says_why(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        mc.note_user_commanded_motion()
        with mock.patch("audio.speech_queue.enqueue") as enqueue, \
             mock.patch.object(mc.motion, "send"):
            mc.move_forward(1.0)
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.kwargs.get("tag"), "motion_tof_blocked")

    def test_autonomous_legs_refuse_silently(self):
        """Exploration retries constantly — it must not narrate every attempt."""
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        with mock.patch("audio.speech_queue.enqueue") as enqueue, \
             mock.patch.object(mc.motion, "send"):
            for _ in range(5):
                mc.move(1.0)
        enqueue.assert_not_called()

    def test_the_spoken_refusal_has_a_cooldown(self):
        self._settle_healthy()
        mc.motion.tof_matrix.return_value = None
        mc.note_user_commanded_motion()
        with mock.patch("audio.speech_queue.enqueue") as enqueue, \
             mock.patch.object(mc.motion, "send"):
            mc.move_forward(1.0)
            mc.move_forward(1.0)
            mc.turn_left()
        self.assertEqual(enqueue.call_count, 1)

    def test_other_suppression_reasons_do_not_blame_the_sensor(self):
        self._settle_healthy()
        mc.charging.return_value = True
        mc.note_user_commanded_motion()
        with mock.patch("audio.speech_queue.enqueue") as enqueue, \
             mock.patch.object(mc.motion, "send"):
            mc.move_forward(1.0)
        enqueue.assert_not_called()


class RadialAliveTest(unittest.TestCase):
    """hardware.motion.radial_tof_alive encodes the firmware's -1 convention."""

    def _alive(self, tof_mm):
        from hardware import motion
        with mock.patch.object(motion, "telemetry", return_value={"tof_mm": tof_mm}):
            return motion.radial_tof_alive()

    def test_all_dead(self):
        self.assertEqual(self._alive({k: -1 for k in "ab cd ef gh".split()}), (0, 4))

    def test_clear_readings_count_as_alive(self):
        self.assertEqual(self._alive({"fl": 4000, "fr": 4000, "rl": -1}), (2, 3))

    def test_zero_is_a_real_reading_not_an_error(self):
        self.assertEqual(self._alive({"fl": 0}), (1, 1))

    def test_no_telemetry(self):
        from hardware import motion
        with mock.patch.object(motion, "telemetry", return_value=None):
            self.assertEqual(motion.radial_tof_alive(), (0, 0))


if __name__ == "__main__":
    unittest.main()
