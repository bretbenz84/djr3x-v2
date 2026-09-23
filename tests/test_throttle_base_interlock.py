"""Base/arm ordering and failure tests, with no physical I/O."""
import json
import unittest
from unittest import mock

import config
from hardware import motion
from sequences import throttle_arm as arm
from tests import test_throttle_runtime as runtime


class TransportTest(unittest.TestCase):
    def setUp(self):
        for patch in (mock.patch.object(motion, '_ser'),
                      mock.patch.object(arm, 'prepare_base_motion', return_value=True),
                      mock.patch.object(arm, 'base_motion_sent'),
                      mock.patch('intelligence.voice_learning.mic_moved')):
            patch.start()
            self.addCleanup(patch.stop)

    def test_all_moving_commands_wait_for_park(self):
        for cmd in ('turn', 'move', 'come', 'drive', 'wheel'):
            motion._ser.reset_mock()
            def prepared():
                motion._ser.write.assert_not_called()
                return True
            arm.prepare_base_motion.side_effect = prepared
            seq = motion.send({'cmd': cmd, 'lin': .2})
            self.assertIsNotNone(seq)
            arm.base_motion_sent.assert_called_with(seq)
            self.assertEqual(json.loads(motion._ser.write.call_args.args[0])['cmd'], cmd)

    def test_failure_or_exception_never_sends(self):
        for result in (False, RuntimeError('serial failed')):
            arm.prepare_base_motion.side_effect = result if isinstance(result, Exception) else None
            arm.prepare_base_motion.return_value = False
            self.assertIsNone(motion.send({'cmd': 'turn', 'deg': 90}))
            motion._ser.write.assert_not_called()

    def test_stops_bypass_guard_and_cancel_pending_motion(self):
        for obj in ({'cmd': 'stop'}, {'cmd': 'estop'}, {'cmd': 'drive', 'lin': 0, 'ang': 0}):
            motion._ser.reset_mock()
            arm.prepare_base_motion.side_effect = lambda: (motion.send(obj), True)[1]
            self.assertIsNone(motion.send({'cmd': 'turn', 'deg': 90}))
            self.assertEqual(motion._ser.write.call_count, 1)
            self.assertEqual(json.loads(motion._ser.write.call_args.args[0])['cmd'], obj['cmd'])


class ArmInterlockTest(runtime.RuntimeTest):
    def test_worker_retracts_and_holds_during_speech(self):
        self.assertTrue(arm.start())
        self.assertTrue(arm.prepare_base_motion(timeout=3))
        controller = arm._controller
        self.assertTrue(controller.base_hold)
        self.assertEqual(self.port.pose, arm.PARK)
        arm.speech_start()
        arm.introduction()
        self.assertEqual(self.port.pose, arm.PARK)
        with mock.patch('hardware.motion.telemetry', return_value={'state': 'idle'}):
            controller.stop_event.wait(.2)
            self.assertTrue(controller.base_hold)
        arm.stop()
        self.assertFalse(arm.prepare_base_motion(timeout=.1))

    def test_base_packet_follows_worker_retraction_and_settling(self):
        self.assertTrue(arm.start())
        base = mock.Mock()
        def write(packet):
            self.assertEqual(self.port.pose, arm.PARK)
            self.assertTrue(arm._controller.base_ready.is_set())
            return len(packet)
        base.write.side_effect = write
        with mock.patch.object(motion, '_ser', base), mock.patch('intelligence.voice_learning.mic_moved'):
            seq = motion.send({'cmd': 'turn', 'deg': 90})
        self.assertIsNotNone(seq)
        base.write.assert_called_once()
        self.assertEqual(arm._controller.base_seq, seq)
        self.assertTrue(arm._controller.base_hold)

    def test_fresh_idle_resumes_but_stale_idle_does_not(self):
        import time
        self.assertTrue(arm.start())
        self.assertTrue(arm.prepare_base_motion(timeout=3))
        controller = arm._controller
        arm.base_motion_sent(42)
        with controller.lock:
            controller.base_sent_at = time.monotonic() - 1
        with mock.patch('hardware.motion.telemetry', return_value={
                'state': 'idle', 'owner': 'auto', 'cmd_seq': 41,
                'rx_monotonic': time.monotonic()}):
            controller.stop_event.wait(.2)
            self.assertTrue(controller.base_hold)
        with mock.patch('hardware.motion.telemetry', side_effect=lambda: {
                'state': 'idle', 'owner': 'auto', 'cmd_seq': 42,
                'rx_monotonic': time.monotonic()}):
            deadline = time.monotonic() + 2
            while controller.base_hold and time.monotonic() < deadline:
                controller.stop_event.wait(.02)
            self.assertFalse(controller.base_hold)
            self.assertFalse(controller.base_ready.is_set())

    def test_enabled_arm_without_worker_blocks(self):
        self.assertFalse(arm.prepare_base_motion(timeout=.01))
        with mock.patch.object(config, 'THROTTLE_ARM_ENABLED', False):
            self.assertTrue(arm.prepare_base_motion())
