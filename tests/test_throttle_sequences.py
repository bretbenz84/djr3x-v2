import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tools import rex_servo_menubar as helper
from tools.throttle_sequences import Recorder, load_sequence, require_start_pose


class SequenceTest(unittest.TestCase):
    def setUp(self):
        self.limits = {cfg['ch']: cfg for cfg in helper._SERVO_DEFAULTS.values()}
        self.pose = {8: 9088, 9: 9984, 10: 2048}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        helper._recorder = helper._playback = None
        helper._measurement_mode.clear()
        helper._playback_stop.clear()
        self.addCleanup(helper._cancel_sequence_state)
        self.port_guard = mock.patch('serial.Serial', side_effect=AssertionError('No robot IO in tests'))
        self.port_guard.start()
        self.addCleanup(self.port_guard.stop)

    def saved(self):
        recorder = Recorder('park to reach', self.pose, self.limits, clock=mock.Mock(side_effect=[100, 101, 104]))
        recorder.record(8, 6000, 30, 6)
        return recorder.save(dict(self.pose, **{}), self.limits, self.directory)

    def test_saved_sequence_preserves_timing_profiles_and_poses(self):
        data = load_sequence(self.saved(), self.limits)
        self.assertEqual(data['start_pose'], self.pose)
        self.assertEqual(data['duration'], 4)
        self.assertEqual(data['events'], [dict(at=1, channel=8, target=6000, speed=30, acceleration=6)])

    def test_invalid_start_is_read_only_and_refused(self):
        data = load_sequence(self.saved(), self.limits)
        require_start_pose(data, self.pose, self.limits)
        for pose in ({8: 0, 9: 9984, 10: 2048}, {8: 6000, 9: 9984, 10: 2048}):
            with self.assertRaises(ValueError):
                require_start_pose(data, pose, self.limits)

    def test_reject_non_throttle_unlimited_speed_bad_timing_and_limits(self):
        path = self.saved()
        original = json.loads(path.read_text())
        for key, value in [('channel', 0), ('speed', 0), ('target', 10000), ('at', -1)]:
            data = json.loads(json.dumps(original))
            data['events'][0][key] = value
            path.write_text(json.dumps(data))
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_sequence(path, self.limits)

    def test_start_recording_and_save_never_move_hardware(self):
        ser = mock.Mock()
        with (mock.patch.object(helper, '_sequence_pose', return_value=self.pose),
              mock.patch.object(helper, '_SEQUENCE_DIR', self.directory)):
            helper._sequence_request(ser, self.limits, ('start', 'test'))
            helper._recorder.record(9, 9000, 70, 12)
            helper._sequence_request(ser, self.limits, ('save', None))
        ser.write.assert_not_called()
        self.assertEqual(len(list(self.directory.glob('*.json'))), 1)

    def test_only_successful_throttle_writes_are_recorded_with_actual_slow_profile(self):
        helper._recorder = Recorder('demo', self.pose, self.limits)
        helper._measurement_mode.set()
        self.addCleanup(helper._measurement_mode.clear)
        ser = mock.Mock()
        ser.write.side_effect = len
        helper._write_target(ser, self.limits[8], 6000)
        event = helper._recorder.data['events'][0]
        self.assertEqual((event['speed'], event['acceleration']), (10, 2))
        ser.write.side_effect = OSError('disconnected')
        with self.assertRaises(OSError):
            helper._write_target(ser, self.limits[9], 9000)
        self.assertEqual(len(helper._recorder.data['events']), 1)

    def test_play_request_requires_start_pose_and_does_not_move_immediately(self):
        ser = mock.Mock()
        path = self.saved()
        with mock.patch.object(helper, '_sequence_pose', return_value={**self.pose, 8: 6000}):
            with self.assertRaises(ValueError):
                helper._sequence_request(ser, self.limits, ('play', path))
        self.assertIsNone(helper._playback)
        with mock.patch.object(helper, '_sequence_pose', return_value=self.pose):
            helper._sequence_request(ser, self.limits, ('play', path))
        ser.write.assert_not_called()

    def test_playback_waits_until_event_and_uses_recorded_profile(self):
        data = load_sequence(self.saved(), self.limits)
        helper._playback = dict(data=data, started=100, index=0)
        ser = mock.Mock()
        with mock.patch.object(helper, '_write_target') as write:
            with mock.patch.object(helper.time, 'monotonic', return_value=100.5):
                helper._playback_tick(ser, self.limits)
            write.assert_not_called()
            with mock.patch.object(helper.time, 'monotonic', return_value=101.01):
                helper._playback_tick(ser, self.limits)
            self.assertEqual(write.call_args.kwargs, {'playback': True})
            self.assertEqual(write.call_args.args[1]['speed'], 30)

    def test_stop_cancels_future_events_and_holds_without_disabling_torque(self):
        helper._playback = dict(data=load_sequence(self.saved(), self.limits), started=0, index=0)
        helper._playback_stop.set()
        ser = mock.Mock()
        ser.write.side_effect = len
        with mock.patch.object(helper, '_read_positions', return_value=self.pose):
            helper._playback_tick(ser, self.limits)
        self.assertIsNone(helper._playback)
        self.assertEqual(ser.write.call_args_list, [mock.call(helper._encode_set_target(ch, self.pose[ch])) for ch in (8, 9, 10)])

    def test_late_playback_does_not_burst_commands(self):
        helper._playback = dict(data=load_sequence(self.saved(), self.limits), started=100, index=0)
        ser = mock.Mock()
        with mock.patch.object(helper.time, 'monotonic', return_value=105), self.assertRaises(ValueError):
            helper._playback_tick(ser, self.limits)
        ser.write.assert_not_called()

    def test_discard_does_not_unlock_manual_controls_during_playback(self):
        helper._playback = {'active': True}
        with self.assertRaises(ValueError):
            helper._sequence_request(mock.Mock(), self.limits, ('discard', None))
        self.assertIsNotNone(helper._playback)


if __name__ == '__main__':
    unittest.main()
