import csv
import tempfile
from pathlib import Path
import unittest
from unittest import mock
from tools import rex_servo_menubar as helper


class ThrottleMenuTest(unittest.TestCase):
    def tearDown(self):
        helper._measurement_mode.clear()

    def test_measurement_mode_uses_slow_profile(self):
        helper._measurement_mode.set()
        serial = mock.Mock()
        serial.write.side_effect = len
        helper._write_target(serial, helper._SERVO_DEFAULTS['throttle_wrist'], 6000)
        self.assertEqual(serial.write.call_args_list[:2], [
            mock.call(bytes([0x89, 10, 3, 0])), mock.call(bytes([0x87, 10, 20, 0]))])

    def test_nudge_uses_readback_and_clamps_at_limit(self):
        helper._measurement_mode.set()
        serial = mock.Mock()
        with (mock.patch.object(helper, '_require_stationary'),
              mock.patch.object(helper, '_read_positions', return_value={8: 9110}),
              mock.patch.object(helper, '_write_target') as write):
            self.assertEqual(helper._nudge(serial, helper._SERVO_DEFAULTS['throttle_shoulder'], 5), 9120)
        write.assert_called_once_with(serial, helper._SERVO_DEFAULTS['throttle_shoulder'], 9120)

    def test_nudge_refuses_off_servo_without_target(self):
        helper._measurement_mode.set()
        serial = mock.Mock()
        with (mock.patch.object(helper, '_require_stationary'),
              mock.patch.object(helper, '_read_positions', return_value={8: 0}),
              mock.patch.object(helper, '_write_target') as write,
              self.assertRaises(ValueError)):
            helper._nudge(serial, helper._SERVO_DEFAULTS['throttle_shoulder'], 1)
        write.assert_not_called()

    def test_record_saves_fresh_pose_and_note_without_motion(self):
        serial = mock.Mock()
        serial.read.side_effect = [b'\x00', (9120).to_bytes(2, 'little'),
                                   (10000).to_bytes(2, 'little'), (2000).to_bytes(2, 'little'), b'\x00']
        channels = {cfg['ch']: cfg for cfg in helper._SERVO_DEFAULTS.values()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurements.csv'
            helper._capture_measurement(serial, channels, 'floor, visible gap', path)
            with path.open() as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual([row[k] for k in ('shoulder_us', 'elbow_us', 'wrist_us')], ['2280.0', '2500.0', '500.0'])
            self.assertEqual(row['note'], 'floor, visible gap')
        self.assertTrue(all(call.args[0][0] in (0x90, 0x93) for call in serial.write.call_args_list))

    def test_record_refuses_motion_or_missing_readback(self):
        channels = {cfg['ch']: cfg for cfg in helper._SERVO_DEFAULTS.values()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurements.csv'
            serial = mock.Mock()
            serial.read.return_value = b'\x01'
            with self.assertRaises(ValueError):
                helper._capture_measurement(serial, channels, '', path)
            with (mock.patch.object(helper, '_require_stationary'),
                  mock.patch.object(helper, '_read_positions', return_value={8: 9120}),
                  self.assertRaises(ValueError)):
                helper._capture_measurement(serial, channels, '', path)
            self.assertFalse(path.exists())

    def test_dialog_callback_is_deferred_and_timer_stops_before_open(self):
        rumps = mock.Mock()
        callback = mock.Mock()
        timer = helper._after_menu_closes(rumps, callback)
        callback.assert_not_called()
        timer.start.assert_called_once()
        fire = rumps.Timer.call_args.args[0]
        callback.side_effect = lambda: timer.stop.assert_called_once()
        fire(timer)
        callback.assert_called_once()

    def test_recording_dialog_promotes_background_app_before_opening(self):
        appkit = mock.Mock()
        app = appkit.NSApplication.sharedApplication.return_value
        rumps = mock.Mock()
        def create_window(*args, **kwargs):
            app.setActivationPolicy_.assert_called_once_with(appkit.NSApplicationActivationPolicyRegular)
            app.activateIgnoringOtherApps_.assert_called_once()
            return mock.Mock()
        rumps.Window.side_effect = create_window
        with mock.patch.dict('sys.modules', {'AppKit': appkit}):
            dialog = helper._pose_note_dialog(rumps)
        dialog._textfield.setEditable_.assert_called_once_with(True)
        dialog._textfield.selectText_.assert_called_once_with(None)
        dialog._alert.window.return_value.makeKeyAndOrderFront_.assert_called_once_with(None)

    def test_all_channels_and_directions(self):
        with mock.patch.object(helper, '_read_env_file', return_value={}):
            table = helper._servos()
        self.assertEqual([cfg['ch'] for cfg in table.values()], list(range(11)))
        self.assertEqual(table['throttle_shoulder']['min'], 535 * 4)
        self.assertEqual(table['throttle_shoulder']['max'], 2280 * 4)
        self.assertEqual(len(helper._THROTTLE_DIRECTIONS), 3)

    def test_override_limits_cannot_expand_or_be_incomplete(self):
        prefix = 'SERVO_THROTTLE_SHOULDER_'
        for values in ({prefix + 'MIN_US': '500', prefix + 'MAX_US': '2280'},
                       {prefix + 'MIN_US': '544'}):
            with mock.patch.object(helper, '_read_env_file', return_value=values):
                with self.assertRaises(ValueError):
                    helper._servos()

    def test_profile_precedes_clamped_target(self):
        serial = mock.Mock()
        serial.write.side_effect = lambda data: len(data)
        helper._write_target(serial, helper._SERVO_DEFAULTS['throttle_shoulder'], 10000)
        self.assertEqual(serial.write.call_args_list, [
            mock.call(bytes([0x89, 8, 6, 0])),
            mock.call(bytes([0x87, 8, 30, 0])),
            mock.call(helper._encode_set_target(8, 9120)),
        ])

    def test_failed_profile_prevents_target(self):
        serial = mock.Mock()
        serial.write.return_value = 0
        with self.assertRaises(OSError):
            helper._write_target(serial, helper._SERVO_DEFAULTS['throttle_shoulder'], 6000)
        self.assertEqual(serial.write.call_count, 1)

    def test_connect_position_reads_send_no_motion(self):
        serial = mock.Mock()
        serial.read.return_value = bytes([0, 0])
        positions = helper._read_positions(serial, list(range(11)))
        self.assertEqual(positions, dict.fromkeys(range(11), 0))
        self.assertEqual(serial.write.call_args_list,
                         [mock.call(bytes([0x90, ch])) for ch in range(11)])


if __name__ == '__main__':
    unittest.main()
