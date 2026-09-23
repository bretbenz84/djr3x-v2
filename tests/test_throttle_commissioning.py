"""Offline commissioning checks: opening the robot serial port is forbidden."""
import copy
import unittest
from unittest import mock


class ThrottleCommissioningTest(unittest.TestCase):
    def setUp(self):
        self.serial_guard = mock.patch('serial.Serial', side_effect=AssertionError('No hardware in tests'))
        self.serial_guard.start()
        self.addCleanup(self.serial_guard.stop)
        import config
        from hardware import servos
        self.config, self.servos = config, servos
        mock.patch.object(config, 'THROTTLE_ARM_ENABLED', False).start()
        self.channels = copy.deepcopy(config.THROTTLE_SERVO_CHANNELS)
        self.channel_patch = mock.patch.object(config, 'THROTTLE_SERVO_CHANNELS', self.channels)
        self.channel_patch.start()
        self.addCleanup(self.channel_patch.stop)
        self.wire = mock.patch.object(servos, '_send_command_locked', return_value=True).start()
        self.addCleanup(mock.patch.stopall)

    def settings(self, values):
        with mock.patch.object(self.config, '_servo_env_raw', side_effect=lambda key: values.get(key, '')):
            return self.config._throttle_startup_settings()

    def test_channels_direction_and_default_disabled(self):
        self.assertEqual([c['ch'] for c in self.channels.values()], [8, 9, 10])
        self.assertEqual(self.settings({}), (False, None))
        self.assertNotIn(8, self.servos._ALL_CHANNELS)
        self.assertGreater(self.channels['throttle_shoulder']['down'], self.channels['throttle_shoulder']['up'])
        self.assertLess(self.channels['throttle_elbow']['down'], self.channels['throttle_elbow']['up'])
        self.assertGreater(self.channels['throttle_wrist']['down'], self.channels['throttle_wrist']['up'])
        with mock.patch.object(self.config, 'THROTTLE_SHOULDER_ENABLED', False):
            self.servos._apply_throttle_startup_locked()
        self.wire.assert_not_called()

    def test_elbow_clearance_measurements_and_conservative_boundaries(self):
        for shoulder, minimum in ((2280, 1546), (1702, 636), (1636, 500),
                                  (1702.25, 1546), (1636.25, 636), (535, 500)):
            with self.subTest(shoulder=shoulder):
                self.assertEqual(self.config.throttle_elbow_limits(int(shoulder * 4)),
                                 (minimum * 4, 2500 * 4))
        for invalid in (0, 534 * 4, 2281 * 4, float('nan')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.config.throttle_elbow_limits(invalid)

    def test_clearance_intersects_individual_elbow_limits(self):
        elbow = self.channels['throttle_elbow']
        elbow.update(min=600 * 4, max=2400 * 4)
        self.assertEqual(self.config.throttle_elbow_limits(1636 * 4), (2400, 9600))
        self.assertEqual(self.config.throttle_elbow_limits(2280 * 4), (6184, 9600))
        elbow['max'] = 1500 * 4
        with self.assertRaises(ValueError):
            self.config.throttle_elbow_limits(2280 * 4)

    def test_explicit_startup_required_and_validated(self):
        key = 'SERVO_THROTTLE_SHOULDER_STARTUP_US'
        enabled = {'SERVO_THROTTLE_SHOULDER_ENABLED': 'true'}
        for value in ('', '534', '2281', 'nan', 'invalid'):
            with self.subTest(value=value), self.assertRaises((RuntimeError, ValueError)):
                self.settings({**enabled, key: value})
        self.assertEqual(self.settings({**enabled, key: '1500'}), (True, 6000))

    def test_limits_cannot_widen_or_be_incomplete(self):
        lo = 'SERVO_THROTTLE_SHOULDER_MIN_US'
        hi = 'SERVO_THROTTLE_SHOULDER_MAX_US'
        for values in ({lo: '544'}, {lo: '500', hi: '2280'}, {lo: '1000', hi: '900'}):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.settings(values)
        self.settings({lo: '544', hi: '2272'})
        self.assertEqual(self.channels['throttle_shoulder']['min'], 2176)

    def test_startup_orders_profile_before_shoulder_target_only(self):
        with (mock.patch.object(self.config, 'THROTTLE_SHOULDER_ENABLED', True),
              mock.patch.object(self.config, 'THROTTLE_SHOULDER_STARTUP', 6000)):
            self.servos._apply_throttle_startup_locked()
        self.assertEqual(self.wire.call_args_list, [
            mock.call(bytes([0x89, 8, 6, 0])),
            mock.call(bytes([0x87, 8, 30, 0])),
            mock.call(bytes([0x84, 8, 112, 46])),
        ])

    def test_failed_profile_does_not_send_target(self):
        self.wire.return_value = False
        with (mock.patch.object(self.config, 'THROTTLE_SHOULDER_ENABLED', True),
              mock.patch.object(self.config, 'THROTTLE_SHOULDER_STARTUP', 6000),
              self.assertRaises(RuntimeError)):
            self.servos._apply_throttle_startup_locked()
        self.assertEqual(self.wire.call_count, 1)

    def test_connect_applies_opt_in_startup_without_opening_hardware(self):
        s = self.servos
        with (mock.patch.object(s, 'SERVOS_ENABLED', True),
              mock.patch.object(s, '_ser', None),
              mock.patch.object(s, '_open_serial_with_retries', return_value=mock.Mock()),
              mock.patch.object(s, '_apply_startup_motion_profile_locked'),
              mock.patch.object(s, '_assert_startup_rest_pose_locked', return_value={}),
              mock.patch.object(self.config, 'THROTTLE_SHOULDER_ENABLED', True),
              mock.patch.object(self.config, 'THROTTLE_SHOULDER_STARTUP', 6000)):
            self.assertTrue(s.connect())
        self.assertEqual(self.wire.call_count, 3)
        self.assertEqual(self.wire.call_args.args[0], bytes([0x84, 8, 112, 46]))

    def test_generic_paths_cannot_command_throttle_even_when_enabled(self):
        s = self.servos
        with (mock.patch.object(s, 'SERVOS_ENABLED', True),
              mock.patch.object(s, '_program_servo_updates_blocked', return_value=False),
              mock.patch.object(s._manual_override, 'is_set', return_value=True)):
            for ch in (8, 9, 10):
                calls = (
                    lambda: s.set_servo(ch, 6000),
                    lambda: s.set_servos({0: 6000, ch: 6000}),
                    lambda: s.move_to({ch: 6000}),
                    lambda: s.set_manual_servo(ch, 6000),
                    lambda: s._send_set_target(ch, 6000),
                    lambda: s.set_speed(ch, 0),
                    lambda: s.set_acceleration(ch, 0),
                )
                for call in calls:
                    with self.subTest(channel=ch, call=call), self.assertRaises(ValueError):
                        call()
        self.wire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
