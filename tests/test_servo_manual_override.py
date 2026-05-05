import unittest
from unittest import mock


class ServoManualOverrideTest(unittest.TestCase):
    def tearDown(self):
        from hardware import servos

        servos.set_manual_override_enabled(False)

    def test_programmatic_set_servo_is_blocked_during_manual_override(self):
        from hardware import servos

        servos.set_manual_override_enabled(True)
        with (
            mock.patch.object(servos, "SERVOS_ENABLED", True),
            mock.patch.object(servos, "_send_set_target") as send_target,
            mock.patch.object(servos, "_remember_positions") as remember,
            mock.patch.object(servos, "_record_servo_positions") as record,
        ):
            servos.set_servo(0, 6200)
            servos.set_servos({1: 6100})

        send_target.assert_not_called()
        remember.assert_not_called()
        record.assert_not_called()

    def test_manual_set_servo_bypasses_override_gate_and_records_pose(self):
        from hardware import servos

        servos.set_manual_override_enabled(True)
        with (
            mock.patch.object(servos, "SERVOS_ENABLED", True),
            mock.patch.object(servos, "_send_set_target") as send_target,
            mock.patch.object(servos, "_remember_positions") as remember,
            mock.patch.object(servos, "_record_servo_positions") as record,
        ):
            self.assertTrue(servos.set_manual_servo(0, 6200))

        send_target.assert_called_once_with(0, 6200)
        remember.assert_called_once_with({0: 6200})
        record.assert_called_once_with({0: 6200})

    def test_manual_set_servo_requires_manual_override(self):
        from hardware import servos

        servos.set_manual_override_enabled(False)
        with mock.patch.object(servos, "_send_set_target") as send_target:
            self.assertFalse(servos.set_manual_servo(0, 6200))

        send_target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
