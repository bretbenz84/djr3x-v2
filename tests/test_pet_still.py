"""Photo capture ordering; all servo I/O and clock waits are mocked."""
import contextlib
import unittest
from unittest import mock

import numpy as np
import config
from hardware import servos as S
from vision import camera as C


class HeadHoldTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for patch in (
            mock.patch.object(S, "SERVOS_ENABLED", True),
            mock.patch.object(S, "_program_servo_updates_blocked", return_value=False),
            mock.patch.object(S, "_photo_hold", None),
            mock.patch.object(S, "_record_servo_positions"),
            mock.patch.object(S, "_commanded_positions", dict(S._commanded_positions)),
        ):
            self.stack.enter_context(patch)
        self.positions = {config.SERVO_CHANNELS[n]["ch"]: config.SERVO_CHANNELS[n]["neutral"]
                          for n in ("neck", "headlift", "headtilt")}
        self.stack.enter_context(mock.patch.object(S, "get_servo", side_effect=self.positions.get))
        self.wire = self.stack.enter_context(mock.patch.object(S, "_send_command_locked", return_value=True))

    def test_tracking_and_direct_animation_targets_cannot_move_held_head(self):
        cfg = config.SERVO_CHANNELS["neck"]
        ch, held = cfg["ch"], self.positions[cfg["ch"]]
        with S.hold_head_for_photo():
            S.set_servo(ch, cfg["max"])
            self.assertEqual(self.wire.call_args.args[0], S._encode(S._CMD_SET_TARGET, ch, held))
            S._send_set_target(ch, cfg["min"])
            self.assertEqual(self.wire.call_args.args[0], S._encode(S._CMD_SET_TARGET, ch, held))
            S.set_servos({ch: cfg["max"]})
            self.assertEqual(S._commanded_positions[ch], held)
            visor = config.SERVO_CHANNELS["visor"]
            S.set_servo(visor["ch"], visor["max"])
            self.assertEqual(self.wire.call_args.args[0], S._encode(S._CMD_SET_TARGET, visor["ch"], visor["max"]))
        S.set_servo(ch, cfg["max"])
        self.assertEqual(self.wire.call_args.args[0], S._encode(S._CMD_SET_TARGET, ch, cfg["max"]))

    def test_exception_releases_hold(self):
        with self.assertRaises(ValueError):
            with S.hold_head_for_photo():
                raise ValueError("camera failed")
        self.assertIsNone(S._photo_hold)

    def test_lease_expires_even_if_caller_stalls(self):
        with S.hold_head_for_photo():
            ch = config.SERVO_CHANNELS["neck"]["ch"]
            with mock.patch.object(S.time, "monotonic", return_value=S._photo_hold[0] + 1):
                self.assertEqual(S._voice_hold_position(ch, 123), 123)

    def test_dev_capture_does_not_write_servos(self):
        with mock.patch.object(S, "SERVOS_ENABLED", False):
            with S.hold_head_for_photo():
                pass
        self.wire.assert_not_called()


class FreshFrameTests(unittest.TestCase):
    def capture(self, fresh):
        clock = [10.]
        held = [False]
        image = np.ones((60, 80, 3), dtype=np.uint8)
        @contextlib.contextmanager
        def hold(*args):
            held[0] = True
            try:
                yield
            finally:
                held[0] = False
        def sleep(seconds):
            self.assertTrue(held[0])
            clock[0] += seconds
            if fresh:
                C._last_frame_at = clock[0]
        with mock.patch.object(C, "CAMERA_ENABLED", True), \
             mock.patch.object(C, "_frame", image), \
             mock.patch.object(C, "_last_frame_at", 9.), \
             mock.patch.object(C.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(C.time, "sleep", side_effect=sleep), \
             mock.patch.object(S, "hold_head_for_photo", side_effect=hold):
            result = C.capture_pet_still()
        self.assertFalse(held[0])
        self.assertGreater(clock[0], 10.6)
        return result

    def test_capture_waits_for_frame_after_settle(self):
        self.assertTrue(np.all(self.capture(True) == 1))

    def test_stopped_camera_cannot_return_stale_frame(self):
        self.assertIsNone(self.capture(False))
