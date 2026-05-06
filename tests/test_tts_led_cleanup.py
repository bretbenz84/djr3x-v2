import contextlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


@contextlib.contextmanager
def _acquired_output_gate(_name):
    yield True


class TtsLedCleanupTests(unittest.TestCase):
    def test_mouth_stop_failure_does_not_skip_tts_cleanup(self):
        from audio import tts

        fake_sd = types.SimpleNamespace(
            play=mock.Mock(),
            wait=mock.Mock(),
        )
        old_sd = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = fake_sd
        try:
            with (
                mock.patch.object(tts.output_gate, "hold", side_effect=_acquired_output_gate),
                mock.patch.object(tts.animations, "speech_activity_start"),
                mock.patch.object(tts.animations, "speech_activity_stop") as activity_stop,
                mock.patch.object(tts.servos, "begin_speech_motion"),
                mock.patch.object(tts.servos, "end_speech_motion") as servo_stop,
                mock.patch.object(tts.servos, "speech_reactive_move"),
                mock.patch.object(tts.leds_head, "speak"),
                mock.patch.object(tts.leds_head, "speak_level"),
                mock.patch.object(tts.leds_head, "speak_stop", side_effect=RuntimeError("serial hiccup")),
                mock.patch.object(tts.leds_chest, "speak"),
                mock.patch.object(tts.leds_chest, "active") as chest_active,
                mock.patch.object(tts.echo_cancel, "set_playing") as set_playing,
                mock.patch.object(tts.echo_cancel, "was_canceled", return_value=False),
            ):
                tts._play(np.zeros(1, dtype=np.float32), 100, "neutral")
        finally:
            if old_sd is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = old_sd

        chest_active.assert_called_once()
        servo_stop.assert_called_once()
        activity_stop.assert_called_once()
        self.assertTrue(any(call.args and call.args[0] is False for call in set_playing.call_args_list))


if __name__ == "__main__":
    unittest.main()
