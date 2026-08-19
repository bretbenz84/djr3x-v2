import contextlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


@contextlib.contextmanager
def _acquired_output_gate(_name, **_kw):
    # **_kw absorbs the bounded-acquire timeout: every TTS hold passes one now,
    # so a wedged holder can never mute Rex for the rest of the session.
    yield True


class TtsLedCleanupTests(unittest.TestCase):
    def test_mouth_stop_failure_does_not_skip_tts_cleanup(self):
        from audio import tts
        from world_state import world_state

        fake_sd = types.SimpleNamespace(
            play=mock.Mock(),
            wait=mock.Mock(),
        )
        old_sd = sys.modules.get("sounddevice")
        old_self_state = world_state.get("self_state")
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
            world_state.update("self_state", old_self_state)

        chest_active.assert_called_once()
        servo_stop.assert_called_once()
        activity_stop.assert_called_once()
        self.assertTrue(any(call.args and call.args[0] is False for call in set_playing.call_args_list))

    def test_play_maps_semantic_emotion_into_shared_frame(self):
        from audio import tts
        from world_state import world_state

        fake_sd = types.SimpleNamespace(
            play=mock.Mock(),
            wait=mock.Mock(),
        )
        old_sd = sys.modules.get("sounddevice")
        old_self_state = world_state.get("self_state")
        sys.modules["sounddevice"] = fake_sd
        try:
            with (
                mock.patch.object(tts.output_gate, "hold", side_effect=_acquired_output_gate),
                mock.patch.object(tts.animations, "speech_activity_start"),
                mock.patch.object(tts.animations, "speech_activity_stop"),
                mock.patch.object(tts.servos, "begin_speech_motion") as begin_motion,
                mock.patch.object(tts.servos, "end_speech_motion"),
                mock.patch.object(tts.servos, "speech_reactive_move"),
                mock.patch.object(tts.leds_head, "speak") as head_speak,
                mock.patch.object(tts.leds_head, "speak_level"),
                mock.patch.object(tts.leds_head, "speak_stop"),
                mock.patch.object(tts.leds_chest, "speak") as chest_speak,
                mock.patch.object(tts.leds_chest, "active"),
                mock.patch.object(tts.echo_cancel, "set_playing"),
                mock.patch.object(tts.echo_cancel, "was_canceled", return_value=False),
            ):
                tts._play(np.zeros(1, dtype=np.float32), 100, "surprised")
        finally:
            if old_sd is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = old_sd
            world_state.update("self_state", old_self_state)

        frame = begin_motion.call_args.args[0]
        self.assertEqual(frame.affect, "surprised")
        self.assertEqual(frame.body_beat, "surprise_pop")
        head_speak.assert_called_once_with("excited")
        chest_speak.assert_called_once_with("excited")

    def test_shutdown_state_turns_leds_off_after_tts(self):
        from audio import tts
        from world_state import world_state

        fake_sd = types.SimpleNamespace(
            play=mock.Mock(),
            wait=mock.Mock(),
        )
        old_sd = sys.modules.get("sounddevice")
        old_self_state = world_state.get("self_state")
        sys.modules["sounddevice"] = fake_sd
        try:
            with (
                mock.patch.object(tts.output_gate, "hold", side_effect=_acquired_output_gate),
                mock.patch.object(tts.animations, "speech_activity_start"),
                mock.patch.object(tts.animations, "speech_activity_stop"),
                mock.patch.object(tts.servos, "begin_speech_motion"),
                mock.patch.object(tts.servos, "end_speech_motion"),
                mock.patch.object(tts.servos, "speech_reactive_move"),
                mock.patch.object(tts.leds_head, "speak"),
                mock.patch.object(tts.leds_head, "speak_level"),
                mock.patch.object(tts.leds_head, "speak_stop") as head_speak_stop,
                mock.patch.object(tts.leds_head, "off") as head_off,
                mock.patch.object(tts.leds_chest, "speak"),
                mock.patch.object(tts.leds_chest, "active") as chest_active,
                mock.patch.object(tts.leds_chest, "off") as chest_off,
                mock.patch.object(tts, "_is_shutdown_state", return_value=True),
                mock.patch.object(tts.echo_cancel, "set_playing"),
                mock.patch.object(tts.echo_cancel, "was_canceled", return_value=False),
            ):
                tts._play(np.zeros(1, dtype=np.float32), 100, "neutral")
        finally:
            if old_sd is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = old_sd
            world_state.update("self_state", old_self_state)

        head_off.assert_called_once()
        chest_off.assert_called_once()
        head_speak_stop.assert_not_called()
        chest_active.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class LedChunkingTests(unittest.TestCase):
    """A clone take renders as ONE unit, so playback receives the whole line as a
    single array. Driving the mouth once per received chunk then meant one RMS
    for the entire bit: motionless head, dead mouth LEDs, and a blocking write
    that barge-in could not interrupt."""

    def test_whole_take_is_resliced_into_led_frames(self):
        from audio import tts

        sr = 24000
        whole = np.zeros(int(sr * 12.0), dtype=np.float32)
        pieces = list(tts._led_chunks(whole, sr))
        self.assertGreater(len(pieces), 100)
        # ~30 fps by default, so a 12 s line is a few hundred updates
        self.assertAlmostEqual(len(pieces), 12.0 / 0.033, delta=30)

    def test_no_samples_are_lost_or_duplicated(self):
        from audio import tts

        sr = 24000
        whole = np.arange(int(sr * 1.5), dtype=np.float32)
        rebuilt = np.concatenate(list(tts._led_chunks(whole, sr)))
        np.testing.assert_array_equal(rebuilt, whole)

    def test_small_chunks_pass_through_untouched(self):
        from audio import tts

        sr = 24000
        small = np.zeros(int(sr * 0.02), dtype=np.float32)
        pieces = list(tts._led_chunks(small, sr))
        self.assertEqual(len(pieces), 1)
        self.assertIs(pieces[0], small)
