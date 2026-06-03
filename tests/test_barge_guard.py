"""Tests for the proactive-speech "yield the floor" guard.

Covers audio.barge_guard.user_speaking_now (the last-moment mic re-check) and
interaction._speak_proactive (pre-cache + abort-if-user-speaking), which together
stop Rex from talking over a reply that the user began during his TTS-generation
lag. See PROACTIVE_SPEECH_YIELD_* in config.
"""

import unittest
from unittest import mock

import numpy as np

from audio import barge_guard


class UserSpeakingNowTests(unittest.TestCase):
    def _audio(self, secs=0.6):
        # Non-silent placeholder; VAD is mocked, so contents don't matter.
        return np.ones(int(secs * 16000), dtype=np.float32)

    def test_returns_true_when_enough_speech_in_window(self):
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", return_value=[(0.1, 0.45)]):
            self.assertTrue(barge_guard.user_speaking_now(window_secs=0.6, min_speech_secs=0.1))

    def test_returns_false_for_brief_blip_below_threshold(self):
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", return_value=[(0.20, 0.23)]):
            self.assertFalse(
                barge_guard.user_speaking_now(window_secs=0.6, min_speech_secs=0.1, poll_secs=0.0)
            )

    def test_returns_false_when_no_speech(self):
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", return_value=[]):
            self.assertFalse(
                barge_guard.user_speaking_now(window_secs=0.6, min_speech_secs=0.1, poll_secs=0.0)
            )

    def test_forward_poll_catches_onset_after_initial_silence(self):
        # Look-back is silent at first; the user starts mid-poll and is caught on a
        # later sample (sleep stubbed so the loop spins without real waiting).
        segments = [[], [], [(0.0, 0.3)]]
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", side_effect=segments), \
             mock.patch.object(barge_guard.time, "sleep"):
            self.assertTrue(barge_guard.user_speaking_now(min_speech_secs=0.1, poll_secs=1.0))

    def test_no_poll_does_single_lookback_only(self):
        # poll_secs=0 must not loop even if speech would have appeared later.
        seg = mock.Mock(return_value=[])
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", seg):
            self.assertFalse(barge_guard.user_speaking_now(poll_secs=0.0))
        self.assertEqual(seg.call_count, 1)

    def test_returns_false_while_rex_playback_suppresses_mic(self):
        # Even with speech segments, a suppressed mic means the buffer holds Rex's
        # own voice — never treat that as the user talking.
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=True), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=self._audio()) as get_chunk, \
             mock.patch.object(barge_guard.vad, "get_speech_segments", return_value=[(0.0, 0.6)]):
            self.assertFalse(barge_guard.user_speaking_now())
            get_chunk.assert_not_called()

    def test_returns_false_on_empty_buffer(self):
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(barge_guard.stream, "get_audio_chunk", return_value=np.array([], dtype=np.float32)), \
             mock.patch.object(barge_guard.vad, "get_speech_segments", return_value=[(0.0, 0.6)]) as seg:
            self.assertFalse(barge_guard.user_speaking_now(poll_secs=0.0))
            seg.assert_not_called()

    def test_swallows_exceptions(self):
        with mock.patch.object(barge_guard.echo_cancel, "is_suppressed", side_effect=RuntimeError("boom")):
            self.assertFalse(barge_guard.user_speaking_now())


class SpeakProactiveTests(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction

    def test_yields_without_speaking_when_user_already_talking(self):
        ix = self.interaction
        with mock.patch.object(ix.config, "PROACTIVE_SPEECH_YIELD_ENABLED", True), \
             mock.patch.object(ix, "_text_only_mode", False), \
             mock.patch("audio.tts.ensure_cached", return_value=True), \
             mock.patch.object(ix.barge_guard, "user_speaking_now", return_value=True), \
             mock.patch.object(ix, "_speak_blocking") as speak:
            result = ix._speak_proactive("Hey, what are you into?", emotion="curious", label="idle_banter")
        self.assertFalse(result)
        speak.assert_not_called()

    def test_speaks_when_user_is_silent(self):
        ix = self.interaction
        with mock.patch.object(ix.config, "PROACTIVE_SPEECH_YIELD_ENABLED", True), \
             mock.patch.object(ix, "_text_only_mode", False), \
             mock.patch("audio.tts.ensure_cached", return_value=True) as ensure, \
             mock.patch.object(ix.barge_guard, "user_speaking_now", return_value=False), \
             mock.patch.object(ix, "_speak_blocking", return_value=True) as speak:
            result = ix._speak_proactive("A strong Rex opinion.", emotion="curious", label="idle_banter")
        self.assertTrue(result)
        ensure.assert_called_once()
        speak.assert_called_once()

    def test_guard_disabled_skips_check_and_speaks(self):
        ix = self.interaction
        with mock.patch.object(ix.config, "PROACTIVE_SPEECH_YIELD_ENABLED", False), \
             mock.patch.object(ix, "_text_only_mode", False), \
             mock.patch.object(ix.barge_guard, "user_speaking_now", return_value=True) as check, \
             mock.patch.object(ix, "_speak_blocking", return_value=True) as speak:
            result = ix._speak_proactive("line", emotion="neutral")
        self.assertTrue(result)
        check.assert_not_called()
        speak.assert_called_once()

    def test_text_only_mode_skips_check_and_speaks(self):
        ix = self.interaction
        with mock.patch.object(ix.config, "PROACTIVE_SPEECH_YIELD_ENABLED", True), \
             mock.patch.object(ix, "_text_only_mode", True), \
             mock.patch.object(ix.barge_guard, "user_speaking_now", return_value=True) as check, \
             mock.patch.object(ix, "_speak_blocking", return_value=True) as speak:
            result = ix._speak_proactive("line", emotion="neutral")
        self.assertTrue(result)
        check.assert_not_called()
        speak.assert_called_once()


if __name__ == "__main__":
    unittest.main()
