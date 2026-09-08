"""Speech-only names: preserve display spelling and bypass old MP3/WAV takes."""
import tempfile
import unittest
from unittest import mock

import numpy as np

import config
from audio import tts


class PronunciationTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(config, "ELEVENLABS_PRONUNCIATIONS", {"T'Joy": "Tee-Joy"}))

    def test_apostrophes_case_and_possessives(self):
        for name in ("T'Joy", "T’Joy", "T‘Joy", "TʼJoy", "t'joy", "T'JOY"):
            with self.subTest(name=name):
                self.assertEqual(tts._elevenlabs_pronunciation_text(f"{name}'s turn, {name}."),
                                 "Tee-Joy's turn, Tee-Joy.")

    def test_other_names_and_word_fragments_are_unchanged(self):
        text = "Joy, T'Pol, Not'Joy, and T'Joyful."
        self.assertEqual(tts._elevenlabs_pronunciation_text(text), text)

    def test_alias_is_speech_only_and_independent_of_audio_tags(self):
        text = "[excited] T'Joy, your turn."
        self.assertEqual(tts.strip_audio_tags(tts._normalize_for_speech(text)), "T'Joy, your turn.")
        with mock.patch.object(config, "TTS_V3_AUDIO_TAGS_ENABLED", False):
            synth, _ = tts._apply_audio_tags(text, "neutral", None, None)
        self.assertEqual(synth, "Tee-Joy, your turn.")

    def test_supported_request_stitching_uses_same_pronunciation(self):
        with mock.patch.object(config, "TTS_V3_STITCH_ENABLED", True):
            self.assertEqual(tts._stitch_previous_text("T'Joy, you're up.", "eleven_multilingual_v2"),
                             "Tee-Joy, you're up.")


class CacheAndDeliveryTest(unittest.TestCase):
    LINE = "T'Joy, pick a category and dollar value."

    def setUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        for name, value in {
            "TTS_CACHE_DIR": root, "TTS_MODEL_ID": "eleven_v3",
            "ELEVENLABS_PRONUNCIATIONS": {"T'Joy": "Tee-Joy"},
            "NO_AUDIO_MODE": False, "AUDIO_OUTPUT_SUPPRESSED": False,
        }.items():
            self.enterContext(mock.patch.object(config, name, value))
        self.enterContext(mock.patch.object(tts, "_use_local_backend", return_value=False))
        self.enterContext(mock.patch.object(tts.delivery, "allowed", return_value=True))
        self.enterContext(mock.patch("intelligence.gaze_engine.note_about_to_speak"))
        self.settings = tts._resolve_voice_settings("neutral", None)
        self.old = tts._cache_path(self.LINE, config.ELEVENLABS_VOICE_ID, config.TTS_MODEL_ID, self.settings)
        synth, settings = tts._apply_audio_tags(self.LINE, "neutral", None, self.settings)
        self.new = tts._cache_path(synth, config.ELEVENLABS_VOICE_ID, config.TTS_MODEL_ID, settings)
        self.assertNotEqual(self.old, self.new)

    def test_old_mp3_and_wav_are_not_cache_hits(self):
        self.old.write_bytes(b"old wrong pronunciation")
        self.old.with_suffix(".wav").write_bytes(b"old streamed pronunciation")
        self.assertFalse(tts.is_cached(self.LINE))
        self.new.with_suffix(".wav").write_bytes(b"correct streamed pronunciation")
        self.assertTrue(tts.is_cached(self.LINE))

    def test_prefill_fetches_corrected_text_and_playback_uses_same_key(self):
        self.old.write_bytes(b"old wrong pronunciation")
        with mock.patch.object(tts, "_fetch_from_api", return_value=b"new pronunciation") as fetch, \
             self.assertLogs(tts.logger, level="INFO") as logs:
            self.assertTrue(tts.ensure_cached(self.LINE))
        self.assertIn(self.LINE, " ".join(logs.output))
        self.assertNotIn("Tee-Joy", " ".join(logs.output))
        self.assertEqual(fetch.call_args.args[0], "Tee-Joy, pick a category and dollar value.")
        self.assertTrue(self.new.exists())
        self.assertTrue(tts.is_cached(self.LINE))
        with mock.patch.object(tts, "_fetch_from_api") as fetch, \
             mock.patch.object(tts, "_speak_streaming") as stream, \
             mock.patch.object(tts, "_read_audio", return_value=(np.ones(20), 44100)) as read, \
             mock.patch.object(tts, "_play"), \
             mock.patch.object(tts.conv_log, "log_rex") as log:
            tts.speak(self.LINE)
        read.assert_called_once_with(self.new)
        fetch.assert_not_called()
        stream.assert_not_called()
        log.assert_called_once_with(self.LINE)

    def test_streamed_cache_miss_gets_new_speech_but_original_display_text(self):
        self.old.with_suffix(".wav").write_bytes(b"old wrong pronunciation")
        with mock.patch.object(config, "TTS_STREAMING_PLAYBACK_ENABLED", True), \
             mock.patch.object(tts, "_speak_streaming", return_value=True) as stream:
            tts.speak(self.LINE)
        args = stream.call_args.args
        self.assertEqual(args[0], "Tee-Joy, pick a category and dollar value.")
        self.assertEqual(args[1], self.LINE)
        self.assertEqual(args[7], self.new)

    def test_buffered_cache_miss_logs_original_spelling(self):
        with mock.patch.object(config, "TTS_STREAMING_PLAYBACK_ENABLED", False), \
             mock.patch.object(tts, "_fetch_from_api", return_value=b"new pronunciation") as fetch, \
             mock.patch.object(tts, "_read_audio", return_value=(np.ones(20), 44100)), \
             mock.patch.object(tts, "_play"), \
             mock.patch.object(tts.conv_log, "log_rex") as log:
            tts.speak(self.LINE)
        self.assertEqual(fetch.call_args.args[0], "Tee-Joy, pick a category and dollar value.")
        log.assert_called_once_with(self.LINE)
        self.assertTrue(self.new.exists())

    def test_unrelated_lines_keep_existing_cache_keys(self):
        text = "Bret, your turn."
        with mock.patch.object(config, "ELEVENLABS_PRONUNCIATIONS", {}):
            before, settings = tts._apply_audio_tags(text, "neutral", None, self.settings)
        after, _ = tts._apply_audio_tags(text, "neutral", None, self.settings)
        self.assertEqual(tts._cache_path(before, "v", "m", settings),
                         tts._cache_path(after, "v", "m", settings))


if __name__ == "__main__":
    unittest.main()
