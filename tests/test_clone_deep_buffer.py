"""Deep playback buffer while the local clone engine is busy (field 2026-08-19:
the ElevenLabs reply stuttered through a 16.7s Jimmy Carter render). Covers the
local_tts engine-busy counter and the playback_stream_kwargs switch."""

import threading
import unittest
from unittest import mock

import config
from audio import local_tts
from audio import tts
from features import organic_impersonation as organic


class EngineBusyTest(unittest.TestCase):
    def test_counter_nests_and_clears(self):
        self.assertFalse(local_tts.engine_busy())
        with local_tts._engine_busy():
            self.assertTrue(local_tts.engine_busy())
            with local_tts._engine_busy():
                self.assertTrue(local_tts.engine_busy())
            self.assertTrue(local_tts.engine_busy())
        self.assertFalse(local_tts.engine_busy())

    def test_generate_stream_marks_busy(self):
        seen = {}

        def fake_chunks(model, seg, ref, interval):
            seen["busy"] = local_tts.engine_busy()
            return iter(())

        with mock.patch.object(local_tts, "_ensure_model", return_value=object()), \
             mock.patch.object(local_tts, "_segment_chunks", side_effect=fake_chunks):
            list(local_tts.generate_stream("Hello there.", local_tts.VoiceRef("x.wav", "x", "rex")))
        self.assertTrue(seen["busy"])
        self.assertFalse(local_tts.engine_busy())

    def test_synthesize_unit_marks_busy_and_releases_on_error(self):
        with mock.patch.object(local_tts, "_ensure_model", return_value=object()), \
             mock.patch.object(local_tts, "_segment_chunks", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                local_tts._synthesize_unit("Hi.", local_tts.VoiceRef("x.wav", "x", "rex"))
        self.assertFalse(local_tts.engine_busy())


class PlaybackKwargsTest(unittest.TestCase):
    def setUp(self):
        organic.reset_state()
        self.addCleanup(organic.reset_state)
        tts.set_boot_deep_buffer(False)
        self.addCleanup(lambda: tts.set_boot_deep_buffer(False))

    def test_normal_kwargs_unchanged(self):
        kw = tts.playback_stream_kwargs()
        self.assertEqual(kw["blocksize"], int(getattr(config, "AUDIO_PLAYBACK_BLOCKSIZE", 4096)))
        self.assertEqual(kw["latency"], getattr(config, "AUDIO_PLAYBACK_LATENCY", "high"))

    def test_engine_busy_deepens_buffer(self):
        with local_tts._engine_busy():
            kw = tts.playback_stream_kwargs()
        self.assertEqual(kw["latency"], float(getattr(config, "AUDIO_PLAYBACK_CLONE_LATENCY_SECS", 1.2)))
        self.assertGreaterEqual(kw["blocksize"], int(getattr(config, "AUDIO_PLAYBACK_CLONE_BLOCKSIZE", 8192)))

    def test_pending_organic_impression_deepens_buffer(self):
        organic._pending = mock.Mock(cancelled=False)
        kw = tts.playback_stream_kwargs()
        self.assertEqual(kw["latency"], float(getattr(config, "AUDIO_PLAYBACK_CLONE_LATENCY_SECS", 1.2)))
        organic._pending = None

    def test_kill_switch(self):
        with mock.patch.object(config, "AUDIO_PLAYBACK_CLONE_DEEP_BUFFER_ENABLED", False, create=True), \
             local_tts._engine_busy():
            kw = tts.playback_stream_kwargs()
        self.assertEqual(kw["latency"], getattr(config, "AUDIO_PLAYBACK_LATENCY", "high"))

    def test_boot_window_still_wins(self):
        tts.set_boot_deep_buffer(True)
        with local_tts._engine_busy():
            kw = tts.playback_stream_kwargs()
        self.assertEqual(kw["latency"], float(getattr(config, "AUDIO_PLAYBACK_BOOT_LATENCY_SECS", 1.0)))


if __name__ == "__main__":
    unittest.main()
