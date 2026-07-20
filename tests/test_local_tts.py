"""
Tests for the on-device Qwen3-TTS backend: engine helpers, tts.py backend
dispatch, the automatic ElevenLabs->local fallback + circuit breaker, and the
local cache path. mlx-audio and the audio device are mocked — no model loads and
no real playback happen here.
"""

import unittest
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import config
from audio import local_tts, tts


REX_REF = local_tts.VoiceRef("/nonexistent/rex.wav", "reference text", "rex")
IMPERSONATION_REF = local_tts.VoiceRef("/nonexistent/carter.wav", "ref", "famous:jimmy-carter")


class _FakeStream:
    """Records writes instead of touching a real audio device."""

    def __init__(self, *a, **k):
        self.writes = []
        self.stopped = False
        self.aborted = False

    def start(self):
        pass

    def write(self, samples):
        self.writes.append(np.asarray(samples))

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True

    def close(self):
        pass


def _fake_gen_factory(n_chunks=3, samples_per=1200):
    def _gen(text, voice_ref):
        for _ in range(n_chunks):
            yield np.full(samples_per, 0.1, dtype=np.float32)
    return _gen


class EngineHelpersTest(unittest.TestCase):
    def test_split_line_short_passthrough(self):
        self.assertEqual(local_tts._split_line("Hi there."), ["Hi there."])

    def test_split_line_long_segments(self):
        long = "One sentence here. " * 20
        parts = local_tts._split_line(long)
        self.assertEqual(len(parts), 20)

    def test_split_line_empty(self):
        self.assertEqual(local_tts._split_line("   "), [])

    def test_is_available_false_without_weights(self):
        # Point the model dir at an empty temp dir → sentinel files absent.
        with TemporaryDirectory() as d, \
             mock.patch.object(local_tts, "_model_dir", return_value=__import__("pathlib").Path(d)):
            self.assertFalse(local_tts.is_available())

    def test_unavailable_reason_missing_model(self):
        from pathlib import Path
        with TemporaryDirectory() as d, \
             mock.patch.object(local_tts, "_model_dir", return_value=Path(d)):
            reason = local_tts.unavailable_reason()
            self.assertIsNotNone(reason)
            self.assertIn("setup_assets", reason)
            self.assertFalse(local_tts.is_available())

    def test_unavailable_reason_no_mlx_audio(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            reason = local_tts.unavailable_reason()
            self.assertIsNotNone(reason)
            self.assertIn("mlx-audio", reason)

    def test_unavailable_reason_find_spec_raises_is_surfaced(self):
        # A raising find_spec must be reported, not silently swallowed to False.
        with mock.patch("importlib.util.find_spec", side_effect=RuntimeError("boom")):
            reason = local_tts.unavailable_reason()
            self.assertIsNotNone(reason)
            self.assertIn("boom", reason)

    def test_unavailable_reason_requires_rex_ref_when_asked(self):
        # Model present but Rex's reference clip missing: fine for impersonation
        # (default), a named failure for --local-tts (require_rex_ref=True) —
        # the dev-mac silent-ElevenLabs-fallback failure mode.
        from pathlib import Path
        with TemporaryDirectory() as d:
            md = Path(d) / "model"
            (md / "speech_tokenizer").mkdir(parents=True)
            (md / "model.safetensors").write_bytes(b"x")
            (md / "speech_tokenizer" / "model.safetensors").write_bytes(b"x")
            with mock.patch.object(local_tts, "_model_dir", return_value=md), \
                 mock.patch.object(local_tts, "rex_voice_ref", return_value=None):
                self.assertIsNone(local_tts.unavailable_reason())
                reason = local_tts.unavailable_reason(require_rex_ref=True)
                self.assertIsNotNone(reason)
                self.assertIn("reference", reason)

    def test_voice_ref_from_files_missing(self):
        self.assertIsNone(
            local_tts.voice_ref_from_files("/no/a.wav", "/no/a.txt", "x")
        )


class LocalBackendDispatchTest(unittest.TestCase):
    def setUp(self):
        tts._note_api_success()  # clear breaker
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(config, "NO_AUDIO_MODE", False, create=True))
        self._stack.enter_context(
            mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False, create=True)
        )

    def tearDown(self):
        self._stack.close()
        tts._note_api_success()

    def test_local_mode_routes_local_strips_tags_skips_elevenlabs(self):
        seen = {}

        def fake_local(clean_text, ref, emotion, **kw):
            seen["text"] = clean_text
            seen["ref"] = ref
            return True

        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(local_tts, "rex_voice_ref", return_value=REX_REF), \
             mock.patch.object(tts, "_speak_local", side_effect=fake_local), \
             mock.patch.object(tts, "_get_el_client") as get_client, \
             mock.patch.object(tts, "_fetch_from_api") as fetch:
            tts.speak("Hello there [laughs]")

        self.assertEqual(seen.get("ref"), REX_REF)
        self.assertEqual(seen.get("text"), "Hello there")  # [laughs] stripped for Qwen
        get_client.assert_not_called()
        fetch.assert_not_called()

    def test_explicit_voice_ref_routes_local(self):
        seen = {}

        def fake_local(clean_text, ref, emotion, **kw):
            seen["ref"] = ref
            return True

        # No --local-tts mode; the explicit impersonation ref must still go local.
        with mock.patch.object(config, "LOCAL_TTS_MODE", False), \
             mock.patch.object(tts, "_speak_local", side_effect=fake_local), \
             mock.patch.object(tts, "_fetch_from_api") as fetch:
            tts.speak("I am the president.", voice_ref=IMPERSONATION_REF)

        self.assertEqual(seen.get("ref"), IMPERSONATION_REF)
        fetch.assert_not_called()

    def test_explicit_voice_ref_failure_does_not_speak_in_rex_voice(self):
        # Impersonation synth fails → must NOT fall through to ElevenLabs.
        with mock.patch.object(tts, "_speak_local", return_value=False), \
             mock.patch.object(tts, "_fetch_from_api") as fetch, \
             mock.patch.object(tts, "_speak_streaming") as streaming:
            tts.speak("parody line", voice_ref=IMPERSONATION_REF)
        fetch.assert_not_called()
        streaming.assert_not_called()

    def test_no_audio_short_circuits_before_local(self):
        with mock.patch.object(config, "NO_AUDIO_MODE", True), \
             mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(tts, "_speak_local") as local:
            tts.speak("hello")
        local.assert_not_called()

    def test_unavailable_model_falls_back_to_elevenlabs(self):
        # --local-tts requested but model not installed → ElevenLabs path runs.
        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(local_tts, "is_available", return_value=False), \
             mock.patch.object(tts, "_speak_local") as local, \
             mock.patch.object(tts, "_speak_streaming", return_value=True) as streaming:
            tts.speak("hello world")
        local.assert_not_called()
        streaming.assert_called_once()


class FallbackCircuitBreakerTest(unittest.TestCase):
    def setUp(self):
        tts._note_api_success()
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(config, "NO_AUDIO_MODE", False, create=True))
        self._stack.enter_context(
            mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False, create=True)
        )
        self._stack.enter_context(mock.patch.object(config, "LOCAL_TTS_FALLBACK_ENABLED", True))
        # Skip the streaming path so speak() reaches the buffered _fetch_from_api.
        self._stack.enter_context(mock.patch.object(config, "TTS_STREAMING_PLAYBACK_ENABLED", False))
        self._tmp = TemporaryDirectory()
        self._stack.enter_context(mock.patch.object(config, "TTS_CACHE_DIR", self._tmp.name))

    def tearDown(self):
        self._stack.close()
        self._tmp.cleanup()
        tts._note_api_success()

    def test_breaker_state_machine(self):
        self.assertFalse(tts._api_circuit_open())
        tts._note_api_failure()
        self.assertTrue(tts._api_circuit_open())
        tts._note_api_success()
        self.assertFalse(tts._api_circuit_open())

    def test_breaker_disabled_never_opens(self):
        with mock.patch.object(config, "LOCAL_TTS_FALLBACK_ENABLED", False):
            tts._note_api_failure()
            self.assertFalse(tts._api_circuit_open())

    def test_api_failure_triggers_local_fallback(self):
        seen = {}

        def failing_fetch(*a, **k):
            tts._note_api_failure()   # faithful: the real _fetch does this
            return None

        def fake_local(clean_text, ref, emotion, **kw):
            seen["text"] = clean_text
            return True

        with mock.patch.object(config, "LOCAL_TTS_MODE", False), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(local_tts, "rex_voice_ref", return_value=REX_REF), \
             mock.patch.object(tts, "_fetch_from_api", side_effect=failing_fetch), \
             mock.patch.object(tts, "_speak_local", side_effect=fake_local):
            tts.speak("weather please")

        self.assertEqual(seen.get("text"), "weather please")
        self.assertTrue(tts._api_circuit_open())

    def test_open_breaker_routes_next_line_directly_local(self):
        tts._note_api_failure()  # breaker already open
        with mock.patch.object(config, "LOCAL_TTS_MODE", False), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(local_tts, "rex_voice_ref", return_value=REX_REF), \
             mock.patch.object(tts, "_speak_local", return_value=True) as local, \
             mock.patch.object(tts, "_fetch_from_api") as fetch:
            tts.speak("second line")
        local.assert_called_once()
        fetch.assert_not_called()


class LocalCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._p = mock.patch.object(config, "TTS_CACHE_DIR", self._tmp.name)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def test_local_key_differs_from_elevenlabs(self):
        local = tts._local_cache_wav("hello")
        eleven = tts._cache_path("hello", config.ELEVENLABS_VOICE_ID, config.TTS_MODEL_ID)
        self.assertNotEqual(str(local), str(eleven.with_suffix(".wav")))

    def test_is_cached_local_checks_local_wav(self):
        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", True), \
             mock.patch.object(local_tts, "is_available", return_value=True):
            self.assertFalse(tts.is_cached("boot line"))
            wav = tts._local_cache_wav("boot line")
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"stub")
            self.assertTrue(tts.is_cached("boot line"))

    def test_is_cached_false_when_cache_disabled(self):
        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", False), \
             mock.patch.object(local_tts, "is_available", return_value=True):
            wav = tts._local_cache_wav("boot line")
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"stub")   # even a stray file must be ignored
            self.assertFalse(tts.is_cached("boot line"))

    def test_ensure_cached_local_uses_engine_not_elevenlabs(self):
        audio = np.full(2400, 0.2, dtype=np.float32)
        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", True), \
             mock.patch.object(config, "NO_AUDIO_MODE", False, create=True), \
             mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False, create=True), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(local_tts, "rex_voice_ref", return_value=REX_REF), \
             mock.patch.object(local_tts, "synthesize", return_value=(audio, 24000)), \
             mock.patch.object(tts, "_fetch_from_api") as fetch:
            ok = tts.ensure_cached("boot line")
        self.assertTrue(ok)
        self.assertTrue(tts._local_cache_wav("boot line").exists())
        fetch.assert_not_called()

    def test_ensure_cached_local_noop_when_disabled(self):
        with mock.patch.object(config, "LOCAL_TTS_MODE", True), \
             mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", False), \
             mock.patch.object(config, "NO_AUDIO_MODE", False, create=True), \
             mock.patch.object(config, "AUDIO_OUTPUT_SUPPRESSED", False, create=True), \
             mock.patch.object(local_tts, "is_available", return_value=True), \
             mock.patch.object(local_tts, "synthesize") as synth:
            ok = tts.ensure_cached("boot line")
        self.assertFalse(ok)
        synth.assert_not_called()
        self.assertFalse(tts._local_cache_wav("boot line").exists())


class SpeakLocalPlaybackTest(unittest.TestCase):
    """End-to-end _speak_local with a fake audio device + fake synthesis stream."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._stack = ExitStack()
        self._stack.enter_context(mock.patch.object(config, "TTS_CACHE_DIR", self._tmp.name))
        self._stack.enter_context(mock.patch("sounddevice.OutputStream", _FakeStream))
        self._stack.enter_context(mock.patch.object(local_tts, "sample_rate", return_value=24000))
        from audio import echo_cancel
        self._stack.enter_context(mock.patch.object(echo_cancel, "was_canceled", return_value=False))

    def tearDown(self):
        self._stack.close()
        self._tmp.cleanup()

    def test_streams_and_caches_rex_voice(self):
        with mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", True), \
             mock.patch.object(local_tts, "generate_stream", _fake_gen_factory()):
            handled = tts._speak_local("hi rex", REX_REF, "neutral", log_text=False)
        self.assertTrue(handled)
        # Rex voice is cacheable (cache enabled) → a WAV take was written.
        self.assertTrue(tts._local_cache_wav("hi rex").exists())

    def test_cache_disabled_by_default_resynthesizes(self):
        # Default (LOCAL_TTS_CACHE_ENABLED off): no WAV written, and a repeat line
        # re-synthesizes rather than replaying via _play.
        with mock.patch.object(local_tts, "generate_stream", _fake_gen_factory()):
            tts._speak_local("fresh line", REX_REF, "neutral", log_text=False)
        self.assertFalse(tts._local_cache_wav("fresh line").exists())
        with mock.patch.object(local_tts, "generate_stream", _fake_gen_factory()), \
             mock.patch.object(tts, "_play") as play:
            tts._speak_local("fresh line", REX_REF, "neutral", log_text=False)
            play.assert_not_called()

    def test_impersonation_take_not_cached(self):
        with mock.patch.object(local_tts, "generate_stream", _fake_gen_factory()):
            handled = tts._speak_local("i am carter", IMPERSONATION_REF, "neutral", log_text=False)
        self.assertTrue(handled)
        imp_wav = tts._cache_path(
            "i am carter", f"local:{IMPERSONATION_REF.label}", config.LOCAL_TTS_MODEL_ID
        ).with_suffix(".wav")
        self.assertFalse(imp_wav.exists())

    def test_second_call_is_cache_hit(self):
        with mock.patch.object(config, "LOCAL_TTS_CACHE_ENABLED", True), \
             mock.patch.object(local_tts, "generate_stream", _fake_gen_factory()):
            # First call streams + writes the WAV cache (no _play — streamed path).
            tts._speak_local("cache me", REX_REF, "neutral", log_text=False)
            # Second call finds the WAV and plays it through _play (the buffered path).
            with mock.patch.object(tts, "_play") as play:
                tts._speak_local("cache me", REX_REF, "neutral", log_text=False)
            play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
