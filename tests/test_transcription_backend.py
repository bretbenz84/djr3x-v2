"""
Backend selection for audio/transcription.py (Qwen3-ASR primary since 2026-07-31).

Pins: qwen3 is used when selected and ready; a qwen failure degrades to the
local whisper path (not straight to the API); an empty qwen decode counts as
"local decoded silence" so the API fallback is still skipped; and the
.confident gate applies the backend-specific logprob floor (Qwen3's scale is
far more peaked than Whisper's — clean decodes ~0.0, garbage below -0.7).
"""

import unittest
from unittest import mock

import numpy as np

import config
from audio import transcription as tr

AUDIO = np.zeros(16000, dtype=np.float32)


class QwenBackendTest(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(config, "TRANSCRIPTION_BACKEND", "qwen3", create=True),
            mock.patch.object(tr, "_qwen_ready", return_value=True),
            mock.patch.object(tr, "_QWEN_LOAD_FAILED", False),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def test_qwen_result_is_used_and_whisper_not_called(self):
        with mock.patch.object(tr, "_qwen_transcribe",
                               return_value=("Come here.", -0.01)) as q, \
             mock.patch.object(tr, "mlx_whisper") as mw:
            out = tr.transcribe(AUDIO)
        q.assert_called_once()
        mw.transcribe.assert_not_called()
        self.assertEqual(str(out), "Come here.")
        self.assertEqual(out.backend, "qwen3_asr")
        self.assertTrue(out.confident)

    def test_low_logprob_decode_is_not_confident(self):
        # -0.75 was the truncated "Okay." capture in the calibration set — heard
        # and replied to, but never learned from.
        with mock.patch.object(tr, "_qwen_transcribe",
                               return_value=("Okay, let's do that thing.", -0.75)):
            out = tr.transcribe(AUDIO)
        self.assertEqual(out.backend, "qwen3_asr")
        self.assertFalse(out.confident)

    def test_whisper_floor_does_not_apply_to_qwen(self):
        # -0.5 fails Qwen's floor (-0.35) but would PASS Whisper's (-0.85):
        # the scales are different and must not be conflated.
        with mock.patch.object(tr, "_qwen_transcribe",
                               return_value=("Some plausible sentence here.", -0.5)):
            out = tr.transcribe(AUDIO)
        self.assertFalse(out.confident)

    def test_qwen_failure_falls_back_to_local_whisper(self):
        with mock.patch.object(tr, "_qwen_transcribe", side_effect=RuntimeError("boom")), \
             mock.patch.object(tr, "_local_model_ready", return_value=True), \
             mock.patch.object(tr, "mlx_whisper") as mw:
            mw.transcribe.return_value = {"text": "Hello there, my friend.", "segments": []}
            out = tr.transcribe(AUDIO)
        self.assertEqual(str(out), "Hello there, my friend.")
        self.assertEqual(out.backend, "mlx_whisper")

    def test_empty_qwen_decode_skips_api_fallback(self):
        # Silence decoded locally IS an answer; the API second-opinion path is
        # how the YouTube-outro hallucination reached the reply path once.
        with mock.patch.object(tr, "_qwen_transcribe", return_value=("", None)), \
             mock.patch("openai.OpenAI") as api:
            out = tr.transcribe(AUDIO)
        api.assert_not_called()
        self.assertEqual(str(out), "")
        self.assertEqual(out.backend, "qwen3_asr")

    def test_missing_qwen_model_uses_whisper(self):
        with mock.patch.object(tr, "_qwen_ready", return_value=False), \
             mock.patch.object(tr, "_qwen_transcribe") as q, \
             mock.patch.object(tr, "_local_model_ready", return_value=True), \
             mock.patch.object(tr, "mlx_whisper") as mw:
            mw.transcribe.return_value = {"text": "Fallback words spoken.", "segments": []}
            out = tr.transcribe(AUDIO)
        q.assert_not_called()
        self.assertEqual(out.backend, "mlx_whisper")


class WhisperBackendUnchangedTest(unittest.TestCase):
    def test_whisper_selected_never_touches_qwen(self):
        with mock.patch.object(config, "TRANSCRIPTION_BACKEND", "whisper", create=True), \
             mock.patch.object(tr, "_qwen_transcribe") as q, \
             mock.patch.object(tr, "_local_model_ready", return_value=True), \
             mock.patch.object(tr, "mlx_whisper") as mw:
            mw.transcribe.return_value = {"text": "Regular whisper result here.",
                                          "segments": [{"avg_logprob": -0.3,
                                                        "no_speech_prob": 0.1}]}
            out = tr.transcribe(AUDIO)
        q.assert_not_called()
        self.assertEqual(out.backend, "mlx_whisper")
        self.assertTrue(out.confident)

    def test_whisper_confidence_floor_unchanged(self):
        self.assertTrue(tr._is_confident(-0.5, 0.1, "mlx_whisper"))
        self.assertFalse(tr._is_confident(-0.9, 0.1, "mlx_whisper"))
        self.assertFalse(tr._is_confident(-0.5, 0.1, "qwen3_asr"))
        self.assertTrue(tr._is_confident(-0.02, None, "qwen3_asr"))


if __name__ == "__main__":
    unittest.main()
