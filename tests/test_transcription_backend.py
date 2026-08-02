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


class StandaloneCorrectionTest(unittest.TestCase):
    """Whole-utterance homophone guard (field 2026-08-02: qwen3 heard the bare
    command "roast me" as "Roast meat." — not common English, so the decoder
    snapped to the nearby real phrase). Fires ONLY on the standalone phrase."""

    def test_bare_roast_meat_becomes_roast_me(self):
        self.assertEqual(tr._apply_standalone_corrections("Roast meat."), "Roast me.")
        self.assertEqual(tr._apply_standalone_corrections("roast meet"), "roast me")
        self.assertEqual(
            tr._apply_standalone_corrections("Rex, roast meat."), "Rex, roast me."
        )
        self.assertEqual(
            tr._apply_standalone_corrections("roast meat please"), "roast me please"
        )

    def test_real_sentences_about_roasting_meat_untouched(self):
        for text in (
            "I like to roast meat on Sundays.",
            "We should roast meat tonight",
            "how do you roast meat",
            "roast meatballs",
        ):
            with self.subTest(text=text):
                self.assertEqual(tr._apply_standalone_corrections(text), text)


class ContextBiasTest(unittest.TestCase):
    """Qwen3-ASR context biasing (field 2026-08-02: Rex said 'Lake Folsom
    today...' and the reply decoded as 'like falsum' — the trip cancellation
    never reached memory). Rex's recent lines + static vocab ride the system
    prompt; the echo guard rejects the decoder copying that context back out
    on silence (measured: verbatim copy at logprob 0.0)."""

    def setUp(self):
        with tr._context_lock:
            self._saved = list(tr._recent_rex_lines)
            tr._recent_rex_lines.clear()

    def tearDown(self):
        with tr._context_lock:
            tr._recent_rex_lines.clear()
            tr._recent_rex_lines.extend(self._saved)

    def test_context_prompt_includes_vocab_and_rex_lines(self):
        tr.note_rex_line("Hey Bret, Lake Folsom today.")
        prompt = tr._asr_context_prompt()
        self.assertIn("Lake Folsom", prompt)
        self.assertIn("just said", prompt)

    def test_kill_switch(self):
        with mock.patch.object(
            tr.config, "QWEN_ASR_CONTEXT_BIAS_ENABLED", False, create=True
        ):
            self.assertIsNone(tr._asr_context_prompt())

    def test_echo_guard_rejects_context_copies_only(self):
        tr.note_rex_line(
            "Hey Bret, Lake Folsom today—let's hope your friends bring snacks."
        )
        # Verbatim copies of the context are hallucinations...
        self.assertTrue(tr._context_echo_hallucination(
            "Hey Bret, Lake Folsom today, let's hope your friends bring snacks"
        ))
        # ...but a real reply that merely re-uses an entity passes through.
        self.assertFalse(tr._context_echo_hallucination(
            "We're not going to Lake Folsom anymore."
        ))
        self.assertFalse(tr._context_echo_hallucination("Yes."))

    def test_falsum_corrections(self):
        self.assertEqual(
            tr._apply_corrections("We're not going to like falsum anymore."),
            "We're not going to Lake Folsom anymore.",
        )
        self.assertEqual(
            tr._apply_corrections("I like Folsom a lot"),
            "I like Folsom a lot",
        )


if __name__ == "__main__":
    unittest.main()
