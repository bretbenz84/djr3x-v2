"""ASR hallucination guards and the streamed-take caching test.

All three from logs/djr3x-2026-08-20-20-06-34.log.
"""

import unittest
from unittest import mock

import numpy as np

import config
from audio import transcription, tts
from intelligence import action_router


class ContextPromptRegurgitationTests(unittest.TestCase):
    """20:16:38 — the decoder emitted its own biasing prompt back as a 54-word
    "utterance" at logprob 0.0. Only the speaking-rate backstop caught it, and that
    backstop does not fire on a longer capture (a 10s VAD segment is ordinary).
    Such a decode lands trusted=True carrying every name in QWEN_ASR_CONTEXT_VOCAB
    — the shape that poisons people.db and resurfaces as a proactive question.

    The echo guard's candidate set held the two KNOWN shapes (a Rex line copied
    back, the vocab list copied back). The fixed preamble was never in it.
    """

    PREAMBLE = ("This audio is one side of a live spoken conversation. "
                "Names and places that may occur: Bret, Lake Folsom, Sacramento.")

    def setUp(self):
        transcription._last_context_prompt = None
        self.addCleanup(setattr, transcription, "_last_context_prompt", None)

    def test_the_field_decode_is_rejected(self):
        self.assertTrue(transcription._context_echo_hallucination(self.PREAMBLE))

    def test_rejected_even_without_a_remembered_prompt(self):
        """The structural markers must stand alone — the prompt cache can be cold
        (first decode of a run) exactly when biasing is most likely to leak."""
        transcription._last_context_prompt = None
        self.assertTrue(transcription._context_echo_hallucination(
            "this audio is one side of a live spoken conversation"))

    def test_partial_regurgitation_is_rejected(self):
        self.assertTrue(transcription._context_echo_hallucination(
            "Names and places that may occur: Bret, Sacramento"))
        self.assertTrue(transcription._context_echo_hallucination(
            "The audio replies to a droid who just said hello there"))

    def test_the_exact_prompt_we_sent_is_a_candidate(self):
        transcription._last_context_prompt = "Some other biasing preamble entirely."
        self.assertTrue(transcription._context_echo_hallucination(
            "Some other biasing preamble entirely."))

    def test_real_speech_is_untouched(self):
        for utterance in (
            "Impersonate Barack Obama.",
            "Yeah, it should be fun to go to Jimmy Carter's house.",
            "We had a conversation about Sacramento last week.",
            "That's Max.",
        ):
            with self.subTest(utterance=utterance):
                self.assertFalse(
                    transcription._context_echo_hallucination(utterance))

    def test_prompt_builder_records_what_it_sent(self):
        with (
            mock.patch.object(config, "QWEN_ASR_CONTEXT_BIAS_ENABLED", True),
            mock.patch.object(config, "QWEN_ASR_CONTEXT_VOCAB", ("Bret", "Sacramento")),
        ):
            prompt = transcription._asr_context_prompt()
        self.assertTrue(prompt)
        self.assertEqual(transcription._last_context_prompt, prompt)


class ImpersonateInflectionTests(unittest.TestCase):
    """20:25:22 — "Impersonate Barack Obama" decoded as "Impersonates Barack
    Obama", matched no pattern, and the imperative was answered as small talk."""

    def _target(self, utterance):
        for pat in action_router._IMPERSONATE_PATTERNS:
            m = pat.search(utterance)
            if m:
                return m.groupdict().get("target")
        return None

    def test_asr_inflections_route(self):
        self.assertEqual(self._target("Impersonates Barack Obama."), "Barack Obama.")
        self.assertEqual(self._target("Impersonating me."), "me.")
        self.assertEqual(self._target("Rex, impersonates JT"), "JT")
        self.assertEqual(self._target("Imitates JT"), "JT")

    def test_base_forms_still_route(self):
        self.assertEqual(self._target("Impersonate Barack Obama."), "Barack Obama.")
        self.assertEqual(self._target("Do an impersonation of Jimmy Carter"),
                         "Jimmy Carter")

    def test_a_gerund_mid_sentence_is_description_not_a_command(self):
        """This module keeps to unambiguous verb shapes on purpose; softer
        phrasings stay on the LLM route. An inflected verb is a command only when
        it IS the utterance."""
        self.assertIsNone(self._target("I was imitating a bird earlier"))
        self.assertIsNone(self._target("He kept mimicking my accent all night"))
        self.assertIsNone(self._target("She was impersonating a customer for the demo"))

    def test_existing_false_positive_guard_still_holds(self):
        self.assertIsNone(self._target("Do you like my voice?"))


class EndsHotRelativeTests(unittest.TestCase):
    """The absolute 0.010 cutoff refused 28% of streamed takes (14 of 50),
    including ordinary complete sentences ("Backing up.", 1.56s). No absolute
    threshold separates the populations: accepted takes ran a smooth continuum
    from 0.00005 to 0.00935 against that line, with no gap. Real truncations end
    at FULL VOICED LEVEL, so the ratio to the take's own median separates them.

    Cost of a false positive is cache churn, not audio: a refused line can never
    be reused, so every repeat pays a fresh generation — and has nothing to fall
    back on when the network dies, as it did at 20:32.
    """

    SR = 22050

    def _speech(self, secs=2.0, level=0.08):
        n = int(self.SR * secs)
        t = np.arange(n) / self.SR
        # Syllable-rate envelope so the median voiced level is meaningful.
        env = 0.35 + 0.65 * np.abs(np.sin(2 * np.pi * 3.0 * t))
        return (np.sin(2 * np.pi * 180.0 * t) * level * env).astype(np.float32)

    def test_a_decayed_tail_is_not_hot(self):
        a = self._speech()
        fade = int(self.SR * 0.15)
        a[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        self.assertFalse(tts._ends_hot(a, self.SR))

    def test_a_mid_word_cut_is_hot(self):
        a = self._speech()
        # Cut where the envelope is at a peak — a truncated generation.
        peak = int(self.SR * (1.0 / 3.0 / 2.0))
        self.assertTrue(tts._ends_hot(a[:peak * 3], self.SR))

    def test_a_quiet_line_is_not_punished_for_being_quiet(self):
        """The absolute rail must not flag a decayed tail just because the whole
        line was loud, nor pass a hot tail because the line was quiet."""
        loud = self._speech(level=0.30)
        fade = int(self.SR * 0.15)
        loud[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        self.assertFalse(tts._ends_hot(loud, self.SR),
                         "a complete LOUD line was refused caching")

    def test_a_very_quiet_tail_is_never_a_truncation(self):
        quiet = self._speech(level=0.004)
        self.assertFalse(tts._ends_hot(quiet, self.SR),
                         "absolute rail should stop a whisper reading as truncated")

    def test_ratio_of_zero_disables(self):
        a = self._speech()
        with mock.patch.object(config, "TTS_HOT_END_RMS_RATIO", 0.0):
            self.assertFalse(tts._ends_hot(a[:int(self.SR * 1.0)], self.SR))

    def test_unmeasurable_input_is_not_hot(self):
        self.assertFalse(tts._ends_hot(None, self.SR))
        self.assertFalse(tts._ends_hot(np.zeros(0, dtype=np.float32), self.SR))
        self.assertFalse(tts._ends_hot(np.zeros(100, dtype=np.float32), self.SR))
        self.assertIsNone(tts._tail_rms_ratio(np.zeros(10, dtype=np.float32), self.SR))

    def test_ratio_is_reported_for_the_log(self):
        a = self._speech()
        tail, ratio = tts._tail_rms_ratio(a, self.SR)
        self.assertGreater(tail, 0.0)
        self.assertGreater(ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
