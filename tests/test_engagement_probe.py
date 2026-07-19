"""
tests/test_engagement_probe.py — the disengagement protocol (owner 2026-07-18:
"treat a lack of response as a gauge of possible lack of interest").

Covers: the deferral regex, the speech-clears-state hook, the probe speak path
(mocked TTS), and the no-answer → long-snooze resolution inside the impulse gate
machinery. No audio, no LLM.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from intelligence import interaction as I


def _reset_state():
    I._engagement_probe_at = 0.0
    I._engagement_probed_this_silence = False
    I._impulse_snooze_until = 0.0
    I._impulse_snooze_reason = ""
    I._impulse_snooze_person = None
    I._consecutive_lean_impulses = 0
    I._lean_impulse_spoken_times[:] = []


class DeferralRegexTest(unittest.TestCase):
    def test_deferral_shapes_match(self):
        for phrase in (
            "give me a few minutes",
            "Give me two minutes, Rex",
            "gimme a sec",
            "hold on",
            "hang on a moment",
            "just a second",
            "not right now",
            "cant talk right now",       # apostrophe-less (Whisper)
            "im busy",
            "I'm kinda busy here",
            "later, Rex",
        ):
            self.assertIsNotNone(
                I._ENGAGEMENT_DEFERRAL_RE.search(phrase), phrase
            )

    def test_normal_chat_does_not_match(self):
        for phrase in (
            "yep",
            "what's the news today",
            "I went on a river float",
            "that's my bed",
            "the concert is in a big stadium",
        ):
            self.assertIsNone(I._ENGAGEMENT_DEFERRAL_RE.search(phrase), phrase)


class DeferralCaptureTest(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def tearDown(self):
        _reset_state()

    def test_deferral_arms_snooze(self):
        with mock.patch.object(I, "_primary_session_person_id", return_value=1):
            self.assertTrue(I._maybe_capture_engagement_deferral("give me a few minutes"))
        self.assertGreater(I._impulse_snooze_until, time.monotonic())
        self.assertEqual(I._impulse_snooze_reason, "deferred")
        self.assertEqual(I._impulse_snooze_person, 1)

    def test_plain_reply_does_not_snooze(self):
        self.assertFalse(I._maybe_capture_engagement_deferral("pretty good, you?"))
        self.assertEqual(I._impulse_snooze_until, 0.0)

    def test_user_speech_clears_probe_and_snooze(self):
        I._engagement_probe_at = time.monotonic()
        I._engagement_probed_this_silence = True
        I._impulse_snooze_until = time.monotonic() + 500
        I._impulse_snooze_reason = "no_answer"
        I._note_user_speech_for_engagement()
        self.assertEqual(I._engagement_probe_at, 0.0)
        self.assertFalse(I._engagement_probed_this_silence)
        self.assertEqual(I._impulse_snooze_until, 0.0)

    def test_person_left_clears_everything(self):
        I._engagement_probe_at = time.monotonic()
        I._impulse_snooze_until = time.monotonic() + 500
        I._clear_engagement_state("person left")
        self.assertEqual(I._engagement_probe_at, 0.0)
        self.assertEqual(I._impulse_snooze_until, 0.0)
        self.assertIsNone(I._impulse_snooze_person)


class ProbeSpeakTest(unittest.TestCase):
    def setUp(self):
        _reset_state()

    def tearDown(self):
        _reset_state()

    def test_probe_speaks_and_arms_window(self):
        with mock.patch.object(I, "_speak_proactive", return_value=True) as spk, \
             mock.patch.object(I, "_register_rex_utterance"), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch("memory.people.get_person", return_value={"name": "Bret Benziger"}), \
             mock.patch.object(I.profile_questions, "profile_fact_count", return_value=40):
            self.assertTrue(I._speak_engagement_probe(1))
        line = spk.call_args[0][0]
        self.assertTrue(line)                       # a known-person (non-shy) probe line
        self.assertNotIn("{name}", line)            # placeholder resolved
        self.assertGreater(I._engagement_probe_at, 0.0)
        self.assertTrue(I._engagement_probed_this_silence)
        self.assertEqual(I._consecutive_lean_impulses, 1)

    def test_shy_variant_for_sparse_profile_uses_name(self):
        with mock.patch.object(I, "_speak_proactive", return_value=True) as spk, \
             mock.patch.object(I, "_register_rex_utterance"), \
             mock.patch.object(I.conv_memory, "add_to_transcript"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch("memory.people.get_person", return_value={"name": "Jamie Doe"}), \
             mock.patch.object(I.profile_questions, "profile_fact_count", return_value=1):
            self.assertTrue(I._speak_engagement_probe(2))
        self.assertIn("Jamie", spk.call_args[0][0])  # "I don't bite, Jamie" register

    def test_failed_speech_does_not_arm_probe(self):
        with mock.patch.object(I, "_speak_proactive", return_value=False), \
             mock.patch("memory.people.get_person", return_value={"name": "Bret"}), \
             mock.patch.object(I.profile_questions, "profile_fact_count", return_value=40):
            self.assertFalse(I._speak_engagement_probe(1))
        self.assertEqual(I._engagement_probe_at, 0.0)
        self.assertFalse(I._engagement_probed_this_silence)


if __name__ == "__main__":
    unittest.main()
