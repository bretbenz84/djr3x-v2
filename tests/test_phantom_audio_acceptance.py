"""Regression for the 2026-09-27 Hiss / Spray / scarcity phantom turns."""
import unittest
from unittest import mock

import numpy as np
from intelligence import interaction as I


class PhantomAudioAcceptanceTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(I, "_game_suppresses_conversation", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def reject(self, text, **overrides):
        args = dict(trusted=False, text_input=False, under_playback=False,
                    speaker_id=1, speaker_score=0.519, speaker_margin=0.066,
                    required_margin=0.07)
        args.update(overrides)
        return I._audio_acceptance_rejection(text, **args)

    def test_short_guesses_are_not_acknowledgments(self):
        for text in ("Hiss.", "Spray.", "Scared.", "Purple elephant"):
            self.assertEqual(self.reject(text), "uncertain_short_fragment")
            self.assertIsNone(self.reject(text, trusted=True))
            self.assertIsNone(self.reject(text, text_input=True))

    def test_acknowledgments_and_stops_survive(self):
        for text in ("Okay.", "Yeah.", "No.", "Thank you.", "I'm sorry."):
            self.assertIsNone(self.reject(text))
        for text in ("Stop.", "Stop moving.", "Stop stop stop", "Freeze!"):
            self.assertIsNone(self.reject(text, under_playback=True))

    def test_game_answers_and_explicit_motion_keep_existing_handling(self):
        with mock.patch.object(I, "_game_suppresses_conversation", return_value=True):
            self.assertIsNone(self.reject("Paris."))
        with mock.patch.object(I, "_eager_motion_transcript_matches", return_value=True):
            self.assertIsNone(self.reject("Turn left."))

    def test_scarcity_confidence_does_not_prove_a_human_spoke(self):
        self.assertEqual(self.reject("You mean scarcity?", trusted=True, under_playback=True),
                         "uncorroborated_under_playback")
        self.assertIsNone(self.reject("You mean scarcity?", trusted=True))

    def test_playback_requires_trusted_unambiguous_known_voice(self):
        good = dict(trusted=True, under_playback=True, speaker_score=0.85, speaker_margin=0.2)
        self.assertIsNone(self.reject("An AWS outage?", **good))
        for override in (dict(speaker_id=None), dict(speaker_margin=0.02),
                         dict(speaker_score=0.5)):
            self.assertEqual(self.reject("An AWS outage?", **(good | override)),
                             "uncorroborated_under_playback")
        self.assertEqual(self.reject("An AWS outage?", **(good | dict(trusted=False))),
                         "untrusted_under_playback")

    def test_field_turns_drop_before_history_routing_or_reply_even_during_reprompt_cooldown(self):
        cases = [("Hiss.", False, False, 0.498, 0.227),
                 ("Spray.", False, False, 0.470, 0.221),
                 ("You mean scarcity?", True, True, 0.519, 0.066)]
        for text, trusted, overlap, score, margin in cases:
            with self.subTest(text=text), \
                 mock.patch.object(I, "_shutdown_requested", return_value=False), \
                 mock.patch.object(I.random, "randint", return_value=0), \
                 mock.patch.object(I, "_split_sequential_capture", side_effect=lambda audio, **kw: audio), \
                 mock.patch.object(I, "_split_audio_speakers", return_value=None), \
                 mock.patch.object(I, "_process_audio", return_value=(
                     I.transcription.Transcript(text, confident=trusted), 1, "Bret", score, margin, 0.07)), \
                 mock.patch.object(I, "_last_low_trust_reprompt_at", I.time.monotonic()), \
                 mock.patch.object(I, "_new_character_loop_trace") as trace, \
                 mock.patch.object(I, "_speak_blocking") as speak:
                I._handle_speech_segment(np.zeros(1600, dtype=np.float32), require_trusted=overlap)
                trace.assert_not_called()
                speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
