"""
Field fixes from the 2026-08-05 21:18 session (five distinct issues, one run).

1. WAVE at a motionless arm on a chair — four wave-backs fired, every one logged
   speed=n/a: flickering clutter keypoints wipe the wrist motion history, so the
   phantom is exactly the unmeasured case the old None-passes back-compat let
   through. Unmeasured now fails; and wrist speed is measured RELATIVE to the
   shoulder, so Rex's own neck pans (camera egomotion) can't read as wrist motion.
2. "An AWS outage?" — Bret repeating Rex's words back as a follow-up question was
   eaten by the own-echo rejector despite a 0.868 voiceprint match. A confident
   human voice match now overrides the text match.
3. A triumphant "proud" chirp minutes after Rex said he was feeling sluggish —
   celebratory chirps now defer to the day mood.
4. "Why thank you!" transcribed as "Why? Thank you." — the LLM answered the "Why?".
   ASR idiom correction.
5. "I don't know. Hey, I'm gonna go now. Can you shut down, please?" →
   "I couldn't safely parse that whole route." The sequence parser's negation
   guard fired on the "don't" in "I don't know" before any clause was classified;
   the shutdown request was eaten by a rejection for a route nobody asked for.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config


class WaveSpeedGateTests(unittest.TestCase):

    def test_unmeasured_speed_is_not_a_wave(self):
        # The four field firings all logged speed=n/a. The veto must treat "no
        # stable motion history" as furniture, not grant it a pass.
        from intelligence import consciousness as c
        self.assertFalse(bool(getattr(config, "WAVE_BACK_UNMEASURED_IS_WAVE", False)))
        import inspect
        src = inspect.getsource(c._step_wave_reaction)
        self.assertIn("WAVE_BACK_UNMEASURED_IS_WAVE", src)
        self.assertIn("wrist speed unmeasured", src)

    def test_wrist_position_is_shoulder_relative(self):
        # Camera egomotion (Rex's neck pans) sweeps every landmark laterally in
        # frame coordinates. Wrist-minus-shoulder cancels the pan while a real
        # wave keeps its amplitude.
        from vision import pose
        kp = {
            "LEFT_SHOULDER": [0.40, 0.50, 0.99],
            "LEFT_WRIST": [0.55, 0.30, 0.99],   # raised, 0.15 right of shoulder
        }
        self.assertAlmostEqual(pose._raised_wrist_x(kp), 0.15, places=6)
        # A pure pan: everything shifts +0.2 in frame — relative x is unchanged,
        # so the recorded "motion" between these two frames is zero.
        panned = {
            "LEFT_SHOULDER": [0.60, 0.50, 0.99],
            "LEFT_WRIST": [0.75, 0.30, 0.99],
        }
        self.assertAlmostEqual(pose._raised_wrist_x(kp),
                               pose._raised_wrist_x(panned), places=6)

    def test_a_real_wave_still_measures(self):
        # Hand sweeping over a still shoulder: relative x changes by the sweep.
        from vision import pose
        left = {"LEFT_SHOULDER": [0.40, 0.50, 0.99], "LEFT_WRIST": [0.30, 0.30, 0.99]}
        right = {"LEFT_SHOULDER": [0.40, 0.50, 0.99], "LEFT_WRIST": [0.55, 0.30, 0.99]}
        delta = abs(pose._raised_wrist_x(right) - pose._raised_wrist_x(left))
        self.assertAlmostEqual(delta, 0.25, places=6)

    def test_lowered_wrist_records_nothing(self):
        from vision import pose
        kp = {
            "LEFT_SHOULDER": [0.40, 0.50, 0.99],
            "LEFT_WRIST": [0.42, 0.80, 0.99],   # below the shoulder
        }
        self.assertIsNone(pose._raised_wrist_x(kp))


class OwnEchoVoiceOverrideTests(unittest.TestCase):

    def test_confident_human_voice_keeps_the_transcript(self):
        # The rejection condition at the call site must be disarmed by a strong
        # voiceprint match. Asserted structurally (the full segment handler needs
        # a heavyweight harness): the override score gates the echo check.
        import inspect
        from intelligence import interaction as I
        src = inspect.getsource(I._handle_speech_segment)
        idx = src.index("_looks_like_own_echo(text)")
        window = src[max(0, idx - 900): idx]
        self.assertIn("OWN_ECHO_VOICE_OVERRIDE_SCORE", window)
        self.assertIn("speaker_score", window)

    def test_override_threshold_is_stricter_than_ordinary_matches(self):
        # The field session's ordinary turns scored 0.56-0.73; the echoed
        # follow-up scored 0.868. The bar must sit between those bands so real
        # residual (which matches no human strongly) still gets rejected.
        score = float(config.OWN_ECHO_VOICE_OVERRIDE_SCORE)
        self.assertGreaterEqual(score, 0.75)
        self.assertLessEqual(score, 0.9)

    def test_matcher_itself_still_rejects_verbatim_prefix(self):
        # The override lives at the CALL SITE — the text matcher keeps catching
        # true echo (a clean prefix of a recent Rex line, no voice information).
        from intelligence import interaction as I
        with mock.patch.object(I, "_recent_rex_lines",
                               [(I._normalize_echo_text(
                                   "An AWS outage has been living rent-free in my "
                                   "processors all day."), I.time.monotonic())]):
            self.assertTrue(I._looks_like_own_echo("An AWS outage."))


class CelebratoryChirpGateTests(unittest.TestCase):

    def setUp(self) -> None:
        from intelligence import rex_mood
        self.rex_mood = rex_mood
        rex_mood.clear()
        self.addCleanup(rex_mood.clear)
        self._patches = [
            mock.patch.object(config, "REX_MOOD_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_GATES_CELEBRATORY_CHIRPS", True),
            mock.patch.object(rex_mood, "_SIGNALS", ()),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _mint(self, valence: float, energy: float) -> None:
        seed = {"id": "t", "label": "test", "valence": valence, "energy": energy,
                "line": "A test mood.", "fits": ["any"]}
        with mock.patch.object(config, "REX_MOOD_SEEDS", [seed]):
            self.rex_mood.clear()
            self.rex_mood.ensure_today()

    def test_proud_chirp_suppressed_on_a_sluggish_day(self):
        # The field case: day mood "sluggish" (v -0.15, e 0.15) → no fanfare.
        from intelligence import body_mood
        self._mint(-0.15, 0.15)
        with mock.patch("audio.sound_effects.play") as play:
            body_mood._mood_chirp("proud")
        play.assert_not_called()

    def test_proud_chirp_plays_on_a_good_day(self):
        from intelligence import body_mood
        self._mint(0.6, 0.7)
        with mock.patch("audio.sound_effects.play") as play:
            body_mood._mood_chirp("proud")
        play.assert_called_once_with("proud")

    def test_no_day_mood_means_no_gate(self):
        # Unminted mood (feature off, early boot) must not silence the chirp.
        from intelligence import body_mood
        self.rex_mood.clear()
        with mock.patch("audio.sound_effects.play") as play:
            body_mood._mood_chirp("proud")
        play.assert_called_once_with("proud")

    def test_gate_flag_off_restores_old_behavior(self):
        from intelligence import body_mood
        self._mint(-0.15, 0.15)
        with (mock.patch.object(config, "REX_MOOD_GATES_CELEBRATORY_CHIRPS", False),
              mock.patch("audio.sound_effects.play") as play):
            body_mood._mood_chirp("proud")
        play.assert_called_once_with("proud")


class WhyThankYouTests(unittest.TestCase):

    def test_the_idiom_is_repunctuated(self):
        from audio import transcription as T
        self.assertEqual(T._apply_corrections("Why? Thank you."), "Why, thank you.")
        self.assertEqual(T._apply_corrections("why? thanks!"), "Why, thanks!")

    def test_a_real_question_plus_thanks_is_untouched(self):
        from audio import transcription as T
        self.assertEqual(
            T._apply_corrections("Why did you do that? Thank goodness."),
            "Why did you do that? Thank goodness.",
        )
        self.assertEqual(T._apply_corrections("Why?"), "Why?")


class MotionSequenceNegationTests(unittest.TestCase):
    """The tri-state: [] = not a sequence (conversation), None = motion-shaped but
    refused (spoken rejection), list = execute."""

    def _seq(self, text: str):
        from intelligence import action_router
        return action_router.classify_explicit_motion_sequence(text, max_steps=8)

    def test_the_field_utterance_is_conversation_not_a_rejected_route(self):
        self.assertEqual(
            self._seq("I don't know. Hey, I'm gonna go now. Can you shut down, please?"),
            [],
        )

    def test_dont_know_with_commas_is_conversation(self):
        for text in (
            "I don't know, maybe later",
            "I can't tell, honestly",
            "no, I don't think so, thanks",
        ):
            with self.subTest(text=text):
                self.assertEqual(self._seq(text), [])

    def test_negated_actual_motion_is_still_refused_whole(self):
        # The guard's original purpose survives: nothing may execute.
        self.assertIsNone(self._seq("don't turn left, then move forward"))

    def test_explanation_over_motion_clauses_is_still_refused(self):
        self.assertIsNone(self._seq("why didn't you move forward, then turn left?"))

    def test_a_real_sequence_still_executes(self):
        result = self._seq("turn left, then move forward")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_mixed_motion_and_nonsense_is_still_refused(self):
        self.assertIsNone(self._seq("turn left then sing"))


if __name__ == "__main__":
    unittest.main()
