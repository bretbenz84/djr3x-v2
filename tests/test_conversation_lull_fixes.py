"""
Conversation-quality fixes from the 2026-07-06-22-17 field log:

1. "nope" answering Rex's own question got hijacked by the bare-negation REPAIR
   ("Fair enough — let me reset and try that again." — a non-sequitur). When the
   dialogue act binds the turn as answer_to_rex, a bare negation is CONTENT for
   the reply path, never a correction.
2. Rex went silent for 42s after a user turn: the impulse flow-guard demanded 30s
   of mutual silence after ANY user content. Now 14s — a human beat, while the
   sub-10s question-machine failure the guard exists for stays blocked.
3. The lull impulse was FORBIDDEN from following up a flat half-answer ("It's
   okay") by the new-thread-only rule, so it reached for generic interview
   questions instead. The instruction now carves out loose-end follow-ups.
"""

import unittest
from types import SimpleNamespace

import config
from intelligence import interaction as I
from intelligence import lean_brain as LB
from intelligence import repair_moves as r


class NegationIsAnswerTest(unittest.TestCase):
    def _dd(self, label):
        return SimpleNamespace(label=label)

    def test_nope_answering_rex_question_is_not_a_repair(self):
        move = {"kind": "bare_negation", "severity": "low"}
        self.assertTrue(I._negation_is_answer(move, self._dd("answer_to_rex")))

    def test_unprompted_negation_still_repairs(self):
        move = {"kind": "bare_negation", "severity": "low"}
        self.assertFalse(I._negation_is_answer(move, self._dd("new_topic")))
        self.assertFalse(I._negation_is_answer(move, None))

    def test_other_repair_kinds_unaffected(self):
        move = {"kind": "misheard", "correction": "I said tacos"}
        self.assertFalse(I._negation_is_answer(move, self._dd("answer_to_rex")))

    def test_no_repair_move_is_false(self):
        self.assertFalse(I._negation_is_answer(None, self._dd("answer_to_rex")))

    def test_bare_negation_detection_itself_still_works(self):
        # The repair layer still catches a genuine unprompted correction shape.
        r.note_assistant_turn("So you're heading to the lake tomorrow, right?")
        detected = r.detect("nope")
        self.assertIsNotNone(detected)
        self.assertEqual(detected.get("kind"), "bare_negation")


class FlowQuietTimingTest(unittest.TestCase):
    def test_flow_quiet_is_a_human_beat_not_half_a_minute(self):
        # 42s of dead air after "It's okay" was the field complaint; the guard's
        # job is only to block sub-10s question-machine stacking.
        self.assertLessEqual(float(config.LEAN_IMPULSE_FLOW_QUIET_SECS), 15.0)
        self.assertGreaterEqual(float(config.LEAN_IMPULSE_FLOW_QUIET_SECS), 10.0)


class ImpulseLooseEndTest(unittest.TestCase):
    def test_instruction_allows_following_up_flat_answers(self):
        self.assertIn("loose end", LB._IMPULSE_INSTRUCTION)
        self.assertIn("following up, NOT reheating", LB._IMPULSE_INSTRUCTION)

    def test_new_thread_rules_survive(self):
        # The carve-out must not have deleted the anti-repeat machinery.
        self.assertIn("RESIST", LB._IMPULSE_INSTRUCTION)
        self.assertIn("ALREADY COVERED", LB._IMPULSE_INSTRUCTION)


class SceneWindowGuardTest(unittest.TestCase):
    """False take-a-bow (owner: 'I didn't laugh', log 2026-07-06-22-28): the scene
    analyzer samples a rolling window that reaches BACK in time, so the skip must
    hold until no part of that window overlaps Rex's own playback tail."""

    def _skip(self, since_release):
        from unittest import mock
        from audio import scene
        with (
            mock.patch.object(scene.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(scene.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(scene.output_gate, "seconds_since_release",
                              return_value=since_release),
        ):
            return scene._should_skip_cycle()

    def test_window_straddling_speech_tail_is_skipped(self):
        # The field case: gate released ~2.1s ago, window 2.0s — the old `< WINDOW`
        # check let this through and the window's leading edge held Rex's bleed.
        self.assertTrue(self._skip(2.1))

    def test_fully_clear_window_analyzes(self):
        window = float(config.SCENE_ANALYSIS_WINDOW_SECS)
        guard = float(config.SCENE_POST_OUTPUT_GUARD_SECS)
        self.assertFalse(self._skip(window + guard + 0.1))


class WhisperOutroHallucinationTest(unittest.TestCase):
    """YouTube-outro hallucination family (live 2026-07-06-22-39: 'and more. I hope
    you enjoyed this video. I'll see you in the next video.' — spoken by nobody).
    The filter must catch outro boilerplate as a SUBSTRING while leaving genuine
    speech about enjoying things alone."""

    def test_field_string_is_filtered(self):
        from audio.transcription import _is_hallucination
        self.assertTrue(_is_hallucination(
            "and more. I hope you enjoyed this video. I'll see you in the next video."
        ))

    def test_outro_variants_filtered(self):
        from audio.transcription import _is_hallucination
        for t in ("I hope you enjoyed the video",
                  "see you in the next one",
                  "I'll see you guys in the next episode"):
            self.assertTrue(_is_hallucination(t), t)

    def test_genuine_speech_survives(self):
        from audio.transcription import _is_hallucination
        for t in ("I really enjoyed this movie last night",
                  "see you tomorrow",
                  "I hope you enjoyed the party"):
            self.assertFalse(_is_hallucination(t), t)

    def test_local_silence_skips_api_second_opinion(self):
        # Local Whisper decoded empty -> no API call, empty return.
        from unittest import mock
        import numpy as np
        from audio import transcription as tr
        with (
            mock.patch.object(tr, "_MLX_AVAILABLE", True),
            mock.patch.object(tr, "_local_model_ready", return_value=True),
            mock.patch.object(tr, "mlx_whisper", create=True) as mlx,
            mock.patch.object(config, "WHISPER_FALLBACK_ON_EMPTY", False),
        ):
            mlx.transcribe.return_value = {"text": "  "}
            with mock.patch("openai.OpenAI") as api:
                out = tr.transcribe(np.zeros(16000, dtype=np.float32))
        self.assertEqual(out, "")
        api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
