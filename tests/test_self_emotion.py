"""
Tests for the self-emotion classifier: reading the tone of REX'S OWN reply
(excited/happy/curious/neutral) so his body can express it. The classifier runs on
the local qwen sidecar with a keyword fallback; these tests exercise the pure
heuristic and the disable/guard paths (no sidecar required).
"""

from __future__ import annotations

import unittest
from unittest import mock


class SelfEmotionHeuristicTest(unittest.TestCase):
    def _h(self, text):
        from intelligence import llm
        return llm._self_emotion_heuristic(text)

    def test_excited_on_stacked_exclaim(self):
        self.assertEqual(self._h("Let's go!! That was incredible!"), "excited")

    def test_excited_on_marker_phrase(self):
        self.assertEqual(self._h("No way you actually did that."), "excited")

    def test_curious_on_genuine_question(self):
        self.assertEqual(self._h("How did you pull that off?"), "curious")

    def test_happy_on_fond_marker(self):
        self.assertEqual(self._h("Ha, nice one."), "happy")

    def test_happy_on_single_exclaim(self):
        self.assertEqual(self._h("Good to see you again!"), "happy")

    def test_neutral_on_flat_statement(self):
        self.assertEqual(self._h("It is about three in the afternoon."), "neutral")


class ClassifySelfEmotionTest(unittest.TestCase):
    def test_disabled_flag_returns_neutral(self):
        from intelligence import llm
        import config
        with mock.patch.object(config, "SELF_EMOTION_CLASSIFY_ENABLED", False):
            self.assertEqual(llm.classify_self_emotion("Let's gooo!!"), "neutral")

    def test_empty_returns_neutral(self):
        from intelligence import llm
        self.assertEqual(llm.classify_self_emotion("   "), "neutral")

    def test_falls_back_to_heuristic_when_sidecar_unavailable(self):
        from intelligence import llm, local_llm
        with mock.patch.object(local_llm, "enabled", return_value=False):
            self.assertEqual(llm.classify_self_emotion("How did you do that?"), "curious")

    def test_only_returns_valid_emotions(self):
        from intelligence import llm, local_llm
        with mock.patch.object(local_llm, "enabled", return_value=False):
            for text in ["Let's go!!", "How so?", "Ha, classic.", "Noted.", ""]:
                self.assertIn(llm.classify_self_emotion(text), llm._SELF_EMOTIONS)


class SelfEmotionBodyMoodMappingTest(unittest.TestCase):
    """Every emotion the classifier can return must map to a real body mood, or the
    afterglow silently no-ops."""

    def test_each_self_emotion_maps_to_a_body_mood(self):
        from intelligence import llm, body_mood
        for emotion in llm._SELF_EMOTIONS:
            if emotion == "neutral":
                continue
            self.assertIsNotNone(
                body_mood.canonical_mood(emotion),
                f"self-emotion {emotion!r} has no body_mood mapping",
            )

    def test_excited_maps_to_giddy(self):
        from intelligence import body_mood
        self.assertEqual(body_mood.canonical_mood("excited"), "giddy")


if __name__ == "__main__":
    unittest.main()
