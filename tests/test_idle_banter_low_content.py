"""
Low-content / quiet-turn gate for idle banter: a curt, content-free answer ("not much",
"Hello") is a legitimate reply Rex should let breathe, NOT keep mining as a topic.
Regression for the 2026-06-26 live log where "not much" got editorialized twice ~18s apart.
"""

import unittest
from unittest import mock

from intelligence import interaction


class IdleBanterLowContentTest(unittest.TestCase):
    def _patch_transcript(self, entries):
        # entries: list of (speaker, text)
        return mock.patch.object(
            interaction.conv_memory,
            "get_session_transcript",
            return_value=[{"speaker": s, "text": t} for s, t in entries],
        )

    def test_last_user_turn_text_skips_rex(self):
        with self._patch_transcript([("rex", "Hey Bret"), ("Bret", "not much"), ("rex", "ok")]):
            self.assertEqual(interaction._last_user_turn_text(), "not much")

    def test_low_content_true_for_curt_answer(self):
        with self._patch_transcript([("rex", "what's up"), ("Bret", "not much")]):
            self.assertTrue(interaction._last_user_turn_was_low_content())

    def test_low_content_true_for_one_word(self):
        with self._patch_transcript([("Bret", "Hello")]):
            self.assertTrue(interaction._last_user_turn_was_low_content())

    def test_low_content_false_for_substantive_answer(self):
        with self._patch_transcript([("Bret", "I'm fixing the time of flight sensors")]):
            self.assertFalse(interaction._last_user_turn_was_low_content())

    def test_low_content_false_for_empty_or_rex_only(self):
        with self._patch_transcript([("rex", "Hey Bret")]):
            self.assertFalse(interaction._last_user_turn_was_low_content())
        with self._patch_transcript([]):
            self.assertFalse(interaction._last_user_turn_was_low_content())

    def test_max_words_threshold_is_config_driven(self):
        with self._patch_transcript([("Bret", "one two three four")]):  # 4 words
            self.assertFalse(interaction._last_user_turn_was_low_content())  # default 3
            with mock.patch.object(interaction.config, "IDLE_BANTER_LOW_CONTENT_MAX_WORDS", 4):
                self.assertTrue(interaction._last_user_turn_was_low_content())


if __name__ == "__main__":
    unittest.main()
