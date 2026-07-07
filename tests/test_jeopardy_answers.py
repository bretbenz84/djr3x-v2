"""Jeopardy spoken-answer judging (features/jeopardy.py + the games.py LLM judge).

Players answer by VOICE, so the matcher must absorb what speech recognition
actually produces: phonetically mangled proper nouns ("day cart" for Descartes),
number words for digits ("fourteen ninety two"), spoken ordinals for regnal
numerals ("Henry the eighth"), contractions ("what's Paris"), and conversational
lead-ins ("um, I think it's...") — while never crediting a genuinely different
answer. 2026-07-07 review: these were the live quality gaps.
"""

import unittest
from unittest import mock

import config
from features import games, jeopardy


class IsCorrectSpeechShapesTest(unittest.TestCase):
    """Spoken shapes that must be accepted."""

    def _yes(self, user, expected):
        self.assertTrue(jeopardy.is_correct(user, expected), f"{user!r} vs {expected!r}")

    def _no(self, user, expected):
        self.assertFalse(jeopardy.is_correct(user, expected), f"{user!r} vs {expected!r}")

    def test_phonetic_garbles_accepted(self):
        self._yes("day cart", "Descartes")
        self._yes("shack", "Shaq")
        self._yes("van go", "van Gogh")

    def test_surname_alone_accepted(self):
        # Real Jeopardy accepts the surname ("Who is Poe?").
        self._yes("Poe", "Edgar Allan Poe")
        self._yes("Twain", "Mark Twain")
        self._yes("who is Curie", "Marie Curie")
        self._yes("washington", "Denzel Washington")

    def test_regnal_numbers_spoken_as_ordinals(self):
        self._yes("Henry the eighth", "Henry VIII")
        self._yes("henry the fifth", "Henry V")

    def test_spoken_years_and_numbers(self):
        self._yes("fourteen ninety two", "1492")
        self._yes("nineteen sixty nine", "1969")
        self._yes("seventeen seventy six", "1776")
        self._yes("nineteen oh five", "1905")
        self._yes("two thousand one", "2001")

    def test_contractions_and_leadins_stripped(self):
        self._yes("what's Paris", "Paris")
        self._yes("is it gold", "gold")
        self._yes("um, I think it is Shakespeare", "William Shakespeare")
        self._yes("what is the mississippi", "the Mississippi")

    def test_wrong_answers_stay_wrong(self):
        self._no("London", "Paris")
        self._no("Beethoven", "Mozart")
        self._no("cat", "dog")
        self._no("the moon", "the sun")
        self._no("Jefferson", "Thomas Edison")
        self._no("fourteen ninety three", "1492")

    def test_tiny_substrings_not_credited(self):
        # partial_ratio matches any 2-letter fragment inside a long answer;
        # the length guard must reject these.
        self._no("an", "Edgar Allan Poe")
        self._no("ed", "Edgar Allan Poe")

    def test_multi_part_answers_still_require_all_parts(self):
        # The phonetic path must not re-open the shared-prefix hole.
        self._no("license", "license & registration")
        self._yes("license and registration", "license & registration")


class NormalizeAnswerTest(unittest.TestCase):
    def test_strips_stacked_spoken_leadins(self):
        self.assertEqual(jeopardy.normalize_answer("um, well, I think it's Paris"), "paris")
        self.assertEqual(jeopardy.normalize_answer("what's a corvette"), "corvette")
        self.assertEqual(jeopardy.normalize_answer("the answer is Rome"), "rome")

    def test_canonicalizes_numbers(self):
        self.assertEqual(jeopardy.normalize_answer("Henry VIII"), "henry 8")
        self.assertEqual(jeopardy.normalize_answer("henry the eighth"), "henry 8")
        self.assertEqual(jeopardy.normalize_answer("forty two"), "42")
        self.assertEqual(jeopardy.normalize_answer("the 8th amendment"), "8 amendment")


class SpokenNumberStringTest(unittest.TestCase):
    def test_year_pairs(self):
        self.assertEqual(jeopardy._spoken_number_string("14 92"), "1492")
        self.assertEqual(jeopardy._spoken_number_string("19 oh 5"), "1905")

    def test_multiplier_forms(self):
        self.assertEqual(jeopardy._spoken_number_string("2 thousand"), "2000")
        self.assertEqual(jeopardy._spoken_number_string("8 hundred 50"), "850")

    def test_non_number_text_returns_none(self):
        self.assertIsNone(jeopardy._spoken_number_string("paris"))
        self.assertIsNone(jeopardy._spoken_number_string("19 paris"))


class PassDetectionTest(unittest.TestCase):
    def test_pass_phrases(self):
        for text in ("no clue", "beats me", "I give up", "dunno", "pass",
                     "I don't know", "no idea", "not sure", "I got nothing"):
            self.assertTrue(jeopardy.is_pass_or_timeout(text), text)

    def test_answers_are_not_passes(self):
        for text in ("the Nile", "what is Paris", "Shakespeare", "no man's land"):
            self.assertFalse(jeopardy.is_pass_or_timeout(text), text)


class LlmJudgeTest(unittest.TestCase):
    """The strict rescue judge: only consulted on deterministic-wrong, fail-safe."""

    _CLUE = {"clue": "This team won Super Bowl XXIX", "category": "NFL"}

    def test_disabled_flag_short_circuits(self):
        with mock.patch.object(config, "JEOPARDY_LLM_JUDGE_ENABLED", False, create=True), \
             mock.patch.object(games, "_quick_call") as call:
            self.assertFalse(games._jeopardy_llm_judge("forty niners", "the 49ers", self._CLUE))
            call.assert_not_called()

    def test_yes_verdict_rescues(self):
        with mock.patch.object(games, "_quick_call", return_value="yes"):
            self.assertTrue(games._jeopardy_llm_judge("forty niners", "the 49ers", self._CLUE))

    def test_no_verdict_stays_wrong(self):
        with mock.patch.object(games, "_quick_call", return_value="no"):
            self.assertFalse(games._jeopardy_llm_judge("the Steelers", "the 49ers", self._CLUE))

    def test_error_fails_safe(self):
        with mock.patch.object(games, "_quick_call", side_effect=RuntimeError("boom")):
            self.assertFalse(games._jeopardy_llm_judge("forty niners", "the 49ers", self._CLUE))

    def test_rambling_turn_not_judged(self):
        long_text = "so anyway like I was telling you earlier about my weekend " * 4
        with mock.patch.object(games, "_quick_call") as call:
            self.assertFalse(games._jeopardy_llm_judge(long_text, "the 49ers", self._CLUE))
            call.assert_not_called()

    def test_handle_answer_consults_judge_only_on_wrong(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": {"category": "NFL", "value": 400, "effective_value": 400,
                             "clue": "This team won Super Bowl XXIX",
                             "answer": "the 49ers"},
            "players": [{"name": "Bret", "score": 0}],
            "current_player_idx": 0,
            "board": {"remaining": 5, "categories": []},
        }
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_jeopardy_llm_judge", return_value=True) as judge:
            resp, done = games._jeopardy_handle_answer("forty niners", None)
        judge.assert_called_once()
        self.assertFalse(done)
        # The rescued answer scores like a normal correct one.
        self.assertEqual(games._game_state["players"][0]["score"], 400)
        games._game_state = {}
        games._active_game = None


class GuiCategoriesReminderTest(unittest.TestCase):
    """With the GUI board on screen, the per-turn spoken category read-out is
    skipped (it's tiresome when you can SEE the board); voice-only play keeps it."""

    def setUp(self):
        games._game_state = {
            "board": {
                "categories": [
                    {"name": "SCIENCE", "clues": {200: {}}},
                    {"name": "HISTORY", "clues": {400: {}}},
                ],
                "remaining": 2,
            },
        }

    def tearDown(self):
        games._game_state = {}

    def test_voice_only_reads_categories(self):
        with mock.patch.object(config, "GUI_ENABLED", False, create=True):
            reminder = games._jeopardy_categories_reminder()
        self.assertIn("SCIENCE", reminder)
        self.assertIn("HISTORY", reminder)

    def test_gui_skips_the_readout(self):
        with mock.patch.object(config, "GUI_ENABLED", True, create=True):
            self.assertEqual(games._jeopardy_categories_reminder(), "")

    def test_override_restores_readout_with_gui(self):
        with mock.patch.object(config, "GUI_ENABLED", True, create=True), \
             mock.patch.object(config, "JEOPARDY_READ_CATEGORIES_WITH_GUI", True, create=True):
            self.assertIn("SCIENCE", games._jeopardy_categories_reminder())


if __name__ == "__main__":
    unittest.main()
