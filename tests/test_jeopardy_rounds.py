"""Daily Double wagers, Final Jeopardy, round jumping, and the selection /
score-cadence fixes (owner batch 2026-08-25).

The live game exposed the shape of play: nobody wagers, nobody reaches Final,
a rejected pick forgets the category it named, and every answer re-reads the
whole scoreboard. These pin the new flows.
"""

import unittest
from unittest import mock

import config
from features import games, jeopardy


_FINAL_CLUE = {
    "category": "WORLD CAPITALS",
    "clue": "Bolivia has two capitals; this one is the seat of government",
    "answer": "La Paz",
}


def _quiet():
    """Silence the physical side effects for a handler call."""
    return [
        mock.patch.object(games, "_body_beat"),
        mock.patch.object(games, "_jeopardy_queue_clip"),
        mock.patch.object(games, "_jeopardy_cancel_timeout"),
        mock.patch.object(games, "_jeopardy_llm_judge", return_value=False),
    ]


class ParseWagerTest(unittest.TestCase):
    def _parse(self, text, mn=5, mx=2000):
        return jeopardy.parse_wager(text, min_wager=mn, max_wager=mx)

    def test_digit_and_word_numbers(self):
        self.assertEqual(self._parse("500"), 500)
        self.assertEqual(self._parse("I'll wager 750"), 750)
        self.assertEqual(self._parse("five hundred"), 500)
        self.assertEqual(self._parse("let's bet fifteen hundred"), 1500)
        self.assertEqual(self._parse("two thousand"), 2000)
        self.assertEqual(self._parse("eight hundred fifty"), 850)
        self.assertEqual(self._parse("a thousand"), 1000)
        self.assertEqual(self._parse("make it a hundred"), 100)

    def test_all_in_shapes(self):
        for text in ["everything", "all of it", "true daily double", "the max", "all in"]:
            self.assertEqual(self._parse(text, mx=1700), 1700, text)

    def test_minimum_shapes(self):
        self.assertEqual(self._parse("the minimum", mn=5), 5)

    def test_out_of_range_returned_raw_for_the_reask(self):
        # The caller owns validation so it can re-ask with the real bounds.
        self.assertEqual(self._parse("nine thousand", mx=2000), 9000)

    def test_no_number_is_none(self):
        for text in ["I have no idea", "hang on", "what do you think?"]:
            self.assertIsNone(self._parse(text), text)


class DailyDoubleWagerTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "jeopardy_round": 1,
            "players": [{"name": "Bret", "score": 600}, {"name": "PJ", "score": 0}],
            "current_player_idx": 0,
            "board": {
                "remaining": 2,
                "categories": [
                    {"name": "SCIENCE", "clues": {
                        400: {"category": "SCIENCE", "value": 400,
                              "clue": "This planet is red", "answer": "Mars",
                              "daily_double": True},
                        600: {"category": "SCIENCE", "value": 600,
                              "clue": "c", "answer": "a"},
                    }},
                ],
            },
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _select_dd(self):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            return games._jeopardy_handle_selection("science for 400", None)

    def test_dd_square_asks_for_a_wager_first(self):
        resp, done = self._select_dd()
        self.assertFalse(done)
        self.assertIn("Daily Double", resp)
        self.assertIn("$1000", resp, "max = max(score 600, round floor 1000)")
        self.assertEqual(games._game_state["phase"], "awaiting_wager")

    def test_wager_reads_the_clue_for_that_amount(self):
        self._select_dd()
        with mock.patch.object(games, "_body_beat"):
            resp, done = games._jeopardy_handle_wager("eight hundred", None)
        self.assertFalse(done)
        self.assertIn("This planet is red", resp)
        self.assertEqual(games._game_state["phase"], "awaiting_answer")
        self.assertEqual(games._game_state["current_clue"]["effective_value"], 800)

    def test_correct_dd_answer_pays_the_wager(self):
        self._select_dd()
        with mock.patch.object(games, "_body_beat"):
            games._jeopardy_handle_wager("1000", None)
        patches = _quiet()
        for p in patches:
            p.start()
        try:
            games._jeopardy_handle_answer("what is Mars", None)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(games._game_state["players"][0]["score"], 1600)

    def test_wrong_dd_answer_costs_the_wager_with_no_rebound(self):
        self._select_dd()
        with mock.patch.object(games, "_body_beat"):
            games._jeopardy_handle_wager("1000", None)
        patches = _quiet()
        for p in patches:
            p.start()
        try:
            resp, done = games._jeopardy_handle_answer("what is Venus", None)
        finally:
            for p in patches:
                p.stop()
        self.assertFalse(done)
        self.assertEqual(games._game_state["players"][0]["score"], -400)
        self.assertNotIn("PJ's turn", resp, "a Daily Double never rebounds")
        self.assertIn("Mars", resp, "the correct response is revealed instead")

    def test_nonsense_wager_reasks(self):
        self._select_dd()
        resp, done = games._jeopardy_handle_wager("hmm let me think", None)
        self.assertFalse(done)
        self.assertIn("number", resp.lower())
        self.assertEqual(games._game_state["phase"], "awaiting_wager")

    def test_out_of_range_wager_reasks_with_the_rails(self):
        self._select_dd()
        resp, _done = games._jeopardy_handle_wager("five thousand", None)
        self.assertIn("$1000", resp)
        self.assertEqual(games._game_state["phase"], "awaiting_wager")

    def test_everything_goes_all_in(self):
        self._select_dd()
        with mock.patch.object(games, "_body_beat"):
            games._jeopardy_handle_wager("everything", None)
        self.assertEqual(games._game_state["current_clue"]["effective_value"], 1000)

    def test_disabled_flag_restores_auto_double(self):
        with mock.patch.object(config, "JEOPARDY_DD_WAGER_ENABLED", False, create=True):
            resp, _done = self._select_dd()
        self.assertEqual(games._game_state["phase"], "awaiting_answer")
        self.assertEqual(games._game_state["current_clue"]["effective_value"], 800)
        self.assertIn("Automatic double", resp)


class PendingCategoryTest(unittest.TestCase):
    """Field 2026-08-25 18:50: "Pop culture for 300" was rejected, then the
    bare "400" picked the last PLAYED category instead of Pop Culture."""

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "players": [{"name": "Bret", "score": 0}],
            "current_player_idx": 0,
            "last_category": "HISTORY",
            "board": {
                "remaining": 4,
                "categories": [
                    {"name": "POP CULTURE", "clues": {
                        400: {"category": "POP CULTURE", "value": 400,
                              "clue": "c1", "answer": "a1"},
                        600: {"category": "POP CULTURE", "value": 600,
                              "clue": "c2", "answer": "a2"},
                    }},
                    {"name": "HISTORY", "clues": {
                        400: {"category": "HISTORY", "value": 400,
                              "clue": "c3", "answer": "a3"},
                    }},
                ],
            },
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _select(self, text):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            return games._jeopardy_handle_selection(text, None)

    def test_bare_value_completes_the_category_the_failed_pick_named(self):
        resp, _done = self._select("pop culture for 300")
        self.assertIn("$300", resp)                       # the rejection
        resp, _done = self._select("400")
        self.assertIn("c1", resp, "POP CULTURE $400, not HISTORY (last played)")
        self.assertNotIn(400, games._game_state["board"]["categories"][0]["clues"])
        self.assertIn(400, games._game_state["board"]["categories"][1]["clues"],
                      "HISTORY's $400 must be untouched")

    def test_hint_survives_a_second_nameless_retry(self):
        self._select("pop culture for 300")
        self._select("the one for 300")                   # still no valid value
        resp, _done = self._select("400")
        self.assertIn("c1", resp)

    def test_successful_pick_consumes_the_hint(self):
        self._select("pop culture for 300")
        self._select("history for 400")                   # explicit pick wins
        self.assertNotIn("pending_category", games._game_state)


class ScoreCadenceTest(unittest.TestCase):
    def setUp(self):
        games._game_state = {
            "players": [{"name": "Bret", "score": 800}, {"name": "PJ", "score": -200}],
        }

    def tearDown(self):
        games._game_state = {}

    def test_short_totals_then_full_board(self):
        player = games._game_state["players"][0]
        with mock.patch.object(config, "JEOPARDY_SCOREBOARD_EVERY", 3, create=True):
            lines = [games._jeopardy_score_announcement(player) for _ in range(3)]
        self.assertIn("That puts Bret at $800", lines[0])
        self.assertIn("That puts Bret at $800", lines[1])
        self.assertIn("Scores:", lines[2])
        self.assertIn("PJ", lines[2])

    def test_round_load_resets_the_counter(self):
        # _jeopardy_load_round zeroes score_events with each fresh board.
        games._game_state["score_events"] = 99
        games._game_state["score_events"] = 0
        line = games._jeopardy_score_announcement(games._game_state["players"][0])
        self.assertIn("That puts", line)


class RoundJumpTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "jeopardy_round": 1,
            "players": [{"name": "Bret", "score": 600}],
            "current_player_idx": 0,
            "board": {"remaining": 20, "categories": [
                {"name": "SCIENCE", "clues": {400: {"category": "SCIENCE",
                                                    "value": 400, "clue": "c",
                                                    "answer": "a"}}},
            ]},
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_jump_phrasings(self):
        for text in ["next round", "let's do double jeopardy", "new categories",
                     "skip to the next round", "fresh board please"]:
            self.assertEqual(jeopardy.round_jump_request(text), "next", text)
        for text in ["final jeopardy", "let's go to the final", "last round"]:
            self.assertEqual(jeopardy.round_jump_request(text), "final", text)

    def test_picks_and_answers_are_not_jumps(self):
        for text in ["pop culture for 400", "double jeopardy for 800",
                     "what is a board game"]:
            self.assertIsNone(jeopardy.round_jump_request(text), text)

    def test_next_round_deals_double_jeopardy(self):
        with mock.patch.object(games, "_jeopardy_load_round",
                               return_value="Double Jeopardy is loaded.") as load:
            resp, done = games._jeopardy_handle_selection("next round", None)
        self.assertFalse(done)
        self.assertIn("Double Jeopardy is loaded.", resp)
        load.assert_called_once_with(2, games._game_state["players"],
                                     current_player_idx=0)

    def test_next_round_from_round_two_goes_to_final(self):
        games._game_state["jeopardy_round"] = 2
        with mock.patch.object(jeopardy, "pick_final_clue",
                               return_value=dict(_FINAL_CLUE)), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            resp, done = games._jeopardy_handle_selection("next round", None)
        self.assertFalse(done)
        self.assertIn("Final Jeopardy", resp)
        self.assertEqual(games._game_state["phase"], "final_wager")

    def test_offer_fires_once_at_half_board(self):
        games._game_state["board"]["remaining"] = 10
        with mock.patch.object(config, "JEOPARDY_ROUND_JUMP_OFFER_REMAINING", 15,
                               create=True):
            first = games._jeopardy_maybe_offer_round_jump()
            second = games._jeopardy_maybe_offer_round_jump()
        self.assertIn("next round", first)
        self.assertEqual(second, "", "once per round only")

    def test_no_offer_on_a_full_board(self):
        games._game_state["board"]["remaining"] = 28
        self.assertEqual(games._jeopardy_maybe_offer_round_jump(), "")


class FinalJeopardyTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "jeopardy_round": 2,
            "players": [{"name": "Bret", "score": 1200}, {"name": "PJ", "score": -200}],
            "current_player_idx": 0,
            "board": {"remaining": 0, "categories": []},
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _begin(self):
        with mock.patch.object(jeopardy, "pick_final_clue",
                               return_value=dict(_FINAL_CLUE)), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            return games._jeopardy_begin_final("")

    def test_begin_announces_category_and_riders_then_asks_first_wager(self):
        resp, done = self._begin()
        self.assertFalse(done)
        self.assertIn("WORLD CAPITALS", resp)
        self.assertIn("ride along", resp, "PJ at -$200 wagers $0 automatically")
        self.assertIn("Bret", resp)
        self.assertIn("wager", resp.lower())
        self.assertEqual(games._game_state["phase"], "final_wager")
        self.assertEqual(games._game_state["final"]["wagers"], {1: 0})

    def test_wager_locks_then_clue_reads_with_the_think_music(self):
        self._begin()
        resp, done = games._jeopardy_handle_final_wager("one thousand", None)
        self.assertFalse(done)
        self.assertIn("Wagers are locked", resp)
        self.assertIn(_FINAL_CLUE["clue"], resp)
        self.assertEqual(games._game_state["phase"], "final_answer")
        self.assertEqual(games._game_state["pending_after_response_clip"], "final_theme")
        self.assertEqual(games._game_state["final"]["wagers"], {1: 0, 0: 1000})
        # Lowest score answers first, show style.
        self.assertIn("PJ, your answer first", resp)

    def test_overbet_reasks_with_the_real_ceiling(self):
        self._begin()
        resp, _done = games._jeopardy_handle_final_wager("two thousand", None)
        self.assertIn("$1200", resp)
        self.assertEqual(games._game_state["phase"], "final_wager")

    def _play_to_answers(self):
        self._begin()
        games._jeopardy_handle_final_wager("one thousand", None)

    def test_answers_collect_in_order_then_reveal_settles_wagers(self):
        self._play_to_answers()
        patches = _quiet()
        for p in patches:
            p.start()
        try:
            resp, done = games._jeopardy_handle_final_answer("what is Sucre", None)
            self.assertFalse(done)
            self.assertIn("Bret, your answer", resp)
            resp, done = games._jeopardy_handle_final_answer("what is La Paz", None)
        finally:
            for p in patches:
                p.stop()
        self.assertTrue(done)
        self.assertIn("La Paz", resp)
        self.assertIn("Bret takes the game", resp)
        # Bret 1200 + 1000; PJ -200 - 0 (rider, wrong, nothing lost).
        self.assertEqual(games._game_state["players"][0]["score"], 2200)
        self.assertEqual(games._game_state["players"][1]["score"], -200)

    def test_clue_repeat_request_rereads_without_consuming_a_turn(self):
        self._play_to_answers()
        resp, done = games._jeopardy_handle_final_answer("can you repeat that", None)
        self.assertFalse(done)
        self.assertIn(_FINAL_CLUE["clue"], resp)
        self.assertEqual(len(games._game_state["final_queue"]), 2,
                         "nobody's answer slot was consumed")

    def test_everyone_broke_skips_final(self):
        games._game_state["players"] = [
            {"name": "Bret", "score": 0}, {"name": "PJ", "score": -400},
        ]
        with mock.patch.object(jeopardy, "pick_final_clue",
                               return_value=dict(_FINAL_CLUE)), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            resp, done = games._jeopardy_begin_final("")
        self.assertTrue(done, "nothing to wager — straight to the finish line")

    def test_disabled_flag_skips_final(self):
        with mock.patch.object(config, "JEOPARDY_FINAL_ENABLED", False, create=True), \
             mock.patch.object(games, "_jeopardy_queue_clip"):
            _resp, done = games._jeopardy_begin_final("")
        self.assertTrue(done)


class ClearGameTimerTest(unittest.TestCase):
    def test_clear_game_cancels_a_live_answer_timer(self):
        timer = mock.MagicMock()
        games._active_game = "jeopardy"
        games._game_state = {"answer_timer": timer, "answer_timer_token": "tok"}
        games._clear_game()
        timer.cancel.assert_called_once()
        self.assertIsNone(games._active_game)


if __name__ == "__main__":
    unittest.main()
