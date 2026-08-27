"""Regressions from the 2026-08-26 20:11 Jeopardy run postmortem.

The owner's report, in his order of annoyance: saying "<category> for 100" got
"pick a dollar value too" over and over; answers came back clipped; and the
transcript credited the wrong people. Each class of defect gets a test here so
the next run cannot quietly re-open one.
"""

import unittest
from unittest import mock

import config
from features import games
from features import jeopardy


def _board(categories):
    return {
        "categories": [
            {"name": name, "clues": {v: {"clue": "c", "answer": "a"} for v in values}}
            for name, values in categories
        ],
        "remaining": sum(len(v) for _n, v in categories),
    }


class SpokenValueTest(unittest.TestCase):
    """"Stadiums for a hundred" must pick a square, not re-ask for the value."""

    def setUp(self):
        self.board = _board([
            ("STADIUMS", [100, 200, 300, 400, 500]),
            ("STATE NICKNAMES", [100, 200]),
        ])

    def _pick(self, text):
        board = _board([
            ("STADIUMS", [100, 200, 300, 400, 500]),
            ("STATE NICKNAMES", [100, 200]),
        ])
        return jeopardy.parse_selection(text, board)

    def test_article_forms_parse(self):
        # The exact shapes that were re-asked four times in a row.
        for text in (
            "Stadiums for a hundred.",
            "Stadiums for a hundred dollars.",
            "State nicknames for a hundred",
        ):
            clue, error = self._pick(text)
            self.assertIsNotNone(clue, f"{text!r} -> {error!r}")
            self.assertEqual(clue["value"], 100)

    def test_bare_and_digit_multipliers_parse(self):
        for text, value in (
            ("stadiums, hundred", 100),
            ("stadiums for 5 hundred", 500),
            ("stadiums for five hundred", 500),
        ):
            clue, error = self._pick(text)
            self.assertIsNotNone(clue, f"{text!r} -> {error!r}")
            self.assertEqual(clue["value"], value)

    def test_established_forms_still_parse(self):
        for text, value in (
            ("stadiums for one hundred", 100),
            ("stadiums for two hundred", 200),
            ("Stadiums for $300.", 300),
            ("stadiums for 400", 400),
        ):
            clue, error = self._pick(text)
            self.assertIsNotNone(clue, f"{text!r} -> {error!r}")
            self.assertEqual(clue["value"], value)

    def test_the_value_never_leaks_into_the_category_query(self):
        self.assertEqual(jeopardy._selection_query("Stadiums for a hundred."), "stadiums")
        self.assertEqual(jeopardy._selection_query("stadiums for 5 hundred"), "stadiums")


class CategoryQuotingTest(unittest.TestCase):
    """The TSV's decorative quotes reached the board, the fuzzy matcher and TTS."""

    def test_escaped_quotes_are_stripped_from_the_name(self):
        self.assertEqual(jeopardy._clean_category('\\"POT"POURRI'), "POTPOURRI")

    def test_apostrophes_survive(self):
        self.assertEqual(jeopardy._clean_category("'80s TV"), "'80s TV")

    def test_a_garbled_category_now_clears_the_fuzzy_gate(self):
        # "Popery" scored 54.5 against the two tokens "pot pourri" and 60.0
        # against the single token "potpourri"; the gate is 58.
        board = _board([(jeopardy._clean_category('\\"POT"POURRI'), [100])])
        clue, error = jeopardy.parse_selection("Popery for a hundred.", board)
        self.assertIsNotNone(clue, error)
        self.assertEqual(clue["category"], "POTPOURRI")

    def test_underscores_are_spoken_as_letters(self):
        self.assertEqual(
            jeopardy.speak_category("TIME TO TAKE THE S_A_T"),
            "TIME TO TAKE THE S A T",
        )


class BareValueFallbackTest(unittest.TestCase):
    """A bare value must not be blamed on a category the player never named."""

    def test_spent_last_category_is_not_reused(self):
        board = _board([("STADIUMS", [100]), ("POTPOURRI", [100])])
        clue, error = jeopardy.parse_selection(
            "One hundred.", board, last_category="STATE NICKNAMES")
        self.assertIsNone(clue)
        self.assertIn("Which category?", error)
        self.assertNotIn("not that category", error)

    def test_live_last_category_still_completes_a_bare_value(self):
        board = _board([("STADIUMS", [100]), ("POTPOURRI", [100])])
        clue, error = jeopardy.parse_selection(
            "One hundred.", board, last_category="STADIUMS")
        self.assertIsNotNone(clue, error)
        self.assertEqual(clue["category"], "STADIUMS")


class NonAnswerGateTest(unittest.TestCase):
    """Nobody gets fined for talking while a clue is live."""

    CLUE = {
        "category": "WORLD HISTORY", "value": 200, "effective_value": 200,
        "clue": "Mussolini abolished all political parties except this one",
        "answer": "the Fascist Party",
    }

    def setUp(self):
        games._active_game = "jeopardy"
        self._fresh()

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _fresh(self):
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": dict(self.CLUE),
            "players": [{"name": "Bret", "score": 0}, {"name": "PJ", "score": 0}],
            "current_player_idx": 0,
            "board": {"remaining": 5, "categories": []},
        }

    def _answer(self, text, verdict="no"):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout") as cancel, \
             mock.patch.object(games, "_quick_call", return_value=verdict) as call:
            response, done = games._jeopardy_handle_answer(text, None)
        return response, games._game_state["players"][0]["score"], cancel, call

    def test_bare_question_stem_is_silent_and_free(self):
        response, score, cancel, call = self._answer("What is?")
        self.assertEqual(response, "")
        self.assertEqual(score, 0)
        self.assertFalse(call.called, "no LLM call for a deterministic ignore")
        self.assertFalse(cancel.called, "the answer clock must keep running")

    def test_a_long_ramble_is_ignored_not_deducted(self):
        rant = (
            "This is rigged because every time it's my turn, it's from a category "
            "I would have never chosen. Also, um, it should have timed out. It "
            "wasn't as buggy yesterday."
        )
        response, score, _cancel, call = self._answer(rant)
        self.assertEqual(response, "")
        self.assertEqual(score, 0)
        self.assertFalse(call.called)

    def test_calling_the_dog_never_reaches_the_judge(self):
        # 6 words, 33 chars — under the length bar and not a stem, so this used
        # to rest entirely on the LLM. It took $400 off Bret in the field.
        response, score, cancel, call = self._answer("Come here, Toby. Come here, baby.")
        self.assertEqual(response, "")
        self.assertEqual(score, 0)
        self.assertFalse(call.called)
        self.assertFalse(cancel.called, "the answer clock must keep running")

    def test_a_bare_player_name_is_not_an_answer(self):
        response, score, _cancel, _call = self._answer("Bret.")
        self.assertEqual(response, "")
        self.assertEqual(score, 0)

    def test_meta_chatter_never_reaches_the_judge(self):
        response, score, _cancel, call = self._answer(
            "Hey, take her points away. She cheated.")
        self.assertEqual(response, "")
        self.assertEqual(score, 0)
        self.assertFalse(call.called)

    def test_judge_none_verdict_is_ignored_and_rearms_the_window(self):
        # A shape no deterministic lane catches — the judge is the only thing
        # that can tell this is not an answer.
        response, score, _cancel, _call = self._answer(
            "Now we could uh start.", verdict="none")
        self.assertEqual(response, "")
        self.assertEqual(score, 0)
        self.assertEqual(games._game_state["phase"], "awaiting_answer")
        self.assertTrue(games._game_state.get("awaiting_prompt_delivery"))
        self.assertNotIn("timeout_rebound", games._game_state)

    def test_the_ignore_streak_settles_the_clue(self):
        with mock.patch.object(config, "JEOPARDY_IGNORE_STREAK_CAP", 2, create=True):
            first, _s, _c, _call = self._answer("Now we could uh start.", verdict="none")
            self.assertEqual(first, "")
            second, _s2, _c2, _call2 = self._answer("And then uh maybe.", verdict="none")
        self.assertNotEqual(second, "", "the square must not hang open forever")
        self.assertEqual(games._game_state["players"][0]["score"], 0,
                         "settling costs nobody anything")

    def test_a_right_answer_that_looks_like_chatter_still_scores(self):
        games._game_state["current_clue"]["answer"] = "Bret"
        response, score, _cancel, _call = self._answer("Who is Bret?")
        self.assertEqual(score, 200)

    def test_a_real_wrong_answer_still_costs(self):
        response, score, _cancel, _call = self._answer("What is the Communist Party?")
        self.assertEqual(score, -200)
        self.assertIn("$200 off Bret", response)

    def test_a_real_right_answer_still_scores(self):
        response, score, _cancel, _call = self._answer("What is the Fascist Party?")
        self.assertEqual(score, 200)

    def test_the_gate_is_switchable(self):
        with mock.patch.object(config, "JEOPARDY_IGNORE_NON_ANSWERS", False, create=True):
            response, score, _cancel, _call = self._answer("What is?")
        self.assertNotEqual(response, "")
        self.assertEqual(score, -200)


class BareStemPredicateTest(unittest.TestCase):
    def test_stems_detected(self):
        for text in ("What is?", "Who is", "what is the", "What are"):
            self.assertTrue(jeopardy.is_bare_question_stem(text), text)

    def test_real_answers_are_not_stems(self):
        for text in ("What is Arizona?", "Who is Fidel Castro?", "Party.", "Alabama"):
            self.assertFalse(jeopardy.is_bare_question_stem(text), text)


class AnswerChargeTest(unittest.TestCase):
    """A miss only costs the current player when it could plausibly be theirs."""

    CLUE = {
        "category": "SAT", "value": 400, "effective_value": 400,
        "clue": "Wrigley's gum is known for this aromatic herb",
        "answer": "spearmint",
    }

    def setUp(self):
        games._active_game = "jeopardy"

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _fresh(self):
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": dict(self.CLUE),
            "players": [
                {"name": "Bret", "score": 0, "person_id": 1},
                {"name": "PJ", "score": 0, "person_id": 7},
            ],
            "current_player_idx": 0,
            "board": {"remaining": 5, "categories": []},
        }

    def _wrong(self, person_id):
        self._fresh()
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_quick_call", return_value="no"):
            response, _done = games._jeopardy_handle_answer("What is a pyramid?", person_id)
        return response, games._game_state["players"]

    def test_the_current_player_pays_for_their_own_miss(self):
        _response, players = self._wrong(1)
        self.assertEqual(players[0]["score"], -400)

    def test_an_unresolved_speaker_still_charges_the_turn(self):
        _response, players = self._wrong(None)
        self.assertEqual(players[0]["score"], -400)

    def test_a_confident_other_contestant_costs_nobody(self):
        response, players = self._wrong(7)
        self.assertEqual(players[0]["score"], 0)
        self.assertEqual(players[1]["score"], 0)
        self.assertIn("PJ", response)

    def test_a_right_answer_from_a_helper_still_scores_for_the_turn(self):
        self._fresh()
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_quick_call", return_value="no"):
            games._jeopardy_handle_answer("What is spearmint?", 7)
        self.assertEqual(games._game_state["players"][0]["score"], 400)

    def test_the_guard_is_switchable(self):
        with mock.patch.object(config, "JEOPARDY_ONLY_CHARGE_THE_ANSWERER",
                               False, create=True):
            _response, players = self._wrong(7)
        self.assertEqual(players[0]["score"], -400)


class ReboundCapTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "players": [{"name": n, "score": 0} for n in ("A", "B", "C", "D")],
            "current_player_idx": 0,
            "current_clue_attempts": [],
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_one_rebound_then_the_clue_is_revealed(self):
        self.assertIsNotNone(games._jeopardy_offer_rebound())
        self.assertIsNone(games._jeopardy_offer_rebound(),
                          "the same clue must not lap the whole table")

    def test_the_cap_is_configurable(self):
        with mock.patch.object(config, "JEOPARDY_MAX_REBOUNDS", 3, create=True):
            self.assertIsNotNone(games._jeopardy_offer_rebound())
            self.assertIsNotNone(games._jeopardy_offer_rebound())
            self.assertIsNotNone(games._jeopardy_offer_rebound())
            self.assertIsNone(games._jeopardy_offer_rebound())

    def test_zero_disables_rebounds(self):
        with mock.patch.object(config, "JEOPARDY_MAX_REBOUNDS", 0, create=True):
            self.assertIsNone(games._jeopardy_offer_rebound())


class CorrectResponsePrefixTest(unittest.TestCase):
    """A stray pronoun in the clue used to outvote the answer's own kind."""

    def test_a_party_is_a_what_not_a_who(self):
        self.assertEqual(
            jeopardy.format_correct_response(
                "Fascist",
                clue="When Mussolini came to power in the 1920s, he abolished all "
                     "political parties in Italy except this one",
                category="WORLD HISTORY",
            ),
            "What is Fascist?",
        )

    def test_a_person_is_still_a_who(self):
        self.assertEqual(
            jeopardy.format_correct_response(
                "Fidel Castro",
                clue="In 1975 this Cuban leader sent several thousand troops to Angola",
                category="WORLD HISTORY",
            ),
            "Who is Fidel Castro?",
        )

    def test_a_place_is_still_a_where(self):
        self.assertEqual(
            jeopardy.format_correct_response(
                "Manila",
                clue="In 1975 Imelda Marcos became the first governor of the "
                     "metropolitan area of this capital",
                category="WORLD HISTORY",
            ),
            "Where is Manila?",
        )


class ShortAnswerFuzzTest(unittest.TestCase):
    """Two short names are always a few edits apart — they need a higher bar."""

    def test_near_neighbours_are_rejected(self):
        for guess, expected in (
            ("What is peppermint?", "spearmint"),
            ("Kansas", "Arkansas"),
            ("Nixon", "Dixon"),
            ("Poland", "Holland"),
        ):
            self.assertFalse(jeopardy.is_correct(guess, expected), f"{guess}/{expected}")

    def test_real_answers_and_garbles_survive(self):
        for guess, expected in (
            ("What is spearmint?", "spearmint"),
            ("Manilla", "Manila"),
            ("Boys are us", 'Toys "R" Us'),
            ("Kennedy", "John F. Kennedy"),
            ("day cart", "Descartes"),
        ):
            self.assertTrue(jeopardy.is_correct(guess, expected), f"{guess}/{expected}")

    def test_the_bump_is_configurable(self):
        with mock.patch.object(config, "JEOPARDY_SHORT_ANSWER_FUZZY_BUMP", 0, create=True):
            self.assertTrue(jeopardy.is_correct("Nixon", "Dixon"))


class TableTalkTest(unittest.TestCase):
    def test_a_side_remark_is_chatter(self):
        self.assertTrue(jeopardy.is_table_chatter(
            "Hey, take her points away. She cheated.", False))

    def test_a_pick_is_never_chatter(self):
        self.assertFalse(jeopardy.is_table_chatter("Stadiums for a hundred", True))
        self.assertFalse(jeopardy.is_table_chatter("can I do the 300 one?", False))

    def test_a_question_is_never_chatter(self):
        self.assertFalse(jeopardy.is_table_chatter("which category is easiest?", False))

    def test_a_short_fragment_keeps_the_canned_retry(self):
        self.assertFalse(jeopardy.is_table_chatter("uh stadiums", False))


class BoardQuestionGateTest(unittest.TestCase):
    """The free-form LLM lane needs a stricter admission test than the
    pattern-matching lanes it sits behind."""

    def test_a_clipped_aux_fragment_is_refused(self):
        self.assertFalse(jeopardy.looks_like_board_question("Is Minneapolis."))
        self.assertTrue(jeopardy.looks_like_question("Is Minneapolis."))

    def test_real_questions_still_pass(self):
        for text in (
            "what's left in pop culture?",
            "Which category is easiest",
            "can we see the scores",
        ):
            self.assertTrue(jeopardy.looks_like_board_question(text), text)


class RoundJumpOfferTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "board": {"remaining": 24, "categories": []},
            "board_size": 30,
            "jeopardy_round": 1,
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_offer_fires_on_clues_played_before_half_a_board(self):
        # 6 played, 24 remaining — far above the 15-remaining trigger.
        self.assertIn("next round", games._jeopardy_maybe_offer_round_jump())

    def test_no_offer_early_in_the_round(self):
        games._game_state["board"]["remaining"] = 28
        self.assertEqual(games._jeopardy_maybe_offer_round_jump(), "")

    def test_once_per_round(self):
        self.assertIn("next round", games._jeopardy_maybe_offer_round_jump())
        self.assertEqual(games._jeopardy_maybe_offer_round_jump(), "")

    def test_both_knobs_zero_disables_the_offer(self):
        with mock.patch.object(config, "JEOPARDY_ROUND_JUMP_OFFER_REMAINING",
                               0, create=True), \
             mock.patch.object(config, "JEOPARDY_ROUND_JUMP_OFFER_AFTER_CLUES",
                               0, create=True):
            self.assertEqual(games._jeopardy_maybe_offer_round_jump(), "")

    def test_the_remaining_trigger_still_works_alone(self):
        games._game_state["board"]["remaining"] = 10
        games._game_state["board_size"] = 12          # only 2 played
        with mock.patch.object(config, "JEOPARDY_ROUND_JUMP_OFFER_AFTER_CLUES",
                               0, create=True):
            self.assertIn("next round", games._jeopardy_maybe_offer_round_jump())


class AnswerTimeoutCeilingTest(unittest.TestCase):
    """A deferral is a courtesy to one in-flight answer, not an open-ended hold."""

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "awaiting_prompt_delivery": True,
            "current_clue": {"category": "C", "value": 100, "clue": "c", "answer": "a"},
            "players": [{"name": "A", "score": 0}, {"name": "B", "score": 0}],
            "current_player_idx": 0,
            "board": {"remaining": 4, "categories": []},
        }

    def tearDown(self):
        timer = games._game_state.pop("answer_timer", None)
        if timer is not None:
            timer.cancel()
        games._game_state = {}
        games._active_game = None

    def test_arming_stamps_a_hard_deadline(self):
        with mock.patch.object(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 12.0, create=True), \
             mock.patch.object(config, "JEOPARDY_TIMEOUT_MAX_DEFER_SECS", 10.0, create=True):
            games._jeopardy_arm_timeout()
        self.assertIn("answer_timer_deadline", games._game_state)

    def test_speech_in_flight_defers_before_the_deadline(self):
        with mock.patch.object(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 12.0, create=True):
            games._jeopardy_arm_timeout()
        token = games._game_state["answer_timer_token"]
        with mock.patch.object(games, "_jeopardy_answer_in_flight", return_value=True), \
             mock.patch.object(games, "_jeopardy_offer_rebound") as rebound:
            games._jeopardy_timeout_fired(token)
        self.assertFalse(rebound.called, "the in-flight answer gets its beat")
        self.assertEqual(games._game_state.get("answer_timer_token"), token)

    def test_past_the_ceiling_the_clue_times_out_through_the_talking(self):
        with mock.patch.object(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 12.0, create=True):
            games._jeopardy_arm_timeout()
        token = games._game_state["answer_timer_token"]
        games._game_state["answer_timer_deadline"] = 0.001    # already elapsed
        with mock.patch.object(games, "_jeopardy_answer_in_flight", return_value=True), \
             mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_offer_rebound", return_value=None), \
             mock.patch.object(games, "_jeopardy_finish_missed_clue",
                               return_value=("Time's up.", False)) as finish, \
             mock.patch("audio.speech_queue.enqueue"):
            games._jeopardy_timeout_fired(token)
        self.assertTrue(finish.called, "13 deferrals held a 12s window open for 31s")

    def test_cancel_clears_the_deadline(self):
        with mock.patch.object(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 12.0, create=True):
            games._jeopardy_arm_timeout()
        games._jeopardy_cancel_timeout()
        self.assertNotIn("answer_timer_deadline", games._game_state)


class AnswerOwnershipDetailTest(unittest.TestCase):
    """An interloper's wrong guess must not cost the current player their square."""

    CLUE = {
        "category": "SAT", "value": 400, "effective_value": 400,
        "clue": "Wrigley's gum is known for this aromatic herb",
        "answer": "spearmint",
    }

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": dict(self.CLUE),
            "players": [
                {"name": "Bret", "score": 0, "person_id": 1},
                {"name": "PJ", "score": 0, "person_id": 7},
            ],
            "current_player_idx": 0,
            "current_clue_attempts": [],
            "board": {"remaining": 5, "categories": []},
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _wrong(self, person_id):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_quick_call", return_value="no"):
            return games._jeopardy_handle_answer("What is a pyramid?", person_id)[0]

    def test_the_square_stays_with_its_owner(self):
        self._wrong(7)
        self.assertIn("current_clue", games._game_state, "the square is not spent")
        self.assertEqual(games._game_state["current_player_idx"], 0)
        self.assertEqual(games._game_state["phase"], "awaiting_answer")
        self.assertTrue(games._game_state.get("awaiting_prompt_delivery"))

    def test_the_rebound_chance_is_not_burned(self):
        self._wrong(7)
        self.assertEqual(games._game_state.get("current_clue_attempts"), [])

    def test_an_off_roster_bystander_is_charged_as_before(self):
        self._wrong(8)
        self.assertEqual(games._game_state["players"][0]["score"], -400)


class RosterAccessorTest(unittest.TestCase):
    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "players": [
                {"name": "Bret", "person_id": 1},
                {"name": "PJ", "person_id": 7},
                {"name": "Guest"},
            ],
            "current_player_idx": 1,
            "phase": "awaiting_answer",
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_roster_ids_skip_players_with_no_row(self):
        self.assertEqual(games.active_roster_person_ids(), frozenset({1, 7}))

    def test_current_player_id(self):
        self.assertEqual(games.active_game_current_player_id(), 7)

    def test_answer_window_probe(self):
        self.assertTrue(games.jeopardy_answer_window_open())
        games._game_state["phase"] = "selecting"
        self.assertFalse(games.jeopardy_answer_window_open())

    def test_everything_is_empty_with_no_game(self):
        games._active_game = None
        self.assertEqual(games.active_roster_person_ids(), frozenset())
        self.assertIsNone(games.active_game_current_player_id())
        self.assertFalse(games.jeopardy_answer_window_open())


class TranscriptDedupeTest(unittest.TestCase):
    """Rex re-asked "Pick a dollar value too" four times; the transcript showed
    two. The 30 s identical-line window is there to swallow a genuine
    double-write within ONE turn, and a human turn in between proves these were
    two real answers."""

    def setUp(self):
        from utils import conv_log
        self.conv_log = conv_log
        self.written = []
        self._append = conv_log._append_locked
        conv_log._append_locked = self.written.append
        conv_log.clear_dedupe_state()
        self.addCleanup(setattr, conv_log, "_append_locked", self._append)
        self.addCleanup(conv_log.clear_dedupe_state)

    def _rex(self, text):
        with mock.patch.object(self.conv_log, "_mirror_to_gui"):
            self.conv_log.log_rex(text)

    def _heard(self, text):
        with mock.patch.object(self.conv_log, "_mirror_to_gui"), \
             mock.patch.object(self.conv_log, "_write") as write:
            self.conv_log.log_heard("Bret", text)
        return write

    def test_a_true_double_write_is_still_deduped(self):
        self._rex("Pick a dollar value too.")
        self._rex("Pick a dollar value too.")
        self.assertEqual(len(self.written), 1)

    def test_a_human_turn_makes_the_next_identical_line_real(self):
        self._rex("Pick a dollar value too.")
        self._heard("Stadiums for a hundred.")
        self._rex("Pick a dollar value too.")
        self.assertEqual(len(self.written), 2,
                         "the re-ask was audible and must appear in the transcript")


if __name__ == "__main__":
    unittest.main()
