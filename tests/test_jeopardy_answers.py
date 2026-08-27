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

    def test_gui_default_now_speaks_the_reminder(self):
        # The blanket GUI mute silently killed the reminder for players sitting
        # around the robot instead of the laptop (field 2026-08-26: zero
        # read-outs across twelve scoring turns). The fatigue curve, not a
        # blanket mute, is what keeps it from being tiresome.
        with mock.patch.object(config, "GUI_ENABLED", True, create=True):
            self.assertIn("SCIENCE", games._jeopardy_categories_reminder())

    def test_gui_mute_opt_out_skips_the_readout(self):
        with mock.patch.object(config, "GUI_ENABLED", True, create=True), \
             mock.patch.object(config, "JEOPARDY_READ_CATEGORIES_WITH_GUI", False, create=True):
            self.assertEqual(games._jeopardy_categories_reminder(), "")


class CategoriesReminderCadenceTest(unittest.TestCase):
    """Voice-only fatigue curve (owner call 2026-08-25): the per-turn category
    read-back is great early game and tiresome once everyone knows the board.
    First FULL_READS scoring turns read it every time, then every EVERY-th."""

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
        self._patches = [
            mock.patch.object(config, "GUI_ENABLED", False, create=True),
            mock.patch.object(config, "JEOPARDY_CATEGORIES_REMINDER_FULL_READS", 2, create=True),
            mock.patch.object(config, "JEOPARDY_CATEGORIES_REMINDER_EVERY", 3, create=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        games._game_state = {}

    def test_full_reads_then_periodic(self):
        # FULL_READS=2, EVERY=3: reads 1-2 speak; then only every 3rd after.
        spoken = [bool(games._jeopardy_categories_reminder()) for _ in range(8)]
        self.assertEqual(spoken, [True, True, False, False, True, False, False, True])

    def test_new_round_resets_the_curve(self):
        for _ in range(3):
            games._jeopardy_categories_reminder()
        self.assertFalse(games._jeopardy_categories_reminder())
        # _jeopardy_load_round zeroes the counter with each fresh board.
        games._game_state["categories_reminder_reads"] = 0
        self.assertTrue(games._jeopardy_categories_reminder())

    def test_every_zero_means_never_again_this_round(self):
        with mock.patch.object(config, "JEOPARDY_CATEGORIES_REMINDER_EVERY", 0, create=True):
            results = [bool(games._jeopardy_categories_reminder()) for _ in range(6)]
        self.assertEqual(results, [True, True, False, False, False, False])

    def test_gui_mute_opt_out_does_not_consume_reads(self):
        with mock.patch.object(config, "GUI_ENABLED", True, create=True), \
             mock.patch.object(config, "JEOPARDY_READ_CATEGORIES_WITH_GUI", False, create=True):
            self.assertEqual(games._jeopardy_categories_reminder(), "")
        self.assertNotIn("categories_reminder_reads", games._game_state)

    def test_the_curve_is_the_same_with_the_gui_up(self):
        results = []
        with mock.patch.object(config, "GUI_ENABLED", True, create=True), \
             mock.patch.object(config, "JEOPARDY_CATEGORIES_REMINDER_FULL_READS", 2, create=True), \
             mock.patch.object(config, "JEOPARDY_CATEGORIES_REMINDER_EVERY", 3, create=True):
            for _ in range(6):
                results.append(bool(games._jeopardy_categories_reminder()))
        self.assertEqual(results, [True, True, False, False, True, False])

    def test_explicit_board_request_ignores_the_curve(self):
        games._game_state["categories_reminder_reads"] = 99
        self.assertIn("SCIENCE", games._jeopardy_board_text())


class BoardQuestionLanesTest(unittest.TestCase):
    """Mid-game board questions (owner ask 2026-08-25): category-specific
    remaining values, value availability, scores, and whose turn — answered
    without consuming a square or grading the question as a wrong answer."""

    def _board(self):
        return {
            "categories": [
                {"name": "POP CULTURE", "clues": {600: {}, 1000: {}}},
                {"name": "STATE ABBREV.", "clues": {200: {}, 400: {}}},
                {"name": "HISTORY", "clues": {}},
            ],
            "remaining": 4,
        }

    def test_category_query_variations(self):
        board = self._board()
        for text in [
            "what's left in pop culture",
            "What is left in pop culture?",
            "what's still open in pop culture",
            "what squares are free in pop culture",
            "what squares are left in pop culture",
            "what values are left in pop culture",
            "what dollar amounts are left in pop culture",
            "what do we have left in pop culture",
            "how much is left in pop culture",
            "how many are left in pop culture",
            "is there anything left in pop culture",
            "what's remaining in pop culture",
            "what does pop culture have left",
            "what does pop culture still have",
        ]:
            result = jeopardy.category_board_query(text, board)
            self.assertIsNotNone(result, text)
            category, _fragment = result
            self.assertIsNotNone(category, text)
            self.assertEqual(category["name"], "POP CULTURE", text)

    def test_cleaned_out_category_still_matches(self):
        # An empty category answers "nothing left", not "no such category".
        result = jeopardy.category_board_query("what's left in history", self._board())
        category, _fragment = result
        self.assertEqual(category["name"], "HISTORY")
        self.assertEqual(category["clues"], {})

    def test_unknown_category_returns_the_fragment(self):
        result = jeopardy.category_board_query("what's left in wibble wobble", self._board())
        self.assertIsNotNone(result)
        category, fragment = result
        self.assertIsNone(category)
        self.assertIn("wibble", fragment)

    def test_picks_are_not_category_queries(self):
        for text in [
            "pop culture for six hundred",
            "I'll take pop culture for 600",
            "what is the McRib",
            "give me history",
        ]:
            self.assertIsNone(jeopardy.category_board_query(text, self._board()), text)

    def test_value_availability_variations(self):
        board = self._board()
        for text in [
            "is the 400 still there in state abbrev",
            "is the $400 still available in state abbrev?",
            "is 400 still open in state abbrev",
            "is the four hundred still there in state abbrev",
            "do you still have the 400 in state abbrev",
            "is there still a 400 in state abbrev",
            "is there a 400 left in state abbrev",
            "is state abbrev for 400 still available",
        ]:
            result = jeopardy.value_availability_query(text, board)
            self.assertIsNotNone(result, text)
            self.assertEqual(result["value"], 400, text)
            self.assertIsNotNone(result["category"], text)
            self.assertEqual(result["category"]["name"], "STATE ABBREV.", text)

    def test_value_availability_without_category_lists_where(self):
        result = jeopardy.value_availability_query("is the 1000 still up", self._board())
        self.assertIsNotNone(result)
        self.assertIsNone(result["category"])
        self.assertEqual(result["open_in"], ["POP CULTURE"])

    def test_picks_are_not_availability_questions(self):
        for text in [
            "pop culture for 400",
            "I'll take the 400 in state abbrev",
            "give me state abbrev for four hundred",
            "400",
        ]:
            self.assertIsNone(jeopardy.value_availability_query(text, self._board()), text)

    def test_score_and_turn_requests(self):
        for text in [
            "what's the score",
            "what are the scores?",
            "who's winning",
            "score check",
            "how much do I have",
            "what am I at",
        ]:
            self.assertTrue(jeopardy.is_score_request(text), text)
        for text in ["whose turn is it", "who's up", "whose pick is it", "is it my turn"]:
            self.assertTrue(jeopardy.is_turn_request(text), text)

    def test_question_shape_gate(self):
        for text in ["what else is on the board?", "can we see the scores", "Pop culture?"]:
            self.assertTrue(jeopardy.looks_like_question(text), text)
        for text in ["pop culture for 600", "the McRib", "I'll take history"]:
            self.assertFalse(jeopardy.looks_like_question(text), text)


class BoardQuestionHandlerTest(unittest.TestCase):
    """The games.py wiring: questions answered in both phases, squares never
    consumed, no deductions, and the LLM fallback gated correctly."""

    def _board(self):
        return {
            "categories": [
                {"name": "POP CULTURE", "clues": {600: {"clue": "c", "answer": "a"},
                                                  1000: {"clue": "c", "answer": "a"}}},
                {"name": "HISTORY", "clues": {400: {"clue": "c", "answer": "a"}}},
            ],
            "remaining": 3,
        }

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "players": [{"name": "Bret", "score": 200}, {"name": "PJ", "score": -400}],
            "current_player_idx": 0,
            "board": self._board(),
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_category_question_answers_without_consuming(self):
        resp, done = games._jeopardy_handle_selection("what's left in pop culture", None)
        self.assertFalse(done)
        self.assertIn("$600", resp)
        self.assertIn("$1000", resp)
        self.assertEqual(games._game_state["board"]["remaining"], 3,
                         "a question must never consume a square")

    def test_availability_question_does_not_pick_the_square(self):
        resp, done = games._jeopardy_handle_selection(
            "is the 400 still there in history", None
        )
        self.assertFalse(done)
        self.assertIn("still on the board", resp.lower())
        self.assertIn(400, games._game_state["board"]["categories"][1]["clues"],
                      "the availability question must not consume HISTORY $400")

    def test_gone_square_reported_gone(self):
        del games._game_state["board"]["categories"][1]["clues"][400]
        resp, _done = games._jeopardy_handle_selection(
            "is the 400 still there in history", None
        )
        self.assertIn("gone", resp.lower())

    def test_score_question_answered_in_selecting_phase(self):
        resp, _done = games._jeopardy_handle_selection("what's the score", None)
        self.assertIn("Bret: $200", resp)
        self.assertIn("negative $400", resp)

    def test_meta_question_during_a_clue_costs_nothing_and_rereads(self):
        games._game_state["phase"] = "awaiting_answer"
        games._game_state["current_clue"] = {
            "category": "POP CULTURE", "value": 600, "effective_value": 600,
            "clue": "This clue", "answer": "zzz-not-a-match",
        }
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_jeopardy_llm_judge", return_value=False):
            resp, done = games._jeopardy_handle_answer("what's left in history", None)
        self.assertFalse(done)
        self.assertIn("$400", resp)
        self.assertIn("This clue", resp)     # the live clue is re-read
        self.assertEqual(games._game_state["players"][0]["score"], 200,
                         "a board question must never cost points")

    def test_llm_fallback_gets_the_board_context(self):
        with mock.patch.object(games, "_rex_respond", return_value="LLM says hi") as rex:
            resp, done = games._jeopardy_handle_selection(
                "which category do you think is easiest?", None
            )
        self.assertFalse(done)
        self.assertEqual(resp, "LLM says hi")
        context = rex.call_args.args[0]
        self.assertIn("POP CULTURE", context)
        self.assertIn("Bret", context)

    def test_llm_fallback_never_swallows_a_value_mention(self):
        # A mangled pick with a value keeps the deterministic retry error.
        with mock.patch.object(games, "_rex_respond") as rex:
            resp, _done = games._jeopardy_handle_selection(
                "can I do the 300 one?", None
            )
        rex.assert_not_called()
        self.assertIn("$300", resp)

    def test_llm_fallback_kill_switch(self):
        with mock.patch.object(config, "JEOPARDY_BOARD_QA_LLM_FALLBACK_ENABLED", False,
                               create=True), \
             mock.patch.object(games, "_rex_respond") as rex:
            games._jeopardy_handle_selection("which category is easiest?", None)
        rex.assert_not_called()

    def test_non_question_gibberish_keeps_the_canned_error(self):
        with mock.patch.object(games, "_rex_respond") as rex:
            resp, _done = games._jeopardy_handle_selection("banana banana", None)
        rex.assert_not_called()
        self.assertIn("dollar value", resp)


class StopConfirmationTest(unittest.TestCase):
    """The are-you-sure guard (owner ask 2026-08-25): a stop attempt asks "But
    we're having so much fun…" and only an affirmative actually ends the game."""

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": {"category": "SPACE", "value": 400, "effective_value": 400,
                             "clue": "The live clue", "answer": "orbit"},
            "players": [{"name": "Bret", "score": 0}],
            "current_player_idx": 0,
            "board": {"remaining": 3, "categories": [{"name": "SPACE", "clues": {200: {}}}]},
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_request_arms_and_freezes_the_answer_clock(self):
        with mock.patch.object(games, "_jeopardy_cancel_timeout") as cancel:
            question = games.request_stop_confirmation()
        self.assertIn("having so much fun", question)
        self.assertIn("stop_confirm_at", games._game_state)
        cancel.assert_called_once()

    def test_no_active_game_returns_none(self):
        games._active_game = None
        self.assertIsNone(games.request_stop_confirmation())

    def test_affirmative_stops_the_game(self):
        games._game_state["stop_confirm_at"] = games.time.monotonic()
        with mock.patch.object(games, "stop_game", return_value="Fine. Scores: ...") as stop:
            result = games.resolve_stop_confirmation("yes, I'm sure", 1)
        kind, line = result
        self.assertEqual(kind, "stop")
        self.assertEqual(line, "Fine. Scores: ...")
        stop.assert_called_once_with(1)

    def test_affirmative_variations(self):
        for text in [
            "yes", "Yeah.", "yep", "sure", "absolutely", "I'm sure",
            "okay yes", "end it", "do it", "go ahead",
        ]:
            self.assertEqual(games._stop_confirm_verdict(text), "yes", text)

    def test_negative_variations_resume(self):
        for text in [
            "no", "Nope.", "nah", "never mind", "just kidding",
            "keep playing", "let's keep going", "no, continue",
        ]:
            self.assertEqual(games._stop_confirm_verdict(text), "no", text)

    def test_decline_rereads_the_live_clue(self):
        games._game_state["stop_confirm_at"] = games.time.monotonic()
        result = games.resolve_stop_confirmation("no way, keep playing", None)
        kind, line = result
        self.assertEqual(kind, "resume")
        self.assertIn("The live clue", line)
        self.assertTrue(games._game_state["awaiting_prompt_delivery"],
                        "the answer window restarts after the re-read")

    def test_decline_in_selecting_phase_reprompts_the_picker(self):
        games._game_state["phase"] = "selecting"
        games._game_state.pop("current_clue")
        games._game_state["stop_confirm_at"] = games.time.monotonic()
        kind, line = games.resolve_stop_confirmation("nope", None)
        self.assertEqual(kind, "resume")
        self.assertIn("Bret", line)
        self.assertIn("pick a category", line)

    def test_unrelated_reply_drops_the_ask_and_passes_through(self):
        games._game_state["stop_confirm_at"] = games.time.monotonic()
        kind, line = games.resolve_stop_confirmation("what is orbit", None)
        self.assertEqual(kind, "pass")
        self.assertIsNone(line)
        self.assertNotIn("stop_confirm_at", games._game_state)

    def test_repeated_stop_demand_counts_as_affirmative(self):
        games._game_state["stop_confirm_at"] = games.time.monotonic()
        with mock.patch.object(games, "stop_game", return_value="Done.") as stop:
            kind, _line = games.resolve_stop_confirmation(
                "stop the game", None, stop_shaped=True
            )
        self.assertEqual(kind, "stop")
        stop.assert_called_once()

    def test_expired_ask_is_ignored(self):
        games._game_state["stop_confirm_at"] = games.time.monotonic() - 10_000.0
        self.assertIsNone(games.resolve_stop_confirmation("yes", None))
        self.assertNotIn("stop_confirm_at", games._game_state)

    def test_nothing_pending_returns_none(self):
        self.assertIsNone(games.resolve_stop_confirmation("yes", None))


class BoardRepeatRequestTest(unittest.TestCase):
    """The categories are announced once when a round loads. A voice-only player
    who missed them must be able to ask for them back — the old selection parser
    answered "pick a dollar value too", which reads as being ignored."""

    def _board(self, uneven: bool = False):
        history = {200: {}, 400: {}} if not uneven else {400: {}}
        return {
            "categories": [
                {"name": "SCIENCE", "clues": {200: {}, 400: {}}},
                {"name": "HISTORY", "clues": history},
            ],
            "remaining": 4 if not uneven else 3,
        }

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "selecting",
            "players": [{"name": "Bret", "score": 0}],
            "current_player_idx": 0,
            "board": self._board(),
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def test_request_phrasings_detected(self):
        for text in [
            "what are the categories",
            "what's on the board",
            "what were the categories again",
            "repeat the categories",
            "read me the board",
            "remind me of the categories",
            "the categories one more time",
            "what categories are left",
            "what's still available",
            "say that again",
            "can you repeat the clue",
        ]:
            self.assertTrue(jeopardy.is_board_request(text), text)

    def test_a_real_selection_is_never_a_board_request(self):
        # A dollar value means they are picking a square, not asking.
        for text in [
            "science for 400",
            "I'll take history for two hundred",
            "same category for 400",
            "give me the science category for 200",
        ]:
            self.assertFalse(jeopardy.is_board_request(text), text)

    def test_selection_phase_reads_the_board_back(self):
        resp, done = games._jeopardy_handle_selection("what are the categories", None)
        self.assertFalse(done)
        self.assertIn("SCIENCE", resp)
        self.assertIn("HISTORY", resp)
        self.assertIn("pick a category", resp)
        # Asking costs nothing: no square consumed, still their turn to select.
        self.assertEqual(games._game_state["board"]["remaining"], 4)
        self.assertEqual(games._game_state["phase"], "selecting")

    def test_explicit_ask_answers_even_with_the_gui_up(self):
        # The GUI mutes the per-turn reminder, not a direct question.
        with mock.patch.object(config, "GUI_ENABLED", True, create=True):
            resp, _done = games._jeopardy_handle_selection("what are the categories", None)
        self.assertIn("SCIENCE", resp)

    def test_even_board_collapses_to_one_value_list(self):
        readout = jeopardy.format_board_readout(self._board())
        self.assertIn("SCIENCE. HISTORY", readout)
        self.assertIn("Each one still has $200, $400", readout)

    def test_uneven_board_itemizes_values_per_category(self):
        readout = jeopardy.format_board_readout(self._board(uneven=True))
        self.assertIn("SCIENCE for $200, $400", readout)
        self.assertIn("HISTORY for $400", readout)

    def test_empty_board_reads_back_as_empty(self):
        self.assertEqual(jeopardy.format_board_readout({"categories": []}), "")


class ClueRepeatRequestTest(unittest.TestCase):
    """Asking to hear a live clue again must not be scored as a wrong answer."""

    _CLUE = {
        "category": "NFL", "value": 400, "effective_value": 400,
        "clue": "This team won Super Bowl XXIX", "answer": "the 49ers",
    }

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": dict(self._CLUE),
            "players": [{"name": "Bret", "score": 0}],
            "current_player_idx": 0,
            "board": {
                "remaining": 2,
                "categories": [{"name": "NFL", "clues": {200: {}}}],
            },
        }

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _handle(self, text):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_jeopardy_llm_judge", return_value=False):
            return games._jeopardy_handle_answer(text, None)

    def test_repeat_request_rereads_the_clue_without_scoring(self):
        resp, done = self._handle("can you repeat the clue")
        self.assertFalse(done)
        self.assertIn("Super Bowl XXIX", resp)
        self.assertEqual(games._game_state["players"][0]["score"], 0)
        # Clue stays live and the answer window restarts.
        self.assertEqual(games._game_state["phase"], "awaiting_answer")
        self.assertTrue(games._game_state["awaiting_prompt_delivery"])
        self.assertTrue(games._game_state.get("current_clue"))

    def test_repeat_request_never_reaches_the_llm_judge(self):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_jeopardy_llm_judge") as judge:
            games._jeopardy_handle_answer("say that again", None)
        judge.assert_not_called()

    def test_categories_asked_mid_clue_costs_nothing(self):
        resp, done = self._handle("wait what are the categories")
        self.assertFalse(done)
        self.assertIn("Still on the board", resp)
        self.assertIn("Super Bowl XXIX", resp)  # the live clue is re-read too
        self.assertEqual(games._game_state["players"][0]["score"], 0)

    def test_a_correct_answer_is_still_scored_not_intercepted(self):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"):
            _resp, _done = games._jeopardy_handle_answer("what is the 49ers", None)
        self.assertEqual(games._game_state["players"][0]["score"], 400)

    def test_repeat_shapes_do_not_swallow_jeopardy_style_answers(self):
        # "What is the board?" is a legal response phrasing — the mid-clue
        # interceptor must not treat it as a request to re-read.
        for text in [
            "what is the board",
            "what are the options",
            "who is Bill Board",
            "the board of directors",
            "a repeat offender",
        ]:
            self.assertFalse(jeopardy.is_clue_repeat_request(text), text)


class SpeakCategoryTest(unittest.TestCase):
    """Dataset category abbreviations are expanded for SPEECH only — the TTS read
    "COMBINED STATE ABBREV." as "abreev" and the players couldn't tell what the
    category was (field 2026-08-25). The raw name stays on the board/GUI and in
    the selection matcher."""

    def test_abbrev_expanded(self):
        self.assertEqual(
            jeopardy.speak_category("COMBINED STATE ABBREV."),
            "COMBINED STATE ABBREVIATIONS",
        )

    def test_case_follows_the_original_token(self):
        self.assertEqual(jeopardy.speak_category("Misc. Facts"), "Miscellaneous Facts")

    def test_plain_names_untouched(self):
        for name in ["VAMPIRE DIARIES", "POP CULTURE", "WE MAKE THAT"]:
            self.assertEqual(jeopardy.speak_category(name), name)

    def test_trailing_period_dropped_even_without_expansion(self):
        # "JUMPING JUPITER!." style double stops read badly; the sentence the
        # name lands in supplies its own period.
        self.assertEqual(jeopardy.speak_category("ODD FACTS."), "ODD FACTS")

    def test_real_words_not_falsely_expanded(self):
        # "lit"/"pres" style tokens are real words in other categories — only the
        # unambiguous map entries expand.
        self.assertEqual(jeopardy.speak_category("GETTING LIT"), "GETTING LIT")

    def test_format_categories_speaks_expanded_names(self):
        board = {
            "categories": [
                {"name": "STATE ABBREV.", "clues": {200: {}}},
                {"name": "HISTORY", "clues": {400: {}}},
            ],
        }
        text = jeopardy.format_categories(board)
        self.assertIn("ABBREVIATIONS", text)
        self.assertNotIn("ABBREV.", text)

    def test_board_readout_speaks_expanded_names(self):
        board = {
            "categories": [
                {"name": "STATE ABBREV.", "clues": {200: {}, 400: {}}},
            ],
        }
        self.assertIn("ABBREVIATIONS", jeopardy.format_board_readout(board))


class TimeoutGraceTest(unittest.TestCase):
    """The answer-timer race (field 2026-08-25, twice in one game): a player
    speaks their answer right at the time's-up beeper, the rebound has already
    advanced the turn, and the points go to the NEXT contestant.

    Two guards: (1) a timer that fires while player speech is in flight defers
    instead of stealing the turn; (2) an answer that lands before the rebound
    announcement finishes is graded for the player whose time ran out."""

    _CLUE = {
        "category": "SPACE", "value": 1000, "effective_value": 1000,
        "clue": "Think of a bouquet: Florida + Oregon + Alabama",
        "answer": "floral",
    }

    def setUp(self):
        games._active_game = "jeopardy"
        games._game_state = {
            "phase": "awaiting_answer",
            "current_clue": dict(self._CLUE),
            "players": [
                {"name": "PJ", "score": 0},
                {"name": "Bret", "score": 0},
            ],
            "current_player_idx": 0,
            "current_clue_attempts": [0],
            "board": {
                "remaining": 2,
                "categories": [{"name": "SPACE", "clues": {200: {}}}],
            },
            # State exactly as _jeopardy_timeout_fired leaves it after offering
            # the rebound to Bret (idx 1), announcement still being spoken.
            "awaiting_prompt_delivery": True,
            "timeout_rebound": {"from_idx": 0, "at": 123.0},
        }
        games._game_state["current_player_idx"] = 1

    def tearDown(self):
        games._game_state = {}
        games._active_game = None

    def _handle(self, text):
        with mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_cancel_timeout"), \
             mock.patch.object(games, "_jeopardy_llm_judge", return_value=False), \
             mock.patch("audio.speech_queue.drop_by_tag", return_value=1):
            return games._jeopardy_handle_answer(text, None)

    def test_grace_answer_scores_the_timed_out_player(self):
        resp, done = self._handle("floral")
        self.assertFalse(done)
        self.assertEqual(games._game_state["players"][0]["score"], 1000,
                         "PJ (timed out mid-answer) gets the points")
        self.assertEqual(games._game_state["players"][1]["score"], 0,
                         "Bret (rebound target) must NOT be credited")
        self.assertNotIn("timeout_rebound", games._game_state)

    def test_grace_wrong_answer_deducts_from_the_timed_out_player(self):
        resp, done = self._handle("what is a garden")
        self.assertFalse(done)
        self.assertEqual(games._game_state["players"][0]["score"], -1000)
        self.assertEqual(games._game_state["players"][1]["score"], 0)
        # The clue rebounds onward to Bret, who has not attempted it.
        self.assertIn("Bret", resp)

    def test_after_the_announcement_the_rebound_player_owns_answers(self):
        # _jeopardy_arm_timeout pops both flags once the announcement finishes.
        games._game_state.pop("awaiting_prompt_delivery")
        games._game_state.pop("timeout_rebound")
        self._handle("floral")
        self.assertEqual(games._game_state["players"][1]["score"], 1000,
                         "Bret answered after the full announcement — his points")
        self.assertEqual(games._game_state["players"][0]["score"], 0)

    def test_arm_timeout_closes_the_grace_window(self):
        with mock.patch.object(config, "JEOPARDY_ANSWER_TIMEOUT_SECS", 0):
            games._jeopardy_arm_timeout()
        self.assertNotIn("timeout_rebound", games._game_state)

    def test_timer_defers_while_an_answer_is_in_flight(self):
        games._game_state["answer_timer_token"] = "tok"
        started = []
        with mock.patch.object(games, "_jeopardy_answer_in_flight", return_value=True), \
             mock.patch.object(games.threading, "Timer") as timer_cls:
            timer_cls.return_value.start = lambda: started.append(True)
            games._jeopardy_timeout_fired("tok")
        self.assertTrue(started, "a grace re-arm timer must be started")
        # The turn was NOT stolen: same player, clue still live.
        self.assertEqual(games._game_state["current_player_idx"], 1)
        self.assertTrue(games._game_state.get("current_clue"))
        self.assertEqual(games._game_state.get("answer_timer_token"), "tok")

    def test_timer_fires_normally_when_nobody_is_speaking(self):
        games._game_state["answer_timer_token"] = "tok"
        games._game_state["current_player_idx"] = 0
        games._game_state["current_clue_attempts"] = []
        with mock.patch.object(games, "_jeopardy_answer_in_flight", return_value=False), \
             mock.patch.object(games, "_body_beat"), \
             mock.patch.object(games, "_jeopardy_queue_clip"), \
             mock.patch.object(games, "_jeopardy_schedule_post_timeout_rebound"), \
             mock.patch("audio.speech_queue.enqueue") as enq:
            enq.return_value = mock.MagicMock()
            games._jeopardy_timeout_fired("tok")
        # Rebound offered to the other player, grace window recorded.
        self.assertEqual(games._game_state["current_player_idx"], 1)
        self.assertIn("timeout_rebound", games._game_state)
        self.assertEqual(games._game_state["timeout_rebound"]["from_idx"], 0)


if __name__ == "__main__":
    unittest.main()
