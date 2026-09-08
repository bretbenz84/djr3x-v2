"""September 7 field regressions: routing -> real board -> scoring and audio seams."""
import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import config
from features import games, jeopardy
from intelligence import interaction as I


def board():
    return {"remaining": 4, "categories": [
        {"name": "LET'S PLAY A GAME", "clues": {
            200: {"clue": "Remove pieces from this patient", "answer": "Operation"},
            1000: {"clue": "This game is also called table soccer", "answer": "foosball"},
        }},
        {"name": "TV TIME", "clues": {
            400: {"clue": "SpongeBob and Squidward work at this restaurant", "answer": "the Krusty Krab"},
            600: {"clue": "The network called ABC this", "answer": "Family"},
        }},
    ]}


class GameRunTest(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        stack.enter_context(mock.patch.object(games, "_active_game", "jeopardy"))
        stack.enter_context(mock.patch.object(games, "_game_state", {
            "phase": "selecting", "board": board(), "current_player_idx": 0,
            "players": [{"name": "Bret", "score": 0}, {"name": "PJ", "score": 0}],
        }))
        for method in ("_body_beat", "_jeopardy_queue_clip", "_jeopardy_cancel_timeout"):
            stack.enter_context(mock.patch.object(games, method))
        self.judge = stack.enter_context(mock.patch.object(games, "_quick_call", return_value="no"))
        self.persona = stack.enter_context(mock.patch.object(games, "_rex_respond", side_effect=AssertionError("No invented game state")))
        stack.enter_context(mock.patch.object(config, "GUI_ENABLED", True))
        stack.enter_context(mock.patch.object(config, "JEOPARDY_READ_SELECTION_WITH_GUI", False))
        stack.enter_context(mock.patch.object(config, "JEOPARDY_LLM_JUDGE_ENABLED", True))

    def test_real_interaction_handler_keeps_category_pick_out_of_general_llm(self):
        import numpy as np
        with mock.patch.object(I, "_shutdown_requested", return_value=False), \
             mock.patch.object(I, "_speak_blocking", return_value=True) as speak, \
             mock.patch.object(I, "_handle_name_update_request", return_value=None), \
             mock.patch.object(I.conv_memory, "_log_turn"), \
             mock.patch.object(I.conv_log, "log_heard"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch.object(games, "on_response_spoken"), \
             mock.patch.object(games, "consume_pending_audio_after_response", return_value=None):
            I._handle_speech_segment(
                np.zeros(1, dtype=np.float32), text_input=True,
                transcribed_text="Let's play a game for one thousand dollars.",
                raw_best_id_override=1, raw_best_name_override="Bret", speaker_score_override=1.,
            )
        snap = games.snapshot()
        self.assertEqual(snap["phase"], "awaiting_answer")
        self.assertEqual(snap["current_clue"]["answer"], "foosball")
        self.assertIn(snap["current_clue"]["clue"], speak.call_args.args[0])

    def test_category_regex_collision_reaches_real_clue_and_scores(self):
        for pick in ("Let's play a game for two hundred.",
                     "Let's play a game. Value 200.",
                     "Category let's play a game for 200."):
            with self.subTest(pick=pick):
                games._game_state.update(phase="selecting", board=board())
                before = copy.deepcopy(games._game_state["board"])
                self.assertIsNone(I._game_escape_command(pick))
                self.assertEqual(before, games._game_state["board"])
                line = games.handle_input(pick)
                snap = games.snapshot()
                self.assertEqual(snap["phase"], "awaiting_answer")
                self.assertIn(snap["current_clue"]["clue"], line)
                self.assertEqual(snap["remaining"], 3)
                self.assertNotIn("$200", line)
                score = games._game_state["players"][0]["score"]
                games.handle_input("Operation.")  # question phrasing is optional
                self.assertEqual(games._game_state["players"][0]["score"], score + 200)

    def test_value_then_category_completes_same_pick(self):
        self.assertIn("Which category?", games.handle_input("Two hundred dollars."))
        self.assertIsNone(I._game_escape_command("Let's play a game."))
        line = games.handle_input("Let's play a game.")
        self.assertIn("Remove pieces", line)
        self.assertEqual(games.snapshot()["current_clue"]["value"], 200)

    def test_spent_category_is_not_rerouted_or_switched(self):
        games.handle_input("Let's play a game for 200")
        games.handle_input("Operation")
        before = copy.deepcopy(games._game_state["board"])
        self.assertIsNone(I._game_escape_command("Let's play a game for 200"))
        games.handle_input("Let's play a game for 200")
        self.assertEqual(before, games._game_state["board"])
        self.assertEqual(games.snapshot()["phase"], "selecting")

    def test_explicit_stop_and_switch_still_escape(self):
        for text, key in (("Stop game", "stop_game"), ("Let's play trivia", "start_game"), ("Shut down", "shutdown")):
            with self.subTest(text=text):
                self.assertEqual(I._game_escape_command(text).command_key, key)

    def test_repeated_abc_family_cannot_invent_freeform_or_score_twice(self):
        games.handle_input("TV time for six hundred")
        self.judge.return_value = "yes"
        games.handle_input("What is ABC Family?")
        self.assertEqual(games._game_state["players"][0]["score"], 600)
        self.judge.reset_mock()
        repeat = games.handle_input("What is ABC Family?")
        self.assertIn("already scored", repeat)
        self.assertIn("Family", repeat)
        self.assertNotIn("Freeform", repeat)
        self.assertEqual(games._game_state["players"][0]["score"], 600)
        self.judge.assert_not_called()

    def test_classifier_cannot_speak_an_invented_clue_or_ruling(self):
        self.judge.return_value = "Correct. What is Freeform? Clue: invented."
        before = copy.deepcopy(games._game_state)
        line = games.handle_input("Do you actually know that?")
        self.assertIn("No clue is open", line)
        self.assertNotIn("Freeform", line)
        self.assertEqual(before, games._game_state)

    def test_unclear_name_and_api_failure_get_repeat_without_deduction(self):
        for verdict in ("unclear", "", "garbage"):
            with self.subTest(verdict=verdict):
                games._game_state.update(phase="selecting", board=board())
                games.handle_input("TV time for 400")
                self.judge.return_value = verdict
                line = games.handle_input("The Percy Crab.")
                self.assertIn("Say your answer again", line)
                self.assertEqual(games._game_state["current_player_idx"], 0)
                self.assertEqual(games._game_state["players"][0]["score"], 0)
                self.assertEqual(games.snapshot()["phase"], "awaiting_answer")

    def test_clear_wrong_answer_still_deducts_and_rebounds(self):
        games.handle_input("TV time for 400")
        line = games.handle_input("What is McDonald's?")
        self.assertIn("$400 off Bret", line)
        self.assertNotIn("Alderaan", line)
        self.assertEqual(games._game_state["players"][0]["score"], -400)
        self.assertEqual(games._game_state["current_player_idx"], 1)

    def test_repeated_unclear_answers_are_bounded_and_never_fined(self):
        games.handle_input("TV time for 400")
        self.judge.return_value = "unclear"
        with mock.patch.object(config, "JEOPARDY_UNCLEAR_ANSWER_RETRIES", 1):
            games.handle_input("The Percy Crab")
            games.handle_input("The Percy Crab")
        self.assertEqual(games._game_state["players"][0]["score"], 0)
        self.assertEqual(games._game_state["current_player_idx"], 1)

    def test_gui_repeat_and_rebound_read_clue_without_repeating_selection(self):
        games.handle_input("TV time for 400")
        for line in (games.handle_input("Repeat the clue"), games.handle_input("McDonald's")):
            self.assertIn("SpongeBob", line)
            self.assertNotIn("TV TIME for", line)

    def test_final_judge_failure_retries_same_player_then_leaves_wager_unchanged(self):
        games._game_state.update(phase="final_answer", final_queue=[0], final={
            "clue": {"clue": "SpongeBob's workplace", "answer": "the Krusty Krab"},
            "order": [0], "wagers": {0: 400},
        })
        self.judge.return_value = ""
        with mock.patch.object(config, "JEOPARDY_UNCLEAR_ANSWER_RETRIES", 1):
            line, done = games._jeopardy_handle_final_answer("Percy Crab", None)
            self.assertFalse(done)
            self.assertIn("Say your answer again", line)
            self.assertEqual(games._game_state["final_queue"], [0])
            line, done = games._jeopardy_handle_final_answer("Percy Crab", None)
        self.assertTrue(done)
        self.assertIn("wager is unchanged", line)
        self.assertEqual(games._game_state["players"][0]["score"], 0)

    def test_voice_only_keeps_category_and_value(self):
        with mock.patch.object(config, "GUI_ENABLED", False):
            line = games.handle_input("TV time for 400")
        self.assertIn("TV TIME for $400", line)

    def test_feedback_never_repeats_consecutively_or_uses_alderaan(self):
        for correct in (True, False):
            lines = [games._jeopardy_feedback(correct) for _ in range(30)]
            self.assertTrue(all(a != b for a, b in zip(lines, lines[1:])))
            self.assertNotIn("Alderaan", " ".join(lines))


class AudioHandoffTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(games, "_active_game", "jeopardy"))
        self.enterContext(mock.patch.object(games, "_game_state", {"phase": "awaiting_answer"}))
        self.enterContext(mock.patch.object(I, "_game_barge_floor_at", 0.0))
        self.enterContext(mock.patch.object(I, "_gap_recovery_floor_at", 0.0))
        self.enterContext(mock.patch.object(I, "_listen_capture_floor_at", 1000.0))
        self.enterContext(mock.patch.object(I.hardware_aec, "is_active", return_value=True))

    def test_cancel_callback_race_preserves_words_spoken_under_music(self):
        self.assertTrue(I._pin_game_barge_capture_floor(1002.0))
        # A late callback overwrites the global after cancellation starts.
        I._listen_capture_floor_at = 1003.0
        with mock.patch.object(I, "_speech_preroll_secs", return_value=1.5):
            seconds = I._speech_capture_secs(1002.0, 1004.0)
        self.assertEqual(1004.0 - seconds, 1000.0)

    def test_music_completion_never_restamps_floor_or_flushes_answer(self):
        item = SimpleNamespace(text=None, audio_path="assets/audio/jeopardy/jeopardy-theme.mp3")
        with mock.patch.object(I, "_apply_post_tts_handoff") as handoff, \
             mock.patch.object(I, "_end_response_sequence_for_text") as release, \
             mock.patch.object(I.speech_queue, "is_drained", return_value=True):
            I._arm_post_tts_window(item)
        handoff.assert_not_called()
        release.assert_called_once()
        self.assertEqual(I._listen_capture_floor_at, 1000.0)

    def test_software_audio_keeps_normal_echo_boundary(self):
        item = SimpleNamespace(text=None, audio_path="assets/audio/jeopardy/jeopardy-theme.mp3")
        with mock.patch.object(I.hardware_aec, "is_active", return_value=False), \
             mock.patch.object(I, "_apply_post_tts_handoff") as handoff, \
             mock.patch.object(I.speech_queue, "is_drained", return_value=False):
            self.assertFalse(I._pin_game_barge_capture_floor(1002.0))
            I._arm_post_tts_window(item)
        handoff.assert_called_once()

    def test_declarative_clue_gets_fast_reply_recovery(self):
        self.assertTrue(I._post_tts_handoff_policy("Clue: This game is table soccer.").fast_response_expected)

    def test_spoken_category_after_playback_is_not_text_echo(self):
        games._game_state.update(phase="selecting", board=board())
        with mock.patch.object(I, "_utterance_observations", {"started_at": 1001., "ended_at": 1003.}), \
             mock.patch.object(I.time, "monotonic", return_value=1004.), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at", return_value=1000.), \
             mock.patch.object(I, "_recent_rex_lines", [("lets play a game", 1000.)]):
            self.assertFalse(I._looks_like_own_echo("Lets play a game"))
            I._utterance_observations["started_at"] = 999.
            self.assertTrue(I._looks_like_own_echo("Lets play a game"))


if __name__ == "__main__":
    unittest.main()
