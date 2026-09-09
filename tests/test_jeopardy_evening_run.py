"""Annotated 20:58 run: voice labels cannot own scores; spoken words survive."""
import copy
import unittest
from unittest import mock

import numpy as np
import config
from audio import transcription as tr
from features import games, jeopardy, spoken_answers
from intelligence import interaction as I


class EveningRunTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.object(games, "_active_game", "jeopardy"))
        self.enterContext(mock.patch.object(games, "_game_state", {}))
        for name in ("_body_beat", "_jeopardy_queue_clip", "_jeopardy_cancel_timeout"):
            self.enterContext(mock.patch.object(games, name))
        self.judge = self.enterContext(mock.patch.object(games, "_quick_call", return_value="no"))
        self.enterContext(mock.patch.object(config, "JEOPARDY_LLM_JUDGE_ENABLED", True))
        self.enterContext(mock.patch.object(config, "GUI_ENABLED", True))
        self.enterContext(mock.patch.object(config, "JEOPARDY_MAX_REBOUNDS", 1))
        self.clue("frets", 1000)

    def clue(self, answer, value=200, idx=1):
        games._game_state.clear()
        games._game_state.update({
            "phase": "awaiting_answer", "current_player_idx": idx,
            "players": [{"name": "Bret", "person_id": 1, "score": 0},
                        {"name": "PJ", "person_id": 7, "score": 0},
                        {"name": "T'Joy", "person_id": 3, "score": 0}],
            "current_clue": {"answer": answer, "value": value, "clue": "The live clue", "category": "TEST"},
            "current_clue_attempts": [],
            "board": {"remaining": 4, "categories": []},
        })

    def test_pj_mislabeled_bret_is_scored_and_turn_advances(self):
        line = games.handle_input("What are struts?", person_id=1)
        self.assertIn("$1000 off PJ", line)
        self.assertNotIn("not your", line)
        self.assertNotIn("No charge", line)
        self.assertEqual(games._game_state["players"][1]["score"], -1000)
        self.assertEqual(games._game_state["players"][0]["score"], 0)
        self.assertEqual(games._game_state["current_player_idx"], 2)
        # Another answer with the same wrong voice label settles the rebound.
        games.handle_input("What are struts?", person_id=1)
        self.assertEqual(games._game_state["phase"], "selecting")

    def test_any_voice_or_unknown_can_answer_the_open_turn(self):
        for pid in (None, 1, 3, 7, 999):
            with self.subTest(pid=pid):
                self.clue("frets")
                line = games.handle_input("What are frets?", person_id=pid)
                self.assertIn("$200 to PJ", line)
                self.assertEqual(games._game_state["players"][1]["score"], 200)

    def test_symbol_is_accepted_as_cymbals_without_model_call(self):
        self.clue("cymbals", 600, idx=0)
        line = games.handle_input("What is the symbol?", person_id=7)
        self.assertIn("$600 to Bret", line)
        self.judge.assert_not_called()

    def test_flag_flood_asks_repeat_then_accepts_clear_answer(self):
        self.clue("flood", idx=0)
        line = games.handle_input("What is the flag?", person_id=3)
        self.assertIn("Say your answer again", line)
        self.assertNotIn("flood", line.lower())  # don't give away the answer
        self.assertEqual(games._game_state["players"][0]["score"], 0)
        self.assertEqual(games._game_state["current_player_idx"], 0)
        line = games.handle_input("What is the flood?", person_id=None)
        self.assertIn("$200 to Bret", line)

    def test_shushing_does_not_consume_a_turn(self):
        before = copy.deepcopy(games._game_state)
        self.assertEqual(games.handle_input("Shh."), "")
        self.assertEqual(before, games._game_state)
        self.judge.assert_not_called()

    def test_setup_needs_no_voiceprints(self):
        from memory import people
        from audio import speaker_id
        with mock.patch.object(games, "_jeopardy_find_or_create_player", side_effect=[(3,"T'Joy"), (1,"Bret"), (7,"PJ")]), \
             mock.patch.object(people, "has_voice_biometric", return_value=False) as has_voice, \
             mock.patch.object(speaker_id, "enroll_voice") as enroll, \
             mock.patch.object(jeopardy, "build_board", return_value={"remaining": 4, "categories": []}):
            line, done = games._jeopardy_begin_board(["T Joy", "Bret", "P J"], None)
        self.assertFalse(done)
        self.assertEqual(games._game_state["phase"], "selecting")
        self.assertNotIn("voice print", line)
        has_voice.assert_not_called()
        enroll.assert_not_called()

    def test_game_speech_cannot_rename_or_enroll(self):
        from audio import speaker_id
        from intelligence.voice_learning_runtime import Runtime
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(speaker_id, "active_backend", return_value="campplus"), \
             mock.patch.object(speaker_id, "enroll_voice") as enroll, \
             mock.patch.object(I.people_memory, "rename_person") as rename:
            runtime = Runtime(I)
            self.assertIsNone(runtime.process(audio, "My name is Joy"))
            self.assertIsNone(runtime.learner.pending)
            self.assertFalse(I._begin_conversational_voice_learning(3,audio,source="new_person",confirmed=True))
            self.assertIsNone(I._handle_name_update_request("My name is Joy", 3, "T'Joy Jackson"))
        enroll.assert_not_called()
        rename.assert_not_called()

    def test_et_passes_actual_asr_and_interaction_to_score(self):
        self.clue("E.T.", idx=0)
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(config, "TRANSCRIPTION_BACKEND", "qwen3"), \
             mock.patch.object(tr, "_qwen_ready", return_value=True), \
             mock.patch.object(tr, "_QWEN_LOAD_FAILED", False), \
             mock.patch.object(tr, "_qwen_transcribe", return_value=("Et.", -0.5)), \
             mock.patch.object(I, "_shutdown_requested", return_value=False), \
             mock.patch.object(I, "_speak_blocking", return_value=True) as speak, \
             mock.patch.object(I.conv_memory, "_log_turn"), \
             mock.patch.object(I.conv_log, "log_heard"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch.object(games, "on_response_spoken"), \
             mock.patch.object(games, "consume_pending_audio_after_response", return_value=None):
            transcript = tr.transcribe(audio)
            self.assertEqual(str(transcript), "Et.")
            self.assertFalse(transcript.confident)  # accepted, never upgraded for learning
            I._handle_speech_segment(audio, text_input=True, transcribed_text=transcript,
                                    raw_best_id_override=7, raw_best_name_override="PJ", speaker_score_override=1.)
        self.assertIn("$200 to Bret", speak.call_args.args[0])
        self.assertEqual(games._game_state["players"][0]["score"], 200)


class SpokenMatchesTest(unittest.TestCase):
    def test_homophones_and_spelling_variants(self):
        for heard, answer in (("symbol", "cymbals"), ("symbols", "cymbal"),
                              ("knight", "night"), ("ore", "oar"),
                              ("Doctor Doolittle", "Dr. Dolittle"), ("Et", "E.T.")):
            self.assertTrue(jeopardy.is_correct(heard, answer), (heard, answer))

    def test_distinct_answers_and_missing_parts_stay_distinct(self):
        for heard, answer in (("struts", "frets"), ("flag", "flood"), ("Boston", "Albany"),
                              ("Kansas", "Arkansas"), ("1493", "1492"),
                              ("license", "license and registration")):
            self.assertFalse(jeopardy.is_correct(heard, answer), (heard, answer))

    def test_near_sound_requires_repeat_not_auto_credit(self):
        self.assertTrue(spoken_answers.possible_mishearing("flag", "flood"))
        self.assertFalse(spoken_answers.possible_mishearing("struts", "frets"))
        self.assertFalse(spoken_answers.possible_mishearing("license", "license and registration"))

    def test_game_short_answer_exception_keeps_noise_filters(self):
        for text in ("E.T.", "Et.", "Pi", "Q", "7"):
            self.assertFalse(tr._is_hallucination(text, allow_short_answer=True), text)
        for text in ("...", "um", "Thanks for watching", "www.example.com", "zzzzzzzzzzzzzzzzzzzz"):
            self.assertTrue(tr._is_hallucination(text, allow_short_answer=True), text)
        self.assertTrue(tr._is_hallucination("Et."))


if __name__ == "__main__":
    unittest.main()
