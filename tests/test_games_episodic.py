"""
Tests for games → episodic memory capture (features/games.py).

The game-played episode is best-effort: trivia's score survives to a natural end so
the outcome reads "scored N out of M"; other games clear their state internally, so
the memory is just "I played X". All writes are gated like every episodic write.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from features import games
from memory import episodes, rex_db


class ExtractGameOutcomeTest(unittest.TestCase):
    def test_trivia_score_outcome(self):
        state = {"score": 4, "total_questions": 5, "history": [1, 2, 3, 4]}
        self.assertEqual(games._extract_game_outcome(state), "scored 4 out of 5")

    def test_total_falls_back_to_history_length(self):
        state = {"score": 2, "history": [1, 2, 3]}
        self.assertEqual(games._extract_game_outcome(state), "scored 2 out of 3")

    def test_empty_or_unscored_state_returns_blank(self):
        self.assertEqual(games._extract_game_outcome({}), "")
        self.assertEqual(games._extract_game_outcome({"question_count": 7}), "")
        self.assertEqual(games._extract_game_outcome(None), "")


class GamePlayedGateTest(unittest.TestCase):
    def test_episodic_game_played_is_a_noop_on_default_path(self):
        default = rex_db._default_db_path()
        existed_before = default.exists()
        with mock.patch.object(games, "_jeopardy_person_name", return_value="Bret"):
            games._episodic_game_played("trivia", 3, "scored 4 out of 5")
        self.assertEqual(default.exists(), existed_before)

    def test_blank_game_logs_nothing(self):
        # No game key → no episode (and no person lookup either).
        with mock.patch.object(games, "_jeopardy_person_name") as lookup:
            games._episodic_game_played(None, 3, "")
            lookup.assert_not_called()


class GamePlayedWriteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "rex.db"
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(self._path))
        self._patch.start()
        episodes.reset_session("run-games")

    def tearDown(self):
        self._patch.stop()
        episodes.reset_session(None)
        self._tmp.cleanup()

    def test_played_trivia_with_person_and_outcome(self):
        with mock.patch.object(games, "_jeopardy_person_name", return_value="Bret"):
            games._episodic_game_played("trivia", 3, "scored 4 out of 5")
        row = episodes.recent_episodes(1)[0]
        self.assertEqual(row["kind"], "game_played")
        self.assertEqual(row["summary"], "I played Trivia with Bret — scored 4 out of 5.")
        self.assertEqual(row["person_id"], 3)

    def test_display_name_mapping_for_underscore_keys(self):
        with mock.patch.object(games, "_jeopardy_person_name", return_value=None):
            games._episodic_game_played("word_association", None, "")
        self.assertEqual(episodes.recent_episodes(1)[0]["summary"], "I played Word Association.")


if __name__ == "__main__":
    unittest.main()
