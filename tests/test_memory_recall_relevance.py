"""
Tier D (started) — topic-relevant recall:

  D3: shared symmetric stemming (memory.text_match) so "dogs" matches a "dog" fact.
  D1: episodic + nostalgia callbacks prefer the memory that connects to the live topic
      instead of a random/most-recent one.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import database as db


# ── D3: stemming / overlap ────────────────────────────────────────────────────

class TextMatchTest(unittest.TestCase):
    def test_plural_singular_unify(self):
        from memory import text_match as tm
        self.assertEqual(tm.stem("dogs"), tm.stem("dog"))
        self.assertEqual(tm.stem("robots"), tm.stem("robot"))
        self.assertEqual(tm.stem("hobbies"), tm.stem("hobby"))

    def test_overlap_is_symmetric_across_number(self):
        from memory import text_match as tm
        # topic says "dogs", memory says "dog" → still a hit.
        self.assertGreaterEqual(tm.overlap_count("has a dog named Scout", {"dogs"}), 1)
        self.assertEqual(tm.overlap_count("lives in Sacramento", {"dogs", "robot"}), 0)

    def test_fact_topic_overlap_uses_stemming(self):
        from memory import facts
        fact = {"key": "pet", "value": "a dog named Scout", "category": "pet"}
        self.assertGreaterEqual(facts.fact_topic_overlap(fact, {"dogs"}), 1)


# ── D1: episodic topic relevance ──────────────────────────────────────────────

class EpisodicTopicRelevanceTest(unittest.TestCase):
    def setUp(self):
        from memory import episodes
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "rex.db"
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(self._path))
        self._patch.start()
        episodes.reset_session("run-test")
        # Two experiential memories for person 1: a higher-salience laugh and a game.
        episodes.record_made_laugh(1, "Bret", kind="laugh")
        episodes.record_game_played("Trivia", "scored 4 out of 5", person_id=1, person_name="Bret")

    def tearDown(self):
        from memory import episodes
        self._patch.stop()
        episodes.reset_session(None)
        self._tmp.cleanup()

    def test_without_topic_higher_salience_wins(self):
        from memory import episodic_recall
        top = episodic_recall.person_episodes(1)[0].lower()
        self.assertIn("laugh", top)   # made_laugh (salience 0.7) outranks the game

    def test_topic_match_lifts_the_connected_memory(self):
        from memory import episodic_recall
        top = episodic_recall.person_episodes(1, topic_tokens={"trivia"})[0].lower()
        self.assertIn("trivia", top)   # the on-topic game callback now wins

    def test_unrelated_topic_falls_back_to_salience(self):
        from memory import episodic_recall
        top = episodic_recall.person_episodes(1, topic_tokens={"weather"})[0].lower()
        self.assertIn("laugh", top)   # nothing connects → recency/salience ranking


# ── D1: nostalgia topic relevance ─────────────────────────────────────────────

class NostalgiaTopicRelevanceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        from setup_assets import DB_SCHEMA
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_nostalgia_prefers_topic_connected_conversation(self):
        from memory import conversations as conv_db
        from intelligence import llm
        # Insert oldest→newest; the newest is skipped (it's the 'last conversation').
        conv_db.save_conversation(1, "we talked about his dog Scout", "warm", "pets")
        conv_db.save_conversation(1, "we talked about his robot build", "warm", "robots")
        conv_db.save_conversation(1, "small talk about the weather", "neutral", "weather")
        llm._nostalgia_used_this_session.clear()
        with (
            mock.patch.object(config, "NOSTALGIA_TRIGGER_PROBABILITY", 1.0),
            mock.patch.object(config, "NOSTALGIA_ELIGIBLE_TIERS", ("friend",)),
            mock.patch.object(config, "NOSTALGIA_HISTORY_DEPTH", 10),
        ):
            chosen = llm._pick_nostalgia_callback(1, "friend", topic_tokens={"dogs"})
        self.assertIsNotNone(chosen)
        self.assertIn("dog", (chosen.get("summary") or "").lower())


if __name__ == "__main__":
    unittest.main()
