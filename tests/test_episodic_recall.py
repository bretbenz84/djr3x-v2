"""
Tests for memory/episodic_recall.py (Phase-2 read side).

Same safety property as test_episodes.py: never touch the real rex.db. Each test
points REX_DB_PATH at a temp file (opting in to real I/O) and flips
EPISODIC_RECALL_ENABLED on via mock.patch.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import config
from memory import episodes, episodic_recall, rex_db


def _ts(days_ago: float = 0.0) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


class _TempRexDb(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "rex.db"
        self._patch = mock.patch.object(config, "REX_DB_PATH", str(self._path))
        self._patch.start()
        episodes.reset_session("run-current")
        rex_db.ensure_schema()
        self._enable = mock.patch.object(config, "EPISODIC_RECALL_ENABLED", True)
        self._enable.start()

    def tearDown(self):
        self._enable.stop()
        self._patch.stop()
        episodes.reset_session(None)
        self._tmp.cleanup()

    def _insert(self, kind, summary, *, person_id=None, person_name=None,
                salience=0.5, created_at=None, session_id="run-old", detail=None):
        rex_db.execute(
            "INSERT INTO rex_episodes "
            "(created_at, kind, summary, person_id, person_name, detail, salience, session_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (created_at or _ts(1), kind, summary, person_id, person_name,
             json.dumps(detail) if detail else None, salience, session_id),
        )


class GateTest(_TempRexDb):
    def test_disabled_returns_empty(self):
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.9)
        with mock.patch.object(config, "EPISODIC_RECALL_ENABLED", False):
            self.assertIsNone(episodic_recall.session_recap())
            self.assertEqual(episodic_recall.person_episodes(1), [])


class ExclusionTest(_TempRexDb):
    def test_conversation_summary_excluded(self):
        self._insert("conversation_summary", "We talked about robots.", person_id=1, salience=0.8)
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.6)
        recap = episodic_recall.session_recap()
        self.assertIsNotNone(recap)
        self.assertIn("laugh", recap)
        self.assertNotIn("talked about robots", recap)
        # person_episodes also excludes it.
        callbacks = episodic_recall.person_episodes(1)
        self.assertTrue(any("laugh" in c for c in callbacks))
        self.assertFalse(any("robots" in c for c in callbacks))

    def test_scenes_never_surface_individually_only_as_vibe(self):
        self._insert("scene", "When I powered up, I saw: a cluttered messy room with boxes.")
        self._insert("scene", "I looked around the room: cluttered boxes and filament everywhere.")
        recap = episodic_recall.session_recap()
        self.assertIsNotNone(recap)
        # The clustered vibe mentions recurring keywords, not the raw caption prefix.
        self.assertIn("cluttered", recap.lower())
        self.assertNotIn("when i powered up", recap.lower())


class SensitiveExclusionTest(_TempRexDb):
    def test_recap_excludes_sensitive_by_default(self):
        self._insert("emotional_checkin", "I checked in on Bret about a hard thing.",
                     person_id=1, salience=0.95)
        self._insert("animal", "I saw a dog.", salience=0.5)
        recap = episodic_recall.session_recap()  # default: exclude_sensitive=True
        self.assertIsNotNone(recap)
        self.assertIn("dog", recap)
        self.assertNotIn("hard thing", recap)

    def test_recap_can_opt_in_to_sensitive(self):
        self._insert("emotional_checkin", "I checked in on Bret about a hard thing.",
                     person_id=1, salience=0.95)
        recap = episodic_recall.session_recap(exclude_sensitive=False)
        self.assertIsNotNone(recap)
        self.assertIn("hard thing", recap)


class CurrentSessionTest(_TempRexDb):
    def test_recap_excludes_current_run(self):
        # An episode from THIS run must not be "remembered" as a past visit.
        self._insert("made_laugh", "I made someone laugh THIS run.",
                     person_id=1, salience=0.9, session_id="run-current")
        self._insert("animal", "I saw a dog.", salience=0.7, session_id="run-old")
        recap = episodic_recall.session_recap()
        self.assertIsNotNone(recap)
        self.assertIn("dog", recap)
        self.assertNotIn("THIS run", recap)


class RankingTest(_TempRexDb):
    def test_recent_beats_stale_at_equal_salience(self):
        self._insert("made_laugh", "I made Bret laugh recently.",
                     person_id=1, salience=0.7, created_at=_ts(0.1))
        self._insert("made_laugh", "I made Bret laugh long ago.",
                     person_id=1, salience=0.7, created_at=_ts(20))
        ranked = episodic_recall.person_episodes(1, limit=2)
        self.assertEqual(ranked[0], "I made Bret laugh recently")

    def test_kind_weight_orders_at_equal_salience_and_time(self):
        ts = _ts(1)
        self._insert("emotional_checkin", "I checked in on Bret.",
                     person_id=1, salience=0.6, created_at=ts)
        self._insert("birthday_wish", "I wished Bret a happy birthday.",
                     person_id=1, salience=0.6, created_at=ts)
        ranked = episodic_recall.person_episodes(1, limit=2)
        self.assertEqual(ranked[0], "I checked in on Bret")  # higher kind weight


class DedupeTest(_TempRexDb):
    def test_repeated_birthday_wish_collapses(self):
        for d in (1, 2, 3):
            self._insert("birthday_wish", "I wished Bret a happy birthday.",
                         person_id=1, salience=0.65, created_at=_ts(d))
        callbacks = episodic_recall.person_episodes(1, limit=5)
        self.assertEqual(callbacks.count("I wished Bret a happy birthday"), 1)


class PersonEpisodesTest(_TempRexDb):
    def test_excludes_person_seen_and_other_people(self):
        self._insert("person_seen", "I saw Bret.", person_id=1, salience=0.45)
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.6)
        self._insert("game_played", "I played trivia with Jeff.", person_id=2, salience=0.6)
        callbacks = episodic_recall.person_episodes(1)
        self.assertTrue(any("laugh" in c for c in callbacks))
        self.assertFalse(any("saw Bret" in c for c in callbacks))
        self.assertFalse(any("Jeff" in c for c in callbacks))

    def test_non_int_person_returns_empty(self):
        self.assertEqual(episodic_recall.person_episodes(None), [])

    def test_exclude_sensitive_drops_checkins_and_boundaries(self):
        self._insert("emotional_checkin", "I checked in on Bret about a hard thing.",
                     person_id=1, salience=0.95)
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.6)
        kept = episodic_recall.person_episodes(1, exclude_sensitive=True)
        self.assertEqual(kept, ["I made Bret laugh"])
        # Without the flag, the sensitive one is included (and ranks first).
        both = episodic_recall.person_episodes(1, exclude_sensitive=False)
        self.assertTrue(any("hard thing" in c for c in both))


class PersonCallbackHookTest(_TempRexDb):
    """llm._pick_episodic_callback — Phase 2b person-context hook."""

    def setUp(self):
        super().setUp()
        from intelligence import llm
        self.llm = llm
        llm._episodic_callbacks_used_this_session.clear()

    def tearDown(self):
        self.llm._episodic_callbacks_used_this_session.clear()
        super().tearDown()

    def test_disabled_returns_none(self):
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.7)
        with mock.patch.object(config, "EPISODIC_RECALL_ENABLED", False):
            self.assertIsNone(self.llm._pick_episodic_callback(1))

    def test_probability_gate(self):
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.7)
        with mock.patch.object(self.llm.random, "random", return_value=1.0):
            self.assertIsNone(self.llm._pick_episodic_callback(1))  # roll fails
        with mock.patch.object(self.llm.random, "random", return_value=0.0):
            self.assertEqual(self.llm._pick_episodic_callback(1), "I made Bret laugh")

    def test_sensitive_excluded_from_hook(self):
        self._insert("emotional_checkin", "I checked in on Bret about a hard thing.",
                     person_id=1, salience=0.95)
        with mock.patch.object(self.llm.random, "random", return_value=0.0):
            self.assertIsNone(self.llm._pick_episodic_callback(1))

    def test_session_dedupe(self):
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.7)
        with mock.patch.object(self.llm.random, "random", return_value=0.0):
            first = self.llm._pick_episodic_callback(1)
            second = self.llm._pick_episodic_callback(1)
        self.assertEqual(first, "I made Bret laugh")
        self.assertIsNone(second)  # only one row, already surfaced this session


class PruneTest(_TempRexDb):
    def test_prune_caps_scenes_keeps_newest(self):
        for d in range(6):
            self._insert("scene", f"scene number {d}", created_at=_ts(d))
        self._insert("made_laugh", "I made Bret laugh.", person_id=1, salience=0.6)
        with mock.patch.object(config, "EPISODIC_RECALL_SCENE_RETENTION", 3):
            deleted = episodic_recall.prune()
        self.assertEqual(deleted, 3)
        rows = rex_db.fetchall("SELECT kind FROM rex_episodes")
        kinds = [r["kind"] for r in rows]
        self.assertEqual(kinds.count("scene"), 3)
        self.assertEqual(kinds.count("made_laugh"), 1)  # non-scene untouched


if __name__ == "__main__":
    unittest.main()
