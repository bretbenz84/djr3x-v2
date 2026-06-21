"""
Tests for the Phase A+B memory-hardening changes:

  * session-end learning suppression (transcript `learnable` flag),
  * write-time dedup for interests/events + the junk-fragment gate,
  * the one-time duplicate-consolidation data migration (PRAGMA user_version gated),
  * the migration-time orphan sweep, and
  * complete cross-table / cross-DB cleanup on delete / merge / delete-all.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import database as db


def _build_people_db(path: Path, people=((1, "Bret Benziger"),)) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        for pid, name in people:
            conn.execute("INSERT INTO people (id, name) VALUES (?, ?)", (pid, name))


class _PeopleDbTestCase(unittest.TestCase):
    """Base: a temp people.db with database._DB_FILE patched to it."""

    PEOPLE = ((1, "Bret Benziger"),)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        _build_people_db(self._path, self.PEOPLE)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _rows(self, sql, params=()):
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Session-end suppression — the highest-impact bug
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptLearnableTest(unittest.TestCase):
    def setUp(self):
        from memory import conversations
        self.conv = conversations
        self.conv.clear_transcript()

    def tearDown(self):
        self.conv.clear_transcript()

    def test_added_turns_default_learnable(self):
        self.conv.add_to_transcript("Bret", "I like jazz")
        entry = self.conv.get_session_transcript()[0]
        self.assertTrue(entry["learnable"])

    def test_mark_flips_only_latest_human_turn(self):
        self.conv.add_to_transcript("Bret", "first real fact")
        self.conv.add_to_transcript("Rex", "neat")
        self.conv.add_to_transcript("Bret", "China")  # e.g. a game answer
        self.assertTrue(self.conv.mark_last_human_turn_unlearnable())
        t = self.conv.get_session_transcript()
        self.assertTrue(t[0]["learnable"])      # earlier real turn preserved
        self.assertTrue(t[1]["learnable"])       # Rex turn untouched
        self.assertFalse(t[2]["learnable"])      # the suppressed turn flipped

    def test_mark_skips_rex_turns(self):
        self.conv.add_to_transcript("Bret", "my dog is Scout")
        self.conv.add_to_transcript("Rex", "good boy")
        self.conv.mark_last_human_turn_unlearnable()
        t = self.conv.get_session_transcript()
        self.assertFalse(t[0]["learnable"])      # the human turn, not Rex's
        self.assertTrue(t[1]["learnable"])

    def test_mark_on_empty_is_noop(self):
        self.assertFalse(self.conv.mark_last_human_turn_unlearnable())

    def test_consolidation_filter_excludes_unlearnable(self):
        self.conv.add_to_transcript("Bret", "real fact about me")
        self.conv.add_to_transcript("Bret", "game answer", learnable=False)
        learnable = [t for t in self.conv.get_session_transcript() if t.get("learnable", True)]
        texts = [t["text"] for t in learnable]
        self.assertIn("real fact about me", texts)
        self.assertNotIn("game answer", texts)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Write-time dedup for interests
# ─────────────────────────────────────────────────────────────────────────────

class InterestDedupTest(_PeopleDbTestCase):
    def test_paraphrase_folds_into_one_row(self):
        from memory import interests
        interests.upsert_interest(1, "R3X droid", interest_strength="high")
        interests.upsert_interest(1, "building an R3X droid", interest_strength="medium")
        interests.upsert_interest(1, "R3X Droid", interest_strength="low")  # case variant
        rows = self._rows("SELECT * FROM person_interests WHERE person_id = 1")
        self.assertEqual(len(rows), 1)
        # Strength is kept at the strongest mention.
        self.assertEqual(rows[0]["interest_strength"], "high")

    def test_distinct_interests_stay_separate(self):
        from memory import interests
        interests.upsert_interest(1, "robotics")
        interests.upsert_interest(1, "camping")
        rows = self._rows("SELECT name FROM person_interests WHERE person_id = 1")
        self.assertEqual(len(rows), 2)

    def test_junk_fragment_is_rejected(self):
        from memory import interests
        self.assertIsNone(interests.upsert_interest(1, "him sassy"))
        rows = self._rows("SELECT name FROM person_interests WHERE person_id = 1")
        self.assertEqual(rows, [])

    def test_possessive_noun_is_kept(self):
        from memory import interests
        self.assertIsNotNone(interests.upsert_interest(1, "my robots"))
        rows = self._rows("SELECT name FROM person_interests WHERE person_id = 1")
        self.assertEqual(len(rows), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Write-time dedup for events
# ─────────────────────────────────────────────────────────────────────────────

class EventDedupTest(_PeopleDbTestCase):
    def test_repeated_event_folds_into_one_open_row(self):
        from memory import events
        events.add_event(1, "camping trip", None, "planning it")
        events.add_event(1, "camping trip", None, "still planning")
        events.add_event(1, "the camping trip", "2026-07-04", "now dated")
        rows = self._rows("SELECT * FROM person_events WHERE person_id = 1")
        self.assertEqual(len(rows), 1)
        # The later mention supplied a date — it should be folded in.
        self.assertEqual(rows[0]["event_date"], "2026-07-04")

    def test_distinct_events_stay_separate(self):
        from memory import events
        events.add_event(1, "camping trip", None, "")
        events.add_event(1, "dentist appointment", None, "")
        rows = self._rows("SELECT event_name FROM person_events WHERE person_id = 1")
        self.assertEqual(len(rows), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. One-time duplicate consolidation (user_version gated)
# ─────────────────────────────────────────────────────────────────────────────

class ConsolidateExistingDuplicatesTest(_PeopleDbTestCase):
    def _seed_duplicate_interests(self):
        with sqlite3.connect(self._path) as conn:
            for name, strength in (
                ("R3X droid", "high"),
                ("building an R3X droid", "medium"),
                ("Droid", "low"),  # token subset of nothing multi → stays unless equal
                ("camping", "medium"),
            ):
                conn.execute(
                    "INSERT INTO person_interests (person_id, name, interest_strength, confidence) "
                    "VALUES (1, ?, ?, 0.9)",
                    (name, strength),
                )

    def test_consolidate_all_collapses_paraphrases(self):
        from memory import dedup
        self._seed_duplicate_interests()
        summary = dedup.consolidate_all()
        self.assertGreaterEqual(summary["interests_removed"], 1)
        names = {r["name"] for r in self._rows("SELECT name FROM person_interests WHERE person_id=1")}
        # The two R3X paraphrases collapsed; camping and the lone "Droid" survive.
        self.assertIn("camping", names)
        # Only ONE R3X-droid row remains.
        r3x = [n for n in names if "r3x" in n.lower()]
        self.assertEqual(len(r3x), 1)

    def test_one_time_migration_is_gated_and_idempotent(self):
        self._seed_duplicate_interests()
        # user_version starts at 0 → migration runs.
        db.verify_schema()
        with sqlite3.connect(self._path) as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        self.assertEqual(version, db._DATA_MIGRATION_VERSION)
        count_after = len(self._rows("SELECT id FROM person_interests WHERE person_id=1"))

        # A NEW duplicate added now must NOT be collapsed by a second verify_schema
        # (the pass is gated by user_version, so it runs once).
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO person_interests (person_id, name, interest_strength, confidence) "
                "VALUES (1, 'building an R3X droid again', 'low', 0.9)"
            )
        db.verify_schema()
        count_after2 = len(self._rows("SELECT id FROM person_interests WHERE person_id=1"))
        self.assertEqual(count_after2, count_after + 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Orphan sweep
# ─────────────────────────────────────────────────────────────────────────────

class OrphanSweepTest(_PeopleDbTestCase):
    def test_migration_removes_orphan_child_rows(self):
        with sqlite3.connect(self._path) as conn:
            # A fact for a person who does not exist (id 42) — the live-DB orphan class.
            conn.execute(
                "INSERT INTO person_facts (person_id, category, key, value) "
                "VALUES (42, 'other', 'name', 'ghost')"
            )
            # A legitimate fact for the real person 1.
            conn.execute(
                "INSERT INTO person_facts (person_id, category, key, value) "
                "VALUES (1, 'other', 'city', 'Sacramento')"
            )
            # A NULL-person voice signature must be PRESERVED (unnamed voice).
            conn.execute(
                "INSERT INTO voice_signatures (embedding, turns, person_id, created_at) "
                "VALUES (X'00', 1, NULL, '2026-01-01')"
            )
        db._run_migrations()
        facts = self._rows("SELECT person_id FROM person_facts")
        self.assertEqual([f["person_id"] for f in facts], [1])  # orphan swept, real kept
        sigs = self._rows("SELECT person_id FROM voice_signatures")
        self.assertEqual(len(sigs), 1)  # NULL-person signature preserved


# ─────────────────────────────────────────────────────────────────────────────
# 6. Complete cross-table / cross-DB cleanup
# ─────────────────────────────────────────────────────────────────────────────

class CrossTableDeleteTest(_PeopleDbTestCase):
    PEOPLE = ((1, "Bret Benziger"), (2, "Wade Odom"))

    def _seed_extras(self, person_id):
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO voice_signatures (embedding, turns, person_id, created_at) "
                "VALUES (X'01', 3, ?, '2026-01-01')",
                (person_id,),
            )
            conn.execute(
                "INSERT INTO proactive_topics_asked (person_id, topic_key, asked_at) "
                "VALUES (?, 'holiday:2026', '2026-01-01')",
                (person_id,),
            )

    def test_delete_person_clears_extra_tables(self):
        from memory import people
        self._seed_extras(2)
        people.delete_person(2)
        self.assertEqual(self._rows("SELECT * FROM voice_signatures WHERE person_id=2"), [])
        self.assertEqual(self._rows("SELECT * FROM proactive_topics_asked WHERE person_id=2"), [])

    def test_merge_person_repoints_extra_tables(self):
        from memory import people
        self._seed_extras(2)
        self.assertTrue(people.merge_person(1, 2))
        self.assertEqual(self._rows("SELECT * FROM voice_signatures WHERE person_id=2"), [])
        moved = self._rows("SELECT * FROM voice_signatures WHERE person_id=1")
        self.assertEqual(len(moved), 1)

    def test_delete_all_preserves_anonymous_voice(self):
        from memory import people
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO voice_signatures (embedding, turns, person_id, created_at) "
                "VALUES (X'02', 1, NULL, '2026-01-01')"
            )
        self._seed_extras(1)
        people.delete_all_people()
        self.assertEqual(self._rows("SELECT * FROM people"), [])
        sigs = self._rows("SELECT person_id FROM voice_signatures")
        self.assertEqual(sigs, [{"person_id": None}])  # anonymous voice survived


class CrossDbEpisodePurgeTest(_PeopleDbTestCase):
    """delete_person/merge/forget reach Rex's diary (rex.db) when it's a real temp DB."""

    def setUp(self):
        super().setUp()
        from memory import episodes
        self._rex_tmp = tempfile.TemporaryDirectory()
        self._rex_path = Path(self._rex_tmp.name) / "rex.db"
        self._rex_patch = mock.patch.object(config, "REX_DB_PATH", str(self._rex_path))
        self._rex_patch.start()
        episodes.reset_session("run-test")

    def tearDown(self):
        from memory import episodes
        self._rex_patch.stop()
        episodes.reset_session(None)
        self._rex_tmp.cleanup()
        super().tearDown()

    def test_delete_person_purges_their_episodes(self):
        from memory import episodes, people
        episodes.record_person_enrolled(1, "Bret")
        episodes.record_made_laugh(1, "Bret", kind="laugh")
        self.assertEqual(episodes.count(), 2)
        people.delete_person(1)
        self.assertEqual(episodes.count(), 0)

    def test_forget_matching_removes_diary_entries(self):
        from memory import episodes
        episodes.record_episode("made_laugh", "I made Bret laugh about Scout the dog",
                                person_id=1, person_name="Bret")
        episodes.record_episode("made_laugh", "I made Bret laugh about robots",
                                person_id=1, person_name="Bret")
        removed = episodes.forget_matching(1, {"scout"})
        self.assertEqual(removed, 1)
        self.assertEqual(episodes.count(), 1)


if __name__ == "__main__":
    unittest.main()
