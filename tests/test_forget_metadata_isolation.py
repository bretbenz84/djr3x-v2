"""
Forget-by-target must match on user CONTENT, never structural metadata columns
(source / category / domain / preference_type / interest_strength). Bug #37: "forget
all explicit memories" matched the `source='explicit'` column and would wipe whole
stores regardless of content. These lock content-matching ON and metadata-matching OFF.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import database as db


class _TempDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Test User')")
        self._p = mock.patch.object(db, "_DB_FILE", self._path)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def _count(self, table):
        with sqlite3.connect(self._path) as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE person_id = 1").fetchone()[0]


class ForgetMetadataIsolationTest(_TempDb):
    def test_forget_explicit_does_not_wipe_explicit_sourced_facts(self):
        from memory import facts, forgetting
        # Two facts, both source='explicit', content unrelated to the word "explicit".
        facts.add_fact(1, "hobby", "pastime", "woodworking", "explicit")
        facts.add_fact(1, "hobby", "sport", "surfing", "explicit")
        self.assertEqual(self._count("person_facts"), 2)

        result = forgetting.forget_specific_memory(1, "forget all explicit memories")

        self.assertEqual(result.deleted["facts"], 0,
                         "'explicit' must not match the source metadata column")
        self.assertEqual(self._count("person_facts"), 2)

    def test_forget_secondhand_does_not_wipe_secondhand_sourced_facts(self):
        from memory import facts, forgetting
        facts.add_fact(1, "other", "name", "Scout", "explicit")
        facts.add_fact(1, "other", "city", "Sacramento", "secondhand")

        result = forgetting.forget_specific_memory(1, "forget anything secondhand")

        self.assertEqual(result.deleted["facts"], 0)
        self.assertEqual(self._count("person_facts"), 2)

    def test_forget_content_word_still_deletes(self):
        from memory import facts, forgetting
        facts.add_fact(1, "hobby", "pastime", "woodworking", "explicit")
        facts.add_fact(1, "hobby", "sport", "surfing", "explicit")

        result = forgetting.forget_specific_memory(1, "forget woodworking")

        self.assertEqual(result.deleted["facts"], 1)
        rows = self._count("person_facts")
        self.assertEqual(rows, 1)

    def test_forget_strength_does_not_match_interest_strength_column(self):
        from memory import forgetting, interests
        interests.upsert_interest(1, "guitar playing", interest_strength="high")
        interests.upsert_interest(1, "jazz music", interest_strength="high")

        # "high" lives only in the interest_strength metadata column now.
        result = forgetting.forget_memory_detail(1, "forget high")

        self.assertEqual(result.deleted["interests"], 0)
        self.assertEqual(self._count("person_interests"), 2)

    def test_forget_interest_name_still_deletes(self):
        from memory import forgetting, interests
        interests.upsert_interest(1, "guitar playing", interest_strength="high")
        interests.upsert_interest(1, "jazz music", interest_strength="medium")

        result = forgetting.forget_memory_detail(1, "forget jazz")

        self.assertEqual(result.deleted["interests"], 1)
        self.assertEqual(self._count("person_interests"), 1)


if __name__ == "__main__":
    unittest.main()
