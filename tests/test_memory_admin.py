"""
memory.admin — the data layer behind the Memory Banks editor GUI. Covers CRUD on Rex's
own memories (rex.db) and on people + their facts/interests/preferences (people.db).
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import admin
from memory import database as db
from memory import rex_db


class _TempDbs(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self._people_path = base / "people.db"
        with sqlite3.connect(self._people_path) as conn:
            conn.executescript(DB_SCHEMA)
        self._rex_path = base / "rex.db"

        self._p_people = mock.patch.object(db, "_DB_FILE", self._people_path)
        self._p_rex = mock.patch.object(config, "REX_DB_PATH", str(self._rex_path))
        self._p_people.start()
        self._p_rex.start()
        rex_db.ensure_schema()

    def tearDown(self):
        self._p_rex.stop()
        self._p_people.stop()
        self._tmp.cleanup()

    def _insert_rex(self, summary="I saw a dog.", kind="animal", salience=0.5):
        return rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, salience, session_id) "
            "VALUES (?,?,?,?,?)",
            ("2026-06-14 10:00:00", kind, summary, salience, "run-1"),
        )


class RexMemoryAdminTest(_TempDbs):
    def test_list_edit_delete(self):
        mid = self._insert_rex("I made Bret laugh.", "made_laugh", 0.6)
        mems = admin.list_rex_memories()
        self.assertEqual(len(mems), 1)
        self.assertEqual(mems[0]["summary"], "I made Bret laugh.")

        admin.update_rex_memory(mid, summary="I made Bret laugh hard.", salience=0.9)
        m = admin.list_rex_memories()[0]
        self.assertEqual(m["summary"], "I made Bret laugh hard.")
        self.assertAlmostEqual(m["salience"], 0.9, places=3)

        admin.delete_rex_memory(mid)
        self.assertEqual(admin.list_rex_memories(), [])

    def test_salience_is_clamped(self):
        mid = self._insert_rex()
        admin.update_rex_memory(mid, salience=5.0)
        self.assertLessEqual(admin.list_rex_memories()[0]["salience"], 1.0)


class PersonAdminTest(_TempDbs):
    def test_create_list_delete_person(self):
        pid = admin.create_person("Jordan Vega")
        self.assertIsNotNone(pid)
        people = admin.list_people()
        self.assertEqual([p["name"] for p in people], ["Jordan Vega"])

        self.assertTrue(admin.delete_person(pid))
        self.assertEqual(admin.list_people(), [])

    def test_create_rejects_junk_name(self):
        self.assertIsNone(admin.create_person("   "))

    def test_update_person_fields(self):
        pid = admin.create_person("Jordan Vega")
        admin.update_person_fields(pid, name="Jordan V", nickname="Jo", hair_color="black")
        detail = admin.get_person_detail(pid)
        self.assertEqual(detail["person"]["name"], "Jordan V")
        self.assertEqual(detail["person"]["nickname"], "Jo")
        self.assertEqual(detail["person"]["hair_color"], "black")

    def test_non_whitelisted_field_is_ignored(self):
        pid = admin.create_person("Jordan Vega")
        # warmth_score is NOT directly editable here; must be untouched.
        admin.update_person_fields(pid, warmth_score=0.99)
        self.assertEqual(float(admin.get_person_detail(pid)["person"]["warmth_score"]), 0.0)


class FactAdminTest(_TempDbs):
    def setUp(self):
        super().setUp()
        self.pid = admin.create_person("Jordan Vega")

    def test_add_edit_delete_fact(self):
        self.assertTrue(admin.add_person_fact(self.pid, "preference", "favorite_color", "teal"))
        facts = admin.get_person_detail(self.pid)["facts"]
        self.assertEqual(len(facts), 1)
        fid = facts[0]["id"]
        self.assertEqual(facts[0]["value"], "teal")

        admin.update_fact(fid, value="cyan", importance=0.8)
        facts = admin.get_person_detail(self.pid)["facts"]
        self.assertEqual(facts[0]["value"], "cyan")
        self.assertAlmostEqual(float(facts[0]["importance"]), 0.8, places=3)

        admin.delete_fact(fid)
        self.assertEqual(admin.get_person_detail(self.pid)["facts"], [])

    def test_add_fact_requires_key_and_value(self):
        self.assertFalse(admin.add_person_fact(self.pid, "other", "", "x"))
        self.assertFalse(admin.add_person_fact(self.pid, "other", "k", ""))

    def test_delete_person_removes_their_facts(self):
        admin.add_person_fact(self.pid, "other", "k", "v")
        admin.delete_person(self.pid)
        # facts table no longer has rows for the deleted person
        rows = db.fetchall("SELECT * FROM person_facts WHERE person_id = ?", (self.pid,))
        self.assertEqual(list(rows), [])


if __name__ == "__main__":
    unittest.main()
