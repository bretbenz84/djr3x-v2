"""
Cross-run dedupe for proactive topic asks (holiday plans, etc.): a date-bound question
Rex already raised in a PRIOR run must not repeat. Backed by the persistent
proactive_topics_asked table so it survives a restart (the in-memory session sets do not).
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory import database as db
from memory import relationships as rel_memory


class _TempPeopleDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()


class ProactiveAskedPersistenceTest(_TempPeopleDb):
    def test_round_trip_and_idempotent(self):
        key = "holiday_plans:2026-06-19"
        self.assertFalse(rel_memory.was_proactive_asked(1, key))
        rel_memory.mark_proactive_asked(1, key)
        rel_memory.mark_proactive_asked(1, key)  # idempotent — no error, still one row
        self.assertTrue(rel_memory.was_proactive_asked(1, key))

    def test_scoped_by_person_and_key(self):
        rel_memory.mark_proactive_asked(1, "holiday_plans:2026-06-19")
        # Different person and different year (key) are independent.
        self.assertFalse(rel_memory.was_proactive_asked(2, "holiday_plans:2026-06-19"))
        self.assertFalse(rel_memory.was_proactive_asked(1, "holiday_plans:2027-06-19"))

    def test_does_not_pollute_pending_questions(self):
        # The proactive marker must NOT show up as a person_qa "pending question"
        # (that would make Rex think he's waiting on an answer).
        rel_memory.mark_proactive_asked(1, "holiday_plans:2026-06-19")
        self.assertIsNone(rel_memory.get_latest_pending_question(1))

    def test_failure_safe(self):
        # Bad input never raises.
        self.assertFalse(rel_memory.was_proactive_asked(None, ""))
        rel_memory.mark_proactive_asked(None, "")  # no-op, no error


class IdlePlansDedupeUsesPersistenceTest(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        interaction._idle_plans_asked.clear()

    def tearDown(self):
        self.interaction._idle_plans_asked.clear()

    def test_already_asked_when_persistently_recorded(self):
        holiday = {"name": "Juneteenth National Independence Day", "date": "2026-06-19"}
        with mock.patch("memory.relationships.was_proactive_asked", return_value=True):
            self.assertTrue(
                self.interaction._idle_plans_already_asked_holiday(1, holiday)
            )

    def test_not_asked_when_not_recorded(self):
        holiday = {"name": "Some Holiday", "date": "2026-12-25"}
        with mock.patch("memory.relationships.was_proactive_asked", return_value=False):
            # also not in the in-memory session set, and no consciousness dedup
            with mock.patch("intelligence.consciousness._holiday_plans_asked", set()):
                self.assertFalse(
                    self.interaction._idle_plans_already_asked_holiday(1, holiday)
                )


if __name__ == "__main__":
    unittest.main()
