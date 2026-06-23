"""
A postponed event must NOT be durably lost (#15). It stays OPEN (re-dated / undated)
so Rex keeps anticipating it, instead of being marked canceled like a real cancellation.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class PostponeKeepsEventOpenTest(_TempDb):
    def test_postpone_keeps_event_open_and_stops_reprompting(self):
        from memory import events
        eid = events.add_event(1, "camping trip", "2020-01-01", "with the crew")  # past date
        self.assertIsNotNone(eid)
        # A past-dated open event is an overdue follow-up.
        self.assertIn(eid, [e["id"] for e in events.get_pending_followups(1)])

        rescheduled = events.postpone_matching_events(1, "we postponed the camping trip")
        self.assertEqual(len(rescheduled), 1)

        # Still OPEN (not canceled), and no longer an overdue follow-up (date cleared,
        # mentioned_at refreshed) so Rex doesn't immediately re-ask.
        self.assertIn(eid, [e["id"] for e in events.get_open_events(1)])
        self.assertNotIn(eid, [e["id"] for e in events.get_pending_followups(1)])

    def test_cancel_still_closes_the_event(self):
        from memory import events
        eid = events.add_event(1, "dentist", "2020-01-01", "")
        canceled = events.cancel_matching_events(1, "I canceled the dentist appointment")
        self.assertEqual(len(canceled), 1)
        self.assertNotIn(eid, [e["id"] for e in events.get_open_events(1)])

    def test_reschedule_with_known_date_sets_it(self):
        from memory import events
        eid = events.add_event(1, "concert", "2020-01-01", "")
        events.reschedule_event(eid, "2030-12-31")
        upcoming = events.get_upcoming_events(1)
        self.assertTrue(
            any(e["id"] == eid and e["event_date"] == "2030-12-31" for e in upcoming)
        )


if __name__ == "__main__":
    unittest.main()
