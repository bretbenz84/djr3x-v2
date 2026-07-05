"""mark_followed_up closes same-date sibling events (memory/events.py).

The extractor stores one outing as several differently-named events ("visit dad" /
"4th of July" / "fireworks" — all the same date), which made Rex ask "how did it go?"
once per duplicate (field log 2026-07-05: three asks in one session, a fourth still
pending next boot). Resolving one dated follow-up must retire the whole outing.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory import database as db
from memory import events


class SiblingCloseTest(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")
            rows = [
                # four names, one outing (same date) — the field shape
                ("visit dad", "2026-07-04", "planned", 0),
                ("4th of July", "2026-07-04", "planned", 0),
                ("fireworks at dad's", "2026-07-04", "planned", 0),
                # different date: must NOT be touched
                ("dentist", "2026-07-09", "planned", 0),
                # same date but already canceled: must NOT be resurrected/overwritten
                ("parade", "2026-07-04", "canceled", 1),
                # undated: must NOT be swept up
                ("camping sometime", None, "planned", 0),
            ]
            for name, date, status, fup in rows:
                conn.execute(
                    "INSERT INTO person_events (person_id, event_name, event_date, "
                    "status, followed_up, mentioned_at) VALUES (1, ?, ?, ?, ?, "
                    "'2026-07-01T00:00:00')",
                    (name, date, status, fup),
                )
        self._patch = mock.patch.object(db, "_DB_FILE", path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _rows(self):
        return {r["event_name"]: dict(r) for r in db.fetchall(
            "SELECT event_name, followed_up, status, outcome FROM person_events"
        )}

    def test_resolving_one_closes_same_date_siblings(self):
        visit = db.fetchone(
            "SELECT id FROM person_events WHERE event_name = 'visit dad'")
        events.mark_followed_up(int(visit["id"]), "I had a good time with my dad")
        rows = self._rows()
        # the answered event carries the real outcome
        self.assertEqual(rows["visit dad"]["followed_up"], 1)
        self.assertEqual(rows["visit dad"]["outcome"], "I had a good time with my dad")
        # same-date planned siblings retired together, marked as the same outing
        for sib in ("4th of July", "fireworks at dad's"):
            self.assertEqual(rows[sib]["followed_up"], 1, sib)
            self.assertEqual(rows[sib]["status"], "completed", sib)
            self.assertIn("same outing", rows[sib]["outcome"] or "", sib)
        # different date, canceled, and undated rows untouched
        self.assertEqual(rows["dentist"]["followed_up"], 0)
        self.assertEqual(rows["parade"]["status"], "canceled")
        self.assertEqual(rows["camping sometime"]["followed_up"], 0)
        # and nothing is pending for that outing anymore
        pending_names = {e["event_name"] for e in events.get_pending_followups(1)}
        self.assertNotIn("4th of July", pending_names)
        self.assertNotIn("fireworks at dad's", pending_names)

    def test_undated_event_resolution_touches_nothing_else(self):
        camp = db.fetchone(
            "SELECT id FROM person_events WHERE event_name = 'camping sometime'")
        events.mark_followed_up(int(camp["id"]), "it was fine")
        rows = self._rows()
        self.assertEqual(rows["camping sometime"]["followed_up"], 1)
        self.assertEqual(rows["visit dad"]["followed_up"], 0)     # dated rows untouched
        self.assertEqual(rows["4th of July"]["followed_up"], 0)


if __name__ == "__main__":
    unittest.main()
