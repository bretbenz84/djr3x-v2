"""
tests/test_open_threads.py — cross-session open-thread follow-ups: freshness
window, once-ever spending (persisted), age phrasing. Temp rex.db, no LLM.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from memory import rex_db


def _iso(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


class OpenThreadsTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = getattr(config, "REX_DB_PATH", None)
        config.REX_DB_PATH = str(Path(self._tmp.name) / "rex.db")
        rex_db.ensure_schema()
        from intelligence import open_threads
        self.ot = open_threads

    def tearDown(self):
        config.REX_DB_PATH = self._orig
        self._tmp.cleanup()

    def _episode(self, *, person_id=1, age_hours=24.0, threads=None, asked=None):
        detail = {}
        if threads is not None:
            detail["open_threads"] = threads
        if asked is not None:
            detail["threads_asked"] = asked
        return rex_db.execute(
            "INSERT INTO rex_episodes "
            "(created_at, kind, summary, person_id, person_name, detail, salience, session_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (_iso(datetime.now() - timedelta(hours=age_hours)), "conversation_summary",
             "Bret told me things.", person_id, "Bret",
             json.dumps(detail), 0.6, "run-test"),
        )

    def test_pending_returns_fresh_threads(self):
        self._episode(age_hours=24, threads=["whether the motor swap happened"])
        got = self.ot.pending_for_person(1)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["thread"], "whether the motor swap happened")

    def test_too_fresh_and_too_old_excluded(self):
        self._episode(age_hours=1, threads=["too fresh"])              # < 6h
        self._episode(age_hours=30 * 24, threads=["too old"])          # > 21d
        self.assertEqual(self.ot.pending_for_person(1), [])

    def test_other_person_excluded(self):
        self._episode(person_id=2, age_hours=24, threads=["their thing"])
        self.assertEqual(self.ot.pending_for_person(1), [])

    def test_mark_asked_is_permanent(self):
        ep = self._episode(age_hours=24, threads=["thread A", "thread B"])
        got = self.ot.pending_for_person(1)
        self.assertEqual(len(got), 2)
        self.ot.mark_asked(ep, "thread A")
        remaining = self.ot.pending_for_person(1)
        self.assertEqual([t["thread"] for t in remaining], ["thread B"])
        # Persisted in the row itself (survives restart).
        row = rex_db.fetchone("SELECT detail FROM rex_episodes WHERE id = ?", (ep,))
        self.assertIn("thread A", json.loads(row["detail"])["threads_asked"])

    def test_pre_asked_threads_skipped(self):
        self._episode(age_hours=24, threads=["done", "open"], asked=["done"])
        self.assertEqual([t["thread"] for t in self.ot.pending_for_person(1)], ["open"])

    def test_describe_age(self):
        self.assertEqual(self.ot.describe_age(0.4), "earlier today")
        self.assertEqual(self.ot.describe_age(1.4), "yesterday")
        self.assertEqual(self.ot.describe_age(4.0), "4 days ago")
        self.assertEqual(self.ot.describe_age(15.0), "a while back")


if __name__ == "__main__":
    unittest.main()


class ConsolidationTest(unittest.TestCase):
    """memory/consolidation.py retention rules on a temp rex.db."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = getattr(config, "REX_DB_PATH", None)
        config.REX_DB_PATH = str(Path(self._tmp.name) / "rex.db")
        rex_db.ensure_schema()
        from memory import consolidation
        self.con = consolidation

    def tearDown(self):
        config.REX_DB_PATH = self._orig
        self._tmp.cleanup()

    def _seen(self, person_id, age_days, hour):
        ts = (datetime.now() - timedelta(days=age_days)).replace(hour=hour, minute=0, second=0)
        rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, person_id, salience, session_id) "
            "VALUES (?,?,?,?,?,?)",
            (_iso(ts), "person_seen", "I saw Bret.", person_id, 0.45, "t"),
        )

    def test_person_seen_dedup_and_ageout(self):
        self._seen(1, 0, 9)
        self._seen(1, 0, 15)        # same day: dedup to newest
        self._seen(1, 2, 12)        # different day: kept
        self._seen(1, 60, 12)       # past retention: deleted
        self.con.run()
        rows = rex_db.fetchall("SELECT created_at FROM rex_episodes WHERE kind='person_seen'")
        self.assertEqual(len(rows), 2)

    def test_stale_pending_room_question_dismissed(self):
        old = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        rex_db.execute(
            "INSERT INTO room_objects (label, first_seen, last_seen, sighting_count, ask_status) "
            "VALUES ('guitar', ?, ?, 5, 'pending')", (old, old))
        fresh = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rex_db.execute(
            "INSERT INTO room_objects (label, first_seen, last_seen, sighting_count, ask_status) "
            "VALUES ('ladder', ?, ?, 5, 'pending')", (fresh, fresh))
        self.con.run()
        rows = {r["label"]: r["ask_status"] for r in rex_db.fetchall(
            "SELECT label, ask_status FROM room_objects")}
        self.assertEqual(rows["guitar"], "dismissed")
        self.assertEqual(rows["ladder"], "pending")
