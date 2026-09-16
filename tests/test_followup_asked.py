"""Delivered follow-ups are durable; time alone never consumes an event."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from memory import database as db, events

class FollowupAskedTests(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'people.db'
        with sqlite3.connect(self.path) as c:
            c.executescript(DB_SCHEMA)
            c.execute("INSERT INTO people (id,name) VALUES (1,'Test')")
        self.patch=mock.patch.object(db,'_DB_FILE',self.path)
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_old_unasked_trip_remains_due_without_mutation(self):
        eid=events.add_event(1,'trip to Georgia','2020-01-01','with dad')
        for _ in range(2):
            self.assertIn(eid,[r['id'] for r in events.get_pending_followups(1)])
        with sqlite3.connect(self.path) as c:
            self.assertEqual(c.execute('SELECT followed_up,followup_asked_at FROM person_events WHERE id=?',(eid,)).fetchone(),(0,None))

    def test_asked_without_answer_is_not_asked_next_session(self):
        eid=events.add_event(1,'trip to Georgia','2020-01-01','')
        events.mark_followup_asked(eid)
        self.assertEqual(events.get_pending_followups(1),[])
        with sqlite3.connect(self.path) as c:
            r=c.execute('SELECT status,outcome,followup_asked_at FROM person_events WHERE id=?',(eid,)).fetchone()
        self.assertEqual(r[:2],('planned',None));self.assertIsNotNone(r[2])
        events.mark_followed_up(eid,'It was fun')
        self.assertEqual(events.get_pending_followups(1),[])

    def test_rescheduled_trip_can_be_asked_again_when_due(self):
        eid=events.add_event(1,'trip','2020-01-01','')
        events.mark_followup_asked(eid)
        events.reschedule_event(eid,'2020-02-01')
        self.assertIn(eid,[r['id'] for r in events.get_pending_followups(1)])

    def test_future_plan_not_due_and_other_person_unaffected(self):
        eid=events.add_event(1,'trip','2099-01-01','')
        self.assertEqual(events.get_pending_followups(1),[])
        self.assertEqual(events.get_pending_followups(2),[])
