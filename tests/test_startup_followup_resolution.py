"""
Regression for the festival re-ask bug: a STARTUP follow-up ("how did the festival go?")
that the user answers ("I never went") must close the event in memory, so it isn't
re-asked on the next run.

The startup follow-up fires from consciousness OUTSIDE interaction._post_response (which
is what normally arms the resolver). The fix has consciousness call
interaction.set_awaiting_followup_event when it fires; this test pins that arming +
resolution end-to-end at the memory layer.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory import database as db


def _build_people_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")


class StartupFollowupResolutionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        _build_people_db(self._path)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()
        interaction._awaiting_followup_event = None

    def tearDown(self):
        from intelligence import interaction
        interaction._awaiting_followup_event = None
        self._patch.stop()
        self._tmp.cleanup()

    def _make_passed_event(self) -> int:
        from memory import events
        # A dated plan whose date has already passed → get_pending_followups returns it.
        return int(events.add_event(1, "festival", "2026-06-01", "going to a festival"))

    def test_event_is_pending_before(self):
        from memory import events
        self._make_passed_event()
        self.assertEqual(len(events.get_pending_followups(1)), 1)

    def test_never_went_resolves_and_stops_reasking(self):
        from intelligence import interaction
        from memory import events
        eid = self._make_passed_event()

        # consciousness arms the resolver when the startup follow-up fires.
        interaction.set_awaiting_followup_event(1, eid, "festival")
        self.assertIsNotNone(interaction._awaiting_followup_event)

        # The user answers that it never happened.
        interaction._resolve_awaiting_followup("I never went", 1)

        # Event is closed (followed_up) and no longer a pending follow-up → no re-ask.
        self.assertIsNone(interaction._awaiting_followup_event)
        self.assertEqual(events.get_pending_followups(1), [])
        self.assertEqual(events.get_open_events(1), [])

    def test_real_outcome_also_resolves(self):
        from intelligence import interaction
        from memory import events
        eid = self._make_passed_event()
        interaction.set_awaiting_followup_event(1, eid, "festival")
        interaction._resolve_awaiting_followup("it was incredible, best night ever", 1)
        self.assertIsNone(interaction._awaiting_followup_event)
        self.assertEqual(events.get_pending_followups(1), [])

    def test_no_arm_means_no_resolution(self):
        from intelligence import interaction
        from memory import events
        self._make_passed_event()
        # Nothing armed → resolver is a no-op and the event stays pending.
        self.assertIsNone(interaction._resolve_awaiting_followup("I never went", 1))
        self.assertEqual(len(events.get_pending_followups(1)), 1)

    def test_set_awaiting_ignores_missing_ids(self):
        from intelligence import interaction
        interaction._awaiting_followup_event = None
        interaction.set_awaiting_followup_event(None, 5, "x")
        interaction.set_awaiting_followup_event(1, None, "x")
        self.assertIsNone(interaction._awaiting_followup_event)


if __name__ == "__main__":
    unittest.main()
