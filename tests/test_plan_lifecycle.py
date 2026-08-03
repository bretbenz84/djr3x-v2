"""Plan-lifecycle fixes (field 2026-07-31 → 08-02).

Four failure shapes, one root: a stored plan is a belief about the future, not
a fact — but every consumer phrased it as one.
  * "I might move the couch this weekend" → next boot: "the couch move is TODAY"
    (hedge lost at storage, asserted at greeting).
  * Lake Folsom planned for TOMORROW, mentioned earlier today → greeted with
    "How'd Lake Folsom go earlier today?" (mention-time conflated with event-time).
  * "We're not going to like falsum anymore" (garbled 'Folsom') shared no token
    with the stored plan → survived to be re-anticipated two hours later.
  * A wrong-name fix became the callback "JT's name change — did that settle in
    okay?" (bookkeeping filed as a life event).
"""

import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
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


class HedgeDetectionTest(unittest.TestCase):
    def test_hedged_phrasings_detected(self):
        from memory import events
        for text in (
            "I might move the couch this weekend",
            "maybe we'll hit the lake",
            "thinking about dyeing my hair",
            "we may go paddleboarding",
            "not sure if I'll make it out there",
        ):
            self.assertTrue(events.looks_like_hedged_plan(text), text)

    def test_committed_phrasings_not_hedged(self):
        from memory import events
        for text in (
            "I'm going to the movies tonight",
            "tomorrow I'm going to Lake Folsom with my friends",
            "we leave Saturday morning",
        ):
            self.assertFalse(events.looks_like_hedged_plan(text), text)


class HedgeStorageTest(_TempDb):
    def _hedged(self, eid):
        from memory import events as _
        row = db.fetchone("SELECT hedged FROM person_events WHERE id = ?", (eid,))
        return bool(row["hedged"])

    def test_explicit_flag_stored(self):
        from memory import events
        eid = events.add_event(1, "couch move", None, "", hedged=True)
        self.assertTrue(self._hedged(eid))

    def test_notes_backstop_detects_hedge(self):
        from memory import events
        eid = events.add_event(
            1, "couch move", None, "said he might move the couch this weekend",
        )
        self.assertTrue(self._hedged(eid))

    def test_firm_restatement_clears_hedge_but_not_vice_versa(self):
        from memory import events
        eid = events.add_event(1, "camping trip", None, "", hedged=True)
        # A firm restatement upgrades the plan…
        eid2 = events.add_event(1, "camping trip", None, "we're going camping", hedged=False)
        self.assertEqual(eid, eid2)
        self.assertFalse(self._hedged(eid))
        # …and a later hedge must NOT downgrade it back.
        eid3 = events.add_event(1, "camping trip", None, "might still go", hedged=True)
        self.assertEqual(eid, eid3)
        self.assertFalse(self._hedged(eid))


class AnticipationPromptTest(unittest.TestCase):
    def _prompt(self, event):
        from intelligence import consciousness
        with mock.patch("random.random", return_value=0.0):
            return consciousness._build_anticipation_prompt("Bret", event, "you just booted")

    def test_hedged_event_asks_instead_of_asserting(self):
        prompt = self._prompt({
            "event_name": "couch move",
            "event_date": date.today().isoformat(),
            "hedged": 1,
        })
        self.assertIn("MIGHT", prompt)
        self.assertIn("still the plan", prompt)
        self.assertIn("Do NOT state that it is happening", prompt)

    def test_firm_event_keeps_preemptive_reference(self):
        prompt = self._prompt({
            "event_name": "dentist appointment",
            "event_date": date.today().isoformat(),
            "hedged": 0,
        })
        self.assertIn("PREEMPTIVELY", prompt)
        self.assertNotIn("MIGHT", prompt)


class FollowupPhrasingTest(unittest.TestCase):
    def test_lean_dated_followup_does_not_assume_completion(self):
        from intelligence import lean_brain
        clause = lean_brain._event_followup_clause(
            {"event_name": "job interview", "kind": "past", "dated": True}
        )
        self.assertIn("whether it ended up happening", clause)
        self.assertIn("do not assume it did", clause)
        self.assertNotIn("almost certainly happened", clause)


class GarbledCancellationTest(_TempDb):
    def test_cancellation_falls_back_to_recent_rex_hint(self):
        """Token match fails on the garbled name; the event Rex just raised wins."""
        from memory import events
        from intelligence import interaction
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        eid = events.add_event(1, "trip to Lake Folsom", tomorrow, "paddleboarding with JT")
        self.assertIsNotNone(eid)
        # A second open event so the "single open event" fallback can't mask the fix.
        events.add_event(1, "website launch", None, "work project")

        hint = "today's the day for Lake Folsom—hope the paddleboarding goes well"
        with mock.patch.object(interaction, "_recent_rex_memory_hint", return_value=hint):
            labels = interaction._cancel_stale_event_memory(
                1, "We're not going to like falsum anymore"
            )
        self.assertTrue(any("Folsom" in label for label in labels), labels)
        open_names = [e["event_name"] for e in events.get_open_events(1)]
        self.assertNotIn("trip to Lake Folsom", open_names)
        self.assertIn("website launch", open_names)


class BookkeepingThreadFilterTest(unittest.TestCase):
    def test_bookkeeping_threads_dropped(self):
        from intelligence import llm
        transcript = [
            {"speaker": "Guest", "text": "My name's not Brad, it's JT. Please forget my info."},
        ]
        kept = llm._filtered_open_threads(
            [
                "whether JT's name change settled in",
                "if Rex's memory banks recovered",
                "whether the asked to forget request went through",
            ],
            transcript,
        )
        self.assertEqual(kept, [])

    def test_real_threads_survive(self):
        from intelligence import llm
        transcript = [
            {"speaker": "Bret", "text": "The dentist appointment is Tuesday and I'm dreading it."},
        ]
        kept = llm._filtered_open_threads(
            ["whether the dentist appointment happened"], transcript,
        )
        self.assertEqual(kept, ["whether the dentist appointment happened"])


if __name__ == "__main__":
    unittest.main()
