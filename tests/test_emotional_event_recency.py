"""Emotional-event recency regression tests.

Live failure: Bret answered "what's the most difficult thing you've been
through?" with "my mother's passing" (13 years ago, no timeframe stated). The
event was stored with an implicit "now" and a 365-day window, so every boot
opened with "Hey Bret, how are you holding up with everything?" — fresh-grief
treatment for settled history.

Policy now:
  - recency is read from EXPLICIT time markers only ("yesterday" → recent,
    "13 years ago" → historical, nothing stated → unknown).
  - heavy events drive greetings/check-ins ONLY when recency == "recent";
    mild same-day venting (bad_day etc.) is inherently about now and exempt.
  - unknown-recency heavy disclosures get a tactful "how long ago was that?"
    directive; the answer updates the stored row.
  - the LLM may never invent "recent" without a marker in the text.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class EventRecencyFromTextTests(unittest.TestCase):
    def _recency(self, text):
        from intelligence import empathy

        return empathy.event_recency_from_text(text)

    def test_no_timeframe_is_unknown(self):
        for text in [
            "my mother's passing",
            "my mom passed away",
            "I lost my dad",
            "the hardest thing was losing my best friend",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self._recency(text), "unknown")

    def test_recent_markers(self):
        for text in [
            "my mom passed away yesterday",
            "my mother died last week",
            "we lost him this week",
            "she passed two days ago",
            "he died a few weeks ago",
            "my dad passed away two months ago",
            "I just lost my dog",
            "she passed away recently",
            "we lost her last month",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self._recency(text), "recent")

    def test_historical_markers(self):
        for text in [
            "my mom passed away 13 years ago",
            "she died three years ago",
            "I lost my dad when I was a kid",
            "he passed away back in 2013",
            "my mother died a long time ago",
            "we lost him years back",
            "she passed last year",
            "he died eight months ago",
            "I lost my mom growing up",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self._recency(text), "historical")

    def test_llm_cannot_invent_recent(self):
        from intelligence import empathy

        # No marker in text: LLM "recent" is rejected, "historical" allowed.
        self.assertEqual(
            empathy.resolve_event_recency("my mother passed away", "recent"),
            "unknown",
        )
        self.assertEqual(
            empathy.resolve_event_recency("my mother passed away", "historical"),
            "historical",
        )
        # Deterministic marker beats a conflicting LLM answer.
        self.assertEqual(
            empathy.resolve_event_recency("she died 13 years ago", "recent"),
            "historical",
        )


_EVENTS_DDL = """
CREATE TABLE person_emotional_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    category TEXT,
    valence REAL,
    description TEXT,
    loss_subject TEXT,
    loss_subject_kind TEXT,
    loss_subject_name TEXT,
    mentioned_at DATETIME,
    last_acknowledged_at DATETIME,
    checkins_muted_at DATETIME,
    checkins_muted_reason TEXT,
    sensitivity_decay_days INTEGER,
    person_invited_topic INTEGER DEFAULT 1,
    recency TEXT DEFAULT 'unknown'
)
"""


class CheckinRecencyGateTests(unittest.TestCase):
    """SQL gating against a real (temp) sqlite DB."""

    def setUp(self):
        from memory import database as db

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "people.db"
        conn = sqlite3.connect(db_path)
        conn.execute(_EVENTS_DDL)
        conn.commit()
        conn.close()
        self._patch = mock.patch.object(db, "_DB_FILE", db_path)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _add(self, **kw):
        from memory import emotional_events as ee

        defaults = dict(
            person_id=1,
            category="grief",
            description="the speaker's mother passed away",
            valence=-1.0,
            loss_subject="mother",
            loss_subject_kind="person",
        )
        defaults.update(kw)
        return ee.add_event(**defaults)

    def test_unknown_recency_grief_never_greets(self):
        from memory import emotional_events as ee

        self._add(recency="unknown")
        self.assertEqual(ee.get_startup_checkins(1, None), [])
        self.assertEqual(ee.get_due_checkins(1), [])

    def test_historical_grief_never_greets(self):
        from memory import emotional_events as ee

        self._add(recency="historical")
        self.assertEqual(ee.get_startup_checkins(1, None), [])
        self.assertEqual(ee.get_due_checkins(1), [])

    def test_recent_grief_still_greets(self):
        from memory import emotional_events as ee

        self._add(recency="recent", description="mother passed away last week")
        startup = ee.get_startup_checkins(1, None)
        self.assertEqual(len(startup), 1)
        self.assertEqual(startup[0]["recency"], "recent")

    def test_mild_same_day_venting_exempt_from_gate(self):
        from memory import emotional_events as ee

        self._add(
            category="bad_day",
            description="had a rough day at work",
            valence=-0.4,
            loss_subject=None,
            loss_subject_kind=None,
            recency="unknown",
        )
        self.assertEqual(len(ee.get_due_checkins(1)), 1)

    def test_recency_probe_answer_updates_row(self):
        from intelligence import empathy
        from memory import emotional_events as ee

        row_id = self._add(recency="unknown")
        empathy.note_recency_probe(1, row_id)
        resolved = empathy.consume_recency_answer(1, "that was 13 years ago")
        self.assertEqual(resolved, "historical")
        self.assertEqual(ee.get_startup_checkins(1, None), [])

        # And the recent case opens check-ins up.
        row2 = self._add(description="father passed away", recency="unknown")
        empathy.note_recency_probe(1, row2)
        resolved = empathy.consume_recency_answer(1, "it happened last week")
        self.assertEqual(resolved, "recent")
        startup = ee.get_startup_checkins(1, None)
        self.assertEqual(len(startup), 1)
        self.assertEqual(startup[0]["id"], row2)

    def test_answer_without_marker_keeps_probe_pending(self):
        from intelligence import empathy

        row_id = self._add(recency="unknown")
        empathy.note_recency_probe(1, row_id)
        self.assertIsNone(empathy.consume_recency_answer(1, "I'd rather not say"))
        # Still pending: a later marker answer lands.
        self.assertEqual(
            empathy.consume_recency_answer(1, "about five years ago"),
            "historical",
        )


class RecencyProbeDirectiveTests(unittest.TestCase):
    def test_unknown_heavy_invited_event_gets_probe_directive(self):
        from intelligence import empathy

        result = {
            "invitation": True,
            "event": {
                "category": "grief",
                "valence": -1.0,
                "recency": "unknown",
                "description": "mother passed away",
            },
        }
        pack = empathy.augment_mode_for_recency_probe(
            result, {"mode": "listen", "directive": "Listen first.", "reason": "x"}
        )
        self.assertIn("how", pack["directive"].lower())
        self.assertIn("long ago", pack["directive"].lower())

    def test_recent_or_mild_events_unchanged(self):
        from intelligence import empathy

        base = {"mode": "listen", "directive": "Listen first.", "reason": "x"}
        recent = {
            "invitation": True,
            "event": {"category": "grief", "valence": -1.0, "recency": "recent"},
        }
        self.assertEqual(
            empathy.augment_mode_for_recency_probe(recent, base)["directive"],
            "Listen first.",
        )
        mild = {
            "invitation": True,
            "event": {"category": "bad_day", "valence": -0.4, "recency": "unknown"},
        }
        self.assertEqual(
            empathy.augment_mode_for_recency_probe(mild, base)["directive"],
            "Listen first.",
        )


if __name__ == "__main__":
    unittest.main()
