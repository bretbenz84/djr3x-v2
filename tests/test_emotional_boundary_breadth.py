"""
tests/test_emotional_boundary_breadth.py — two defects behind three sessions of
"how are you holding up with everything" openers (2026-08-27 .. 09-02):

  * "Tim Curry died." / "She died two days ago." (Dolly Parton) — news chatter on
    2026-08-26 — were stored as PERSONAL bereavements (person_invited_topic=1,
    180-day window) and led every first-sight greeting afterwards.
  * "I'd rather not talk about that" (09-01 23:01) muted ONE of the two rows; the
    other opened the next boot with "everything you're carrying right now" after
    Rex had promised not to bring it up again.

Temp sqlite DB, no LLM.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class _TempEventsDB(unittest.TestCase):
    def setUp(self):
        from memory import database as db
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        db_path = Path(self._tmp.name) / "people.db"
        conn = sqlite3.connect(db_path)
        conn.execute(_EVENTS_DDL)
        conn.commit()
        conn.close()
        patch = mock.patch.object(db, "_DB_FILE", db_path)
        patch.start()
        self.addCleanup(patch.stop)

    def _add(self, **kw):
        from memory import emotional_events as ee
        defaults = dict(
            person_id=1, category="death", valence=-1.0,
            description="she died two days ago", loss_subject="she",
            loss_subject_kind="person", recency="recent", person_invited_topic=True,
        )
        defaults.update(kw)
        return ee.add_event(**defaults)


class PublicFigureLossTest(unittest.TestCase):
    def _loss(self, **kw):
        from memory import emotional_events as ee
        base = dict(category="death", description="", loss_subject=None,
                    loss_subject_kind="person", loss_subject_name=None)
        base.update(kw)
        return ee.is_public_figure_loss(base)

    def test_tim_curry_is_a_public_figure(self):
        # The stored row from 2026-08-26 22:29:40, verbatim.
        self.assertTrue(self._loss(description="Tim Curry passed away.",
                                   loss_subject="tim curry", loss_subject_name="Tim Curry"))

    def test_the_classifier_can_say_so_for_a_pronoun(self):
        # "She died two days ago" is Dolly Parton only in context; the LLM has it.
        self.assertTrue(self._loss(description="she died two days ago",
                                   loss_subject="she", loss_relation="public_figure"))
        self.assertTrue(self._loss(description="she died two days ago",
                                   loss_subject="she", public_figure=True))

    def test_a_bare_pronoun_without_the_flag_stays_personal(self):
        self.assertFalse(self._loss(description="she died two days ago", loss_subject="she"))

    def test_a_named_relative_is_personal(self):
        self.assertFalse(self._loss(description="his mother Mary Smith passed away",
                                    loss_subject="mother", loss_subject_name="Mary Smith"))
        self.assertFalse(self._loss(description="lost their dog Max",
                                    loss_subject="dog", loss_subject_kind="pet",
                                    loss_subject_name="Max"))
        self.assertFalse(self._loss(description="my best friend Sam Jones died last week",
                                    loss_subject="best friend", loss_subject_name="Sam Jones"))

    def test_relation_word_in_the_description_wins(self):
        self.assertFalse(self._loss(description="the speaker is referencing a deceased son",
                                    loss_subject="son"))

    def test_only_death_and_grief_qualify(self):
        self.assertFalse(self._loss(category="illness", description="Tim Curry is ill",
                                    loss_subject_name="Tim Curry"))


class StartupCheckinGuardTest(_TempEventsDB):
    def test_a_public_figure_death_never_leads_a_greeting(self):
        from memory import emotional_events as ee
        self._add(description="Tim Curry passed away.", loss_subject="tim curry",
                  loss_subject_name="Tim Curry")
        self.assertEqual(ee.get_startup_checkins(1, None), [])
        self.assertEqual(ee.get_due_checkins(1), [])

    def test_a_personal_loss_still_does(self):
        from memory import emotional_events as ee
        self._add(description="the speaker's mother passed away", loss_subject="mother")
        self.assertEqual(len(ee.get_startup_checkins(1, None)), 1)

    def test_the_guard_does_not_starve_the_limit(self):
        from memory import emotional_events as ee
        self._add(description="Tim Curry passed away.", loss_subject="tim curry",
                  loss_subject_name="Tim Curry")
        self._add(description="the speaker's mother passed away", loss_subject="mother")
        got = ee.get_startup_checkins(1, None, limit=1)
        self.assertEqual([g["loss_subject"] for g in got], ["mother"])


class CategoryWideMuteTest(_TempEventsDB):
    def test_the_boundary_mutes_every_active_event_of_the_subject(self):
        from memory import emotional_events as ee
        a = self._add(description="Tim Curry passed away.", loss_subject="tim curry",
                      loss_subject_name="Tim Curry", person_invited_topic=True)
        b = self._add(description="she died two days ago")
        other = self._add(category="job_loss", description="his job offer was rescinded",
                          loss_subject=None)
        muted = ee.mute_category_for_person(1, "death", reason="I'd rather not talk about that.")
        self.assertEqual(sorted(muted), sorted([a, b]))
        due = ee.get_startup_checkins(1, None)
        self.assertEqual([d["category"] for d in due if d["category"] == "death"], [],
                         "a second death row must not open the next boot")
        # The other subject is untouched.
        muted_again = ee.mute_category_for_person(1, "job_loss", reason="x")
        self.assertEqual(muted_again, [other])

    def test_already_muted_rows_are_not_touched(self):
        from memory import emotional_events as ee
        a = self._add(description="she died two days ago")
        ee.mute_checkins(a, reason="first")
        self.assertEqual(ee.mute_category_for_person(1, "death", reason="second"), [])

    def test_recent_checkin_pseudo_category_is_a_no_op(self):
        from memory import emotional_events as ee
        self._add(description="she died two days ago")
        self.assertEqual(ee.mute_category_for_person(1, "recent_checkin", reason="x"), [])
        self.assertEqual(ee.mute_category_for_person(1, "", reason="x"), [])


if __name__ == "__main__":
    unittest.main()


class ClassifierPublicFigureTest(unittest.TestCase):
    """The LLM classifier is the only layer that can tell "she died two days ago"
    is about Dolly Parton: it gets a loss_relation field and the recent lines."""

    def _run(self, payload: dict, text="You're wrong. She died two days ago.", context=None):
        import json
        from intelligence import empathy
        captured = {}

        class _Msg:
            content = json.dumps(payload)

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        def _create(**kw):
            captured["prompt"] = kw["messages"][0]["content"]
            return _Resp()

        fake = mock.Mock()
        fake.chat.completions.create.side_effect = _create
        with mock.patch.object(empathy, "_client", fake):
            result = empathy.classify_affect(text, context_lines=context)
        return result, captured.get("prompt", "")

    def _payload(self, **event):
        base = dict(affect="sad", needs="vent", topic_sensitivity="heavy",
                    invitation=True, crisis=False, confidence=0.9)
        ev = dict(category="death", valence=-1.0, description="she died two days ago",
                  loss_subject="she", loss_subject_kind="person",
                  loss_subject_name=None, recency="recent")
        ev.update(event)
        base["event"] = ev
        return base

    def test_public_figure_relation_marks_the_event(self):
        from memory import emotional_events as ee
        result, _ = self._run(self._payload(loss_relation="public_figure"))
        self.assertEqual(result["event"]["loss_relation"], "public_figure")
        self.assertTrue(result["event"]["public_figure"])
        self.assertTrue(ee.is_public_figure_loss(result["event"]))

    def test_family_relation_is_personal(self):
        from memory import emotional_events as ee
        result, _ = self._run(self._payload(loss_relation="family", loss_subject="mother"))
        self.assertFalse(result["event"]["public_figure"])
        self.assertFalse(ee.is_public_figure_loss(result["event"]))

    def test_unknown_relation_values_are_dropped(self):
        result, _ = self._run(self._payload(loss_relation="cousin-ish"))
        self.assertIsNone(result["event"]["loss_relation"])
        self.assertFalse(result["event"]["public_figure"])

    def test_recent_lines_reach_the_prompt(self):
        _, prompt = self._run(
            self._payload(loss_relation="public_figure"),
            context=["Bret: Tim Curry died.", "Rex: Yeah. Tim Curry died at 80.",
                     "Bret: Dolly Parton died at 80.", "Rex: Nope, Dolly Parton is very much alive."],
        )
        self.assertIn("Dolly Parton died at 80", prompt)
        self.assertIn("public_figure", prompt)

    def test_no_context_is_fine(self):
        result, prompt = self._run(self._payload(), context=None)
        self.assertIsNotNone(result)
        self.assertNotIn("Recent conversation", prompt)
