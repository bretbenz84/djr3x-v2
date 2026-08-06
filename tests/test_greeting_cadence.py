"""
Greeting cadence (intelligence/greeting_cadence.py) + its wiring into the boot ladder.

Owner gripe 2026-08-05: "on repeat visits/boots that are recent (20 minutes, a few
hours) he shouldn't ask how I'm doing again — people who leave and enter rooms
repeatedly don't greet each other that way more than once."

The reason it kept happening: every anti-repeat guard Rex had was IN-MEMORY.
`_greeted_this_session` is a set wiped at process start and `_should_fire_presence`
uses monotonic cooldowns that reset with the process — so the single most common
repeat-visit event, a RESTART, was the one thing guaranteed to defeat all of them.
These tests pin the fix to PERSISTED per-person timestamps, which a reboot cannot
reset, and lock in that the quick-return branch asks nothing.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import config
from intelligence import greeting_cadence as gc
from memory import database as db
from memory import people


def _build_people_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA
    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")


class _PeopleDbTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        _build_people_db(self._path)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()
        self._flag = mock.patch.object(config, "GREETING_CADENCE_ENABLED", True)
        self._flag.start()

    def tearDown(self) -> None:
        self._flag.stop()
        self._patch.stop()
        self._tmp.cleanup()

    def _greeted_ago(self, **delta) -> None:
        stamp = (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()
        db.execute("UPDATE people SET last_greeted_at = ? WHERE id = 1", (stamp,))

    def _asked_ago(self, **delta) -> None:
        stamp = (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()
        db.execute("UPDATE people SET last_wellbeing_ask_at = ? WHERE id = 1", (stamp,))


class RecencyTests(_PeopleDbTestCase):

    def test_never_greeted_falls_through_to_the_normal_ladder(self):
        self.assertEqual(gc.recency(1), (None, None))

    def test_minutes_ago_is_a_snap_return(self):
        self._greeted_ago(minutes=5)
        bucket, age = gc.recency(1)
        self.assertEqual(bucket, gc.SNAP)
        self.assertLess(age, 20 * 60)

    def test_an_hour_ago_is_a_recent_return(self):
        self._greeted_ago(hours=1)
        self.assertEqual(gc.recency(1)[0], gc.RECENT)

    def test_this_morning_falls_back_to_the_existing_ladder(self):
        self._greeted_ago(hours=9)
        self.assertIsNone(gc.recency(1)[0])

    def test_thresholds_are_configurable(self):
        self._greeted_ago(minutes=30)
        self.assertEqual(gc.recency(1)[0], gc.RECENT)
        with mock.patch.object(config, "GREETING_CADENCE_SNAP_SECS", 60 * 60):
            self.assertEqual(gc.recency(1)[0], gc.SNAP)

    def test_disabled_returns_nothing_so_behavior_is_unchanged(self):
        self._greeted_ago(minutes=5)
        with mock.patch.object(config, "GREETING_CADENCE_ENABLED", False):
            self.assertEqual(gc.recency(1), (None, None))

    def test_unknown_person_never_suppresses(self):
        for bad in (None, "Bret", 999):
            self.assertIsNone(gc.recency(bad)[0])

    def test_recency_survives_a_restart(self):
        # The whole point: this is read from the DB, not from process memory. Recording
        # a greeting and then re-importing the world must not reset it.
        people.record_greeting(1)
        self.assertEqual(gc.recency(1)[0], gc.SNAP)
        # Nothing in this module holds state — prove it by asserting there is none to hold.
        self.assertFalse(
            [n for n in vars(gc) if n.startswith("_last") or n.startswith("_seen")],
            "greeting_cadence must stay stateless, or a reboot resets it again",
        )

    def test_naive_timestamps_from_older_rows_still_parse(self):
        naive = (datetime.now() - timedelta(minutes=5)).isoformat()
        db.execute("UPDATE people SET last_greeted_at = ? WHERE id = 1", (naive,))
        self.assertIsNotNone(gc.recency(1)[1])


class WellbeingAskTests(_PeopleDbTestCase):

    def test_ask_is_recorded_and_then_spent(self):
        self.assertEqual(gc.wellbeing_ask_spent(1), (False, None))
        self.assertTrue(gc.note_wellbeing_ask(1, "Hey Bret, how are you?"))
        self.assertTrue(gc.wellbeing_ask_spent(1)[0])

    def test_ask_expires_after_the_cooldown(self):
        self._asked_ago(hours=9)
        self.assertFalse(gc.wellbeing_ask_spent(1)[0])
        self._asked_ago(hours=1)
        self.assertTrue(gc.wellbeing_ask_spent(1)[0])

    def test_a_line_that_did_not_ask_records_nothing(self):
        self.assertFalse(gc.note_wellbeing_ask(1, "Hey, you're back."))
        self.assertFalse(gc.wellbeing_ask_spent(1)[0])

    def test_suppression_line_bans_the_reworded_versions_too(self):
        gc.note_wellbeing_ask(1, "How's it going?")
        line = gc.suppression_line(1)
        self.assertIn("ALREADY asked", line)
        for phrase in ("how are you", "how's it going", "what's up", "how's your day"):
            self.assertIn(phrase, line.lower())
        self.assertIn("reworded", line)

    def test_suppression_still_allows_a_specific_followup(self):
        # Banning the RITUAL must not ban genuine interest in what they told him.
        gc.note_wellbeing_ask(1, "How are you?")
        self.assertIn("specific", gc.suppression_line(1).lower())

    def test_no_suppression_line_when_he_has_not_asked(self):
        self.assertEqual(gc.suppression_line(1), "")

    def test_disabled_never_suppresses(self):
        gc.note_wellbeing_ask(1, "How are you?")
        with mock.patch.object(config, "GREETING_CADENCE_ENABLED", False):
            self.assertEqual(gc.suppression_line(1), "")
            self.assertEqual(gc.wellbeing_ask_spent(1), (False, None))

    def test_ask_and_greeting_clocks_are_independent(self):
        # A return hello is fine every few hours; "how are you" is not. They must not
        # share a timestamp.
        people.record_greeting(1)
        self.assertFalse(gc.wellbeing_ask_spent(1)[0])
        gc.note_wellbeing_ask(1, "How are you?")
        self._greeted_ago(hours=9)
        self.assertIsNone(gc.recency(1)[0])
        self.assertTrue(gc.wellbeing_ask_spent(1)[0])


class AskDetectionTests(unittest.TestCase):
    """Detection runs on Rex's FINAL TEXT, not on which prompt-builder ran — the
    builder says what he was told to do, the text says what he actually said."""

    def test_detects_the_ask_in_its_common_shapes(self):
        for line in (
            "Hey Bret, how are you?",
            "How's it going?",
            "How have you been?",
            "How are things?",
            "What's up?",
            "What's new, Bret?",
            "How's your day going?",
            "How'd your weekend go?",
            "You doing ok?",
            "Everything alright?",
            "Back already. How you holding up?",
        ):
            with self.subTest(line=line):
                self.assertTrue(gc.looks_like_wellbeing_ask(line))

    def test_does_not_fire_on_lines_that_are_not_asks(self):
        for line in (
            "Hey, you're back.",
            "That was quick.",
            "Good to see you.",
            "I'm contemplative today, since you ask.",
            "You never ask how are you, and it shows.",   # no question mark
            "How does that thing even work?",             # a question, not about them
            "How is it powered?",                         # about a THING (review find:
            "How was it?",                                # bare "it" used to match)
            "What's the capital of Peru?",
            "",
        ):
            with self.subTest(line=line):
                self.assertFalse(gc.looks_like_wellbeing_ask(line))

    def test_a_question_mark_is_required(self):
        self.assertFalse(gc.looks_like_wellbeing_ask("I asked how you are"))
        self.assertTrue(gc.looks_like_wellbeing_ask("I asked how you are?"))


class ConstraintTextTests(unittest.TestCase):

    def test_snap_constraint_forbids_a_hello_and_any_question(self):
        text = gc.greeting_constraint(gc.SNAP, 300)
        self.assertIn("ask NOTHING", text)
        self.assertIn("Do NOT run a hello", text)
        self.assertIn("Under eight words", text)
        for phrase in ("how are you", "what's up", "what's new"):
            self.assertIn(phrase, text.lower())
        self.assertIn("never annoyed", text)

    def test_recent_constraint_allows_the_return_beat_but_not_the_question(self):
        text = gc.greeting_constraint(gc.RECENT, 60 * 90)
        self.assertIn("RETURN, not a", text)
        self.assertIn("do NOT ask", text)
        self.assertIn("about an hour ago", gc.describe_gap(60 * 90) or "")
        self.assertNotIn("Under eight words", text)   # a return line can breathe

    def test_no_constraint_when_there_is_no_bucket(self):
        self.assertEqual(gc.greeting_constraint(None, 99999), "")

    def test_gap_phrasing_scales(self):
        self.assertEqual(gc.describe_gap(None), "")
        self.assertIn("minute", gc.describe_gap(60))
        self.assertIn("minutes", gc.describe_gap(15 * 60))
        self.assertIn("an hour", gc.describe_gap(60 * 60))
        self.assertIn("hours", gc.describe_gap(3 * 60 * 60))


class LadderWiringTests(_PeopleDbTestCase):
    """The consciousness side: the ladder must consult recency and append the
    no-re-ask clause to whichever branch won."""

    def test_greeting_recency_helper_reads_the_db(self):
        from intelligence import consciousness
        self._greeted_ago(minutes=5)
        self.assertEqual(consciousness._greeting_recency(1)[0], gc.SNAP)

    def test_wellbeing_clause_helper_returns_the_suppression(self):
        from intelligence import consciousness
        gc.note_wellbeing_ask(1, "How are you?")
        self.assertIn("ALREADY asked", consciousness._wellbeing_ask_clause(1))

    def test_helpers_degrade_to_empty_rather_than_raising(self):
        from intelligence import consciousness
        with mock.patch.object(gc, "recency", side_effect=RuntimeError("db gone")):
            self.assertEqual(consciousness._greeting_recency(1), (None, None))
        with mock.patch.object(gc, "suppression_line", side_effect=RuntimeError("db gone")):
            self.assertEqual(consciousness._wellbeing_ask_clause(1), "")

    def test_quick_return_prompt_never_asks_how_they_are(self):
        # The end-to-end contract, asserted on the prompt string the ladder builds.
        # Both buckets must forbid the wellbeing question by name, and neither may
        # carry the "ends in a question mark" requirement the normal builders impose.
        for bucket, age in ((gc.SNAP, 300), (gc.RECENT, 5400)):
            with self.subTest(bucket=bucket):
                prompt = f"You see Bret again. {gc.greeting_constraint(bucket, age)}"
                lowered = prompt.lower()
                self.assertIn("Bret", prompt)
                for phrase in ("how are you", "what's up", "what's new"):
                    self.assertIn(phrase, lowered)
                self.assertNotIn("ends in a question mark", prompt)
                self.assertNotIn("must end in a question mark", prompt)


class PeopleColumnTests(_PeopleDbTestCase):

    def test_record_and_read_round_trip(self):
        self.assertIsNone(people.last_wellbeing_ask_age_secs(1))
        people.record_wellbeing_ask(1)
        self.assertLess(people.last_wellbeing_ask_age_secs(1), 5.0)

    def test_bad_ids_are_swallowed(self):
        people.record_wellbeing_ask("not-an-id")      # must not raise
        self.assertIsNone(people.last_wellbeing_ask_age_secs("not-an-id"))
        self.assertIsNone(people.last_greeted_age_secs(None))

    def test_last_greeted_age_tracks_record_greeting(self):
        self.assertIsNone(people.last_greeted_age_secs(1))
        people.record_greeting(1)
        self.assertLess(people.last_greeted_age_secs(1), 5.0)

    def test_column_is_added_to_a_pre_migration_database(self):
        # Old DBs in the field predate the column; verify_schema must heal them.
        from setup_assets import DB_SCHEMA
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.db"
            old_schema = DB_SCHEMA.replace("    last_wellbeing_ask_at   DATETIME,\n", "")
            self.assertNotIn("last_wellbeing_ask_at", old_schema)
            with sqlite3.connect(path) as conn:
                conn.executescript(old_schema)
                conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")
            with mock.patch.object(db, "_DB_FILE", path):
                # Reads degrade to None instead of exploding before the migration...
                self.assertIsNone(people.last_wellbeing_ask_age_secs(1))
                db._run_migrations()
                cols = {r[1] for r in sqlite3.connect(path).execute(
                    "PRAGMA table_info(people)")}
                self.assertIn("last_wellbeing_ask_at", cols)
                people.record_wellbeing_ask(1)
                self.assertIsNotNone(people.last_wellbeing_ask_age_secs(1))


if __name__ == "__main__":
    unittest.main()
