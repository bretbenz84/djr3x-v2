"""Open commitments — accountability ribbing (docs/comedy_improvements.md §8).

A first-person FUTURE promise ("I'll fix that sensor", "I'm gonna call my mom") is filed as
a status='promised' person_event; Rex may dryly needle the still-open promise on a LATER
turn, and it's cleared on a cancel/never-mind or a "did it". The hard part is the linguistic
gate (a hedge like "I should really…" / "maybe I'll…" / "I might…" / a question is NOT a
commitment), and the structural guarantee that a promise never collides with the shipped
open-plans / proactive-followup paths (those read status='planned' only).
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from memory import events


def _make_db(path: Path) -> None:
    from setup_assets import DB_SCHEMA

    with sqlite3.connect(path) as conn:
        conn.executescript(DB_SCHEMA)


class CommitmentDetectionTest(unittest.TestCase):
    """The first-person future-intent gate — the explicit hard 80%."""

    COMMITMENTS = [
        "I'll fix the sensor this weekend",
        "I'm gonna call my mom later",
        "I am going to send you the photos",
        "I promise to clean the garage",
        "I will finally read that book",
        "yeah ok, I'll handle that tomorrow",
        "I'll get around to the dishes",
        "I swear I'll email her back",
        "I gotta finish the deck",
    ]

    NOT_COMMITMENTS = [
        "I should really fix that sensor",      # hedge
        "I ought to call my mom",               # hedge
        "maybe I'll get to it",                 # maybe
        "I might fix it later",                 # might
        "I may come by",                        # may
        "I could do that I guess",              # could
        "I wish I could help",                  # wish
        "I'd love to see it sometime",          # would-love-to
        "I was gonna call but forgot",          # past intent
        "I think I'll head out",                # think-I'll
        "I hope to make it",                    # hope-to
        "I keep meaning to do it",              # keep-meaning-to
        "we'll see how it goes",                # we'll see
        "if I get time I'll do it",             # conditional
        "should I fix the sensor?",             # question
        "I'll be honest, that was rough",       # I'll be honest idiom
        "I'll say this once",                   # I'll say idiom
        "I'll bet you ten credits",             # I'll bet idiom
        "tell me about your day",               # no first-person commitment at all
        "I'm going to bed",                     # movement/state, not a task
        "I am going to sleep",                  # state
        "I'm going to the store",               # destination
        "I'll be right back",                   # be-filler
        "I will be there in a sec",             # will-be-filler
        "I'm gonna grab a coffee",              # immediate filler
        "I gotta run to work",                  # immediate departure
    ]

    TASKY_GOING_TO = [
        "I'm going to work on the deck this weekend",  # "work ON" is a task, not the destination
    ]

    def test_commitments_match(self):
        for t in self.COMMITMENTS:
            self.assertTrue(events.looks_like_commitment(t), f"should be a commitment: {t!r}")

    def test_hedges_and_questions_excluded(self):
        for t in self.NOT_COMMITMENTS:
            self.assertFalse(events.looks_like_commitment(t), f"should NOT be a commitment: {t!r}")

    def test_going_to_work_on_something_is_still_a_task(self):
        for t in self.TASKY_GOING_TO:
            self.assertTrue(events.looks_like_commitment(t), f"should be a commitment: {t!r}")

    def test_completion_detection(self):
        for t in ["I finally fixed it", "I already called them", "it's done now",
                  "took care of it", "I did it", "we got it done"]:
            self.assertTrue(events.looks_like_completion(t), f"should read as done: {t!r}")
        for t in ["I'll fix it", "I'm gonna do it", "I should fix it"]:
            self.assertFalse(events.looks_like_completion(t), f"should NOT read as done: {t!r}")

    def test_action_phrase_strips_the_commissive_head(self):
        self.assertEqual(events._commitment_action("I'll fix the sensor this weekend"),
                         "fix the sensor this weekend")
        self.assertEqual(events._commitment_action("yeah I'm gonna call my mom later"),
                         "call my mom later")
        self.assertEqual(events._commitment_action("I promise to send the photos"),
                         "send the photos")
        self.assertEqual(events._commitment_action("I'll get to it"), "get to it")
        # Too little left after the head -> fall back to the full utterance.
        self.assertEqual(events._commitment_action("I'll cook"), "I'll cook")


class CommitmentNeedleTest(unittest.TestCase):
    """_open_commitments_prompt_line — one aged promise, restraint rule, throttles."""

    def _promise(self, action, eid=1, hours_ago=48):
        made = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return {"id": eid, "event_name": action, "mentioned_at": made.isoformat()}

    def _line(self, promises, *, anticipated=()):
        from intelligence import llm

        with mock.patch("memory.events.get_open_commitments", return_value=list(promises)), \
             mock.patch.object(llm, "_note_commitment_needled"), \
             mock.patch.object(llm, "_open_plan_anticipated",
                               side_effect=lambda pid, eid: eid in anticipated):
            return llm._open_commitments_prompt_line(1)

    def test_disabled_returns_empty(self):
        from intelligence import llm

        with mock.patch.object(llm.config, "OPEN_COMMITMENTS_ENABLED", False):
            self.assertEqual(self._line([self._promise("fix the sensor")]), "")

    def test_no_promises_returns_empty(self):
        self.assertEqual(self._line([]), "")

    def test_aged_promise_surfaces_with_restraint(self):
        line = self._line([self._promise("fix the sensor")])
        self.assertIn("fix the sensor", line)
        self.assertIn("accountability ribbing", line)
        self.assertIn("do", line.lower())            # the "do NOT nag/lead" restraint
        self.assertEqual(line.count("\n"), 0)        # one line

    def test_fresh_promise_is_not_ribbed_yet(self):
        # The joke is the LATER callback — a promise made minutes ago is gated out.
        self.assertEqual(self._line([self._promise("fix the sensor", hours_ago=0)]), "")

    def test_already_needled_is_skipped(self):
        self.assertEqual(self._line([self._promise("fix the sensor", eid=7)], anticipated={7}), "")

    def test_capped_to_one(self):
        line = self._line([self._promise("fix the sensor", eid=1),
                           self._promise("call your mom", eid=2)])
        self.assertIn("fix the sensor", line)
        self.assertNotIn("call your mom", line)      # OPEN_COMMITMENTS_MAX = 1


class CommitmentResolutionTest(unittest.TestCase):
    """resolve_matching_commitments — retract vs done, scoped to the promised pool."""

    def _promise(self, action, eid):
        return {"id": eid, "event_name": action, "event_notes": action}

    def test_cancel_clears_a_token_matched_promise(self):
        promises = [self._promise("fix the sensor", 1)]
        with mock.patch.object(events, "get_open_commitments", return_value=promises), \
             mock.patch.object(events, "cancel_event") as cancel, \
             mock.patch.object(events, "mark_followed_up") as done:
            resolved = events.resolve_matching_commitments(1, "never mind the sensor, scrap it")
        self.assertEqual(len(resolved), 1)
        cancel.assert_called_once()
        self.assertEqual(cancel.call_args[0][0], 1)
        done.assert_not_called()

    def test_completion_marks_followed_up(self):
        promises = [self._promise("fix the sensor", 1)]
        with mock.patch.object(events, "get_open_commitments", return_value=promises), \
             mock.patch.object(events, "cancel_event") as cancel, \
             mock.patch.object(events, "mark_followed_up") as done:
            resolved = events.resolve_matching_commitments(1, "I finally fixed the sensor")
        self.assertEqual(len(resolved), 1)
        done.assert_called_once()
        self.assertEqual(done.call_args[0][0], 1)
        cancel.assert_not_called()

    def test_neutral_turn_resolves_nothing(self):
        promises = [self._promise("fix the sensor", 1)]
        with mock.patch.object(events, "get_open_commitments", return_value=promises), \
             mock.patch.object(events, "cancel_event") as cancel, \
             mock.patch.object(events, "mark_followed_up") as done:
            resolved = events.resolve_matching_commitments(1, "the weather is nice today")
        self.assertEqual(resolved, [])
        cancel.assert_not_called()
        done.assert_not_called()

    def test_generic_cancel_clears_the_lone_promise(self):
        promises = [self._promise("fix the sensor", 1)]
        with mock.patch.object(events, "get_open_commitments", return_value=promises), \
             mock.patch.object(events, "cancel_event") as cancel, \
             mock.patch.object(events, "mark_followed_up"):
            resolved = events.resolve_matching_commitments(1, "eh, changed my mind")
        self.assertEqual(len(resolved), 1)      # single-promise fallback for a cancel
        cancel.assert_called_once()

    def test_bare_done_does_not_nuke_a_promise(self):
        # Completion always requires a token match — a vague "I did it" can't retire a promise.
        promises = [self._promise("fix the sensor", 1)]
        with mock.patch.object(events, "get_open_commitments", return_value=promises), \
             mock.patch.object(events, "cancel_event") as cancel, \
             mock.patch.object(events, "mark_followed_up") as done:
            resolved = events.resolve_matching_commitments(1, "I did it")
        self.assertEqual(resolved, [])
        done.assert_not_called()
        cancel.assert_not_called()


class CommitmentStorageTest(unittest.TestCase):
    """add_commitment / get_open_commitments against a temp people.db, and the structural
    guarantee that a promise is invisible to the plan readers (no double-mention)."""

    def setUp(self):
        from memory import database as db

        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "people.db"
        _make_db(db_path)
        self._patch = mock.patch.object(db, "_DB_FILE", db_path)
        self._patch.start()
        from memory import people as people_memory
        self.pid = people_memory.enroll_person("Bret")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_promise_round_trips_and_is_invisible_to_plan_readers(self):
        events.add_commitment(self.pid, "I'll fix the sensor this weekend")
        open_c = events.get_open_commitments(self.pid)
        self.assertEqual(len(open_c), 1)
        self.assertEqual(open_c[0]["event_name"], "fix the sensor this weekend")
        # Structural partition: a 'promised' row never appears as a plan/follow-up.
        self.assertEqual(events.get_upcoming_events(self.pid), [])
        self.assertEqual(events.get_open_events(self.pid), [])
        self.assertEqual(events.get_pending_followups(self.pid), [])

    def test_planned_event_is_not_a_commitment(self):
        from datetime import date
        events.add_event(self.pid, "camping trip",
                         (date.today() + timedelta(days=3)).isoformat(), "the big camping trip")
        self.assertEqual(events.get_open_commitments(self.pid), [])   # plans aren't promises

    def test_cancel_resolution_clears_the_promise(self):
        events.add_commitment(self.pid, "I'm gonna call my mom")
        self.assertEqual(len(events.get_open_commitments(self.pid)), 1)
        events.resolve_matching_commitments(self.pid, "never mind calling my mom, scrap it")
        self.assertEqual(events.get_open_commitments(self.pid), [])

    def test_dedup_refreshes_one_row(self):
        events.add_commitment(self.pid, "I'll fix the sensor")
        events.add_commitment(self.pid, "I'll fix the sensor")
        self.assertEqual(len(events.get_open_commitments(self.pid)), 1)


if __name__ == "__main__":
    unittest.main()
