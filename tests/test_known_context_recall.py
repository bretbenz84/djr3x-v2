"""Known-context recall + outcome resolution + shutdown persistence (field 2026-08-03).

Three faces of one failure — "the memory of what was said before doesn't affect the
conversation today":

  * 18:53 log: Bret reported "I got all the new interns set up." The intern-training
    plan sat in person_events (stored the night before; Rex asked about it at 23:56),
    yet the reply was "How many interns were there?" — a stranger's question. The lean
    reply prompt never read events at all: recall fired only on memory QUESTIONS.
  * The plan then STAYED open: resolution only ever happened when Rex asked and the
    human answered — a spontaneous outcome report changed nothing, so the same plan
    would come back later as "so did that happen?".
  * people.db `conversations` had ZERO rows ever — _end_session only fires on the
    idle timeout, and every real session ends in a spoken shutdown. "Last time you
    talked", nostalgia callbacks, and trends were reading an empty table.
"""

import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import config
from memory import database as db
from memory import text_match


def _utcnow_iso(days_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


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
        # rex.db is the REAL diary on a dev box — statement-time recall must not
        # read live data in a unit test.
        self._ep = mock.patch("memory.recall.search_episodes", return_value=[])
        self._ep.start()

    def tearDown(self):
        self._ep.stop()
        self._p.stop()
        self._tmp.cleanup()

    def _add_event(self, name, *, event_date=None, status="planned", followed_up=0,
                   outcome=None, hedged=0, mentioned_days_ago=1.0, notes=""):
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """INSERT INTO person_events
                   (person_id, event_name, event_date, event_notes, mentioned_at,
                    followed_up, status, outcome, hedged, updated_at)
                   VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, event_date, notes, _utcnow_iso(mentioned_days_ago),
                 followed_up, status, outcome, hedged, _utcnow_iso()),
            )


# ── The conservative matcher ────────────────────────────────────────────────────

class StrongOverlapTest(unittest.TestCase):
    def test_two_shared_stems_match(self):
        tokens = {"new", "interns", "set"}
        self.assertTrue(text_match.strong_overlap(tokens, "work and train new interns"))

    def test_one_distinctive_stem_matches_across_inflection(self):
        # "intern" (utterance) vs "interns" (stored) — the stemming gap that broke
        # cancel-matching on 'like falsum' shapes must not break this.
        self.assertTrue(text_match.strong_overlap({"intern", "orientation"},
                                                  "work and train new interns"))

    def test_one_short_shared_stem_does_not_match(self):
        # "new" alone must never tie an utterance to a plan.
        self.assertFalse(text_match.strong_overlap({"new", "shoes"},
                                                   "work and train new interns"))

    def test_no_tokens_no_match(self):
        self.assertFalse(text_match.strong_overlap(set(), "anything at all"))


# ── Statement-time known-context block ──────────────────────────────────────────

class KnownContextLinesTest(_TempDb):
    UTTERANCE = "I got all the new interns set up."

    def _lines(self, utterance=None, person_id=1):
        from memory import recall
        return recall.known_context_lines(person_id, utterance or self.UTTERANCE)

    def test_open_plan_surfaces_with_still_open_framing(self):
        self._add_event("work and train new interns", event_date=date.today().isoformat())
        lines = self._lines()
        joined = " ".join(lines)
        self.assertTrue(lines, "expected a known-context block")
        self.assertIn("KNOWN CONTEXT", joined)
        self.assertIn("work and train new interns", joined)
        self.assertIn("STILL OPEN", joined)

    def test_inflection_gap_bridged(self):
        self._add_event("work and train new interns")
        lines = self._lines("The intern orientation went well.")
        self.assertIn("work and train new interns", " ".join(lines))

    def test_completed_event_carries_its_outcome(self):
        self._add_event("work and train new interns", status="completed",
                        followed_up=1, outcome="all seven interns survived")
        joined = " ".join(self._lines("How are the interns doing this week?"))
        # (a question, but not a memory question — still known context)
        self.assertIn("already happened", joined)
        self.assertIn("all seven interns survived", joined)

    def test_canceled_event_says_called_off(self):
        self._add_event("trip to lake folsom", status="canceled", followed_up=1,
                        outcome="We're not going anymore.")
        joined = " ".join(self._lines("Lake Folsom is beautiful this time of year"))
        self.assertIn("CALLED IT OFF", joined)

    def test_hedged_open_plan_flagged_tentative(self):
        self._add_event("move the couch downstairs", hedged=1)
        joined = " ".join(self._lines("I still need to move that couch downstairs"))
        self.assertIn("tentative", joined)

    def test_single_midlength_noun_graze_stays_silent(self):
        # Deliberate conservatism: one shared 5-char noun ("couch") is NOT enough —
        # a wrong "you already know this" is worse than a missed connection.
        self._add_event("move the couch downstairs", hedged=1)
        self.assertEqual(self._lines("That couch is blocking the hallway again"), [])

    def test_unrelated_utterance_matches_nothing(self):
        self._add_event("work and train new interns")
        self.assertEqual(self._lines("it is really hot outside right now"), [])

    def test_memory_question_defers_to_rich_recall(self):
        self._add_event("work and train new interns")
        self.assertEqual(self._lines("do you remember the new interns I mentioned?"), [])

    def test_short_utterance_never_triggers(self):
        self._add_event("work and train new interns")
        self.assertEqual(self._lines("interns arrived"), [])

    def test_unknown_person_and_kill_switch(self):
        self._add_event("work and train new interns")
        self.assertEqual(self._lines(person_id=None), [])
        with mock.patch.object(config, "KNOWN_CONTEXT_RECALL_ENABLED", False,
                               create=True):
            self.assertEqual(self._lines(), [])

    def test_prior_session_summary_matches(self):
        from memory import conversations as conv_db
        conv_db.save_conversation(
            1, "They talked about training seven new interns at work.",
            emotion_tone="happy", topics="interns, work",
        )
        joined = " ".join(self._lines())
        self.assertIn("a past chat of yours covered", joined)
        self.assertIn("seven new interns", joined)

    def test_stale_event_outside_lookback_ignored(self):
        self._add_event("work and train new interns", mentioned_days_ago=90.0)
        self.assertEqual(self._lines(), [])


class LeanPersonLinesKnownContextTest(_TempDb):
    def test_reply_prompt_carries_the_block(self):
        self._add_event("work and train new interns", event_date=date.today().isoformat())
        from intelligence import lean_brain
        with mock.patch("memory.episodic_recall.recent_conversation_topics",
                        return_value=[]):
            lines = lean_brain._person_lines(1, "I got all the new interns set up.")
        joined = " ".join(lines)
        self.assertIn("KNOWN CONTEXT", joined)
        self.assertIn("work and train new interns", joined)

    def test_directive_path_empty_user_text_carries_none(self):
        self._add_event("work and train new interns")
        from intelligence import lean_brain
        with mock.patch("memory.episodic_recall.recent_conversation_topics",
                        return_value=[]):
            lines = lean_brain._person_lines(1, "")
        self.assertNotIn("KNOWN CONTEXT", " ".join(lines))


# ── Spontaneous outcome reports resolve the open plan ───────────────────────────

class CompleteMatchingEventsTest(_TempDb):
    def _open_events(self):
        from memory import events
        return events.get_open_events(1)

    def test_outcome_report_resolves_the_matching_plan(self):
        from memory import events
        self._add_event("work and train new interns",
                        event_date=date.today().isoformat())
        done = events.complete_matching_events(1, "I got all the new interns set up")
        self.assertEqual(len(done), 1)
        self.assertEqual(self._open_events(), [])
        with sqlite3.connect(self._path) as conn:
            status, followed, outcome = conn.execute(
                "SELECT status, followed_up, outcome FROM person_events "
                "WHERE event_name='work and train new interns'"
            ).fetchone()
        self.assertEqual(status, "completed")
        self.assertTrue(followed)
        self.assertIn("interns set up", outcome)

    def test_went_well_shape_resolves_through_stemming(self):
        from memory import events
        self._add_event("work and train new interns",
                        event_date=date.today().isoformat())
        done = events.complete_matching_events(1, "The intern orientation went well")
        self.assertEqual(len(done), 1)

    def test_no_single_open_event_fallback(self):
        # Unlike cancellation, a generic report must NEVER close an unrelated plan.
        from memory import events
        self._add_event("dentist appointment", event_date=date.today().isoformat())
        done = events.complete_matching_events(1, "the barbecue went really well")
        self.assertEqual(done, [])
        self.assertEqual(len(self._open_events()), 1)

    def test_future_dated_plan_cannot_have_happened(self):
        from memory import events
        self._add_event("work and train new interns",
                        event_date=(date.today() + timedelta(days=2)).isoformat())
        done = events.complete_matching_events(1, "I got all the new interns set up")
        self.assertEqual(done, [])

    def test_questions_and_non_reports_never_resolve(self):
        from memory import events
        self._add_event("work and train new interns",
                        event_date=date.today().isoformat())
        for text in (
            "did the interns get set up?",
            "tomorrow I have to train the new interns",
            "the interns seem nice",
        ):
            self.assertEqual(events.complete_matching_events(1, text), [], text)

    def test_kill_switch(self):
        from memory import events
        self._add_event("work and train new interns",
                        event_date=date.today().isoformat())
        with mock.patch.object(config, "EVENT_COMPLETION_RESOLUTION_ENABLED", False,
                               create=True):
            self.assertEqual(
                events.complete_matching_events(1, "I got all the new interns set up"),
                [],
            )


class EventCompletionShapeTest(unittest.TestCase):
    def test_outcome_report_shapes(self):
        from memory import events
        for text in (
            "The intern orientation went well",
            "it went pretty smoothly actually",
            "I got all the new interns set up",
            "we survived the big move",
            "that's all done now",
        ):
            self.assertTrue(events.looks_like_event_completion(text), text)

    def test_non_reports_rejected(self):
        from memory import events
        for text in (
            "did it go well?",
            "I hope it goes well",
            "tomorrow I have to train new interns",
        ):
            self.assertFalse(events.looks_like_event_completion(text), text)


# ── Session persistence at spoken shutdown ──────────────────────────────────────

class ShutdownPersistenceTest(_TempDb):
    """persist_session_memories_at_shutdown writes the people.db rows that a spoken
    shutdown historically skipped entirely (conversations had 0 rows ever)."""

    def setUp(self):
        super().setUp()
        from intelligence import interaction
        self.interaction = interaction
        self._saved_ids = set(interaction._session_person_ids)
        interaction._session_person_ids.clear()
        interaction._session_person_ids.add(1)
        self.addCleanup(self._restore_ids)
        self._transcript = [
            {"speaker": "Bret", "text": "I got all the new interns set up.",
             "learnable": True},
            {"speaker": "Rex", "text": "Nice — the tiny bureaucrats are online."},
            {"speaker": "Bret", "text": "Seven of them.", "learnable": True},
        ]
        self._tp = mock.patch.object(
            interaction.conv_memory, "get_session_transcript",
            return_value=list(self._transcript),
        )
        self._tp.start()
        self.addCleanup(self._tp.stop)
        # Arc fast-path: instant summary fields, zero LLM calls at shutdown.
        self._arc = mock.patch.object(
            interaction.topic_thread, "arc_persistence_fields",
            return_value=("Interns got set up; good mood.", "happy", "interns, work"),
        )
        self._arc.start()
        self.addCleanup(self._arc.stop)
        self._consolidate = mock.patch.object(
            interaction, "_consolidate_session_memories", return_value=True,
        )
        self._mock_consolidate = self._consolidate.start()
        self.addCleanup(self._consolidate.stop)

    def _restore_ids(self):
        self.interaction._session_person_ids.clear()
        self.interaction._session_person_ids.update(self._saved_ids)

    def _conversation_rows(self):
        with sqlite3.connect(self._path) as conn:
            return conn.execute(
                "SELECT summary, topics FROM conversations WHERE person_id=1"
            ).fetchall()

    def test_shutdown_persists_summary_and_visit_without_consolidation(self):
        self.interaction.persist_session_memories_at_shutdown()
        rows = self._conversation_rows()
        self.assertEqual(len(rows), 1)
        self.assertIn("Interns got set up", rows[0][0])
        self.assertEqual(rows[0][1], "interns, work")
        self._mock_consolidate.assert_not_called()
        with sqlite3.connect(self._path) as conn:
            visits = conn.execute(
                "SELECT visit_count FROM people WHERE id=1"
            ).fetchone()[0]
        self.assertGreaterEqual(int(visits or 0), 1)

    def test_idle_timeout_profile_still_consolidates(self):
        self.interaction._end_session()
        self.assertEqual(len(self._conversation_rows()), 1)
        self._mock_consolidate.assert_called_once()

    def test_substance_gate_blocks_command_only_sessions(self):
        with mock.patch.object(
            self.interaction.conv_memory, "get_session_transcript",
            return_value=[
                {"speaker": "Bret", "text": "shut down", "learnable": True},
                {"speaker": "Rex", "text": "Cockpit going dark."},
            ],
        ):
            self.interaction.persist_session_memories_at_shutdown()
        self.assertEqual(self._conversation_rows(), [])

    def test_kill_switch(self):
        with mock.patch.object(config, "SESSION_SUMMARY_ON_SHUTDOWN_ENABLED", False,
                               create=True):
            self.interaction.persist_session_memories_at_shutdown()
        self.assertEqual(self._conversation_rows(), [])


if __name__ == "__main__":
    unittest.main()
