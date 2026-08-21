"""A session that outruns the shutdown budget must degrade, not vanish.

Field 2026-08-20 20:33 (log L7466-L7515): `persist_session_memories_at_shutdown`
runs `_end_session` on a DAEMON thread and joins for 10 s. The join expired while
the summary LLM call was in flight during a network outage; teardown continued and
the process exit killed the thread mid-call. An 8-minute, 20-turn conversation with
person_id=1 left no `conversations` row at all — confirmed in the DB, not inferred.
That table is what "last time you talked", nostalgia callbacks and cross-session
trends read from, so the loss is invisible until weeks later.

The fix is not a longer budget (power-off must stay fast) — it is a deterministic
LLM-free row written for whoever the worker did not already claim.
"""

import threading
import time
import unittest
from unittest import mock

import config
from intelligence import interaction


TRANSCRIPT = [
    {"speaker": "Bret Benziger", "text": "We're going to Georgia in a few weeks."},
    {"speaker": "Rex", "text": "A pilgrimage powered by vibes."},
    {"speaker": "Bret Benziger", "text": "My dog Max is coming too."},
    {"speaker": "Rex", "text": "Max remains on station."},
]


class ShutdownSalvageTests(unittest.TestCase):
    def setUp(self):
        interaction._session_person_ids.clear()
        interaction._session_person_ids.add(1)
        with interaction._session_row_lock:
            interaction._session_rows_written.clear()
        self.addCleanup(interaction._session_person_ids.clear)

    def _run_shutdown(self, end_session_impl):
        saved = []
        with (
            mock.patch.object(config, "SESSION_SUMMARY_ON_SHUTDOWN_ENABLED", True),
            mock.patch.object(config, "SESSION_SUMMARY_MIN_HUMAN_TURNS", 2),
            mock.patch.object(config, "SESSION_SUMMARY_SHUTDOWN_TIMEOUT_SECS", 0.3),
            mock.patch.object(interaction.conv_memory, "get_session_transcript",
                              return_value=list(TRANSCRIPT)),
            mock.patch.object(interaction.conv_memory, "save_conversation",
                              side_effect=lambda pid, summary, **kw: saved.append((pid, summary, kw))),
            mock.patch.object(interaction, "_filter_forgotten_transcript",
                              side_effect=lambda t, pid: t),
            mock.patch.object(interaction, "_end_session", side_effect=end_session_impl),
        ):
            interaction.persist_session_memories_at_shutdown()
        return saved

    def test_worker_overrunning_its_budget_still_leaves_a_row(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def slow_end_session(**kw):
            release.wait(5.0)            # stands in for the wedged LLM call

        saved = self._run_shutdown(slow_end_session)

        self.assertEqual(len(saved), 1, "the session vanished — no row written")
        person_id, summary, _ = saved[0]
        self.assertEqual(person_id, 1)
        self.assertIn("unsummarized", summary,
                      "salvaged rows must be labelled, not passed off as a real recap")
        # The actual content has to survive, or the row is worthless.
        self.assertIn("Georgia", summary)
        self.assertIn("Max", summary)

    def test_no_duplicate_row_when_the_worker_finishes_in_time(self):
        def fast_end_session(**kw):
            interaction.conv_memory.save_conversation(
                1, "Talked about Georgia and Max.", emotion_tone="warm", topics="travel",
            )
            with interaction._session_row_lock:
                interaction._session_rows_written.add(1)

        saved = self._run_shutdown(fast_end_session)
        self.assertEqual(len(saved), 1, "salvage double-wrote a session that succeeded")
        self.assertNotIn("unsummarized", saved[0][1])

    def test_worker_that_claims_late_cannot_produce_a_second_row(self):
        """The worker is still alive when the join expires; whichever writer claims
        the person first wins and the other must skip."""
        started = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def racing_end_session(**kw):
            with interaction._session_row_lock:
                interaction._session_rows_written.add(1)   # claim, then be slow
            started.set()
            release.wait(5.0)

        saved = self._run_shutdown(racing_end_session)
        self.assertTrue(started.is_set())
        self.assertEqual(saved, [], "salvage wrote over a claim the worker already made")

    def test_substance_gate_still_skips_trivial_visits(self):
        interaction._session_person_ids.clear()
        interaction._session_person_ids.add(1)
        saved = []
        with (
            mock.patch.object(config, "SESSION_SUMMARY_MIN_HUMAN_TURNS", 5),
            mock.patch.object(config, "SESSION_SUMMARY_SHUTDOWN_TIMEOUT_SECS", 0.2),
            mock.patch.object(interaction.conv_memory, "get_session_transcript",
                              return_value=list(TRANSCRIPT)),
            mock.patch.object(interaction.conv_memory, "save_conversation",
                              side_effect=lambda *a, **k: saved.append(a)),
            mock.patch.object(interaction, "_end_session"),
        ):
            interaction.persist_session_memories_at_shutdown()
        self.assertEqual(saved, [], "a shut-down-only visit should earn no row")


class PlainRecapTests(unittest.TestCase):
    def test_recap_is_capped_and_marked(self):
        long_transcript = [
            {"speaker": "Bret", "text": "word " * 60} for _ in range(20)
        ]
        with mock.patch.object(config, "SESSION_SUMMARY_SALVAGE_MAX_CHARS", 300):
            out = interaction._plain_transcript_recap(long_transcript)
        self.assertTrue(out.startswith("[unsummarized"))
        self.assertLess(len(out), 420, "cap not applied")
        self.assertTrue(out.endswith("..."), "truncation not signalled")

    def test_empty_transcript_yields_nothing(self):
        self.assertEqual(interaction._plain_transcript_recap([]), "")
        self.assertEqual(
            interaction._plain_transcript_recap([{"speaker": "Bret", "text": "  "}]), "")

    def test_speakers_are_preserved(self):
        out = interaction._plain_transcript_recap(TRANSCRIPT)
        self.assertIn("Bret Benziger:", out)
        self.assertIn("Rex:", out)


if __name__ == "__main__":
    unittest.main()
