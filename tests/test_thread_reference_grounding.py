"""
Two context fixes from the 2026-08-01 12:22/12:28 field logs:

1. Proactive speech (speech_engine paths) must land in conv_memory's session
   transcript — the reply model reads it, and a greeting it can't see makes the
   next human turn contextless (asked "How did watching The Odyssey go?", heard
   "I loved the movie", replied "Which one?").

2. "I don't remember saying that" right after Rex referenced a diary open
   thread is answered by QUOTING the source (episode summary + when), not the
   stock "Nothing recent to discard" line — while an explicit "forget that"
   still reaches the real discard flow.
"""

import time
import unittest
from unittest import mock

import numpy as np  # noqa: F401  (keeps import parity with sibling audio tests)

import config
from intelligence import consciousness, speech_engine
from intelligence import interaction


class ProactiveTranscriptTest(unittest.TestCase):
    def test_presence_reaction_is_written_to_the_transcript(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            def p(name, **kw):
                stack.enter_context(mock.patch.object(consciousness, name, **kw))

            p("_observe_governor_candidate", return_value="cg-test")
            p("_mark_governor_candidate", return_value=None)
            p("_claim_proactive_purpose", return_value="tok")
            p("_release_proactive_purpose", return_value=None)
            p("_proactive_purpose_current", return_value=True)
            p("_can_proactive_speak", return_value=True)
            p("note_rex_utterance", return_value=None)
            p("_record_proactive_question", return_value=None)
            p("_utterance_expects_reply", return_value=False)
            p("_presence_line_counts_as_greeting", return_value=False)
            stack.enter_context(mock.patch.object(
                speech_engine.config, "PRESENCE_REACTION_DELAY_SECS", 0.0, create=True))
            stack.enter_context(
                mock.patch("audio.speech_queue.enqueue", return_value=mock.Mock()))
            transcript = stack.enter_context(
                mock.patch("memory.conversations.add_to_transcript"))

            self.assertTrue(speech_engine.generate_and_speak_presence(
                "prompt", "greeting", 1, direct_text="How did the Odyssey go?"))
            deadline = time.monotonic() + 3.0
            while not transcript.called and time.monotonic() < deadline:
                time.sleep(0.05)
        transcript.assert_called_once_with("Rex", "How did the Odyssey go?")


class ThreadReferenceGroundingTest(unittest.TestCase):
    def setUp(self):
        interaction._last_thread_reference = {
            "episode_id": 468,
            "thread": "what the wheels in the parking lot turned into",
            "when": "3 days ago",
            "at": time.monotonic(),
        }
        self.addCleanup(lambda: setattr(interaction, "_last_thread_reference", None))

    def _row(self, summary):
        return {"summary": summary}

    def test_confusion_quotes_the_diary_source(self):
        with mock.patch("memory.rex_db.fetchone",
                        return_value=self._row("Bret mentioned wheels in the parking lot.")), \
             mock.patch.object(interaction, "_speak_blocking") as speak:
            resp = interaction._execute_memory_boundary_command(
                1, utterance="I don't remember saying what that was.")
        speak.assert_called_once()
        self.assertIn("Bret mentioned wheels in the parking lot.", resp)
        self.assertIn("3 days ago", resp)
        self.assertIn("wheels in the parking lot turned into", resp)
        self.assertIsNone(interaction._last_thread_reference)  # one grounding per ref

    def test_explicit_forget_still_reaches_the_discard_flow(self):
        with mock.patch.object(interaction, "_speak_blocking") as speak, \
             mock.patch.object(interaction, "_recent_memory_candidates", []):
            resp = interaction._execute_memory_boundary_command(
                1, utterance="Forget that, please.")
        self.assertIn("Nothing recent to discard", resp)
        speak.assert_called_once()

    def test_stale_reference_falls_through(self):
        interaction._last_thread_reference["at"] = time.monotonic() - 600.0
        with mock.patch.object(interaction, "_speak_blocking"), \
             mock.patch.object(interaction, "_recent_memory_candidates", []):
            resp = interaction._execute_memory_boundary_command(
                1, utterance="I have no idea what you mean.")
        self.assertIn("Nothing recent to discard", resp)

    def test_missing_summary_still_names_the_thread(self):
        with mock.patch("memory.rex_db.fetchone", return_value=None), \
             mock.patch.object(interaction, "_speak_blocking"):
            resp = interaction._execute_memory_boundary_command(
                1, utterance="What are you talking about?")
        self.assertIn("wheels in the parking lot turned into", resp)
        self.assertIn("misheard", resp)


if __name__ == "__main__":
    unittest.main()
