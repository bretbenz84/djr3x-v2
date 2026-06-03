"""
Tests for Bet 1 arc-memory: a running, local-LLM-maintained summary of the live
conversation, folded into intelligence/topic_thread.py.

Covers:
  - the summarize/fold step updates the running summary
  - the prompt directive carries the callback + anti-repeat guard
  - graceful degradation: a local-LLM failure retains the previous summary
  - the kill switch (config flag) and local-LLM-unavailable both make it inert
  - cursor bookkeeping (no redundant call when nothing is new; reset on a
    shrunken/cleared transcript)
  - clear() wipes the arc
  - note_user_turn triggers a real background refresh
"""

from __future__ import annotations

import unittest
from unittest import mock


def _transcript(*pairs):
    return [{"speaker": s, "text": t} for s, t in pairs]


class _ArcTestBase(unittest.TestCase):
    def setUp(self):
        from intelligence import topic_thread
        topic_thread.clear()

    def tearDown(self):
        from intelligence import topic_thread
        topic_thread.clear()


class ArcRefreshTest(_ArcTestBase):
    def test_fold_new_exchange_into_summary(self):
        from intelligence import topic_thread as tt
        canned = (
            "Topics: astrophotography\nLanded: deep-sky obsession\nFlopped: -\n"
            "Mood: engaged, enthusiastic\nOpen threads: which galaxy he shoots next"
        )
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value=canned) as gen,
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(
                    ("Rex", "What are you obsessed with?"),
                    ("Bret", "astrophotography"),
                ),
            ),
        ):
            ran = tt._arc_refresh_core()
        self.assertTrue(ran)
        self.assertTrue(gen.called)
        self.assertEqual(tt.arc_summary(), canned)
        directive = tt.build_arc_directive()
        self.assertIn("astrophotography", directive)
        self.assertIn("Conversation arc", directive)

    def test_directive_carries_callback_and_no_repeat_guard(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch(
                "intelligence.local_llm.generate", return_value="Topics: x\nMood: ok"
            ),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "hi")),
            ),
        ):
            tt._arc_refresh_core()
        directive = tt.build_arc_directive().lower()
        self.assertIn("call back", directive)   # callbacks are enabled
        self.assertIn("never force", directive)  # ...but not forced
        self.assertIn("re-ask", directive)       # anti-repeat steer

    def test_retains_previous_summary_on_local_llm_error(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: seeded"),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "first")),
            ),
        ):
            tt._arc_refresh_core()
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

        # Next refresh: the local LLM falls over. The old summary must survive and
        # no exception may escape.
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch(
                "intelligence.local_llm.generate",
                side_effect=RuntimeError("ollama down"),
            ),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(
                    ("Bret", "first"), ("Rex", "huh"), ("Bret", "second")
                ),
            ),
        ):
            ran = tt._arc_refresh_core()
        self.assertFalse(ran)
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

    def test_empty_model_output_retains_previous_summary(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="seeded"),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "first")),
            ),
        ):
            tt._arc_refresh_core()
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="   "),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "first"), ("Bret", "second")),
            ),
        ):
            self.assertFalse(tt._arc_refresh_core())
        self.assertEqual(tt.arc_summary(), "seeded")

    def test_no_new_lines_skips_the_call(self):
        from intelligence import topic_thread as tt
        transcript = _transcript(("Bret", "hello"))
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: a") as gen,
            mock.patch(
                "memory.conversations.get_session_transcript", return_value=transcript
            ),
        ):
            self.assertTrue(tt._arc_refresh_core())   # folds the one new line
            self.assertEqual(gen.call_count, 1)
            self.assertFalse(tt._arc_refresh_core())  # nothing new -> no second call
            self.assertEqual(gen.call_count, 1)

    def test_transcript_reset_resummarizes_from_scratch(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: a"),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "1"), ("Rex", "2"), ("Bret", "3")),
            ),
        ):
            tt._arc_refresh_core()
        # A shorter transcript means it was cleared: cursor was 3, now len 1, so the
        # arc must reset and re-fold rather than silently fold nothing.
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: b") as gen,
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "fresh")),
            ),
        ):
            ran = tt._arc_refresh_core()
        self.assertTrue(ran)
        self.assertTrue(gen.called)
        self.assertEqual(tt.arc_summary(), "Topics: b")


class ArcKillSwitchTest(_ArcTestBase):
    def test_inert_when_local_llm_unavailable(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=False),
            mock.patch("intelligence.local_llm.generate") as gen,
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "hi")),
            ),
        ):
            self.assertFalse(tt._arc_refresh_core())
            gen.assert_not_called()
        self.assertEqual(tt.build_arc_directive(), "")

    def test_inert_when_config_flag_off(self):
        import config
        from intelligence import topic_thread as tt
        with (
            mock.patch.object(config, "CONVERSATION_ARC_ENABLED", False),
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate") as gen,
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "hi")),
            ),
        ):
            self.assertFalse(tt._arc_refresh_core())
            gen.assert_not_called()
            self.assertEqual(tt.build_arc_directive(), "")

    def test_flag_off_hides_an_existing_summary(self):
        import config
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: seeded"),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "hi")),
            ),
        ):
            tt._arc_refresh_core()
        self.assertNotEqual(tt.build_arc_directive(), "")
        with mock.patch.object(config, "CONVERSATION_ARC_ENABLED", False):
            self.assertEqual(tt.build_arc_directive(), "")


class ArcClearTest(_ArcTestBase):
    def test_clear_wipes_summary_and_directive(self):
        from intelligence import topic_thread as tt
        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value="Topics: seeded"),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "x")),
            ),
        ):
            tt._arc_refresh_core()
        self.assertNotEqual(tt.arc_summary(), "")
        tt.clear()
        self.assertEqual(tt.arc_summary(), "")
        self.assertEqual(tt.build_arc_directive(), "")


class ArcSnapshotTest(_ArcTestBase):
    def test_snapshot_preserves_label_and_unresolved_and_adds_arc(self):
        # snapshot() has three live consumers in interaction.py that read
        # label / unresolved_question — adding arc_summary must not disturb them.
        from intelligence import topic_thread as tt
        tt.note_assistant_turn("What are you obsessed with?")
        tt.note_user_turn("astrophotography", 1, answered_question={"question_key": "obsession"})
        snap = tt.snapshot()
        self.assertIsNotNone(snap)
        self.assertIn("label", snap)
        self.assertIn("unresolved_question", snap)
        self.assertIn("arc_summary", snap)


class ArcTriggerThreadTest(_ArcTestBase):
    def test_note_user_turn_triggers_background_refresh(self):
        from intelligence import topic_thread as tt
        canned = "Topics: triggered\nMood: fine"
        seen = []

        def _fake_generate(*a, **k):
            seen.append(True)
            return canned

        with (
            mock.patch("intelligence.local_llm.enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", side_effect=_fake_generate),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "I love astrophotography")),
            ),
        ):
            tt.note_user_turn("I love astrophotography", 1)
            # The refresh runs on a daemon thread — wait for it inside the mocks.
            if tt._arc_thread is not None:
                tt._arc_thread.join(timeout=3.0)
        self.assertTrue(seen)
        self.assertEqual(tt.arc_summary(), canned)


if __name__ == "__main__":
    unittest.main()
