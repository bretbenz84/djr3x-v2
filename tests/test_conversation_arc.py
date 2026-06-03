"""
Tests for Bet 1 arc-memory: a running summary of the live conversation, folded
into intelligence/topic_thread.py and fed back into the system prompt.

The summary backend is configurable (config.CONVERSATION_ARC_BACKEND): "openai"
(gpt-4o-mini, default, rich 5-field schema) or "local" (qwen2.5:1.5b sidecar,
3-field factual schema). These tests are backend-agnostic — they mock the dispatch
seam `_arc_generate` (and `_arc_enabled`) so NO real Ollama/OpenAI call is ever
made (the suite runs with a live OpenAI key, so this matters).

Covers: fresh-window fold/update, the callback+anti-repeat directive text,
retain-on-error, empty/echo/runaway-loop rejection, markdown/dedup sanitize,
speaker→role normalization, rich-vs-local schema, the test-runner safety gate,
backend-availability gating, cursor bookkeeping, clear(), snapshot() back-compat,
the real background-thread trigger, and the OpenAI helper.
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

    def _run_refresh(self, transcript, *, returns=None, raises=None):
        """Run _arc_refresh_core with the arc force-enabled and the backend call
        mocked (no real network), against the given transcript. Returns (ran, gen)."""
        from intelligence import topic_thread as tt
        gen = mock.Mock(side_effect=raises) if raises is not None else mock.Mock(return_value=returns)
        with (
            mock.patch.object(tt, "_arc_enabled", return_value=True),
            mock.patch.object(tt, "_arc_generate", gen),
            mock.patch("memory.conversations.get_session_transcript", return_value=transcript),
        ):
            ran = tt._arc_refresh_core()
        return ran, gen


class ArcRefreshTest(_ArcTestBase):
    def test_fold_new_exchange_into_summary(self):
        from intelligence import topic_thread as tt
        canned = (
            "Topics: astrophotography\n"
            "Shared: shoots deep-sky from his backyard\n"
            "Mood: relaxed, enthusiastic\n"
            "Landed/flopped: deep-sky imaging landed; small talk flopped\n"
            "Open threads: which galaxy he shoots next"
        )
        ran, gen = self._run_refresh(
            _transcript(("Rex", "What are you into?"), ("Bret", "astrophotography")),
            returns=canned,
        )
        self.assertTrue(ran)
        self.assertTrue(gen.called)
        self.assertEqual(tt.arc_summary(), canned)
        directive = tt.build_arc_directive()
        self.assertIn("astrophotography", directive)
        self.assertIn("Conversation arc", directive)

    def test_directive_carries_callback_and_no_repeat_guard(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "hi")), returns="Topics: x\nMood: ok")
        directive = tt.build_arc_directive().lower()
        self.assertIn("call back", directive)   # callbacks enabled
        self.assertIn("never force", directive)  # ...but not forced
        self.assertIn("re-ask", directive)       # anti-repeat steer

    def test_retains_previous_summary_on_backend_error(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "first")), returns="Topics: seeded")
        self.assertEqual(tt.arc_summary(), "Topics: seeded")
        # The backend (Ollama or OpenAI) falls over — old summary must survive, no raise.
        ran, _ = self._run_refresh(
            _transcript(("Bret", "first"), ("Rex", "x"), ("Bret", "second")),
            raises=RuntimeError("backend down"),
        )
        self.assertFalse(ran)
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

    def test_empty_model_output_retains_previous_summary(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "first")), returns="Topics: seeded")
        ran, _ = self._run_refresh(
            _transcript(("Bret", "first"), ("Bret", "second")), returns="   "
        )
        self.assertFalse(ran)
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

    def test_rejects_transcript_echo_output(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "first")), returns="Topics: seeded")
        # The model parrots the dialogue back — reject it, keep prior.
        echo = "User: I love astrophotography\nRex: Cool, which galaxy?"
        ran, _ = self._run_refresh(
            _transcript(("Bret", "first"), ("Rex", "hi"), ("Bret", "second")), returns=echo
        )
        self.assertFalse(ran)
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

    def test_rejects_runaway_repetition_loop(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "first")), returns="Topics: seeded")
        loop = "Topics: " + ", ".join(["motivation"] * 20)
        ran, _ = self._run_refresh(
            _transcript(("Bret", "first"), ("Rex", "x"), ("Bret", "y")), returns=loop
        )
        self.assertFalse(ran)
        self.assertEqual(tt.arc_summary(), "Topics: seeded")

    def test_sanitizes_markdown_and_dedups_lists(self):
        from intelligence import topic_thread as tt
        messy = "**Topics:** astro, astro, robots\nShared: testing the program"
        self._run_refresh(_transcript(("Bret", "hi")), returns=messy)
        self.assertEqual(
            tt.arc_summary(), "Topics: astro, robots\nShared: testing the program"
        )

    def test_refresh_noop_when_disabled(self):
        from intelligence import topic_thread as tt
        gen = mock.Mock(return_value="Topics: x")
        with (
            mock.patch.object(tt, "_arc_enabled", return_value=False),
            mock.patch.object(tt, "_arc_generate", gen),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "hi")),
            ),
        ):
            self.assertFalse(tt._arc_refresh_core())
            gen.assert_not_called()

    def test_no_new_lines_skips_the_call(self):
        from intelligence import topic_thread as tt
        transcript = _transcript(("Bret", "hello"))
        gen = mock.Mock(return_value="Topics: a")
        with (
            mock.patch.object(tt, "_arc_enabled", return_value=True),
            mock.patch.object(tt, "_arc_generate", gen),
            mock.patch("memory.conversations.get_session_transcript", return_value=transcript),
        ):
            self.assertTrue(tt._arc_refresh_core())   # folds the one new line
            self.assertEqual(gen.call_count, 1)
            self.assertFalse(tt._arc_refresh_core())  # nothing new -> no second call
            self.assertEqual(gen.call_count, 1)

    def test_transcript_reset_resummarizes_from_scratch(self):
        from intelligence import topic_thread as tt
        self._run_refresh(
            _transcript(("Bret", "1"), ("Rex", "2"), ("Bret", "3")), returns="Topics: a"
        )
        # A shorter transcript means a reset: cursor was 3, now len 1 -> re-fold.
        ran, gen = self._run_refresh(_transcript(("Bret", "fresh")), returns="Topics: b")
        self.assertTrue(ran)
        self.assertTrue(gen.called)
        self.assertEqual(tt.arc_summary(), "Topics: b")


class ArcSchemaAndRenderTest(_ArcTestBase):
    def test_render_normalizes_speakers_to_roles(self):
        # The person's name must never reach the summarizer (it was landing in
        # "Topics:" as the speaker name). Everyone but Rex is "User".
        from intelligence import topic_thread as tt
        rendered = tt._render_transcript_lines(
            _transcript(("Bret Benziger", "hi"), ("Rex", "yo"), ("unknown_voice_1", "hm"))
        )
        self.assertIn("User: hi", rendered)
        self.assertIn("Rex: yo", rendered)
        self.assertIn("User: hm", rendered)
        self.assertNotIn("Bret", rendered)

    def test_rich_schema_has_mood_local_schema_does_not(self):
        from intelligence import topic_thread as tt
        rich = tt._build_arc_prompt("User: hi", rich=True)
        local = tt._build_arc_prompt("User: hi", rich=False)
        self.assertIn("Mood:", rich)
        self.assertIn("Landed/flopped:", rich)
        self.assertNotIn("Mood:", local)
        self.assertIn("Topics:", local)


class ArcGateTest(_ArcTestBase):
    def test_disabled_under_test_runner_by_default(self):
        # Safety: with unittest loaded and no opt-in, the arc must NOT fire from
        # note_user_turn (it would make a real cloud call with the live key).
        from intelligence import topic_thread as tt
        self.assertFalse(tt._arc_enabled())

    def test_backend_available_false_when_config_flag_off(self):
        import config
        from intelligence import topic_thread as tt
        with mock.patch.object(config, "CONVERSATION_ARC_ENABLED", False):
            self.assertFalse(tt._arc_backend_available())

    def test_openai_backend_available_by_default(self):
        from intelligence import topic_thread as tt
        self.assertTrue(tt._arc_backend_available())

    def test_local_backend_requires_local_llm(self):
        import config
        from intelligence import topic_thread as tt
        with mock.patch.object(config, "CONVERSATION_ARC_BACKEND", "local"):
            with mock.patch("intelligence.local_llm.enabled", return_value=False):
                self.assertFalse(tt._arc_backend_available())
            with mock.patch("intelligence.local_llm.enabled", return_value=True):
                self.assertTrue(tt._arc_backend_available())

    def test_flag_off_hides_an_existing_summary(self):
        import config
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "hi")), returns="Topics: seeded")
        self.assertNotEqual(tt.build_arc_directive(), "")
        with mock.patch.object(config, "CONVERSATION_ARC_ENABLED", False):
            self.assertEqual(tt.build_arc_directive(), "")


class ArcClearTest(_ArcTestBase):
    def test_clear_wipes_summary_and_directive(self):
        from intelligence import topic_thread as tt
        self._run_refresh(_transcript(("Bret", "x")), returns="Topics: seeded")
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
        with (
            mock.patch.object(tt, "_arc_enabled", return_value=True),
            mock.patch.object(tt, "_arc_generate", return_value=canned),
            mock.patch(
                "memory.conversations.get_session_transcript",
                return_value=_transcript(("Bret", "I love astrophotography")),
            ),
        ):
            tt.note_user_turn("I love astrophotography", 1)
            if tt._arc_thread is not None:
                tt._arc_thread.join(timeout=3.0)
        self.assertEqual(tt.arc_summary(), canned)


class ArcOpenAIHelperTest(unittest.TestCase):
    def test_summarize_conversation_arc_calls_chat_and_returns_text(self):
        from intelligence import llm
        fake = mock.Mock()
        fake.choices = [mock.Mock(message=mock.Mock(content="Topics: x\nMood: ok"))]
        with mock.patch.object(
            llm._client.chat.completions, "create", return_value=fake
        ) as create:
            out = llm.summarize_conversation_arc(
                "PROMPT", system="SYS", max_tokens=50, timeout_secs=5.0
            )
        self.assertEqual(out, "Topics: x\nMood: ok")
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertEqual(kwargs["messages"][0]["content"], "SYS")
        self.assertEqual(kwargs["messages"][1]["content"], "PROMPT")
        self.assertEqual(kwargs["max_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
