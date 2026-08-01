"""
Guards for the speak-engine extraction (intelligence/speech_engine.py).

These pin the contracts that the extraction must preserve and that the full suite
didn't already cover — most importantly the governor metadata key (a string literal
that a refactor pass once corrupted) and the consciousness re-export shims.
"""

import unittest
from unittest import mock

from intelligence import consciousness, speech_engine, action_governor


class ShimIdentityTest(unittest.TestCase):
    """consciousness re-exports each engine function as a shim alias, so its ~100 call
    sites and the test patch targets keep resolving to the moved implementation."""

    def test_shims_point_at_speech_engine(self):
        pairs = [
            ("_generate_and_speak", "generate_and_speak"),
            ("_speak_async", "speak_async"),
            ("_can_proactive_speak", "can_proactive_speak"),
            ("_generate_and_speak_presence", "generate_and_speak_presence"),
            ("_observe_governor_candidate", "observe_governor_candidate"),
            ("_governor_speech_metadata", "governor_speech_metadata"),
        ]
        for shim, real in pairs:
            self.assertIs(getattr(consciousness, shim), getattr(speech_engine, real), shim)

    def test_note_rex_utterance_stayed_in_consciousness(self):
        # The cross-cutting bookkeeping (which rebinds shared globals) must NOT have moved.
        self.assertFalse(hasattr(speech_engine, "note_rex_utterance"))
        self.assertTrue(callable(consciousness.note_rex_utterance))


class GovernorMetadataKeyTest(unittest.TestCase):
    """Regression guard: governor_speech_metadata() must emit the key the action governor
    actually reads. A refactor once mangled it to '_c._can_proactive_speak', silently
    breaking the 'can_proactive_speak_false' rejection reason (no test caught it)."""

    def test_metadata_uses_the_key_the_governor_reads(self):
        meta = speech_engine.governor_speech_metadata()
        self.assertIn("can_proactive_speak", meta)
        self.assertNotIn("_c._can_proactive_speak", meta)  # the corrupted form

    def test_action_governor_reads_that_exact_key(self):
        # Pin both sides of the contract together so they can't drift apart.
        import inspect
        gov_src = inspect.getsource(action_governor)
        self.assertIn('metadata.get("can_proactive_speak")', gov_src)


class PatchTransparencyTest(unittest.TestCase):
    """The intra-engine calls route through the consciousness shim ON PURPOSE, so patching
    consciousness._can_proactive_speak overrides the engine's internal gate exactly as it
    did before the extraction. This is the property the routing exists to preserve."""

    def test_internal_gate_respects_a_consciousness_patch(self):
        # speak_async's internal _c._can_proactive_speak() must see the consciousness patch.
        with mock.patch.object(consciousness, "_can_proactive_speak", return_value=False), \
             mock.patch("audio.speech_queue.enqueue") as enq:
            spoke = speech_engine.speak_async("hello", governed=False)
        self.assertFalse(spoke)        # gate said no → no speech
        enq.assert_not_called()


class ReactiveBypassTest(unittest.TestCase):
    """reactive=True (wave-back) must break through the 'awaiting a reply' gate so a wave
    is acknowledged even right after Rex asked a question — while a normal proactive line
    stays blocked. Regression for the logged failure where every wave landed during Rex's
    await window and was silently dropped."""

    def _allow_everything_except_await(self, stack):
        from contextlib import ExitStack  # noqa: F401 (documents intent)

        def p(target, **kw):
            stack.enter_context(mock.patch(target, **kw))

        # Every non-await gate set to "allow".
        p("intelligence.consciousness._can_speak", return_value=True)
        p("intelligence.consciousness.is_waiting_for_response", return_value=True)
        p("intelligence.interaction.tell_about_flow_active", return_value=False)
        p("intelligence.interaction.onboarding_flow_active", return_value=False)
        p("intelligence.callback_engine.recently_heavy", return_value=False)
        p("features.dj.is_playing", return_value=False)
        p("features.games.suppresses_conversation_interruptions", return_value=False)
        p("audio.speech_queue.is_speaking", return_value=False)
        p("audio.output_gate.is_busy", return_value=False)
        stack.enter_context(
            mock.patch.object(speech_engine._situation_assessor, "is_interaction_busy",
                              return_value=False)
        )
        stack.enter_context(
            mock.patch.object(speech_engine.state_module, "get_state",
                              return_value=speech_engine.State.IDLE)
        )
        stack.enter_context(
            mock.patch.object(consciousness._proactive_speech_pending, "is_set",
                              return_value=False)
        )

    def test_reactive_bypasses_awaiting_reply_but_normal_does_not(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            self._allow_everything_except_await(stack)
            # Awaiting a reply → a normal proactive line is blocked …
            self.assertFalse(speech_engine.can_proactive_speak())
            # … but a reactive one (wave-back) breaks through.
            self.assertTrue(speech_engine.can_proactive_speak(reactive=True))


class PresenceLockWaitTest(unittest.TestCase):
    """A presence reaction must WAIT for the shared reaction lock (bounded), not
    try-once-and-die. Field 2026-07-31 20:10:29: the startup greeting hit the lock
    while a low-value donut room-change ask slept its 2s pre-speak delay inside it;
    the greeting was skipped silently, the donut ask was then dropped as superseded
    (the greeting had taken its purpose claim) — and Rex greeted nobody."""

    def _patched(self, stack):
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
        stack.enter_context(mock.patch.object(
            speech_engine.config, "PRESENCE_SPEAK_GRACE_SECS", 5.0, create=True))
        done = mock.Mock()
        enq = stack.enter_context(
            mock.patch("audio.speech_queue.enqueue", return_value=done))
        return enq

    def test_greeting_waits_out_a_busy_lock_instead_of_dying(self):
        import time
        from contextlib import ExitStack

        with ExitStack() as stack:
            enq = self._patched(stack)
            consciousness._presence_reaction_lock.acquire()   # a loser holds it
            try:
                self.assertTrue(speech_engine.generate_and_speak_presence(
                    "prompt", "greeting under test", 1,
                    direct_text="hello there"))
                time.sleep(0.4)                # well past the old try-once skip
                enq.assert_not_called()        # still waiting, not dropped
            finally:
                consciousness._presence_reaction_lock.release()
            deadline = time.monotonic() + 3.0
            while not enq.called and time.monotonic() < deadline:
                time.sleep(0.05)
        enq.assert_called_once()               # spoke once the lock freed
        self.assertEqual(enq.call_args[0][0], "hello there")


if __name__ == "__main__":
    unittest.main()
