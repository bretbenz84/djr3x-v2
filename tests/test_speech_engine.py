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


if __name__ == "__main__":
    unittest.main()
