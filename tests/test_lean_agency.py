"""Lean agency (Phase 1): under LEAN_BRAIN_ENABLED the old silence-fill proactive is suppressed
and replaced by ONE motivated impulse that Rex chooses to make — or passes on. Covers the
governor suppression gate and the impulse's PASS/act parsing (mocked — no API)."""

import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import lean_brain
from intelligence.action_governor import ActionGovernor, CandidateMove


class GovernorSuppressionTest(unittest.TestCase):
    def _rejected(self, purpose, lean):
        prev = config.LEAN_BRAIN_ENABLED
        try:
            config.LEAN_BRAIN_ENABLED = lean
            g = ActionGovernor()
            return g._score(CandidateMove(source="t", purpose=purpose, priority=50, label=purpose)).rejected
        finally:
            config.LEAN_BRAIN_ENABLED = prev

    def test_silence_fill_suppressed_only_under_lean(self):
        self.assertTrue(self._rejected("idle_monologue", lean=True))
        self.assertFalse(self._rejected("idle_monologue", lean=False))   # classic path intact

    def test_perception_reactors_never_suppressed(self):
        for reactor in ("presence_reaction", "world.animal_arrival", "world.scenery_change"):
            self.assertFalse(self._rejected(reactor, lean=True), reactor)


def _one_chunk_stream(text):
    """A fake OpenAI stream yielding one delta with `text`."""
    return [NS(choices=[NS(delta=NS(content=text))])]


class ImpulseDecisionParsingTest(unittest.TestCase):
    def test_pass_means_watch(self):
        with mock.patch.object(lean_brain.llm_compat, "create", return_value=_one_chunk_stream("PASS")):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]), "")

    def test_pass_with_trailing_junk_still_watches(self):
        with mock.patch.object(lean_brain.llm_compat, "create", return_value=_one_chunk_stream('PASS.')):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]), "")

    def test_a_real_line_is_spoken(self):
        with mock.patch.object(lean_brain.llm_compat, "create",
                               return_value=_one_chunk_stream("Nice hat, Bret.")):
            self.assertEqual(lean_brain.consider_initiating(person_id=None, transcript=[]),
                             "Nice hat, Bret.")


if __name__ == "__main__":
    unittest.main()
