"""Deterministic conversational skip in action_router.decide (latency work 2026-07-06).

The LLM routing call cost ~0.8s on every plain chat turn and returned
conversation.reply essentially always. Now a turn with no action-domain cue words, no
active game/music, and a deterministic 'general' intent skips the LLM entirely; any
cue word or active-mode context keeps full routing.
"""

import unittest
from unittest import mock

from intelligence import action_router as AR


class DeterministicSkipTest(unittest.TestCase):
    def _decide_with_llm_guard(self, text, context=None):
        """decide() with the router LLM mocked to COUNT calls (and fail the routing
        so a consulted call surfaces as the error-fallback decision)."""
        calls = []

        def _fake_create(**kw):
            calls.append(1)
            raise RuntimeError("llm consulted")

        with mock.patch.object(AR._client.chat.completions, "create", side_effect=_fake_create):
            decision = AR.decide(text, context or {})
        return decision, len(calls)

    def test_chat_turns_skip_the_llm(self):
        # Real utterances from the 2026-07-05/06 field logs — all plain conversation.
        for text in (
            "Just testing your program, how are you doing",
            "what's new with you",
            "Oh, what planet are you on?",
            "I did not smile",
            "my guts hurt",
            "It'll be nothing like the Rebel Alliance",
        ):
            decision, llm_calls = self._decide_with_llm_guard(text)
            self.assertEqual(llm_calls, 0, text)
            self.assertEqual(decision.action, "conversation.reply", text)
            self.assertIn("deterministic", decision.reason, text)

    def test_action_cues_keep_the_llm_router(self):
        for text in (
            "something about the weather maybe",
            "let's do that trivia thing again",
            "could you look over there",
            "I want to hear a song",
        ):
            _decision, llm_calls = self._decide_with_llm_guard(text)
            self.assertEqual(llm_calls, 1, text)

    def test_active_game_blocks_the_skip(self):
        # Mid-game answers can be arbitrary words — full routing must stay on.
        _d, llm_calls = self._decide_with_llm_guard("purple elephants", {"active_game": True})
        self.assertEqual(llm_calls, 1)

    def test_active_music_blocks_the_skip(self):
        _d, llm_calls = self._decide_with_llm_guard("that one is nice", {"active_music": True})
        self.assertEqual(llm_calls, 1)

    def test_kill_switch(self):
        import config
        with mock.patch.object(config, "ACTION_ROUTER_DETERMINISTIC_SKIP_ENABLED", False, create=True):
            _d, llm_calls = self._decide_with_llm_guard("my guts hurt")
        self.assertEqual(llm_calls, 1)


if __name__ == "__main__":
    unittest.main()
