"""Deterministic conversational skip in action_router.decide (latency work 2026-07-06).

The LLM routing call cost ~0.8s on every plain chat turn and returned
conversation.reply essentially always. Now a turn with no action-domain cue words, no
active game/music, and a deterministic 'general' intent skips the LLM entirely; any
cue word or active-mode context keeps full routing.
"""

import unittest
from unittest import mock

from intelligence import action_router as AR


class _LlmGuardMixin(unittest.TestCase):
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

class DeterministicSkipTest(_LlmGuardMixin):
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


class SelfQuerySkipTest(_LlmGuardMixin):
    """Mirror-image skip (latency work 2026-08-02): when the deterministic intent
    classifier already claims the turn as a self-knowledge query, the LLM router
    could only agree — 'What day is it?' burned a 0.91s routing call to return
    conversation.reply at 0.00 confidence, then executed via the intent
    classifier anyway (field log 2026-08-02 13:03)."""

    def test_self_queries_skip_the_llm(self):
        for text in (
            "What day is it?",
            "What time is it?",
            "What's the weather like?",
            "What's the temperature inside?",   # indoor BME280 branch
            "What can you do?",
            "Who is speaking?",
            "what games can you play?",
        ):
            decision, llm_calls = self._decide_with_llm_guard(text)
            self.assertEqual(llm_calls, 0, text)
            self.assertEqual(decision.action, "conversation.reply", text)
            self.assertIn("self-query", decision.reason, text)

    def test_loose_intent_claim_without_evidence_keeps_the_llm(self):
        # classify_deterministic says query_weather, but the router's stricter
        # evidence regex disagrees — same outcome the downstream execution gate
        # would enforce, so full routing must stay on.
        _d, llm_calls = self._decide_with_llm_guard("something about the weather maybe")
        self.assertEqual(llm_calls, 1)

    def test_active_game_blocks_the_self_query_skip(self):
        # Jeopardy answers are phrased "what is ..." — game.answer must win.
        _d, llm_calls = self._decide_with_llm_guard("What day is it?", {"active_game": True})
        self.assertEqual(llm_calls, 1)

    def test_self_query_kill_switch(self):
        import config
        with mock.patch.object(config, "ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED", False, create=True):
            _d, llm_calls = self._decide_with_llm_guard("What day is it?")
        self.assertEqual(llm_calls, 1)

    def test_music_and_memory_intents_still_route(self):
        # Deliberately excluded from the skip: router owns args / disambiguation.
        for text in ("play some jazz", "what music can you play?", "what do you remember about me?"):
            _d, llm_calls = self._decide_with_llm_guard(text)
            self.assertEqual(llm_calls, 1, text)


if __name__ == "__main__":
    unittest.main()
