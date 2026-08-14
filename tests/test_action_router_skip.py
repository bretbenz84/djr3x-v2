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

    def test_action_cues_without_an_owned_intent_keep_the_llm_router(self):
        # Cue words that do NOT resolve to a tool-router-owned intent still pay
        # the JSON-prose call. Games and the bare look/song phrasings are the
        # families still on that lane (games migrate in Stage 3).
        for text in (
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

    def test_loose_intent_claim_skips_the_llm_once_the_tool_owns_it(self):
        # Was: classify_deterministic says query_weather, the router's stricter
        # regex disagrees, so full routing stayed on. Since the Stage 1 demotion
        # (2026-08-13) weather.query is tool-router-owned, so that stricter regex
        # is no longer a safety property on this lane — it only decided whether to
        # pay ~0.8s on the way to the same conversation.reply. The reply call now
        # makes the call for free.
        decision, llm_calls = self._decide_with_llm_guard("something about the weather maybe")
        self.assertEqual(llm_calls, 0)
        self.assertEqual(decision.action, "conversation.reply")

    def test_active_game_blocks_the_self_query_skip(self):
        # Jeopardy answers are phrased "what is ..." — game.answer must win.
        _d, llm_calls = self._decide_with_llm_guard("What day is it?", {"active_game": True})
        self.assertEqual(llm_calls, 1)

    def test_self_query_kill_switch(self):
        import config
        with mock.patch.object(config, "ACTION_ROUTER_SELF_QUERY_SKIP_ENABLED", False, create=True):
            _d, llm_calls = self._decide_with_llm_guard("What day is it?")
        self.assertEqual(llm_calls, 1)

    def test_music_and_memory_intents_are_handed_to_the_reply_call(self):
        # Was: "deliberately excluded from the skip — router owns args /
        # disambiguation." Since Stage 1 the reply call owns all four
        # (vision/memory/music.play/music.options), so the router has nothing left
        # to add and the extra round-trip is pure cost.
        for text in ("play some jazz", "what music can you play?",
                     "what do you remember about me?", "what do you see?"):
            decision, llm_calls = self._decide_with_llm_guard(text)
            self.assertEqual(llm_calls, 0, text)
            self.assertEqual(decision.action, "conversation.reply", text)

    def test_offline_restores_the_pre_migration_routing(self):
        from unittest import mock as _mock

        with _mock.patch("intelligence.connectivity.is_offline", return_value=True):
            # Offline the intent lane CLAIMS again, so the stricter evidence regex
            # is back to being the only filter — and a loose claim it rejects must
            # keep full routing exactly as it did before the migration.
            _d, llm_calls = self._decide_with_llm_guard("something about the weather maybe")
            self.assertEqual(llm_calls, 1)


if __name__ == "__main__":
    unittest.main()
