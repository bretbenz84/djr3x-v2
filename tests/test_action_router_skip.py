"""action_router.decide after the JSON-prose LLM router was retired.

Everything the deterministic ladder (shutdown pre-pass + the explicit control,
humor and performance classifiers) does not claim is conversation: the lean
reply call picks tools natively.
"""

import unittest
from unittest import mock

from intelligence import action_router as AR


class LlmFallbackRetirementTest(unittest.TestCase):
    """Phase 4 (2026-08-13): the JSON-prose fallback is RETIRED.

    Measured before flipping it: across 1,340 audited field turns the LLM branch
    produced exactly TWO executions, both character.preference_query -- an action
    retired the same day (6267d38). Every other router_takeover.* in the log
    corpus came from decide()'s deterministic pre-LLM ladder.
    """

    def test_default_off_pays_no_call_and_hands_the_turn_to_conversation(self):
        # The phrasings that still reached the JSON router: a game phrasing
        # command_parser misses, a bare gaze request, an off-pattern music ask.
        # All three are LIVE tools on the reply call, which is why this is a
        # handoff and not a loss.
        #
        # Plain chat (2026-07-05/06 field logs) and the self-knowledge queries must
        # come out as conversation.reply too: that is what lets the zero-LLM intent
        # lane answer time/date/weather/etc. and the reply call pick tools, so an
        # explicit classifier that starts claiming one of them is a regression.
        for text in (
            "let's do that trivia thing again",
            "could you look over there",
            "I want to hear a song",
            "Just testing your program, how are you doing",
            "what's new with you",
            "Oh, what planet are you on?",
            "I did not smile",
            "my guts hurt",
            "It'll be nothing like the Rebel Alliance",
            "What day is it?",
            "What time is it?",
            "What's the weather like?",
            "What's the temperature inside?",   # indoor BME280 branch
            "What can you do?",
            "Who is speaking?",
            "what games can you play?",
        ):
            decision = AR.decide(text, {})
            self.assertEqual(decision.action, "conversation.reply", text)

    def test_offline_unclaimed_turns_stay_conversation(self):
        # Offline the explicit classifiers are the only routing Rex has; a turn
        # none of them claims still hands off to conversation (the intent lane
        # answers the self-query offline).
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            for text in ("something about the weather maybe", "What day is it?"):
                self.assertEqual(
                    AR.decide(text, {}).action, "conversation.reply", text
                )

    def test_active_game_and_music_no_longer_buy_a_routing_call(self):
        # These kept full routing so game.answer / a bare "stop" could win. Mid-game
        # the active-game claim in interaction._handle_speech_segment returns before
        # decide() is ever called, and game.answer is blocked "game_inactive"
        # outside one -- so the call bought nothing in either state.
        for ctx in ({"active_game": True}, {"active_music": True}):
            decision = AR.decide("purple elephants", ctx)
            self.assertEqual(decision.action, "conversation.reply", str(ctx))

    def test_deterministic_lanes_are_untouched(self):
        # The pre-LLM ladder is where every logged router_takeover.* actually came
        # from, so it must survive the retirement unchanged.
        self.assertEqual(
            AR.decide("I would like you to shut down.", {}).action,
            "system.shutdown",
        )
        self.assertEqual(
            AR.decide("Call me JT.", {}).action, "identity.name_correction"
        )
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            self.assertEqual(
                AR.decide("tell me a joke", {}).action, "humor.tell_joke"
            )


if __name__ == "__main__":
    unittest.main()
