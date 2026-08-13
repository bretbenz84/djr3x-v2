"""
Field fix from the 2026-08-12 22:54 session: a full answer to Rex's own
question was silently dropped.

Rex asked "New sensors for me?"; 14 seconds later Bret answered "Yeah, I'm
gonna be adding three radar sensors to your body, so you can sense where
people are in relation to your robot body." The word "robot" tripped the
address-mode keyword gate, no hard rule fired, and the gpt-4o-mini fallback
labeled the reply "instructional" (Rex as the object of an action directed at
someone else — a plausible cold read of "adding sensors to your body" with no
conversational context). The turn was diverted to being_discussed and
returned before ever being recorded as HEARD — no reply, no transcript entry,
and the lull impulse then swerved to an unrelated follow-up.

Three guards now stand between a mid-conversation answer and that drop:
1. A fresh Rex-turn frame for the speaker makes classify() skip the LLM guess
   entirely (hard rules still run, so an explicit "say hi to Rex" mid-chat
   keeps its instructional read).
2. Utterances dense with second-person pronouns ("your body", "you can
   sense") are direct address by hard rule — no LLM needed.
3. When the LLM is consulted, its context carries Rex's most recent line to
   the speaker, and the prompt says a second-person reply to it is direct.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from awareness import address_mode


FIELD_UTTERANCE = (
    "Yeah, I'm gonna be adding three radar sensors to your body, "
    "so you can sense where people are in relation to your robot body."
)


class SecondPersonHardRuleTests(unittest.TestCase):

    def test_the_field_utterance_is_direct_by_hard_rule(self):
        # Three second-person hits — the LLM must never see this sentence.
        with mock.patch.object(address_mode, "_llm_classify") as llm:
            result = address_mode.classify(FIELD_UTTERANCE)
        llm.assert_not_called()
        self.assertEqual(result.label, address_mode.ADDRESS_DIRECT)
        self.assertEqual(result.rule, "second_person")

    def test_a_single_conversational_you_does_not_trip_it(self):
        # "you know" filler aimed at a third party must still reach the LLM.
        result = address_mode.classify(
            "That robot is really something, you know.", skip_llm=True
        )
        self.assertEqual(result.rule, "skip_llm_default")

    def test_instructional_verbs_keep_precedence(self):
        # Density must not swallow an explicit relay instruction.
        result = address_mode.classify(
            "You should tell Rex you loved the show.", skip_llm=True
        )
        self.assertEqual(result.label, address_mode.ADDRESS_INSTRUCTIONAL)
        self.assertEqual(result.rule, "instructional_verb")

    def test_direct_prefix_rule_still_first(self):
        result = address_mode.classify("Hey Rex, how are you?", skip_llm=True)
        self.assertEqual(result.label, address_mode.ADDRESS_DIRECT)
        self.assertEqual(result.rule, "direct_prefix")


class ActiveExchangeBypassTests(unittest.TestCase):

    def test_mid_exchange_reply_skips_the_llm(self):
        # Keyword present, no hard rule — outside an exchange this would be
        # the LLM's call; mid-exchange it stays Rex's turn to answer.
        with mock.patch.object(address_mode, "_llm_classify") as llm:
            result = address_mode.classify(
                "The robot deserves the upgrade honestly.",
                in_active_exchange=True,
            )
        llm.assert_not_called()
        self.assertEqual(result.label, address_mode.ADDRESS_DIRECT)
        self.assertEqual(result.rule, "active_exchange")

    def test_explicit_relay_instruction_survives_the_bypass(self):
        result = address_mode.classify("Say hi to Rex.", in_active_exchange=True)
        self.assertEqual(result.label, address_mode.ADDRESS_INSTRUCTIONAL)

    def test_outside_an_exchange_the_llm_still_decides(self):
        with mock.patch.object(
            address_mode, "_llm_classify", return_value=("referential", "positive")
        ) as llm:
            result = address_mode.classify("That robot is so fun.")
        llm.assert_called_once()
        self.assertEqual(result.label, address_mode.ADDRESS_REFERENTIAL)


class FieldScenarioTests(unittest.TestCase):

    def setUp(self):
        from intelligence import dialogue_act
        dialogue_act.clear()
        self.addCleanup(dialogue_act.clear)

    def test_rexs_question_leaves_a_fresh_frame_for_the_answer(self):
        from intelligence import dialogue_act
        dialogue_act.note_rex_turn("New sensors for me?")
        frame = dialogue_act.active_frame(
            person_id=1,
            max_age_secs=float(config.ADDRESS_MODE_EXCHANGE_FRESH_SECS),
        )
        self.assertIsNotNone(frame)
        result = address_mode.classify(FIELD_UTTERANCE, in_active_exchange=True)
        self.assertEqual(result.label, address_mode.ADDRESS_DIRECT)

    def test_llm_prompt_carries_the_context_guidance(self):
        self.assertIn("Context:", address_mode._LLM_PROMPT)
        self.assertIn("second person", address_mode._LLM_PROMPT)


class GateWiringTests(unittest.TestCase):
    """The full segment handler needs a heavyweight harness — asserted
    structurally, same as the own-echo voice override."""

    def test_handler_passes_exchange_state_and_rex_context(self):
        import inspect
        from intelligence import interaction as I
        src = inspect.getsource(I._handle_speech_segment)
        idx = src.index("address_mode.classify(")
        window = src[max(0, idx - 1500): idx + 300]
        self.assertIn("in_active_exchange", window)
        self.assertIn("ADDRESS_MODE_EXCHANGE_FRESH_SECS", window)
        self.assertIn("Rex's most recent line to this speaker", window)
        self.assertIn("dialogue_act.active_frame", window)

    def test_freshness_window_sits_inside_the_frame_ttl(self):
        from intelligence import dialogue_act
        secs = float(config.ADDRESS_MODE_EXCHANGE_FRESH_SECS)
        self.assertGreaterEqual(secs, 20.0)  # covers the field 14s gap with margin
        self.assertLess(secs, dialogue_act._FRAME_TTL_SECS)


if __name__ == "__main__":
    unittest.main()
