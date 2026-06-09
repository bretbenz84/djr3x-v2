"""Vision-query routing regression tests.

Replays of live failures from logs/djr3x-2026-06-09-15-34-52.log where genuine
camera questions were blocked by the turn-policy evidence gate and answered by
the plain text LLM instead — producing a hallucinated "I see it" about a
telescope and a blind "can't say I can" about a held-up object:

  1. "look to your right"      → directed_look parsed, but blocked_by_dialogue_act.
  2. "You can see the telescope, look at my telescope"
                               → router said vision.describe_scene @0.95, but
                                 missing_vision_query_evidence blocked it.
  3. "Can you see what I'm holding?"
                               → same block ("what I'm holding" didn't match the
                                 old "what am i holding" pattern).

These tests pin the widened evidence patterns, the dialogue-act breakout for
camera requests, and the question-aware vision answer path.
"""

import unittest
from unittest import mock


def _answer_to_rex_decision():
    from intelligence import dialogue_act

    return dialogue_act.DialogueActDecision(
        label="answer_to_rex",
        confidence=0.90,
        reason="reply to last Rex turn",
    )


class VisionQueryEvidenceTests(unittest.TestCase):
    """has_vision_query_evidence: real camera questions pass, idioms do not."""

    def test_live_failure_phrasings_pass(self):
        from intelligence import action_router

        for text in [
            "You can see the telescope, look at my telescope",
            "Can you see what I'm holding?",
            "what am I holding",
            "what i'm holding",
            "what am I wearing",
            "do you see my dog",
            "can you see the telescope",
            "what do you see",
            "what can you see",
            "describe the room",
            "what's in front of you",
            "tell me what you see",
        ]:
            with self.subTest(text=text):
                self.assertTrue(action_router.has_vision_query_evidence(text))

    def test_conversational_idioms_do_not_pass(self):
        from intelligence import action_router

        for text in [
            "do you see what I mean",
            "can you see my point",
            "look at the bright side",
            "we'll see how it goes",
            "see you later",
            "I see your point",
            "yeah that's not happening anymore",
        ]:
            with self.subTest(text=text):
                self.assertFalse(action_router.has_vision_query_evidence(text))

    def test_evidence_gate_unblocks_vision_action(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="vision.describe_scene",
            confidence=0.95,
            args={},
            reason="vision question",
        )
        self.assertIsNone(
            action_router.missing_required_evidence_reason(
                "Can you see what I'm holding?", decision
            )
        )
        self.assertEqual(
            action_router.missing_required_evidence_reason(
                "yeah that's not happening anymore", decision
            ),
            "missing_vision_query_evidence",
        )


class DirectedLookEvidenceTests(unittest.TestCase):
    def test_gaze_imperatives_pass(self):
        from intelligence import action_router

        for text in [
            "look to your right",
            "look left",
            "look at this",
            "look behind you",
            "look at my telescope",
            "look for the dog",
        ]:
            with self.subTest(text=text):
                self.assertTrue(action_router.has_directed_look_evidence(text))

    def test_look_idioms_do_not_pass(self):
        from intelligence import action_router

        for text in [
            "look at the bright side",
            "look sharp",
            "looking good there",
            "look out below",
        ]:
            with self.subTest(text=text):
                self.assertFalse(action_router.has_directed_look_evidence(text))


class DialogueBreakoutTests(unittest.TestCase):
    """Camera requests break out of a stale answer_to_rex binding; replies don't."""

    def test_directed_look_breaks_out_of_answer_binding(self):
        from intelligence import command_parser, interaction

        match = command_parser.parse("look to your right")
        self.assertIsNotNone(match)
        self.assertEqual(match.command_key, "directed_look")
        self.assertIsNone(
            interaction._legacy_command_execution_block_reason(
                match,
                text="look to your right",
                dialogue_decision=_answer_to_rex_decision(),
            )
        )

    def test_look_idiom_reply_stays_blocked(self):
        from intelligence import command_parser, interaction

        match = command_parser.parse("look at the bright side")
        if match is None:
            self.skipTest("parser no longer matches the idiom at all (fine)")
        self.assertEqual(
            interaction._legacy_command_execution_block_reason(
                match,
                text="look at the bright side",
                dialogue_decision=_answer_to_rex_decision(),
            ),
            "blocked_by_dialogue_act",
        )

    def test_vision_question_breaks_out_of_answer_binding(self):
        from intelligence import interaction

        self.assertIsNone(
            interaction._intent_execution_block_reason(
                "query_what_do_you_see",
                text="can you see what I'm holding",
                dialogue_decision=_answer_to_rex_decision(),
            )
        )

    def test_non_action_reply_still_blocked(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._intent_execution_block_reason(
                "query_time",
                text="yeah whenever you have time",
                dialogue_decision=_answer_to_rex_decision(),
            ),
            "missing_time_query_evidence",
        )


class VisionQuestionAnswerTests(unittest.TestCase):
    """The vision answer path sends the user's question to the vision call."""

    def _fake_frame(self):
        import numpy as np

        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_question_reaches_vision_call_and_grounds_prompt(self):
        from intelligence import interaction

        analysis = {
            "target_summary": "A soda can held up toward the camera",
            "target_visible": True,
            "subject_type": "object",
            "visible_people_count": 1,
            "animals": [],
            "notable_details": ["red aluminum can"],
            "roast_angle": "hydration tech",
            "confidence": "high",
        }
        with (
            mock.patch("vision.camera.get_frame", return_value=self._fake_frame()),
            mock.patch(
                "vision.scene.analyze_directed_attention", return_value=analysis
            ) as ada,
        ):
            prompt = interaction._vision_question_answer_prompt(
                "Can you see what I'm holding?"
            )
        self.assertEqual(
            ada.call_args.kwargs.get("utterance"), "Can you see what I'm holding?"
        )
        self.assertIn("soda can", prompt)
        self.assertIn("never claim to see", prompt)

    def test_honest_miss_keeps_target_visible_false_in_prompt(self):
        from intelligence import interaction

        analysis = {
            "target_summary": "No held object is clearly visible",
            "target_visible": False,
            "subject_type": "unknown",
            "visible_people_count": 1,
            "animals": [],
            "notable_details": [],
            "roast_angle": "",
            "confidence": "low",
        }
        with (
            mock.patch("vision.camera.get_frame", return_value=self._fake_frame()),
            mock.patch(
                "vision.scene.analyze_directed_attention", return_value=analysis
            ),
        ):
            prompt = interaction._vision_question_answer_prompt(
                "what am I holding"
            )
        self.assertIn('"target_visible": false', prompt)
        self.assertIn("honestly", prompt)

    def test_no_frame_falls_back_to_scene_summary(self):
        from intelligence import interaction

        with (
            mock.patch("vision.camera.get_frame", return_value=None),
            mock.patch(
                "vision.scene.describe_scene",
                return_value="home. 1 person visible.",
            ),
        ):
            prompt = interaction._vision_question_answer_prompt("what do you see")
        self.assertIn("home. 1 person visible.", prompt)

    def test_target_hint_extraction(self):
        from intelligence import interaction

        self.assertIn(
            "holding",
            interaction._vision_question_target_hint("can you see what I'm holding"),
        )
        self.assertEqual(
            interaction._vision_question_target_hint("look at my telescope"),
            "telescope",
        )
        self.assertIn(
            "wearing",
            interaction._vision_question_target_hint("what am I wearing today"),
        )


if __name__ == "__main__":
    unittest.main()
