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


class IntentClassifierPrecedenceTests(unittest.TestCase):
    """A vision phrasing must out-rank the greedy person-memory heuristic.

    Field 2026-07-23: 'Well, what do you see in my hand? And maybe that gives you the
    answer.' (a wine glass held up) was classified query_memory — the
    references_person_memory_target fallback fired on the bare 'my' — and answered
    'Name the person, and I'll stop pretending my photoreceptors are psychic.'
    """

    def test_see_in_my_hand_routes_to_vision_not_memory(self):
        from intelligence import intent_classifier as ic
        for text in [
            "Well, what do you see in my hand? And maybe that gives you the answer.",
            "what do you see in my hand",
            "what am I holding",
            "can you see what I'm holding",
        ]:
            with self.subTest(text=text):
                self.assertEqual(ic.classify_deterministic(text), "query_what_do_you_see")

    def test_real_memory_queries_still_route_to_memory(self):
        from intelligence import intent_classifier as ic
        for text in ["what do you know about me", "tell me about my dog",
                     "what do you remember about me"]:
            with self.subTest(text=text):
                self.assertEqual(ic.classify_deterministic(text), "query_memory")


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

        # Since the Stage 1 demotion (2026-08-13) vision.describe_scene is
        # tool-router-owned, so this lane hands the turn to the reply call rather
        # than claiming it. The property under test is unchanged: a camera
        # question must NOT be swallowed as an answer to Rex's last turn.
        reason = interaction._intent_execution_block_reason(
            "query_what_do_you_see",
            text="can you see what I'm holding",
            dialogue_decision=_answer_to_rex_decision(),
        )
        self.assertEqual(reason, "tool_router_owns_action")
        self.assertNotEqual(reason, "blocked_by_dialogue_act")

        # Offline the lane claims again and the breakout must still fire.
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
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


class CompoundLookCommandTests(unittest.TestCase):
    """'Look to your right and tell me what you see' describes the new view.

    Live failure (logs/djr3x-2026-06-09-15-59-14.log): the directional half
    parsed as a bare look, the head turned, and the face-greet branch answered
    "Oh hi, Bret" — swallowing the vision question entirely.
    """

    def _fake_frame(self):
        import numpy as np

        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_compound_look_runs_directed_view_analysis(self):
        from intelligence import command_parser, interaction

        text = "Look to your right and tell me what you see"
        match = command_parser.parse(text)
        self.assertEqual(match.command_key, "directed_look")

        analysis = {
            "target_summary": "A telescope on a tripod",
            "target_visible": True,
            "subject_type": "object",
            "visible_people_count": 1,
            "animals": [],
            "notable_details": ["white 3D printed telescope"],
            "roast_angle": "",
            "confidence": "high",
        }
        with (
            mock.patch.object(
                interaction,
                "_move_and_capture_gaze",
                return_value=("right", self._fake_frame()),
            ),
            mock.patch(
                "vision.scene.analyze_directed_attention", return_value=analysis
            ) as ada,
            mock.patch.object(
                interaction, "_visible_known_face_candidate"
            ) as face_check,
            mock.patch.object(
                interaction.llm, "get_response", return_value="One telescope."
            ),
            mock.patch.object(
                interaction.llm, "clean_response_text", side_effect=lambda x: x
            ),
            mock.patch.object(interaction, "_speak_blocking"),
        ):
            resp = interaction._execute_directed_look_command(
                match.args, 1, "Bret", text
            )
        self.assertFalse(face_check.called)
        self.assertEqual(ada.call_args.kwargs.get("utterance"), text)
        self.assertEqual(resp, "One telescope.")

    def test_front_truncated_look_still_turns_head(self):
        """Far-field ASR drops the leading "look", leaving "to your right, what
        do you see" — which no longer parses as a directed_look and routes here
        as a plain scene description. Rex must still physically turn before
        describing, not list what is straight ahead (logged 2026-06-21). This
        guards the _handle_classified_intent choke point both the router and
        intent-classifier paths funnel through.
        """
        from intelligence import command_parser, interaction

        text = "to your right, what do you see?"
        # Without "look", the directional half does NOT parse as a directed_look.
        match = command_parser.parse(text)
        self.assertTrue(match is None or match.command_key != "directed_look")

        analysis = {
            "target_summary": "A cluttered shelf of cans",
            "target_visible": True,
            "subject_type": "object",
            "visible_people_count": 0,
            "animals": [],
            "notable_details": [],
            "roast_angle": "",
            "confidence": "high",
        }
        with (
            mock.patch.object(
                interaction,
                "_move_and_capture_gaze",
                return_value=("right", self._fake_frame()),
            ) as move,
            mock.patch(
                "vision.scene.analyze_directed_attention", return_value=analysis
            ) as ada,
            mock.patch.object(
                interaction.people_memory, "get_person", return_value=None
            ),
            mock.patch.object(
                interaction.llm, "get_response", return_value="A shelf of cans."
            ),
            mock.patch.object(
                interaction.llm, "clean_response_text", side_effect=lambda x: x
            ),
            mock.patch.object(interaction, "_speak_blocking"),
        ):
            resp = interaction._handle_classified_intent(
                "query_what_do_you_see", text, 1
            )
        self.assertTrue(move.called, "head must physically turn before describing")
        self.assertEqual(move.call_args.args[0], "right")
        self.assertTrue(ada.called, "the new view must be analyzed")
        self.assertEqual(resp, "A shelf of cans.")

    def test_plain_what_do_you_see_does_not_turn(self):
        """A direction-free "what do you see" must NOT trigger a head-turn — it
        describes straight ahead via the normal prompt path."""
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_move_and_capture_gaze") as move,
            mock.patch.object(
                interaction, "_vision_question_answer_prompt", return_value="PROMPT"
            ),
            mock.patch.object(
                interaction.llm, "get_response", return_value="A desk."
            ),
            mock.patch.object(interaction, "_speak_blocking"),
        ):
            resp = interaction._handle_classified_intent(
                "query_what_do_you_see", "what do you see?", 1
            )
        self.assertFalse(move.called, "no direction → no head-turn")
        self.assertEqual(resp, "A desk.")

    def test_plain_directional_look_reports_the_view_instead_of_greeting(self):
        """A directed look must not be hijacked by a greeting.

        _visible_known_face_candidate reads world_state (stale — face tracking is
        suspended across the turn) and filters to person_name, i.e. the speaker
        who gave the order, so the only line it can produce is "Oh hi, <whoever
        is talking to me>". On 2026-08-13 21:02:45 that landed on top of a
        "look down and then look left" aimed at a dog. With a frame in hand the
        report path wins; the greet is the no-frame fallback only.
        """
        from intelligence import command_parser, interaction

        analysis = {
            "target_summary": "A dog lying on the floor to the left",
            "target_visible": True,
            "subject_type": "animal",
            "visible_people_count": 0,
            "animals": [{"species": "dog", "position": "floor, left"}],
            "notable_details": ["dog on the rug"],
            "roast_angle": "",
            "confidence": "high",
        }
        match = command_parser.parse("look to your right")
        with (
            mock.patch.object(
                interaction,
                "_move_and_capture_gaze",
                return_value=("right", self._fake_frame()),
            ),
            mock.patch(
                "vision.scene.analyze_directed_attention", return_value=analysis
            ) as ada,
            mock.patch.object(
                interaction,
                "_visible_known_face_candidate",
                return_value={"name": "Bret", "person_id": 1},
            ),
            mock.patch.object(
                interaction, "_greet_directed_face_once", return_value="Oh hi, Bret."
            ) as greet,
            mock.patch.object(
                interaction.llm, "get_response", return_value="There is a dog down there."
            ),
            mock.patch.object(
                interaction.llm, "clean_response_text", side_effect=lambda x: x
            ),
            mock.patch.object(interaction, "_speak_blocking"),
        ):
            resp = interaction._execute_directed_look_command(
                match.args, 1, "Bret", "look to your right"
            )
        self.assertTrue(ada.called, "the captured frame must be analyzed, not discarded")
        self.assertFalse(greet.called, "a greeting must not hijack a directed look")
        self.assertEqual(resp, "There is a dog down there.")

    def test_directional_look_falls_back_to_greeting_without_a_frame(self):
        """Camera off / capture failed: the greeting is the only social signal
        left, so it stays as the last resort."""
        from intelligence import command_parser, interaction

        match = command_parser.parse("look to your right")
        with (
            mock.patch.object(
                interaction, "_move_and_capture_gaze", return_value=("right", None)
            ),
            mock.patch("vision.scene.analyze_directed_attention") as ada,
            mock.patch.object(
                interaction,
                "_visible_known_face_candidate",
                return_value={"name": "Bret", "person_id": 1},
            ),
            mock.patch.object(
                interaction, "_greet_directed_face_once", return_value="Oh hi, Bret."
            ) as greet,
        ):
            resp = interaction._execute_directed_look_command(
                match.args, 1, "Bret", "look to your right"
            )
        self.assertFalse(ada.called)
        self.assertTrue(greet.called)
        self.assertEqual(resp, "Oh hi, Bret.")


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
