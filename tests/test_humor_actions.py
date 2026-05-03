import unittest
from unittest import mock


class HumorActionExecutionTests(unittest.TestCase):
    def test_router_tell_joke_action_generates_single_punchline_contract(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.tell_joke",
            confidence=0.96,
            args={},
            reason="explicit joke request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Joke line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "tell me a joke",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "Joke line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("Tell exactly ONE short in-character DJ-R3X joke", prompt)
        self.assertIn("Deliver the punchline and stop", prompt)
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "Joke line.")
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")
        beat.assert_called_once_with("dramatic_visor_peek")

    def test_router_roast_action_keeps_prompt_non_sensitive(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.roast",
            confidence=0.96,
            args={"target": "speaker"},
            reason="explicit roast request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Roast line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "roast me",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "Roast line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("consent-based Rex roast", prompt)
        self.assertIn("Do NOT joke about body, age, gender", prompt)
        self.assertIn("No question. One line only.", prompt)
        self.assertEqual(speak.call_args.kwargs["emotion"], "curious")
        beat.assert_called_once_with("suspicious_glance")

    def test_fast_local_takeover_handles_explicit_free_humor_without_router_flag(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Funny line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat"),
        ):
            response = interaction._handle_fast_local_takeover(
                "say something funny",
                person_id=None,
                person_name=None,
            )

        self.assertEqual(response, "Funny line.")
        self.assertIn("asked Rex to be funny or do a bit", get_response.call_args.args[0])
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")

    def test_fast_local_takeover_ignores_plain_joke_mentions(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._handle_fast_local_takeover(
                "that joke was funny",
                person_id=None,
                person_name=None,
            )

        self.assertIsNone(response)
        get_response.assert_not_called()
        speak.assert_not_called()

    def test_router_dj_bit_action_uses_performance_plan(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.dj_bit",
            confidence=0.95,
            args={},
            reason="explicit DJ performance request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="DJ line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "do your DJ thing",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "DJ line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("DJ-R3X cantina patter", prompt)
        self.assertIn("Do not start music", prompt)
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")
        beat.assert_called_once_with("proud_dj_pose")

    def test_fast_local_takeover_handles_explicit_dj_bit(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Hype line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat"),
        ):
            response = interaction._handle_fast_local_takeover(
                "hype the room",
                person_id=None,
                person_name=None,
            )

        self.assertEqual(response, "Hype line.")
        self.assertIn("station-break", get_response.call_args.args[0])
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")

    def test_router_body_beat_action_executes_without_llm_generation(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.body_beat",
            confidence=0.95,
            args={"body_beat": "suspicious_glance"},
            reason="explicit body beat performance request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "look suspicious",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertIn("Suspicious glance", response)
        get_response.assert_not_called()
        beat.assert_called_once_with("suspicious_glance")
        self.assertEqual(speak.call_args.args[0], response)
        self.assertEqual(speak.call_args.kwargs["emotion"], "curious")

    def test_fast_local_takeover_handles_explicit_body_beat(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_fast_local_takeover(
                "do a victory dance",
                person_id=None,
                person_name=None,
            )

        self.assertIn("Tiny victory dance", response)
        get_response.assert_not_called()
        beat.assert_called_once_with("tiny_victory_dance")
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")


if __name__ == "__main__":
    unittest.main()
