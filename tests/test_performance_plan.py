import unittest


class PerformancePlanTests(unittest.TestCase):
    def test_tell_joke_plan_sets_delivery_body_and_memory_policy(self):
        from intelligence import performance_plan

        plan = performance_plan.plan_for_action(
            "humor.tell_joke",
            user_text="tell me a joke",
        )

        self.assertIsInstance(plan, performance_plan.PerformancePlan)
        self.assertEqual(plan.action, "humor.tell_joke")
        self.assertEqual(plan.emotion, "happy")
        self.assertEqual(plan.body_beat, "dramatic_visor_peek")
        self.assertEqual(plan.delivery_style, "quick_punchline")
        self.assertEqual(plan.memory_policy, performance_plan.MEMORY_DO_NOT_STORE)
        self.assertIn("Deliver the punchline and stop", plan.prompt_contract)

    def test_roast_plan_keeps_target_and_safety_contract(self):
        from intelligence import performance_plan

        plan = performance_plan.plan_for_action(
            "humor.roast",
            user_text="roast me",
            args={"target": "speaker"},
        )

        self.assertEqual(plan.emotion, "curious")
        self.assertEqual(plan.body_beat, "suspicious_glance")
        self.assertEqual(plan.delivery_style, "consent_roast")
        self.assertIn("Roast target: 'speaker'", plan.prompt_contract)
        self.assertIn("Do NOT joke about body, age, gender", plan.prompt_contract)

    def test_free_bit_plan_prefers_self_deprecating_riff(self):
        from intelligence import performance_plan

        plan = performance_plan.plan_for_action(
            "humor.free_bit",
            user_text="say something funny",
        )

        self.assertEqual(plan.body_beat, "proud_dj_pose")
        self.assertEqual(plan.delivery_style, "quick_riff")
        self.assertIn("self-deprecation", plan.prompt_contract)

    def test_future_performance_actions_have_non_executing_plans(self):
        from intelligence import performance_plan

        dj = performance_plan.plan_for_action("performance.dj_bit", user_text="do your DJ thing")
        beat = performance_plan.plan_for_action(
            "performance.body_beat",
            args={"body_beat": "tiny_victory_dance"},
        )
        mood = performance_plan.plan_for_action(
            "performance.mood_pose",
            args={"mood": "embarrassed"},
        )

        self.assertEqual(dj.body_beat, "proud_dj_pose")
        self.assertEqual(dj.delivery_style, "dj_stinger")
        self.assertEqual(beat.body_beat, "tiny_victory_dance")
        self.assertFalse(beat.requires_llm)
        self.assertEqual(mood.body_beat, "dramatic_visor_peek")
        self.assertEqual(mood.delivery_style, "mood_pose")
        self.assertFalse(mood.requires_llm)

    def test_body_beat_plan_uses_named_fallback_and_emotion(self):
        from intelligence import performance_plan

        victory = performance_plan.plan_for_action(
            "performance.body_beat",
            args={"body_beat": "victory dance"},
        )
        offended = performance_plan.plan_for_action(
            "performance.body_beat",
            args={"body_beat": "offended recoil"},
        )

        self.assertEqual(victory.body_beat, "tiny_victory_dance")
        self.assertEqual(victory.emotion, "happy")
        self.assertIn("Tiny victory dance", victory.fallback_text)
        self.assertEqual(offended.body_beat, "offended_recoil")
        self.assertEqual(offended.emotion, "angry")

    def test_mood_pose_plan_maps_emotions_to_body_beats(self):
        from intelligence import performance_plan

        examples = {
            "bashful": ("embarrassed", "dramatic_visor_peek"),
            "annoyed": ("annoyed", "offended_recoil"),
            "proud": ("proud", "proud_dj_pose"),
        }

        for mood, (_canonical, beat) in examples.items():
            with self.subTest(mood=mood):
                plan = performance_plan.plan_for_action(
                    "performance.mood_pose",
                    args={"mood": mood},
                )
                self.assertEqual(plan.body_beat, beat)
                self.assertEqual(plan.memory_policy, performance_plan.MEMORY_DO_NOT_STORE)

    def test_body_beat_for_event_maps_physical_stage_directions(self):
        from intelligence import performance_plan

        examples = {
            "insult.detected": "offended_recoil",
            "repair.misunderstood": "thinking_tilt",
            "idle.empty_room": "thinking_tilt",
            "game.correct": "tiny_victory_dance",
            "game.wrong": "suspicious_glance",
            "dj.bit": "proud_dj_pose",
        }

        for event, beat in examples.items():
            with self.subTest(event=event):
                self.assertEqual(performance_plan.body_beat_for_event(event), beat)

    def test_body_beat_for_event_uses_action_outcome_and_aliases(self):
        from intelligence import performance_plan

        self.assertEqual(
            performance_plan.body_beat_for_event("action", action="humor.roast"),
            "suspicious_glance",
        )
        self.assertEqual(
            performance_plan.body_beat_for_event("game", outcome="win"),
            "tiny_victory_dance",
        )
        self.assertEqual(
            performance_plan.body_beat_for_event("manual", body_beat="side eye"),
            "suspicious_glance",
        )

    def test_unknown_action_has_no_plan(self):
        from intelligence import performance_plan

        self.assertIsNone(performance_plan.plan_for_action("conversation.reply"))


if __name__ == "__main__":
    unittest.main()
