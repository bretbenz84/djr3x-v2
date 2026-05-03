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

        self.assertEqual(dj.body_beat, "proud_dj_pose")
        self.assertEqual(dj.delivery_style, "dj_stinger")
        self.assertEqual(beat.body_beat, "tiny_victory_dance")
        self.assertFalse(beat.requires_llm)

    def test_unknown_action_has_no_plan(self):
        from intelligence import performance_plan

        self.assertIsNone(performance_plan.plan_for_action("conversation.reply"))


if __name__ == "__main__":
    unittest.main()
