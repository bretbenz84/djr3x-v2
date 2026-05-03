import unittest


class ActionRouterExecutionGateTests(unittest.TestCase):
    def test_default_execute_allowlist_is_conservative(self):
        import config

        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS)

        self.assertTrue(config.ACTION_ROUTER_EXECUTE_ENABLED)
        self.assertTrue({
            "humor.tell_joke",
            "humor.roast",
            "humor.free_bit",
            "performance.dj_bit",
            "time.query",
            "date.query",
            "weather.query",
            "status.uptime",
            "status.capabilities",
        }.issubset(allowed))
        self.assertTrue({
            "memory.forget_specific",
            "memory.forget_person",
            "event.cancel",
            "music.play",
            "game.start",
            "system.sleep",
            "performance.body_beat",
        }.isdisjoint(allowed))

    def test_allowed_high_confidence_action_is_executable(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="weather.query",
            confidence=0.95,
            args={},
            reason="weather question",
        )

        self.assertTrue(interaction._router_decision_executable(decision))
        self.assertIsNone(interaction._router_execution_block_reason(decision))

    def test_executable_but_unallowlisted_action_is_blocked(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="memory.forget_specific",
            confidence=0.99,
            args={"target": "Disneyland"},
            reason="explicit forget",
        )

        self.assertFalse(interaction._router_decision_executable(decision))
        self.assertEqual(
            interaction._router_execution_block_reason(decision),
            "not_in_execute_allowlist",
        )

    def test_low_confidence_allowed_action_is_blocked(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="time.query",
            confidence=0.50,
            args={},
            reason="weak time question",
        )

        self.assertFalse(interaction._router_decision_executable(decision))
        self.assertEqual(
            interaction._router_execution_block_reason(decision),
            "below_confidence_threshold",
        )

    def test_confirmation_blocks_even_allowlisted_action(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="weather.query",
            confidence=0.99,
            args={},
            requires_confirmation=True,
            reason="confirmation requested",
        )

        self.assertFalse(interaction._router_decision_executable(decision))
        self.assertEqual(
            interaction._router_execution_block_reason(decision),
            "requires_confirmation",
        )


if __name__ == "__main__":
    unittest.main()
