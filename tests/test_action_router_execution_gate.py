import json
import unittest
from unittest import mock


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
            "performance.body_beat",
            "identity.who_is_speaking",
            "music.options",
            "music.stop",
            "music.skip",
            "game.stop",
            "vision.describe_scene",
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
            "game.answer",
            "game.start",
            "system.sleep",
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

    def test_promoted_safe_actions_are_executable(self):
        from intelligence import action_router, interaction

        cases = {
            "music.options": {},
            "music.stop": {},
            "music.skip": {},
            "game.stop": {},
            "performance.body_beat": {"body_beat": "tiny_victory_dance"},
            "identity.who_is_speaking": {},
            "vision.describe_scene": {},
        }

        for action, args in cases.items():
            with self.subTest(action=action):
                decision = action_router.ActionDecision(
                    action=action,
                    confidence=0.95,
                    args=args,
                    reason="safe promoted action",
                )
                self.assertTrue(interaction._router_decision_executable(decision))
                self.assertIsNone(interaction._router_execution_block_reason(decision))

    def test_body_beat_action_requires_known_beat_to_execute(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.body_beat",
            confidence=0.95,
            args={"body_beat": "spin the mystery servo"},
            reason="unknown physical request",
        )

        self.assertFalse(interaction._router_decision_executable(decision))
        self.assertEqual(
            interaction._router_execution_block_reason(decision),
            "unknown_body_beat",
        )

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

    def test_router_audit_logs_decision_allowlist_legacy_and_final_path(self):
        from intelligence import action_router, interaction

        audit = interaction._new_router_audit(
            "tell me a joke",
            {
                "legacy": {
                    "command_match": {
                        "command_key": "memory_review",
                        "match_type": "regex",
                    },
                },
            },
        )
        decision = action_router.ActionDecision(
            action="humor.tell_joke",
            confidence=0.954,
            reason="explicit joke",
        )

        interaction._router_audit_note_decision(audit, decision)

        with mock.patch.object(interaction._log, "info") as info:
            interaction._log_router_audit(
                audit,
                "router_takeover.humor.tell_joke",
                completed=True,
                spoken_text="Joke line.",
            )

        info.assert_called_once()
        self.assertEqual(info.call_args.args[0], "[action_router_audit] %s")
        payload = json.loads(info.call_args.args[1])
        self.assertEqual(payload["utterance"], "tell me a joke")
        self.assertEqual(payload["router_action"], "humor.tell_joke")
        self.assertEqual(payload["router_confidence"], 0.954)
        self.assertEqual(payload["allowlist_result"], "allowed")
        self.assertEqual(payload["legacy_command"], "memory_review")
        self.assertEqual(payload["legacy_match_type"], "regex")
        self.assertEqual(payload["final_executed_path"], "router_takeover.humor.tell_joke")
        self.assertTrue(payload["completed"])
        self.assertIsNone(payload["handler_error"])
        self.assertTrue(payload["spoken_text_present"])

    def test_router_audit_records_unallowlisted_router_action(self):
        import config
        from intelligence import action_router, command_parser, interaction

        audit = interaction._new_router_audit("play music", {})
        decision = action_router.ActionDecision(
            action="music.play",
            confidence=0.99,
            reason="music request",
        )

        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", {"humor.tell_joke"}):
            interaction._router_audit_note_decision(audit, decision)
        interaction._router_audit_note_legacy_command(
            audit,
            command_parser.CommandMatch("dj_start", "keyword", {"genre": "jazz"}),
        )

        with mock.patch.object(interaction._log, "info") as info:
            interaction._log_router_audit(
                audit,
                "legacy_command.dj_start",
                completed=False,
                handler_error="router blocked; legacy fallback not simulated",
                spoken_text_present=False,
            )

        payload = json.loads(info.call_args.args[1])
        self.assertEqual(payload["router_action"], "music.play")
        self.assertEqual(payload["allowlist_result"], "not_in_execute_allowlist")
        self.assertEqual(payload["legacy_command"], "dj_start")
        self.assertEqual(payload["final_executed_path"], "legacy_command.dj_start")
        self.assertFalse(payload["completed"])
        self.assertEqual(payload["handler_error"], "router blocked; legacy fallback not simulated")
        self.assertFalse(payload["spoken_text_present"])


if __name__ == "__main__":
    unittest.main()
