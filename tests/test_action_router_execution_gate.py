import json
import unittest
from unittest import mock


class ActionRouterExecutionGateTests(unittest.TestCase):
    def test_default_execute_allowlist_is_conservative(self):
        import config

        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS)

        self.assertTrue(config.ACTION_ROUTER_EXECUTE_ENABLED)
        self.assertTrue({
            "conversation.repair",
            "humor.tell_joke",
            "humor.roast",
            "humor.free_bit",
            "performance.dj_bit",
            "performance.body_beat",
            "performance.mood_pose",
            "memory.query",
            "memory.recent_discard",
            "identity.who_is_speaking",
            "identity.name_correction",
            "music.options",
            "music.stop",
            "music.skip",
            "game.answer",
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
            "game.start",
            "vision.snapshot",
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
            "conversation.repair": {},
            "music.options": {},
            "music.stop": {},
            "music.skip": {},
            "game.stop": {},
            "performance.body_beat": {"body_beat": "tiny_victory_dance"},
            "performance.mood_pose": {"mood": "embarrassed"},
            "memory.query": {},
            "memory.recent_discard": {},
            "identity.who_is_speaking": {},
            "identity.name_correction": {"name": "Daniel"},
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

    def test_game_answer_requires_active_game_context(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="game.answer",
            confidence=0.95,
            args={},
            reason="short active-game answer",
        )

        with mock.patch("features.games.is_active", return_value=False):
            self.assertFalse(interaction._router_decision_executable(decision))
            self.assertEqual(
                interaction._router_execution_block_reason(decision),
                "game_inactive",
            )

        with mock.patch("features.games.is_active", return_value=True):
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

    def test_mood_pose_action_requires_known_mood_to_execute(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.mood_pose",
            confidence=0.95,
            args={"mood": "servo chaos"},
            reason="unknown mood pose",
        )

        self.assertFalse(interaction._router_decision_executable(decision))
        self.assertEqual(
            interaction._router_execution_block_reason(decision),
            "unknown_mood_pose",
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

    def test_character_loop_trace_logs_speaker_interpretation_execution_and_outcome(self):
        from intelligence import action_router, interaction

        trace = interaction._new_character_loop_trace(
            "Tell me a joke",
            from_idle_activation=True,
            turn_start=10.0,
            raw_best_id=1,
            raw_best_name="Bret Benziger",
            speaker_score=0.91234,
        )
        interaction._character_loop_note_speaker(
            trace,
            person_id=1,
            person_name="Bret Benziger",
            speaker_label="Bret Benziger",
            identity_resolution="voice_hard_match",
            off_camera_unknown=False,
            visible_known_count=1,
            has_unknown_visible_or_recent=False,
        )
        audit = interaction._new_router_audit("Tell me a joke", {})
        decision = action_router.ActionDecision(
            action="humor.tell_joke",
            confidence=0.956,
            reason="explicit joke",
        )
        interaction._router_audit_note_decision(audit, decision)

        with (
            mock.patch.object(interaction.time, "monotonic", return_value=11.234),
            mock.patch.object(interaction._log, "info") as info,
        ):
            interaction._log_character_loop_trace(
                trace,
                router_audit=audit,
                final_executed_path="fast_local_takeover.humor.tell_joke",
                completed=True,
                spoken_text="One tiny joke.",
                assistant_asked_question=False,
                suppress_memory_learning=True,
                intent="general",
            )

        info.assert_called_once()
        self.assertEqual(info.call_args.args[0], "[character_loop] %s")
        payload = json.loads(info.call_args.args[1])
        self.assertEqual(payload["utterance"], "Tell me a joke")
        self.assertEqual(payload["duration_ms"], 1234)
        self.assertEqual(payload["speaker"]["person_id"], 1)
        self.assertEqual(payload["speaker"]["identity_resolution"], "voice_hard_match")
        self.assertTrue(payload["speaker"]["from_idle_activation"])
        self.assertEqual(payload["speaker"]["raw_candidate"]["score"], 0.912)
        self.assertEqual(payload["interpretation"]["router_action"], "humor.tell_joke")
        self.assertEqual(payload["interpretation"]["router_confidence"], 0.956)
        self.assertEqual(payload["interpretation"]["allowlist_result"], "allowed")
        self.assertEqual(payload["interpretation"]["intent"], "general")
        self.assertEqual(
            payload["execution"]["final_executed_path"],
            "fast_local_takeover.humor.tell_joke",
        )
        self.assertTrue(payload["execution"]["completed"])
        self.assertTrue(payload["execution"]["spoken_text_present"])
        self.assertFalse(payload["execution"]["assistant_asked_question"])
        self.assertTrue(payload["execution"]["suppress_memory_learning"])

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

    def test_fast_local_stop_music_audit_uses_stable_action_path(self):
        from intelligence import interaction

        audit = interaction._new_router_audit("stop the music", {})

        with (
            mock.patch("features.dj.is_playing", return_value=False),
            mock.patch("features.dj.stop") as stop,
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
        ):
            response = interaction._handle_fast_local_takeover(
                "stop the music",
                person_id=1,
                person_name="Bret",
                router_audit=audit,
            )

        self.assertEqual(response, "No music is playing.")
        stop.assert_called_once()
        self.assertEqual(audit.router_action, "music.stop")
        self.assertEqual(audit.router_confidence, 1.0)
        self.assertEqual(audit.allowlist_result, "allowed")
        self.assertEqual(audit.legacy_command, "dj_stop")
        self.assertEqual(
            interaction._router_audit_fast_local_final_path(audit),
            "fast_local_takeover.music.stop",
        )

    def test_fast_local_specific_memory_forget_audit_remains_legacy_path(self):
        from intelligence import interaction

        audit = interaction._new_router_audit("forget I like Star Wars", {})

        with mock.patch.object(interaction, "_execute_command", return_value="Forgotten.") as execute:
            response = interaction._handle_fast_local_takeover(
                "forget I like Star Wars",
                person_id=1,
                person_name="Bret",
                router_audit=audit,
            )

        self.assertEqual(response, "Forgotten.")
        self.assertEqual(execute.call_args.args[0].command_key, "forget_specific")
        self.assertEqual(audit.router_action, "memory.forget_specific")
        self.assertEqual(audit.allowlist_result, "not_in_execute_allowlist")
        self.assertEqual(audit.legacy_command, "forget_specific")
        self.assertEqual(
            interaction._router_audit_fast_local_final_path(audit),
            "legacy_command.forget_specific",
        )

    def test_fast_local_recent_discard_uses_stable_action_path(self):
        from intelligence import interaction

        audit = interaction._new_router_audit("forget I said that", {})

        with mock.patch.object(
            interaction,
            "_execute_memory_boundary_command",
            return_value="Recent memory discarded.",
        ) as discard:
            response = interaction._handle_fast_local_takeover(
                "forget I said that",
                person_id=1,
                person_name="Bret",
                router_audit=audit,
            )

        self.assertEqual(response, "Recent memory discarded.")
        discard.assert_called_once_with(1)
        self.assertEqual(audit.router_action, "memory.recent_discard")
        self.assertEqual(audit.allowlist_result, "allowed")
        self.assertEqual(audit.legacy_command, "memory_boundary")
        self.assertEqual(
            interaction._router_audit_fast_local_final_path(audit),
            "fast_local_takeover.memory.recent_discard",
        )

    def test_fast_local_name_correction_uses_stable_action_path(self):
        from intelligence import interaction

        audit = interaction._new_router_audit("call me Daniel", {})

        with mock.patch.object(interaction, "_execute_command", return_value="Name corrected.") as execute:
            response = interaction._handle_fast_local_takeover(
                "call me Daniel",
                person_id=1,
                person_name="Bret",
                router_audit=audit,
            )

        self.assertEqual(response, "Name corrected.")
        self.assertEqual(execute.call_args.args[0].command_key, "rename_me")
        self.assertEqual(audit.router_action, "identity.name_correction")
        self.assertEqual(audit.allowlist_result, "allowed")
        self.assertEqual(audit.legacy_command, "rename_me")
        self.assertEqual(
            interaction._router_audit_fast_local_final_path(audit),
            "fast_local_takeover.identity.name_correction",
        )


if __name__ == "__main__":
    unittest.main()
