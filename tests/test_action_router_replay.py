import json
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest import mock


@dataclass(frozen=True)
class RouterReplayCase:
    utterance: str
    router_action: Optional[str]
    expected_allowlist_result: str
    expected_final_path: str
    expected_legacy_command: Optional[str] = None
    expected_router_action: Optional[str] = None
    confidence: float = 0.95
    args: dict = field(default_factory=dict)
    explicit_fast_path: bool = False
    active_game: bool = False
    # humor.* / performance.* migrated to the live tool router 2026-08-13, so
    # ONLINE their fast lane stands down and the reply call picks the tool. These
    # cases still pin the OFFLINE lane, where the local model gets no tools and
    # the deterministic classifier remains the only thing that performs
    # (docs/tool_router_scope.md 2.4). The online contract is pinned separately in
    # tests/test_regex_routing_guards.py::HumorPerformanceToolMigrationTest.
    offline: bool = False


class ActionRouterReplayTests(unittest.TestCase):
    def _legacy_context(self, utterance: str, *, active_game: bool = False) -> dict:
        from intelligence import command_parser

        match = command_parser.parse(utterance)
        legacy = None
        if match is not None:
            legacy = {
                "command_key": match.command_key,
                "match_type": match.match_type,
                "args": match.args,
            }
        return {
            "active_game": active_game,
            "legacy": {
                "command_key": legacy.get("command_key") if legacy else None,
                "command_match": legacy,
            },
        }

    def _decision_for_case(self, case: RouterReplayCase, context: dict):
        from intelligence import action_router

        if case.explicit_fast_path:
            return action_router.decide(case.utterance, context)
        return action_router._apply_context_overrides(
            action_router.ActionDecision(
                action=case.router_action,
                confidence=case.confidence,
                args=dict(case.args),
                reason="router replay fixture",
            ),
            case.utterance,
            context,
        )

    def _run_case(self, case: RouterReplayCase) -> dict:
        from intelligence import command_parser, interaction

        context = self._legacy_context(case.utterance, active_game=case.active_game)
        audit = interaction._new_router_audit(case.utterance, context)
        legacy_match = command_parser.parse(case.utterance)
        response = None

        with (
            mock.patch("intelligence.connectivity.is_offline", return_value=case.offline),
            mock.patch.object(interaction, "_handle_classified_intent", return_value="classified response"),
            mock.patch.object(interaction, "_execute_command", return_value="command response"),
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
            mock.patch("sequences.animations.play_body_beat"),
            mock.patch("features.games.is_active", return_value=case.active_game),
            mock.patch("features.games.handle_input", return_value="game response"),
            mock.patch("features.games.on_response_spoken"),
            mock.patch("features.games.consume_pending_audio_after_response", return_value=None),
            mock.patch.object(interaction, "_handle_name_update_request", return_value="name response") as name_update,
            mock.patch.object(interaction, "_execute_memory_boundary_command", return_value="recent discard response") as recent_discard,
        ):
            if case.explicit_fast_path:
                response = interaction._handle_fast_local_takeover(
                    case.utterance,
                    person_id=1,
                    person_name="Bret",
                    router_audit=audit,
                )
                final_path = (
                    interaction._router_audit_fast_local_final_path(audit)
                    if response
                    else "response_text.unknown"
                )
            else:
                decision = self._decision_for_case(case, context)
                interaction._router_audit_note_decision(audit, decision)
                block_reason = interaction._router_execution_block_reason(decision)
                if block_reason is None:
                    response = interaction._handle_router_takeover_action(
                        decision,
                        case.utterance,
                        person_id=1,
                        person_name="Bret",
                        raw_best_id=1,
                        raw_best_name="Bret",
                        raw_best_score=0.99,
                        router_audit=audit,
                    )
                    final_path = (
                        f"router_takeover.{decision.action}"
                        if response
                        else "response_text.unknown"
                    )
                elif legacy_match is not None:
                    response = interaction._execute_command(
                        legacy_match,
                        1,
                        "Bret",
                        case.utterance,
                    )
                    final_path = f"legacy_command.{legacy_match.command_key}"
                else:
                    response = "llm response"
                    final_path = "llm.stream"

            with mock.patch.object(interaction._log, "info") as info:
                interaction._log_router_audit(
                    audit,
                    final_path,
                    completed=bool(response),
                    spoken_text=response,
                )

        info.assert_called_once()
        payload = json.loads(info.call_args.args[1])
        payload["_name_update_calls"] = [
            call.args for call in name_update.call_args_list
        ]
        payload["_recent_discard_calls"] = [
            call.args for call in recent_discard.call_args_list
        ]
        return payload

    def test_replay_core_router_outcomes_before_more_promotions(self):
        cases = [
            RouterReplayCase(
                utterance="that's not Bret, I'm Daniel",
                router_action="identity.name_correction",
                expected_allowlist_result="allowed",
                expected_legacy_command=None,
                expected_final_path="router_takeover.identity.name_correction",
                args={"name": "Daniel"},
            ),
            RouterReplayCase(
                utterance="forget I said that",
                router_action="memory.recent_discard",
                expected_allowlist_result="allowed",
                expected_legacy_command="memory_boundary",
                expected_final_path="fast_local_takeover.memory.recent_discard",
                explicit_fast_path=True,
                offline=True,  # see RouterReplayCase.offline
            ),
            RouterReplayCase(
                utterance="act embarrassed",
                router_action="performance.mood_pose",
                expected_allowlist_result="allowed",
                expected_legacy_command=None,
                expected_final_path="fast_local_takeover.performance.mood_pose",
                explicit_fast_path=True,
                offline=True,  # see RouterReplayCase.offline
                
            ),
            RouterReplayCase(
                utterance="remember what you see",
                router_action="vision.snapshot",
                expected_allowlist_result="not_executable",
                expected_legacy_command=None,
                expected_final_path="llm.stream",
            ),
            RouterReplayCase(
                utterance="do a victory dance",
                router_action="performance.body_beat",
                expected_allowlist_result="allowed",
                expected_legacy_command=None,
                expected_final_path="fast_local_takeover.performance.body_beat",
                explicit_fast_path=True,
                offline=True,  # see RouterReplayCase.offline
                
            ),
            RouterReplayCase(
                utterance="look at this",
                router_action="conversation.reply",
                expected_allowlist_result="not_executable",
                expected_legacy_command="directed_look",
                expected_final_path="legacy_command.directed_look",
            ),
            RouterReplayCase(
                utterance="look to your left",
                router_action=None,
                expected_allowlist_result="not_run",
                expected_legacy_command="directed_look",
                expected_final_path="legacy_command.directed_look",
                explicit_fast_path=True,
            ),
        ]

        for case in cases:
            with self.subTest(utterance=case.utterance):
                payload = self._run_case(case)

                self.assertEqual(payload["utterance"], case.utterance)
                self.assertEqual(
                    payload["router_action"],
                    case.expected_router_action or case.router_action,
                )
                self.assertEqual(
                    payload["allowlist_result"],
                    case.expected_allowlist_result,
                )
                self.assertEqual(
                    payload["legacy_command"],
                    case.expected_legacy_command,
                )
                self.assertEqual(
                    payload["final_executed_path"],
                    case.expected_final_path,
                )
                self.assertTrue(payload["completed"])
                self.assertIsNone(payload["handler_error"])
                self.assertTrue(payload["spoken_text_present"])

                if case.expected_final_path == "router_takeover.identity.name_correction":
                    self.assertEqual(len(payload["_name_update_calls"]), 1)
                    self.assertEqual(payload["_name_update_calls"][0][0], "that's not Bret, I'm Daniel")
                if case.expected_final_path == "fast_local_takeover.memory.recent_discard":
                    self.assertEqual(payload["_recent_discard_calls"], [(1,)])


if __name__ == "__main__":
    unittest.main()
