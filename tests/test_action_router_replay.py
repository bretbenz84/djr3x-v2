import json
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest import mock


@dataclass(frozen=True)
class RouterReplayCase:
    utterance: str
    router_action: str
    expected_allowlist_result: str
    expected_final_path: str
    expected_legacy_command: Optional[str] = None
    confidence: float = 0.95
    args: dict = field(default_factory=dict)
    explicit_fast_path: bool = False


class ActionRouterReplayTests(unittest.TestCase):
    def _legacy_context(self, utterance: str) -> dict:
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

        context = self._legacy_context(case.utterance)
        audit = interaction._new_router_audit(case.utterance, context)
        legacy_match = command_parser.parse(case.utterance)
        response = None

        with (
            mock.patch.object(interaction, "_handle_classified_intent", return_value="classified response") as classified,
            mock.patch.object(interaction, "_execute_command", return_value="command response") as execute_command,
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
            mock.patch("sequences.animations.play_body_beat"),
        ):
            if case.explicit_fast_path:
                response = interaction._handle_fast_local_takeover(
                    case.utterance,
                    person_id=1,
                    person_name="Bret",
                    router_audit=audit,
                )
                decision_action = audit.router_action
                final_path = (
                    f"fast_local_takeover.{decision_action}"
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
        payload["_classified_calls"] = [
            call.args[0] for call in classified.call_args_list
        ]
        payload["_executed_command_keys"] = [
            call.args[0].command_key for call in execute_command.call_args_list
        ]
        return payload

    def test_replay_core_router_outcomes_before_more_promotions(self):
        cases = [
            RouterReplayCase(
                utterance="what music can you play?",
                router_action="music.options",
                expected_allowlist_result="allowed",
                expected_legacy_command=None,
                expected_final_path="router_takeover.music.options",
            ),
            RouterReplayCase(
                utterance="skip this",
                router_action="music.skip",
                expected_allowlist_result="allowed",
                expected_legacy_command="dj_skip",
                expected_final_path="router_takeover.music.skip",
            ),
            RouterReplayCase(
                utterance="what do you see?",
                router_action="vision.describe_scene",
                expected_allowlist_result="allowed",
                expected_legacy_command="vision_describe",
                expected_final_path="router_takeover.vision.describe_scene",
            ),
            RouterReplayCase(
                utterance="play jazz",
                router_action="music.play",
                expected_allowlist_result="not_in_execute_allowlist",
                expected_legacy_command=None,
                expected_final_path="llm.stream",
                args={"music_query": "jazz"},
            ),
            RouterReplayCase(
                utterance="forget I like Star Wars",
                router_action="memory.forget_specific",
                expected_allowlist_result="not_in_execute_allowlist",
                expected_legacy_command="forget_specific",
                expected_final_path="legacy_command.forget_specific",
                args={"target": "I like Star Wars"},
            ),
            RouterReplayCase(
                utterance="do a victory dance",
                router_action="performance.body_beat",
                expected_allowlist_result="allowed",
                expected_legacy_command=None,
                expected_final_path="fast_local_takeover.performance.body_beat",
                explicit_fast_path=True,
            ),
            RouterReplayCase(
                utterance="look at this",
                router_action="conversation.reply",
                expected_allowlist_result="not_executable",
                expected_legacy_command="directed_look",
                expected_final_path="legacy_command.directed_look",
            ),
        ]

        for case in cases:
            with self.subTest(utterance=case.utterance):
                payload = self._run_case(case)

                self.assertEqual(payload["utterance"], case.utterance)
                self.assertEqual(payload["router_action"], case.router_action)
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

                if case.expected_final_path == "router_takeover.music.options":
                    self.assertEqual(payload["_classified_calls"], ["query_music_options"])
                if case.expected_final_path == "router_takeover.vision.describe_scene":
                    self.assertEqual(payload["_classified_calls"], ["query_what_do_you_see"])
                if case.expected_final_path == "router_takeover.music.skip":
                    self.assertEqual(payload["_executed_command_keys"], ["dj_skip"])


if __name__ == "__main__":
    unittest.main()
