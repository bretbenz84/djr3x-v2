"""
Phase 0 tool-router shadow (intelligence/tool_router.py — docs/tool_router_scope.md).

Pins the contracts that keep the shadow trustworthy: every ActionSpec has a tool
definition (a new action without one fails HERE, not silently at runtime), names
round-trip, conversation.reply is the no-tool default, disabled means no-op, and
the tool-call/none responses parse into the right decision records.
"""

import json
import unittest
from unittest import mock

import config
from intelligence import tool_router
from intelligence.action_router import ACTION_SPECS


class CatalogCoverageTest(unittest.TestCase):
    def test_every_spec_has_a_tool_definition(self):
        missing = [s.key for s in ACTION_SPECS if s.key not in tool_router._TOOL_DEFS]
        self.assertEqual(missing, [], f"add _TOOL_DEFS entries for: {missing}")

    def test_no_orphan_tool_definitions(self):
        keys = {s.key for s in ACTION_SPECS}
        orphans = [k for k in tool_router._TOOL_DEFS if k not in keys]
        self.assertEqual(orphans, [], f"stale _TOOL_DEFS entries: {orphans}")

    def test_schema_shape_and_name_round_trip(self):
        tools = tool_router.tool_schemas()
        # reply is represented by "no tool call", so: every spec except reply.
        self.assertEqual(len(tools), len(ACTION_SPECS) - 1)
        names = [t["function"]["name"] for t in tools]
        self.assertEqual(len(names), len(set(names)), "tool names must be unique")
        for t in tools:
            fn = t["function"]
            self.assertEqual(t["type"], "function")
            self.assertNotIn(".", fn["name"])
            self.assertIn(tool_router._NAME_TO_KEY[fn["name"]],
                          {s.key for s in ACTION_SPECS})
            params = fn["parameters"]
            self.assertEqual(params["type"], "object")
            for req in params["required"]:
                self.assertIn(req, params["properties"])

    def test_reply_is_not_a_tool(self):
        names = {t["function"]["name"] for t in tool_router.tool_schemas()}
        self.assertNotIn("conversation_reply", names)


def _resp(tool_calls):
    msg = mock.Mock()
    msg.tool_calls = tool_calls
    resp = mock.Mock()
    resp.choices = [mock.Mock(message=msg)]
    return resp


class ShadowDecideTest(unittest.TestCase):
    def test_tool_call_maps_back_to_action_key(self):
        call = mock.Mock()
        call.function = mock.Mock()
        call.function.name = "motion_turn"
        call.function.arguments = '{"direction": "left", "degrees": 90}'
        with mock.patch("intelligence.llm_compat.create", return_value=_resp([call])):
            out = tool_router.shadow_decide("Turn to your left.", {})
        self.assertEqual(out["action"], "motion.turn")
        self.assertEqual(out["args"], {"direction": "left", "degrees": 90})

    def test_no_tool_call_means_reply(self):
        with mock.patch("intelligence.llm_compat.create", return_value=_resp([])):
            out = tool_router.shadow_decide("I loved the movie.", {})
        self.assertEqual(out["action"], "conversation.reply")

    def test_error_is_captured_not_raised(self):
        with mock.patch("intelligence.llm_compat.create", side_effect=RuntimeError("api down")):
            out = tool_router.shadow_decide("hello", {})
        self.assertIsNone(out["action"])
        self.assertIn("api down", out["error"])


class StartShadowTest(unittest.TestCase):
    def test_disabled_is_a_no_op(self):
        with mock.patch.object(config, "TOOL_ROUTER_SHADOW_ENABLED", False, create=True), \
             mock.patch.object(tool_router, "shadow_decide") as decide:
            tool_router.start_shadow("hello", {}, "conversation.reply")
        decide.assert_not_called()

    def test_enabled_logs_a_parseable_json_record(self):
        import threading
        done = threading.Event()
        captured = {}

        def fake_decide(text, context):
            return {"action": "music.play", "args": {"music_query": "jazz"}, "secs": 0.42}

        real_info = tool_router._log.info

        def capture(fmt, payload):
            captured["payload"] = payload
            done.set()

        with mock.patch.object(config, "TOOL_ROUTER_SHADOW_ENABLED", True, create=True), \
             mock.patch.object(tool_router, "shadow_decide", side_effect=fake_decide), \
             mock.patch.object(tool_router._log, "info", side_effect=capture):
            tool_router.start_shadow("play some jazz", {"active_game": None}, "music.play")
            self.assertTrue(done.wait(3.0), "shadow thread never logged")
        record = json.loads(captured["payload"])
        self.assertTrue(record["agree"])
        self.assertEqual(record["shipped"], "music.play")
        self.assertEqual(record["tool"], "music.play")
        del real_info  # (kept for clarity that we patched the module logger)


class LiveCutoverTest(unittest.TestCase):
    """Phase 1+2: the live subset rides the lean reply call as native tools."""

    def test_live_tools_are_the_expected_subset_only(self):
        tools = tool_router.live_reply_tools()
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {
            "time_query", "date_query", "weather_query", "status_capabilities",
            "status_uptime", "status_battery", "vision_describe_scene",
            "music_options", "system_sleep", "system_shutdown", "web_search",
            "event_cancel", "memory_query", "identity_who_is_speaking",
            "music_play", "music_stop", "music_skip", "vision_snapshot",
            "identity_name_correction", "memory_forget_person",
            # Phase 2 (2026-08-13): humor + performance. Their regex fast lanes
            # "worked", which is what the routing audit disputed — the classifier
            # decided the whole turn and anything off-pattern became prose.
            "humor_tell_joke", "humor_roast", "humor_free_bit",
            "performance_dj_bit", "performance_body_beat",
            "performance_mood_pose", "performance_impersonate",
            # Phase 2b (2026-08-13): the last regex-owned writes/deletes.
            "memory_forget_specific", "memory_recent_discard",
            "emotional_boundary",
            # Phase 2 games. game_answer is deliberately absent — mid-game
            # answer capture stays deterministic (scope doc 2.2).
            "game_start", "game_stop",
            # Phase 3 motion (2026-08-13), the last family. motion_stop and
            # motion_explore are deliberately absent: 2.2 keeps bare "stop"
            # deterministic forever, and an explore invite already has a
            # purpose-built imperative test plus a minutes-long floor grab.
            "motion_turn", "motion_move", "motion_arc", "motion_come",
        })

    def test_physical_performance_tools_carry_canonical_enums(self):
        """A free-text beat/pose would reach the servos as a shrug.

        performance_plan coerces an unrecognized name to thinking_tilt/thinking,
        so the schema is where an invented gesture has to become impossible.
        """
        from intelligence import performance_plan

        byname = {t["function"]["name"]: t["function"] for t in tool_router.live_reply_tools()}
        beat = byname["performance_body_beat"]["parameters"]
        self.assertEqual(
            set(beat["properties"]["body_beat"]["enum"]),
            set(performance_plan.BODY_BEAT_NAMES),
        )
        self.assertEqual(beat["required"], ["body_beat"])
        pose = byname["performance_mood_pose"]["parameters"]
        self.assertEqual(
            set(pose["properties"]["mood"]["enum"]),
            set(performance_plan.MOOD_POSE_NAMES),
        )

    def test_impersonate_arg_is_target_across_every_router(self):
        # The tool def said "who" while ActionSpec, the JSON-prose prompt and the
        # regex classifier all said target — the same arg-name drift class as the
        # documented tool_args/args bug.
        byname = {t["function"]["name"]: t["function"] for t in tool_router.live_reply_tools()}
        params = byname["performance_impersonate"]["parameters"]
        self.assertIn("target", params["properties"])
        self.assertEqual(params["required"], ["target"])

    def test_kill_switch_detaches_all_tools(self):
        with mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", False, create=True):
            self.assertIsNone(tool_router.live_reply_tools())

    def test_resolve_refuses_non_live_actions(self):
        # game_answer is a valid catalog tool but NOT live — must never execute.
        # (This used to use motion_turn, which went live in Phase 3.) game_answer
        # is the right stand-in now: scope doc 2.2 keeps mid-game answer capture
        # deterministic, so it is a tool that must never resolve.
        self.assertIsNone(tool_router.resolve_tool_call("game_answer", "{}"))
        self.assertEqual(
            tool_router.resolve_tool_call("weather_query", "{}"),
            ("weather.query", {}),
        )
        key, args = tool_router.resolve_tool_call("time_query", "not json")
        self.assertEqual((key, args), ("time.query", {}))


def _stream(deltas):
    chunks = []
    for d in deltas:
        delta = mock.Mock()
        delta.content = d.get("content")
        if "tool" in d:
            fn = mock.Mock()
            fn.name = d["tool"]
            fn.arguments = d.get("args", "")
            tc = mock.Mock()
            tc.function = fn
            delta.tool_calls = [tc]
        else:
            delta.tool_calls = None
        chunks.append(mock.Mock(choices=[mock.Mock(delta=delta)]))
    return chunks


class LeanStreamToolTest(unittest.TestCase):
    def _run(self, deltas):
        from intelligence import lean_brain
        with mock.patch.object(lean_brain.llm_compat, "create",
                               return_value=_stream(deltas)), \
             mock.patch.object(lean_brain, "_messages", return_value=[]):
            return list(lean_brain.stream_reply("test utterance", 1))

    def test_tool_only_stream_raises_tool_call_requested(self):
        from intelligence import lean_brain  # noqa: F401
        with self.assertRaises(tool_router.ToolCallRequested) as ctx:
            self._run([{"tool": "weather_query", "args": ""},
                       {"tool": "", "args": "{}"}])
        self.assertEqual(ctx.exception.action, "weather.query")

    def test_prose_stream_yields_normally(self):
        out = self._run([{"content": "Hello "}, {"content": "there."}])
        self.assertEqual("".join(out), "Hello there.")

    def test_prose_wins_when_both_appear(self):
        out = self._run([{"content": "Sure thing."},
                         {"tool": "weather_query", "args": "{}"}])
        self.assertEqual("".join(out), "Sure thing.")

    def test_non_live_tool_call_degrades_to_hiccup_not_execution(self):
        # motion_turn went live in Phase 3 (2026-08-13), so this uses
        # motion_explore — still deliberately NOT live, because a floor-seizing
        # wander should not start from an ambient model read.
        out = self._run([{"tool": "motion_explore", "args": "{}"}])
        self.assertTrue(out and "circuits" in out[0])


class DispatcherTest(unittest.TestCase):
    def test_live_action_dispatches_to_intent_executor(self):
        from intelligence import interaction
        with mock.patch.object(interaction, "_handle_classified_intent",
                               return_value="It is 3 PM.") as handler:
            resp = interaction._execute_tool_routed_action(
                "time.query", {}, "any idea what time it is?", 1)
        handler.assert_called_once()
        self.assertEqual(handler.call_args[0][0], "query_time")
        self.assertEqual(resp, "It is 3 PM.")
        self.assertEqual(interaction._consume_tool_routed_path(),
                         "tool_router.time.query")
        self.assertIsNone(interaction._consume_tool_routed_path())  # one-shot

    def test_executor_decline_falls_back_to_classic_reply(self):
        from intelligence import interaction
        with mock.patch.object(interaction, "_handle_classified_intent",
                               return_value=None), \
             mock.patch.object(interaction.llm, "get_response",
                               return_value="fallback words") as classic, \
             mock.patch.object(interaction, "_speak_blocking") as speak:
            resp = interaction._execute_tool_routed_action(
                "weather.query", {}, "how's it looking", 1)
        classic.assert_called_once()
        speak.assert_called_once_with("fallback words")
        self.assertEqual(resp, "fallback words")
        self.assertIsNone(interaction._consume_tool_routed_path())


class MotionPhase3Test(unittest.TestCase):
    """Phase 3 (docs/tool_router_scope.md §3): motion is the one family where the
    regex KEEPS the first claim and the tool only catches what it missed."""

    def test_motion_is_not_a_detector_demotion(self):
        # Every other migrated family put its key in TOOL_ROUTER_OWNED_ACTIONS so the
        # classifier stops claiming the turn. Motion must NOT: §3 keeps the >=0.95
        # fast lane executing immediately at today's latency, and 2.2 keeps bare
        # "stop" deterministic forever. This is the test that fails if someone
        # "finishes" the migration by pattern-matching the other four stages.
        from intelligence import action_router
        self.assertEqual(
            [a for a in action_router.TOOL_ROUTER_OWNED_ACTIONS
             if a.startswith("motion")], [])

    def test_live_motion_set_excludes_stop_and_explore(self):
        self.assertEqual(
            sorted(a for a in tool_router.live_actions() if a.startswith("motion")),
            ["motion.arc", "motion.come", "motion.move", "motion.turn"])
        self.assertIsNone(tool_router.resolve_tool_call("motion_stop", "{}"))
        self.assertIsNone(tool_router.resolve_tool_call("motion_explore", "{}"))

    def test_motion_schema_args_are_the_keys_the_executor_reads(self):
        # Arg-name drift, three times over, every one silent: `degrees` (the executor
        # reads `deg`), `distance`+`unit` (it reads `dist_m`) and arc's lone
        # `direction` (it reads ang_dir/lin_dir) all shipped, so a commanded angle or
        # distance became the default and every arc curved forward-LEFT. The move
        # enum said "backward" while the executor tests == "back" and falls through
        # to move_forward — "back up" would have driven him FORWARD.
        byname = {t["function"]["name"]: t["function"]
                  for t in tool_router.live_reply_tools()}
        turn = byname["motion_turn"]["parameters"]["properties"]
        self.assertEqual(set(turn), {"direction", "deg"})
        self.assertIn("DEGREES", turn["deg"]["description"])
        move = byname["motion_move"]["parameters"]["properties"]
        self.assertEqual(set(move), {"direction", "dist_m"})
        self.assertEqual(move["direction"]["enum"], ["forward", "back"])
        self.assertIn("METRES", move["dist_m"]["description"])
        arc = byname["motion_arc"]["parameters"]["properties"]
        self.assertEqual(set(arc), {"ang_dir", "lin_dir", "small"})

    def test_dispatcher_translates_tool_args_to_executor_keys(self):
        from intelligence import interaction
        self.assertEqual(
            interaction._motion_args_from_tool(
                "motion.move", {"direction": "backward", "distance": 2,
                                "unit": "feet"}, "back yourself up two feet"),
            {"direction": "back", "dist_m": 0.6096})
        self.assertEqual(
            interaction._motion_args_from_tool(
                "motion.turn", {"direction": "right", "degrees": 90}, "turn right"),
            {"direction": "right", "deg": 90.0})
        self.assertEqual(
            interaction._motion_args_from_tool(
                "motion.arc", {"direction": "right"}, "scootch to your right"),
            {"ang_dir": "right", "lin_dir": "forward", "small": False})

    def test_gate_admits_the_commands_the_regex_misses(self):
        # Every one of these classifies as None today and becomes conversation.
        from intelligence import action_router
        for text in ("rotate ninety degrees", "back yourself up a bit",
                     "scoot a little closer", "get closer", "back it up",
                     "hang a left", "face me", "drive up here",
                     "point yourself at the window", "why don't you scoot forward"):
            self.assertIsNone(action_router.classify_explicit_motion(text), text)
            self.assertIsNone(
                action_router.motion_command_refusal_reason(text, "motion.move"),
                text)

    def test_gate_refuses_figurative_motion_that_guards_only_admitted(self):
        # The measurement that decided this design: a gate built only from
        # _MOTION_NEGATED_RE / _MOTION_EXPLANATION_RE / _MOTION_REPORTED_SPEECH_RE
        # admitted 31 of 31 of these, because not one carries a negator, a leading
        # "why", or a speech verb for those guards to catch.
        from intelligence import action_router
        for text in ("I think we should move on from that topic", "let's move on",
                     "moving forward, I want to try something",
                     "I need to run to the store", "she moved forward with the plan",
                     "can you back me up on this", "go right ahead and tell him",
                     "come to think of it, that's wrong", "my head is spinning",
                     "I'm going to head out soon"):
            self.assertEqual(
                action_router.motion_command_refusal_reason(text, "motion.move"),
                "missing_motion_command_evidence", text)

    def test_bare_stop_never_migrates(self):
        # scope doc 2.2. The deterministic escape owns these
        # (interaction._errand_stop_demanded + motion_controller.is_moving()), and an
        # LLM-chosen motion.stop still has to satisfy classify_explicit_motion.
        from intelligence import action_router
        for text in ("stop", "whoa stop", "hold up", "cut it out"):
            self.assertEqual(
                action_router.motion_command_refusal_reason(text, "motion.stop"),
                "missing_motion_command_evidence", text)
        self.assertIsNone(
            action_router.motion_command_refusal_reason("stop moving", "motion.stop"))


if __name__ == "__main__":
    unittest.main()
