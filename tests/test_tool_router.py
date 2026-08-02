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
    """Phase 1: the live subset rides the lean reply call as native tools."""

    def test_live_tools_are_the_phase1_subset_only(self):
        tools = tool_router.live_reply_tools()
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {
            "time_query", "date_query", "weather_query", "status_capabilities",
            "status_uptime", "vision_describe_scene", "music_options",
            "system_sleep", "system_shutdown", "web_search",
            "event_cancel", "memory_query", "identity_who_is_speaking",
            "music_play", "music_stop", "music_skip", "vision_snapshot",
        })

    def test_kill_switch_detaches_all_tools(self):
        with mock.patch.object(config, "TOOL_ROUTER_LIVE_ENABLED", False, create=True):
            self.assertIsNone(tool_router.live_reply_tools())

    def test_resolve_refuses_non_live_actions(self):
        # motion_turn is a valid catalog tool but NOT live — must never execute.
        self.assertIsNone(tool_router.resolve_tool_call("motion_turn", "{}"))
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
        out = self._run([{"tool": "motion_turn", "args": '{"direction":"left"}'}])
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


if __name__ == "__main__":
    unittest.main()
