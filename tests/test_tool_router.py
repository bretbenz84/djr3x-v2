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


if __name__ == "__main__":
    unittest.main()
