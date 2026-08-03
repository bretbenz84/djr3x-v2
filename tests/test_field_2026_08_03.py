"""Fixes from the 2026-08-02 23:55 field session.

Three verified failures:
  * "I will talk to you later, and I would like you to shut down." — the LLM
    router scored it conversation (0.20), the closure-cue agenda took over, and
    the reply model generated "Powering down." as a FAREWELL QUIP without
    powering down (the desire-form leader defeated the parser, and the
    standalone dispatch guard would have bounced it anyway).
  * "What's your state of charge?" → capabilities list; "how are your
    batteries?" → "I don't have a battery meter" — a capability lie; there was
    simply no battery action to route to.
  * "The data tracking thing came up the other day — did JT ever make peace
    with it?" — a guest's privacy request, stored as a diary open thread BEFORE
    the write-side bookkeeping filter shipped, fired as a callback.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from intelligence import action_router as AR
from intelligence import command_parser as cp
from intelligence import intent_classifier


class ShutdownRequestParsingTest(unittest.TestCase):
    def test_desire_form_directives_accepted(self):
        for text in (
            "I will talk to you later, and I would like you to shut down.",
            "I would like you to shut down.",
            "I'd like you to shut down",
            "I want you to power down",
            "I need you to shut down now",
            "Can you shut down, please?",
            "Shut down.",
        ):
            self.assertTrue(cp.is_shutdown_request(text), text)

    def test_guards_still_hold(self):
        for text in (
            "don't shut down",
            "can you shut down the music",
            "I would like you to shut down the music",
            "why would you shut down",
            "should I shut down the server",
            "I had to shut down my old server yesterday",
        ):
            self.assertFalse(cp.is_shutdown_request(text), text)


class RouterShutdownPrePassTest(unittest.TestCase):
    def _decide_no_llm(self, text):
        calls = []

        def _fake_create(**kw):
            calls.append(1)
            raise RuntimeError("llm consulted")

        with mock.patch.object(
            AR._client.chat.completions, "create", side_effect=_fake_create
        ):
            decision = AR.decide(text, {})
        return decision, len(calls)

    def test_compound_farewell_routes_to_shutdown_without_llm(self):
        decision, llm_calls = self._decide_no_llm(
            "I will talk to you later, and I would like you to shut down."
        )
        self.assertEqual(decision.action, "system.shutdown")
        self.assertEqual(llm_calls, 0)
        self.assertGreaterEqual(decision.confidence, 0.9)

    def test_object_scoped_shutdown_not_prerouted(self):
        decision, _ = self._decide_no_llm("can you shut down the music")
        self.assertNotEqual(decision.action, "system.shutdown")

    def test_shutdown_in_execute_allowlist(self):
        self.assertIn("system.shutdown", config.ACTION_ROUTER_EXECUTE_ACTIONS)


class BatteryIntentTest(unittest.TestCase):
    def test_battery_questions_classify(self):
        for text in (
            "What's your state of charge?",
            "No, I was asking how how are your batteries? What's the charge level?",
            "how's your battery doing",
            "are you charging?",
            "what's your battery level",
        ):
            self.assertEqual(
                intent_classifier.classify_deterministic(text), "query_battery", text
            )

    def test_other_batteries_stay_general(self):
        for text in (
            "my phone battery died again",
            "I need to buy batteries for the remote",
            "the car battery is toast",
        ):
            self.assertNotEqual(
                intent_classifier.classify_deterministic(text), "query_battery", text
            )

    def test_battery_action_fully_wired(self):
        """The six-point checklist, greppable: catalog spec, evidence gate,
        self-query skip, execute allowlist, tool-router catalog + live set,
        intent-action map."""
        from intelligence import tool_router
        from intelligence import interaction as I
        self.assertIn("status.battery", {s.key for s in AR.ACTION_SPECS})
        self.assertIsNone(AR.missing_required_evidence_reason(
            "what's your state of charge?",
            AR.ActionDecision(action="status.battery", confidence=0.9),
        ))
        self.assertEqual(
            AR._SELF_QUERY_SKIP_INTENTS.get("query_battery"), "status.battery")
        self.assertIn("status.battery", config.ACTION_ROUTER_EXECUTE_ACTIONS)
        self.assertIn("status.battery", config.TOOL_ROUTER_LIVE_ACTIONS)
        self.assertIn("status.battery", tool_router._TOOL_DEFS)
        self.assertIn("status.battery", tool_router._DEFAULT_LIVE_ACTIONS)
        self.assertEqual(I._INTENT_ACTION_MAP.get("query_battery"), "status.battery")

    def test_router_self_query_skip_saves_llm_call(self):
        calls = []

        def _fake_create(**kw):
            calls.append(1)
            raise RuntimeError("llm consulted")

        with mock.patch.object(
            AR._client.chat.completions, "create", side_effect=_fake_create
        ):
            decision = AR.decide("What's your state of charge?", {})
        self.assertEqual(decision.action, "conversation.reply")
        self.assertIn("self-query", decision.reason)
        self.assertEqual(len(calls), 0)


class BatteryHandlerTest(unittest.TestCase):
    def _handler_prompt(self, mv):
        from intelligence import interaction as I
        from intelligence import battery_awareness
        captured = {}

        def _fake_say(prompt, *a, **k):
            captured["prompt"] = prompt
            return "spoken"

        with mock.patch.object(battery_awareness, "current_mv", return_value=mv), \
             mock.patch.object(I.llm, "get_response", side_effect=lambda p, *a, **k: p), \
             mock.patch.object(I, "_speak_blocking", return_value=True):
            resp = I._handle_classified_intent(
                "query_battery", "what's your state of charge?", None
            )
        return resp

    def test_no_telemetry_answers_honestly_without_capability_lie(self):
        prompt = self._handler_prompt(-1)
        self.assertIn("drive base", prompt)
        self.assertIn("isn't connected", prompt)
        self.assertIn("do NOT invent a number", prompt)

    def test_nominal_pack_reports_voltage_and_band(self):
        prompt = self._handler_prompt(13100)
        self.assertIn("13.10 volts", prompt)
        self.assertIn("'nominal'", prompt)


class StoredBookkeepingThreadTest(unittest.TestCase):
    """Read-time filter: threads stored BEFORE the write-side guard shipped
    (the JT data-tracking thread) must die at pending_for_person."""

    def setUp(self):
        from memory import rex_db
        self._tmp = tempfile.TemporaryDirectory()
        path = str(Path(self._tmp.name) / "rex.db")
        self._p = mock.patch.object(config, "REX_DB_PATH", path, create=True)
        self._p.start()
        rex_db.ensure_schema()
        self.rex_db = rex_db

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def _store_episode(self, threads):
        import json
        from datetime import datetime, timedelta
        created = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        self.rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, person_id, detail)"
            " VALUES (?, 'conversation_summary', 'diary note', 1, ?)",
            (created, json.dumps({"open_threads": threads})),
        )

    def test_stored_bookkeeping_threads_filtered_at_read(self):
        from intelligence import open_threads
        self._store_episode([
            "whether JT made peace with the data tracking thing",
            "whether JT's name change settled in",
            "whether the camping trip happened",
        ])
        pending = open_threads.pending_for_person(1)
        threads = [p["thread"] for p in pending]
        self.assertEqual(threads, ["whether the camping trip happened"])


if __name__ == "__main__":
    unittest.main()
