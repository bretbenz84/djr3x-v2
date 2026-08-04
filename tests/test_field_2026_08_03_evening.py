"""Fixes from the 2026-08-03 17:55 field session.

  * TOTAL DEAFNESS: the first live web.search tool call crashed the turn —
    ToolCallRequested stored its arguments on ``args``, BaseException's
    reserved attribute, which silently coerces a dict to a tuple of its KEYS
    (``('query',)``); the executor's ``.get`` raised AttributeError and the
    exception killed the listening loop thread. Wake word kept hearing
    ("Hey_rex" 0.883) but no turn was ever processed again — manual shutdown.
    Every argument-less tool had masked the bug (``()`` is falsy).
  * "52%": "No, I mean, what percentage is your battery at?" classified as
    query_battery but the answer_to_rex frame blocked it, and the
    conversational LLM invented a percentage.
  * Charging: on the bench charger the handler recited charger voltage as if
    it were fill state.
  * "Brad's 'maintaining his freedom' thing": Brad corrected his name to JT
    the day before, but the diary froze "Brad" into episode text — the rename
    never propagated into rex.db.
  * "I heard the correction, but I need one clear fact to update": a story
    ELABORATION ("Actually, they changed the deadline on him...") was
    hijacked by the memory-correction executor's canned fallthrough.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from intelligence import dialogue_act
from intelligence import tool_router
from memory import database as db


class ToolCallArgsNotClobberedTest(unittest.TestCase):
    def test_tool_args_survive_as_dict(self):
        exc = tool_router.ToolCallRequested("web.search", {"query": "top headlines"})
        self.assertEqual(exc.tool_args, {"query": "top headlines"})
        self.assertEqual(exc.action, "web.search")

    def test_raise_and_catch_roundtrip(self):
        try:
            raise tool_router.ToolCallRequested("web.search", {"query": "x"})
        except tool_router.ToolCallRequested as tc:
            self.assertEqual(tc.tool_args.get("query"), "x")

    def test_no_consumer_reads_the_clobbered_attribute(self):
        """BaseException.args coerces a dict to a tuple of keys — nothing may
        ever read tool arguments from .args again."""
        import re
        src = Path("intelligence/interaction.py").read_text()
        for m in re.finditer(r"except .*ToolCallRequested as (\w+)", src):
            var = m.group(1)
            self.assertNotIn(
                f"{var}.args", src,
                f"a ToolCallRequested consumer reads the clobbered .{var}.args",
            )


class BatteryAnswerFrameBreakoutTest(unittest.TestCase):
    def _answer_frame(self):
        return dialogue_act.DialogueActDecision(
            label="answer_to_rex", confidence=0.9, reason="reply to last Rex turn"
        )

    def test_battery_question_breaks_out_of_answer_frame(self):
        from intelligence import interaction as I
        self.assertTrue(I._dialogue_allows_action_breakout(
            "status.battery",
            "No, I mean, what percentage is your battery at?",
            self._answer_frame(),
        ))

    def test_plain_answer_stays_bound(self):
        from intelligence import interaction as I
        self.assertFalse(I._dialogue_allows_action_breakout(
            "status.battery", "It's a Delorean.", self._answer_frame(),
        ))

    def test_shutdown_request_breaks_out_of_answer_frame(self):
        from intelligence import interaction as I
        self.assertTrue(I._dialogue_allows_action_breakout(
            "system.shutdown",
            "I would like you to shut down.",
            self._answer_frame(),
        ))


class ChargingHonestyTest(unittest.TestCase):
    def _handler_prompt(self, mv):
        from intelligence import interaction as I
        from intelligence import battery_awareness
        with mock.patch.object(battery_awareness, "current_mv", return_value=mv), \
             mock.patch.object(I.llm, "get_response", side_effect=lambda p, *a, **k: p), \
             mock.patch.object(I, "_speak_blocking", return_value=True):
            return I._handle_classified_intent(
                "query_battery", "what is your battery state of charge?", None
            )

    def test_on_charger_reports_charging_not_voltage(self):
        prompt = self._handler_prompt(13950)
        self.assertIn("ON THE CHARGER", prompt)
        self.assertIn("not your fill level", prompt)
        self.assertIn("NEVER invent a percentage", prompt)
        self.assertNotIn("13.95", prompt)


class RenamePropagationTest(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        from memory import rex_db
        self._tmp = tempfile.TemporaryDirectory()
        people_path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(people_path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (5, 'Brad')")
        self._p1 = mock.patch.object(db, "_DB_FILE", people_path)
        self._p1.start()
        rex_path = str(Path(self._tmp.name) / "rex.db")
        self._p2 = mock.patch.object(config, "REX_DB_PATH", rex_path, create=True)
        self._p2.start()
        rex_db.ensure_schema()
        self.rex_db = rex_db

    def tearDown(self):
        self._p1.stop()
        self._p2.stop()
        self._tmp.cleanup()

    def test_rename_rewrites_diary_episodes(self):
        from memory import people
        detail = json.dumps({
            "open_threads": ["whether Brad's freedom project is going well"],
        })
        self.rex_db.execute(
            "INSERT INTO rex_episodes (created_at, kind, summary, person_id, "
            "person_name, detail) VALUES ('2026-08-02 13:55:00', "
            "'conversation_summary', 'Brad shared that he is maintaining his "
            "freedom.', 5, 'Brad', ?)",
            (detail,),
        )
        self.assertTrue(people.rename_person(5, "JT"))
        row = self.rex_db.fetchone(
            "SELECT summary, person_name, detail FROM rex_episodes WHERE person_id = 5"
        )
        self.assertEqual(row["person_name"], "JT")
        self.assertIn("JT shared", row["summary"])
        self.assertNotIn("Brad", row["summary"])
        threads = json.loads(row["detail"])["open_threads"]
        self.assertEqual(threads, ["whether JT's freedom project is going well"])


class CorrectionFallthroughTest(unittest.TestCase):
    def test_story_elaboration_gets_real_reply_not_canned_line(self):
        from intelligence import interaction as I
        spoken = []
        with mock.patch.object(
            I.llm, "get_response",
            return_value="Moving the deadline after he made it? That's rigged.",
        ), mock.patch.object(
            I, "_speak_blocking", side_effect=lambda t, *a, **k: spoken.append(t) or True
        ), mock.patch.object(
            I, "_extract_memory_statement_target",
            return_value=(None, None, "", False),
        ):
            resp = I._execute_memory_correct_fact_command(
                {"correction": "Actually, they changed the deadline on him"},
                1, "Bret",
            )
        self.assertNotIn("one clear fact", resp)
        self.assertIn("rigged", resp)
        self.assertEqual(spoken, [resp])


if __name__ == "__main__":
    unittest.main()
