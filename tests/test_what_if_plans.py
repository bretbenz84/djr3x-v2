"""
What-if / plans feature: when the user states a plan, Rex asks a clarifying question if
details are sparse, offers a concrete suggestion once a place is known (same turn or via
the clarify→answer handoff), and suggests something near WEATHER_LOCATION when there are
no plans — instead of a generic "that sounds fun" riff.

qwen confirm is disabled here so the regex path is tested deterministically.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import plan_intent
from intelligence import conversation_agenda as ca
from intelligence.turn_plan import TurnPlan


class PlanIntentClassifyTest(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(config, "PLAN_INTENT_QWEN_CONFIRM_ENABLED", False)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _c(self, text):
        return plan_intent.classify(text)

    def test_sparse_plan(self):
        r = self._c("I'm going camping this weekend")
        self.assertTrue(r["is_plan"])
        self.assertEqual(r["specificity"], "sparse")
        self.assertEqual(r["plan_key"], "camping")

    def test_specific_plan_named_place(self):
        r = self._c("I'm going camping at Fraser Flats")
        self.assertTrue(r["is_plan"])
        self.assertEqual(r["specificity"], "specific")
        self.assertEqual(r["place"], "Fraser Flats")

    def test_no_plans(self):
        r = self._c("I have no plans this weekend")
        self.assertTrue(r["is_no_plans"])
        self.assertFalse(r["is_plan"])

    def test_mundane_going_to_bed_is_not_a_plan(self):
        self.assertFalse(self._c("I'm going to bed")["is_plan"])

    def test_non_plan_statement(self):
        r = self._c("I love pizza")
        self.assertFalse(r["is_plan"])
        self.assertFalse(r["is_no_plans"])

    def test_failure_safe_returns_default(self):
        with mock.patch.object(plan_intent, "_extract_place", side_effect=RuntimeError):
            r = self._c("I'm going to Yosemite tomorrow")
            self.assertFalse(r["is_plan"])  # error → safe default


class PlanBranchTest(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(config, "PLAN_INTENT_QWEN_CONFIRM_ENABLED", False)
        self._patch.start()
        ca.reset_plans_state()

    def tearDown(self):
        self._patch.stop()
        ca.reset_plans_state()

    def _branch(self, text, person_id=1):
        plan, lines = TurnPlan(), []
        out = ca._plan_branch(plan, lines, text, person_id)
        return out, plan, lines

    def test_sparse_plan_asks_a_clarifier(self):
        out, plan, lines = self._branch("I'm going camping this weekend")
        self.assertIsNotNone(out)
        # explicit_followup is DERIVED from the directive (validates it matches the
        # question-allow patterns), so the one clarifier survives a full question budget.
        self.assertTrue(plan.explicit_followup)
        joined = " ".join(lines).lower()
        self.assertIn("clarifying follow-up question", joined)
        self.assertIn("do not give a generic", joined)
        # the clarify armed the cross-turn handoff
        self.assertIn(1, ca._pending_plan_clarify)

    def test_specific_plan_offers_a_suggestion(self):
        out, plan, lines = self._branch("I'm going to Yosemite next week")
        self.assertIsNotNone(out)
        self.assertTrue(plan.explicit_followup)
        self.assertIn("what if", " ".join(lines).lower())

    def test_no_plans_suggests_local_activity(self):
        out, plan, lines = self._branch("I have no plans this weekend")
        self.assertIsNotNone(out)
        self.assertTrue(plan.explicit_followup)
        loc = config.WEATHER_LOCATION.split(",")[0]
        self.assertIn(loc.lower(), " ".join(lines).lower())

    def test_dedupe_second_same_plan_falls_back(self):
        self.assertIsNotNone(self._branch("I'm going camping this weekend")[0])
        # same plan_key again → None (caller uses the generic acknowledgment, no nag)
        self.assertIsNone(self._branch("I'm going camping again")[0])

    def test_feature_flag_off_falls_back(self):
        with mock.patch.object(config, "WHAT_IF_PLANS_ENABLED", False):
            self.assertIsNone(self._branch("I'm going camping this weekend")[0])

    def test_not_a_plan_falls_back(self):
        self.assertIsNone(self._branch("I love pizza")[0])


class PlanClarifyHandoffTest(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(config, "PLAN_INTENT_QWEN_CONFIRM_ENABLED", False)
        self._patch.start()
        ca.reset_plans_state()

    def tearDown(self):
        self._patch.stop()
        ca.reset_plans_state()

    def test_answer_to_clarifier_triggers_suggestion(self):
        # Turn 1: sparse plan → clarify (arms pending)
        ca._plan_branch(TurnPlan(), [], "I'm going camping this weekend", 1)
        # Turn 2: the place answer
        plan, lines = TurnPlan(), []
        out = ca._plan_clarify_answer(
            plan, lines, "Fraser Flats", 1,
            {"answer_text": "Fraser Flats", "question_text": "Where are you camping?"},
        )
        self.assertIsNotNone(out)
        self.assertTrue(plan.explicit_followup)
        self.assertIn("fraser flats", " ".join(lines).lower())
        # pending consumed
        self.assertNotIn(1, ca._pending_plan_clarify)

    def test_no_pending_returns_none(self):
        out = ca._plan_clarify_answer(TurnPlan(), [], "Fraser Flats", 1, {"answer_text": "x"})
        self.assertIsNone(out)

    def test_expired_pending_returns_none(self):
        ca._plan_branch(TurnPlan(), [], "I'm going camping this weekend", 1)
        ca._pending_plan_clarify[1]["at"] -= 10_000  # force past TTL
        out = ca._plan_clarify_answer(TurnPlan(), [], "Fraser Flats", 1, {"answer_text": "x"})
        self.assertIsNone(out)

    def test_reset_clears_state(self):
        ca._plan_branch(TurnPlan(), [], "I'm going camping this weekend", 1)
        ca.reset_plans_state()
        self.assertEqual(ca._pending_plan_clarify, {})
        self.assertEqual(ca._plans_clarified, set())


if __name__ == "__main__":
    unittest.main()
