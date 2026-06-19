"""
Tests for Bet 2's TurnPlan — the typed conversation_agenda → social_frame handoff.

The key safety property is EQUIVALENCE: social_frame.build_frame must derive the same
purpose whether it reads the structured TurnPlan (new path) or regex-reparses the
directive string (the retained fallback). If those agree, swapping the live pipeline
onto the plan path is behavior-preserving — which is why every existing string-only
test (which passes no plan → regex path) stays green.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

# Representative agenda directives for the branches whose purpose the agenda now sets
# structurally, paired with the purpose _purpose_from derives from the same string.
_CASES = {
    "closure": (
        "Primary purpose: close the current thread gracefully. Give a brief "
        "acknowledgement or soft final beat, then stop. No new questions, no "
        "unrelated memory hooks, no visual riff."
    ),
    "interest": (
        "Conversation steering: The current thread matches a known/active interest: "
        "'astrophotography'.\nPrimary purpose: deepen the interest thread the human "
        "opened. Give one specific reaction, then ask one natural follow-up."
    ),
    "answer_ack": (
        "Primary purpose: the human just answered a question Rex asked. "
        "Question: 'what are you into?'. Answer: 'astrophotography'. React to the "
        "actual content with genuine, specific interest."
    ),
    "answer": (
        "Primary purpose: answer the human's question directly first. After "
        "answering, ask at most one short follow-up only if it flows from their "
        "question."
    ),
}


class TurnPlanDataclassTest(unittest.TestCase):
    def test_defaults_are_inert(self):
        from intelligence.turn_plan import TurnPlan
        p = TurnPlan()
        self.assertEqual(p.directive, "")
        self.assertIsNone(p.purpose)
        self.assertIsNone(p.explicit_followup)
        self.assertIsNone(p.hard_no_question)


class PurposeEquivalenceTest(unittest.TestCase):
    """build_frame gives the same purpose via TurnPlan as via the directive regex."""

    def _purpose(self, directive, *, plan):
        from intelligence import social_frame as sf
        with (
            mock.patch.object(sf.world_state, "snapshot", return_value={"people": []}),
            mock.patch.object(sf, "_question_budget_allows", return_value=True),
        ):
            return sf.build_frame(
                "some user text",
                None,
                agenda_directive=directive,
                turn_plan=plan,
            ).purpose

    def test_plan_path_matches_regex_path_for_each_branch(self):
        from intelligence.turn_plan import TurnPlan
        for expected, directive in _CASES.items():
            regex_purpose = self._purpose(directive, plan=None)            # fallback path
            plan = TurnPlan(directive=directive, purpose=expected)
            plan_purpose = self._purpose(directive, plan=plan)             # structured path
            self.assertEqual(regex_purpose, expected, f"regex path for {expected}")
            self.assertEqual(plan_purpose, expected, f"plan path for {expected}")
            self.assertEqual(plan_purpose, regex_purpose, f"divergence for {expected}")

    def test_plan_with_no_purpose_falls_back_to_regex(self):
        # A generic turn: the agenda leaves purpose=None, so build_frame must still
        # derive it from the directive/energy exactly as before (no behavior change).
        from intelligence import social_frame as sf
        from intelligence.turn_plan import TurnPlan
        directive = _CASES["answer"]
        plan = TurnPlan(directive=directive, purpose=None)
        with (
            mock.patch.object(sf.world_state, "snapshot", return_value={"people": []}),
            mock.patch.object(sf, "_question_budget_allows", return_value=True),
        ):
            with_plan = sf.build_frame("x", None, agenda_directive=directive, turn_plan=plan)
            without = sf.build_frame("x", None, agenda_directive=directive)
        self.assertEqual(with_plan.purpose, without.purpose)
        self.assertEqual(with_plan.purpose, "answer")


class AgendaPopulatesPlanTest(unittest.TestCase):
    """build_turn_plan sets the structured purpose, and build_turn_directive stays a
    faithful back-compat wrapper (its string == the plan's rendered directive)."""

    @staticmethod
    @contextlib.contextmanager
    def _agenda_mocks():
        from intelligence import conversation_agenda as ca
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                ca.world_state, "snapshot",
                return_value={"people": [], "environment": {}},
            ))
            stack.enter_context(mock.patch.object(
                ca.empathy, "classify_local_sensitivity", return_value=None))
            stack.enter_context(mock.patch.object(ca.empathy, "peek", return_value={}))
            stack.enter_context(mock.patch.object(
                ca.rel_memory, "get_latest_pending_question", return_value=None))
            yield

    def test_answered_question_sets_answer_ack_purpose(self):
        from intelligence import conversation_agenda as ca
        answered = {
            "question_key": "obsession",
            "question_text": "What are you into?",
            "answer_text": "astrophotography",
        }
        with self._agenda_mocks():
            plan = ca.build_turn_plan("astrophotography", 1, answered_question=answered)
        self.assertEqual(plan.purpose, "answer_ack")
        self.assertTrue(plan.explicit_followup)  # earned on-thread follow-up offered
        self.assertIn("just answered a question", plan.directive)

    def test_grounding_correction_branch(self):
        from intelligence import conversation_agenda as ca
        with self._agenda_mocks():
            for text in ("What? That makes no sense", "You mean my telescope?"):
                with self.subTest(text=text):
                    plan = ca.build_turn_plan(text, 1)
                    self.assertEqual(plan.purpose, "grounding_repair")
                    low = plan.directive.lower()
                    self.assertIn("drop that thread", low)
                    self.assertIn("do not re-explain", low)

    def test_grounding_correction_precedes_user_question(self):
        # A correction that is also question-shaped resolves to grounding_repair,
        # not the generic "answer the human's question" purpose.
        from intelligence import conversation_agenda as ca
        with self._agenda_mocks():
            plan = ca.build_turn_plan("What? That makes no sense", 1)
        self.assertEqual(plan.purpose, "grounding_repair")

    def test_build_turn_directive_wrapper_equals_plan_directive(self):
        from intelligence import conversation_agenda as ca
        with self._agenda_mocks():
            plan = ca.build_turn_plan("just a plain statement about my day", 1)
            directive = ca.build_turn_directive("just a plain statement about my day", 1)
        self.assertEqual(directive, plan.directive)


class AllowQuestionEquivalenceTest(unittest.TestCase):
    """build_frame derives the same allow_question via the TurnPlan's explicit_followup
    as via the _explicit_followup_allowed regex, for the branches the agenda migrated."""

    _ANSWERED_DIR = (
        "Primary purpose: the human just answered a question Rex asked. "
        "Question: 'what are your hobbies?'. Answer: 'woodworking'. React to the "
        "actual content with genuine, specific interest. After answering, ask at "
        "most one short follow-up that stays on this exact topic, or carry the turn "
        "with a light roast instead. Do not pivot into a new interview topic."
    )
    _INTEREST_DIR = (
        "Primary purpose: deepen the interest thread the human opened. Give one "
        "specific reaction, then ask one natural follow-up about their experience "
        "with that topic — what got them into it or their favorite part."
    )

    def _equiv(self, directive, *, purpose, answered_question, user_text):
        from intelligence import social_frame as sf
        from intelligence.turn_plan import TurnPlan
        plan = TurnPlan(directive=directive, purpose=purpose, explicit_followup=True)
        with (
            mock.patch.object(sf.world_state, "snapshot", return_value={"people": []}),
            mock.patch.object(sf, "_question_budget_allows", return_value=True),
        ):
            via_plan = sf.build_frame(
                user_text, None, answered_question=answered_question,
                agenda_directive=directive, turn_plan=plan,
            )
            via_regex = sf.build_frame(
                user_text, None, answered_question=answered_question,
                agenda_directive=directive,
            )
        return via_plan, via_regex

    def test_answered_followup_equivalent_and_allowed(self):
        via_plan, via_regex = self._equiv(
            self._ANSWERED_DIR, purpose="answer_ack",
            answered_question={"question_key": "hobbies", "answer_text": "woodworking"},
            user_text="woodworking and hiking on weekends",
        )
        self.assertEqual(via_plan.allow_question, via_regex.allow_question)
        self.assertTrue(via_plan.allow_question)

    def test_interest_followup_equivalent_and_allowed(self):
        via_plan, via_regex = self._equiv(
            self._INTEREST_DIR, purpose="interest", answered_question=None,
            user_text="I'm really into astrophotography these days",
        )
        self.assertEqual(via_plan.allow_question, via_regex.allow_question)
        self.assertTrue(via_plan.allow_question)


class FullEquivalenceTest(unittest.TestCase):
    """End-to-end: build_turn_plan → build_frame(plan) yields the SAME purpose AND
    allow_question as build_frame on the rendered directive with NO plan (regex). This
    is what proves the now-fully-populated plan is behavior-preserving on the live path
    across the migrated signals (ask_allowed / hard_no_question / urgent_identity / …)."""

    @contextlib.contextmanager
    def _mocks(self):
        from intelligence import conversation_agenda as ca, social_frame as sf
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                ca.world_state, "snapshot",
                return_value={"people": [], "environment": {}},
            ))
            stack.enter_context(mock.patch.object(
                ca.empathy, "classify_local_sensitivity", return_value=None))
            stack.enter_context(mock.patch.object(ca.empathy, "peek", return_value={}))
            stack.enter_context(mock.patch.object(
                ca.rel_memory, "get_latest_pending_question", return_value=None))
            stack.enter_context(mock.patch.object(
                sf, "_question_budget_allows", return_value=True))
            yield

    def _assert_equiv(self, user_text, *, person_id=1, answered_question=None):
        from intelligence import conversation_agenda as ca, social_frame as sf
        with self._mocks():
            plan = ca.build_turn_plan(user_text, person_id, answered_question=answered_question)
            via_plan = sf.build_frame(
                user_text, person_id, answered_question=answered_question,
                agenda_directive=plan.directive, turn_plan=plan,
            )
            via_regex = sf.build_frame(
                user_text, person_id, answered_question=answered_question,
                agenda_directive=plan.directive,
            )
        self.assertEqual(via_plan.purpose, via_regex.purpose, f"purpose: {user_text!r}")
        self.assertEqual(
            via_plan.allow_question, via_regex.allow_question,
            f"allow_question: {user_text!r}",
        )
        # every signal the agenda populated should be non-None (live path is plan-driven)
        for name in ("ask_allowed", "hard_no_question", "explicit_followup",
                     "fresh_interest_followup", "urgent_identity"):
            self.assertIsNotNone(getattr(plan, name), f"{name} unset for {user_text!r}")

    def test_generic_statement_equivalent(self):
        self._assert_equiv("I had a really long day at work today")

    def test_plan_statement_equivalent(self):
        self._assert_equiv("I'm going to the beach tomorrow")

    def test_user_question_equivalent(self):
        self._assert_equiv("what's the best telescope for a beginner?")

    def test_reassurance_equivalent(self):
        self._assert_equiv("I'm not upset, it's okay, no worries")


if __name__ == "__main__":
    unittest.main()
