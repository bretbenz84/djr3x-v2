"""
Two issues from the 2026-06-14 16:53 run:
  1. A transient detail ("piles of boxes contributing to the heat in the room") was
     baked into the persisted last-conversation summary and kept resurfacing in
     unrelated conversations ("…those piles of boxes"). Summaries now exclude transient
     surroundings.
  2. Rex over-responded to short, friendly throwaway answers ("I've been good", "it's
     going good") by roasting the brevity or dragging in a remembered detail. A phatic-
     answer branch now steers a brief, warm, natural reply.
"""

from __future__ import annotations

import unittest

from intelligence import conversation_agenda as ca
from intelligence import llm
from intelligence import topic_thread


class PhaticAnswerDetectorTest(unittest.TestCase):
    def test_matches_short_throwaway_answers(self):
        for t in ("good", "I'm good", "I've been good", "it's going good", "doing well",
                  "pretty good", "not much", "can't complain", "all good", "good, you?"):
            self.assertTrue(ca._looks_like_phatic_answer(t), t)

    def test_rejects_answers_with_real_content(self):
        for t in ("good, just recovering from my camping trip",
                  "I captured the Eagle Nebula",
                  "good because I finished my 3D print",
                  "I'm good but my dog is sick",
                  "the photo came out so good"):
            self.assertFalse(ca._looks_like_phatic_answer(t), t)

    def test_empty_is_not_phatic(self):
        self.assertFalse(ca._looks_like_phatic_answer(""))
        self.assertFalse(ca._looks_like_phatic_answer("   "))


class PhaticTurnPlanTest(unittest.TestCase):
    def test_phatic_answer_gets_brief_warm_directive(self):
        plan = ca.build_turn_plan("I've been good", person_id=1)
        d = plan.directive.lower()
        self.assertEqual(plan.purpose, "small_talk")
        self.assertIn("throwaway answer", d)
        self.assertIn("do not analyze, roast", d)
        # It explicitly forbids the exact bad behaviors seen in the logs.
        self.assertIn("droid-approved script", d)
        self.assertIn("surroundings", d)

    def test_contentful_answer_does_not_take_phatic_path(self):
        plan = ca.build_turn_plan("good, just recovering from my camping trip", person_id=1)
        self.assertNotEqual(plan.purpose, "small_talk")


class SummaryHygieneTest(unittest.TestCase):
    def test_session_summary_prompt_excludes_transient_surroundings(self):
        # The prompt is built inline; assert the guidance is present by checking the
        # module source carries the exclusion (cheap, no LLM call).
        import inspect
        src = inspect.getsource(llm.generate_session_summary)
        self.assertIn("transient", src.lower())
        self.assertIn("boxes", src.lower())

    def test_arc_rich_schema_shared_excludes_surroundings(self):
        rich = topic_thread._build_arc_prompt("User: hi", rich=True)
        # The Shared field now says DURABLE facts, not transient surroundings.
        self.assertIn("DURABLE", rich)
        self.assertIn("boxes", rich.lower())


if __name__ == "__main__":
    unittest.main()
