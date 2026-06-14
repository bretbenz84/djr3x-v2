"""
Tier 2 / item 4 — child-aware + low-engagement question suppression. Rex must not
interview a child (10-12yo) or keep questioning a shy/disengaged speaker giving
one-word answers (the live session interrogated a kid with "where are you from?"
then "what do you do professionally?").
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import profile_questions as pq
from intelligence import conversation_agenda as ca
from intelligence import social_frame as sf
from intelligence.turn_plan import TurnPlan


class ChildProfileGuardTest(unittest.TestCase):
    def test_person_is_minor_from_record_and_facts(self):
        with mock.patch.object(pq.people_memory, "get_person", return_value={"age_category": "child"}), \
             mock.patch.object(pq.facts_memory, "get_facts", return_value=[]):
            self.assertTrue(pq.person_is_minor(4))
        with mock.patch.object(pq.people_memory, "get_person", return_value={"age_category": "adult"}), \
             mock.patch.object(pq.facts_memory, "get_facts",
                               return_value=[{"key": "age_category", "value": "teen"}]):
            self.assertTrue(pq.person_is_minor(4))
        with mock.patch.object(pq.people_memory, "get_person", return_value={"age_category": "adult"}), \
             mock.patch.object(pq.facts_memory, "get_facts", return_value=[]):
            self.assertFalse(pq.person_is_minor(1))

    def test_next_profile_question_skips_minor(self):
        with mock.patch.object(pq.people_memory, "get_person", return_value={"age_category": "child"}), \
             mock.patch.object(pq.facts_memory, "get_facts", return_value=[]):
            self.assertIsNone(pq.next_profile_question(4))

    def test_friendship_question_blocked_for_minor(self):
        with mock.patch.object(pq, "person_is_minor", return_value=True):
            self.assertFalse(ca._friendship_question_allowed("tell me more about it please", 4))
        # an adult with substantive text is allowed past the minor gate
        with mock.patch.object(pq, "person_is_minor", return_value=False), \
             mock.patch.object(ca.empathy, "peek", return_value={}):
            self.assertTrue(ca._friendship_question_allowed(
                "I have really been getting into astrophotography and deep sky imaging lately", 1))


class LowEngagementGateTest(unittest.TestCase):
    def _build(self, energy):
        # Force allow_question True via an explicit earned follow-up, then check the
        # energy gate. Mock build_frame's helpers to a benign baseline.
        plan = mock.Mock(target="short", max_words=40, max_sentences=2, reason="x")
        tp = TurnPlan(purpose="interest", explicit_followup=True, ask_allowed=True)
        with mock.patch.object(sf.response_length, "classify", return_value=plan), \
             mock.patch.object(sf, "_safe_user_energy", return_value=energy), \
             mock.patch.object(sf, "_safe_empathy", return_value=None), \
             mock.patch.object(sf, "_unknown_visible_count", return_value=0), \
             mock.patch.object(sf, "_looks_like_user_question", return_value=False), \
             mock.patch.object(sf, "_question_budget_allows", return_value=True), \
             mock.patch.object(sf, "_visual_allowed", return_value=False), \
             mock.patch.object(sf, "_roast_level", return_value="normal"), \
             mock.patch.object(sf, "_addressee", return_value="Wade"):
            return sf.build_frame("China", 4, turn_plan=tp)

    def test_low_engagement_suppresses_question(self):
        self.assertFalse(self._build({"engagement": "low", "mode": "quiet",
                                       "question_appetite": "low"}).allow_question)

    def test_quiet_mode_suppresses_question(self):
        self.assertFalse(self._build({"engagement": "medium", "mode": "quiet",
                                      "question_appetite": "normal"}).allow_question)

    def test_engaged_speaker_keeps_question(self):
        self.assertTrue(self._build({"engagement": "engaged", "mode": "depth",
                                     "question_appetite": "open"}).allow_question)


if __name__ == "__main__":
    unittest.main()
