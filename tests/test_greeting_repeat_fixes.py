"""
Regression tests for two "Rex repeats himself every boot" bugs:

1. Milestone visit greeting ("your 5th visit") fired on EVERY startup because
   visit_count sat at a milestone-minus-one value (update_visit only advances
   after a real conversation). `_pick_milestone` now suppresses a milestone once
   `last_milestone_greeted` has recorded it.

2. A transient "today is the speaker's birthday" statement was extracted as a
   durable, permanently-injected person fact, so Rex wished happy birthday for
   days afterward. `get_prompt_worthy_facts` now drops relative-day statements,
   while the structured 'birthday' MM-DD fact (owned by the birthday-window
   path) is untouched.
"""

import unittest
from unittest import mock


class MilestoneFiresOnceTest(unittest.TestCase):
    def _person(self, visit_count, last_marked):
        return {"visit_count": visit_count, "last_milestone_greeted": last_marked}

    def test_milestone_fires_when_unacknowledged(self):
        from intelligence import consciousness as c
        with mock.patch("memory.people.get_person", return_value=self._person(4, 0)):
            self.assertEqual(c._pick_milestone(1), 5)  # incoming visit 5

    def test_milestone_suppressed_once_acknowledged(self):
        from intelligence import consciousness as c
        with mock.patch("memory.people.get_person", return_value=self._person(4, 5)):
            self.assertIsNone(c._pick_milestone(1))  # already announced #5

    def test_later_milestone_still_fires_after_earlier_acked(self):
        from intelligence import consciousness as c
        # visit_count 9 → incoming 10; only #5 acknowledged so far → fires.
        with mock.patch("memory.people.get_person", return_value=self._person(9, 5)):
            self.assertEqual(c._pick_milestone(1), 10)

    def test_non_milestone_visit_is_silent(self):
        from intelligence import consciousness as c
        with mock.patch("memory.people.get_person", return_value=self._person(5, 0)):
            self.assertIsNone(c._pick_milestone(1))  # incoming 6 not a milestone


class EphemeralFactInjectionTest(unittest.TestCase):
    def test_relative_day_statement_is_ephemeral(self):
        from memory import facts
        for value in (
            "today is the speaker's birthday",
            "got the promotion yesterday",
            "flying out tomorrow",
            "had a rough night last night",
        ):
            self.assertTrue(
                facts._is_ephemeral_statement({"value": value}),
                f"expected ephemeral: {value!r}",
            )

    def test_durable_trait_is_not_ephemeral(self):
        from memory import facts
        for value in ("06-06", "works as an engineer", "loves jazz", "has a dog named Mo"):
            self.assertFalse(
                facts._is_ephemeral_statement({"value": value}),
                f"expected durable: {value!r}",
            )

    def test_ephemeral_birthday_fact_dropped_structured_kept(self):
        from memory import facts
        rows = [
            {"key": "birthday", "value": "06-06", "confidence": 0.95, "importance": 0.85},
            {"key": "birthday_event", "value": "today is the speaker's birthday",
             "confidence": 0.95, "importance": 0.75},
            {"key": "job", "value": "works as an engineer", "confidence": 0.9, "importance": 0.6},
        ]
        with mock.patch("memory.facts.get_facts", return_value=rows):
            out = facts.get_prompt_worthy_facts(1, limit=10)
        values = [f.get("value") for f in out]
        self.assertIn("06-06", values)                       # structured birthday survives
        self.assertIn("works as an engineer", values)
        self.assertNotIn("today is the speaker's birthday", values)


if __name__ == "__main__":
    unittest.main()
