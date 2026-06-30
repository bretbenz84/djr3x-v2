"""Move-on / disengagement override (field bug 2026-06-30: Rex ground a 'bed/mattress' metaphor
for five straight turns, and when Bret said "Don't you have anything else to say?" the agenda
classified it as a normal answer and STAYED on the bed bit — "do not pivot into a new topic").

When the human signals they want off the thread, the agenda now DROPS the topic and forces a
genuine change of direction, overriding the topic-lock.
"""

import unittest
from unittest import mock

from intelligence import conversation_agenda as ca


class WantsNewDirectionTest(unittest.TestCase):
    def test_real_move_on_cues_fire(self):
        for t in ["Don't you have anything else to say?",
                  "I think you've lost the metaphor",
                  "can we talk about something else",
                  "you keep saying that",
                  "change the subject",
                  "this is boring",
                  "you're repeating yourself",
                  "enough about the bed",
                  "drop it"]:
            self.assertTrue(ca._wants_new_direction(t), f"should pivot: {t!r}")

    def test_normal_answers_do_not_fire(self):
        for t in ["the bed is winning",
                  "probably the full dad visit",
                  "life's good, cant complain",
                  "I might go see my dad for the 4th of July",
                  "what?",                       # bare 'what?' -> repair path, not a pivot
                  "yeah it's usually sitting there looking lovely"]:
            self.assertFalse(ca._wants_new_direction(t), f"should NOT pivot: {t!r}")


class PivotAgendaTest(unittest.TestCase):
    def _directive(self, text):
        with (
            mock.patch.object(ca.people_memory, "get_person",
                              return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"}),
            mock.patch.object(ca.rel_memory, "get_latest_pending_question", return_value=None),
            mock.patch.object(ca, "_next_useful_question", return_value=None),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            return ca.build_turn_directive(text, 1)

    def test_move_on_cue_forces_a_subject_change(self):
        d = self._directive("Don't you have anything else to say?")
        self.assertIn("MOVE ON", d)
        self.assertIn("CHANGE THE SUBJECT", d)
        self.assertIn("DROP the current topic", d)

    def test_normal_answer_keeps_the_thread(self):
        d = self._directive("the bed is winning")
        self.assertNotIn("MOVE ON", d)


if __name__ == "__main__":
    unittest.main()
