"""
Introductions: Rex must get curious about the NEW person, not interview the
introducer by proxy. Live failure (logs/conversation-2026-06-13-15-45-53.log):
"I'd like you to meet my nephew, Wade" -> Rex talked about himself / asked
rhetorical questions about Bret, never about Wade ("the program is supposed to
get to know new people").
"""

from __future__ import annotations

import unittest
from unittest import mock


class IntroVocabTest(unittest.TestCase):
    def test_nephew_niece_detected_and_self_explanatory(self):
        from intelligence import interaction, introductions
        self.assertTrue(interaction._intro_relationship_self_explanatory("nephew"))
        self.assertTrue(interaction._intro_relationship_self_explanatory("niece"))
        parse = introductions.detect("I'd like you to meet my nephew, Wade")
        self.assertTrue(parse.is_introduction)
        self.assertEqual(parse.name, "Wade")
        self.assertEqual(parse.relationship, "nephew")

    def test_inverse_relationship_for_nephew(self):
        from intelligence import interaction
        self.assertEqual(interaction._intro_inverse_relationship("nephew"), "aunt_or_uncle")


class IntroQuestionTargetsNewcomerTest(unittest.TestCase):
    def test_question_instruction_is_about_the_newcomer(self):
        from intelligence import interaction
        q = interaction._intro_relationship_question_instruction("nephew", "Bret", "Wade")
        self.assertIn("Wade", q)
        self.assertNotIn("Bret", q)              # not steered toward the introducer
        self.assertIn("get-to-know-you", q)


class IntroAckBuildsNewcomerCuriosityTest(unittest.TestCase):
    def test_ack_asks_wade_about_himself_with_his_person_id(self):
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction.llm, "get_response", return_value="Wade! Welcome."
            ) as get_response,
            mock.patch.object(
                interaction, "_told_about_teller_name", return_value=None
            ),
            mock.patch.object(
                interaction.consciousness, "note_person_greeted_this_session"
            ),
        ):
            interaction._intro_ack_and_followup(
                introducer_id=1,
                introducer_name="Bret Benziger",
                introduced_id=4,
                introduced_name="Wade",
                relationship="nephew",
                visible_newcomer=True,
            )

        self.assertTrue(get_response.called)
        prompt = get_response.call_args.args[0]
        person_id = get_response.call_args.args[1]
        # Generated AS the newcomer's turn so the cast frames Wade as the subject.
        self.assertEqual(person_id, 4)
        # Curiosity is directed at Wade about himself. (The prompt may still carry a
        # NEGATIVE guard like "do NOT ask how they know each other" — that's correct,
        # so we assert the positive instruction rather than the phrase's absence.)
        self.assertIn("Wade", prompt)
        self.assertIn("get-to-know-you", prompt)
        self.assertIn("themselves", prompt.lower())
        self.assertNotIn("origin story", prompt.lower())


if __name__ == "__main__":
    unittest.main()
