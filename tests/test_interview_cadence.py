"""Anti-interview cadence: after several consecutive question-ending Rex turns, the
next reply is forced to be a statement so Rex doesn't interrogate once a topic opens
(live-logged 2026-06-20: six question-ending turns in a row about a favourite movie)."""

import unittest
from unittest import mock

from intelligence import question_budget as qb


class QuestionBudgetCadenceTest(unittest.TestCase):
    def setUp(self):
        qb.clear()

    def test_consecutive_questions_force_statement(self):
        self.assertFalse(qb.should_force_statement_turn())
        qb.note_rex_utterance("What's your favorite movie?")
        self.assertEqual(qb.consecutive_question_turns(), 1)
        self.assertFalse(qb.should_force_statement_turn())
        qb.note_rex_utterance("What part do you love most?")
        qb.note_rex_utterance("What got you into it?")
        self.assertEqual(qb.consecutive_question_turns(), 3)
        self.assertTrue(qb.should_force_statement_turn())  # K=3 reached

    def test_statement_turn_resets_the_streak(self):
        for _ in range(4):
            qb.note_rex_utterance("oh really? and then? hm?")
        self.assertTrue(qb.should_force_statement_turn())
        qb.note_rex_utterance("Eddie Izzard is genuinely brilliant.")  # no '?'
        self.assertEqual(qb.consecutive_question_turns(), 0)
        self.assertFalse(qb.should_force_statement_turn())

    def test_clamp_can_be_disabled(self):
        import config
        old = config.INTERVIEW_CADENCE_CLAMP_ENABLED
        config.INTERVIEW_CADENCE_CLAMP_ENABLED = False
        try:
            for _ in range(6):
                qb.note_rex_utterance("really? sure? ok?")
            self.assertFalse(qb.should_force_statement_turn())
        finally:
            config.INTERVIEW_CADENCE_CLAMP_ENABLED = old

    def test_question_budget_still_tracks_window(self):
        # The consecutive counter is independent of the time-window question budget.
        qb.note_rex_utterance("a question?")
        snap = qb.snapshot()
        self.assertGreaterEqual(snap["recent_questions"], 1)


class SocialFrameCadenceClampTest(unittest.TestCase):
    """The clamp actually suppresses the trailing question in build_frame for an
    otherwise-allowed earned follow-up, but an urgent identity ask overrides it."""

    _INTEREST_DIRECTIVE = (
        "Conversation steering: The current thread matches a known/active interest: "
        "'british comedy'.\nPrimary purpose: deepen the interest thread the human "
        "opened. Give one specific reaction or tidbit, then ask one natural follow-up "
        "about their experience with that topic."
    )

    def setUp(self):
        qb.clear()

    def _frame(self, directive, text="I like Eddie Izzard", person_id=1):
        from intelligence import social_frame as sf
        with (
            mock.patch.object(sf.world_state, "snapshot", return_value={"people": []}),
            mock.patch.object(sf, "_question_budget_allows", return_value=True),
            mock.patch.object(sf, "_safe_user_energy", return_value={}),
        ):
            return sf.build_frame(text, person_id, agenda_directive=directive)

    def test_earned_followup_allows_question_until_streak(self):
        frame = self._frame(self._INTEREST_DIRECTIVE)
        self.assertTrue(frame.allow_question)  # earned follow-up: normally allowed
        for _ in range(3):
            qb.note_rex_utterance("and? really? hm?")
        clamped = self._frame(self._INTEREST_DIRECTIVE)
        self.assertFalse(clamped.allow_question)  # streak → forced statement turn


if __name__ == "__main__":
    unittest.main()
