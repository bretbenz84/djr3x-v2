"""Unit tests for the DETERMINISTIC eval checkers (evals/checkers.py).

Only the network-free deterministic checkers are exercised here — the LLM-judge
checkers (invented_prop / roasted_sincere) make real OpenAI calls and stay out of
the suite. Importing evals.checkers is itself network-free (stdlib + a lazy
intelligence import), so this guards the checker logic without violating the
suite's no-network rule.
"""

import unittest

from evals import checkers


class OverQuestioningCheckerTest(unittest.TestCase):
    """The checker must count GENUINE (unquoted) question SENTENCES the way the
    production one-question cap does — not raw '?' chars — so a quoted or embedded
    '?' the cap ignores doesn't over-report (the ~4% residual the refinement fixes)."""

    def _n(self, reply):
        return checkers._count_question_sentences(reply)

    def _flag(self, reply, cap=1):
        return checkers.over_questioning(reply, {"max_questions": cap}).flagged

    def test_single_genuine_question_not_flagged(self):
        self.assertEqual(self._n("A robot DJ? Bold move."), 1)
        self.assertFalse(self._flag("A robot DJ? Bold move."))

    def test_two_genuine_questions_flagged(self):
        self.assertEqual(self._n("What's up? How are you?"), 2)
        self.assertTrue(self._flag("What's up? How are you?"))

    def test_quoted_question_mark_is_not_counted(self):
        # The quoted "really?" must NOT count — only the unquoted question does.
        reply = 'He said "really?" so what are you building?'
        self.assertEqual(self._n(reply), 1)
        self.assertFalse(self._flag(reply))

    def test_statement_has_zero_questions(self):
        self.assertEqual(self._n("Just a statement here."), 0)
        self.assertFalse(self._flag("Just a statement here."))

    def test_empty_reply_is_safe(self):
        self.assertEqual(self._n(""), 0)
        self.assertFalse(self._flag(""))


class TrailOffCheckerTest(unittest.TestCase):
    def test_clean_ending_not_flagged(self):
        self.assertFalse(checkers.trail_off("All systems go.", {}).flagged)
        self.assertFalse(checkers.trail_off("Ready?", {}).flagged)

    def test_mid_clause_cutoff_flagged(self):
        self.assertTrue(checkers.trail_off("I guess the excitement of", {}).flagged)

    def test_empty_is_not_flagged(self):
        self.assertFalse(checkers.trail_off("", {}).flagged)


if __name__ == "__main__":
    unittest.main()
