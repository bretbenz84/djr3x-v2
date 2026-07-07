"""
First-greeting style variety (owner gripe 2026-07-06: "hey Bret, what's up?" fires
too often at startup — fine sometimes, stale as the default). The regular's first
greeting of the day rotates through STYLES (question, statement, time-of-day), not
just question phrasings, and statement styles drop the question-mark mandate.
"""

import unittest
from unittest import mock

from intelligence import consciousness as c


class FirstGreetingStyleTest(unittest.TestCase):
    def test_rotates_across_visit_counts(self):
        styles = {c._first_greeting_style(v)[0] for v in range(len(c._FIRST_GREETING_STYLES))}
        self.assertGreaterEqual(len(styles), 5)   # genuinely different openers

    def test_includes_statement_styles(self):
        flags = [c._first_greeting_style(v)[1] for v in range(len(c._FIRST_GREETING_STYLES))]
        self.assertIn(False, flags)   # some greetings are NOT questions
        self.assertIn(True, flags)    # and some still are

    def test_question_default_still_present(self):
        # "how are you" stays in the rotation — it's fine sometimes.
        phrases = [c._first_greeting_style(v)[0] for v in range(len(c._FIRST_GREETING_STYLES))]
        self.assertIn("how are you", phrases)

    def test_time_of_day_renders_concrete_phrase(self):
        idx = next(i for i, (p, _q) in enumerate(c._FIRST_GREETING_STYLES)
                   if p == "time_of_day")
        phrase, is_question = c._first_greeting_style(idx)
        self.assertIn(phrase, ("good morning", "good afternoon", "good evening"))
        self.assertFalse(is_question)


class SimpleGreetingPromptTest(unittest.TestCase):
    def test_question_style_mandates_question_mark(self):
        prompt = c._build_simple_greeting_prompt("Bret", "Warm.", opener="what's up",
                                                 require_question=True)
        self.assertIn("ends in a question mark", prompt)

    def test_statement_style_makes_question_optional(self):
        prompt = c._build_simple_greeting_prompt("Bret", "Warm.",
                                                 opener="good to see you",
                                                 require_question=False)
        self.assertNotIn("ends in a question mark", prompt)
        self.assertIn("question is OPTIONAL", prompt)
        self.assertIn("good to see you", prompt)

    def test_bans_survive_both_shapes(self):
        for rq in (True, False):
            prompt = c._build_simple_greeting_prompt("Bret", "Warm.", require_question=rq)
            self.assertIn("NO roast", prompt)
            self.assertIn("NO interest callbacks", prompt)


if __name__ == "__main__":
    unittest.main()
