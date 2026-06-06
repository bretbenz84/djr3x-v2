"""
Birthday recognition: on a known person's birthday, Rex recognizes them and opens
with a birthday line — the highest-priority first-sight greeting.

The birthday is stored as a PERMANENT, immutable per-person FACT (category/key
'birthday', value 'MM-DD'), learned by the LLM fact-extractor from "it's my birthday
today" / "my birthday is X". `_pick_birthday_window` reads it and returns days_until
(0 = today) when within BIRTHDAY_REMINDER_WINDOW_DAYS; that's Priority 1 in the
first-sight greeting chain (consciousness._step_presence_tracking), so the birthday
line beats every other opener. Because the stored value is MM-DD (year-agnostic), it
recurs EVERY year.

These tests lock that chain — previously the feature was only ever MOCKED away in the
suite (to isolate lower greeting tiers), so it had no direct coverage and a refactor
could silently break "Happy birthday, Bret!" without a single red test.
"""

import unittest
from datetime import date, timedelta
from unittest import mock


class DaysUntilBirthdayTest(unittest.TestCase):
    def test_zero_on_the_day_every_year(self):
        from awareness.holidays import days_until_birthday
        self.assertEqual(days_until_birthday("06-05", today=date(2026, 6, 5)), 0)
        self.assertEqual(days_until_birthday("06-05", today=date(2027, 6, 5)), 0)  # next year too
        self.assertEqual(days_until_birthday("06-05", today=date(2030, 6, 5)), 0)

    def test_counts_down_then_wraps_after_it_passes(self):
        from awareness.holidays import days_until_birthday
        self.assertEqual(days_until_birthday("06-05", today=date(2026, 6, 1)), 4)
        self.assertEqual(days_until_birthday("06-05", today=date(2026, 6, 4)), 1)
        # The day after, the next occurrence is ~a year out (not 0).
        self.assertGreater(days_until_birthday("06-05", today=date(2026, 6, 6)), 300)

    def test_bad_input_is_safe(self):
        from awareness.holidays import days_until_birthday
        self.assertIsNone(days_until_birthday(""))
        self.assertIsNone(days_until_birthday("nope"))


class BirthdayWindowTest(unittest.TestCase):
    def _facts(self, *facts):
        return mock.patch("memory.facts.get_facts", return_value=list(facts))

    def test_returns_zero_on_the_day(self):
        from intelligence import consciousness as c
        today_md = date.today().strftime("%m-%d")
        with self._facts({"key": "birthday", "value": today_md}):
            self.assertEqual(c._pick_birthday_window(1), 0)

    def test_returns_days_within_window(self):
        from intelligence import consciousness as c
        soon = (date.today() + timedelta(days=3)).strftime("%m-%d")
        with self._facts({"key": "birthday", "value": soon}):
            self.assertEqual(c._pick_birthday_window(1), 3)

    def test_none_outside_window(self):
        from intelligence import consciousness as c
        far = (date.today() + timedelta(days=30)).strftime("%m-%d")
        with self._facts({"key": "birthday", "value": far}):
            self.assertIsNone(c._pick_birthday_window(1))

    def test_none_when_no_birthday_fact(self):
        from intelligence import consciousness as c
        with self._facts({"key": "hometown", "value": "Paris"}):
            self.assertIsNone(c._pick_birthday_window(1))

    def test_none_for_non_int_person(self):
        from intelligence import consciousness as c
        self.assertIsNone(c._pick_birthday_window(None))


class BirthdayPromptTest(unittest.TestCase):
    def test_says_today_on_the_day_and_names_the_person(self):
        from intelligence import consciousness as c
        prompt = c._build_birthday_prompt("Bret", 0)
        self.assertIn("is TODAY", prompt)
        self.assertIn("Bret", prompt)
        self.assertIn("Don't sing", prompt)  # the snark guardrail

    def test_lead_up_phrasing(self):
        from intelligence import consciousness as c
        self.assertIn("tomorrow", c._build_birthday_prompt("Bret", 1))
        self.assertIn("in 5 days", c._build_birthday_prompt("Bret", 5))


if __name__ == "__main__":
    unittest.main()
