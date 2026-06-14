"""
Relationship-aware greetings. Rex greeted Bret (his CREATOR) with a hostile roast
("Oh, it's you again, Bret! What do you need this time?") regardless of relationship.
Greetings now default to a plain, warm, human hello ("Hey Bret, how are you?") scaled
by tier/creator, with the roast/interest-hook openers removed.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import consciousness as c
from memory import people as pm


def _profile(name, tier, *, creator=False):
    with mock.patch.object(pm, "get_person", return_value={"name": name, "friendship_tier": tier}), \
         mock.patch.object(c.person_specials, "is_rex_creator", return_value=creator):
        return c._greeting_profile(1)


class GreetingProfileTest(unittest.TestCase):
    def test_creator_is_warm_default(self):
        tone, warm = _profile("Bret Benziger", "friend", creator=True)
        self.assertTrue(warm)
        self.assertIn("MAKER", tone)

    def test_friends_get_warm_default(self):
        for tier in ("friend", "close_friend", "best_friend"):
            _, warm = _profile("Sam", tier)
            self.assertTrue(warm, tier)

    def test_acquaintance_and_stranger_are_not_warm_default(self):
        self.assertFalse(_profile("Sam", "acquaintance")[1])
        self.assertFalse(_profile("Sam", "stranger")[1])


class GreetingPromptTest(unittest.TestCase):
    def test_simple_greeting_is_plain_and_warm(self):
        p = c._build_simple_greeting_prompt("Bret", "This is a friend.")
        self.assertIn("how are you", p.lower())
        self.assertIn("NO roast", p)             # explicitly forbids a roast
        self.assertIn("tinkering with", p)       # and forbids the try-hard curiosity hook
        self.assertIn("This is a friend.", p)    # the relationship tone is woven in

    def test_same_day_return_is_no_longer_a_roast(self):
        p = c._build_same_day_return_prompt("Bret", 1, tone="This is a friend.")
        self.assertIn("how are you", p.lower())
        self.assertNotIn("punch up", p)                 # old roast instruction gone
        self.assertNotIn("powered you up for the", p)   # old "won't leave you alone" tally gone
        self.assertNotIn("what do you need this time", p)

    def test_recent_and_long_absence_route_through_simple(self):
        for p in (c._build_recent_return_prompt("Bret", 3.0, tone="t."),
                  c._build_long_absence_prompt("Bret", 30.0, tone="t.")):
            self.assertIn("how are you", p.lower())
            self.assertNotIn("teasing", p)
            self.assertNotIn("accusatory", p)


class RepeatGreetingOpenerTest(unittest.TestCase):
    """Repeat visits in the same window vary the opener instead of always 'how are you'."""

    def test_first_greeting_uses_default(self):
        self.assertIsNone(c._repeat_greeting_opener(1))  # None -> default "how are you"

    def test_repeats_rotate_and_skip_the_default(self):
        openers = [c._repeat_greeting_opener(n) for n in range(2, 8)]
        self.assertTrue(all(o for o in openers))           # all non-None
        self.assertNotIn("how are you", openers)           # never the default on a repeat
        self.assertEqual(len(set(openers)), len(openers))  # all distinct within one cycle

    def test_rotation_wraps(self):
        # Cycles back to the first variant after exhausting the pool.
        self.assertEqual(c._repeat_greeting_opener(2), c._repeat_greeting_opener(2 + len(c._GREETING_OPENERS) - 1))

    def test_same_day_return_prompt_honors_opener(self):
        p = c._build_same_day_return_prompt("Bret", 1, tone="t.", opener="what's up")
        self.assertIn("what's up", p.lower())
        self.assertNotIn("how are you", p.lower())  # the rotated opener replaces the default


if __name__ == "__main__":
    unittest.main()
