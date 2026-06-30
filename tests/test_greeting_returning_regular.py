"""Returning-regular greeting variety (field gripe 2026-06-30: "Hey Bret, how are you?" /
"Hey Bret, what's up?" identical every boot, no "it's you again" vibe).

The first-sight warm greeting hard-defaulted to "how are you" on every first-of-day boot and
its template banned any "it's you again" familiarity. For an established regular it now adds a
warm familiarity note, rotates the opener by visit_count, and drops ONLY the familiarity ban.
"""

import unittest

from intelligence import consciousness as c


class SimpleGreetingPromptTest(unittest.TestCase):
    def test_plain_greeting_keeps_all_bans(self):
        p = c._build_simple_greeting_prompt("Bret", "TONE.", opener="how are you")
        self.assertIn("oh it's you again", p)   # familiarity still banned for non-regulars
        self.assertIn("NO roast", p)

    def test_familiar_greeting_drops_only_the_it_is_you_again_ban(self):
        p = c._build_simple_greeting_prompt(
            "Bret", "TONE.", note="You know Bret well — look who's back.",
            opener="what's good", allow_familiarity=True)
        self.assertNotIn("oh it's you again", p)   # familiarity now allowed
        self.assertIn("look who's back", p)         # the note is carried
        self.assertIn("NO roast", p)                # every other ban intact
        self.assertIn("NO clever", p)
        self.assertIn("NO interest callbacks", p)

    def test_opener_rotation_varies_across_visits(self):
        openers = {c._GREETING_OPENERS[v % len(c._GREETING_OPENERS)] for v in range(4, 11)}
        self.assertGreater(len(openers), 1)          # not a single fixed opener


if __name__ == "__main__":
    unittest.main()
