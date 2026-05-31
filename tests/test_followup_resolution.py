"""Regression test for the event-follow-up "did not happen" resolver.

When Rex asks "how did X go?" and the user says they never went / didn't go,
the follow-up must resolve (so it stops being re-injected into the agenda),
rather than being held open as a repair and re-asked every turn. This pins the
denial detector that drives that resolution.
"""

import unittest

from intelligence import interaction


class FollowupDidNotHappenTest(unittest.TestCase):
    def test_denials_resolve_followup(self):
        for text in [
            "I never went to that",
            "I'm not going to that, I didn't go to that",
            "I never went",
            "I didn't go",
            "I did not go",
            "I never made it",
            "I didn't end up going",
            "Couldn't make it",
            "I wasn't able to go",
            "that never happened",
        ]:
            with self.subTest(text=text):
                self.assertTrue(interaction._followup_event_did_not_happen(text))

    def test_real_answers_do_not_resolve(self):
        for text in [
            "It was amazing",
            "I'm doing great",
            "I went and it was fun",
            "Not going to lie, it was incredible",
            "what are you talking about?",
            "Best concert of my life",
        ]:
            with self.subTest(text=text):
                self.assertFalse(interaction._followup_event_did_not_happen(text))

    def test_empty(self):
        self.assertFalse(interaction._followup_event_did_not_happen(""))
        self.assertFalse(interaction._followup_event_did_not_happen(None))


if __name__ == "__main__":
    unittest.main()
