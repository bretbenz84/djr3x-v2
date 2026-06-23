"""
Recall/comprehension requests must NOT be treated as misheard-corrections.

From a live run (2026-06-22): "Did you not follow what I said?" and "Do you recall what
I said?" were classified kind=misheard (the bare "i said" alternative in _MISHEARD_PAT)
and answered with a canned recovery line ("Consider it logged. Onward.") instead of Rex
actually recalling what was said. detect() must hand these to normal conversation while
still catching genuine corrections.
"""

import unittest

from intelligence import repair_moves as rm


class RecallRequestNotRepairTest(unittest.TestCase):
    def setUp(self):
        rm.note_assistant_turn("What's been the best photon you've caught lately?")

    def test_recall_requests_are_not_repairs(self):
        # The exact live failures (post-Whisper) plus near variants.
        for text in [
            "Do you recall what I said for the answer to that question?",
            "I believe I answered that question. Did you not follow what I said?",
            "Did you not follow what I said?",
            "do you remember what I said?",
            "what did I say?",
            "what did I just say?",
            "did you even hear what I said?",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(rm.detect(text),
                                  f"recall/comprehension request misrouted to repair: {text!r}")

    def test_genuine_corrections_still_route_to_repair(self):
        # Real corrections must still be caught (no regression from the guard).
        self.assertEqual(rm.detect("No, I said blues, not jazz")["kind"], "misheard")
        self.assertEqual(rm.detect("That's not what I said")["kind"], "misheard")
        self.assertEqual(rm.detect("You misheard me")["kind"], "misheard")
        self.assertEqual(rm.detect("You mean Sarah?")["kind"], "wrong_person")


if __name__ == "__main__":
    unittest.main()
