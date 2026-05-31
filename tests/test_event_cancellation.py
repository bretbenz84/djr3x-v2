"""Regression tests for memory.events.looks_like_cancellation.

The follow-up handler and _cancel_stale_event_memory both gate a DURABLE
"mark event canceled" write on this function, before the dialogue gate runs.
It must not fire on conversational outcome replies that merely contain a loose
negation ("not going to lie...", "I'm not doing too bad").
"""

import unittest

from memory import events


class LooksLikeCancellationTest(unittest.TestCase):
    def test_conversational_outcome_replies_are_not_cancellations(self):
        # These answer Rex's "how did it go?" and must NOT cancel the event.
        for text in [
            "Not going to lie, the trip was amazing",
            "Not going to lie it was incredible",
            "Honestly I'm not doing too bad",
            "We're not going to forget that concert",
            "I'm not going to miss that show, it was perfect",
            "I'm on my way there right now",
        ]:
            with self.subTest(text=text):
                self.assertFalse(events.looks_like_cancellation(text))

    def test_real_cancellations_still_detected(self):
        for text in [
            "Oh, it got canceled",
            "I'm not going anymore",
            "changed my mind about it",
            "I can't make it",
            "they postponed it",
            "not going to make it after all",
            "we scrubbed the whole thing",
        ]:
            with self.subTest(text=text):
                self.assertTrue(events.looks_like_cancellation(text))

    def test_empty_and_none(self):
        self.assertFalse(events.looks_like_cancellation(""))
        self.assertFalse(events.looks_like_cancellation(None))


if __name__ == "__main__":
    unittest.main()
