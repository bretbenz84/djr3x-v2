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
            "not going to make it after all",
            "we scrubbed the whole thing",
        ]:
            with self.subTest(text=text):
                self.assertTrue(events.looks_like_cancellation(text))

    def test_gap_phrasings_now_detected(self):
        # #32: plan-collapsed phrasings that used to slip past and let Rex re-ask.
        for text in [
            "that's no longer happening",
            "the trip's not on anymore",
            "we scrapped the whole thing",
            "the plan fell through",
            "yeah we called it off",
        ]:
            with self.subTest(text=text):
                self.assertTrue(events.looks_like_cancellation(text))
                self.assertFalse(events.looks_like_postponement(text))

    def test_postponements_are_not_cancellations(self):
        # A reschedule is NOT a cancellation — it must not durably lose the plan.
        for text in [
            "they postponed it",
            "we rescheduled for next week",
            "moved it to Friday",
            "we're pushing it back a bit",
            "had to put it off",
        ]:
            with self.subTest(text=text):
                self.assertFalse(events.looks_like_cancellation(text))

    def test_empty_and_none(self):
        self.assertFalse(events.looks_like_cancellation(""))
        self.assertFalse(events.looks_like_cancellation(None))


class LooksLikePostponementTest(unittest.TestCase):
    def test_postponements_detected(self):
        for text in [
            "they postponed it",
            "postponed until next week",
            "we rescheduled for next Tuesday",
            "moved it to Friday",
            "we're pushing it back",
            "had to put it off",
            "we set a new date",
        ]:
            with self.subTest(text=text):
                self.assertTrue(events.looks_like_postponement(text))

    def test_cancellations_are_not_postponements(self):
        for text in [
            "I'm not going anymore",
            "canceled the whole thing",
            "changed my mind about it",
        ]:
            with self.subTest(text=text):
                self.assertFalse(events.looks_like_postponement(text))

    def test_false_positive_idioms_guarded(self):
        # "on my way" shares the cancellation false-positive guard.
        self.assertFalse(events.looks_like_postponement(
            "I'm on my way to the postponed session"))
        self.assertFalse(events.looks_like_postponement(""))
        self.assertFalse(events.looks_like_postponement(None))


if __name__ == "__main__":
    unittest.main()
