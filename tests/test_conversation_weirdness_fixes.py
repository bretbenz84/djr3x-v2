"""Regression tests for the conversation bugs found in the 2026-06-08 run log.

Bug B: a joke-collapse regex truncated ordinary replies ("I can't see you." -> "I can't").
Bug C: a sentence-initial discourse marker was registered as a person's name ("Also").
Bug D: chit-chat ("what are you doing today?") was answered as a date query.
Bug E: a proactive line could play after a shutdown command (queue not suppressed in SHUTDOWN).
"""

import unittest


class JokeCollapseRegexTest(unittest.TestCase):
    def setUp(self):
        from intelligence.comedy_modes import _collapse_overexplained_joke
        self.collapse = _collapse_overexplained_joke

    def test_keeps_ordinary_verbs_and_conjunctions(self):
        # Bug B: these must NOT be truncated.
        self.assertEqual(self.collapse("I can't see you."), "I can't see you.")
        self.assertEqual(self.collapse("I don't get it."), "I don't get it.")
        self.assertEqual(
            self.collapse("I can't because the hyperdrive is offline."),
            "I can't because the hyperdrive is offline.",
        )

    def test_still_collapses_genuine_explainer_tags(self):
        self.assertEqual(
            self.collapse("That's the joke, see, robots don't sleep."), "That's the joke"
        )
        self.assertEqual(self.collapse("It's funny — because droids never tire."), "It's funny")
        self.assertEqual(self.collapse("Wakey wakey, get it?"), "Wakey wakey")


class NameDiscourseMarkerTest(unittest.TestCase):
    def setUp(self):
        from memory.name_validation import normalize_person_name
        self.normalize = normalize_person_name

    def test_rejects_sentence_initial_discourse_markers(self):
        # Bug C: "Also, what are you doing today?" -> cleaned to "Also" -> must be rejected.
        for bad in ("Also, what are you doing today?", "also", "Frankly", "So", "Well", "Anyway"):
            self.assertIsNone(self.normalize(bad), f"should reject {bad!r}")

    def test_keeps_real_names(self):
        self.assertEqual(self.normalize("Bret"), "Bret")
        self.assertEqual(self.normalize("Bret Benziger"), "Bret Benziger")
        self.assertEqual(self.normalize("Han Solo"), "Han Solo")
        self.assertEqual(self.normalize("Luke"), "Luke")


class DateQueryRegexTest(unittest.TestCase):
    REAL_DATE_QUERIES = [
        "what's the date",
        "what is the date",
        "what's today's date",
        "what day is it",
        "what day is it today",
        "what day of the week is it",
        "what weekday is it",
        "tell me the date",
        "do you know the date",
        "what's the current date",
    ]
    CHITCHAT = [
        "what are you doing today",
        "what are you up to today",
        "what are you working on today",
        "what's happening today",
        "how are you today",
        "what are we doing today",
        "what should we do today",
    ]

    def _patterns(self):
        from intelligence.intent_classifier import _DATE_QUERY_RE as classifier_re
        from intelligence.action_router import _DATE_QUERY_RE as router_re
        return classifier_re, router_re

    def test_matches_real_date_queries(self):
        classifier_re, router_re = self._patterns()
        for q in self.REAL_DATE_QUERIES:
            self.assertTrue(classifier_re.search(q), f"classifier should match {q!r}")
            self.assertTrue(router_re.search(q), f"router should match {q!r}")

    def test_rejects_chitchat_with_today(self):
        # Bug D: an action verb + "today" must not route to the date handler.
        classifier_re, router_re = self._patterns()
        for q in self.CHITCHAT:
            self.assertFalse(classifier_re.search(q), f"classifier should reject {q!r}")
            self.assertFalse(router_re.search(q), f"router should reject {q!r}")


class ShutdownSuppressesSpeechTest(unittest.TestCase):
    def test_shutdown_state_suppresses_queue_output(self):
        # Bug E: the speech queue must drop output while SHUTTING DOWN, just like SLEEP,
        # so a proactive line cannot play over the power-down animation.
        import state as state_module
        from state import State
        from audio import speech_queue

        prev = state_module.get_state()
        try:
            state_module.set_state(State.SHUTDOWN)
            self.assertTrue(speech_queue._state_suppresses_output())
            state_module.set_state(State.SLEEP)
            self.assertTrue(speech_queue._state_suppresses_output())
            state_module.set_state(State.IDLE)
            self.assertFalse(speech_queue._state_suppresses_output())
        finally:
            state_module.set_state(prev)


if __name__ == "__main__":
    unittest.main()
