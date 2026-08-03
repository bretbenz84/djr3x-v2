"""Misheard-repair echo/duplication fixes (field bug 2026-06-30).

Two defects produced "We'll get there — recalibrating. I watch a lot of Netflix specials.
We'll get there — recalibrating." after the user re-stated content:
  (A) the recovery-line dedup missed because the LLM rendered a curly apostrophe (U+2019)
      while the constant uses a straight one (U+0027) → the preamble was appended twice;
  (B) a bare "I said X" was treated as a mishear-correction and echoed instead of answered.
"""

import unittest

from intelligence import repair_moves as r


class RecoveryLineDedupTest(unittest.TestCase):
    # The 2026-06-30 field string ("We'll get there — recalibrating.") left the pool in
    # the 2026-08-02 deflection purge; the apostrophe-folding dedup is exercised against
    # a line still in rotation.
    def test_curly_apostrophe_recovery_line_is_recognized(self):
        self.assertTrue(r._contains_recovery_line("That one’s on my wiring, not you."))

    def test_straight_apostrophe_still_recognized(self):
        self.assertTrue(r._contains_recovery_line("That one's on my wiring, not you."))

    def test_unrelated_text_not_flagged(self):
        self.assertFalse(r._contains_recovery_line("Tell me about the festival."))

    def test_deflection_lines_removed_from_pool(self):
        """The topic-closing deflectors must never come back: they sound like
        acknowledgment while refusing the repair (field 2026-07-31: 'What do you
        mean?' → 'Consider it logged. Onward.')."""
        pool = " | ".join(r._RECOVERY_LINES).lower()
        for banned in ("recalibrating", "route around", "consider it logged"):
            self.assertNotIn(banned, pool)


class ConfusionComplaintTest(unittest.TestCase):
    """'You're not saying anything that makes sense' must trigger the humble factual
    repair, never a roast comeback (field log 2026-07-03: it got 'your conversation
    pacing is the one doing barrel rolls')."""

    def test_second_person_confusion_routes_to_factual_repair(self):
        r.note_assistant_turn("Some Rex line just spoken.")
        for text in (
            "What is going on? You're not saying anything that makes sense",
            "you're not making sense",
            "nothing you're saying makes sense",
        ):
            detected = r.detect(text)
            self.assertIsNotNone(detected, text)
            self.assertEqual(detected.get("kind"), "factual", text)


class SubjectChangeTest(unittest.TestCase):
    """'Can we talk about something else?' is a TRANSIENT steer — tagged subject_change
    with the topic resolved from the live thread, never a durable boundary built from
    the request's own words (field log: banned topic 'can / talk')."""

    def test_subject_change_tagged_with_fallback_topic(self):
        from memory import boundaries
        d = boundaries.detect_boundary(
            "Can we talk about something else?", fallback_topic="car repairs"
        )
        self.assertIsNotNone(d)
        self.assertEqual(d.get("kind"), "subject_change")
        self.assertEqual(d.get("topic"), "car repairs")

    def test_real_boundary_still_tagged_boundary(self):
        from memory import boundaries
        d = boundaries.detect_boundary("don't ask about work", fallback_topic="x")
        self.assertIsNotNone(d)
        self.assertEqual(d.get("kind"), "boundary")
        self.assertEqual(d.get("topic"), "work")


class BareRestatementTest(unittest.TestCase):
    def test_plain_restatements_reroute(self):
        for t in ["I said I watch a lot of Netflix specials",
                  "I meant the blue one",
                  "um, I said pizza",
                  "I said yes"]:
            self.assertTrue(r.is_bare_restatement(t), f"should re-route: {t!r}")

    def test_contrastive_corrections_stay_in_repair(self):
        for t in ["I said blues, not jazz",
                  "I said blue, not red",
                  "That's not what I said",
                  "no, I said Tuesday"]:
            self.assertFalse(r.is_bare_restatement(t), f"should NOT re-route: {t!r}")

    def test_empty_is_false(self):
        self.assertFalse(r.is_bare_restatement(""))
        self.assertFalse(r.is_bare_restatement(None))


class QueryCorrectionRerouteTest(unittest.TestCase):
    """A correction that restates a routable QUERY ('no I said what's the weather') must re-run the
    query, not fire a repair-ack (field bug: user had to ask a 3rd time). Distinct from a bare
    restatement — 'no …' is not bare, but a query correction still re-routes."""

    def setUp(self):
        from unittest import mock
        import intelligence.interaction as I
        self.I = I
        self._p = mock.patch.object(I.config, "INTENT_CLASSIFIER_ENABLED", True)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_query_corrections_route(self):
        for c in ["what's the weather", "what time is it", "what can you do"]:
            self.assertTrue(self.I._correction_routes_to_query(c), f"query should route: {c!r}")

    def test_statement_corrections_do_not_route(self):
        for c in ["I went to my dad's for the 4th", "I like jazz", "my name is Bob", "", None]:
            self.assertFalse(self.I._correction_routes_to_query(c), f"statement must NOT route: {c!r}")

    def test_field_case_reroutes_but_statement_correction_does_not(self):
        # The exact field utterance re-routes (kind=misheard, correction is a query), while a
        # restated non-query statement with the same "no …" shape keeps the repair-ack path.
        weather = r.detect("no I said what's the weather")
        self.assertEqual(weather.get("kind"), "misheard")
        self.assertTrue(self.I._correction_routes_to_query(weather.get("correction")))
        self.assertFalse(r.is_bare_restatement("no I said what's the weather"))  # not bare, but routes

        dad = r.detect("no I said I went to my dad's")
        self.assertFalse(self.I._correction_routes_to_query(dad.get("correction")))


if __name__ == "__main__":
    unittest.main()
