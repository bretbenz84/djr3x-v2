"""Semantic-recall breaker observability, and one missing motion vocabulary token."""

import unittest
from unittest import mock

import numpy as np

from intelligence import action_router
from memory import semantic


class SemanticBreakerTests(unittest.TestCase):
    """The 2026-08-20 run logged ONE warning at 20:09:08 and then 24 minutes of
    silence, so it cannot answer whether semantic recall was live or degraded
    across its 78 conversational turns. _warned latched for the life of the
    process: no recovery line, and no re-trip line either."""

    def setUp(self):
        semantic._fail_count = 0
        semantic._disabled_until = 0.0
        semantic._warned = False
        semantic._last_error = ""
        semantic._cand_cache.clear()
        semantic._topic_cache = ("", None)
        self.addCleanup(semantic._cand_cache.clear)

    def _trip(self):
        for _ in range(semantic._FAIL_THRESHOLD):
            semantic._note_failure()

    def test_recovery_is_logged_and_rearms_the_warning(self):
        with mock.patch.object(semantic._log, "warning") as warn:
            self._trip()
            self.assertEqual(warn.call_count, 1)
        with mock.patch.object(semantic._log, "info") as info:
            semantic._note_success()
            info.assert_called_once()
            self.assertIn("recovered", info.call_args[0][0])
        # A LATER outage must warn again — that is the edge the old latch ate.
        with mock.patch.object(semantic._log, "warning") as warn:
            self._trip()
            self.assertEqual(warn.call_count, 1,
                             "a re-trip after recovery was silent")

    def test_success_without_a_trip_is_quiet(self):
        with mock.patch.object(semantic._log, "info") as info:
            semantic._note_success()
            info.assert_not_called()

    def test_warning_reports_the_real_error(self):
        semantic._last_error = "ConnectionError: [Errno 61] Connection refused"
        with mock.patch.object(semantic._log, "warning") as warn:
            self._trip()
        rendered = warn.call_args[0][0] % warn.call_args[0][1:]
        self.assertIn("Connection refused", rendered,
                      "the warning must name the cause, not guess at it")


class NegativeCachingTests(unittest.TestCase):
    """A None memoized during an outage was never retried — _cand_cache clears only
    on overflow (1024 entries), and the topic key holds until the conversation's
    topic tokens change. One transient blip permanently demoted those candidates to
    keyword matching for the rest of the run."""

    def setUp(self):
        semantic._cand_cache.clear()
        semantic._topic_cache = ("", None)
        self.addCleanup(semantic._cand_cache.clear)

    def test_candidate_failure_is_not_memoized(self):
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(semantic, "_embed", side_effect=[None, vec]) as embed:
            self.assertIsNone(semantic._embed_candidate("Max is a dog"))
            self.assertIsNotNone(semantic._embed_candidate("Max is a dog"))
            self.assertEqual(embed.call_count, 2, "the failure was cached")

    def test_topic_failure_is_not_memoized(self):
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(semantic, "_embed", side_effect=[None, vec]) as embed:
            self.assertIsNone(semantic._topic_vector(["georgia", "trip"]))
            self.assertIsNotNone(semantic._topic_vector(["georgia", "trip"]))
            self.assertEqual(embed.call_count, 2, "the failure was cached")

    def test_successes_are_still_memoized(self):
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(semantic, "_embed", return_value=vec) as embed:
            semantic._embed_candidate("Max is a dog")
            semantic._embed_candidate("Max is a dog")
            self.assertEqual(embed.call_count, 1, "lost the cache entirely")

    def test_relevance_still_degrades_to_keyword(self):
        """The whole point of the breaker: never worse than keyword matching."""
        with mock.patch.object(semantic, "_embed", return_value=None):
            score = semantic.relevance(["georgia", "trip"], "the georgia trip", 3)
        self.assertGreater(score, 0.0)


class MotionVocabularyTests(unittest.TestCase):
    """20:30:17 — "Turn slight left, then go forward three feet." produced zero
    motion and "I couldn't safely parse that whole route."

    _MOTION_AMOUNT whitelisted "slightly" but not the bare adverbial "slight",
    which is how turn-by-turn navigation phrases it. One unclassifiable clause
    correctly refuses the WHOLE sequence — partial execution of "turn left then
    sing" is exactly what that guard exists to prevent — so the missing token, not
    the guard, was the defect.
    """

    def _actions(self, utterance):
        seq = action_router.classify_explicit_motion_sequence(utterance)
        return [s.action for s in seq] if seq else None

    def test_the_field_route_parses(self):
        self.assertEqual(
            self._actions("Turn slight left, then go forward three feet."),
            ["motion.turn", "motion.move"],
        )

    def test_it_matches_the_slightly_form_it_was_missing(self):
        self.assertEqual(
            self._actions("Turn slight left, then go forward three feet."),
            self._actions("Turn slightly left, then go forward three feet."),
        )

    def test_single_clause_forms(self):
        for utterance in ("turn slight right", "turn slight left",
                          "turn slightly right"):
            with self.subTest(utterance=utterance):
                decision = action_router.classify_explicit_motion(utterance)
                self.assertIsNotNone(decision, "a plain steering command missed")
                self.assertEqual(decision.action, "motion.turn")

    def test_idiom_guards_still_hold(self):
        """The turn-verb filler whitelist exists because English idioms spun the
        base at conversation (field 2026-08-13)."""
        for utterance in ("She had to turn her whole life around.",
                          "Turn that frown around.",
                          "I can't face them right now."):
            with self.subTest(utterance=utterance):
                self.assertIsNone(action_router.classify_explicit_motion(utterance))


if __name__ == "__main__":
    unittest.main()
