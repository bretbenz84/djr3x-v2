"""Semantic-recall breaker observability, and one missing motion vocabulary token."""

import time
import unittest
from unittest import mock

import numpy as np

from intelligence import action_router
from memory import semantic


class SemanticBreakerTests(unittest.TestCase):
    """Reply-path contract (rewritten 2026-09-01). The robot logs from 08-21 through
    09-01 showed the old breaker — three 2.0s inline timeouts to trip, an inline
    retry every 60s — stalling llm_first_sentence by ~6s on every turn that came a
    minute after the last trip. Now ONE inline failure opens it, inline calls never
    block while it is open, and recovery is a background probe that must be FAST."""

    def setUp(self):
        semantic.reset_cache()
        self.addCleanup(semantic.reset_cache)
        # Keep un-asserted breaker warnings out of the test output; patched
        # loggers (mock.patch.object on .warning/.info) are unaffected by level.
        import logging
        level = semantic._log.level
        semantic._log.setLevel(logging.CRITICAL)
        self.addCleanup(semantic._log.setLevel, level)

    def _trip(self):
        for _ in range(semantic._FAIL_THRESHOLD):
            semantic._note_failure()

    def test_one_inline_failure_opens_the_breaker(self):
        self.assertEqual(semantic._FAIL_THRESHOLD, 1)
        with mock.patch.object(semantic._log, "warning") as warn:
            semantic._note_failure()
            self.assertEqual(warn.call_count, 1)
        self.assertTrue(semantic.is_open())
        self.assertFalse(semantic._healthy())

    def test_inline_embed_never_touches_the_endpoint_while_open(self):
        self._trip()
        with mock.patch.object(semantic, "_request_embedding",
                               side_effect=AssertionError("inline retry")) as req:
            self.assertIsNone(semantic._embed("Max is a dog"))
            req.assert_not_called()

    def test_inline_timeout_opens_after_a_single_miss(self):
        with mock.patch.object(semantic, "_request_embedding",
                               side_effect=TimeoutError("read timed out")):
            self.assertIsNone(semantic._embed("Max is a dog"))
        self.assertTrue(semantic.is_open())
        self.assertIn("read timed out", semantic._last_error)

    def test_expired_cooldown_launches_a_probe_but_stays_off_inline(self):
        self._trip()
        semantic._disabled_until = 0.0     # cooldown over
        with mock.patch.object(semantic, "_launch_recovery_probe",
                               return_value=True) as launch:
            self.assertFalse(semantic._healthy(),
                             "an expired cooldown must not re-enable inline calls")
            launch.assert_called_once()

    def test_unexpired_cooldown_launches_nothing(self):
        self._trip()
        with mock.patch.object(semantic, "_launch_recovery_probe") as launch:
            self.assertFalse(semantic._healthy())
            launch.assert_not_called()

    def test_fast_probe_closes_the_breaker_and_logs_recovery(self):
        self._trip()
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(semantic, "_request_embedding", return_value=vec), \
             mock.patch.object(semantic._log, "info") as info:
            self.assertTrue(semantic._recovery_probe())
        self.assertFalse(semantic.is_open())
        self.assertTrue(semantic._healthy())
        self.assertTrue(any("recovered" in c[0][0] for c in info.call_args_list))
        # A LATER outage must warn again — that is the edge the old latch ate.
        with mock.patch.object(semantic._log, "warning") as warn:
            self._trip()
            self.assertEqual(warn.call_count, 1, "a re-trip after recovery was silent")

    def test_failed_probe_reopens_with_a_longer_cooldown(self):
        self._trip()
        first_cooldown = semantic._cooldown_secs
        with mock.patch.object(semantic, "_request_embedding",
                               side_effect=ConnectionError("refused")):
            self.assertFalse(semantic._recovery_probe())
        self.assertTrue(semantic.is_open())
        self.assertGreater(semantic._cooldown_secs, first_cooldown,
                           "consecutive failures must back off")
        self.assertLessEqual(semantic._cooldown_secs, semantic._max_cooldown())

    def test_slow_probe_keeps_the_breaker_open(self):
        self._trip()
        vec = np.ones(4, dtype=np.float32)

        def _slow(*_a, **_k):
            time.sleep(0.02)
            return vec

        with mock.patch.object(semantic, "_request_embedding", side_effect=_slow), \
             mock.patch.object(semantic, "_probe_budget", return_value=0.001):
            self.assertFalse(semantic._recovery_probe())
        self.assertTrue(semantic.is_open())
        self.assertIn("probe took", semantic._last_error)

    def test_only_one_probe_in_flight(self):
        self._trip()
        with mock.patch.object(semantic.threading, "Thread") as thread:
            thread.return_value.start = mock.Mock()
            self.assertTrue(semantic._launch_recovery_probe())
            self.assertFalse(semantic._launch_recovery_probe())
            self.assertEqual(thread.call_count, 1)

    def test_warmup_failure_opens_the_breaker_before_the_first_turn(self):
        with mock.patch.object(semantic, "_cfg",
                               side_effect=lambda n, d: True if n == "MEMORY_SEMANTIC_RECALL_ENABLED" else d), \
             mock.patch.object(semantic, "_request_embedding",
                               side_effect=ConnectionError("refused")):
            self.assertFalse(semantic.warmup())
        self.assertTrue(semantic.is_open())

    def test_warmup_success_leaves_it_closed(self):
        vec = np.ones(4, dtype=np.float32)
        with mock.patch.object(semantic, "_cfg",
                               side_effect=lambda n, d: True if n == "MEMORY_SEMANTIC_RECALL_ENABLED" else d), \
             mock.patch.object(semantic, "_request_embedding", return_value=vec):
            self.assertTrue(semantic.warmup())
        self.assertFalse(semantic.is_open())

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

    def test_keep_alive_rides_on_every_request(self):
        seen = {}

        class _Resp:
            content = b"x"
            def raise_for_status(self): pass
            def json(self): return {"embedding": [1.0, 0.0]}

        def _post(url, json=None, timeout=None):
            seen.update(json or {})
            seen["timeout"] = timeout
            return _Resp()

        with mock.patch.dict("sys.modules", {"requests": mock.Mock(post=_post)}):
            semantic._request_embedding("hi")
        self.assertIn("keep_alive", seen)
        self.assertLessEqual(float(seen["timeout"]), semantic._inline_timeout())


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
