"""Bounded prompt retrieval and optional inference admission, without a model."""
import threading
import time
import unittest
from unittest import mock
import numpy as np
from memory import semantic
from utils import local_work


class OptionalWorkTest(unittest.TestCase):
    def setUp(self):
        semantic.reset_cache()
        self.addCleanup(semantic.reset_cache)

    def test_foreground_prevents_new_optional_work(self):
        with local_work.foreground():
            with local_work.optional() as admitted:
                self.assertFalse(admitted)
        with local_work.optional() as admitted:
            self.assertTrue(admitted)
            with local_work.optional() as second:
                self.assertFalse(second)

    def test_candidate_misses_never_embed_during_retrieval(self):
        with (
            mock.patch.object(semantic, '_embed', side_effect=AssertionError('inline')),
            mock.patch.object(semantic, '_ensure_prewarm_worker'),
            semantic.turn_budget(1),
        ):
            self.assertIsNone(semantic._embed_candidate('sailing'))
        self.assertIn('sailing', semantic._prewarm_pending)

    def test_nested_budget_cannot_extend_outer_deadline(self):
        with semantic.turn_budget(0):
            with semantic.turn_budget(10):
                self.assertLessEqual(semantic._budget_remaining(), 0)
            self.assertLessEqual(semantic._budget_remaining(), 0)
        self.assertIsNone(semantic._budget_remaining())

    def test_budget_is_not_shared_with_other_threads(self):
        result = []
        with semantic.turn_budget(0):
            thread = threading.Thread(target=lambda: result.append(semantic._budget_remaining()))
            thread.start()
            thread.join(1)
        self.assertEqual(result, [None])

    def test_query_deadline_does_not_wait_for_slow_request(self):
        release = threading.Event()
        finished = threading.Event()
        def embed(*args):
            try:
                release.wait(1)
                return np.array([1.0, 0.0])
            finally:
                finished.set()
        with mock.patch.object(semantic, '_embed', side_effect=embed):
            try:
                started = time.monotonic()
                with semantic.turn_budget(0.02):
                    self.assertIsNone(semantic._topic_vector({'ocean'}))
                self.assertLess(time.monotonic() - started, 0.3)
            finally:
                release.set()
                self.assertTrue(finished.wait(1))
        self.assertEqual(semantic._topic_cache, ('', None))

    def test_background_queue_is_bounded(self):
        with mock.patch.object(semantic, '_ensure_prewarm_worker'):
            semantic.prewarm_texts([str(i) for i in range(1000)])
        self.assertLessEqual(len(semantic._prewarm_pending), 128)
