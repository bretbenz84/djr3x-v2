"""
Tier D — semantic embedding relevance (memory/semantic.py), the pluggable backend for
the unified retrieval layer. Embeddings are MOCKED here (a tiny 3-axis fake) so the
tests need no Ollama / model download.

Properties covered:
  * cosine relevance discriminates meaning (ocean↔sailing high, ocean↔dog ~0),
  * it falls back to keyword overlap when embeddings are unavailable (never worse),
  * the circuit breaker disables a dead endpoint,
  * it stays OFF by default and auto-installs only when the flag is on,
  * end-to-end it surfaces a semantically-related interest with no shared word.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from memory import database as db, retrieval, semantic


def _fake_embed(text):
    """3-axis stand-in: ocean / pet / neutral. Unit-normalized, never zero-norm."""
    t = (text or "").lower()
    v = np.array([
        1.0 if any(w in t for w in ("ocean", "sail", "boat", "sea", "water")) else 0.0,
        1.0 if any(w in t for w in ("dog", "pet", "puppy", "cat")) else 0.0,
        0.1,
    ], dtype=np.float32)
    return v / float(np.linalg.norm(v))


class SemanticRelevanceTest(unittest.TestCase):
    def setUp(self):
        semantic.reset_cache()
        self._patch = mock.patch.object(semantic, "_embed", side_effect=_fake_embed)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        semantic.reset_cache()

    def test_meaning_match_scores_high(self):
        rel = semantic.relevance({"ocean", "trip"}, "loves sailing boats", 3)
        self.assertGreater(rel, 1.5)

    def test_unrelated_meaning_scores_zero(self):
        rel = semantic.relevance({"ocean"}, "has a dog named Scout", 3)
        self.assertEqual(rel, 0.0)


class SemanticFallbackTest(unittest.TestCase):
    def tearDown(self):
        semantic.reset_cache()

    def test_falls_back_to_keyword_when_embedding_unavailable(self):
        semantic.reset_cache()
        with mock.patch.object(semantic, "_embed", return_value=None):
            # No embeddings → keyword overlap ("dog" topic matches "dog" text) ≈ 1.
            rel = semantic.relevance({"dog"}, "has a dog named Scout", 3)
        self.assertGreaterEqual(rel, 1.0)

    def test_circuit_breaker_trips_after_failures(self):
        semantic.reset_cache()
        self.assertTrue(semantic._healthy())
        for _ in range(semantic._FAIL_THRESHOLD):
            semantic._note_failure()
        self.assertFalse(semantic._healthy())   # endpoint parked
        semantic.reset_cache()
        self.assertTrue(semantic._healthy())     # reset re-arms


class SemanticInstallTest(unittest.TestCase):
    def tearDown(self):
        retrieval.set_relevance_backend(None)
        semantic.reset_cache()

    def test_off_by_default_no_backend(self):
        retrieval.set_relevance_backend(None)   # re-arms auto-install
        with mock.patch.object(config, "MEMORY_SEMANTIC_RECALL_ENABLED", False):
            retrieval._ensure_backend()
        self.assertIsNone(retrieval._relevance_backend)

    def test_flag_on_installs_semantic_backend(self):
        retrieval.set_relevance_backend(None)
        with mock.patch.object(config, "MEMORY_SEMANTIC_RECALL_ENABLED", True):
            retrieval._ensure_backend()
        self.assertIs(retrieval._relevance_backend, semantic.relevance)


class SemanticEndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        from setup_assets import DB_SCHEMA
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret')")
        self._db_patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._db_patch.start()
        semantic.reset_cache()
        self._embed_patch = mock.patch.object(semantic, "_embed", side_effect=_fake_embed)
        self._embed_patch.start()
        retrieval.set_relevance_backend(semantic.relevance)

    def tearDown(self):
        retrieval.set_relevance_backend(None)
        semantic.reset_cache()
        self._embed_patch.stop()
        self._db_patch.stop()
        self._tmp.cleanup()

    def test_semantic_surfaces_related_interest_with_no_shared_word(self):
        from memory import interests
        interests.upsert_interest(1, "sailing", interest_strength="low")
        interests.upsert_interest(1, "gardening", interest_strength="high")
        interests.upsert_interest(1, "baking", interest_strength="high")
        # Topic "ocean" shares NO word with "sailing" — only meaning. With a tight budget,
        # semantic relevance must still pull sailing in over the higher-strength others.
        with mock.patch.object(config, "MEMORY_PROMPT_BUDGET_ITEMS", 1):
            bundle = retrieval.retrieve_person_memory(1, topic_tokens={"ocean"})
        names = {it["name"] for it in bundle["interests"]}
        self.assertEqual(names, {"sailing"})


if __name__ == "__main__":
    unittest.main()
