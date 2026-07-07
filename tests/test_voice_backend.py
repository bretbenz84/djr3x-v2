"""
ECAPA-TDNN voice embedder migration (owner spec 2026-07-06): backend dispatch with
Resemblyzer fallback, the score-offset mapping onto the Resemblyzer-calibrated
threshold scale, and dimension-aware matching so 192-d ECAPA and 256-d Resemblyzer
prints coexist but never cross-match.
"""

import unittest
from unittest import mock

import numpy as np

import config
from audio import speaker_id, voice_score
from memory import people


class MapSimilarityTest(unittest.TestCase):
    def setUp(self):
        self._backend = voice_score._active_backend

    def tearDown(self):
        voice_score._active_backend = self._backend

    def test_ecapa_scores_are_offset(self):
        voice_score.set_active_backend("ecapa")
        with mock.patch.object(config, "VOICE_SCORE_OFFSET_ECAPA", 0.25, create=True):
            self.assertAlmostEqual(voice_score.map_similarity(0.30), 0.55)
            self.assertAlmostEqual(voice_score.map_similarity(0.00), 0.25)

    def test_ecapa_clamps_high(self):
        voice_score.set_active_backend("ecapa")
        self.assertAlmostEqual(voice_score.map_similarity(0.90), 0.99)

    def test_resemblyzer_is_passthrough(self):
        voice_score.set_active_backend("resemblyzer")
        self.assertAlmostEqual(voice_score.map_similarity(0.55), 0.55)

    def test_offset_preserves_margins(self):
        # The reason a constant offset is safe: score GAPS are unchanged, so every
        # margin threshold keeps its meaning.
        voice_score.set_active_backend("ecapa")
        a, b = voice_score.map_similarity(0.45), voice_score.map_similarity(0.38)
        self.assertAlmostEqual(a - b, 0.07, places=6)


class BackendDispatchTest(unittest.TestCase):
    def setUp(self):
        self._enc = speaker_id._encoder
        self._unavail = speaker_id._UNAVAILABLE
        self._backend = speaker_id._active_backend
        self._vs_backend = voice_score._active_backend
        speaker_id._encoder = None
        speaker_id._UNAVAILABLE = False
        speaker_id._active_backend = None

    def tearDown(self):
        speaker_id._encoder = self._enc
        speaker_id._UNAVAILABLE = self._unavail
        speaker_id._active_backend = self._backend
        voice_score._active_backend = self._vs_backend

    def test_ecapa_failure_falls_back_to_resemblyzer(self):
        sentinel = object()
        with (
            mock.patch.object(config, "VOICE_EMBEDDER", "ecapa", create=True),
            mock.patch.object(speaker_id, "_load_ecapa", return_value=None),
            mock.patch.object(speaker_id, "_load_resemblyzer", return_value=sentinel),
        ):
            enc = speaker_id._get_encoder()
        self.assertIs(enc, sentinel)
        self.assertEqual(speaker_id.active_backend(), "resemblyzer")
        self.assertEqual(voice_score.active_backend(), "resemblyzer")

    def test_ecapa_load_sets_backend(self):
        sentinel = object()
        with (
            mock.patch.object(config, "VOICE_EMBEDDER", "ecapa", create=True),
            mock.patch.object(speaker_id, "_load_ecapa", return_value=sentinel),
        ):
            enc = speaker_id._get_encoder()
        self.assertIs(enc, sentinel)
        self.assertEqual(speaker_id.active_backend(), "ecapa")
        self.assertEqual(voice_score.active_backend(), "ecapa")

    def test_both_unavailable_disables_cleanly(self):
        with (
            mock.patch.object(config, "VOICE_EMBEDDER", "ecapa", create=True),
            mock.patch.object(speaker_id, "_load_ecapa", return_value=None),
            mock.patch.object(speaker_id, "_load_resemblyzer", return_value=None),
        ):
            self.assertIsNone(speaker_id._get_encoder())
        self.assertTrue(speaker_id._UNAVAILABLE)


class DimensionCoexistenceTest(unittest.TestCase):
    """192-d ECAPA and 256-d Resemblyzer prints in the same biometrics table."""

    def _unit(self, dim, seed=0):
        v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_rank_speakers_skips_other_dim_rows(self):
        query = self._unit(192, 1)
        rows = [
            {"person_id": 1, "encoding": query.tobytes()},            # ECAPA print
            {"person_id": 2, "encoding": self._unit(256, 2).tobytes()},  # stale legacy
        ]
        with (
            mock.patch.object(speaker_id, "get_embedding", return_value=query),
            mock.patch.object(speaker_id.db, "fetchall", return_value=rows),
            mock.patch.object(speaker_id.people, "get_person",
                              side_effect=lambda pid: {"name": f"P{pid}"}),
        ):
            ranked = speaker_id.rank_speakers(np.zeros(4, dtype=np.float32))
        self.assertEqual([r[0] for r in ranked], [1])   # legacy row silently skipped

    def test_find_by_voice_skips_other_dim_rows(self):
        query = self._unit(192, 1)
        rows = [{"person_id": 5, "encoding": self._unit(256, 3).tobytes()}]
        with (
            mock.patch.object(people.db, "fetchall", return_value=rows),
            mock.patch.object(people, "get_person",
                              side_effect=lambda pid: {"id": pid}),
        ):
            self.assertIsNone(people.find_by_voice(query))

    def test_rank_speakers_scores_are_mapped(self):
        # A perfect self-match under ECAPA maps to the clamp (0.99), not raw 1.0.
        query = self._unit(192, 1)
        rows = [{"person_id": 1, "encoding": query.tobytes()}]
        try:
            voice_score.set_active_backend("ecapa")
            with (
                mock.patch.object(speaker_id, "get_embedding", return_value=query),
                mock.patch.object(speaker_id.db, "fetchall", return_value=rows),
                mock.patch.object(speaker_id.people, "get_person",
                                  side_effect=lambda pid: {"name": f"P{pid}"}),
            ):
                ranked = speaker_id.rank_speakers(np.zeros(4, dtype=np.float32))
            self.assertAlmostEqual(ranked[0][2], 0.99, places=3)
        finally:
            voice_score.set_active_backend(
                str(getattr(config, "VOICE_EMBEDDER", "ecapa")))


if __name__ == "__main__":
    unittest.main()
