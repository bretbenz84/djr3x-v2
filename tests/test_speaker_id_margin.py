"""
Tier-1 speaker-ID (fixes symptom #3 — a returning user's own voice was rejected):
per-person centroid scoring + a lowered hard threshold (0.50) guarded by a margin over
the next different person, so Bret's ~0.55 self-match (margin ~0.10 over the runner-up)
is accepted while two close candidates stay ambiguous.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import config
from audio import speaker_id


class IdentifySpeakerAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self._thr = config.SPEAKER_ID_SIMILARITY_THRESHOLD
        self._margin = config.SPEAKER_ID_KNOWN_MARGIN
        config.SPEAKER_ID_SIMILARITY_THRESHOLD = 0.50
        config.SPEAKER_ID_KNOWN_MARGIN = 0.07

    def tearDown(self):
        config.SPEAKER_ID_SIMILARITY_THRESHOLD = self._thr
        config.SPEAKER_ID_KNOWN_MARGIN = self._margin

    def _identify(self, ranked):
        with mock.patch.object(speaker_id, "rank_speakers", return_value=ranked):
            return speaker_id.identify_speaker(np.zeros(4, dtype=np.float32))

    def test_real_self_match_is_accepted(self):
        # Bret 0.55 beats Cheers 0.445 by 0.105 (>= 0.07) and clears 0.50.
        pid, name, score = self._identify([(1, "Bret", 0.55), (2, "Cheers", 0.445)])
        self.assertEqual(pid, 1)
        self.assertEqual(name, "Bret")
        self.assertAlmostEqual(score, 0.55, places=3)

    def test_below_threshold_rejected(self):
        self.assertEqual(self._identify([(1, "Bret", 0.48), (2, "Cheers", 0.30)]),
                         (None, None, 0.0))

    def test_ambiguous_within_margin_rejected(self):
        # 0.55 vs 0.52 = 0.03 margin < 0.07 -> no confident match.
        self.assertEqual(self._identify([(1, "Bret", 0.55), (4, "Wade", 0.52)]),
                         (None, None, 0.0))

    def test_single_candidate_has_infinite_margin(self):
        pid, name, _ = self._identify([(1, "Bret", 0.55)])
        self.assertEqual(pid, 1)


class RankSpeakersCentroidTest(unittest.TestCase):
    def test_one_entry_per_person_via_centroid(self):
        # Bret(1) has TWO prints; Cheers(2) one. Query aligned with Bret's mean direction.
        def vec(*c):
            v = np.zeros(4, dtype=np.float32)
            for i, x in enumerate(c):
                v[i] = x
            return v
        rows = [
            {"person_id": 1, "encoding": vec(1.0, 0.0).tobytes()},
            {"person_id": 1, "encoding": vec(0.0, 1.0).tobytes()},   # Bret centroid ~ (.707,.707)
            {"person_id": 2, "encoding": vec(-1.0, 0.0).tobytes()},  # Cheers opposite
        ]
        with (
            mock.patch.object(speaker_id, "get_embedding", return_value=vec(1.0, 1.0)),
            mock.patch.object(speaker_id.db, "fetchall", return_value=rows),
            mock.patch.object(speaker_id.people, "get_person",
                              side_effect=lambda pid: {"name": f"P{pid}"}),
        ):
            ranked = speaker_id.rank_speakers(np.zeros(4, dtype=np.float32))
        # Exactly one entry per person, Bret first (aligned), Cheers last (opposite).
        ids = [r[0] for r in ranked]
        self.assertEqual(ids.count(1), 1)
        self.assertEqual(ranked[0][0], 1)
        self.assertGreater(ranked[0][2], ranked[-1][2])


if __name__ == "__main__":
    unittest.main()
