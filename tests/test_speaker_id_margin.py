"""
Tier-1 speaker-ID (fixes symptom #3 — a returning user's own voice was rejected):
per-person centroid scoring + a lowered hard threshold (0.50) guarded by a margin over
the next different person, so Bret's ~0.55 self-match (margin ~0.10 over the runner-up)
is accepted while two close candidates stay ambiguous.

Thin-challenger relief (field log 2026-07-06-19-23): a runner-up whose centroid is a
SINGLE unverified clip halves the required margin when the top candidate is mature and
scores above the cross-match band — a 1-print centroid must not challenge the owner's
6-print match at a 0.056 gap. The reverse direction (thin print on top) keeps the full
margin so the who's-that ask still fires for the newcomer.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import config
from audio import speaker_id, voice_score


class IdentifySpeakerAcceptanceTest(unittest.TestCase):
    def setUp(self):
        backend = mock.patch.object(voice_score, "_active_backend", "ecapa")
        backend.start()
        self.addCleanup(backend.stop)
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
        pid, name, score = self._identify([(1, "Bret", 0.55, 3), (2, "Cheers", 0.445, 3)])
        self.assertEqual(pid, 1)
        self.assertEqual(name, "Bret")
        self.assertAlmostEqual(score, 0.55, places=3)

    def test_below_threshold_rejected(self):
        self.assertEqual(self._identify([(1, "Bret", 0.48, 3), (2, "Cheers", 0.30, 3)]),
                         (None, None, 0.0))

    def test_ambiguous_within_margin_rejected(self):
        # 0.55 vs 0.52 = 0.03 margin < 0.07 -> no confident match (both mature).
        self.assertEqual(self._identify([(1, "Bret", 0.55, 3), (4, "Wade", 0.52, 3)]),
                         (None, None, 0.0))

    def test_single_candidate_has_infinite_margin(self):
        pid, name, _ = self._identify([(1, "Bret", 0.55, 3)])
        self.assertEqual(pid, 1)


class ThinChallengerReliefTest(unittest.TestCase):
    """required_ambiguity_margin — the scoreboard-specific ambiguity bar."""

    def setUp(self):
        backend = mock.patch.object(voice_score, "_active_backend", "ecapa")
        backend.start()
        self.addCleanup(backend.stop)
        self._saved = {}
        for key, val in {
            "SPEAKER_ID_SIMILARITY_THRESHOLD": 0.50,
            "SPEAKER_ID_KNOWN_MARGIN": 0.07,
            "SPEAKER_ID_THIN_PRINT_MAX_ROWS": 1,
            "SPEAKER_ID_THIN_RUNNER_MARGIN_FACTOR": 0.5,
            "SPEAKER_ID_THIN_RUNNER_MIN_TOP_SCORE": 0.55,
        }.items():
            self._saved[key] = getattr(config, key, None)
            setattr(config, key, val)

    def tearDown(self):
        for key, val in self._saved.items():
            setattr(config, key, val)

    def test_field_case_owner_vs_thin_print_gets_relief(self):
        # Log 2026-07-06-19-23: Bret (6 prints) 0.558 vs JT (1 print) 0.502.
        ranked = [(1, "Bret", 0.558, 6), (2, "JT", 0.502, 1)]
        self.assertAlmostEqual(speaker_id.required_ambiguity_margin(ranked), 0.035)
        # And the full identify path now accepts Bret.
        with mock.patch.object(speaker_id, "rank_speakers", return_value=ranked):
            pid, name, _ = speaker_id.identify_speaker(np.zeros(4, dtype=np.float32))
        self.assertEqual((pid, name), (1, "Bret"))

    def test_field_case_thin_print_on_top_keeps_full_margin(self):
        # Log 2026-07-05-21-22: JT (1 print) 0.563 vs Bret (6 prints) 0.529 —
        # the challenge must still fire while JT's print is unverified.
        ranked = [(2, "JT", 0.563, 1), (1, "Bret", 0.529, 6)]
        self.assertAlmostEqual(speaker_id.required_ambiguity_margin(ranked), 0.07)
        with mock.patch.object(speaker_id, "rank_speakers", return_value=ranked):
            self.assertEqual(
                speaker_id.identify_speaker(np.zeros(4, dtype=np.float32)),
                (None, None, 0.0),
            )

    def test_low_top_score_gets_no_relief(self):
        # A cross-match impostor lands ~0.53 on the mature centroid (measured JT →
        # Bret 0.529): below the 0.55 bar, relief must not apply.
        ranked = [(1, "Bret", 0.53, 6), (2, "JT", 0.50, 1)]
        self.assertAlmostEqual(speaker_id.required_ambiguity_margin(ranked), 0.07)

    def test_two_mature_candidates_keep_full_margin(self):
        ranked = [(1, "Bret", 0.60, 6), (4, "Wade", 0.55, 5)]
        self.assertAlmostEqual(speaker_id.required_ambiguity_margin(ranked), 0.07)

    def test_two_thin_candidates_keep_full_margin(self):
        # Both unverified: no one has earned the benefit of the doubt.
        ranked = [(2, "JT", 0.58, 1), (3, "Sam", 0.54, 1)]
        self.assertAlmostEqual(speaker_id.required_ambiguity_margin(ranked), 0.07)

    def test_single_candidate_returns_base(self):
        self.assertAlmostEqual(
            speaker_id.required_ambiguity_margin([(1, "Bret", 0.7, 6)]), 0.07)


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
        # Print counts ride along: Bret 2, Cheers 1.
        self.assertEqual(ranked[0][3], 2)
        self.assertEqual(ranked[-1][3], 1)


if __name__ == "__main__":
    unittest.main()
