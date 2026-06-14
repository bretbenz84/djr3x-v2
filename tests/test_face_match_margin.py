"""
Tier-1 face matcher (fixes symptoms #1 name-on-wrong-face and #2 no-re-recognition):
find_by_face now aggregates a person's multiple encodings to their CLOSEST one and
requires the winner to beat the next-closest DIFFERENT person by a margin, so the
identity can't flip between two confusable (e.g. family) faces.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import config
from memory import people


def _row(person_id, vec):
    return {"person_id": person_id, "encoding": np.asarray(vec, dtype=np.float32).tobytes()}


def _vec(*coords, dim=128):
    v = np.zeros(dim, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    return v


class FindByFaceTest(unittest.TestCase):
    def setUp(self):
        self._orig_thr = config.FACE_RECOGNITION_DISTANCE_THRESHOLD
        self._orig_margin = config.FACE_RECOGNITION_MARGIN
        config.FACE_RECOGNITION_DISTANCE_THRESHOLD = 0.6
        config.FACE_RECOGNITION_MARGIN = 0.06

    def tearDown(self):
        config.FACE_RECOGNITION_DISTANCE_THRESHOLD = self._orig_thr
        config.FACE_RECOGNITION_MARGIN = self._orig_margin

    def _find(self, rows, query):
        with (
            mock.patch.object(people.db, "fetchall", return_value=rows),
            mock.patch.object(people, "get_person", side_effect=lambda pid: {"id": pid, "name": f"P{pid}"}),
        ):
            return people.find_by_face(np.asarray(query, dtype=np.float32))

    # Query is the zero vector, so each stored encoding's distance == its coord value.
    def test_clear_winner_with_margin_is_accepted(self):
        # Bret(1) d=0.30, Wade(4) d=0.50 -> margin 0.20 -> accept Bret.
        rows = [_row(1, _vec(0.30)), _row(4, _vec(0.50))]
        out = self._find(rows, _vec(0.0))
        self.assertIsNotNone(out)
        self.assertEqual(out["id"], 1)

    def test_ambiguous_within_margin_returns_none(self):
        # Bret(1) d=0.30, Wade(4) d=0.34 -> margin 0.04 < 0.06 -> ambiguous, no guess.
        rows = [_row(1, _vec(0.30)), _row(4, _vec(0.34))]
        self.assertIsNone(self._find(rows, _vec(0.0)))

    def test_above_threshold_returns_none(self):
        rows = [_row(1, _vec(0.70))]  # d=0.70 >= 0.6
        self.assertIsNone(self._find(rows, _vec(0.0)))

    def test_per_person_min_aggregation_uses_closest_encoding(self):
        # Bret(1) has TWO encodings: far (0.55) and close (0.10). Wade(4) d=0.40.
        # Aggregating Bret to his CLOSEST (0.10) beats Wade by margin -> accept Bret.
        rows = [_row(1, _vec(0.55)), _row(1, _vec(0.10)), _row(4, _vec(0.40))]
        out = self._find(rows, _vec(0.0))
        self.assertIsNotNone(out)
        self.assertEqual(out["id"], 1)

    def test_two_encodings_of_same_person_do_not_count_as_ambiguous(self):
        # A person's own two close encodings must NOT trip the margin gate (same id,
        # so there is no second DIFFERENT person).
        rows = [_row(1, _vec(0.10)), _row(1, _vec(0.12))]
        out = self._find(rows, _vec(0.0))
        self.assertIsNotNone(out)
        self.assertEqual(out["id"], 1)


if __name__ == "__main__":
    unittest.main()
