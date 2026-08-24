"""
Face false-accept hardening, from the 2026-08-23 18:15 PJ run
(logs/djr3x-2026-08-23-18-15-49.log).

At 18:18:08, 18:21:30 and 18:24:11 "face identified → Bret Benziger" fired while
only PJ (un-enrolled) was in front of the camera — once from 2-3 ft. PJ's live
embedding landed inside the old 1.10 ArcFace accept bar against Bret's single
unverified auto-captured reference, bound on ONE frame, and then suppressed the
camera-contradiction voice guards (5adc6bb) that trust "known person visible".

Under test:
  - find_by_face: tightened ArcFace bar (1.10 → 1.00) + match-quality metadata
    (face_match_distance / face_match_strong) attached to accepted matches.
  - _confirm_new_identity: a gray-zone match (strong < d < accept) may not
    create a NEW identity binding until it repeats for
    FACE_IDENTIFY_CONFIRM_FRAMES consecutive ticks; strong matches, existing
    bindings, and carried (sticky/hysteresis) identities bind as before.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from intelligence import consciousness as C
from memory import database as db


def _unit_pair(cos_sim: float, dim: int = 512) -> np.ndarray:
    """A unit vector at the given cosine similarity to e1 (L2-normalized)."""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = cos_sim
    v[1] = math.sqrt(max(0.0, 1.0 - cos_sim * cos_sim))
    return v


def _e(idx: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


class _TempPeopleDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA

        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
        self._patch = mock.patch.object(db, "_DB_FILE", self._path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()


class FindByFaceThresholdTest(_TempPeopleDb):
    """ArcFace bar and metadata. cos↔distance: d = sqrt(2 - 2*cos)."""

    def setUp(self):
        super().setUp()
        from memory import people

        self.people = people
        self.bret = people.enroll_person("Bret Benziger")
        self.jt = people.enroll_person("JT")
        # Bret's reference along e1; JT far away along e3 so the inter-person
        # margin gate never interferes with the threshold under test.
        people.add_biometric(self.bret, "face", _e(0))
        people.add_biometric(self.jt, "face", _e(2))

    def test_strong_match_is_flagged_strong(self):
        # cos 0.70 → d 0.775: comfortably same-person.
        match = self.people.find_by_face(_unit_pair(0.70))
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], self.bret)
        self.assertTrue(match["face_match_strong"])
        self.assertAlmostEqual(match["face_match_distance"], 0.775, places=3)

    def test_gray_zone_match_is_not_strong(self):
        # cos 0.52 → d 0.980: accepted, but inside the gray band [0.90, 1.00).
        match = self.people.find_by_face(_unit_pair(0.52))
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], self.bret)
        self.assertFalse(match["face_match_strong"])

    def test_old_accept_band_now_rejected(self):
        # cos 0.45 → d 1.049: accepted under the old 1.10 bar (the PJ-as-Bret
        # regime), rejected under 1.00 — the face stays unknown and routes to
        # the who-are-you prompt instead of stealing an identity.
        self.assertIsNone(self.people.find_by_face(_unit_pair(0.45)))

    def test_impostor_band_rejected(self):
        # cos 0.10 → d 1.342: a genuine stranger.
        self.assertIsNone(self.people.find_by_face(_unit_pair(0.10)))


class ConfirmNewIdentityTest(unittest.TestCase):
    """Gray-zone matches need consecutive-tick confirmation to BIND anew."""

    def setUp(self):
        C._identify_streaks.clear()

    def tearDown(self):
        C._identify_streaks.clear()

    def _rec(self, *, strong: bool, dist: float = 0.97, pid: int = 1):
        return {
            "id": pid,
            "name": "Bret Benziger",
            "face_match_distance": dist,
            "face_match_strong": strong,
        }

    def test_strong_match_binds_first_frame(self):
        out = C._confirm_new_identity(self._rec(strong=True, dist=0.78), [], set())
        self.assertIsNotNone(out)

    def test_gray_match_held_on_first_frame(self):
        # The PJ shape: nothing bound yet, d in the gray band → no new binding.
        out = C._confirm_new_identity(self._rec(strong=False), [], set())
        self.assertIsNone(out)

    def test_gray_match_confirms_on_consecutive_ticks(self):
        confirm = int(getattr(config, "FACE_IDENTIFY_CONFIRM_FRAMES", 2))
        out = None
        for _tick in range(confirm):
            out = C._confirm_new_identity(self._rec(strong=False), [], set())
        self.assertIsNotNone(out)

    def test_gray_match_on_existing_binding_passes(self):
        people = [{"person_db_id": 1, "face_id": "Bret Benziger"}]
        out = C._confirm_new_identity(self._rec(strong=False), people, set())
        self.assertIsNotNone(out)

    def test_carried_identity_passes_untouched(self):
        # Sticky/switch-hysteresis records carry no match metadata — they were
        # vouched for when first bound.
        carried = {"id": 1, "name": "Bret Benziger"}
        out = C._confirm_new_identity(carried, [], set())
        self.assertIs(out, carried)

    def test_streak_reset_restarts_confirmation(self):
        C._confirm_new_identity(self._rec(strong=False), [], set())
        # The recognition loop prunes streaks on a tick with no match for this
        # person — after which one new gray tick must hold again.
        C._identify_streaks.clear()
        out = C._confirm_new_identity(self._rec(strong=False), [], set())
        self.assertIsNone(out)

    def test_same_tick_double_detection_counts_once(self):
        tick = set()
        C._confirm_new_identity(self._rec(strong=False), [], tick)
        C._confirm_new_identity(self._rec(strong=False), [], tick)
        self.assertEqual(C._identify_streaks.get(1), 1)


if __name__ == "__main__":
    unittest.main()
