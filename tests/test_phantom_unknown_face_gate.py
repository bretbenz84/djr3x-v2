"""Regression tests for the phantom "unknown person" false-positive (logs 2026-06-23).

While the user was alone (lying down), MediaPipe hallucinated a SECOND pose skeleton — a
person-slot with person_db_id=None but NO face_box. Both the social-scene unknown count and
the llm.py "UNKNOWN PERSON IN FRAME" curiosity counted any person_db_id-less slot as an
unknown person, so a faceless pose-phantom kept firing "who's your friend?" ~13× and hijacked
the 20 Questions game.

Fixes under test:
  1. social_scene only counts a slot as unknown if it has an actual visible face.
  2. The unknown-person identity handoff stands down entirely while a game owns the turn.
"""

import unittest
from unittest import mock

from features import games
from intelligence import social_scene as ss


def _snapshot(*people, crowd=None):
    snap = {"people": list(people)}
    if crowd is not None:
        snap["crowd"] = {"count": crowd}
    return snap


_KNOWN = {"person_db_id": 1, "face_id": "Bret Benziger", "face_box": [1, 2, 3, 4],
          "face_visible": True}
_PHANTOM_POSE = {"person_db_id": None}                       # no face_box → not a real person
_REAL_UNKNOWN = {"person_db_id": None, "face_box": [5, 6, 7, 8], "face_visible": True}


class FaceGateTest(unittest.TestCase):
    def setUp(self):
        games._active_game = None
        self.addCleanup(setattr, games, "_active_game", None)

    def test_phantom_pose_does_not_count_as_unknown(self):
        scene = ss.from_snapshot(_snapshot(_KNOWN, _PHANTOM_POSE))
        self.assertEqual(scene.unknown_count, 0)

    def test_real_unknown_face_still_counts(self):
        scene = ss.from_snapshot(_snapshot(_KNOWN, _REAL_UNKNOWN))
        self.assertEqual(scene.unknown_count, 1)

    def test_face_marked_not_visible_does_not_count(self):
        slot = {"person_db_id": None, "face_box": [5, 6, 7, 8], "face_visible": False}
        self.assertEqual(ss.from_snapshot(_snapshot(_KNOWN, slot)).unknown_count, 0)
        slot2 = {"person_db_id": None, "face_box": [5, 6, 7, 8], "face_missing": True}
        self.assertEqual(ss.from_snapshot(_snapshot(_KNOWN, slot2)).unknown_count, 0)

    def test_slot_has_visible_face_helper(self):
        self.assertTrue(ss._slot_has_visible_face(_REAL_UNKNOWN))
        self.assertFalse(ss._slot_has_visible_face(_PHANTOM_POSE))


class GameSuppressionTest(unittest.TestCase):
    def setUp(self):
        games._active_game = None
        self.addCleanup(setattr, games, "_active_game", None)

    def test_unknown_handoff_present_when_no_game(self):
        snap = _snapshot(_KNOWN, _REAL_UNKNOWN, crowd=2)
        out = ss.unknown_group_context(snap, current_person_id=1, current_person_name="Bret")
        self.assertIsNotNone(out)

    def test_unknown_handoff_suppressed_during_game(self):
        games._active_game = "20_questions"
        snap = _snapshot(_KNOWN, _REAL_UNKNOWN, crowd=2)
        out = ss.unknown_group_context(snap, current_person_id=1, current_person_name="Bret")
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
