"""Tests for face-tracking jump rejection (consciousness._evaluate_face_jump).

A real face can't teleport across the frame in one ~0.08s tick, so a detection box
that jumps too far from the last accepted position is ignored (head holds its gaze) —
unless the new position persists, which means a genuine fast move.
"""

import unittest
from unittest import mock


class FaceJumpRejectionTest(unittest.TestCase):
    FRAME_W, FRAME_H = 1920, 1080  # diag ~2203; frac 0.33 -> max jump ~727 px

    def setUp(self):
        from intelligence import consciousness

        self.c = consciousness

    def _eval(self, cx, cy, key, now, last, pending, **cfg):
        c = self.c
        patches = {
            "FACE_TRACKING_MAX_JUMP_FRAC": 0.33,
            "FACE_TRACKING_JUMP_CONFIRM_SECS": 0.5,
            "FACE_TRACKING_JUMP_MAX_AGE_SECS": 0.5,
        }
        patches.update(cfg)
        with mock.patch.multiple(c.config, **patches):
            return c._evaluate_face_jump(
                cx, cy, key, now, self.FRAME_W, self.FRAME_H, last, pending
            )

    def test_small_move_accepted(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        accept, new_last, new_pend = self._eval(950, 560, "p1", 100.05, last, None)
        self.assertTrue(accept)
        self.assertEqual((new_last["cx"], new_last["cy"]), (950, 560))
        self.assertIsNone(new_pend)

    def test_big_jump_rejected_and_pending_set(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        accept, new_last, new_pend = self._eval(100, 1000, "p1", 100.05, last, None)  # ~923px > 727
        self.assertFalse(accept)
        # Gaze reference keeps its POSITION but the timestamp refreshes: a
        # rejection must not let the reference age past max_age, or the same
        # spurious box would be auto-accepted via staleness a tick later
        # (see test_face_jump_guard for the live-failure replay).
        self.assertEqual(
            (new_last["key"], new_last["cx"], new_last["cy"]),
            (last["key"], last["cx"], last["cy"]),
        )
        self.assertEqual(float(new_last["at"]), 100.05)
        self.assertIsNotNone(new_pend)
        self.assertEqual((new_pend["cx"], new_pend["cy"]), (100, 1000))

    def test_persistent_jump_confirmed_after_window(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        pend = {"cx": 100, "cy": 1000, "since": 100.0}
        # same far spot, 0.6s later (> confirm 0.5) -> accept it as a real move
        accept, new_last, new_pend = self._eval(110, 1010, "p1", 100.6, last, pend)
        self.assertTrue(accept)
        self.assertEqual((new_last["cx"], new_last["cy"]), (110, 1010))
        self.assertIsNone(new_pend)

    def test_jump_still_rejected_before_confirm_window(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        pend = {"cx": 100, "cy": 1000, "since": 100.0}
        accept, _, new_pend = self._eval(105, 1005, "p1", 100.2, last, pend)  # only 0.2s < 0.5
        self.assertFalse(accept)
        self.assertIsNotNone(new_pend)

    def test_first_lock_seeds_reference(self):
        accept, new_last, new_pend = self._eval(500, 500, "p1", 100.0, None, None)
        self.assertTrue(accept)
        self.assertEqual(new_last["key"], "p1")

    def test_stale_reference_reseeds(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        # 2s later (> max_age 0.5): treat as a fresh seed, accept even a far box
        accept, new_last, _ = self._eval(100, 1000, "p1", 102.0, last, None)
        self.assertTrue(accept)

    def test_different_person_seeds_not_rejected(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        accept, new_last, _ = self._eval(100, 1000, "p2", 100.05, last, None)
        self.assertTrue(accept)
        self.assertEqual(new_last["key"], "p2")

    def test_disabled_when_frac_zero(self):
        last = {"key": "p1", "cx": 900, "cy": 540, "at": 100.0}
        accept, _, _ = self._eval(100, 1000, "p1", 100.05, last, None, FACE_TRACKING_MAX_JUMP_FRAC=0.0)
        self.assertTrue(accept)


if __name__ == "__main__":
    unittest.main()
