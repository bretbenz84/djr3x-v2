"""
Unknown-face persistence gate: a detected unknown face must persist
FACE_UNKNOWN_CONFIRM_FRAMES consecutive recognition ticks before Rex treats it as a real
person (and arms the "who's the mystery guest?" agenda). This filters transient phantom
faces (clutter / a shape on the wall) — the live-run failure where Rex kept asking Bret
about non-existent guests — while a genuine newcomer, who persists, still clears the gate.

The recognition step computes `expose_unknown = (streak + 1) >= confirm` at tick start,
then commits the streak via _update_unknown_streak at tick end; these tests exercise that
exact sequence through the public helpers.
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness as c


class UnknownFacePersistenceTest(unittest.TestCase):
    def setUp(self):
        c._update_unknown_streak(False)  # reset streak to 0

    def _tick(self, had_unknown: bool) -> bool:
        """One recognition tick: returns whether an unknown face would be EXPOSED this
        tick, then commits the streak — mirroring _step_person_recognition."""
        expose = (c._unknown_visible_streak + 1) >= c._unknown_confirm_frames()
        c._update_unknown_streak(had_unknown)
        return bool(had_unknown and expose)

    def test_transient_phantom_is_never_exposed(self):
        with mock.patch.object(config, "FACE_UNKNOWN_CONFIRM_FRAMES", 3):
            c._update_unknown_streak(False)
            self.assertFalse(self._tick(True))   # tick 1 — phantom appears
            self.assertFalse(self._tick(True))   # tick 2 — still there
            self._tick(False)                    # phantom gone
            self.assertEqual(c._unknown_visible_streak, 0)

    def test_persistent_face_exposed_on_the_confirm_frame(self):
        with mock.patch.object(config, "FACE_UNKNOWN_CONFIRM_FRAMES", 3):
            c._update_unknown_streak(False)
            self.assertFalse(self._tick(True))   # 1
            self.assertFalse(self._tick(True))   # 2
            self.assertTrue(self._tick(True))    # 3 — confirmed, now a person
            self.assertTrue(self._tick(True))    # 4 — stays confirmed

    def test_a_gap_resets_the_streak(self):
        with mock.patch.object(config, "FACE_UNKNOWN_CONFIRM_FRAMES", 3):
            c._update_unknown_streak(False)
            self._tick(True)
            self._tick(True)                     # streak 2
            self._tick(False)                    # gap → reset
            self.assertFalse(self._tick(True))   # back to square one, not exposed

    def test_confirm_frames_1_restores_immediate_behavior(self):
        with mock.patch.object(config, "FACE_UNKNOWN_CONFIRM_FRAMES", 1):
            c._update_unknown_streak(False)
            self.assertTrue(self._tick(True))    # gate disabled → exposed immediately


if __name__ == "__main__":
    unittest.main()
