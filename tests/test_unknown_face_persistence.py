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


class UnknownFaceConfidenceGateTest(unittest.TestCase):
    """Detector-confidence floor for UNKNOWN faces (live-logged 2026-08-05): a busy-room
    shelf minted a PERSISTENT phantom face — so the confirm-frames streak couldn't help —
    that survived the pose-face guard once the room emptied (the only remaining pose head
    was a phantom pose on the same clutter) and got the full "what name should I save for
    you?" prompt. An unidentified face must now also clear FACE_UNKNOWN_MIN_CONFIDENCE on
    the detector's own score; known-face tracking (embedding-vouched) is unaffected."""

    def setUp(self):
        c._last_lowconf_face_log_at = 0.0

    def test_low_score_unknown_is_ignored(self):
        with mock.patch.object(config, "FACE_UNKNOWN_MIN_CONFIDENCE", 0.62):
            self.assertFalse(c._unknown_face_conf_ok(
                {"confidence": 0.55, "bounding_box": (545, 519, 60, 70)}))

    def test_high_score_unknown_passes(self):
        with mock.patch.object(config, "FACE_UNKNOWN_MIN_CONFIDENCE", 0.62):
            self.assertTrue(c._unknown_face_conf_ok({"confidence": 0.80}))

    def test_score_at_the_floor_passes(self):
        with mock.patch.object(config, "FACE_UNKNOWN_MIN_CONFIDENCE", 0.62):
            self.assertTrue(c._unknown_face_conf_ok({"confidence": 0.62}))

    def test_dlib_detection_without_score_passes(self):
        # The dlib backend carries no det score — the gate is insightface-only.
        self.assertTrue(c._unknown_face_conf_ok({"bounding_box": (1, 2, 3, 4)}))

    def test_floor_zero_disables_gate(self):
        with mock.patch.object(config, "FACE_UNKNOWN_MIN_CONFIDENCE", 0.0):
            self.assertTrue(c._unknown_face_conf_ok({"confidence": 0.10}))


if __name__ == "__main__":
    unittest.main()
