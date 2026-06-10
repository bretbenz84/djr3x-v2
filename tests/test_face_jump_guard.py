"""Face-tracking jump-guard regression tests.

Replays of the wild-head-movement failure from logs/djr3x-2026-06-09-18-39-25.log:
spurious detector boxes were accepted unconditionally because the guard's
reference was stale (detections land ~every 2s; max_age was 1.5s), and a
rejection left the reference un-refreshed so the same box auto-accepted one
tick later (rejected (938,838) at 18:43:16, accepted (937,835) at 18:43:17).
The head chased clutter across the lift servo's range (1995..7270 qus).

Guard policy: "random UNKNOWN face detected far away = noise."
  - A box dlib freshly identified as a known person is followed immediately —
    a seated person low in the frame is real, clutter can't match an encoding.
  - live_tracked boxes inherit person_id from an older recognition pass, so
    they do NOT get the identity instant-accept.
  - Unknown teleports are held until the position persists (confirm path);
    staleness is no longer a way in.
"""

import unittest
from unittest import mock

FRAME_W, FRAME_H = 1920, 1080
# diag ≈ 2203 px; max_jump = 0.15 × diag ≈ 330 px


def _evaluate(cx, cy, now, last_center, pending_center, **kw):
    from intelligence import consciousness

    return consciousness._evaluate_face_jump(
        cx, cy, "db:1", now, FRAME_W, FRAME_H, last_center, pending_center, **kw
    )


def _config_patches():
    import config

    return (
        mock.patch.object(config, "FACE_TRACKING_MAX_JUMP_FRAC", 0.15),
        mock.patch.object(config, "FACE_TRACKING_JUMP_CONFIRM_SECS", 0.5),
        mock.patch.object(config, "FACE_TRACKING_JUMP_MAX_AGE_SECS", 5.0),
    )


class FaceJumpGuardTests(unittest.TestCase):
    def setUp(self):
        self._patches = _config_patches()
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def test_guard_engages_across_real_detection_cadence(self):
        """With max_age above the ~2s HOG cadence, a clutter teleport is rejected."""
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        # 2s later (one real detection cycle): box teleports 480+ px to bottom edge.
        accept, last, pending = _evaluate(940, 975, 2.0, last, None)
        self.assertFalse(accept)
        self.assertIsNotNone(pending)

    def test_rejection_refreshes_reference_no_staleness_backdoor(self):
        """A flickering clutter box can't sneak in via reference staleness."""
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        accept, last, pending = _evaluate(940, 975, 2.0, last, None)
        self.assertFalse(accept)
        self.assertEqual(float(last["at"]), 2.0)  # reference kept alive
        # 2s later the clutter flickers to a DIFFERENT far spot. Old behaviour
        # (no refresh, max_age 1.5) accepted this unconditionally; now it must
        # be rejected and the pending resets to the new spot.
        accept, last, pending = _evaluate(300, 200, 4.0, last, pending)
        self.assertFalse(accept)
        self.assertEqual(float(last["at"]), 4.0)
        self.assertEqual((pending["cx"], pending["cy"]), (300, 200))

    def test_identified_fresh_detection_followed_immediately(self):
        """Dlib-recognized known face far away = the person really moved (seated,
        bottom of frame) — follow it, never treat it as noise."""
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        accept, last, pending = _evaluate(
            940, 975, 2.0, last, None, identified=True, live_tracked=False
        )
        self.assertTrue(accept)
        self.assertEqual((last["cx"], last["cy"]), (940, 975))

    def test_live_tracked_identity_does_not_instant_accept(self):
        """A correlation-tracked box inherits its person_id from an older pass —
        a drifted tracker is not identity evidence for the new position."""
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        accept, last, pending = _evaluate(
            940, 975, 2.0, last, None, identified=True, live_tracked=True
        )
        self.assertFalse(accept)
        self.assertIsNotNone(pending)

    def test_unknown_jump_accepted_after_persistence(self):
        """A genuinely moved/new unknown face still gets in via the confirm path."""
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        accept, last, pending = _evaluate(940, 975, 2.0, last, None)
        self.assertFalse(accept)
        # Same spot one detection cycle later: persisted ≥ confirm_secs → accepted.
        accept, last, pending = _evaluate(945, 970, 4.0, last, pending)
        self.assertTrue(accept)
        self.assertEqual((last["cx"], last["cy"]), (945, 970))

    def test_small_moves_always_accepted(self):
        accept, last, _ = _evaluate(960, 540, 0.0, None, None)
        self.assertTrue(accept)
        accept, last, _ = _evaluate(1050, 600, 2.0, last, None)  # ~108 px
        self.assertTrue(accept)


if __name__ == "__main__":
    unittest.main()
