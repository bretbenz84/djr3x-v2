"""Engagement misread + phantom wall faces (field 2026-08-03, 23:55 session).

Two failures with one session of evidence:
  * The owner answered "It's a Delorean." (terse but DIRECTLY responsive), the
    energy read flipped to quiet, and the lean brain mirrored his silence for a
    full minute while he sat looking straight at Rex, deliberately waiting for
    him to strike up conversation.
  * The busy workshop wall kept minting phantom dlib faces on pose-miss ticks
    (the guard used to stand down with no pose anchors) — the head snapped
    up/down chasing them and Rex waved at the wall.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import consciousness
from intelligence import user_energy


class TerseAnswerNotDisengagementTest(unittest.TestCase):
    def test_short_answer_to_rex_question_is_not_quiet(self):
        profile = user_energy._classify(
            "It's a Delorean.",
            prosody_features=None,
            affect_result=None,
            answered_question={"question_key": "gift_kind"},
        )
        self.assertNotEqual(profile.mode, "quiet")
        self.assertNotEqual(profile.engagement, "low")
        self.assertNotEqual(profile.question_appetite, "low")
        self.assertIn("terse but direct answer", profile.signals)

    def test_unprompted_terseness_still_reads_quiet(self):
        profile = user_energy._classify(
            "Yeah, sure.",
            prosody_features=None,
            affect_result=None,
            answered_question=None,
        )
        self.assertEqual(profile.mode, "quiet")


class PersonVisiblyFacingTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(consciousness._face_tracking_lock)
        self.addCleanup(
            lambda: consciousness._face_tracking_lock.update(self._saved)
            or None
        )

    def _set_lock(self, person_id, age_secs):
        consciousness._face_tracking_lock.clear()
        consciousness._face_tracking_lock.update({
            "key": f"db:{person_id}",
            "person_id": person_id,
            "last_seen_at": time.monotonic() - age_secs,
        })

    def test_fresh_lock_on_person_reads_facing(self):
        self._set_lock(1, age_secs=2.0)
        self.assertTrue(consciousness.person_visibly_facing(1, max_age_secs=6.0))

    def test_stale_lock_does_not_read_facing(self):
        self._set_lock(1, age_secs=30.0)
        self.assertFalse(consciousness.person_visibly_facing(1, max_age_secs=6.0))

    def test_lock_on_other_person_does_not_read_facing(self):
        self._set_lock(2, age_secs=1.0)
        self.assertFalse(consciousness.person_visibly_facing(1, max_age_secs=6.0))

    def test_no_person_no_facing(self):
        self._set_lock(1, age_secs=1.0)
        self.assertFalse(consciousness.person_visibly_facing(None))


class PhantomFaceGuardStickyAnchorsTest(unittest.TestCase):
    """The guard must keep working through pose-miss ticks via cached anchors."""

    def setUp(self):
        consciousness._pose_anchor_cache = []
        consciousness._pose_anchor_cache_at = 0.0
        self.addCleanup(setattr, consciousness, "_pose_anchor_cache", [])
        self.addCleanup(setattr, consciousness, "_pose_anchor_cache_at", 0.0)

    # A real face near the pose head at (900, 600, head_w=120) and a wall
    # phantom far away at (500, 300).
    _REAL = {"bounding_box": [850, 550, 100, 100]}
    _PHANTOM = {"bounding_box": [450, 250, 100, 100]}

    def _guard(self, faces, anchors):
        with mock.patch("vision.pose.head_anchors_px", return_value=anchors):
            return consciousness._reject_faces_off_body(list(faces), 1920, 1080)

    def test_live_anchors_drop_wall_phantom(self):
        kept = self._guard([self._REAL, self._PHANTOM], [(900, 600, 120)])
        self.assertEqual(kept, [self._REAL])

    def test_pose_miss_tick_uses_cached_anchors(self):
        # Tick 1: pose present — anchors cached, phantom dropped.
        self._guard([self._REAL, self._PHANTOM], [(900, 600, 120)])
        # Tick 2: pose MISSED — the cached anchors must still kill the phantom
        # (this exact gap is where the wall faces leaked before).
        kept = self._guard([self._REAL, self._PHANTOM], [])
        self.assertEqual(kept, [self._REAL])

    def test_expired_cache_stands_down(self):
        self._guard([self._REAL], [(900, 600, 120)])
        consciousness._pose_anchor_cache_at = (
            time.monotonic()
            - float(getattr(config, "POSE_FACE_GUARD_ANCHOR_TTL_SECS", 2.5))
            - 1.0
        )
        kept = self._guard([self._PHANTOM], [])
        self.assertEqual(kept, [self._PHANTOM])  # can't guard — trust detection

    def test_cached_radius_is_wider_than_live(self):
        # A face 300px from the anchor (2.5 head-widths): dropped against LIVE
        # anchors (mult 1.5) but kept against CACHED ones (mult 3.0) — the head
        # may have panned since the anchors were captured.
        borderline = {"bounding_box": [1150, 550, 100, 100]}
        kept_live = self._guard([borderline], [(900, 600, 120)])
        self.assertEqual(kept_live, [])
        kept_cached = self._guard([borderline], [])
        self.assertEqual(kept_cached, [borderline])


if __name__ == "__main__":
    unittest.main()
