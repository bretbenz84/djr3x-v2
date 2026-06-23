"""
Phantom-face guard: dlib sometimes throws a spurious face off the body (the GUI box jumps
up the frame). The MediaPipe pose head is the source of truth, so a detected face far from
it is dropped. Covers the pose head anchor and the rejection helper.
"""

import unittest
from unittest import mock


class HeadAnchorTest(unittest.TestCase):
    def setUp(self):
        from world_state import world_state
        self.ws = world_state
        self.ws.mutate("people", lambda _c: [{
            "id": "person_1",
            "pose_keypoints": {
                "NOSE": (0.50, 0.40, 0.95),
                "LEFT_EAR": (0.56, 0.40, 0.90),
                "RIGHT_EAR": (0.44, 0.40, 0.90),
                "LEFT_SHOULDER": (0.62, 0.60, 0.90),
                "RIGHT_SHOULDER": (0.38, 0.60, 0.90),
            },
        }])
        self.addCleanup(lambda: self.ws.mutate("people", lambda _c: []))

    def test_anchor_from_nose_and_ear_span(self):
        from vision import pose
        anchor = pose.head_anchor_px(1000, 800)
        self.assertIsNotNone(anchor)
        hx, hy, head_w = anchor
        self.assertAlmostEqual(hx, 500.0)        # nose.x 0.50 * 1000
        self.assertAlmostEqual(hy, 320.0)        # nose.y 0.40 * 800
        self.assertAlmostEqual(head_w, 120.0, delta=1.0)  # ear span 0.12 * 1000

    def test_no_keypoints_returns_none(self):
        from vision import pose
        self.ws.mutate("people", lambda _c: [{"id": "p", "face_box": (1, 2, 3, 4)}])
        self.assertIsNone(pose.head_anchor_px(1000, 800))


class RejectFacesOffBodyTest(unittest.TestCase):
    def _face(self, cx, cy, size=100):
        return {"bounding_box": (cx - size // 2, cy - size // 2, size, size)}

    def _centers(self, faces):
        return [(b[0] + b[2] // 2, b[1] + b[3] // 2) for b in (f["bounding_box"] for f in faces)]

    def test_drops_far_face_keeps_near(self):
        from intelligence import consciousness as c
        near = self._face(510, 330)   # ~14px from pose head → real
        far = self._face(510, 60)     # ~260px above → phantom
        with mock.patch("vision.pose.head_anchors_px", return_value=[(500.0, 320.0, 120.0)]):
            out = c._reject_faces_off_body([near, far], 1000, 800)
        self.assertIn((510, 330), self._centers(out))
        self.assertNotIn((510, 60), self._centers(out))

    def test_keeps_face_near_a_second_person(self):
        # Two tracked bodies: a face near the SECOND head must be kept. This is the
        # multi-person regression — the old single-head guard dropped a real second
        # person's face (the "boss had no bounding box" report).
        from intelligence import consciousness as c
        p1 = self._face(300, 320)     # near head A
        p2 = self._face(710, 330)     # near head B (the "boss")
        phantom = self._face(510, 60)  # far from both heads
        with mock.patch(
            "vision.pose.head_anchors_px",
            return_value=[(300.0, 320.0, 120.0), (700.0, 320.0, 120.0)],
        ):
            out = c._reject_faces_off_body([p1, p2, phantom], 1000, 800)
        centers = self._centers(out)
        self.assertIn((300, 320), centers)
        self.assertIn((710, 330), centers)
        self.assertNotIn((510, 60), centers)

    def test_no_pose_anchor_keeps_all(self):
        from intelligence import consciousness as c
        faces = [self._face(10, 10), self._face(900, 700)]
        with mock.patch("vision.pose.head_anchors_px", return_value=[]):
            self.assertEqual(c._reject_faces_off_body(faces, 1000, 800), faces)

    def test_disabled_keeps_all(self):
        from intelligence import consciousness as c
        faces = [self._face(10, 10)]
        with mock.patch.object(c.config, "POSE_FACE_GUARD_ENABLED", False):
            self.assertEqual(c._reject_faces_off_body(faces, 1000, 800), faces)


if __name__ == "__main__":
    unittest.main()
