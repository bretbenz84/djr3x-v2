"""
Render test for the pose-wireframe overlay: the skeleton must be clipped to the
displayed video rect so limbs running off the edge are cut at the frame boundary
instead of bleeding into the panel's letterbox/border. Runs headless via the Qt
'offscreen' platform; skips cleanly if PySide6 / a Qt platform isn't available.
"""

from __future__ import annotations

import os
import time
import unittest

import numpy as np

# Force headless BEFORE any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from gui.vision_panel import VisionPanel, _scaled_rect
    _GUI_OK = True
except Exception:  # pragma: no cover - environment without a usable Qt platform
    _GUI_OK = False

_SKELETON_RGB = np.array([54, 217, 255])  # _SKELETON_COLOR "#36d9ff"


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class DrawablePersonGateTest(unittest.TestCase):
    """A faceless pose-only phantom slot (POSE_MAX_PEOPLE>1 hallucinating a stray skeleton)
    must NOT be drawn as a bogus 'Unknown' marker over the real person's labelled face."""

    def setUp(self):
        from gui.vision_panel import _slot_is_drawable_person
        self.drawable = _slot_is_drawable_person

    def test_real_face_is_drawn(self):
        self.assertTrue(self.drawable(
            {"person_db_id": 1, "face_id": "Bret Benziger",
             "face_box": (10, 10, 80, 80), "face_visible": True}))

    def test_unidentified_real_face_is_drawn(self):
        # A genuine unknown FACE (has a detected face box) still draws — that's a real person.
        self.assertTrue(self.drawable(
            {"person_db_id": None, "face_box": (10, 10, 80, 80), "face_visible": True}))

    def test_known_identity_without_box_is_drawn(self):
        # A momentarily box-less but identified slot keeps its marker.
        self.assertTrue(self.drawable(
            {"person_db_id": 1, "face_id": "Bret Benziger",
             "pose_keypoints": {"NOSE": (0.5, 0.4, 0.9)}}))

    def test_faceless_phantom_pose_slot_is_not_drawn(self):
        # No face box, no identity — the phantom that rendered "Unknown" at the pose nose.
        self.assertFalse(self.drawable(
            {"id": "person_2", "person_db_id": None, "face_id": None,
             "voice_id": None, "pose_keypoints": {"NOSE": (0.5, 0.4, 0.9)}}))


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class SkeletonClipTest(unittest.TestCase):
    def _render(self, people, *, frame=(540, 820), size=(640, 480)):
        panel = VisionPanel()
        panel.resize(*size)
        fh, fw = frame
        panel.set_snapshot({
            "frame": np.ones((fh, fw, 3), np.uint8) * 40,
            "world_state": {"people": people},
            "camera_stats": {"last_frame_monotonic": time.monotonic()},
        })
        img = QImage(panel.size(), QImage.Format.Format_RGB888)
        panel.render(img)
        arr = np.frombuffer(img.constBits().tobytes(), dtype=np.uint8)
        arr = arr.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 3]
        arr = arr.reshape(img.height(), img.width(), 3)
        content = QRectF(panel.rect().adjusted(16, 16, -16, -16))
        image_rect = _scaled_rect(fw, fh, QRectF(content.adjusted(0, 0, 0, -42)))
        return arr, image_rect

    def test_skeleton_does_not_bleed_past_video_edges(self):
        # Landmarks deliberately pushed PAST every frame edge.
        people = [{
            "id": "p1", "face_id": "Edge", "face_box": (0, 90, 210, 260),
            "face_visible": True, "face_missing": False,
            "pose_keypoints": {
                "NOSE": (0.05, 0.30, 1.0),
                "LEFT_SHOULDER": (0.12, 0.52, 1.0), "RIGHT_SHOULDER": (-0.04, 0.52, 1.0),
                "LEFT_ELBOW": (0.20, 0.66, 1.0), "RIGHT_ELBOW": (-0.10, 0.60, 1.0),
                "LEFT_WRIST": (1.10, 0.80, 1.0), "RIGHT_WRIST": (-0.08, 0.10, 1.0),
                "LEFT_HIP": (0.10, 1.05, 1.0), "RIGHT_HIP": (-0.02, 1.08, 1.0),
            },
        }]
        arr, ir = self._render(people)
        dist = np.abs(arr.astype(int) - _SKELETON_RGB).sum(axis=2)
        ys, xs = np.where(dist < 60)
        self.assertGreater(len(xs), 50, "skeleton should still render inside the frame")
        tol = 1.5  # allow sub-pixel antialiasing at the boundary
        outside = (
            (xs < ir.left() - tol) | (xs > ir.right() + tol)
            | (ys < ir.top() - tol) | (ys > ir.bottom() + tol)
        ).sum()
        self.assertEqual(int(outside), 0, "skeleton bled outside the video rect")


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class CoarseHandPointsTest(unittest.TestCase):
    """The wireframe extends past the wrist using the coarse pinky/index/thumb points the
    Pose model already publishes (no Hand Landmarker). They must be in the skeleton
    definitions AND actually render."""

    def _cyan_count(self, kp):
        from gui.vision_panel import VisionPanel
        panel = VisionPanel()
        panel.resize(640, 480)
        panel.set_snapshot({
            "frame": np.ones((480, 640, 3), np.uint8) * 40,
            "world_state": {"people": [{"id": "p", "face_visible": True, "pose_keypoints": kp}]},
            "camera_stats": {"last_frame_monotonic": time.monotonic()},
        })
        img = QImage(panel.size(), QImage.Format.Format_RGB888)
        panel.render(img)
        arr = np.frombuffer(img.constBits().tobytes(), dtype=np.uint8)
        arr = arr.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 3]
        arr = arr.reshape(img.height(), img.width(), 3)
        dist = np.abs(arr.astype(int) - _SKELETON_RGB).sum(axis=2)
        return int((dist < 60).sum())

    def test_finger_landmarks_in_skeleton_definitions(self):
        from gui import vision_panel as vp
        for n in ("LEFT_THUMB", "LEFT_INDEX", "LEFT_PINKY",
                  "RIGHT_THUMB", "RIGHT_INDEX", "RIGHT_PINKY"):
            self.assertIn(n, vp._POSE_JOINTS)
        self.assertIn(("LEFT_WRIST", "LEFT_INDEX"), vp._POSE_EDGES)
        self.assertIn(("RIGHT_WRIST", "RIGHT_PINKY"), vp._POSE_EDGES)

    def test_coarse_hand_points_render(self):
        # Same body; adding the in-frame finger points must draw MORE cyan (the hand fan
        # past the wrist) — proves the new dots/edges actually render, not silently no-op.
        base = {
            "NOSE": (0.50, 0.20, 1.0),
            "LEFT_SHOULDER": (0.60, 0.40, 1.0), "RIGHT_SHOULDER": (0.40, 0.40, 1.0),
            "LEFT_ELBOW": (0.66, 0.55, 1.0), "RIGHT_ELBOW": (0.34, 0.55, 1.0),
            "LEFT_WRIST": (0.70, 0.68, 1.0), "RIGHT_WRIST": (0.30, 0.68, 1.0),
        }
        fingers = {
            "LEFT_THUMB": (0.73, 0.71, 1.0), "LEFT_INDEX": (0.76, 0.69, 1.0),
            "LEFT_PINKY": (0.74, 0.74, 1.0),
            "RIGHT_THUMB": (0.27, 0.71, 1.0), "RIGHT_INDEX": (0.24, 0.69, 1.0),
            "RIGHT_PINKY": (0.26, 0.74, 1.0),
        }
        self.assertGreater(
            self._cyan_count({**base, **fingers}), self._cyan_count(dict(base)),
            "coarse hand points/edges did not add to the wireframe",
        )


if __name__ == "__main__":
    unittest.main()
