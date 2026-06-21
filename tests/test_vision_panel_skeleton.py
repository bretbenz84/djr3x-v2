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


if __name__ == "__main__":
    unittest.main()
