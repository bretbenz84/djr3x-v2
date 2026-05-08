import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None


class VisionPanelMoodLabelTests(unittest.TestCase):
    def test_face_mood_wins_over_generic_expression(self):
        from gui.vision_panel import _person_expression

        self.assertEqual(
            _person_expression({
                "expression": "neutral",
                "face_mood": {"mood": "happy", "confidence": 0.86},
            }),
            "happy",
        )

    def test_face_expression_wins_over_mood_for_live_box_label(self):
        from gui.vision_panel import _person_expression

        self.assertEqual(
            _person_expression({
                "expression": "neutral",
                "face_mood": {"mood": "happy", "confidence": 0.86},
                "face_expression": {
                    "expression": "smile",
                    "mood": "happy",
                    "confidence": 0.81,
                },
            }),
            "smile",
        )

    def test_missing_mood_does_not_invent_neutral(self):
        from gui.vision_panel import _person_expression

        self.assertEqual(_person_expression({"face_id": "Bret"}), "")

    def test_dlib_panel_formats_local_face_expression(self):
        from gui.dashboard import _format_expression

        self.assertEqual(
            _format_expression({
                "face_expression": {
                    "expression": "smile",
                    "mood": "happy",
                    "confidence": 0.82,
                    "notes": "smiling",
                    "source": "mediapipe_face_landmarker",
                },
            }),
            "smile / happy / 82% / smiling / mediapipe face landmarker",
        )


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class LiveFaceBoxTrackerTests(unittest.TestCase):
    def _frame_with_patch(self, x: int, y: int) -> np.ndarray:
        rng = np.random.default_rng(1234)
        patch = rng.integers(0, 255, size=(36, 36), dtype=np.uint8)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[y:y + 36, x:x + 36] = np.repeat(patch[:, :, None], 3, axis=2)
        return frame

    def test_live_box_tracks_between_stale_world_state_boxes(self):
        from gui.live_face_tracker import LiveFaceBoxTracker

        tracker = LiveFaceBoxTracker()
        people = [{
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (20, 30, 36, 36),
        }]

        first = tracker.update(self._frame_with_patch(20, 30), people, now=1.0)
        second = tracker.update(self._frame_with_patch(32, 37), people, now=1.05)

        self.assertEqual(first[0]["face_box"], (20.0, 30.0, 36.0, 36.0))
        self.assertGreater(second[0]["face_box"][0], 28.0)
        self.assertGreater(second[0]["face_box"][1], 34.0)
        self.assertTrue(second[0]["gui_live_tracked"])

    def test_new_recognition_box_reseeds_tracker(self):
        from gui.live_face_tracker import LiveFaceBoxTracker

        tracker = LiveFaceBoxTracker()
        original = [{
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (20, 30, 36, 36),
        }]
        updated = [{
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (70, 42, 36, 36),
        }]

        tracker.update(self._frame_with_patch(20, 30), original, now=1.0)
        result = tracker.update(self._frame_with_patch(70, 42), updated, now=1.05)

        self.assertEqual(result[0]["face_box"], (70.0, 42.0, 36.0, 36.0))

    def test_live_box_expires_without_fresh_source(self):
        from gui.live_face_tracker import LiveFaceBoxTracker

        tracker = LiveFaceBoxTracker(stale_secs=0.10)
        visible = [{
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (20, 30, 36, 36),
        }]
        missing = [{
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": False,
            "face_missing": True,
            "face_box": None,
        }]

        tracker.update(self._frame_with_patch(20, 30), visible, now=1.0)
        bridged = tracker.update(self._frame_with_patch(32, 37), missing, now=1.05)
        expired = tracker.update(self._frame_with_patch(44, 44), missing, now=1.25)

        self.assertTrue(bridged[0]["gui_live_tracked"])
        self.assertTrue(bridged[0]["face_visible"])
        self.assertFalse(expired[0].get("gui_live_tracked"))
        self.assertFalse(expired[0]["face_visible"])
        self.assertTrue(expired[0]["face_missing"])
        self.assertIsNone(expired[0]["face_box"])


if __name__ == "__main__":
    unittest.main()
