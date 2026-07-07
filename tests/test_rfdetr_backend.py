"""
RF-DETR object-detector backend (owner spec 2026-07-06): replaces EfficientDet-
Lite0 as the default local animal/object detector. The adapter converts
supervision.Detections into the MediaPipe detection duck-type, so the existing
record builders (species thresholds, no-screens exclusions, position phrasing)
run backend-independently. MediaPipe remains the automatic fallback.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import config
from vision import animal_detector as ad


def _sv_detections(rows):
    """rows: [(x1, y1, x2, y2, score, class_id)] → supervision-shaped namespace."""
    return SimpleNamespace(
        xyxy=np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32),
        confidence=np.array([r[4] for r in rows], dtype=np.float32),
        class_id=np.array([r[5] for r in rows], dtype=np.int64),
    )


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self._classes = ad._rf_classes
        ad._rf_classes = {1: "person", 18: "dog", 47: "cup", 63: "laptop"}

    def tearDown(self):
        ad._rf_classes = self._classes

    def test_converts_to_mediapipe_duck_type(self):
        result = _sv_detections([(10, 20, 110, 220, 0.91, 18)])
        adapted = ad._rf_detections_to_mp(result)
        self.assertEqual(len(adapted), 1)
        det = adapted[0]
        self.assertEqual(det.categories[0].category_name, "dog")
        self.assertAlmostEqual(det.categories[0].score, 0.91, places=2)
        self.assertEqual(det.bounding_box.origin_x, 10.0)
        self.assertEqual(det.bounding_box.width, 100.0)
        self.assertEqual(det.bounding_box.height, 200.0)

    def test_unknown_class_id_is_skipped(self):
        result = _sv_detections([(0, 0, 10, 10, 0.9, 999)])
        self.assertEqual(ad._rf_detections_to_mp(result), [])

    def test_animal_records_flow_through_existing_builder(self):
        # A confident dog through the REAL record builder: species threshold,
        # position phrasing, furred flag — all backend-independent.
        result = _sv_detections([(100, 400, 500, 700, 0.85, 18)])
        adapted = ad._rf_detections_to_mp(result)
        records = ad._records_from_detections(adapted, (720, 1280, 3), now=123.0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["species"], "dog")
        self.assertEqual(records[0]["last_seen"], 123.0)
        self.assertIn("position", records[0])
        self.assertEqual(records[0]["box"], (100.0, 400.0, 400.0, 300.0))

    def test_object_records_apply_exclusions(self):
        # person + laptop (banned screen) + cup: only the cup becomes an object.
        result = _sv_detections([
            (0, 0, 100, 100, 0.95, 1),     # person -> excluded (world_state.people)
            (0, 0, 100, 100, 0.90, 63),    # laptop -> no-screens rule
            (200, 500, 300, 600, 0.80, 47) # cup -> object
        ])
        adapted = ad._rf_detections_to_mp(result)
        with mock.patch.object(config, "OBJECT_DETECTION_BANNED_CLASSES",
                               {"laptop"}, create=True):
            records = ad._object_records_from_detections(adapted, (720, 1280, 3))
        self.assertEqual([r["label"] for r in records], ["cup"])


class BackendDispatchTest(unittest.TestCase):
    def setUp(self):
        self._state = (ad._detector, ad._mp, ad._rf_model, ad._active_backend,
                       ad._load_attempted, ad._load_ok)
        ad._detector = None
        ad._mp = None
        ad._rf_model = None
        ad._active_backend = None
        ad._load_attempted = False
        ad._load_ok = False

    def tearDown(self):
        (ad._detector, ad._mp, ad._rf_model, ad._active_backend,
         ad._load_attempted, ad._load_ok) = self._state

    def test_rfdetr_failure_falls_back_to_mediapipe_path(self):
        with (
            mock.patch.object(config, "OBJECT_DETECTOR_BACKEND", "rfdetr", create=True),
            mock.patch.object(ad, "_load_rfdetr", return_value=False),
            mock.patch.object(ad, "_model_path") as mp_path,
        ):
            mp_path.return_value.exists.return_value = False  # mediapipe model absent too
            self.assertFalse(ad._load_model())
        self.assertIsNone(ad.active_backend())

    def test_rfdetr_success_sets_backend(self):
        with (
            mock.patch.object(config, "OBJECT_DETECTOR_BACKEND", "rfdetr", create=True),
            mock.patch.object(ad, "_load_rfdetr", return_value=True),
        ):
            self.assertTrue(ad._load_model())
        self.assertEqual(ad.active_backend(), "rfdetr")

    def test_source_field_tracks_backend(self):
        ad._active_backend = "rfdetr"
        self.assertEqual(ad._source(), "rfdetr_object_detector")
        ad._active_backend = "mediapipe"
        self.assertEqual(ad._source(), "mediapipe_object_detector")


if __name__ == "__main__":
    unittest.main()
