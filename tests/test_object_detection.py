"""Local COCO object stream → world_state.objects (+ GUI bounding boxes).

The animal detector already runs the full 80-class model and discards every
non-animal box; this stream KEEPS the rest as world_state.objects (the §2 curiosity
substrate), minus screens/devices (no-screens rule), people, and animals.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class _FakeBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class _FakeDetection:
    def __init__(self, name, score, box=None):
        self.categories = [_FakeCategory(name, score)]
        self.bounding_box = box if box is not None else _FakeBox(10, 10, 60, 60)


class ObjectDetectorRecordsTests(unittest.TestCase):
    def _records(self, detections):
        from vision import animal_detector

        with (
            mock.patch.object(animal_detector.config, "LOCAL_ANIMAL_DETECTION_SPECIES", {"dog", "cat"}),
            mock.patch.object(
                animal_detector.config, "OBJECT_DETECTION_BANNED_CLASSES",
                {"laptop", "tv", "cell phone"},
            ),
            mock.patch.object(animal_detector.config, "OBJECT_DETECTION_SCORE_THRESHOLD", 0.35),
        ):
            return animal_detector._object_records_from_detections(
                detections, (720, 1280, 3), now=42.0
            )

    def test_keeps_room_items_drops_screens_animals_and_people(self):
        records = self._records([
            _FakeDetection("chair", 0.90),
            _FakeDetection("potted plant", 0.80),
            _FakeDetection("laptop", 0.98),   # screen → no-screens rule
            _FakeDetection("dog", 0.97),       # animal → world_state.animals
            _FakeDetection("person", 0.99),    # person → world_state.people
        ])
        self.assertEqual({r["label"] for r in records}, {"chair", "potted plant"})

    def test_record_shape(self):
        from vision import animal_detector

        records = self._records([_FakeDetection("cup", 0.72, _FakeBox(100, 120, 40, 50))])
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["label"], "cup")
        self.assertEqual(rec["source"], animal_detector._SOURCE)
        self.assertEqual(rec["last_seen"], 42.0)
        self.assertEqual(rec["box"], (100.0, 120.0, 40.0, 50.0))
        self.assertTrue(rec["id"].startswith("object_"))
        self.assertTrue(0.0 <= rec["confidence"] <= 1.0)
        self.assertIn("position", rec)

    def test_below_score_threshold_is_dropped(self):
        records = self._records([_FakeDetection("cup", 0.20)])
        self.assertEqual(records, [])

    def test_excluded_classes_union_of_banned_animals_and_person(self):
        from vision import animal_detector

        with (
            mock.patch.object(animal_detector.config, "LOCAL_ANIMAL_DETECTION_SPECIES", {"dog", "cat"}),
            mock.patch.object(animal_detector.config, "OBJECT_DETECTION_BANNED_CLASSES", {"laptop"}),
        ):
            excluded = animal_detector._object_excluded_classes()
        self.assertTrue({"laptop", "dog", "cat", "person"} <= excluded)

    def test_detect_objects_returns_none_when_disabled(self):
        from vision import animal_detector

        with mock.patch.object(animal_detector.config, "OBJECT_DETECTION_ENABLED", False):
            self.assertIsNone(animal_detector.detect_objects(object()))


class SceneObjectStreamTests(unittest.TestCase):
    def setUp(self):
        from vision import scene

        scene._object_confirm_streak.clear()
        from world_state import world_state

        self._saved = world_state.get("objects")
        self.addCleanup(lambda: world_state.update("objects", self._saved))
        self.addCleanup(scene._object_confirm_streak.clear)

    def test_confirm_persistent_objects_requires_consecutive_scans(self):
        from vision import scene

        objs = [{"label": "chair", "position": "center"}]
        with mock.patch.object(scene.config, "OBJECT_DETECTION_CONFIRM_SCANS", 2):
            self.assertEqual(scene._confirm_persistent_objects(objs), [])      # scan 1
            confirmed = scene._confirm_persistent_objects(objs)                 # scan 2
        self.assertEqual([o["label"] for o in confirmed], ["chair"])

    def test_detect_objects_local_publishes_to_world_state(self):
        from vision import scene
        from world_state import world_state

        objs = [{"id": "object_1", "label": "chair", "position": "center", "confidence": 0.9}]
        with (
            mock.patch.object(scene.local_animal_detector, "detect_objects", return_value=objs),
            mock.patch.object(scene.config, "OBJECT_DETECTION_CONFIRM_SCANS", 1),
        ):
            result = scene.detect_objects_local(object())
        self.assertEqual([o["label"] for o in result], ["chair"])
        self.assertEqual([o["label"] for o in world_state.get("objects")], ["chair"])

    def test_detect_objects_local_preserves_state_when_unavailable(self):
        from vision import scene
        from world_state import world_state

        world_state.update("objects", [{"label": "lamp"}])
        with mock.patch.object(scene.local_animal_detector, "detect_objects", return_value=None):
            result = scene.detect_objects_local(object())
        self.assertEqual([o["label"] for o in result], ["lamp"])


class WorldStateObjectsKeyTests(unittest.TestCase):
    def test_objects_key_exists_and_is_writable(self):
        from world_state import world_state

        saved = world_state.get("objects")
        try:
            self.assertIsInstance(saved, list)  # present in _DEFAULTS
            world_state.update("objects", [{"label": "chair"}])
            self.assertEqual(world_state.get("objects"), [{"label": "chair"}])
        finally:
            world_state.update("objects", saved)


try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    _HAVE_QT = True
except Exception:  # pragma: no cover
    _HAVE_QT = False


@unittest.skipUnless(_HAVE_QT, "PySide6 not available")
class ObjectLabelTests(unittest.TestCase):
    def test_object_label_reads_label_then_falls_back(self):
        from gui import vision_panel

        self.assertEqual(vision_panel._object_label({"label": "potted plant"}, 0), "Potted Plant")
        self.assertEqual(vision_panel._object_label({"class": "chair"}, 0), "Chair")
        self.assertEqual(vision_panel._object_label({}, 2), "Object 3")


if __name__ == "__main__":
    unittest.main()
