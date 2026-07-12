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


class SelfOcclusionMaskTests(unittest.TestCase):
    """Detections sitting on Rex's own eye stalks (self-occlusion zones) are dropped
    at the source — the field-logged phantom "chair" that was really his face."""

    FRAME = (1080, 1920, 3)   # 1080p capture

    def _records(self, detections, zones):
        from vision import animal_detector as ad
        import config
        with mock.patch.object(config, "CAMERA_SELF_OCCLUSION_ZONES", zones, create=True), \
             mock.patch.object(config, "CAMERA_SELF_OCCLUSION_MAX_OVERLAP", 0.55, create=True):
            return ad._object_records_from_detections(detections, self.FRAME, now=123.0)

    def test_detection_inside_zone_is_suppressed(self):
        # Box fully inside the bottom-right zone (0.60-1.0 x, 0.45-1.0 y of 1920x1080).
        zone_box = _FakeBox(1300, 600, 400, 400)
        recs = self._records(
            [_FakeDetection("chair", 0.90, box=zone_box)],
            zones=[(0.60, 0.45, 1.00, 1.00)],
        )
        self.assertEqual(recs, [])

    def test_detection_outside_zone_is_kept(self):
        center_box = _FakeBox(800, 300, 300, 300)
        recs = self._records(
            [_FakeDetection("chair", 0.90, box=center_box)],
            zones=[(0.60, 0.45, 1.00, 1.00)],
        )
        self.assertEqual([r["label"] for r in recs], ["chair"])

    def test_partial_overlap_below_threshold_is_kept(self):
        # Box straddling the zone edge with well under 55% inside survives.
        straddle = _FakeBox(950, 100, 400, 400)   # only a sliver inside x>=1152
        recs = self._records(
            [_FakeDetection("potted plant", 0.90, box=straddle)],
            zones=[(0.60, 0.45, 1.00, 1.00)],
        )
        self.assertEqual([r["label"] for r in recs], ["potted plant"])

    def test_no_zones_configured_keeps_everything(self):
        recs = self._records(
            [_FakeDetection("chair", 0.90, box=_FakeBox(1300, 600, 400, 400))],
            zones=[],
        )
        self.assertEqual([r["label"] for r in recs], ["chair"])


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


class PersonAdjacentObjectTests(unittest.TestCase):
    """Person-oriented salience (2026-07-08): a small object inside a visible
    person's body zone is tagged near_person, so a cup in someone's hand beats
    the background chair for curiosity. Live-logged failure: Bret held a cup for
    minutes while Rex riffed on a chair."""

    # Face at (900, 200), 120x140 → body zone x∈[720,1200], y∈[200,1040], max_h=350.
    _PEOPLE = [{"face_id": "Bret", "face_box": (900, 200, 120, 140)}]

    def _tag(self, objects, people=None):
        from vision import scene
        return scene.tag_person_adjacent_objects(objects, self._PEOPLE if people is None else people)

    def test_cup_in_hand_is_tagged_with_name(self):
        objs = self._tag([{"label": "cup", "box": (930, 620, 60, 90)}])
        self.assertTrue(objs[0].get("near_person"))
        self.assertEqual(objs[0].get("near_person_name"), "Bret")

    def test_background_object_not_tagged(self):
        objs = self._tag([{"label": "cup", "box": (100, 620, 60, 90)}])
        self.assertFalse(objs[0].get("near_person"))

    def test_furniture_label_never_tagged(self):
        # A chair's box contains the sitter — overlap doesn't mean holding.
        objs = self._tag([{"label": "chair", "box": (930, 620, 60, 90)}])
        self.assertFalse(objs[0].get("near_person"))

    def test_too_big_to_hold_not_tagged(self):
        objs = self._tag([{"label": "backpack", "box": (930, 400, 300, 600)}])
        self.assertFalse(objs[0].get("near_person"))

    def test_no_people_leaves_objects_unchanged(self):
        objs = self._tag([{"label": "cup", "box": (930, 620, 60, 90)}], people=[])
        self.assertFalse(objs[0].get("near_person"))

    def test_kill_switch(self):
        import config
        with mock.patch.object(config, "OBJECT_NEAR_PERSON_ENABLED", False, create=True):
            objs = self._tag([{"label": "cup", "box": (930, 620, 60, 90)}])
        self.assertFalse(objs[0].get("near_person"))

    def test_object_without_box_skipped(self):
        objs = self._tag([{"label": "cup"}])
        self.assertFalse(objs[0].get("near_person"))


class PersonOrientedCuriosityTests(unittest.TestCase):
    """The two curiosity consumers put held objects FIRST with an explicit
    'this beats the furniture' instruction."""

    _OBJECTS = [
        {"label": "chair", "position": "background", "confidence": 0.9},
        {"label": "cup", "position": "center", "confidence": 0.6,
         "near_person": True, "near_person_name": "Bret"},
    ]

    def test_lean_scene_summary_leads_with_held_object(self):
        from intelligence import lean_brain, llm

        with mock.patch.object(llm, "_summarize_world_state", return_value=""):
            summary = lean_brain._scene_summary({"objects": list(self._OBJECTS)})
        self.assertIn("IN THEIR HANDS", summary)
        self.assertIn("cup", summary)
        # The held item leads; the chair is relegated to the generic object list.
        self.assertLess(summary.index("cup"), summary.index("chair"))
        self.assertIn("beats ANY furniture", summary)

    def test_visual_curiosity_line_floats_held_object_first(self):
        from intelligence import consciousness

        with mock.patch.object(consciousness.world_state, "get", return_value=list(self._OBJECTS)):
            line = consciousness._visual_curiosity_objects_line()
        self.assertIn("cup (in their hands)", line)
        self.assertIn("IN Bret's hands", line)
        # Held ordering wins even though the chair has higher detector confidence.
        self.assertLess(line.index("cup"), line.index("chair"))

    def test_visual_curiosity_line_without_held_objects_unchanged(self):
        from intelligence import consciousness

        plain = [{"label": "chair", "position": "background", "confidence": 0.9}]
        with mock.patch.object(consciousness.world_state, "get", return_value=plain):
            line = consciousness._visual_curiosity_objects_line()
        self.assertIn("chair (background)", line)
        self.assertNotIn("hands", line)


if __name__ == "__main__":
    unittest.main()
