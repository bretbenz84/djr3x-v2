import unittest
from unittest import mock


class SceneMonitorTests(unittest.TestCase):
    def setUp(self):
        from world_state import world_state

        self.old_animals = world_state.get("animals")
        self.old_crowd = world_state.get("crowd")
        self.old_people = world_state.get("people")
        self.old_environment = world_state.get("environment")

    def tearDown(self):
        from world_state import world_state

        world_state.update("animals", self.old_animals)
        world_state.update("crowd", self.old_crowd)
        world_state.update("people", self.old_people)
        world_state.update("environment", self.old_environment)

    def test_detect_lifeforms_updates_people_and_furry_animals(self):
        from vision import scene
        from world_state import world_state

        raw = (
            '{"people_count": 1, "animals": ['
            '{"species": "dog", "position": "lower right", '
            '"furred": true, "confidence": "high"}]}'
        )

        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.detect_lifeforms(object())

        self.assertEqual(result["people_count"], 1)
        self.assertEqual(world_state.get("crowd")["count"], 1)
        animals = world_state.get("animals")
        self.assertEqual(len(animals), 1)
        self.assertEqual(animals[0]["species"], "dog")
        self.assertTrue(animals[0]["furred"])

    def test_detect_lifeforms_ignores_low_confidence_animals(self):
        from vision import scene
        from world_state import world_state

        raw = (
            '{"people_count": 1, "animals": ['
            '{"species": "cat", "position": "background", '
            '"furred": true, "confidence": "low"}]}'
        )

        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.detect_lifeforms(object())

        self.assertEqual(result["animals"], [])
        self.assertEqual(world_state.get("animals"), [])

    def test_local_animal_detection_updates_world_state_without_openai(self):
        from vision import scene
        from world_state import world_state

        local_animals = [{
            "id": "animal_1",
            "species": "dog",
            "position": "foreground right",
            "last_seen": 123.0,
            "confidence": 0.88,
            "furred": True,
            "source": "mediapipe_object_detector",
        }]

        # Persistence debounce: a species must be seen ANIMAL_ARRIVAL_CONFIRM_SCANS
        # consecutive scans before it's confirmed into world_state (so a flickering
        # lamp-as-bird can't fire an arrival). Drive enough scans to confirm.
        scene._animal_confirm_streak.clear()
        need = int(getattr(scene.config, "ANIMAL_ARRIVAL_CONFIRM_SCANS", 1))
        with mock.patch.object(scene.local_animal_detector, "detect_animals", return_value=local_animals):
            for _ in range(need):
                animals = scene.detect_animals_local(object())

        self.assertEqual(animals, local_animals)
        self.assertEqual(world_state.get("animals"), local_animals)

    def test_local_animal_detection_preserves_state_when_model_unavailable(self):
        from vision import scene
        from world_state import world_state

        existing = [{
            "id": "animal_1",
            "species": "cat",
            "position": "center",
            "last_seen": 123.0,
        }]
        world_state.update("animals", existing)

        with mock.patch.object(scene.local_animal_detector, "detect_animals", return_value=None):
            animals = scene.detect_animals_local(object())

        self.assertEqual(animals, existing)
        self.assertEqual(world_state.get("animals"), existing)

    def test_environment_grounded_by_local_visible_people(self):
        from vision import scene
        from world_state import world_state

        world_state.update("people", [
            {"id": "person_1", "person_db_id": 1, "face_visible": True, "face_box": [1, 2, 3, 4]},
            {"id": "person_2", "person_db_id": None, "face_visible": True, "face_box": [5, 6, 7, 8]},
        ])
        raw = (
            '{"scene_type":"home","indoor_outdoor":"indoor","lighting":"dim",'
            '"crowd_density":"empty","time_of_day":"night","description":"A quiet room."}'
        )

        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.analyze_environment(object(), force=True)

        self.assertEqual(result["crowd_density"], "sparse")
        self.assertEqual(result["local_people_count"], 2)
        self.assertEqual(world_state.get("crowd")["count"], 2)

    def test_lifeform_scan_preserves_recent_local_animals_when_cloud_misses(self):
        import time
        from vision import scene
        from world_state import world_state

        existing = [{
            "id": "animal_1",
            "species": "dog",
            "position": "foreground",
            "last_seen": time.time(),
        }]
        world_state.update("animals", existing)

        with mock.patch.object(scene, "_call_gpt4o", return_value='{"people_count":0,"animals":[]}'):
            result = scene.detect_lifeforms(object())

        self.assertEqual(result["animals"], existing)
        self.assertEqual(world_state.get("animals"), existing)

    def test_locate_people_reads_presence_and_vertical(self):
        from vision import scene

        raw = '{"present": true, "count": 1, "vertical": "low", "posture": "seated", "confidence": "high"}'
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.locate_people(object())

        self.assertTrue(result["present"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["vertical"], "low")
        self.assertEqual(result["posture"], "seated")
        self.assertEqual(result["confidence"], "high")

    def test_locate_people_normalizes_unknown_values(self):
        from vision import scene

        raw = '{"present": true, "count": 99, "vertical": "floor", "posture": "dancing", "confidence": "certain"}'
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.locate_people(object())

        self.assertEqual(result["count"], 5)            # capped
        self.assertEqual(result["vertical"], "center")  # unknown → center
        self.assertEqual(result["posture"], "unknown")  # unknown → unknown
        self.assertEqual(result["confidence"], "low")   # unknown → low

    def test_locate_people_safe_fallback_on_no_response(self):
        from vision import scene

        # None frame and a None model response (e.g. missing API key) both degrade
        # to "nobody, low confidence" — never a false positive.
        self.assertFalse(scene.locate_people(None)["present"])
        with mock.patch.object(scene, "_call_gpt4o", return_value=None):
            result = scene.locate_people(object())
        self.assertFalse(result["present"])
        self.assertEqual(result["confidence"], "low")

    def test_locate_people_present_when_count_positive(self):
        from vision import scene

        # A count > 0 implies presence even if the model omits/false the flag.
        raw = '{"present": false, "count": 2, "vertical": "center", "posture": "standing", "confidence": "medium"}'
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            result = scene.locate_people(object())
        self.assertTrue(result["present"])


if __name__ == "__main__":
    unittest.main()
