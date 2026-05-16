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

        with mock.patch.object(scene.local_animal_detector, "detect_animals", return_value=local_animals):
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


if __name__ == "__main__":
    unittest.main()
