import unittest
from unittest import mock


class SceneMonitorTests(unittest.TestCase):
    def setUp(self):
        from world_state import world_state

        self.old_animals = world_state.get("animals")
        self.old_crowd = world_state.get("crowd")

    def tearDown(self):
        from world_state import world_state

        world_state.update("animals", self.old_animals)
        world_state.update("crowd", self.old_crowd)

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


if __name__ == "__main__":
    unittest.main()
