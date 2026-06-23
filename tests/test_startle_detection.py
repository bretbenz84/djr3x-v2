"""
#29 — in continuous (local-detection) mode the local MediaPipe detector only knows
bird/cat/dog/horse, so the startle species (snake/spider/wasp) were never seen and the
startle reaction could never fire. The periodic OpenAI startle scan must ADD startle
species into world_state.animals without clobbering the local detections or the crowd.
"""

import json
import unittest
from unittest import mock

from vision import scene
from world_state import world_state


class StartleScanTest(unittest.TestCase):
    def setUp(self):
        world_state.update("animals", [])

    def tearDown(self):
        world_state.update("animals", [])

    def test_merges_startle_species_additively(self):
        # A locally-detected dog is already tracked; the OpenAI scan reports a snake + dog.
        world_state.update("animals", [{"species": "dog", "position": "left", "id": "animal_1"}])
        raw = json.dumps({"animals": [
            {"species": "snake", "position": "floor right", "furred": False, "confidence": "high"},
            {"species": "dog", "position": "left", "furred": True, "confidence": "high"},
        ]})
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            added = scene._scan_for_startle_species(object())

        # Only the snake (a startle species) is newly added; the local dog is preserved.
        self.assertEqual([a["species"] for a in added], ["snake"])
        self.assertEqual({a["species"] for a in world_state.get("animals")}, {"dog", "snake"})

    def test_non_startle_species_are_ignored(self):
        world_state.update("animals", [{"species": "cat", "position": "couch"}])
        raw = json.dumps({"animals": [{"species": "dog", "position": "door", "confidence": "high"}]})
        with mock.patch.object(scene, "_call_gpt4o", return_value=raw):
            added = scene._scan_for_startle_species(object())

        self.assertEqual(added, [])
        self.assertEqual([a["species"] for a in world_state.get("animals")], ["cat"])

    def test_none_frame_returns_empty(self):
        self.assertEqual(scene._scan_for_startle_species(None), [])

    def test_startle_species_recognized_by_orchestrator(self):
        # The merged species must actually trip the startle classifier the reaction uses.
        from intelligence import emotion_orchestrator
        for sp in ("snake", "spider", "wasp"):
            self.assertTrue(emotion_orchestrator.is_startling_animal(sp), sp)
        for sp in ("dog", "cat", "bird"):
            self.assertFalse(emotion_orchestrator.is_startling_animal(sp), sp)


if __name__ == "__main__":
    unittest.main()
