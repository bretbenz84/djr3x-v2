"""
GPT-4o vision descriptions should fold in dlib face-recognition identity: a
recognized visible person is named ("Bret at his desk") instead of anonymized
("a man at a desk"). This covers both the periodic environment description shown
in the GUI (analyze_environment) and the "what do you see?" path
(analyze_directed_attention), plus the shared name resolver.
"""

from __future__ import annotations

import unittest
from unittest import mock

from vision import face, scene


class VisibleKnownNamesTest(unittest.TestCase):
    def test_resolves_visible_person_db_id_to_name(self):
        snap = {"people": [
            {"person_db_id": 1, "face_visible": True},
            {"person_db_id": None, "face_visible": True},   # unknown → skipped
            {"person_db_id": 2, "face_missing": True},       # gone → skipped
            {"person_db_id": 1, "face_visible": True},        # dup id → de-duped
        ]}
        with mock.patch("memory.people.get_person",
                        side_effect=lambda pid: {"name": "Bret"} if pid == 1 else None):
            self.assertEqual(face.visible_known_names(snap), ["Bret"])

    def test_no_recognized_people_returns_empty(self):
        snap = {"people": [{"person_db_id": None, "face_visible": True}]}
        self.assertEqual(face.visible_known_names(snap), [])


class AnalyzeEnvironmentNamesTest(unittest.TestCase):
    def _prompt_for(self, known_names):
        captured = {}

        def _fake(frame, prompt, key, **kw):
            captured["prompt"] = prompt
            return ('{"scene_type":"office","indoor_outdoor":"indoor",'
                    '"lighting":"bright","crowd_density":"sparse",'
                    '"time_of_day":"night","description":"Bret is at a desk."}')

        with mock.patch.object(scene, "_call_gpt4o", side_effect=_fake):
            result = scene.analyze_environment(object(), force=True, known_names=known_names)
        return captured.get("prompt", ""), result

    def test_known_name_injected_into_prompt(self):
        prompt, result = self._prompt_for(["Bret"])
        self.assertIn("KNOWN to you by name: Bret", prompt)
        self.assertIn('instead of "a man"', prompt)
        self.assertEqual(result["description"], "Bret is at a desk.")

    def test_empty_names_keeps_generic_prompt(self):
        prompt, _ = self._prompt_for([])
        self.assertNotIn("KNOWN to you by name", prompt)

    def test_none_auto_resolves_from_world_state(self):
        captured = {}

        def _fake(frame, prompt, key, **kw):
            captured["prompt"] = prompt
            return ('{"scene_type":"home","indoor_outdoor":"indoor","lighting":"dim",'
                    '"crowd_density":"sparse","time_of_day":"night","description":"x"}')

        with mock.patch.object(face, "visible_known_names", return_value=["Jasmine"]), \
             mock.patch.object(scene, "_call_gpt4o", side_effect=_fake):
            scene.analyze_environment(object(), force=True)   # known_names=None
        self.assertIn("KNOWN to you by name: Jasmine", captured["prompt"])


class AnalyzeDirectedAttentionNamesTest(unittest.TestCase):
    def _prompt_for(self, known_names):
        captured = {}

        def _fake(frame, prompt, key, **kw):
            captured["prompt"] = prompt
            return ('{"target_summary":"Bret at his desk","target_visible":true,'
                    '"subject_type":"person","visible_people_count":1,"animals":[],'
                    '"notable_details":[],"roast_angle":"","confidence":"high"}')

        with mock.patch.object(scene, "_call_gpt4o", side_effect=_fake):
            scene.analyze_directed_attention(
                object(), direction="current", utterance="what do you see",
                known_names=known_names,
            )
        return captured.get("prompt", "")

    def test_recognized_people_may_be_named(self):
        prompt = self._prompt_for(["Bret"])
        self.assertIn("You MAY name these specific people", prompt)
        self.assertIn("Bret", prompt)
        # The blanket "do not identify anyone" ban is lifted when we know who it is.
        self.assertNotIn("Do not identify anyone.", prompt)

    def test_no_known_people_keeps_identity_ban(self):
        prompt = self._prompt_for([])
        self.assertIn("Do not identify anyone.", prompt)
        self.assertNotIn("You MAY name", prompt)


if __name__ == "__main__":
    unittest.main()
