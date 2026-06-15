"""
First-person scene memories should record WHO was there, not "a man at a desk". The
startup caption and scene-change episodes now resolve recognized people and name them.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import episodic_hooks
from vision import scene


class QuickCaptionNamesPeopleTest(unittest.TestCase):
    def _prompt_for(self, known_people):
        captured = {}

        def _fake(frame, prompt, profile, max_tokens=120):
            captured["prompt"] = prompt
            return "a person at a desk"

        with mock.patch.object(scene, "_call_gpt4o", side_effect=_fake):
            scene.quick_caption(object(), known_people=known_people)
        return captured.get("prompt", "")

    def test_known_people_are_injected_by_name(self):
        p = self._prompt_for(["Bret Benziger"])
        self.assertIn("Bret Benziger", p)
        self.assertIn("BY NAME", p)
        self.assertIn("never as 'a man'", p)

    def test_no_known_people_keeps_generic_prompt(self):
        p = self._prompt_for([])
        self.assertNotIn("BY NAME", p)


class KnownVisibleNamesTest(unittest.TestCase):
    def test_resolves_person_db_id_to_name_from_snapshot(self):
        snapshot = {"people": [
            {"person_db_id": 1, "face_visible": True},
            {"person_db_id": None, "face_visible": True},          # unknown → skipped
            {"person_db_id": 2, "face_missing": True},             # not visible → skipped
        ]}
        with mock.patch.object(
            episodic_hooks, "_known_visible_names",
            wraps=episodic_hooks._known_visible_names,
        ), mock.patch("memory.people.get_person",
                      side_effect=lambda pid: {"name": "Bret Benziger"} if pid == 1 else None):
            names = episodic_hooks._known_visible_names(snapshot)
        self.assertEqual(names, ["Bret Benziger"])

    def test_join_names(self):
        self.assertEqual(episodic_hooks._join_names(["A"]), "A")
        self.assertEqual(episodic_hooks._join_names(["A", "B"]), "A and B")
        self.assertEqual(episodic_hooks._join_names(["A", "B", "C"]), "A, B, and C")


class SceneChangedNamesWhoTest(unittest.TestCase):
    def test_scene_episode_names_the_person_present(self):
        episodic_hooks._last_scene_episode_sig = None
        snapshot = {
            "environment": {"scene_type": "workshop", "lighting": "dim",
                            "crowd_density": "sparse", "description": "a cluttered workshop"},
            "people": [{"person_db_id": 1, "face_visible": True}],
        }
        recorded = {}
        with mock.patch("memory.people.get_person", return_value={"name": "Bret"}), \
             mock.patch("memory.episodes.record_scene",
                        side_effect=lambda summary, detail=None: recorded.update(summary=summary)):
            episodic_hooks.scene_changed(snapshot)
        self.assertIn("cluttered workshop", recorded["summary"])
        self.assertIn("Bret was there", recorded["summary"])

    def test_no_known_people_leaves_summary_generic(self):
        episodic_hooks._last_scene_episode_sig = None
        snapshot = {
            "environment": {"description": "an empty room"},
            "people": [{"person_db_id": None, "face_visible": True}],
        }
        recorded = {}
        with mock.patch("memory.episodes.record_scene",
                        side_effect=lambda summary, detail=None: recorded.update(summary=summary)):
            episodic_hooks.scene_changed(snapshot)
        self.assertIn("empty room", recorded["summary"])
        self.assertNotIn("was there", recorded["summary"])


if __name__ == "__main__":
    unittest.main()
