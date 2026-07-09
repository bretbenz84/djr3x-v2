"""Lean-owned visual riffs stay grounded, consent-safe, and non-competing."""

import unittest
from types import SimpleNamespace as NS
from unittest import mock

from intelligence import consciousness, interaction, lean_brain


def _one_chunk_stream(text):
    return [NS(choices=[NS(delta=NS(content=text))])]


class LeanVisualRiffCueTest(unittest.TestCase):
    def _adult_facts(self):
        return [{"key": "age_category", "value": "adult"}]

    def test_known_adult_with_safe_accessory_gets_a_familiar_detail_cue(self):
        world = {"people": [{"person_db_id": 7, "pose": "standing"}]}
        with (
            mock.patch.object(interaction.random, "random", return_value=0.0),
            mock.patch.object(interaction.people_memory, "get_person", return_value={}),
            mock.patch.object(interaction.facts_memory, "get_facts", return_value=self._adult_facts()),
            mock.patch.object(interaction.boundary_memory, "is_blocked", return_value=False),
            mock.patch.object(interaction.consciousness, "_pick_appearance_hint", return_value="a familiar hat"),
        ):
            cue = interaction._lean_visual_riff_cue(7, world)
        self.assertEqual(cue, {"cue": "a familiar visual detail: a familiar hat"})

    def test_no_cue_for_minor_or_boundary(self):
        world = {"people": [{"person_db_id": 7, "age_category": "teen", "pose": "standing"}]}
        with (
            mock.patch.object(interaction.random, "random", return_value=0.0),
            mock.patch.object(interaction.people_memory, "get_person", return_value={}),
            mock.patch.object(interaction.facts_memory, "get_facts", return_value=self._adult_facts()),
        ):
            self.assertIsNone(interaction._lean_visual_riff_cue(7, world))

        world = {"people": [{"person_db_id": 7, "pose": "standing"}]}
        with (
            mock.patch.object(interaction.random, "random", return_value=0.0),
            mock.patch.object(interaction.people_memory, "get_person", return_value={}),
            mock.patch.object(interaction.facts_memory, "get_facts", return_value=self._adult_facts()),
            mock.patch.object(interaction.boundary_memory, "is_blocked", return_value=True),
        ):
            self.assertIsNone(interaction._lean_visual_riff_cue(7, world))

    def test_lean_instruction_requires_one_safe_non_question_line(self):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("That hat is still doing heroic work, Bret.")

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(
                person_id=None,
                transcript=[],
                visual_riff={"cue": "a familiar visual detail: a familiar hat"},
            )
        self.assertEqual(line, "That hat is still doing heroic work, Bret.")
        self.assertIn("not a question", captured[0])
        self.assertIn("Never mention or joke about body, age", captured[0])
        self.assertNotIn("fresh angles", captured[0])

    def test_shared_appearance_hint_never_uses_build(self):
        captured = []

        def pick(options):
            captured.extend(options)
            return options[0] if options else None

        rows = [
            {"key": "build", "value": "athletic"},
            {"key": "hair_color", "value": "brown"},
            {"key": "notable_features", "value": '["glasses", "tattoo"]'},
        ]
        with (
            mock.patch("memory.facts.get_facts_by_category", return_value=rows),
            mock.patch.object(consciousness.random, "choice", side_effect=pick),
        ):
            consciousness._pick_appearance_hint(7)
        self.assertIn("brown hair", captured)
        self.assertIn("a familiar glasses", captured)
        self.assertFalse(any("athletic" in option or "tattoo" in option for option in captured))


if __name__ == "__main__":
    unittest.main()
