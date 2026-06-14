"""
Tier 2 / item 6 — identity & memory hygiene. A joking child renamed himself
Wade->Bro->Broski (each obeyed instantly) and a full appearance dossier + junk facts
were stored on a 10-12yo. Guards: reject non-name renames, cooldown rapid renames,
and don't persist an appearance dossier for a minor.
"""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

import numpy as np

from intelligence import interaction as I


class NonNameAndCooldownTest(unittest.TestCase):
    def setUp(self):
        I._canonical_rename_at.clear()

    def test_looks_like_non_name(self):
        for n in ("bro", "broski", "Bro Broski", "dude", "yo"):
            self.assertTrue(I._looks_like_non_name(n), n)
        for n in ("Wade", "Bret Benziger", "Robert", "O'Brien"):
            self.assertFalse(I._looks_like_non_name(n), n)

    def test_rename_recently_window(self):
        self.assertFalse(I._rename_recently(4))
        I._canonical_rename_at[4] = time.monotonic()
        self.assertTrue(I._rename_recently(4))
        I._canonical_rename_at[4] = time.monotonic() - 10_000
        self.assertFalse(I._rename_recently(4))


class RenameHandlerGateTest(unittest.TestCase):
    def setUp(self):
        I._canonical_rename_at.clear()

    def test_non_name_rename_rejected(self):
        with mock.patch.object(I, "_extract_name_update", return_value="Bro"), \
             mock.patch.object(I, "_speak_blocking"):
            r = I._handle_name_update_request("call me bro", 4, "Wade")
        self.assertIsNotNone(r)
        self.assertIn("name field", r)

    def test_cooldown_blocks_rapid_second_rename(self):
        I._canonical_rename_at[4] = time.monotonic()
        with mock.patch.object(I, "_extract_name_update", return_value="Robert"), \
             mock.patch.object(I.people_memory, "find_person_by_name", return_value=None), \
             mock.patch.object(I, "_resolve_name_update_target", return_value=(4, "Wade")), \
             mock.patch.object(I, "_speak_blocking"):
            r = I._handle_name_update_request("my name is Robert", 4, "Wade")
        self.assertIn("just changed your name", r)

    def test_valid_rename_passes_and_records_cooldown(self):
        with mock.patch.object(I, "_extract_name_update", return_value="Robert"), \
             mock.patch.object(I.people_memory, "find_person_by_name", return_value=None), \
             mock.patch.object(I, "_resolve_name_update_target", return_value=(4, "Wade")), \
             mock.patch.object(I.people_memory, "rename_person", return_value=True), \
             mock.patch.object(I, "_refresh_world_state_person_name"), \
             mock.patch.object(I, "_identity_enrollment_ack", return_value="ok"), \
             mock.patch.object(I.person_specials, "special_intro_ack", return_value=None), \
             mock.patch.object(I.repair_moves, "add_better_luck_line", return_value="Got it."), \
             mock.patch.object(I, "_speak_blocking"):
            I._handle_name_update_request("my name is Robert", 4, "Wade")
        self.assertIn(4, I._canonical_rename_at)


class AppearanceDossierMinorTest(unittest.TestCase):
    def _run(self, age_category):
        from vision import face
        payload = json.dumps({
            "age_category": age_category, "age_range": "10-12", "build": "slim",
            "hair_color": "brown", "hair_style": "short", "height_estimate": "average",
            "notable_features": [],
        })
        choice = mock.Mock(); choice.message = mock.Mock(content=payload)
        resp = mock.Mock(); resp.choices = [choice]
        client = mock.Mock(); client.chat.completions.create.return_value = resp
        stored = []
        with mock.patch.object(face, "encode_jpeg_base64", return_value="x"), \
             mock.patch("openai.OpenAI", return_value=client), \
             mock.patch.object(face.facts, "add_fact",
                               side_effect=lambda **k: stored.append(k["key"])):
            face.update_appearance(4, np.zeros((8, 8, 3), dtype=np.uint8))
        return set(stored)

    def test_child_stores_only_age(self):
        self.assertEqual(self._run("child"), {"age_category", "age_range"})

    def test_adult_stores_full_dossier(self):
        self.assertIn("build", self._run("adult"))
        self.assertIn("hair_color", self._run("adult"))


if __name__ == "__main__":
    unittest.main()
