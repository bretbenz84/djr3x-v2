import unittest


class InteractionPersonResolutionTests(unittest.TestCase):
    def test_single_visible_face_overrides_conflicting_voice_id(self):
        from intelligence import interaction

        override = interaction._single_visible_face_voice_override(
            resolved_person_id=2,
            ws_person={
                "person_db_id": 1,
                "face_id": "Bret Penziger",
            },
            visible_known_by_id={
                1: {
                    "person_db_id": 1,
                    "face_id": "Bret Penziger",
                },
            },
            has_unknown_visible_or_recent=False,
        )

        self.assertEqual(override, (1, "Bret Penziger"))

    def test_single_visible_face_does_not_override_when_unknown_visible(self):
        from intelligence import interaction

        override = interaction._single_visible_face_voice_override(
            resolved_person_id=2,
            ws_person={
                "person_db_id": 1,
                "face_id": "Bret Penziger",
            },
            visible_known_by_id={
                1: {
                    "person_db_id": 1,
                    "face_id": "Bret Penziger",
                },
            },
            has_unknown_visible_or_recent=True,
        )

        self.assertIsNone(override)

    def test_multiple_visible_known_faces_do_not_override_voice_id(self):
        from intelligence import interaction

        override = interaction._single_visible_face_voice_override(
            resolved_person_id=2,
            ws_person={
                "person_db_id": 1,
                "face_id": "Bret Penziger",
            },
            visible_known_by_id={
                1: {
                    "person_db_id": 1,
                    "face_id": "Bret Penziger",
                },
                3: {
                    "person_db_id": 3,
                    "face_id": "Gloria Carter",
                },
            },
            has_unknown_visible_or_recent=False,
        )

        self.assertIsNone(override)


if __name__ == "__main__":
    unittest.main()
