import unittest
from unittest import mock


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

    def test_recent_question_attribution_uses_topic_thread_question(self):
        from intelligence import interaction, topic_thread

        topic_thread.clear()
        try:
            topic_thread.note_assistant_turn("What wore you out?")
            with (
                mock.patch.object(interaction, "_latest_pending_question", return_value=None),
                mock.patch.object(interaction, "_has_unknown_visible_or_recent", return_value=False),
            ):
                person_id, person_name, attributed = (
                    interaction._pending_question_recent_attribution(
                        person_id=None,
                        person_name=None,
                        recent_engagement={
                            "person_id": 1,
                            "name": "Bret Benziger",
                        },
                        raw_best_id=1,
                        speaker_score=0.413,
                        text="I only got 4 hours of sleep",
                    )
                )

            self.assertEqual(person_id, 1)
            self.assertEqual(person_name, "Bret Benziger")
            self.assertTrue(attributed)
        finally:
            topic_thread.clear()

    def test_single_visible_matching_candidate_uses_lower_floor(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._single_visible_engaged_continuity_floor(
                ws_pid=1,
                raw_best_id=1,
            ),
            0.35,
        )
        self.assertEqual(
            interaction._single_visible_engaged_continuity_floor(
                ws_pid=1,
                raw_best_id=2,
            ),
            0.45,
        )


if __name__ == "__main__":
    unittest.main()
