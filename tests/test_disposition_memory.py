import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class DispositionMemoryTests(unittest.TestCase):
    def _create_db(self, path: Path) -> None:
        from setup_assets import DB_SCHEMA

        with sqlite3.connect(path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Bret Benziger')")

    def test_record_expression_samples_builds_smiley_disposition(self):
        from memory import disposition

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "people.db"
            self._create_db(db_path)

            with mock.patch.object(disposition.db, "_DB_FILE", db_path):
                for _ in range(24):
                    disposition.record_expression_sample(
                        1,
                        expression="smile",
                        mood="happy",
                        confidence=0.82,
                    )
                for _ in range(3):
                    disposition.record_expression_sample(
                        1,
                        expression="neutral",
                        mood="neutral",
                        confidence=0.90,
                    )
                stats = disposition.get_stats(1)
                summary = disposition.summarize_for_prompt(1, min_samples=20)

        self.assertIsNotNone(stats)
        self.assertEqual(stats["total_samples"], 27)
        self.assertEqual(stats["dominant_expression"], "smile")
        self.assertEqual(stats["disposition_label"], "smiley")
        self.assertGreater(stats["smile_samples"], stats["neutral_samples"])
        self.assertIn("Facial disposition trend", summary)
        self.assertIn("smiley", summary)

    def test_canonical_expression_maps_mood_aliases(self):
        from memory import disposition

        self.assertEqual(disposition.canonical_expression("happy", None), "smile")
        self.assertEqual(disposition.canonical_expression(None, "sad"), "frown")
        self.assertEqual(disposition.canonical_expression("angry", None), "brow_furrow")
        self.assertEqual(disposition.canonical_expression("surprised", None), "surprise")


class ConsciousnessDispositionTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.c = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_sampled_at = dict(consciousness._disposition_sampled_at)
        self.old_lines = dict(consciousness._last_expression_reaction_line_by_kind)
        consciousness._disposition_sampled_at.clear()
        consciousness._last_expression_reaction_line_by_kind.clear()

    def tearDown(self):
        c = self.c
        c.world_state.update("people", self.old_people)
        c._disposition_sampled_at.clear()
        c._disposition_sampled_at.update(self.old_sampled_at)
        c._last_expression_reaction_line_by_kind.clear()
        c._last_expression_reaction_line_by_kind.update(self.old_lines)

    def _person(self, expression="smile", confidence=0.8):
        return {
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 80, 200, 220),
            "face_expression": {
                "expression": expression,
                "mood": "happy" if expression == "smile" else expression,
                "confidence": confidence,
                "source": "mediapipe_face_landmarker",
                "updated_at": None,
                "blendshapes": {},
            },
        }

    def test_disposition_sampling_records_known_mediapipe_expression(self):
        c = self.c
        c.world_state.update("people", [self._person("smile", 0.81)])
        snapshot = c.world_state.snapshot()

        with mock.patch("memory.disposition.record_expression_sample") as record:
            c._step_disposition_memory(snapshot)

        record.assert_called_once()
        self.assertEqual(record.call_args.args[0], 1)
        self.assertEqual(record.call_args.kwargs["expression"], "smile")
        self.assertEqual(record.call_args.kwargs["confidence"], 0.81)

    def test_disposition_sampling_ignores_non_mediapipe_expression(self):
        c = self.c
        person = self._person("smile", 0.81)
        person["face_expression"]["source"] = "person.expression"
        c.world_state.update("people", [person])

        with mock.patch("memory.disposition.record_expression_sample") as record:
            c._step_disposition_memory(c.world_state.snapshot())

        record.assert_not_called()

    def test_first_sight_disposition_greeting_is_random_and_canned(self):
        c = self.c
        stats = {
            "total_samples": 42,
            "disposition_label": "smiley",
            "confidence": 0.78,
            "last_mentioned_at": None,
        }

        with (
            mock.patch("memory.disposition.get_stats", return_value=stats),
            mock.patch.object(c.config, "FACIAL_DISPOSITION_FIRST_SIGHT_PROBABILITY", 1.0),
            mock.patch.object(c.config, "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_SAMPLES", 20),
            mock.patch.object(c.config, "FACIAL_DISPOSITION_FIRST_SIGHT_MIN_CONFIDENCE", 0.5),
            mock.patch.object(c.random, "choice", side_effect=lambda choices: choices[0]),
        ):
            label, line = c._pick_first_sight_disposition_greeting(1, "Bret")

        self.assertEqual(label, "smiley")
        self.assertIn("Bret", line)
        self.assertIn("smiling", line)


if __name__ == "__main__":
    unittest.main()
