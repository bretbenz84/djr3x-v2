import unittest
from unittest import mock


class FaceExpressionTelemetryTests(unittest.TestCase):
    def tearDown(self):
        from world_state import world_state

        world_state.update("people", [])

    def test_smile_blendshapes_classify_as_happy(self):
        from vision import face_expression

        with mock.patch.object(face_expression.config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.35):
            result = face_expression._classify_expression({
                "mouthSmileLeft": 0.72,
                "mouthSmileRight": 0.68,
            })

        self.assertEqual(result["mood"], "happy")
        self.assertEqual(result["expression"], "smile")
        self.assertGreaterEqual(result["confidence"], 0.68)

    def test_frown_blendshapes_classify_as_sad(self):
        from vision import face_expression

        with mock.patch.object(face_expression.config, "FACE_EXPRESSION_FROWN_THRESHOLD", 0.35):
            result = face_expression._classify_expression({
                "mouthFrownLeft": 0.61,
                "mouthFrownRight": 0.57,
            })

        self.assertEqual(result["mood"], "sad")
        self.assertEqual(result["expression"], "frown")
        self.assertIn("mouth", result["notes"])

    def test_low_expression_scores_classify_as_neutral(self):
        from vision import face_expression

        result = face_expression._classify_expression({
            "mouthSmileLeft": 0.05,
            "mouthSmileRight": 0.04,
            "mouthFrownLeft": 0.03,
            "mouthFrownRight": 0.02,
        })

        self.assertEqual(result["mood"], "neutral")
        self.assertEqual(result["expression"], "neutral")

    def test_merge_expression_updates_existing_visible_face_slot(self):
        from vision import face_expression
        from world_state import world_state

        world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 80, 200, 220),
            "expression": "neutral",
        }])

        changed = face_expression.merge_expressions_into_world_state([{
            "mood": "happy",
            "expression": "smile",
            "confidence": 0.82,
            "notes": "smiling",
            "face_box": (105, 85, 190, 210),
            "blendshapes": {
                "mouthSmileLeft": 0.84,
                "mouthSmileRight": 0.80,
            },
        }])

        self.assertEqual(changed, 1)
        person = world_state.get("people")[0]
        self.assertEqual(person["expression"], "happy")
        self.assertEqual(person["face_mood"]["mood"], "happy")
        self.assertEqual(person["face_mood"]["source"], "mediapipe_face_landmarker")
        self.assertEqual(person["face_expression"]["expression"], "smile")
        self.assertIn("mouthSmileLeft", person["face_expression"]["blendshapes"])

    def test_merge_expression_does_not_create_identity_slots(self):
        from vision import face_expression
        from world_state import world_state

        world_state.update("people", [])

        changed = face_expression.merge_expressions_into_world_state([{
            "mood": "happy",
            "expression": "smile",
            "confidence": 0.75,
            "face_box": (0, 0, 100, 100),
        }])

        self.assertEqual(changed, 0)
        self.assertEqual(world_state.get("people"), [])


if __name__ == "__main__":
    unittest.main()
