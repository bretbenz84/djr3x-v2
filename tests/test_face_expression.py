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

    def test_brow_furrow_classifies_as_focused_not_angry(self):
        from vision import face_expression

        with mock.patch.object(face_expression.config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45):
            result = face_expression._classify_expression({
                "browDownLeft": 0.74,
                "browDownRight": 0.70,
            })

        self.assertEqual(result["mood"], "focused")
        self.assertEqual(result["expression"], "brow_furrow")

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


class AdaptiveBrowBaselineTests(unittest.TestCase):
    """Brow-furrow fires on a rise above the person's RESTING brow, not an absolute
    value — so a high-neutral face (MediaPipe over-reads browDown for some face/camera
    geometries) stops being tagged 'furrowing' every frame, while a low-neutral face
    keeps its original sensitivity. Floored at the absolute threshold, so it can only
    reduce false positives, never add them."""

    BOX = (100, 80, 200, 220)

    def setUp(self):
        from vision import face_expression
        self.fe = face_expression
        face_expression.reset_brow_baselines()

    def tearDown(self):
        self.fe.reset_brow_baselines()

    def _warm(self, value, n=8):
        baseline = None
        for _ in range(n):
            baseline = self.fe._brow_furrow_baseline(self.BOX, value, 1000.0)
        return baseline

    def test_high_neutral_brow_would_false_furrow_without_baseline(self):
        # Control: the absolute threshold (warmup / disabled) tags a high-neutral
        # browDown as furrowing — this is the logged misfire.
        fe = self.fe
        with mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45):
            result = fe._classify_expression({"browDownLeft": 0.87, "browDownRight": 0.85})
        self.assertEqual(result["expression"], "brow_furrow")

    def test_high_neutral_brow_suppressed_after_warmup(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_ADAPTIVE_BASELINE_ENABLED", True),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA", 0.18),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45),
        ):
            baseline = self._warm(0.86)
            self.assertIsNotNone(baseline)
            result = fe._classify_expression(
                {"browDownLeft": 0.87, "browDownRight": 0.85}, brow_baseline=baseline
            )
        self.assertEqual(result["expression"], "neutral")

    def test_genuine_furrow_above_relaxed_baseline_still_detected(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA", 0.18),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45),
        ):
            self._warm(0.50)  # relaxed floor settles low
            baseline = fe._brow_furrow_baseline(self.BOX, 0.81, 1000.0)  # the furrow frame
            result = fe._classify_expression(
                {"browDownLeft": 0.82, "browDownRight": 0.80}, brow_baseline=baseline
            )
        self.assertEqual(result["expression"], "brow_furrow")

    def test_low_neutral_face_keeps_original_sensitivity(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_BASELINE_DELTA", 0.18),
            mock.patch.object(fe.config, "FACE_EXPRESSION_BROW_FURROW_THRESHOLD", 0.45),
        ):
            baseline = self._warm(0.05)
            # baseline ~0.05 → effective threshold floored at 0.45, so a 0.50 furrow fires.
            result = fe._classify_expression(
                {"browDownLeft": 0.52, "browDownRight": 0.50}, brow_baseline=baseline
            )
        self.assertEqual(result["expression"], "brow_furrow")

    def test_disabled_baseline_returns_none(self):
        fe = self.fe
        with mock.patch.object(
            fe.config, "FACE_EXPRESSION_BROW_ADAPTIVE_BASELINE_ENABLED", False
        ):
            self.assertIsNone(fe._brow_furrow_baseline(self.BOX, 0.9, 1000.0))

    def test_unknown_face_box_uses_absolute_threshold(self):
        fe = self.fe
        # No box → no track → None baseline → absolute threshold path.
        self.assertIsNone(fe._brow_furrow_baseline(None, 0.9, 1000.0))


class AdaptiveSmileBaselineTests(unittest.TestCase):
    """Smile fires on a rise above the person's RESTING mouth, not an absolute value — so a
    face MediaPipe over-reads as faintly smiling at rest (a robot camera angled up at a seated
    talker) stops being tagged 'happy' every frame and leaking 'looks amused / smiling' into
    Rex's prompt. Floored at the absolute threshold, so it can only reduce false positives."""

    BOX = (100, 80, 200, 220)

    def setUp(self):
        from vision import face_expression
        self.fe = face_expression
        face_expression.reset_smile_baselines()

    def tearDown(self):
        self.fe.reset_smile_baselines()

    def _warm(self, value, n=8):
        baseline = None
        for _ in range(n):
            baseline = self.fe._smile_baseline(self.BOX, value, 1000.0)
        return baseline

    def test_high_neutral_smile_would_false_happy_without_baseline(self):
        # Control: absolute threshold (warmup / disabled) tags a high-neutral resting mouth as
        # smiling — this is the "carbon-based approval / there it is, a smile" misfire.
        fe = self.fe
        with mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.50):
            result = fe._classify_expression({"mouthSmileLeft": 0.56, "mouthSmileRight": 0.54})
        self.assertEqual(result["expression"], "smile")

    def test_high_neutral_smile_suppressed_after_warmup(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_ADAPTIVE_BASELINE_ENABLED", True),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_DELTA", 0.22),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.50),
        ):
            baseline = self._warm(0.55)
            self.assertIsNotNone(baseline)
            result = fe._classify_expression(
                {"mouthSmileLeft": 0.56, "mouthSmileRight": 0.54}, smile_baseline=baseline
            )
        self.assertEqual(result["expression"], "neutral")

    def test_genuine_smile_above_relaxed_baseline_still_detected(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_DELTA", 0.22),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.50),
        ):
            self._warm(0.10)  # resting mouth settles low
            baseline = fe._smile_baseline(self.BOX, 0.82, 1000.0)  # the smile frame
            result = fe._classify_expression(
                {"mouthSmileLeft": 0.83, "mouthSmileRight": 0.81}, smile_baseline=baseline
            )
        self.assertEqual(result["expression"], "smile")

    def test_low_neutral_face_keeps_original_sensitivity(self):
        fe = self.fe
        with (
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_WARMUP_SAMPLES", 5),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_BASELINE_DELTA", 0.22),
            mock.patch.object(fe.config, "FACE_EXPRESSION_SMILE_THRESHOLD", 0.50),
        ):
            baseline = self._warm(0.05)
            # baseline ~0.05 → effective threshold floored at 0.50, so a 0.55 smile fires.
            result = fe._classify_expression(
                {"mouthSmileLeft": 0.56, "mouthSmileRight": 0.54}, smile_baseline=baseline
            )
        self.assertEqual(result["expression"], "smile")

    def test_disabled_baseline_returns_none(self):
        fe = self.fe
        with mock.patch.object(
            fe.config, "FACE_EXPRESSION_SMILE_ADAPTIVE_BASELINE_ENABLED", False
        ):
            self.assertIsNone(fe._smile_baseline(self.BOX, 0.9, 1000.0))

    def test_unknown_face_box_uses_absolute_threshold(self):
        fe = self.fe
        self.assertIsNone(fe._smile_baseline(None, 0.9, 1000.0))


if __name__ == "__main__":
    unittest.main()
