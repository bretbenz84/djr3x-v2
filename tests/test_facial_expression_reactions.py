import unittest
from unittest import mock


class FacialExpressionReactionTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.c = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_observed = dict(consciousness._facial_expression_observed)
        self.old_reacted_at = dict(consciousness._facial_expression_reacted_at)
        self.old_global_at = consciousness._last_facial_expression_reaction_at
        self.old_lines = dict(consciousness._last_expression_reaction_line_by_kind)
        with consciousness._smile_reaction_lock:
            self.old_smile_watch = consciousness._smile_reaction_watch
            consciousness._smile_reaction_watch = None
        with consciousness._engaged_lock:
            self.old_engaged_person_id = consciousness._engaged_person_id
            self.old_engaged_last_touch_at = consciousness._engaged_last_touch_at
            self.old_recent_engaged_person_id = consciousness._recent_engaged_person_id
            self.old_recent_engaged_touch_at = consciousness._recent_engaged_touch_at
            consciousness._engaged_person_id = None
            consciousness._engaged_last_touch_at = 0.0
            consciousness._recent_engaged_person_id = None
            consciousness._recent_engaged_touch_at = 0.0
        consciousness._facial_expression_observed.clear()
        consciousness._facial_expression_reacted_at.clear()
        consciousness._last_facial_expression_reaction_at = 0.0
        consciousness._last_expression_reaction_line_by_kind.clear()

    def tearDown(self):
        c = self.c
        c.world_state.update("people", self.old_people)
        c._facial_expression_observed.clear()
        c._facial_expression_observed.update(self.old_observed)
        c._facial_expression_reacted_at.clear()
        c._facial_expression_reacted_at.update(self.old_reacted_at)
        c._last_facial_expression_reaction_at = self.old_global_at
        c._last_expression_reaction_line_by_kind.clear()
        c._last_expression_reaction_line_by_kind.update(self.old_lines)
        with c._smile_reaction_lock:
            c._smile_reaction_watch = self.old_smile_watch
        with c._engaged_lock:
            c._engaged_person_id = self.old_engaged_person_id
            c._engaged_last_touch_at = self.old_engaged_last_touch_at
            c._recent_engaged_person_id = self.old_recent_engaged_person_id
            c._recent_engaged_touch_at = self.old_recent_engaged_touch_at

    def _person(self, expression="neutral", mood=None, confidence=0.9, blendshapes=None):
        return {
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 80, 200, 220),
            "face_expression": {
                "expression": expression,
                "mood": mood or expression,
                "confidence": confidence,
                "source": "mediapipe_face_landmarker",
                "blendshapes": dict(blendshapes or {}),
            },
        }

    def test_surprise_reaction_speaks_a_question(self):
        c = self.c
        c.world_state.update("people", [
            self._person(
                "surprise",
                "surprised",
                0.84,
                {
                    "eyeWideLeft": 0.82,
                    "eyeWideRight": 0.80,
                    "jawOpen": 0.74,
                },
            )
        ])

        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SURPRISE_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 0.0),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())

        speak.assert_called_once()
        kind, text = speak.call_args.args
        self.assertEqual(kind, "surprise")
        self.assertIn("?", text)

    def test_neutral_expression_is_ignored(self):
        c = self.c
        c.world_state.update("people", [self._person("neutral", "neutral", 0.99)])

        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS", 0.0),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())

        speak.assert_not_called()

    def test_brow_furrow_reaction_respects_per_expression_cooldown(self):
        c = self.c
        c.world_state.update("people", [
            self._person(
                "brow_furrow",
                "angry",
                0.88,
                {"browDownLeft": 0.88, "browDownRight": 0.86},
            )
        ])
        c._facial_expression_reacted_at[("db:1", "brow_furrow")] = 160.0

        with (
            mock.patch.object(c.time, "monotonic", return_value=200.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 120.0),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())

        speak.assert_not_called()

    def test_reaction_line_choice_does_not_repeat_immediate_previous(self):
        c = self.c
        lines = c._FACIAL_EXPRESSION_REACTION_LINES["frown"]
        c._last_expression_reaction_line_by_kind["frown"] = lines[0]

        with mock.patch.object(c.random, "choice", side_effect=lambda choices: choices[0]) as choice:
            selected = c._choose_expression_reaction_line("frown", lines)

        self.assertNotEqual(selected, lines[0])
        self.assertNotIn(lines[0], choice.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
