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
        # The reaction-mechanics tests assert on the authored bank (e.g. a surprise line
        # ends with "?"); keep the conversation-aware LLM path out of them so they're
        # deterministic and never hit the network. test_contextual_reaction_is_preferred
        # exercises the LLM branch explicitly.
        self._ctx_patch = mock.patch.object(
            consciousness, "_generate_contextual_expression_reaction", return_value=""
        )
        self._ctx_patch.start()

    def tearDown(self):
        c = self.c
        self._ctx_patch.stop()
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

    def test_contextual_reaction_is_preferred_over_bank(self):
        # When the conversation-aware LLM path returns a line, it wins over the authored
        # bank — that's how a surprised face gets read in context.
        c = self.c
        c.world_state.update("people", [
            self._person(
                "surprise", "surprised", 0.84,
                {"eyeWideLeft": 0.82, "eyeWideRight": 0.80, "jawOpen": 0.74},
            )
        ])
        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SURPRISE_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 0.0),
            mock.patch.object(
                c, "_generate_contextual_expression_reaction",
                return_value="Whoa — what did I miss?",
            ) as gen,
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())
        gen.assert_called_once()
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[1], "Whoa — what did I miss?")

    def test_habitual_disposition_suppresses_reaction(self):
        # Bret reads as habitually brow-furrowed/intense (logged: 60 samples, 85%
        # brow-furrow). His RESTING face must not trigger "you're not exactly sold on
        # this, are you?" — that mistakes a visual habit for a live emotional signal.
        c = self.c
        c.world_state.update("people", [
            self._person(
                "brow_furrow", "focused", 0.88,
                {"browDownLeft": 0.90, "browDownRight": 0.86},
            )
        ])
        stats = {"total_samples": 60, "dominant_expression": "brow_furrow", "confidence": 0.86}
        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 0.0),
            mock.patch("memory.disposition.get_stats", return_value=stats),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())
        speak.assert_not_called()

    def test_non_habitual_brow_furrow_still_reacts(self):
        # Same expression, but it is NOT this person's dominant resting face → the
        # reaction fires normally. Proves the guard is disposition-specific, not a
        # blanket mute of brow-furrow reactions.
        c = self.c
        c.world_state.update("people", [
            self._person(
                "brow_furrow", "focused", 0.88,
                {"browDownLeft": 0.90, "browDownRight": 0.86},
            )
        ])
        stats = {"total_samples": 60, "dominant_expression": "smile", "confidence": 0.80}
        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 0.0),
            mock.patch("memory.disposition.get_stats", return_value=stats),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "brow_furrow")

    def test_disposition_guard_helper_gates(self):
        c = self.c
        dominant = {"total_samples": 60, "dominant_expression": "brow_furrow", "confidence": 0.86}
        thin = {"total_samples": 5, "dominant_expression": "brow_furrow", "confidence": 0.86}
        with mock.patch("memory.disposition.get_stats", return_value=dominant):
            self.assertTrue(c._expression_is_habitual_disposition(1, "brow_furrow"))
            self.assertFalse(c._expression_is_habitual_disposition(1, "smile"))
            self.assertFalse(c._expression_is_habitual_disposition(None, "brow_furrow"))
        with mock.patch("memory.disposition.get_stats", return_value=thin):
            self.assertFalse(c._expression_is_habitual_disposition(1, "brow_furrow"))
        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_RESPECT_DISPOSITION", False),
            mock.patch("memory.disposition.get_stats", return_value=dominant),
        ):
            self.assertFalse(c._expression_is_habitual_disposition(1, "brow_furrow"))

    def test_neutral_expression_is_ignored(self):
        c = self.c
        c.world_state.update("people", [self._person("neutral", "neutral", 0.99)])

        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SUSTAIN_SECS", 0.0),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())

        speak.assert_not_called()

    def test_sustained_smile_can_trigger_general_reaction(self):
        c = self.c
        c.world_state.update("people", [
            self._person(
                "smile",
                "happy",
                0.78,
                {"mouthSmileLeft": 0.78, "mouthSmileRight": 0.76},
            )
        ])

        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_SMILE_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_COOLDOWN_SECS", 0.0),
            mock.patch.object(c, "_speak_facial_expression_reaction", return_value=True) as speak,
        ):
            c._step_facial_expression_reactions(c.world_state.snapshot(), mock.Mock())

        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "smile")

    def test_moderate_brow_furrow_is_ignored_as_pensive(self):
        c = self.c
        c.world_state.update("people", [
            self._person(
                "brow_furrow",
                "focused",
                0.72,
                {"browDownLeft": 0.72, "browDownRight": 0.70},
            )
        ])

        with (
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_BROW_FURROW_SUSTAIN_SECS", 0.0),
            mock.patch.object(c.config, "FACIAL_EXPRESSION_REACTION_GLOBAL_COOLDOWN_SECS", 0.0),
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


class LiveExpressionInReplyTests(unittest.TestCase):
    """`llm._live_expression_prompt_line` surfaces the engaged person's NOTABLE
    live expression into the reply prompt so Rex can react to a smile WITHIN his
    reply (not only via the proactive smile reaction, which is often suppressed
    mid-conversation). The detection gating is consciousness's job (tested above);
    this pins person-finding, phrasing, and the kill switch."""

    def _ws(self, person_id=1, name="Bret Benziger"):
        return {"people": [{"person_db_id": person_id, "name": name,
                            "face_expression": {"expression": "smile", "confidence": 0.9}}]}

    def test_notable_expression_surfaces_for_engaged_person(self):
        from intelligence import llm, consciousness
        with mock.patch.object(consciousness, "_person_reactable_expression",
                               return_value=("smile", 0.9)):
            line = llm._live_expression_prompt_line(self._ws(), 1)
        self.assertIn("smiling", line.lower())
        self.assertIn("Bret", line)
        self.assertIn("right now", line.lower())
        self.assertIn("never say a camera", line.lower())  # instructs against narrating the camera

    def test_no_notable_expression_is_silent(self):
        from intelligence import llm, consciousness
        with mock.patch.object(consciousness, "_person_reactable_expression",
                               return_value=(None, 0.0)):
            self.assertEqual(llm._live_expression_prompt_line(self._ws(), 1), "")

    def test_kill_switch_disables_injection(self):
        from intelligence import llm, consciousness
        with mock.patch.object(llm.config, "LIVE_EXPRESSION_IN_REPLY_ENABLED", False), \
             mock.patch.object(consciousness, "_person_reactable_expression",
                               return_value=("smile", 0.9)):
            self.assertEqual(llm._live_expression_prompt_line(self._ws(), 1), "")

    def test_no_person_is_silent(self):
        from intelligence import llm
        self.assertEqual(llm._live_expression_prompt_line({"people": []}, 1), "")
        self.assertEqual(llm._live_expression_prompt_line({}, None), "")


if __name__ == "__main__":
    unittest.main()
