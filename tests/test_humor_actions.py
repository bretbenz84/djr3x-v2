import unittest
from unittest import mock


class HumorActionExecutionTests(unittest.TestCase):
    def test_router_tell_joke_action_generates_single_punchline_contract(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.tell_joke",
            confidence=0.96,
            args={},
            reason="explicit joke request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Joke line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "tell me a joke",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "Joke line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("Tell exactly ONE short in-character DJ-R3X joke", prompt)
        self.assertIn("Deliver the punchline and stop", prompt)
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "Joke line.")
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")
        # The visor-peek button now LANDS in the post-line silence: it is deferred to
        # the line's on_audio_end hook rather than fired upfront over the line.
        beat.assert_not_called()
        self.assertIsNotNone(speak.call_args.kwargs.get("on_audio_end"))

    def test_router_roast_action_keeps_prompt_non_sensitive(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.roast",
            confidence=0.96,
            args={"target": "speaker"},
            reason="explicit roast request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Roast line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "roast me",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "Roast line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("consent-based Rex roast", prompt)
        self.assertIn("Do NOT joke about body, age, gender", prompt)
        self.assertIn("No question. One line only.", prompt)
        self.assertEqual(speak.call_args.kwargs["emotion"], "curious")
        # The side-eye button now lands in the post-line silence (deferred to the
        # line's on_audio_end), not fired upfront over the roast.
        beat.assert_not_called()
        self.assertIsNotNone(speak.call_args.kwargs.get("on_audio_end"))

    def test_fast_local_takeover_handles_explicit_free_humor_without_router_flag(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Funny line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat"),
        ):
            response = interaction._handle_fast_local_takeover(
                "say something funny",
                person_id=None,
                person_name=None,
            )

        self.assertEqual(response, "Funny line.")
        self.assertIn("asked Rex to be funny or do a bit", get_response.call_args.args[0])
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")

    def test_fast_local_takeover_ignores_plain_joke_mentions(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._handle_fast_local_takeover(
                "that joke was funny",
                person_id=None,
                person_name=None,
            )

        self.assertIsNone(response)
        get_response.assert_not_called()
        speak.assert_not_called()

    def test_router_dj_bit_action_uses_performance_plan(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.dj_bit",
            confidence=0.95,
            args={},
            reason="explicit DJ performance request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="DJ line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "do your DJ thing",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertEqual(response, "DJ line.")
        prompt = get_response.call_args.args[0]
        self.assertIn("DJ-R3X cantina patter", prompt)
        self.assertIn("Do not start music", prompt)
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")
        beat.assert_called_once_with("proud_dj_pose")

    def test_fast_local_takeover_handles_explicit_dj_bit(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Hype line.") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat"),
        ):
            response = interaction._handle_fast_local_takeover(
                "hype the room",
                person_id=None,
                person_name=None,
            )

        self.assertEqual(response, "Hype line.")
        self.assertIn("station-break", get_response.call_args.args[0])
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")

    def test_router_body_beat_action_executes_without_llm_generation(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="performance.body_beat",
            confidence=0.95,
            args={"body_beat": "suspicious_glance"},
            reason="explicit body beat performance request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_router_takeover_action(
                decision,
                "look suspicious",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        self.assertIn("Suspicious glance", response)
        get_response.assert_not_called()
        beat.assert_called_once_with("suspicious_glance")
        self.assertEqual(speak.call_args.args[0], response)
        self.assertEqual(speak.call_args.kwargs["emotion"], "curious")

    def test_fast_local_takeover_handles_explicit_body_beat(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.llm, "get_response") as get_response,
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch("sequences.animations.play_body_beat") as beat,
        ):
            response = interaction._handle_fast_local_takeover(
                "do a victory dance",
                person_id=None,
                person_name=None,
            )

        self.assertIn("Tiny victory dance", response)
        get_response.assert_not_called()
        beat.assert_called_once_with("tiny_victory_dance")
        self.assertEqual(speak.call_args.kwargs["emotion"], "happy")

    def test_router_roast_action_makes_rex_smug(self):
        # After landing a deliberate roast, Rex basks: a 'smug' body mood (resolves
        # to the proud chin-up posture) — the affective mirror of compliment->proud.
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.roast",
            confidence=0.96,
            args={"target": "speaker"},
            reason="explicit roast request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Roast line."),
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
            mock.patch("sequences.animations.play_body_beat"),
            mock.patch.object(interaction, "_set_body_mood") as set_mood,
        ):
            interaction._handle_router_takeover_action(
                decision,
                "roast me",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        set_mood.assert_called_once_with("smug", source="roast_landed")

    def test_joke_action_does_not_make_rex_smug(self):
        # Only a roast triggers the smug afterglow — a plain joke must not.
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="humor.tell_joke",
            confidence=0.96,
            args={},
            reason="explicit joke request",
        )

        with (
            mock.patch.object(interaction.llm, "get_response", return_value="Joke line."),
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
            mock.patch("sequences.animations.play_body_beat"),
            mock.patch.object(interaction, "_set_body_mood") as set_mood,
        ):
            interaction._handle_router_takeover_action(
                decision,
                "tell me a joke",
                person_id=1,
                person_name="Bret",
                raw_best_id=1,
                raw_best_name="Bret",
                raw_best_score=0.99,
            )

        for call in set_mood.call_args_list:
            self.assertNotEqual(call.args[0], "smug")


class RoastVisionTests(unittest.TestCase):
    """'Roast me' → roast what Rex SEES (a consent self/room roast). The vision read
    is scoped to the speaker/their room and a non-minor, and falls back to the verbal
    roast otherwise."""

    def _decision(self, target="speaker"):
        from intelligence import action_router

        return action_router.ActionDecision(
            action="humor.roast",
            confidence=0.96,
            args={"target": target},
            reason="explicit roast request",
        )

    # --- plan_for_action: vision contract vs verbal contract ---
    def test_plan_with_visual_material_roasts_what_rex_sees(self):
        from intelligence import performance_plan

        plan = performance_plan.plan_for_action(
            "humor.roast",
            user_text="roast me",
            args={"target": "speaker"},
            visual_material="rumpled gray hoodie, three energy drinks, cables everywhere",
        )
        self.assertIn("rumpled gray hoodie", plan.prompt_contract)
        # Targets the PERSON, and explicitly steers OFF the room/furniture.
        self.assertIn("THE PERSON", plan.prompt_contract)
        self.assertIn("not their furniture", plan.prompt_contract)
        # The protected-category floor survives even in the loosened consent roast.
        self.assertIn("race", plan.prompt_contract)
        self.assertIn("disability", plan.prompt_contract)
        self.assertIn("medical condition", plan.prompt_contract)
        self.assertEqual(plan.delivery_style, "consent_roast")

    def test_plan_without_visual_material_uses_verbal_roast(self):
        from intelligence import performance_plan

        plan = performance_plan.plan_for_action(
            "humor.roast", user_text="roast me", args={"target": "speaker"}
        )
        self.assertIn("consent-based Rex roast", plan.prompt_contract)
        self.assertIn("Do NOT joke about body", plan.prompt_contract)
        self.assertNotIn("what Rex SEES", plan.prompt_contract)

    # --- self/room vs third-party target detection ---
    def test_self_and_room_targets_are_eligible(self):
        from intelligence import interaction

        self.assertTrue(interaction._roast_targets_speaker(self._decision("speaker"), 1))
        self.assertTrue(interaction._roast_targets_speaker(self._decision("room"), 1))
        self.assertTrue(interaction._roast_targets_speaker(self._decision(""), 1))

    def test_named_third_party_is_not_a_self_roast(self):
        from intelligence import interaction

        with mock.patch("memory.people.get_person", return_value={"name": "Bret"}):
            self.assertFalse(
                interaction._roast_targets_speaker(self._decision("Dave"), 1)
            )

    def test_target_matching_speaker_name_is_a_self_roast(self):
        from intelligence import interaction

        with mock.patch("memory.people.get_person", return_value={"name": "Bret Benziger"}):
            self.assertTrue(
                interaction._roast_targets_speaker(self._decision("bret"), 1)
            )

    # --- _roast_visual_material gating ---
    def test_visual_material_empty_when_feature_disabled(self):
        from intelligence import interaction

        with mock.patch.object(interaction.config, "ROAST_VISION_ENABLED", False):
            self.assertEqual(
                interaction._roast_visual_material(self._decision("speaker"), 1), ""
            )

    def test_visual_material_empty_for_third_party(self):
        from intelligence import interaction

        with mock.patch("memory.people.get_person", return_value={"name": "Bret"}):
            self.assertEqual(
                interaction._roast_visual_material(self._decision("Dave"), 1), ""
            )

    def test_visual_material_empty_for_minor(self):
        from intelligence import interaction

        with mock.patch("intelligence.profile_questions.person_is_minor", return_value=True):
            self.assertEqual(
                interaction._roast_visual_material(self._decision("speaker"), 1), ""
            )

    def test_visual_material_empty_for_unidentified_speaker(self):
        from intelligence import interaction

        # No person_id → could be a minor the camera sees → fail safe (verbal roast).
        self.assertEqual(
            interaction._roast_visual_material(self._decision("speaker"), None), ""
        )

    def test_visual_material_failsafe_when_minor_check_errors(self):
        from intelligence import interaction

        with mock.patch(
            "intelligence.profile_questions.person_is_minor",
            side_effect=RuntimeError("db down"),
        ):
            # Can't confirm adult → fail safe, NOT proceed.
            self.assertEqual(
                interaction._roast_visual_material(self._decision("speaker"), 1), ""
            )

    def test_visual_material_empty_in_tender_empathy_mode(self):
        from intelligence import interaction

        with (
            mock.patch("intelligence.profile_questions.person_is_minor", return_value=False),
            mock.patch(
                "intelligence.empathy.get_delivery_overrides",
                return_value={"mode": "support"},
            ),
        ):
            self.assertEqual(
                interaction._roast_visual_material(self._decision("speaker"), 1), ""
            )

    def test_visual_material_empty_when_no_camera_frame(self):
        from intelligence import interaction

        with (
            mock.patch("intelligence.profile_questions.person_is_minor", return_value=False),
            mock.patch("intelligence.empathy.get_delivery_overrides", return_value=None),
            mock.patch("vision.camera.get_frame", return_value=None),
        ):
            self.assertEqual(
                interaction._roast_visual_material(self._decision("speaker"), 1), ""
            )

    def test_visual_material_describes_frame_for_consenting_adult(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.config, "ROAST_VISION_ENABLED", True),
            mock.patch("intelligence.profile_questions.person_is_minor", return_value=False),
            mock.patch("intelligence.empathy.get_delivery_overrides", return_value=None),
            mock.patch("vision.camera.get_frame", return_value=object()),
            mock.patch("vision.face.visible_known_names", return_value=["Bret"]),
            mock.patch("vision.scene.describe_for_roast", return_value="slouched, messy desk") as desc,
        ):
            material = interaction._roast_visual_material(self._decision("speaker"), 1)
        self.assertEqual(material, "slouched, messy desk")
        desc.assert_called_once()

    # --- vision.scene.describe_for_roast ---
    def test_describe_for_roast_returns_empty_on_none_frame(self):
        from vision import scene

        self.assertEqual(scene.describe_for_roast(None), "")

    def test_describe_for_roast_returns_model_text(self):
        from vision import scene

        with mock.patch.object(scene, "_call_gpt4o", return_value="hoodie, clutter") as call:
            out = scene.describe_for_roast(object(), known_names=["Bret"])
        self.assertEqual(out, "hoodie, clutter")
        # The cheap detail tier is used.
        self.assertEqual(call.call_args.args[2], "roast")

    def test_describe_for_roast_swallows_errors(self):
        from vision import scene

        with mock.patch.object(scene, "_call_gpt4o", side_effect=RuntimeError("boom")):
            self.assertEqual(scene.describe_for_roast(object(), known_names=[]), "")


if __name__ == "__main__":
    unittest.main()
