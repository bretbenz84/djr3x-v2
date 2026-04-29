import unittest
from unittest import mock


class PostTtsHandoffPolicyTest(unittest.TestCase):
    def test_question_handoff_preserves_buffer_and_uses_short_delay(self):
        from intelligence import interaction
        import config

        policy = interaction._post_tts_handoff_policy("What's your favorite movie?")

        self.assertTrue(policy.asked_question)
        self.assertEqual(
            policy.listen_delay_secs,
            float(config.POST_QUESTION_LISTEN_DELAY_SECS),
        )
        self.assertFalse(policy.flush_buffer)

    def test_statement_handoff_flushes_buffer_and_uses_general_delay(self):
        from intelligence import interaction
        import config

        policy = interaction._post_tts_handoff_policy("Classic choice.")

        self.assertFalse(policy.asked_question)
        self.assertEqual(
            policy.listen_delay_secs,
            float(config.POST_SPEECH_LISTEN_DELAY_SECS),
        )
        self.assertTrue(policy.flush_buffer)

    def test_apply_question_handoff_does_not_flush_stream(self):
        from intelligence import interaction

        with mock.patch.object(interaction.stream, "flush") as flush:
            interaction._apply_post_tts_handoff(
                "What do you do for work?",
                source="test",
            )

        flush.assert_not_called()

    def test_no_response_recovery_waits_for_cooldown_and_user_speech(self):
        from intelligence import interaction

        with mock.patch.object(
            interaction,
            "_question_recovery_cooldown_secs",
            return_value=7.0,
        ):
            self.assertFalse(
                interaction._should_no_response_recovery_fire(
                    asked_at=100.0,
                    now=106.9,
                    last_speech_at=100.0,
                )
            )
            self.assertFalse(
                interaction._should_no_response_recovery_fire(
                    asked_at=100.0,
                    now=108.0,
                    last_speech_at=101.0,
                )
            )
            self.assertTrue(
                interaction._should_no_response_recovery_fire(
                    asked_at=100.0,
                    now=108.0,
                    last_speech_at=100.0,
                )
            )

    def test_wake_ack_never_repeats_back_to_back_and_requires_cache(self):
        from intelligence import interaction

        interaction._last_wake_ack = None
        with (
            mock.patch.object(
                interaction.config,
                "WAKE_ACKNOWLEDGMENTS",
                ["yeah?", "what?"],
            ),
            mock.patch.object(interaction.config, "WAKE_ACK_REQUIRE_CACHE", True),
            mock.patch("audio.tts.is_cached", return_value=True),
            mock.patch.object(interaction, "_speak_blocking") as speak,
            mock.patch.object(interaction.random, "choice", side_effect=lambda seq: seq[0]),
        ):
            interaction._wake_ack()
            interaction._wake_ack()

        self.assertEqual(
            [call.args[0] for call in speak.call_args_list],
            ["yeah?", "what?"],
        )

    def test_wake_ack_skips_uncached_lines(self):
        from intelligence import interaction

        interaction._last_wake_ack = None
        with (
            mock.patch.object(interaction.config, "WAKE_ACKNOWLEDGMENTS", ["yeah?"]),
            mock.patch.object(interaction.config, "WAKE_ACK_REQUIRE_CACHE", True),
            mock.patch("audio.tts.is_cached", return_value=False),
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            interaction._wake_ack()

        speak.assert_not_called()

    def test_bare_wake_address_detection(self):
        from intelligence import interaction

        for text in ("Hey Rex", "hey dj-rex", "DJ Rex", "yo robot", "R3X"):
            self.assertTrue(interaction._is_bare_wake_address(text), text)

        self.assertFalse(interaction._is_bare_wake_address("Hey Rex what time is it"))
        self.assertFalse(interaction._is_bare_wake_address("Rex play jazz"))


class ConversationGatingTest(unittest.TestCase):
    def test_conversation_steering_detects_interest_declarations(self):
        from intelligence import conversation_steering

        self.assertEqual(
            conversation_steering.detect_interest("I'm into astrophotography."),
            "astrophotography",
        )
        self.assertEqual(
            conversation_steering.detect_interest("3D printing is my hobby."),
            "3D printing",
        )
        self.assertEqual(
            conversation_steering.detect_interest("My favorite activity is hair styling."),
            "hair styling",
        )
        self.assertIsNone(conversation_steering.detect_interest("I do not know."))

    def test_interest_declaration_is_stored_and_steers_agenda(self):
        from intelligence import conversation_agenda, conversation_steering

        conversation_steering.clear()
        with (
            mock.patch.object(
                conversation_agenda.world_state,
                "snapshot",
                return_value={"people": [], "environment": {}},
            ),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ) as add_fact,
        ):
            directive = conversation_agenda.build_turn_directive(
                "I'm really into astrophotography.",
                1,
            )

        self.assertIn("Conversation steering", directive)
        self.assertIn("astrophotography", directive)
        self.assertIn("subject-specific observation", directive)
        self.assertIn("natural follow-up", directive)
        add_fact.assert_any_call(
            1,
            "interest",
            "interest_astrophotography",
            "astrophotography",
            "interest_declaration",
            confidence=0.95,
        )
        conversation_steering.clear()

    def test_interest_thread_stores_notable_followup_notes(self):
        from intelligence import conversation_steering

        conversation_steering.clear()
        with (
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ) as add_fact,
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
        ):
            conversation_steering.note_user_turn(1, "I like 3D printing.")
            conversation_steering.note_user_turn(
                1,
                "I usually build little brackets because my printer is tiny.",
            )

        calls = add_fact.call_args_list
        self.assertTrue(
            any(call.args[1] == "interest_note" for call in calls),
            calls,
        )
        conversation_steering.clear()

    def test_interest_steering_respects_topic_boundaries(self):
        from intelligence import conversation_steering

        conversation_steering.clear()
        with (
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ),
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=True,
            ),
        ):
            ctx = conversation_steering.note_user_turn(1, "I'm into hair styling.")

        self.assertIsNone(ctx)
        conversation_steering.clear()

    def test_curiosity_followup_prefers_active_interest_thread(self):
        from intelligence import conversation_steering, interaction

        conversation_steering.clear()
        with (
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ),
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
        ):
            conversation_steering.note_user_turn(1, "I'm into hair styling.")

        with (
            mock.patch.object(interaction.random, "random", return_value=0.0),
            mock.patch.object(interaction.question_budget, "can_ask", return_value=True),
            mock.patch.object(interaction.empathy, "peek", return_value=None),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(
                interaction.llm,
                "get_response",
                return_value="What's the hardest hair disaster you've rescued?",
            ) as get_response,
            mock.patch.object(interaction, "_speak_blocking") as speak,
            mock.patch.object(interaction.rel_memory, "save_question_asked") as save_qa,
        ):
            question = interaction._curiosity_check(
                "Hair styling logged. My circuits fear curling irons.",
                "I'm into hair styling.",
                1,
                "Joy",
            )

        self.assertEqual(
            question,
            "What's the hardest hair disaster you've rescued?",
        )
        self.assertIn("hair styling", get_response.call_args.args[0])
        speak.assert_called_once_with(question)
        self.assertEqual(save_qa.call_args.args[1], "interest_hair_styling_followup")
        conversation_steering.clear()

    def test_local_sensitive_classifier_detects_death_subject(self):
        from intelligence import empathy

        result = empathy.classify_local_sensitivity("My dad died yesterday.")

        self.assertIsNotNone(result)
        self.assertEqual(result["topic_sensitivity"], "heavy")
        self.assertEqual(result["affect"], "sad")
        self.assertFalse(result["crisis"])
        self.assertEqual(result["event"]["category"], "death")
        self.assertEqual(result["event"]["loss_subject"], "dad")
        self.assertEqual(result["event"]["loss_subject_kind"], "person")

    def test_local_sensitive_classifier_avoids_common_death_false_alarms(self):
        from intelligence import empathy

        self.assertIsNone(empathy.classify_local_sensitivity("I lost my keys."))
        self.assertIsNone(empathy.classify_local_sensitivity("I'm dead tired."))

    def test_agenda_suppresses_roasts_for_same_turn_sensitive_disclosure(self):
        from intelligence import conversation_agenda

        with (
            mock.patch.object(
                conversation_agenda.world_state,
                "snapshot",
                return_value={"people": [], "environment": {}},
            ),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            directive = conversation_agenda.build_turn_directive(
                "My dog died last night.",
                1,
            )

        self.assertIn("sensitive disclosure detected in this exact user turn", directive)
        self.assertIn("Drop roast-first mode completely", directive)
        self.assertIn("No personal roasts", directive)

    def test_local_sensitive_prepass_records_current_turn_mode(self):
        from intelligence import empathy, interaction

        empathy.clear()
        with (
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"},
            ),
            mock.patch.object(interaction.world_state, "get", return_value=[]),
        ):
            result = interaction._apply_local_sensitive_topic_prepass(
                1,
                "My mom passed away.",
            )

        cached = empathy.peek(1)
        self.assertIsNotNone(result)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["result"]["event"]["loss_subject"], "mom")
        self.assertEqual(cached["mode"]["mode"], "listen")
        empathy.clear()

    def test_local_sensitive_prepass_allows_grief_flow_if_async_classifier_times_out(self):
        from intelligence import empathy, interaction

        empathy.clear()
        interaction._grief_flow_state.clear()
        with (
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"},
            ),
            mock.patch.object(interaction.world_state, "get", return_value=[]),
        ):
            interaction._apply_local_sensitive_topic_prepass(1, "My cat died.")

        cached = empathy.peek(1)
        response = interaction._maybe_start_grief_flow(
            1,
            cached["result"]["event"],
        )

        self.assertIsNotNone(response)
        self.assertIn("cat", response)
        self.assertIn(1, interaction._grief_flow_state)
        empathy.clear()
        interaction._grief_flow_state.clear()

    def test_late_weaker_empathy_result_cannot_erase_local_sensitive_prepass(self):
        from intelligence import empathy, interaction

        local = empathy.classify_local_sensitivity("My dad died yesterday.")
        late_neutral = {
            "affect": "neutral",
            "needs": "none",
            "topic_sensitivity": "none",
            "invitation": False,
            "crisis": False,
            "confidence": 0.7,
            "event": None,
        }

        merged = interaction._merge_with_local_sensitive_prepass(late_neutral, local)

        self.assertEqual(merged["topic_sensitivity"], "heavy")
        self.assertEqual(merged["event"]["loss_subject"], "dad")

    def test_social_scene_cast_summarizes_group_and_pronouns(self):
        from intelligence import social_scene

        ws = {
            "people": [
                {
                    "id": "person_1",
                    "person_db_id": 1,
                    "face_id": "Bret Benziger",
                },
                {
                    "id": "person_2",
                    "person_db_id": 2,
                    "face_id": "JT Example",
                },
            ],
        }
        facts = {
            1: [],
            2: [{"category": "identity", "key": "pronouns", "value": "they/them"}],
        }
        with mock.patch(
            "memory.facts.get_facts",
            side_effect=lambda person_id: facts.get(person_id, []),
        ):
            cast = social_scene.conversation_cast_context(
                ws,
                current_person_id=1,
            )

        self.assertIn("Bret primarily; visible group", cast.addressee)
        self.assertIn("JT (they/them)", cast.directive)
        self.assertIn("Referent candidates besides the speaker: JT", cast.directive)
        self.assertIn("Pronoun and group-address rules", cast.directive)

    def test_social_frame_uses_group_addressee_when_multiple_known_people_visible(self):
        from intelligence import social_frame

        ws = {
            "people": [
                {"id": "person_1", "person_db_id": 1, "face_id": "Bret Benziger"},
                {"id": "person_2", "person_db_id": 2, "face_id": "JT Example"},
            ],
        }
        with (
            mock.patch.object(social_frame.world_state, "snapshot", return_value=ws),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("memory.facts.get_facts", return_value=[]),
        ):
            frame = social_frame.build_frame(
                "that was funny",
                person_id=1,
                agenda_directive="Primary purpose: respond to the human's latest thought.",
            )

        self.assertIn("Bret primarily; visible group", frame.addressee)
        self.assertIn("Bret and JT", frame.addressee)

    def test_visible_unknown_followup_arms_relationship_parser(self):
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction.world_state,
                "get",
                return_value=[
                    {"id": "person_1", "person_db_id": 1, "face_id": "Bret"},
                    {"id": "person_2", "person_db_id": None, "face_id": None},
                ],
            ),
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret Benziger"},
            ),
            mock.patch.object(
                interaction.consciousness,
                "set_relationship_prompt_context",
            ) as set_ctx,
        ):
            interaction._arm_visible_unknown_identity_followup(
                1,
                source="test",
            )

        set_ctx.assert_called_once()
        ctx = set_ctx.call_args.args[0]
        self.assertEqual(ctx["engaged_person_id"], 1)
        self.assertEqual(ctx["engaged_name"], "Bret Benziger")
        self.assertEqual(ctx["slot_id"], "person_2")

    def test_pronoun_repair_stores_explicit_named_pronouns(self):
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction.people_memory,
                "find_person_by_name",
                return_value={"id": 2, "name": "JT Example"},
            ),
            mock.patch.object(interaction.facts_memory, "add_fact") as add_fact,
        ):
            interaction._maybe_store_pronoun_repair(
                1,
                "JT uses they/them pronouns.",
            )

        add_fact.assert_called_once_with(
            2,
            "identity",
            "pronouns",
            "they/them",
            "pronoun_repair",
            confidence=0.95,
        )

    def test_agenda_allows_related_followup_when_curated_pool_is_exhausted(self):
        from intelligence import conversation_agenda

        with (
            mock.patch.object(
                conversation_agenda.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"},
            ),
            mock.patch.object(
                conversation_agenda.rel_memory,
                "get_latest_pending_question",
                return_value=None,
            ),
            mock.patch.object(
                conversation_agenda,
                "_next_useful_question",
                return_value=None,
            ),
            mock.patch(
                "intelligence.question_budget.can_ask",
                return_value=True,
            ),
            mock.patch(
                "intelligence.question_budget.build_directive",
                return_value="",
            ),
        ):
            directive = conversation_agenda.build_turn_directive(
                "I work in computers",
                1,
            )

        self.assertIn("tightly related follow-up question", directive)
        self.assertIn("do not pivot into a new interview topic", directive)

    def test_social_frame_allows_the_related_followup_directive(self):
        from intelligence import social_frame

        directive = (
            "Primary purpose: respond to the human's latest thought. "
            "You may ask one tightly related follow-up question if it naturally "
            "continues this exact thread; do not pivot into a new interview topic."
        )
        with (
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch.object(
                social_frame.world_state,
                "snapshot",
                return_value={"people": []},
            ),
        ):
            frame = social_frame.build_frame(
                "I work in computers",
                person_id=1,
                agenda_directive=directive,
            )

        self.assertTrue(frame.allow_question)

    def test_social_frame_allows_followup_after_user_question_when_agenda_allows(self):
        from intelligence import social_frame

        directive = (
            "Primary purpose: answer the human's question directly first. "
            "After answering, ask at most one short follow-up only if it flows "
            "from their question."
        )
        with (
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch.object(
                social_frame.world_state,
                "snapshot",
                return_value={"people": []},
            ),
        ):
            frame = social_frame.build_frame(
                "What do you do for work?",
                person_id=1,
                agenda_directive=directive,
            )

        self.assertTrue(frame.allow_question)

    def test_unknown_group_agenda_prioritizes_identity_handoff(self):
        from intelligence import conversation_agenda

        ws = {
            "crowd": {"count": 2, "interaction_mode": "small_group"},
            "people": [
                {
                    "id": "person_1",
                    "person_db_id": 1,
                    "face_id": "Bret Benziger",
                },
                {
                    "id": "person_2",
                    "person_db_id": None,
                    "face_id": None,
                },
            ],
            "environment": {},
        }
        with (
            mock.patch.object(conversation_agenda.world_state, "snapshot", return_value=ws),
            mock.patch("intelligence.question_budget.can_ask", return_value=False),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            directive = conversation_agenda.build_turn_directive("hello there", 1)

        self.assertIn("urgent group identity handoff", directive)
        self.assertIn("Bret", directive)
        self.assertIn("may bypass the optional question budget", directive)

    def test_unknown_group_social_frame_keeps_identity_question(self):
        from intelligence import social_frame

        ws = {
            "people": [
                {
                    "id": "person_1",
                    "person_db_id": 1,
                    "face_id": "Bret Benziger",
                },
                {
                    "id": "person_2",
                    "person_db_id": None,
                    "face_id": None,
                },
            ],
        }
        directive = (
            "Primary purpose: urgent group identity handoff. "
            "There is an unfamiliar guest visible. Ask who they are and get a name. "
            "This identity question may bypass the optional question budget."
        )
        with (
            mock.patch("intelligence.question_budget.can_ask", return_value=False),
            mock.patch.object(social_frame.world_state, "snapshot", return_value=ws),
        ):
            frame = social_frame.build_frame(
                "thanks",
                person_id=1,
                agenda_directive=directive,
            )
            governed = social_frame.govern_response(
                "Great, Bret. Who is your mystery guest, and should I be concerned?",
                frame,
            )

        self.assertEqual(frame.purpose, "identity")
        self.assertTrue(frame.allow_question)
        self.assertIn("mystery guest", frame.addressee)
        self.assertIn("?", governed.text)

    def test_social_frame_removes_novel_roast_in_no_roast_mode(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="support",
            max_words=40,
            max_sentences=3,
            allow_question=False,
            allow_roast="none",
            allow_visual_comment=True,
            reason="test",
        )
        governed = social_frame.govern_response(
            "I hear you. You are a walking software outage in sneakers.",
            frame,
        )

        self.assertEqual(governed.text, "I hear you.")
        self.assertIn("removed_roast", governed.notes)

    def test_social_frame_removes_condescending_organic_roast_in_no_roast_mode(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="the room",
            purpose="support",
            max_words=40,
            max_sentences=3,
            allow_question=False,
            allow_roast="none",
            allow_visual_comment=True,
            reason="test",
        )
        governed = social_frame.govern_response(
            "That sounds hard. Classic fragile organic decision-making.",
            frame,
        )

        self.assertEqual(governed.text, "That sounds hard.")
        self.assertIn("removed_roast", governed.notes)

    def test_social_frame_removes_tiny_tap_in_no_roast_mode(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="support",
            max_words=40,
            max_sentences=3,
            allow_question=False,
            allow_roast="none",
            allow_visual_comment=True,
            reason="test",
        )
        governed = social_frame.govern_response(
            "That sounds hard. Bold choice, captain.",
            frame,
        )

        self.assertEqual(governed.text, "That sounds hard.")
        self.assertIn("removed_roast", governed.notes)

    def test_social_frame_allows_tiny_tap_but_removes_sharp_roast_in_light_mode(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="banter",
            max_words=40,
            max_sentences=3,
            allow_question=False,
            allow_roast="light",
            allow_visual_comment=True,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Bold choice, captain. You are a pathetic disaster.",
            frame,
        )

        self.assertEqual(governed.text, "Bold choice, captain.")
        self.assertIn("removed_sharp_roast", governed.notes)

    def test_agenda_invites_opinions_and_roasts_after_simple_ack(self):
        from intelligence import conversation_agenda

        with (
            mock.patch.object(
                conversation_agenda.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"},
            ),
            mock.patch.object(
                conversation_agenda.rel_memory,
                "get_latest_pending_question",
                return_value=None,
            ),
            mock.patch(
                "intelligence.question_budget.can_ask",
                return_value=True,
            ),
            mock.patch(
                "intelligence.question_budget.build_directive",
                return_value="",
            ),
        ):
            directive = conversation_agenda.build_turn_directive(
                "You're such a good robot",
                1,
            )

        self.assertIn("specific Rex opinion", directive)
        self.assertIn("light roast", directive)
        self.assertIn("Do not pivot into a new interview question", directive)

    def test_llm_places_turn_contract_after_behavioral_rules(self):
        from intelligence import llm

        directive = "Final response shape contract:\n- Hard shape: max_words=24."
        with (
            mock.patch.object(
                llm.world_state,
                "snapshot",
                return_value={
                    "environment": {},
                    "crowd": {"count": 0},
                    "audio_scene": {},
                    "self_state": {},
                    "time": {},
                    "animals": [],
                    "people": [],
                },
            ),
            mock.patch.object(llm.conv_db, "get_session_transcript", return_value=[]),
            mock.patch.object(llm, "_get_personality_params", return_value={}),
        ):
            prompt = llm.assemble_system_prompt(None, agenda_directive=directive)

        self.assertGreater(
            prompt.rfind("Turn-specific response contract"),
            prompt.rfind("Behavioral rules"),
        )
        self.assertIn(directive, prompt)

    def test_startup_group_prompt_uses_conversation_steering_openers(self):
        from intelligence import social_scene

        scene = social_scene.SocialScene(
            known=(
                social_scene.VisiblePerson(1, "Bret Benziger", "Bret", "person_1"),
                social_scene.VisiblePerson(2, "Joy Example", "Joy", "person_2"),
            ),
            unknown_count=0,
            crowd_count=2,
        )

        prompt = social_scene.startup_group_prompt(scene)

        self.assertIn("conversation-steering question", prompt)
        self.assertIn("what are you up to today", prompt)
        self.assertIn("what do you want to talk about", prompt)
        self.assertIn("What mission are we pretending is important today?", prompt)

    def test_acknowledge_on_return_prompt_ends_with_steering_question(self):
        from intelligence import llm

        with (
            mock.patch.object(
                llm.world_state,
                "snapshot",
                return_value={
                    "environment": {},
                    "crowd": {"count": 1},
                    "audio_scene": {},
                    "self_state": {},
                    "time": {},
                    "animals": [],
                    "people": [],
                },
            ),
            mock.patch.object(llm.conv_db, "get_session_transcript", return_value=[]),
            mock.patch.object(llm.people_db, "get_person", return_value={"id": 1, "name": "Bret"}),
            mock.patch.object(llm.facts_db, "get_prompt_facts", return_value=[]),
            mock.patch.object(llm, "_get_personality_params", return_value={}),
            mock.patch(
                "memory.emotional_events.summarize_for_prompt",
                return_value="Recent emotional context: had a hard week.",
            ),
            mock.patch(
                "memory.emotional_events.get_active_events",
                return_value=[{"id": 1, "last_acknowledged_at": None}],
            ),
            mock.patch(
                "memory.emotional_events.can_surface_event",
                return_value=True,
            ),
            mock.patch(
                "memory.emotional_events.is_heavy_event",
                return_value=False,
            ),
        ):
            prompt = llm.assemble_system_prompt(1)

        self.assertIn("ACKNOWLEDGE-ON-RETURN", prompt)
        self.assertIn("conversation-steering question", prompt)
        self.assertIn("what are you up to today", prompt)
        self.assertIn("What topic gets the honor", prompt)

    def test_agenda_surfaces_intimate_personal_space_cue(self):
        from intelligence import conversation_agenda

        ws = {
            "crowd": {
                "count": 1,
                "interaction_mode": "one_on_one",
                "engaged_count": 1,
            },
            "people": [
                {
                    "id": "person_1",
                    "face_id": "Bret",
                    "distance_zone": "intimate",
                }
            ],
            "environment": {},
        }
        with (
            mock.patch.object(conversation_agenda.world_state, "snapshot", return_value=ws),
            mock.patch.object(
                conversation_agenda.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "friend"},
            ),
            mock.patch.object(
                conversation_agenda.rel_memory,
                "get_latest_pending_question",
                return_value=None,
            ),
            mock.patch.object(conversation_agenda, "_next_useful_question", return_value=None),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            directive = conversation_agenda.build_turn_directive("hello", 1)

        self.assertIn("Proxemics cue", directive)
        self.assertIn("American norms", directive)
        self.assertIn("boundary joke or roast", directive)


class PendingMusicPreferenceTest(unittest.TestCase):
    def tearDown(self):
        from intelligence import interaction

        interaction._pending_music_offer = None

    def test_bare_music_preference_answer_is_not_played_immediately(self):
        from intelligence import interaction

        pending = {
            "question_key": "favorite_music",
            "question_text": "What kind of music are you into?",
        }
        answered = {"question_key": "favorite_music", "answer": "classical music"}

        with (
            mock.patch.object(
                interaction.rel_memory,
                "answer_latest_pending_question",
                return_value=answered,
            ) as answer,
            mock.patch.object(interaction.facts_memory, "add_fact") as add_fact,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response, captured = interaction._handle_pending_music_preference_answer(
                1,
                "classical music",
                pending_question=pending,
            )

        self.assertEqual(captured, answered)
        self.assertIn("Want me to play some classical music", response)
        answer.assert_called_once_with(1, "classical music")
        add_fact.assert_called_once_with(
            1,
            "preference",
            "favorite_music",
            "classical music",
            "pending_qa:favorite_music",
            confidence=0.95,
        )
        speak.assert_called_once()
        self.assertEqual(
            interaction._pending_music_offer["music_query"],
            "classical music",
        )

    def test_pending_music_offer_yes_starts_playback(self):
        from features import dj
        from intelligence import interaction

        track = dj.TrackInfo(
            source="radio",
            name="Classical Test",
            url_or_path="http://example.test/stream",
            description="test station",
        )
        interaction._pending_music_offer = {
            "person_id": 1,
            "music_query": "classical music",
            "asked_at": 100.0,
        }

        with (
            mock.patch.object(interaction.time, "monotonic", return_value=101.0),
            mock.patch.object(dj, "handle_request", return_value=track) as handle,
            mock.patch.object(dj, "play") as play,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._handle_pending_music_offer_reply(1, "yes")

        self.assertIn("Spinning Classical Test", response)
        handle.assert_called_once_with("classical music")
        play.assert_called_once_with(track)
        speak.assert_called_once()
        self.assertIsNone(interaction._pending_music_offer)

    def test_router_downgrades_bare_music_answer_under_pending_question(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="music.play",
            confidence=0.90,
            args={"music_query": "classical music"},
            reason="genre phrase",
        )
        context = {
            "pending": {
                "pending_question": {
                    "question_key": "favorite_music",
                    "question_text": "What kind of music are you into?",
                }
            }
        }

        routed = action_router._apply_context_overrides(
            decision,
            "classical music",
            context,
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_allows_explicit_music_play_under_pending_question(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="music.play",
            confidence=0.90,
            args={"music_query": "classical music"},
            reason="explicit play",
        )
        context = {
            "pending": {
                "pending_question": {
                    "question_key": "favorite_music",
                    "question_text": "What kind of music are you into?",
                }
            }
        }

        routed = action_router._apply_context_overrides(
            decision,
            "play classical music",
            context,
        )

        self.assertEqual(routed.action, "music.play")

    def test_dj_vibe_match_does_not_confuse_classical_with_classic_rock(self):
        import config
        from features import dj

        stations = [
            {
                "name": "Left Coast 70s",
                "url": "https://example.test/70s.pls",
                "vibes": ["70s", "classic rock", "retro"],
            }
        ]
        with mock.patch.object(config, "RADIO_STATIONS", stations):
            self.assertIsNone(dj._vibe_match("classical music", []))

    def test_dj_vibe_match_still_allows_exact_classic_rock(self):
        import config
        from features import dj

        stations = [
            {
                "name": "Left Coast 70s",
                "url": "https://example.test/70s.pls",
                "vibes": ["70s", "classic rock", "retro"],
            }
        ]
        with mock.patch.object(config, "RADIO_STATIONS", stations):
            match = dj._vibe_match("classic rock", [])

        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Left Coast 70s")


class SocialVisionIntegrationTest(unittest.TestCase):
    def test_social_crowd_updates_count_and_engagement(self):
        from awareness import social

        with mock.patch.object(social.world_state, "get", return_value={}):
            updated = {}
            with mock.patch.object(
                social.world_state,
                "update",
                side_effect=lambda field, value: updated.setdefault(field, value),
            ):
                result = social.analyze_crowd([
                    {"id": "a", "engagement": "high", "distance_zone": "social"},
                    {"id": "b", "engagement": "low", "distance_zone": "public"},
                ])

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["count_label"], "pair")
        self.assertEqual(result["engaged_count"], 1)
        self.assertEqual(result["interaction_mode"], "small_group")
        self.assertEqual(updated["crowd"]["count"], 2)

    def test_personal_space_helper_treats_intimate_as_too_close(self):
        from intelligence import consciousness

        self.assertTrue(
            consciousness._too_close_for_personal_space(
                {"distance_zone": "intimate"}
            )
        )
        self.assertFalse(
            consciousness._too_close_for_personal_space(
                {"distance_zone": "social"}
            )
        )

    def test_llm_world_summary_includes_visible_social_cues(self):
        from intelligence import llm

        summary = llm._summarize_world_state({
            "environment": {},
            "crowd": {
                "count": 1,
                "count_label": "alone",
                "interaction_mode": "one_on_one",
                "engaged_count": 1,
            },
            "people": [
                {
                    "face_id": "Bret",
                    "distance_zone": "intimate",
                    "approach_vector": "approaching",
                    "pose": "facing_forward",
                    "gesture": "leaning_in",
                    "engagement": "high",
                }
            ],
            "audio_scene": {},
            "self_state": {},
            "time": {},
            "animals": [],
        })

        self.assertIn("Interaction mode: one_on_one", summary)
        self.assertIn("Bret: distance=intimate", summary)
        self.assertIn("too close for comfort", summary)


if __name__ == "__main__":
    unittest.main()
