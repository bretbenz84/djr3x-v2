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


class ConversationGatingTest(unittest.TestCase):
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
