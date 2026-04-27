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


if __name__ == "__main__":
    unittest.main()
