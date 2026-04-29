import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory


class PostTtsHandoffPolicyTest(unittest.TestCase):
    def test_first_text_enqueue_inserts_startup_chime_once(self):
        from audio import speech_queue

        with TemporaryDirectory() as tmp:
            chime = Path(tmp) / "startup_chime.mp3"
            chime.write_bytes(b"fake")
            with (
                mock.patch.object(speech_queue._SpeechQueue, "_worker", lambda self: None),
                mock.patch("config.PLAY_LISTENING_CHIME", True),
                mock.patch("config.LISTENING_CHIME_FILE", str(chime)),
            ):
                queue = speech_queue._SpeechQueue()
                queue.enqueue("Hello there.", priority=1)
                queue.enqueue("Second line.", priority=1)

            queued = sorted(queue._heap, key=lambda item: item.seq)

        self.assertEqual(len(queued), 3)
        self.assertEqual(queued[0].tag, "system:first_listening_chime")
        self.assertEqual(queued[0].audio_path, str(chime))
        self.assertEqual(queued[1].text, "Hello there.")
        self.assertEqual(queued[2].text, "Second line.")

    def test_conversation_log_dedupes_same_rex_line_briefly(self):
        from utils import conv_log

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "conversation.log"
            with mock.patch.object(conv_log, "_LOG_PATH", log_path):
                conv_log.clear_dedupe_state()
                conv_log.log_rex("Bret, what mission are we pretending is important today?")
                conv_log.log_rex("  Bret, what mission are we pretending is important today?  ")

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertIn("REX", lines[0])
        conv_log.clear_dedupe_state()

    def test_conversation_log_dedupes_same_rex_line_after_blocking_tts_return(self):
        from utils import conv_log

        times = iter([100.0, 110.0])
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "conversation.log"
            with (
                mock.patch.object(conv_log, "_LOG_PATH", log_path),
                mock.patch.object(conv_log.time, "monotonic", side_effect=lambda: next(times)),
            ):
                conv_log.clear_dedupe_state()
                line = "Ah, Star Trek! Where humans boldly go where no one has gone before."
                conv_log.log_rex(line)
                conv_log.log_rex(line)

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        conv_log.clear_dedupe_state()

    def test_conversation_log_trims_to_debug_line_limit(self):
        from utils import conv_log

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "conversation.log"
            with (
                mock.patch.object(conv_log, "_LOG_PATH", log_path),
                mock.patch("config.DEBUG_MODE", True),
                mock.patch("config.CONVERSATION_LOG_DEBUG_MAX_LINES", 3),
            ):
                conv_log.clear_dedupe_state()
                for idx in range(5):
                    conv_log.log_heard("Bret", f"line {idx}")

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 3)
        self.assertIn("line 2", lines[0])
        self.assertIn("line 4", lines[-1])
        conv_log.clear_dedupe_state()

    def test_conversation_log_labels_unknown_speakers_explicitly(self):
        from utils import conv_log

        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "conversation.log"
            with mock.patch.object(conv_log, "_LOG_PATH", log_path):
                conv_log.log_heard(None, "hello")
                conv_log.log_heard("  ", "still me")
                conv_log.log_heard("Bret Benziger", "known voice")

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("HEARD | Unknown: hello", lines[0])
        self.assertIn("HEARD | Unknown: still me", lines[1])
        self.assertIn("HEARD | Bret Benziger: known voice", lines[2])

    def test_tts_speak_logs_spoken_text_to_conversation_log(self):
        import numpy as np
        from audio import tts

        cache_file = mock.MagicMock()
        cache_file.exists.return_value = True
        cache_file.name = "cached.mp3"

        with (
            mock.patch.object(tts, "_cache_path", return_value=cache_file),
            mock.patch.object(
                tts,
                "_read_audio",
                return_value=(np.zeros(80, dtype=np.float32), 16000),
            ),
            mock.patch.object(tts, "_play") as play,
            mock.patch.object(tts.conv_log, "log_rex") as log_rex,
        ):
            tts.speak("R3X sees WWII trivia.")

        log_rex.assert_called_once_with("R3X sees World War Two trivia.")
        play.assert_called_once()

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

    def test_post_tts_handoff_refreshes_idle_timer(self):
        from intelligence import interaction

        interaction._last_speech_at = 10.0
        with (
            mock.patch.object(interaction.time, "monotonic", return_value=50.0),
            mock.patch.object(interaction.stream, "flush"),
        ):
            interaction._apply_post_tts_handoff(
                "Long Star Trek answer complete.",
                source="test",
            )

        self.assertEqual(interaction._last_speech_at, 50.0)

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
    def test_latency_fillers_are_in_character_not_human_disfluencies(self):
        import config

        joined = " ".join(config.LATENCY_FILLER_LINES).lower()

        self.assertNotRegex(joined, r"\b(?:um+|uh+|hmm+)\b")
        self.assertTrue(
            any(
                phrase in joined
                for phrase in ("one sec", "processing", "recalibrating", "memory banks")
            )
        )

    def test_startup_solo_greeting_prompt_names_person_and_avoids_they_them(self):
        from intelligence import consciousness

        prompt = consciousness._build_startup_solo_greeting_prompt(
            "Bret",
            "You just started up and immediately see 'Bret'.",
        )

        self.assertIn("Greet Bret", prompt)
        self.assertIn("what are you up to today", prompt.lower())
        self.assertIn("what do you want to talk about", prompt.lower())
        self.assertIn("Pick one from this menu", prompt)
        self.assertIn("do not reuse the same wording every run", prompt)
        self.assertIn("do NOT call this one visible person 'they' or 'them'", prompt)

    def test_startup_known_greeting_pending_suppresses_generic_world_reactions(self):
        from intelligence import consciousness

        old_started = consciousness._process_started_mono
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._process_started_mono = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [
                    {"person_db_id": 1, "face_id": "Bret Benziger"},
                ]
            }

            self.assertTrue(
                consciousness._startup_known_greeting_pending(snapshot, now=110.0)
            )
            consciousness._greeted_this_session.add(1)
            self.assertFalse(
                consciousness._startup_known_greeting_pending(snapshot, now=110.0)
            )
        finally:
            consciousness._process_started_mono = old_started
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_proactive_speech_writes_conversation_log(self):
        from intelligence import consciousness

        class _Done:
            def wait(self):
                return None

        with (
            mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
            mock.patch("audio.speech_queue.enqueue", return_value=_Done()),
            mock.patch.object(consciousness.conv_log, "log_rex") as log_rex,
            mock.patch.object(consciousness, "note_rex_utterance"),
        ):
            ok = consciousness._speak_async(
                "Bret, what mission are we pretending is important today?",
                governed=False,
            )

        self.assertTrue(ok)
        log_rex.assert_called_once_with(
            "Bret, what mission are we pretending is important today?"
        )

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
        self.assertEqual(
            conversation_steering.detect_interest(
                "My favorite kind of ice cream is mint chocolate chip"
            ),
            "mint chocolate chip ice cream",
        )
        self.assertEqual(
            conversation_steering.detect_interest("Let's talk about Star Trek."),
            "Star Trek",
        )
        self.assertEqual(
            conversation_steering.detect_interest("I really want to talk about Star Trek."),
            "Star Trek",
        )
        self.assertIsNone(conversation_steering.detect_interest("I do not know."))

    def test_conversation_steering_detects_topic_knowledge_questions(self):
        from intelligence import conversation_steering

        self.assertEqual(
            conversation_steering.detect_topic_question("What do you know about Star Trek?"),
            "Star Trek",
        )
        self.assertEqual(
            conversation_steering.detect_topic_question("Do you know anything about droid building?"),
            "droid building",
        )

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
        self.assertIn("do not confuse franchises or fields", directive)
        add_fact.assert_any_call(
            1,
            "interest",
            "interest_astrophotography",
            "astrophotography",
            "interest_declaration",
            confidence=0.95,
        )
        conversation_steering.clear()

    def test_topic_question_steers_agenda_without_personal_memory_shrug(self):
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
        ):
            directive = conversation_agenda.build_turn_directive(
                "What do you know about Star Trek?",
                1,
            )

        self.assertIn("Conversation steering", directive)
        self.assertIn("Star Trek", directive)
        self.assertIn("answer from general knowledge first", directive)
        self.assertIn("general topic knowledge question", directive)
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

    def test_visual_curiosity_suppressed_during_active_interest_thread(self):
        from intelligence import consciousness, conversation_steering

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
            conversation_steering.note_user_turn(1, "I really like Star Trek")

        self.assertTrue(consciousness._visual_curiosity_blocked_by_interest_thread(1))
        conversation_steering.clear()

    def test_bare_startup_answer_becomes_interest_thread(self):
        from intelligence import conversation_steering, interaction

        conversation_steering.clear()
        pending = {
            "id": 7,
            "question_key": "startup_conversation_steering",
            "question_text": "What corner of your organic life are we discussing first?",
            "depth_level": 1,
        }
        answered = dict(pending, answer_text="Droid Development")
        with (
            mock.patch.object(
                interaction.rel_memory,
                "get_latest_pending_question",
                return_value=pending,
            ),
            mock.patch.object(
                interaction.rel_memory,
                "answer_latest_pending_question",
                return_value=answered,
            ),
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ) as add_fact,
        ):
            captured = interaction._maybe_capture_pending_qa(
                1,
                "Droid Development",
            )

        ctx = conversation_steering.build_context(1)
        self.assertEqual(captured["question_key"], "startup_conversation_steering")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.topic, "Droid Development")
        add_fact.assert_any_call(
            1,
            "interest",
            "interest_droid_development",
            "Droid Development",
            "startup_steering_answer",
            confidence=0.95,
        )
        conversation_steering.clear()

    def test_topic_thread_startup_answer_fallback_becomes_interest_thread(self):
        from intelligence import conversation_steering, interaction, topic_thread

        conversation_steering.clear()
        topic_thread.clear()
        topic_thread.note_assistant_turn(
            "Hey, Bret. What topic gets the honor of my extremely limited patience today?"
        )
        with (
            mock.patch.object(
                interaction.rel_memory,
                "answer_latest_pending_question",
                return_value=None,
            ),
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ) as add_fact,
        ):
            captured = interaction._maybe_capture_topic_thread_answer(
                1,
                "Star Trek",
            )

        ctx = conversation_steering.build_context(1)
        self.assertIsNotNone(captured)
        self.assertEqual(captured["question_key"], "startup_conversation_steering")
        self.assertEqual(captured["answer_text"], "Star Trek")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.topic, "Star Trek")
        add_fact.assert_any_call(
            1,
            "interest",
            "interest_star_trek",
            "Star Trek",
            "startup_thread_answer",
            confidence=0.95,
        )
        topic_thread.clear()
        conversation_steering.clear()

    def test_mind_opener_bare_topic_gets_startup_interest_budget(self):
        from intelligence import conversation_steering, interaction, response_length, topic_thread

        conversation_steering.clear()
        topic_thread.clear()
        topic_thread.note_assistant_turn(
            "Hey Bret, what's been rolling around in your mind today?"
        )
        with (
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ),
        ):
            captured = interaction._maybe_capture_topic_thread_answer(
                1,
                "Star Trek",
            )

        plan = response_length.classify("Star Trek", answered_question=captured)
        ctx = conversation_steering.build_context(1)

        self.assertIsNotNone(captured)
        self.assertEqual(captured["question_key"], "startup_conversation_steering")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.topic, "Star Trek")
        self.assertGreaterEqual(plan.max_words, 50)
        self.assertGreaterEqual(plan.max_sentences, 3)
        topic_thread.clear()
        conversation_steering.clear()

    def test_topic_thread_startup_correction_clears_question_without_interest(self):
        from intelligence import conversation_steering, interaction, topic_thread

        conversation_steering.clear()
        topic_thread.clear()
        topic_thread.note_assistant_turn(
            "Bret! Look who finally decided to grace us with their presence. "
            "What problem are we pretending I caused?"
        )
        with (
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch(
                "intelligence.conversation_steering.facts_memory.add_fact",
            ) as add_fact,
        ):
            captured = interaction._maybe_capture_topic_thread_answer(
                1,
                "You didn't cause any problem",
            )
            topic_thread.note_answered_question(captured)

        snap = topic_thread.snapshot()
        self.assertIsNotNone(captured)
        self.assertEqual(captured["question_key"], "startup_conversation_steering_reply")
        self.assertIsNone(conversation_steering.build_context(1))
        self.assertIsNone(snap.get("unresolved_question"))
        add_fact.assert_not_called()
        topic_thread.clear()
        conversation_steering.clear()

    def test_startup_answer_gets_room_for_followup_question(self):
        from intelligence import response_length

        plan = response_length.classify(
            "Droid Development",
            answered_question={"question_key": "startup_conversation_steering"},
        )

        self.assertEqual(plan.target, "short")
        self.assertGreaterEqual(plan.max_words, 50)
        self.assertGreaterEqual(plan.max_sentences, 3)
        self.assertIn("follow-up question", plan.instruction)
        self.assertIn("startup steering", plan.reason)

    def test_interest_declaration_gets_room_for_followup_question(self):
        from intelligence import response_length

        plan = response_length.classify("I really like Star Trek")

        self.assertEqual(plan.target, "short")
        self.assertGreaterEqual(plan.max_words, 40)
        self.assertGreaterEqual(plan.max_sentences, 2)
        self.assertIn("follow-up", plan.instruction)
        self.assertIn("topic interest", plan.reason)

    def test_interest_answer_to_startup_question_is_not_micro_shortened(self):
        from intelligence import response_length

        plan = response_length.classify(
            "I really like Star Trek",
            answered_question={"question_key": "startup_conversation_steering_reply"},
        )

        self.assertEqual(plan.target, "short")
        self.assertGreaterEqual(plan.max_words, 40)
        self.assertGreaterEqual(plan.max_sentences, 2)
        self.assertIn("topic interest", plan.reason)

    def test_topic_knowledge_question_gets_longer_budget(self):
        from intelligence import llm, response_length

        directive = response_length.build_directive(
            "What do you know about Star Trek?",
        )
        plan = response_length.classify("What do you know about Star Trek?")

        self.assertEqual(plan.target, "long")
        self.assertGreaterEqual(plan.max_words, 100)
        self.assertIn("general knowledge", plan.instruction)
        self.assertGreaterEqual(llm._max_tokens_for_agenda(directive), 200)

    def test_social_frame_preserves_allowed_interest_followup_when_trimming(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="interest",
            max_words=60,
            max_sentences=2,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=True,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Ah, Star Trek! Tiny starship sermon. What's your favorite corner of the Federation?",
            frame,
        )

        self.assertEqual(
            governed.text,
            "Ah, Star Trek! What's your favorite corner of the Federation?",
        )
        self.assertIn("trimmed_sentences", governed.notes)

    def test_presence_startup_question_is_saved_as_pending_qa(self):
        from intelligence import consciousness

        with mock.patch("memory.relationships.save_question_asked") as save:
            consciousness._record_proactive_question(
                1,
                "Hey there, Bret! What corner of your organic life are we discussing first?",
                label="first-sight greeting for Bret Benziger",
                purpose="presence_reaction",
            )

        save.assert_called_once_with(
            1,
            "startup_conversation_steering",
            "Hey there, Bret! What corner of your organic life are we discussing first?",
            1,
        )

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

    def test_social_frame_allows_interest_natural_followup_directive(self):
        from intelligence import social_frame

        directive = (
            "Conversation steering: The current thread matches a known/active "
            "interest: 'Star Trek'. Keep this turn steered toward that subject. "
            "Primary purpose: deepen the interest thread the human opened. "
            "Give one specific subject-aware reaction or tidbit, then ask one "
            "natural follow-up about their experience with that topic."
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
                "I like Star Trek Voyager",
                person_id=1,
                agenda_directive=directive,
            )
            governed = social_frame.govern_response(
                "Ah, Voyager! The show that proved even the most advanced "
                "starship can get lost in the space equivalent of a parking "
                "garage. What do you love most about it?",
                frame,
            )

        self.assertEqual(frame.purpose, "interest")
        self.assertTrue(frame.allow_question)
        self.assertGreaterEqual(frame.max_sentences, 2)
        self.assertIn("What do you love most about it?", governed.text)

    def test_interest_idle_followup_speaks_before_idle_timeout(self):
        from intelligence import conversation_steering, interaction

        conversation_steering.clear()
        interaction._session_person_ids.clear()
        interaction._interest_idle_followups_spoken.clear()
        interaction._session_person_ids.add(1)
        with (
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch("intelligence.conversation_steering.facts_memory.add_fact"),
        ):
            conversation_steering.note_user_turn(
                1,
                "My favorite kind of ice cream is mint chocolate chip",
            )

        with (
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.question_budget, "can_ask", return_value=True),
            mock.patch.object(
                interaction.llm,
                "get_response",
                return_value="Mint chocolate chip has main-character freezer energy. What makes it your pick?",
            ),
            mock.patch.object(
                interaction,
                "_speak_blocking",
                return_value=True,
            ) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript") as transcript,
            mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
            mock.patch.object(interaction.rel_memory, "save_question_asked") as save_q,
        ):
            spoken = interaction._maybe_interest_idle_followup(
                idle_for=13.0,
                effective_idle_timeout=30.0,
            )

        self.assertTrue(spoken)
        speak.assert_called_once()
        self.assertIn("mint chocolate", speak.call_args.args[0].lower())
        transcript.assert_called_once()
        log_rex.assert_called_once()
        save_q.assert_called_once()
        interaction._session_person_ids.clear()
        interaction._interest_idle_followups_spoken.clear()
        conversation_steering.clear()

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
        self.assertIn("what are you up to today", prompt.lower())
        self.assertIn("what do you want to talk about", prompt.lower())
        self.assertIn("Pick one from this menu", prompt)
        self.assertIn("do not reuse the same wording every run", prompt)
        self.assertIn("What mission are we pretending is important today?", prompt)

    def test_first_sight_mood_prompt_uses_high_confidence_expression(self):
        from intelligence import consciousness

        built = consciousness._build_first_sight_mood_prompt(
            "Bret",
            "You just started up and immediately see 'Bret'.",
            {"mood": "happy", "confidence": 0.9, "notes": "broad smile"},
        )

        self.assertIsNotNone(built)
        prompt, emotion = built
        self.assertEqual(emotion, "happy")
        self.assertIn("what's got them smiling", prompt)
        self.assertIn("apparent read", prompt)

    def test_first_sight_mood_prompt_ignores_low_confidence(self):
        from intelligence import consciousness

        built = consciousness._build_first_sight_mood_prompt(
            "Bret",
            "You just started up and immediately see 'Bret'.",
            {"mood": "sad", "confidence": 0.2, "notes": "unclear"},
        )

        self.assertIsNone(built)

    def test_group_smile_startup_prompt_only_for_two_happy_people(self):
        from intelligence import consciousness, social_scene

        scene = social_scene.SocialScene(
            known=(
                social_scene.VisiblePerson(1, "Bret Benziger", "Bret", "person_1"),
                social_scene.VisiblePerson(2, "Joy Example", "Joy", "person_2"),
            ),
            unknown_count=0,
            crowd_count=2,
        )

        prompt = consciousness._build_group_smile_startup_prompt(
            scene,
            [
                {"mood": "happy", "confidence": 0.9, "notes": "smiling"},
                {"mood": "happy", "confidence": 0.85, "notes": "grinning"},
            ],
        )

        self.assertIsNotNone(prompt)
        self.assertIn("both appear to be smiling", prompt)
        self.assertIn("what's got them both smiling", prompt)

    def test_group_smile_startup_prompt_ignores_three_person_room(self):
        from intelligence import consciousness, social_scene

        scene = social_scene.SocialScene(
            known=(
                social_scene.VisiblePerson(1, "Bret Benziger", "Bret", "person_1"),
                social_scene.VisiblePerson(2, "Joy Example", "Joy", "person_2"),
                social_scene.VisiblePerson(3, "JT Example", "JT", "person_3"),
            ),
            unknown_count=0,
            crowd_count=3,
        )

        prompt = consciousness._build_group_smile_startup_prompt(
            scene,
            [
                {"mood": "happy", "confidence": 0.9, "notes": "smiling"},
                {"mood": "happy", "confidence": 0.85, "notes": "grinning"},
            ],
        )

        self.assertIsNone(prompt)

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
        self.assertIn("what are you up to today", prompt.lower())
        self.assertIn("do not reuse the same wording every run", prompt)
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

    def test_router_downgrades_preference_misread_as_forget(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="memory.forget_specific",
            confidence=0.90,
            args={"target": "Disneyland"},
            reason="misread preference as forget request",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "I like Disneyland",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_allows_explicit_specific_forget_request(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="memory.forget_specific",
            confidence=0.90,
            args={"target": "Disneyland"},
            reason="explicit forget request",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Forget Disneyland from your memory",
            {},
        )

        self.assertEqual(routed.action, "memory.forget_specific")

    def test_router_downgrades_bare_sensitive_topic_as_boundary(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="emotional.boundary",
            confidence=0.90,
            args={},
            reason="misread health topic as boundary",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "back pain",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_allows_explicit_topic_boundary(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="emotional.boundary",
            confidence=0.90,
            args={"topic": "back pain"},
            reason="explicit boundary",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Please don't ask me about back pain again",
            {},
        )

        self.assertEqual(routed.action, "emotional.boundary")

    def test_router_downgrades_general_topic_knowledge_from_memory_query(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="memory.query",
            confidence=0.90,
            args={"person_name": "Star Trek"},
            reason="misread topic as memory target",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "What do you know about Star Trek?",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_keeps_person_memory_question_as_memory_query(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="memory.query",
            confidence=0.90,
            args={},
            reason="person memory question",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "What do you know about my dad?",
            {},
        )

        self.assertEqual(routed.action, "memory.query")

    def test_intent_classifier_short_circuits_topic_knowledge_questions(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("What do you know about Star Trek?"),
            "general",
        )
        self.assertEqual(intent_classifier.classify("Star Trek"), "general")

    def test_intent_classifier_keeps_contextual_followups_in_conversation(self):
        from intelligence import intent_classifier

        self.assertEqual(intent_classifier.classify("what about the tech?"), "general")
        self.assertEqual(intent_classifier.classify("and the transporters?"), "general")

    def test_intent_classifier_does_not_route_closure_to_tools(self):
        from intelligence import intent_classifier

        self.assertEqual(intent_classifier.classify("later."), "general")
        self.assertEqual(
            intent_classifier.classify("Well it was nice speaking, I'll talk to you later."),
            "general",
        )
        self.assertEqual(intent_classifier.classify("Goodbye"), "general")

    def test_intent_classifier_does_not_treat_weekday_statement_as_date_query(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("I'm going to Las Vegas on Thursday"),
            "general",
        )

    def test_intent_classifier_routes_capability_variants_deterministically(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("What sort of stuff are you good for?"),
            "query_capabilities",
        )
        self.assertEqual(
            intent_classifier.classify("what are you good at?"),
            "query_capabilities",
        )

    def test_intent_classifier_routes_self_memory_question_to_memory(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("Can you tell me about myself?"),
            "query_memory",
        )
        self.assertEqual(
            intent_classifier.classify("What are my plans for Thursday?"),
            "query_memory",
        )

    def test_intent_classifier_blocks_false_game_routes_for_star_trek_chat(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("I want to talk about Star Trek The Next Generation"),
            "general",
        )
        self.assertEqual(
            intent_classifier.classify("Star Trek Voyager, and Captain Janeway"),
            "general",
        )

    def test_thanks_for_asking_is_not_closure(self):
        from intelligence import end_thread, response_length

        end_thread.clear()
        closure = end_thread.note_user_turn("I'm doing okay, thanks for asking", 1)
        plan = response_length.classify("I'm doing okay, thanks for asking")

        self.assertIsNone(closure)
        self.assertNotEqual(plan.target, "micro")
        end_thread.clear()

    def test_social_frame_sentence_split_preserves_abbreviations(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="interest",
            max_words=125,
            max_sentences=7,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Star Trek started in 1966. The U.S.S. Enterprise explores strange new worlds. The tech is transporters, tricorders, and warp drive.",
            frame,
        )

        self.assertIn("U.S.S. Enterprise", governed.text)
        self.assertIn("The tech is transporters", governed.text)

    def test_social_frame_repairs_word_trimmed_fragments(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="answer",
            max_words=22,
            max_sentences=2,
            allow_question=False,
            allow_roast="none",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "I excel at spinning tracks that make even the most boring organics want to dance. Plus, I've mastered the art of delivering crushing roasts.",
            frame,
        )
        governed_whatever = social_frame.govern_response(
            'Well, the results are in: still the galaxy\'s best DJ, despite whatever this "test" is.',
            social_frame.SocialFrame(
                addressee="Bret",
                purpose="answer_ack",
                max_words=12,
                max_sentences=1,
                allow_question=False,
                allow_roast="none",
                allow_visual_comment=False,
                reason="test",
            ),
        )

        self.assertNotIn("delivering.", governed.text)
        self.assertTrue(governed.text in {"I hear you.", "Fair enough.", ""} or governed.text.endswith("dance."))
        self.assertNotIn("despite whatever.", governed_whatever.text)

    def test_agenda_does_not_inject_friendship_question_after_short_ack(self):
        from intelligence import conversation_agenda

        ws = {"people": [], "crowd": {}}
        with (
            mock.patch.object(conversation_agenda.world_state, "snapshot", return_value=ws),
            mock.patch.object(
                conversation_agenda.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret", "friendship_tier": "stranger"},
            ),
            mock.patch.object(conversation_agenda.rel_memory, "get_asked_question_keys", return_value=set()),
            mock.patch.object(conversation_agenda.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(conversation_agenda.empathy, "classify_local_sensitivity", return_value=None),
            mock.patch.object(conversation_agenda.empathy, "peek", return_value=None),
        ):
            directive = conversation_agenda.build_turn_directive(
                "It turned out totally cool",
                1,
            )

        self.assertNotIn("How did you end up talking to a droid DJ?", directive)
        self.assertIn("briefly acknowledge", directive)

    def test_agenda_does_not_inject_friendship_question_after_plan_statement(self):
        from intelligence import conversation_agenda

        ws = {"people": [], "crowd": {}}
        with (
            mock.patch.object(conversation_agenda.world_state, "snapshot", return_value=ws),
            mock.patch.object(conversation_agenda.empathy, "classify_local_sensitivity", return_value=None),
        ):
            directive = conversation_agenda.build_turn_directive(
                "I'm going to Las Vegas on Thursday",
                1,
            )

        self.assertIn("upcoming event", directive)
        self.assertNotIn("How did you end up talking to a droid DJ?", directive)

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

    def test_laughter_sound_event_reactions_are_disabled_by_default(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness

        old_snapshot = consciousness._last_snapshot
        profile = SituationProfile(
            conversation_active=False,
            user_mid_sentence=False,
            rapid_exchange=False,
            child_present=False,
            apparent_departure=False,
            likely_still_present=False,
            social_mode="one_on_one",
            suppress_proactive=False,
            suppress_system_comments=False,
            force_family_safe=False,
            being_discussed=False,
            discussion_sentiment="neutral",
            interaction_busy=False,
        )
        prev = {
            "crowd": {"count": 1, "count_label": "alone"},
            "audio_scene": {},
            "animals": [],
            "time": {},
        }
        curr = {
            "crowd": {"count": 1, "count_label": "alone"},
            "audio_scene": {"last_sound_event": "laughter"},
            "animals": [],
            "time": {},
        }
        try:
            consciousness._last_snapshot = prev
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak") as speak,
                mock.patch("config.WORLD_SOUND_EVENT_REACTIONS_ENABLED", False),
            ):
                consciousness._step_proactive_reactions(curr, profile)
        finally:
            consciousness._last_snapshot = old_snapshot

        speak.assert_not_called()


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
