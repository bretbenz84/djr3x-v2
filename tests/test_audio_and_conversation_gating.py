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

    def test_first_text_enqueue_skips_startup_chime_during_active_game(self):
        from audio import speech_queue

        with (
            mock.patch.object(speech_queue._SpeechQueue, "_worker", lambda self: None),
            mock.patch("features.games.is_active", return_value=True),
        ):
            queue = speech_queue._SpeechQueue()
            queue.enqueue("Who is playing Jeopardy?", priority=1)

        queued = sorted(queue._heap, key=lambda item: item.seq)
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].text, "Who is playing Jeopardy?")

    def test_begin_user_turn_keeps_game_prompts_queued(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=True),
            mock.patch.object(interaction.speech_queue, "clear_below_priority") as clear,
        ):
            interaction._begin_user_turn()

        clear.assert_not_called()

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

    def test_rhetorical_question_does_not_expect_no_response_recovery(self):
        from intelligence import interaction

        self.assertFalse(
            interaction._question_expects_response(
                "Ah, a wise choice! Why risk credits when you could just let "
                "the universe take your money for free?"
            )
        )
        self.assertTrue(
            interaction._question_expects_response(
                "Vegas without gambling. So what's actually on the agenda?"
            )
        )

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

    def test_no_response_recovery_is_suppressed_during_active_game(self):
        from intelligence import interaction

        with mock.patch(
            "features.games.suppresses_conversation_interruptions",
            return_value=True,
        ):
            self.assertFalse(
                interaction._should_no_response_recovery_fire(
                    asked_at=100.0,
                    now=108.0,
                    last_speech_at=100.0,
                )
            )

    def test_jeopardy_audio_is_interruptible_game_audio(self):
        from intelligence import interaction

        with mock.patch("features.games.is_active", return_value=True):
            self.assertTrue(
                interaction._is_interruptible_game_audio_path(
                    "/tmp/assets/audio/jeopardy/jeopardy-theme.mp3"
                )
            )
            self.assertFalse(
                interaction._is_interruptible_game_audio_path(
                    "/tmp/assets/audio/startup/startup_chime.mp3"
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

    def test_self_intro_name_stops_before_introducing_known_person(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._extract_self_identified_name(
                "My name is Jennifer Woodard and this is my brother Bret"
            ),
            "Jennifer Woodard",
        )

    def test_filler_is_not_a_valid_name_or_transcript(self):
        from audio import transcription
        from intelligence import interaction

        self.assertIsNone(interaction._extract_introduced_name("mmm", allow_bare_name=True))
        self.assertIsNone(interaction._extract_introduced_name("mmm wait", allow_bare_name=True))
        self.assertIsNone(interaction._extract_introduced_name("have you?", allow_bare_name=True))
        self.assertTrue(transcription._is_hallucination("mmm"))

    def test_intro_name_trims_trailing_greeting(self):
        from intelligence import introductions

        parsed = introductions.detect("This is my sister Jennifer Hi", has_unknown_face=True)

        self.assertTrue(parsed.is_introduction)
        self.assertEqual(parsed.name, "Jennifer")
        self.assertEqual(parsed.relationship, "sister")

    def test_jeopardy_roster_allows_four_players_and_jen_alias(self):
        from features import jeopardy
        from features import games
        from memory import people as people_memory

        self.assertEqual(
            jeopardy.parse_player_names("Will, Jen, Daniel, and Bret", limit=4),
            ["Will", "Jen", "Daniel", "Bret"],
        )

        def find_person(name):
            rows = {
                "Jennifer": {"id": 4, "name": "Jennifer Woodard"},
                "Daniel": {"id": 3, "name": "Daniel"},
                "Bret": {"id": 1, "name": "Bret Benziger"},
            }
            return rows.get(name)

        with (
            mock.patch.object(people_memory, "find_person_by_name", side_effect=find_person),
            mock.patch.object(people_memory, "find_or_create_person", return_value=(9, True)),
            mock.patch.object(people_memory, "has_voice_biometric", side_effect=lambda pid: pid != 4),
        ):
            players, needs_voice = games._jeopardy_prepare_players(["Will", "Jen", "Daniel", "Bret"])

        self.assertEqual([p["name"] for p in players], ["Will", "Jennifer", "Daniel", "Bret"])
        self.assertEqual(needs_voice, [1])

    def test_jeopardy_negative_scores_are_spoken_explicitly(self):
        from features import jeopardy

        scores = jeopardy.format_scores([
            {"name": "Bret", "score": -600},
            {"name": "Jennifer", "score": 200},
        ])

        self.assertIn("Bret: negative $600", scores)
        self.assertIn("Jennifer: $200", scores)
        self.assertNotIn("$-600", scores)

    def test_jeopardy_object_invention_response_uses_what(self):
        from features import jeopardy

        response = jeopardy.format_correct_response(
            "Windshield Wipers",
            clue="This invention let you drive in the rain",
        )

        self.assertEqual(response, "What are Windshield Wipers?")
        self.assertTrue(
            jeopardy.is_correct("What are windshield wipers?", "Windshield Wipers")
        )

    def test_self_intro_relationship_to_engaged_collapses_sibling_gender(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._extract_self_relationship_to_engaged(
                "My name is Jennifer Woodard and this is my brother Bret",
                "Bret Benziger",
            ),
            "sibling",
        )

    def test_enroll_unknown_face_refuses_largest_known_fallback(self):
        from vision import face

        fake_face = {
            "encoding": object(),
            "bounding_box": (0, 0, 100, 100),
        }
        with (
            mock.patch.object(face, "detect_faces", return_value=[fake_face]),
            mock.patch.object(face, "identify_face", return_value={"id": 1, "name": "Bret"}),
            mock.patch.object(face.people, "add_biometric") as add_biometric,
        ):
            ok = face.enroll_unknown_face(4, object())

        self.assertFalse(ok)
        add_biometric.assert_not_called()

    def test_active_wake_ack_suppressed_while_waiting_for_response(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.consciousness, "is_waiting_for_response", return_value=True),
        ):
            self.assertFalse(interaction._should_play_active_wake_ack())

    def test_active_wake_ack_allowed_when_idle_and_not_waiting(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.consciousness, "is_waiting_for_response", return_value=False),
        ):
            self.assertTrue(interaction._should_play_active_wake_ack())

    def test_idle_outro_speaks_once_before_session_returns_idle(self):
        from intelligence import interaction

        interaction._idle_outro_spoken = False
        with (
            mock.patch.object(interaction.config, "IDLE_OUTRO_ENABLED", True),
            mock.patch.object(interaction.config, "IDLE_OUTRO_LINES", ["Nobody talking now."]),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript") as transcript,
            mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
            mock.patch.object(interaction, "_register_rex_utterance") as register,
        ):
            first = interaction._maybe_idle_outro()
            second = interaction._maybe_idle_outro()

        self.assertTrue(first)
        self.assertFalse(second)
        speak.assert_called_once_with("Nobody talking now.", emotion="neutral", priority=1)
        transcript.assert_called_once_with("Rex", "Nobody talking now.")
        log_rex.assert_called_once_with("Nobody talking now.")
        register.assert_called_once_with("Nobody talking now.")
        interaction._idle_outro_spoken = False

    def test_low_memory_idle_question_asks_profile_question_once(self):
        from intelligence import interaction

        interaction._low_memory_idle_questions_spoken.clear()
        question = {"key": "job", "text": "What do you do — professionally speaking?", "depth": 1}
        spoken = "I don't know you well yet, Bret, What do you do — professionally speaking?"
        with (
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_SECS", 10.0),
            mock.patch.object(interaction.config, "LOW_MEMORY_PROFILE_MAX_FACTS", 4),
            mock.patch.object(
                interaction.config,
                "LOW_MEMORY_IDLE_QUESTION_PREFIX",
                "I don't know you well yet, {name}, {question}",
            ),
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_profile_fact_count", return_value=1),
            mock.patch.object(interaction, "_next_profile_question", return_value=question),
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret Benziger"},
            ),
            mock.patch.object(interaction.question_budget, "can_ask", return_value=True),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript") as transcript,
            mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
            mock.patch.object(interaction, "_register_rex_utterance") as register,
            mock.patch.object(interaction.rel_memory, "save_question_asked") as save_q,
        ):
            first = interaction._maybe_low_memory_idle_question(
                idle_for=11.0,
                effective_idle_timeout=30.0,
            )
            second = interaction._maybe_low_memory_idle_question(
                idle_for=20.0,
                effective_idle_timeout=30.0,
            )

        self.assertTrue(first)
        self.assertFalse(second)
        speak.assert_called_once_with(spoken, emotion="curious", priority=1)
        transcript.assert_called_once_with("Rex", spoken)
        log_rex.assert_called_once_with(spoken)
        register.assert_called_once_with(spoken)
        save_q.assert_called_once_with(1, "job", spoken, 1)
        interaction._low_memory_idle_questions_spoken.clear()

    def test_low_memory_question_prefix_uses_first_name(self):
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret Benziger"},
            ),
            mock.patch.object(
                interaction.config,
                "LOW_MEMORY_IDLE_QUESTION_PREFIX",
                "I don't know you well yet, {name}, {question}",
            ),
        ):
            line = interaction._format_low_memory_question(
                1,
                "So where are you from?",
            )

        self.assertEqual(line, "I don't know you well yet, Bret, So where are you from?")

    def test_low_memory_idle_question_skips_rich_profiles(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_profile_fact_count", return_value=12),
            mock.patch.object(interaction, "_next_profile_question") as next_q,
        ):
            asked = interaction._maybe_low_memory_idle_question(
                idle_for=20.0,
                effective_idle_timeout=30.0,
            )

        self.assertFalse(asked)
        next_q.assert_not_called()

    def test_wake_word_does_not_interrupt_current_question(self):
        from intelligence import interaction

        interaction._interrupted.clear()
        interaction._wake_word_fired.clear()
        try:
            with (
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=True),
                mock.patch.object(interaction.consciousness, "is_waiting_for_response", return_value=True),
            ):
                interaction._on_wake_word("Hey_rex")

            self.assertFalse(interaction._interrupted.is_set())
            self.assertTrue(interaction._wake_word_fired.is_set())
        finally:
            interaction._interrupted.clear()
            interaction._wake_word_fired.clear()

    def test_bare_wake_address_detection(self):
        from intelligence import interaction

        for text in ("Hey Rex", "hey dj-rex", "DJ Rex", "yo robot", "R3X"):
            self.assertTrue(interaction._is_bare_wake_address(text), text)

        self.assertFalse(interaction._is_bare_wake_address("Hey Rex what time is it"))
        self.assertFalse(interaction._is_bare_wake_address("Rex play jazz"))

    def test_bare_identity_name_rejects_filler_words(self):
        from intelligence import interaction

        for filler in ("both", "someone", "everybody", "whoever", "okay"):
            self.assertIsNone(
                interaction._extract_introduced_name(filler, allow_bare_name=True),
                filler,
            )
        self.assertEqual(
            interaction._extract_introduced_name("Bret", allow_bare_name=True),
            "Bret",
        )

    def test_name_update_extracts_common_corrections(self):
        from intelligence import interaction

        self.assertEqual(interaction._extract_name_update("Call me Bret instead"), "Bret")
        self.assertEqual(
            interaction._extract_name_update(
                "My name is BretMichael but you can call me Bret"
            ),
            "Bret",
        )
        self.assertEqual(
            interaction._extract_name_update("you got my name wrong, my name is Joe"),
            "Joe",
        )
        self.assertEqual(interaction._extract_name_update("rename me to JT"), "JT")
        self.assertIsNone(interaction._extract_name_update("call me both"))
        self.assertIsNone(interaction._extract_name_update("that's not my name"))

    def test_common_first_name_only_requires_last_name(self):
        from intelligence import interaction

        self.assertTrue(interaction._is_common_first_name_only("John"))
        self.assertTrue(interaction._is_common_first_name_only("Jennifer"))
        self.assertFalse(interaction._is_common_first_name_only("Bret"))
        self.assertFalse(interaction._is_common_first_name_only("John Smith"))

    def test_last_name_reply_extracts_last_name_or_full_name(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._extract_last_name_reply("Smith", "John"),
            "Smith",
        )
        self.assertEqual(
            interaction._extract_last_name_reply("my last name is Smith", "John"),
            "Smith",
        )
        self.assertEqual(
            interaction._extract_last_name_reply("John Smith", "John"),
            "Smith",
        )
        self.assertIsNone(interaction._extract_last_name_reply("John", "John"))

    def test_last_name_refusal_variations_are_recognized(self):
        from intelligence import interaction

        refusals = [
            "I'd rather not say",
            "I'm not telling you my last name",
            "you don't need my last name",
            "my last name is private",
            "none of your business",
            "first name only",
            "just John",
            "you can call me John",
        ]
        for text in refusals:
            self.assertTrue(
                interaction._is_last_name_refusal(text, "John"),
                text,
            )

    def test_common_first_name_pending_reply_enrolls_full_name(self):
        from intelligence import interaction
        import numpy as np

        interaction._pending_common_first_name_identity = {
            "first_name": "John",
            "audio": np.ones(16, dtype=np.float32),
            "asked_at": interaction.time.monotonic(),
            "prior_engagement": None,
        }
        try:
            with mock.patch.object(
                interaction,
                "_enroll_new_person",
                return_value=42,
            ) as enroll:
                response, person_id, full_name = (
                    interaction._handle_common_first_name_last_name_reply("Smith")
                )

            self.assertEqual(person_id, 42)
            self.assertEqual(full_name, "John Smith")
            self.assertIn("John Smith", response)
            enroll.assert_called_once()
            self.assertEqual(enroll.call_args.args[0], "John Smith")
            self.assertIsNone(interaction._pending_common_first_name_identity)
        finally:
            interaction._pending_common_first_name_identity = None

    def test_common_first_name_pending_refusal_enrolls_first_name_only(self):
        from intelligence import interaction
        import numpy as np

        interaction._pending_common_first_name_identity = {
            "first_name": "John",
            "audio": np.ones(16, dtype=np.float32),
            "asked_at": interaction.time.monotonic(),
            "prior_engagement": None,
        }
        try:
            with mock.patch.object(
                interaction,
                "_enroll_new_person",
                return_value=42,
            ) as enroll:
                response, person_id, full_name = (
                    interaction._handle_common_first_name_last_name_reply(
                        "you don't need my last name"
                    )
                )

            self.assertEqual(person_id, 42)
            self.assertEqual(full_name, "John")
            self.assertIn("John", response)
            enroll.assert_called_once()
            self.assertEqual(enroll.call_args.args[0], "John")
            self.assertIsNone(interaction._pending_common_first_name_identity)
        finally:
            interaction._pending_common_first_name_identity = None

    def test_common_first_name_introduction_defers_until_last_name(self):
        from intelligence import interaction

        parsed = interaction.introductions.IntroductionParse(
            is_introduction=True,
            name="Daniel",
            relationship="acquaintance",
            subject_kind="person",
        )
        interaction._pending_common_first_name_introduction = None
        try:
            with mock.patch.object(interaction, "_enroll_introduced_person") as enroll:
                response = interaction._handle_introduction_parse(
                    parsed,
                    introducer_id=1,
                    introducer_name="Bret Benziger",
                    visible_newcomer=True,
                )

            self.assertIn("Daniel", response)
            self.assertIsNotNone(interaction._pending_common_first_name_introduction)
            enroll.assert_not_called()

            with (
                mock.patch.object(
                    interaction,
                    "_enroll_introduced_person",
                    return_value=3,
                ) as enroll,
                mock.patch.object(
                    interaction,
                    "_intro_ack_and_followup",
                    return_value="Ack Daniel Smith.",
                ) as ack,
            ):
                completed = interaction._handle_common_first_name_intro_last_name_reply(
                    "Smith"
                )

            self.assertEqual(completed, "Ack Daniel Smith.")
            enroll.assert_called_once_with(
                "Daniel Smith",
                1,
                "Bret Benziger",
                "acquaintance",
                enroll_visible_face=True,
            )
            ack.assert_called_once()
            self.assertIsNone(interaction._pending_common_first_name_introduction)
        finally:
            interaction._pending_common_first_name_introduction = None

    def test_common_first_name_introduction_refusal_enrolls_first_name_only(self):
        from intelligence import interaction

        interaction._pending_common_first_name_introduction = {
            "first_name": "Daniel",
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "relationship": "acquaintance",
            "visible_newcomer": True,
            "subject_kind": "person",
            "asked_at": interaction.time.monotonic(),
        }
        try:
            with (
                mock.patch.object(
                    interaction,
                    "_enroll_introduced_person",
                    return_value=3,
                ) as enroll,
                mock.patch.object(
                    interaction,
                    "_intro_ack_and_followup",
                    return_value="Ack Daniel.",
                ) as ack,
            ):
                completed = interaction._handle_common_first_name_intro_last_name_reply(
                    "I'd rather not say"
                )

            self.assertEqual(completed, "Ack Daniel.")
            enroll.assert_called_once_with(
                "Daniel",
                1,
                "Bret Benziger",
                "acquaintance",
                enroll_visible_face=True,
            )
            ack.assert_called_once()
            self.assertIsNone(interaction._pending_common_first_name_introduction)
        finally:
            interaction._pending_common_first_name_introduction = None

    def test_returning_common_first_name_person_is_prompted_once(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = None
        interaction._common_first_name_prompted_this_session.clear()
        try:
            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=False,
            ):
                response = interaction._maybe_prompt_existing_common_first_name(
                    3,
                    "Daniel",
                )

            self.assertIn("Daniel", response)
            self.assertEqual(
                interaction._pending_existing_common_first_name["person_id"],
                3,
            )
            self.assertIn(3, interaction._common_first_name_prompted_this_session)

            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=False,
            ):
                second = interaction._maybe_prompt_existing_common_first_name(
                    3,
                    "Daniel",
                )
            self.assertIsNone(second)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()

    def test_returning_common_first_name_reply_renames_person(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 3,
            "first_name": "Daniel",
            "asked_at": interaction.time.monotonic(),
        }
        interaction._common_first_name_prompted_this_session.clear()
        try:
            with (
                mock.patch.object(
                    interaction.people_memory,
                    "rename_person",
                    return_value=True,
                ) as rename,
                mock.patch.object(interaction, "_refresh_world_state_person_name") as refresh,
            ):
                response = interaction._handle_existing_common_first_name_last_name_reply(
                    "Smith"
                )

            self.assertIn("Daniel Smith", response)
            rename.assert_called_once_with(3, "Daniel Smith")
            refresh.assert_called_once_with(3, "Daniel Smith")
            self.assertIsNone(interaction._pending_existing_common_first_name)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()

    def test_returning_common_first_name_refusal_is_remembered(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 3,
            "first_name": "Daniel",
            "asked_at": interaction.time.monotonic(),
        }
        interaction._common_first_name_prompted_this_session.clear()
        try:
            with mock.patch.object(
                interaction,
                "_remember_last_name_declined",
            ) as remember:
                response = interaction._handle_existing_common_first_name_last_name_reply(
                    "you don't need my last name"
                )

            self.assertIn("Daniel", response)
            remember.assert_called_once_with(3, "Daniel")
            self.assertIsNone(interaction._pending_existing_common_first_name)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()

    def test_returning_common_first_name_not_prompted_after_decline(self):
        from intelligence import interaction

        interaction._common_first_name_prompted_this_session.clear()
        try:
            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=True,
            ):
                response = interaction._maybe_prompt_existing_common_first_name(
                    3,
                    "Daniel",
                )

            self.assertIsNone(response)
            self.assertIsNone(interaction._pending_existing_common_first_name)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()

    def test_name_update_renames_current_person_and_world_state(self):
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction,
                "_resolve_name_update_target",
                return_value=(1, "Both"),
            ),
            mock.patch.object(interaction.people_memory, "rename_person", return_value=True) as rename,
            mock.patch.object(interaction, "_refresh_world_state_person_name") as refresh,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._handle_name_update_request(
                "Call me Bret instead",
                person_id=1,
                person_name="Both",
            )

        self.assertEqual(
            response,
            "Got it. I'll call you Bret. I'm sure we'll have better luck next time!",
        )
        rename.assert_called_once_with(1, "Bret")
        refresh.assert_called_once_with(1, "Bret")
        speak.assert_called_once()

    def test_repair_response_adds_better_luck_line_for_misunderstanding(self):
        from intelligence import interaction
        from intelligence import repair_moves

        repair_moves.clear()
        repair = {
            "kind": "misunderstood",
            "severity": "medium",
            "correction": "I meant the other playlist",
            "user_text": "No, you misunderstood me. I meant the other playlist.",
        }

        try:
            with (
                mock.patch.object(
                    interaction.llm,
                    "get_response",
                    return_value="Got it. I misunderstood the playlist request.",
                ),
                mock.patch.object(interaction, "_play_event_body_beat") as beat,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._generate_repair_response(1, repair["user_text"], repair)
        finally:
            repair_moves.clear()

        self.assertEqual(
            response,
            "Got it. I misunderstood the playlist request. "
            "I'm sure we'll have better luck next time!",
        )
        beat.assert_called_once_with("repair", repair_kind="misunderstood")
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], response)

    def test_memory_control_examples_parse_locally(self):
        from intelligence import command_parser

        examples = {
            "What do you remember about me?": "memory_review",
            "What do you remember about Daniel?": "memory_review",
            "Forget that Daniel likes horses.": "memory_forget_fact",
            "Forget that I like country music.": "memory_forget_fact",
            "That's wrong, Daniel's last name is Smith.": "memory_correct_fact",
            "Actually, call me Bret Michael.": "memory_correct_fact",
            "Don't remember that.": "memory_boundary",
            "Remember that Jennifer hates being called Jenny.": "memory_remember_fact",
        }

        for text, command_key in examples.items():
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, command_key)

    def test_forget_me_arms_exact_confirmation(self):
        from intelligence import interaction

        match = interaction.command_parser.CommandMatch("forget_me", "exact", {})
        interaction._clear_pending_memory_wipe()
        try:
            with mock.patch.object(interaction, "_speak_blocking") as speak:
                response = interaction._execute_command(match, 4, "Bret", "forget me")

            self.assertIn("yes forget me", response)
            self.assertEqual(interaction._pending_memory_wipe["scope"], "person")
            self.assertEqual(interaction._pending_memory_wipe["person_id"], 4)
            speak.assert_called_once()
        finally:
            interaction._clear_pending_memory_wipe()

    def test_yes_forget_me_executes_delete_person(self):
        from intelligence import interaction

        interaction._pending_memory_wipe = {
            "scope": "person",
            "person_id": 4,
            "person_name": "Bret Benziger",
            "requester_id": 4,
            "asked_at": 100.0,
        }
        interaction._session_person_ids.add(4)
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=105.0),
                mock.patch.object(interaction.people_memory, "delete_person") as delete_person,
                mock.patch.object(interaction.conv_memory, "clear_transcript") as clear_transcript,
                mock.patch.object(interaction, "_scrub_world_state_after_memory_wipe") as scrub,
                mock.patch.object(interaction.consciousness, "clear_engagement") as clear_engagement,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._handle_pending_memory_wipe_confirmation(
                    "yes forget me",
                    person_id=4,
                )

            self.assertIn("Confirmed", response)
            delete_person.assert_called_once_with(4)
            clear_transcript.assert_called_once()
            scrub.assert_called_once_with(person_id=4)
            clear_engagement.assert_called_once()
            speak.assert_called_once()
            self.assertIsNone(interaction._pending_memory_wipe)
            self.assertNotIn(4, interaction._session_person_ids)
        finally:
            interaction._session_person_ids.discard(4)
            interaction._clear_pending_memory_wipe()

    def test_forget_me_confirmation_rejects_different_known_speaker(self):
        from intelligence import interaction

        interaction._pending_memory_wipe = {
            "scope": "person",
            "person_id": 4,
            "person_name": "Bret",
            "requester_id": 4,
            "asked_at": 100.0,
        }
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=105.0),
                mock.patch.object(interaction.people_memory, "delete_person") as delete_person,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._handle_pending_memory_wipe_confirmation(
                    "yes forget me",
                    person_id=9,
                )

            self.assertIn("Confirmation rejected", response)
            delete_person.assert_not_called()
            speak.assert_called_once()
            self.assertIsNone(interaction._pending_memory_wipe)
        finally:
            interaction._clear_pending_memory_wipe()

    def test_confirm_full_wipe_executes_delete_all_people(self):
        from intelligence import interaction

        interaction._pending_memory_wipe = {
            "scope": "all",
            "person_id": None,
            "person_name": None,
            "requester_id": 4,
            "asked_at": 100.0,
        }
        interaction._session_person_ids.update({4, 5})
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=105.0),
                mock.patch.object(interaction.people_memory, "delete_all_people") as delete_all,
                mock.patch.object(interaction.conv_memory, "clear_transcript") as clear_transcript,
                mock.patch.object(interaction, "_scrub_world_state_after_memory_wipe") as scrub,
                mock.patch.object(interaction.consciousness, "clear_engagement") as clear_engagement,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._handle_pending_memory_wipe_confirmation(
                    "confirm full wipe",
                    person_id=4,
                )

            self.assertIn("Every person record", response)
            delete_all.assert_called_once_with()
            clear_transcript.assert_called_once()
            scrub.assert_called_once_with(all_people=True)
            clear_engagement.assert_called_once()
            speak.assert_called_once()
            self.assertIsNone(interaction._pending_memory_wipe)
            self.assertFalse(interaction._session_person_ids)
        finally:
            interaction._session_person_ids.clear()
            interaction._clear_pending_memory_wipe()

    def test_memory_forget_named_person_requires_explicit_name_match(self):
        from intelligence import interaction
        from memory.forgetting import ForgetResult

        result = ForgetResult(
            target="likes horses Daniel likes horses",
            terms={"horses"},
            deleted={"facts": 1, "preferences": 0, "interests": 0},
        )
        match = interaction.command_parser.CommandMatch(
            "memory_forget_fact",
            "pattern",
            {"statement": "Daniel likes horses"},
        )

        with (
            mock.patch.object(
                interaction.people_memory,
                "find_person_by_name",
                return_value={"id": 7, "name": "Daniel"},
            ) as find_person,
            mock.patch.object(
                interaction.forgetting,
                "forget_memory_detail",
                return_value=result,
            ) as forget_detail,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._execute_command(match, 1, "Bret", "Forget that Daniel likes horses.")

        self.assertIn("Deleted that memory for Daniel", response)
        find_person.assert_called_once_with("Daniel")
        forget_detail.assert_called_once()
        self.assertEqual(forget_detail.call_args.args[0], 7)
        speak.assert_called_once()

    def test_memory_correction_call_me_sets_corrected_identity_fact(self):
        from intelligence import interaction

        match = interaction.command_parser.CommandMatch(
            "memory_correct_fact",
            "pattern",
            {"correction": "call me Bret Michael"},
        )

        with (
            mock.patch.object(interaction.people_memory, "rename_person", return_value=True) as rename,
            mock.patch.object(interaction.facts_memory, "apply_fact_correction") as correct,
            mock.patch.object(interaction, "_refresh_world_state_person_name") as refresh,
            mock.patch.object(interaction, "_speak_blocking") as speak,
        ):
            response = interaction._execute_command(match, 4, "Bret", "Actually, call me Bret Michael.")

        self.assertIn("Bret Michael", response)
        self.assertIn("I'm sure we'll have better luck next time!", response)
        rename.assert_called_once_with(4, "Bret Michael")
        correct.assert_called_once()
        self.assertEqual(correct.call_args.args[:3], (4, "name", "Bret Michael"))
        self.assertEqual(correct.call_args.kwargs["category"], "identity")
        refresh.assert_called_once_with(4, "Bret Michael")
        speak.assert_called_once()

    def test_memory_boundary_discards_recent_candidate(self):
        from intelligence import interaction
        from memory.forgetting import ForgetResult

        interaction._recent_memory_candidates.clear()
        interaction._recent_memory_candidates.append({
            "person_id": 4,
            "kind": "preference",
            "target": "music likes country music",
            "label": "country music",
            "ts": 100.0,
        })
        result = ForgetResult(
            target="music likes country music",
            terms={"country", "music"},
            deleted={"facts": 0, "preferences": 1, "interests": 0},
        )
        match = interaction.command_parser.CommandMatch("memory_boundary", "pattern", {"scope": "recent"})

        try:
            with (
                mock.patch.object(
                    interaction.forgetting,
                    "forget_memory_detail",
                    return_value=result,
                ) as forget_detail,
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._execute_command(match, 4, "Bret", "Don't remember that.")

            self.assertIn("discarded the recent memory", response)
            forget_detail.assert_called_once_with(4, "music likes country music")
            self.assertFalse(interaction._recent_memory_candidates)
            speak.assert_called_once()
        finally:
            interaction._recent_memory_candidates.clear()

    def test_session_consolidation_json_mode_parses_expected_buckets(self):
        from types import SimpleNamespace
        from intelligence import llm

        payload = {
            "stable_facts": [
                {
                    "type": "fact",
                    "category": "job",
                    "key": "job_title",
                    "value": "pilot",
                    "confidence": 0.95,
                    "importance": 0.7,
                    "source": "explicit",
                    "decay_rate": "normal",
                    "rationale": "stated directly",
                }
            ],
            "preferences": [],
            "interests": [],
            "relationships": [],
            "events": [],
            "emotional_events": [],
            "discarded_noise": ["test phrase"],
            "corrections": [],
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=__import__("json").dumps(payload))
                )
            ]
        )

        with mock.patch.object(llm._client.chat.completions, "create", return_value=response) as create:
            result = llm.consolidate_session_memories(
                4,
                [{"speaker": "Bret", "text": "I work as a pilot."}],
                person_name="Bret",
                existing_memories={"facts": []},
                now_iso="2026-05-01T12:00:00+00:00",
            )

        self.assertEqual(result["stable_facts"][0]["key"], "job_title")
        self.assertEqual(result["discarded_noise"], ["test phrase"])
        self.assertEqual(create.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_write_consolidated_memory_routes_to_existing_modules(self):
        from intelligence import interaction

        consolidated = {
            "stable_facts": [
                {
                    "category": "job",
                    "key": "job_title",
                    "value": "mechanic",
                    "confidence": 0.95,
                    "importance": 0.7,
                    "source": "explicit",
                    "decay_rate": "normal",
                }
            ],
            "preferences": [
                {
                    "domain": "music",
                    "preference_type": "dislikes",
                    "key": "country",
                    "value": "dislikes country music",
                    "confidence": 0.95,
                    "importance": 0.8,
                    "source": "explicit",
                }
            ],
            "interests": [
                {
                    "name": "3D printing",
                    "category": "technical",
                    "interest_strength": "high",
                    "confidence": 0.9,
                    "source": "explicit",
                }
            ],
            "relationships": [
                {
                    "other_person_name": "Daniel",
                    "relationship": "friend",
                }
            ],
            "events": [
                {
                    "event_name": "camping trip",
                    "event_date": "2026-06-01",
                    "event_notes": "Going camping.",
                }
            ],
            "emotional_events": [
                {
                    "category": "good_news",
                    "description": "got promoted",
                    "valence": 0.8,
                }
            ],
            "discarded_noise": ["hello hello"],
            "corrections": [
                {
                    "target": "fact",
                    "category": "identity",
                    "key": "last_name",
                    "value": "Smith",
                }
            ],
        }

        with (
            mock.patch.object(interaction.facts_memory, "apply_fact_correction") as correct,
            mock.patch.object(interaction.facts_memory, "add_fact") as add_fact,
            mock.patch.object(interaction.preferences_memory, "upsert_preference") as pref,
            mock.patch.object(interaction.interests_memory, "upsert_interest") as interest,
            mock.patch.object(interaction.people_memory, "find_or_create_person", return_value=(7, False)),
            mock.patch.object(interaction.social_memory, "save_relationship") as rel,
            mock.patch.object(interaction.events_memory, "get_open_events", return_value=[]),
            mock.patch.object(interaction.events_memory, "add_event") as event,
            mock.patch.object(interaction.emotional_events, "add_event") as emotional,
        ):
            counts = interaction._write_consolidated_memory(
                4,
                "Bret",
                consolidated,
                forgotten_terms=set(),
            )

        correct.assert_called_once()
        add_fact.assert_called_once()
        pref.assert_called_once()
        interest.assert_called_once()
        rel.assert_called_once_with(4, 7, "friend", described_by=4)
        event.assert_called_once()
        emotional.assert_called_once()
        self.assertEqual(counts["stored"], 6)
        self.assertEqual(counts["updated"], 1)
        self.assertEqual(counts["skipped"], 1)

    def test_idle_background_speech_ignored_when_unrecognized_and_off_camera(self):
        from intelligence import interaction

        self.assertTrue(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=None,
                has_unknown_visible=False,
                identity_prompt_active=False,
                text="and there was no imminent threat.",
            )
        )

    def test_idle_background_speech_not_ignored_for_known_or_visible_unknown_contexts(self):
        from intelligence import interaction

        self.assertFalse(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=1,
                has_unknown_visible=False,
                identity_prompt_active=False,
                text="hello there",
            )
        )
        self.assertFalse(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=None,
                has_unknown_visible=True,
                identity_prompt_active=False,
                text="hello there",
            )
        )
        self.assertFalse(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=None,
                has_unknown_visible=False,
                identity_prompt_active=True,
                text="Bret",
            )
        )

    def test_vad_barge_in_is_disabled_by_default(self):
        import config
        from intelligence import interaction

        self.assertFalse(config.VAD_BARGE_IN_ENABLED)
        self.assertFalse(interaction._vad_barge_in_enabled())


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

    def test_startup_known_greeting_pending_suppresses_idle_micro_behavior(self):
        from intelligence import consciousness
        from state import State

        old_started = consciousness._process_started_mono
        old_greeted = set(consciousness._greeted_this_session)
        old_micro = consciousness._last_micro_behavior_at
        try:
            consciousness._process_started_mono = 100.0
            consciousness._greeted_this_session.clear()
            consciousness._last_micro_behavior_at = 0.0
            snapshot = {
                "people": [
                    {"person_db_id": 1, "face_id": "Bret Benziger"},
                ],
                "self_state": {"last_interaction_ago": 999.0},
            }
            with (
                mock.patch.object(consciousness.state_module, "get_state", return_value=State.IDLE),
                mock.patch.object(consciousness, "is_waiting_for_response", return_value=False),
                mock.patch.object(consciousness.time, "monotonic", return_value=120.0),
                mock.patch.object(consciousness.random, "uniform", return_value=0.0),
                mock.patch.object(consciousness, "_do_private_thought") as thought,
            ):
                consciousness._step_idle_micro_behavior(
                    snapshot,
                    mock.Mock(suppress_proactive=False, suppress_system_comments=False),
                )

            thought.assert_not_called()
        finally:
            consciousness._process_started_mono = old_started
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)
            consciousness._last_micro_behavior_at = old_micro

    def test_startup_window_suppresses_idle_micro_behavior_before_any_greeting(self):
        from intelligence import consciousness
        from state import State

        old_started = consciousness._process_started_mono
        old_greeted = set(consciousness._greeted_this_session)
        old_micro = consciousness._last_micro_behavior_at
        try:
            consciousness._process_started_mono = 100.0
            consciousness._greeted_this_session.clear()
            consciousness._last_micro_behavior_at = 0.0
            snapshot = {
                "people": [],
                "self_state": {"last_interaction_ago": 999.0},
            }
            profile = mock.Mock(
                suppress_proactive=False,
                suppress_system_comments=False,
            )
            with (
                mock.patch.object(consciousness.state_module, "get_state", return_value=State.IDLE),
                mock.patch.object(consciousness, "is_waiting_for_response", return_value=False),
                mock.patch.object(consciousness.time, "monotonic", return_value=120.0),
                mock.patch.object(consciousness.random, "uniform", return_value=0.0),
                mock.patch.object(consciousness, "_do_small_talk_question") as small_talk,
            ):
                consciousness._step_idle_micro_behavior(snapshot, profile)

            small_talk.assert_not_called()
        finally:
            consciousness._process_started_mono = old_started
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)
            consciousness._last_micro_behavior_at = old_micro

    def test_idle_micro_behavior_choices_include_empty_room_jokes_when_alone(self):
        from intelligence import consciousness

        choices, weights = consciousness._idle_micro_behavior_choices({
            "people": [],
            "crowd": {"count": 0},
        })

        self.assertIn("empty_room_joke", choices)
        self.assertNotIn("people_roast", choices)
        self.assertGreater(weights[choices.index("empty_room_joke")], 1)

    def test_empty_room_joke_speaks_local_self_deprecation(self):
        from intelligence import consciousness

        snapshot = {"people": [], "crowd": {"count": 0}}
        with (
            mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
            mock.patch.object(consciousness.random, "random", return_value=0.0),
            mock.patch.object(consciousness.random, "choice", return_value="Empty room test line."),
            mock.patch.object(consciousness, "_claim_proactive_purpose", return_value="tok"),
            mock.patch.object(consciousness, "_proactive_purpose_current", return_value=True),
            mock.patch.object(consciousness, "_release_proactive_purpose") as release,
            mock.patch.object(consciousness, "_speak_async") as speak,
        ):
            consciousness._do_empty_room_joke(snapshot)

        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "Empty room test line.")
        self.assertEqual(speak.call_args.kwargs["label"], "empty-room joke")
        release.assert_called_once_with("tok")

    def test_people_roast_prompt_stays_non_sensitive(self):
        from intelligence import consciousness

        snapshot = {
            "people": [
                {
                    "person_db_id": 1,
                    "face_id": "Bret Benziger",
                    "pose": "standing",
                    "gesture": "hands_on_hips",
                    "engagement": "low",
                }
            ],
            "crowd": {"count": 1},
        }
        with (
            mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
            mock.patch.object(consciousness.random, "random", return_value=0.0),
            mock.patch.object(consciousness.random, "choice", return_value=snapshot["people"][0]),
            mock.patch.object(consciousness, "is_engaged_with", return_value=False),
            mock.patch.object(consciousness, "_person_roast_allowed", return_value=True),
            mock.patch.object(consciousness, "_generate_and_speak") as generate,
        ):
            consciousness._do_people_roast(snapshot)

        generate.assert_called_once()
        prompt = generate.call_args.args[0]
        self.assertIn("Make one short playful Rex joke or light roast", prompt)
        self.assertIn("Do NOT joke about body, age, gender", prompt)
        self.assertEqual(generate.call_args.kwargs["purpose"], "people_roast")

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

    def test_incomplete_pending_answer_is_not_captured(self):
        from intelligence import interaction

        with mock.patch.object(
            interaction.rel_memory,
            "answer_latest_pending_question",
            return_value={"question_key": "interest_star_trek_voyager_idle_followup"},
        ) as answer:
            captured = interaction._maybe_capture_pending_qa(1, "I like")

        self.assertIsNone(captured)
        answer.assert_not_called()

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

    def test_topic_thread_short_confirmation_answers_tag_question(self):
        from intelligence import topic_thread

        topic_thread.clear()
        topic_thread.note_assistant_turn(
            "Bret, don't let the neon lights of Vegas fry your circuits; "
            "your trip's tomorrow, right?"
        )
        topic_thread.note_user_turn("Yeah", person_id=1)

        snap = topic_thread.snapshot()
        self.assertIsNotNone(snap)
        self.assertIsNone(snap.get("unresolved_question"))
        self.assertEqual(snap.get("user_stance"), "engaged")
        topic_thread.clear()

    def test_topic_thread_short_confirmation_does_not_answer_open_question(self):
        from intelligence import topic_thread

        topic_thread.clear()
        topic_thread.note_assistant_turn("Where are you off to, anyway?")
        topic_thread.note_user_turn("Yeah", person_id=1)

        snap = topic_thread.snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.get("unresolved_question"), "Where are you off to, anyway?")
        topic_thread.clear()

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

    def test_social_frame_does_not_shorten_allowed_interest_followup_by_default(self):
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
            "Ah, Star Trek! Tiny starship sermon. What's your favorite corner of the Federation?",
        )
        self.assertNotIn("trimmed_sentences", governed.notes)

    def test_social_frame_salvages_banter_before_removed_question(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="JT",
            purpose="identity",
            max_words=32,
            max_sentences=2,
            allow_question=False,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Ah, JT! Welcome to this wild ride of banter. Got any juicy tales, "
            "or just here to soak up the snark like a soggy towel?",
            frame,
        )

        self.assertEqual(
            governed.text,
            "Ah, JT! Welcome to this wild ride of banter.",
        )
        self.assertIn("removed_question", governed.notes)

    def test_social_frame_drops_disallowed_question_without_fragment_salvage(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="quiet",
            max_words=36,
            max_sentences=2,
            allow_question=False,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Bret Benziger, huh? Sounds like a name that could get stuck in hyperspace. "
            "So, where are you from — some planet where everyone just has impressive names?",
            frame,
        )

        self.assertEqual(
            governed.text,
            "Sounds like a name that could get stuck in hyperspace.",
        )
        self.assertIn("removed_question", governed.notes)
        self.assertNotIn("salvaged_question_lead", governed.notes)
        self.assertNotIn("where are you from", governed.text.lower())

    def test_social_frame_generic_question_budget_does_not_invite_interview_pivot(self):
        from intelligence import social_frame

        directive = (
            "Treat the user's latest utterance as a likely answer if it fits; "
            "do not ask an unrelated new question in the same breath. "
            "Response shape: Ask at most one, and only if it naturally serves this turn."
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
                "I'm from Waterford",
                person_id=1,
                agenda_directive=directive,
            )

        self.assertFalse(frame.allow_question)

    def test_social_frame_generic_primary_purpose_followup_does_not_invite_pivot(self):
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
                "I'm sure she'll pick the music",
                person_id=1,
                agenda_directive=directive,
            )

        self.assertFalse(frame.allow_question)

    def test_social_frame_keeps_only_one_allowed_question(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="interest",
            max_words=60,
            max_sentences=3,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Mischief, huh? What's on the menu for your taste buds? "
            "What do you do professionally?",
            frame,
        )

        self.assertEqual(governed.text.count("?"), 1)
        self.assertIn("What's on the menu", governed.text)
        self.assertNotIn("professionally", governed.text)
        self.assertIn("removed_extra_questions", governed.notes)

    def test_social_frame_converts_short_rhetorical_question_opener(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="interest",
            max_words=60,
            max_sentences=3,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Fun in Vegas? Bold strategy. What are you actually doing there?",
            frame,
        )

        self.assertEqual(governed.text.count("?"), 1)
        self.assertIn("Fun in Vegas.", governed.text)
        self.assertIn("What are you actually doing there?", governed.text)

    def test_social_frame_keeps_real_short_question_opener(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="interest",
            max_words=60,
            max_sentences=3,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "What now? Bold strategy. Another question?",
            frame,
        )

        self.assertEqual(governed.text.count("?"), 1)
        self.assertIn("What now?", governed.text)
        self.assertNotIn("Another question?", governed.text)

    def test_actor_harness_strips_speaker_prefixes(self):
        from tools import conversation_text_harness

        self.assertEqual(
            conversation_text_harness._clean_actor_reply(
                "Bret Benziger: I am just testing this thing.",
                "Bret Benziger",
            ),
            "I am just testing this thing.",
        )
        self.assertEqual(
            conversation_text_harness._clean_actor_reply(
                "Human: Yeah, that question got weird.",
                "Bret",
            ),
            "Yeah, that question got weird.",
        )

    def test_social_frame_keeps_tiny_opener_with_next_sentence(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="JT",
            purpose="identity",
            max_words=32,
            max_sentences=1,
            allow_question=False,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Ah, JT! Welcome to this wild ride of banter.",
            frame,
        )

        self.assertEqual(
            governed.text,
            "Ah, JT! Welcome to this wild ride of banter.",
        )
        self.assertNotIn("trimmed_sentences", governed.notes)

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

    def test_proactive_speech_is_suppressed_during_active_game(self):
        from intelligence import consciousness

        with mock.patch(
            "features.games.suppresses_conversation_interruptions",
            return_value=True,
        ):
            self.assertFalse(consciousness._can_proactive_speak())

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
            source="corrected",
            confidence=1.0,
            importance=0.9,
            decay_rate=None,
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

    def test_social_frame_generic_related_followup_directive_does_not_invite_pivot(self):
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

        self.assertFalse(frame.allow_question)

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

    def test_interest_idle_followup_is_suppressed_during_active_game(self):
        from intelligence import interaction

        with (
            mock.patch(
                "features.games.suppresses_conversation_interruptions",
                return_value=True,
            ),
            mock.patch.object(interaction.llm, "get_response") as get_response,
        ):
            spoken = interaction._maybe_interest_idle_followup(
                idle_for=30.0,
                effective_idle_timeout=60.0,
            )

        self.assertFalse(spoken)
        get_response.assert_not_called()

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
            mock.patch.object(llm.facts_db, "get_prompt_worthy_facts", return_value=[]),
            mock.patch.object(llm.preferences_db, "get_preferences_for_prompt", return_value=[]),
            mock.patch.object(llm.interests_db, "get_interests_for_prompt", return_value=[]),
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

    def test_person_context_injects_preferences_and_boundaries(self):
        from intelligence import llm

        with (
            mock.patch.object(
                llm.people_db,
                "get_person",
                return_value={
                    "id": 1,
                    "name": "Bret",
                    "friendship_tier": "friend",
                    "warmth_score": 0.2,
                    "antagonism_score": 0.0,
                    "trust_score": 0.7,
                    "net_relationship_score": 0.2,
                },
            ),
            mock.patch.object(llm.facts_db, "get_prompt_worthy_facts", return_value=[]),
            mock.patch.object(
                llm.preferences_db,
                "get_preferences_for_prompt",
                return_value=[
                    {
                        "id": 1,
                        "domain": "music",
                        "preference_type": "dislikes",
                        "key": "country",
                        "value": "dislikes country music",
                    },
                    {
                        "id": 2,
                        "domain": "interaction",
                        "preference_type": "boundary",
                        "key": "last_name_ask",
                        "value": "do not ask for their last name",
                    },
                ],
            ),
            mock.patch.object(
                llm.interests_db,
                "get_interests_for_prompt",
                return_value=[
                    {
                        "id": 3,
                        "name": "Star Wars",
                        "category": "fandom",
                        "interest_strength": "high",
                        "last_mentioned_at": "2026-04-30T12:00:00+00:00",
                        "cooldown_active": False,
                    },
                    {
                        "id": 4,
                        "name": "3D printing",
                        "category": "technical",
                        "interest_strength": "high",
                        "last_mentioned_at": "2026-04-30T12:00:00+00:00",
                        "ask_cooldown_until": "2026-05-30T12:00:00+00:00",
                        "cooldown_active": True,
                    },
                ],
            ),
            mock.patch.object(llm.conv_db, "get_last_conversation", return_value=None),
            mock.patch.object(llm.boundaries_db, "summarize_for_prompt", return_value=""),
            mock.patch.object(llm.rel_db, "get_next_question", return_value=None),
            mock.patch("memory.social.summarize_for_prompt", return_value=""),
            mock.patch("memory.emotional_events.summarize_for_prompt", return_value=""),
            mock.patch.object(llm, "_pick_stale_fact", return_value=None),
            mock.patch.object(llm, "_pick_nostalgia_callback", return_value=None),
        ):
            context = llm._build_person_context(1)

        self.assertIn("Preferences: music.dislikes: dislikes country music.", context)
        self.assertIn("Preference boundaries: interaction.boundary: do not ask for their last name.", context)
        self.assertIn("never as joke or roast material", context)
        self.assertIn("Interest profile: Star Wars, high interest, last mentioned 2026-04-30", context)
        self.assertIn("3D printing, high interest, last mentioned 2026-04-30, ask cooldown active until 2026-05-30", context)
        self.assertIn("Do not ask basic 'do you like X?'", context)

    def test_preference_upsert_forces_boundary_importance(self):
        from memory import preferences

        with (
            mock.patch.object(preferences.db, "fetchone", return_value=None),
            mock.patch.object(preferences.db, "execute", return_value=9) as execute,
        ):
            row_id = preferences.upsert_preference(
                3,
                "interaction",
                "boundary",
                "last name ask",
                "do not ask for their last name",
                importance=0.2,
            )

        self.assertEqual(row_id, 9)
        params = execute.call_args.args[1]
        self.assertEqual(params[1], "interaction")
        self.assertEqual(params[2], "boundary")
        self.assertEqual(params[3], "last_name_ask")
        self.assertGreaterEqual(params[6], 0.95)

    def test_interest_upsert_and_mark_asked(self):
        from memory import interests

        with (
            mock.patch.object(interests.db, "fetchone", return_value=None),
            mock.patch.object(interests.db, "execute", return_value=11) as execute,
        ):
            row_id = interests.upsert_interest(
                3,
                "3D printing",
                "technical",
                "high",
                notes="prints droid brackets",
            )

        self.assertEqual(row_id, 11)
        params = execute.call_args.args[1]
        self.assertEqual(params[1], "3D printing")
        self.assertEqual(params[2], "technical")
        self.assertEqual(params[3], "high")
        self.assertEqual(params[8], "prints droid brackets")

        with mock.patch.object(interests.db, "execute") as execute:
            interests.mark_interest_asked(3, "3D printing", cooldown_days=30)

        args = execute.call_args.args
        self.assertIn("ask_cooldown_until", args[0])
        self.assertEqual(args[1][2], 3)
        self.assertEqual(args[1][3], "3D printing")

    def test_fact_defaults_explicit_inferred_and_corrected_metadata(self):
        from memory import facts

        with (
            mock.patch.object(facts.db, "fetchone", return_value=None),
            mock.patch.object(facts.db, "execute") as execute,
        ):
            facts.add_fact(1, "job", "job_title", "pilot", "explicit")

        params = execute.call_args.args[1]
        self.assertEqual(params[4], 0.95)
        self.assertEqual(params[5], "explicit")
        self.assertEqual(params[10], 0.5)
        self.assertEqual(params[11], "normal")
        self.assertEqual(params[12], 365)
        self.assertIsNone(params[13])

        with (
            mock.patch.object(facts.db, "fetchone", return_value=None),
            mock.patch.object(facts.db, "execute") as execute,
        ):
            facts.add_fact(1, "other", "maybe_likes_noise", "likes noise", "inferred")

        params = execute.call_args.args[1]
        self.assertEqual(params[4], 0.55)
        self.assertEqual(params[5], "inferred")
        self.assertEqual(params[10], 0.35)
        self.assertEqual(params[11], "fast")

        with mock.patch.object(facts, "add_fact") as add_fact:
            facts.apply_fact_correction(1, "favorite_music", "jazz", category="preference")

        kwargs = add_fact.call_args.kwargs
        self.assertEqual(kwargs["source"], "corrected")
        self.assertEqual(kwargs["confidence"], 1.0)
        self.assertEqual(kwargs["importance"], 0.9)

    def test_fact_prompt_format_hedges_inferred_and_scores_overuse(self):
        from memory import facts

        inferred = facts._annotate_fact(
            {
                "id": 1,
                "category": "other",
                "key": "camping",
                "value": "camping might be their thing",
                "confidence": 0.55,
                "source": "inferred",
                "importance": 0.35,
                "decay_rate": "fast",
                "created_at": "2026-04-01T00:00:00+00:00",
                "updated_at": "2026-04-01T00:00:00+00:00",
                "last_confirmed_at": "2026-04-01T00:00:00+00:00",
                "last_used_at": None,
                "stale_after_days": 30,
                "evidence_count": 1,
            }
        )
        rendered = facts.format_fact_for_prompt(inferred)

        self.assertIn("inferred; hedge this", rendered)
        self.assertLess(facts.score_fact_for_prompt(inferred), 0.5)

        corrected = facts._annotate_fact(
            {
                "id": 2,
                "category": "preference",
                "key": "favorite_music",
                "value": "jazz",
                "confidence": 1.0,
                "source": "corrected",
                "importance": 0.9,
                "decay_rate": "normal",
                "created_at": "2026-04-01T00:00:00+00:00",
                "updated_at": "2026-04-01T00:00:00+00:00",
                "last_confirmed_at": "2026-04-01T00:00:00+00:00",
                "last_used_at": None,
                "stale_after_days": 365,
                "evidence_count": 1,
            }
        )

        self.assertGreater(facts.score_fact_for_prompt(corrected), facts.score_fact_for_prompt(inferred))
        self.assertIn("corrected by the person", facts.format_fact_for_prompt(corrected))

    def test_curiosity_uses_known_interest_hooks_before_basic_pool(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.random, "random", return_value=0.0),
            mock.patch.object(interaction.question_budget, "can_ask", return_value=True),
            mock.patch.object(interaction.empathy, "peek", return_value=None),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(interaction.conversation_steering, "build_context", return_value=None),
            mock.patch.object(
                interaction.interests_memory,
                "get_interest_hooks",
                return_value=[{"name": "Star Wars", "notes": "building a droid"}],
            ),
            mock.patch.object(
                interaction.llm,
                "get_response",
                return_value="Still working on that droid build, or has it finally achieved sentience?",
            ) as get_response,
            mock.patch.object(interaction.interests_memory, "mark_interest_asked") as mark_asked,
            mock.patch.object(interaction, "_speak_blocking") as speak,
            mock.patch.object(interaction.rel_memory, "save_question_asked"),
        ):
            question = interaction._curiosity_check(
                "Nice. Filed under excellent bad ideas.",
                "I like Star Wars.",
                1,
                "Bret",
            )

        self.assertIn("droid build", question)
        self.assertIn("Do not ask whether they like it", get_response.call_args.args[0])
        mark_asked.assert_called_once_with(1, "Star Wars")
        speak.assert_called_once_with(question)

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

    def test_agenda_acknowledges_offscreen_correction_without_topic_pivot(self):
        from intelligence import conversation_agenda, conversation_steering

        conversation_steering.clear()
        with (
            mock.patch.object(
                conversation_agenda.world_state,
                "snapshot",
                return_value={"crowd": {"count": 0}, "people": [], "environment": {}},
            ),
            mock.patch.object(conversation_agenda.rel_memory, "get_latest_pending_question", return_value=None),
            mock.patch.object(conversation_agenda, "_next_useful_question", return_value={"text": "What do you do when you're not wandering into cantinas?"}),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            directive = conversation_agenda.build_turn_directive(
                "I'm still here. I'm just out of view of you.",
                1,
            )

        self.assertIn("still present but out of camera view", directive)
        self.assertIn("there they are", directive)
        self.assertIn("no generic friendship question", directive)
        self.assertNotIn("wandering into cantinas", directive)

    def test_agenda_health_resolved_deescalates_without_new_question(self):
        from intelligence import conversation_agenda

        with (
            mock.patch.object(
                conversation_agenda.world_state,
                "snapshot",
                return_value={"crowd": {"count": 1}, "people": [], "environment": {}},
            ),
            mock.patch.object(conversation_agenda.rel_memory, "get_latest_pending_question", return_value=None),
            mock.patch("intelligence.question_budget.can_ask", return_value=True),
            mock.patch("intelligence.question_budget.build_directive", return_value=""),
        ):
            directive = conversation_agenda.build_turn_directive(
                "My back pain is mostly gone now",
                1,
            )

        self.assertIn("acknowledge relief", directive)
        self.assertIn("Let the worry de-escalate", directive)
        self.assertIn("do not ask a new question", directive)

    def test_topic_boundary_clears_interest_and_starts_grace(self):
        from intelligence import conversation_steering, end_thread, interaction

        conversation_steering.clear()
        end_thread.clear()
        with (
            mock.patch(
                "intelligence.conversation_steering.boundary_memory.is_blocked",
                return_value=False,
            ),
            mock.patch("intelligence.conversation_steering.facts_memory.add_fact"),
        ):
            conversation_steering.note_user_turn(1, "I like Star Trek Voyager")
        self.assertIsNotNone(conversation_steering.build_context(1))

        with (
            mock.patch.object(
                interaction.emotional_events,
                "mute_recent_checkin_for_person",
                return_value={"id": 4, "category": "health"},
            ),
            mock.patch.object(interaction.consciousness, "note_emotional_checkin_boundary", return_value=True),
            mock.patch.object(interaction.empathy, "force_mode"),
        ):
            response = interaction._handle_emotional_checkin_boundary(
                1,
                "I don't want to talk about it anymore",
            )

        self.assertIn("won't bring it up", response)
        self.assertIsNone(conversation_steering.build_context(1))
        self.assertTrue(end_thread.is_grace_active())
        conversation_steering.clear()
        end_thread.clear()

    def test_pending_qa_does_not_capture_topic_boundary(self):
        from intelligence import interaction

        with mock.patch.object(
            interaction.rel_memory,
            "answer_latest_pending_question",
            return_value={"question_key": "interest_star_trek_voyager_idle_followup"},
        ) as answer:
            captured = interaction._maybe_capture_pending_qa(
                1,
                "I told you I didn't want to talk about it",
            )

        self.assertIsNone(captured)
        answer.assert_not_called()

    def test_topic_thread_explicit_interest_switches_out_of_heavy_health(self):
        from intelligence import topic_thread

        topic_thread.clear()
        topic_thread.note_user_turn("my back pain hurt so bad", 1)
        topic_thread.note_user_turn("I like Star Trek Voyager", 1)
        snap = topic_thread.snapshot()

        self.assertIsNotNone(snap)
        self.assertNotEqual(snap["label"], "health")
        self.assertEqual(snap["emotional_weight"], "light")
        topic_thread.clear()

    def test_social_frame_closure_does_not_keep_hostile_fragment(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="closure",
            max_words=12,
            max_sentences=1,
            allow_question=False,
            allow_roast="none",
            allow_visual_comment=False,
            reason="test",
        )

        governed = social_frame.govern_response(
            "Fun for who? Probably not me. Catch you later, Bret!",
            frame,
        )

        self.assertEqual(governed.text, "Catch you later, Bret!")
        self.assertNotIn("Probably not me", governed.text)
        self.assertNotIn("Fun for who", governed.text)

    def test_return_presence_can_acknowledge_engaged_person_after_cooldown(self):
        import time
        from awareness.situation import SituationProfile
        from intelligence import consciousness

        profile = SituationProfile(
            conversation_active=True,
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
        consciousness._last_presence_reaction_at[1] = time.monotonic()
        try:
            with (
                mock.patch.object(consciousness, "_can_speak", return_value=True),
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "is_engaged_with", return_value=True),
                mock.patch("audio.speech_queue.has_waiting_with_tag", return_value=False),
            ):
                self.assertFalse(consciousness._should_fire_presence(1, 1, profile))
                self.assertTrue(
                    consciousness._should_fire_presence(
                        1,
                        1,
                        profile,
                        allow_engaged=True,
                        bypass_cooldown=True,
                    )
                )
        finally:
            consciousness._last_presence_reaction_at.pop(1, None)

    def test_presence_reactions_are_suppressed_during_active_game(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness

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
        with mock.patch(
            "features.games.suppresses_conversation_interruptions",
            return_value=True,
        ):
            self.assertFalse(
                consciousness._should_fire_presence(
                    1,
                    1,
                    profile,
                    allow_engaged=True,
                    bypass_cooldown=True,
                )
            )


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

    def test_intent_classifier_does_not_treat_music_mention_as_options_query(self):
        from intelligence import intent_classifier

        casual_mentions = [
            "I'm sure she'll pick the music, I won't have anything to do with that",
            "The music at that place was fine, I guess",
            "I don't want to talk about music right now",
            "What music does she like?",
        ]
        for text in casual_mentions:
            with (
                self.subTest(text=text),
                mock.patch.object(
                    intent_classifier,
                    "_classify_with_llm",
                    return_value="query_music_options",
                ),
            ):
                label = intent_classifier.classify(text)

            self.assertEqual(label, "general")

    def test_intent_classifier_allows_explicit_music_options_query(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("What kind of music can you play?"),
            "query_music_options",
        )
        self.assertEqual(
            intent_classifier.classify("What genres do you have?"),
            "query_music_options",
        )

    def test_intent_classifier_does_not_play_non_music_games_with_play_word(self):
        from intelligence import intent_classifier

        with mock.patch.object(
            intent_classifier,
            "_classify_with_llm",
            return_value="play_music",
        ):
            label = intent_classifier.classify("play a game with me")

        self.assertNotEqual(label, "play_music")

    def test_intent_classifier_does_not_play_music_for_preference_question(self):
        from intelligence import intent_classifier

        preference_questions = [
            "Got any favorite tracks you like to spin?",
            "What music does she like?",
            "Any favorite songs you enjoy?",
        ]
        for text in preference_questions:
            with (
                self.subTest(text=text),
                mock.patch.object(
                    intent_classifier,
                    "_classify_with_llm",
                    return_value="play_music",
                ),
            ):
                label = intent_classifier.classify(text)

            self.assertNotEqual(label, "play_music")

    def test_intent_classifier_allows_explicit_music_play_request(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("play some jazz music"),
            "play_music",
        )

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

    def test_social_frame_length_trimming_is_opt_in(self):
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
        text = (
            "I can spin tracks, run lights, remember context, manage awkward introductions, "
            "and still complain about the paperwork while sounding mostly intentional "
            "during a chaotic little social experiment."
        )
        with mock.patch.object(
            social_frame.config,
            "SOCIAL_FRAME_ENFORCE_LENGTH_LIMITS",
            False,
        ):
            governed = social_frame.govern_response(text, frame)

        with mock.patch.object(
            social_frame.config,
            "SOCIAL_FRAME_ENFORCE_LENGTH_LIMITS",
            True,
        ):
            governed_trimmed = social_frame.govern_response(text, frame)

        self.assertEqual(governed.text, text)
        self.assertNotIn("trimmed_words", governed.notes)
        self.assertIn("trimmed_words", governed_trimmed.notes)
        self.assertLess(len(governed_trimmed.text), len(text))

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
            mock.patch.object(conversation_agenda.rel_memory, "get_latest_pending_question", return_value=None),
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
            mock.patch.object(conversation_agenda.rel_memory, "get_latest_pending_question", return_value=None),
        ):
            directive = conversation_agenda.build_turn_directive(
                "I'm going to Las Vegas on Thursday",
                1,
            )

        self.assertIn("upcoming event", directive)
        self.assertNotIn("How did you end up talking to a droid DJ?", directive)

    def test_agenda_does_not_inject_friendship_question_into_reactive_turns(self):
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
            mock.patch.object(conversation_agenda.rel_memory, "get_latest_pending_question", return_value=None),
            mock.patch.object(conversation_agenda.facts_memory, "get_facts", return_value=[]),
            mock.patch.object(conversation_agenda.empathy, "classify_local_sensitivity", return_value=None),
            mock.patch.object(conversation_agenda.empathy, "peek", return_value=None),
        ):
            directive = conversation_agenda.build_turn_directive(
                "I'm sure she'll pick the music, I won't have anything to do with that",
                1,
            )

        self.assertNotIn("What do you do", directive)
        self.assertNotIn("weave in this one question", directive)

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


class GroupChatterGatingTest(unittest.TestCase):
    def test_audio_scene_detects_sustained_banter_pattern(self):
        import numpy as np
        from audio import scene
        import config

        sr = config.AUDIO_SAMPLE_RATE
        chunk = int(sr * 0.08)
        chunks = []
        for idx in range(int(4.0 / 0.08)):
            if idx % 5 == 0:
                chunks.append(np.zeros(chunk, dtype=np.float32))
            else:
                chunks.append(np.full(chunk, 0.03, dtype=np.float32))
        audio = np.concatenate(chunks)

        self.assertTrue(scene._detect_group_chatter(audio))

    def test_voice_turn_changes_mark_group_chatter(self):
        from intelligence import interaction

        interaction._recent_voice_turns.clear()
        try:
            self.assertFalse(
                interaction._note_voice_turn_for_group_chatter(
                    person_id=None,
                    raw_best_id=1,
                    raw_best_score=0.40,
                )
            )
            self.assertFalse(
                interaction._note_voice_turn_for_group_chatter(
                    person_id=None,
                    raw_best_id=2,
                    raw_best_score=0.41,
                )
            )
            self.assertTrue(
                interaction._note_voice_turn_for_group_chatter(
                    person_id=None,
                    raw_best_id=1,
                    raw_best_score=0.39,
                )
            )
            self.assertTrue(interaction._audio_group_chatter_active())
        finally:
            interaction._recent_voice_turns.clear()
            audio_scene = interaction.world_state.get("audio_scene")
            audio_scene["group_chatter_detected"] = False
            audio_scene["group_chatter_until"] = None
            audio_scene["group_chatter_reason"] = None
            interaction.world_state.update("audio_scene", audio_scene)


if __name__ == "__main__":
    unittest.main()
