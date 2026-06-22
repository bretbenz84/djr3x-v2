import unittest
from contextlib import ExitStack
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory


def _speak_async_spoke(*args, on_spoke=None, **kwargs):
    """Faithful stub for consciousness._speak_async: it returns True ("I spoke")
    AND fires the on_spoke bookkeeping callback, exactly as the real function does
    when a line reaches the speech queue. Use as `side_effect` so tests that assert
    on-spoke side effects (cooldown latch, _prime_emotion_frame, pop pending) see
    them the same way the production winner-only path does."""
    if on_spoke is not None:
        on_spoke()
    return True


class PostTtsHandoffPolicyTest(unittest.TestCase):
    def setUp(self):
        # The post-question "sticky responsive" window is keyed on wall-clock time
        # via a module global; clear it so a prior test's question handoff (often
        # with time.monotonic mocked to a fixed value) can't keep a later
        # statement handoff responsive and skip its flush.
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0
        # Proactive paths now hold off if another proactive line fired in the last
        # few seconds; clear the leaked global so a prior test can't suppress this
        # one's question/banter.
        interaction._last_proactive_line_at = 0.0
        # These tests assert the default (non-hardware-AEC) post-TTS tuning. Force the
        # ReSpeaker hardware-AEC gate OFF so they are deterministic regardless of
        # whether a ReSpeaker Lite is plugged into the test machine. The AEC-on
        # overrides are covered by HardwareAecBoundaryTest below.
        _aec = mock.patch("audio.hardware_aec.is_active", return_value=False)
        _aec.start()
        self.addCleanup(_aec.stop)

    def test_startup_burst_defers_chime_to_ready_cue(self):
        import config

        files = [Path(path).name for path in config.STARTUP_AUDIO_FILES]
        # The chime is intentionally NOT in the opening burst — main.py plays it at
        # the END of startup (once models are loaded and listening) as the ready cue.
        self.assertNotIn("startup_chime.mp3", files)
        self.assertIn("Roger Control.mp3", files)
        # The chime file is still configured; it's just played later, at end-of-load.
        self.assertEqual(Path(config.LISTENING_CHIME_FILE).name, "startup_chime.mp3")

    def test_startup_speech_clip_toggle_contract(self):
        """The spoken startup intro has an easy on/off toggle (default OFF). main.py's
        skip logic keys off STARTUP_AUDIO_FILES slots whose name is in
        STARTUP_SPEECH_CLIP_CHOICES, so that wiring must hold."""
        import config

        # The toggle exists, is a bool, and ships OFF (the requested default).
        self.assertIsInstance(config.PLAY_STARTUP_SPEECH_CLIP, bool)
        self.assertFalse(config.PLAY_STARTUP_SPEECH_CLIP)

        # The randomized intro clips are exactly the ones the user named.
        choice_names = {Path(c).name for c in config.STARTUP_SPEECH_CLIP_CHOICES}
        self.assertEqual(
            choice_names,
            {"Roger Control.mp3", "Outer Rim.mp3", "This is your cap.mp3"},
        )

        # At least one STARTUP_AUDIO_FILES slot is a speech clip (so the toggle has
        # something to skip), and at least one is NOT (so the boot SFX still plays
        # when the intro is off).
        burst = {Path(p).name for p in config.STARTUP_AUDIO_FILES}
        self.assertTrue(burst & choice_names, "no speech-clip slot to toggle off")
        self.assertTrue(burst - choice_names, "no non-speech SFX would survive the toggle")

    def test_direct_startup_clip_arms_aec_and_limits_level(self):
        import numpy as np
        import main
        from world_state import world_state

        source_audio = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        old_self_state = world_state.get("self_state")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_FILES",
                    ["Roger Control.mp3"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_TRANSCRIPTS",
                    {"Roger Control.mp3": "Roger control, all systems go!"},
                )
            )
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_GAIN", 1.0))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT", 0.5))
            stack.enter_context(mock.patch.object(main.sf, "read", return_value=(source_audio, 16000)))
            play = stack.enter_context(mock.patch.object(main.sd, "play"))
            wait = stack.enter_context(mock.patch.object(main.sd, "wait"))
            set_playing = stack.enter_context(mock.patch.object(main.echo_cancel, "set_playing"))
            add_reference = stack.enter_context(mock.patch.object(main.echo_cancel, "add_reference"))
            speech_activity_start = stack.enter_context(
                mock.patch.object(main.animations, "speech_activity_start")
            )
            speech_activity_stop = stack.enter_context(
                mock.patch.object(main.animations, "speech_activity_stop")
            )
            begin_speech_motion = stack.enter_context(
                mock.patch.object(main.servos, "begin_speech_motion")
            )
            end_speech_motion = stack.enter_context(
                mock.patch.object(main.servos, "end_speech_motion")
            )
            stack.enter_context(mock.patch.object(main.servos, "speech_reactive_move"))
            head_speak = stack.enter_context(mock.patch.object(main.leds_head, "speak"))
            head_speak_stop = stack.enter_context(mock.patch.object(main.leds_head, "speak_stop"))
            stack.enter_context(mock.patch.object(main.leds_head, "speak_level"))
            chest_speak = stack.enter_context(mock.patch.object(main.leds_chest, "speak"))
            chest_active = stack.enter_context(mock.patch.object(main.leds_chest, "active"))
            set_rex_speaking = stack.enter_context(
                mock.patch("awareness.situation.assessor.set_rex_speaking")
            )
            log_rex = stack.enter_context(mock.patch("utils.conv_log.log_rex"))
            try:
                main._play_audio_file("assets/audio/startup/Roger Control.mp3")
            finally:
                world_state.update("self_state", old_self_state)

        set_playing.assert_has_calls([mock.call(True), mock.call(False)])
        add_reference.assert_called_once()
        play.assert_called_once()
        wait.assert_called_once()
        played_audio = play.call_args.args[0]
        self.assertLessEqual(float(np.max(np.abs(played_audio))), 0.5)
        speech_activity_start.assert_called_once()
        speech_activity_stop.assert_called_once()
        begin_speech_motion.assert_called_once()
        speech_frame = begin_speech_motion.call_args.args[0]
        self.assertEqual(speech_frame.affect, "neutral")
        end_speech_motion.assert_called_once()
        head_speak.assert_called_once_with("neutral")
        head_speak_stop.assert_called_once()
        chest_speak.assert_called_once_with("neutral")
        chest_active.assert_called_once()
        set_rex_speaking.assert_has_calls([mock.call(True), mock.call(False)])
        log_rex.assert_called_once_with("Roger control, all systems go!")

    def test_direct_sound_effect_skips_speech_animation(self):
        import numpy as np
        import main

        source_audio = np.array([0.0, 0.25, -0.25, 0.1], dtype=np.float32)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_FILES",
                    ["Roger Control.mp3"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_TRANSCRIPTS",
                    {"Roger Control.mp3": "Roger control, all systems go!"},
                )
            )
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_GAIN", 1.0))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT", 0.8))
            stack.enter_context(mock.patch.object(main.sf, "read", return_value=(source_audio, 16000)))
            stack.enter_context(mock.patch.object(main.sd, "play"))
            stack.enter_context(mock.patch.object(main.sd, "wait"))
            stack.enter_context(mock.patch.object(main.echo_cancel, "set_playing"))
            stack.enter_context(mock.patch.object(main.echo_cancel, "add_reference"))
            speech_activity_start = stack.enter_context(
                mock.patch.object(main.animations, "speech_activity_start")
            )
            speech_activity_stop = stack.enter_context(
                mock.patch.object(main.animations, "speech_activity_stop")
            )
            begin_speech_motion = stack.enter_context(
                mock.patch.object(main.servos, "begin_speech_motion")
            )
            end_speech_motion = stack.enter_context(
                mock.patch.object(main.servos, "end_speech_motion")
            )
            speech_reactive_move = stack.enter_context(
                mock.patch.object(main.servos, "speech_reactive_move")
            )
            head_speak = stack.enter_context(mock.patch.object(main.leds_head, "speak"))
            head_speak_stop = stack.enter_context(mock.patch.object(main.leds_head, "speak_stop"))
            head_speak_level = stack.enter_context(mock.patch.object(main.leds_head, "speak_level"))
            chest_speak = stack.enter_context(mock.patch.object(main.leds_chest, "speak"))
            chest_active = stack.enter_context(mock.patch.object(main.leds_chest, "active"))
            set_rex_speaking = stack.enter_context(
                mock.patch("awareness.situation.assessor.set_rex_speaking")
            )
            log_rex = stack.enter_context(mock.patch("utils.conv_log.log_rex"))
            main._play_audio_file("assets/audio/startup/light_speed.mp3")

        speech_activity_start.assert_not_called()
        speech_activity_stop.assert_not_called()
        begin_speech_motion.assert_not_called()
        end_speech_motion.assert_not_called()
        speech_reactive_move.assert_not_called()
        head_speak.assert_not_called()
        head_speak_stop.assert_not_called()
        head_speak_level.assert_not_called()
        chest_speak.assert_not_called()
        chest_active.assert_not_called()
        set_rex_speaking.assert_not_called()
        log_rex.assert_not_called()

    def test_startup_boot_tts_thread_speaks_after_configured_delay(self):
        import main

        line = "Hang on folks while I'm booting up."
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.config, "PLAY_STARTUP_BOOT_TTS", True))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_LINES", [line]))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_LINE", line))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_DELAY_SECS", 1.25))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_EMOTION", "curious"))
            sleep = stack.enter_context(mock.patch.object(main.time, "sleep"))
            speak = stack.enter_context(mock.patch.object(main.tts, "speak"))

            thread = main._start_startup_boot_tts_thread()
            self.assertIsNotNone(thread)
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive())

        sleep.assert_called_once_with(1.25)
        speak.assert_called_once_with(line, "curious")

    def test_startup_boot_tts_chains_behind_startup_audio_thread(self):
        """The boot filler must follow the startup mp3s immediately: the worker
        joins the startup-audio thread (released when the last clip ends), then
        applies only the configured beat. Regression for the ~8s of dead air
        when the filler was started after the blocking startup animation."""
        import main
        import threading

        line = "Hang on folks while I'm booting up."
        audio_done = threading.Event()
        order: list[str] = []

        def _fake_audio() -> None:
            audio_done.wait(timeout=2.0)
            order.append("audio_finished")

        audio_thread = threading.Thread(target=_fake_audio, daemon=True)
        audio_thread.start()

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.config, "PLAY_STARTUP_BOOT_TTS", True))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_LINES", [line]))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_LINE", line))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_DELAY_SECS", 0.0))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_BOOT_TTS_EMOTION", "curious"))
            speak = stack.enter_context(
                mock.patch.object(main.tts, "speak", side_effect=lambda *a, **k: order.append("boot_tts"))
            )

            thread = main._start_startup_boot_tts_thread(wait_for=audio_thread)
            self.assertIsNotNone(thread)
            # Boot line must NOT play while the startup audio is still going.
            thread.join(timeout=0.3)
            self.assertTrue(thread.is_alive())
            speak.assert_not_called()
            # The moment the audio finishes, the boot line follows.
            audio_done.set()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())

        speak.assert_called_once_with(line, "curious")
        self.assertEqual(order, ["audio_finished", "boot_tts"])

    def test_startup_boot_tts_skips_noaudio(self):
        import main

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", True))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.config, "PLAY_STARTUP_BOOT_TTS", True))
            speak = stack.enter_context(mock.patch.object(main.tts, "speak"))

            thread = main._start_startup_boot_tts_thread()

        self.assertIsNone(thread)
        speak.assert_not_called()

    def test_direct_shutdown_clip_leaves_leds_off(self):
        import numpy as np
        import main

        source_audio = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_FILES",
                    ["shutdown.mp3"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    main.config,
                    "SPEECH_ANIMATED_AUDIO_TRANSCRIPTS",
                    {"shutdown.mp3": "Powering down."},
                )
            )
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_GAIN", 1.0))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SHUTDOWN_AUDIO_PEAK_LIMIT", 0.8))
            stack.enter_context(mock.patch.object(main.sf, "read", return_value=(source_audio, 16000)))
            stack.enter_context(mock.patch.object(main.sd, "play"))
            stack.enter_context(mock.patch.object(main.sd, "wait"))
            stack.enter_context(mock.patch.object(main.echo_cancel, "set_playing"))
            stack.enter_context(mock.patch.object(main.echo_cancel, "add_reference"))
            stack.enter_context(mock.patch.object(main.echo_cancel, "was_canceled", return_value=False))
            stack.enter_context(mock.patch.object(main.animations, "speech_activity_start"))
            stack.enter_context(mock.patch.object(main.animations, "speech_activity_stop"))
            stack.enter_context(mock.patch.object(main.servos, "begin_speech_motion"))
            stack.enter_context(mock.patch.object(main.servos, "end_speech_motion"))
            stack.enter_context(mock.patch.object(main.servos, "speech_reactive_move"))
            stack.enter_context(mock.patch.object(main.leds_head, "speak"))
            stack.enter_context(mock.patch.object(main.leds_head, "speak_level"))
            head_speak_stop = stack.enter_context(mock.patch.object(main.leds_head, "speak_stop"))
            head_off = stack.enter_context(mock.patch.object(main.leds_head, "off"))
            stack.enter_context(mock.patch.object(main.leds_chest, "speak"))
            chest_active = stack.enter_context(mock.patch.object(main.leds_chest, "active"))
            chest_off = stack.enter_context(mock.patch.object(main.leds_chest, "off"))
            stack.enter_context(mock.patch.object(main, "_is_shutdown_state", return_value=True))
            stack.enter_context(mock.patch("awareness.situation.assessor.set_rex_speaking"))
            stack.enter_context(mock.patch("utils.conv_log.log_rex"))

            main._play_audio_file("assets/audio/shutdown/shutdown.mp3")

        head_off.assert_called_once()
        chest_off.assert_called_once()
        head_speak_stop.assert_not_called()
        chest_active.assert_not_called()

    def test_startup_device_warning_prefers_combined_line(self):
        import main

        lines = {
            "camera": ["camera offline"],
            "audio": ["audio offline"],
            "both": ["both offline"],
        }
        with mock.patch.object(main.config, "STARTUP_SENSOR_WARNING_LINES", lines):
            self.assertEqual(
                main._startup_device_warning_line(
                    camera_available=False,
                    audio_available=False,
                ),
                "both offline",
            )
            self.assertEqual(
                main._startup_device_warning_line(
                    camera_available=False,
                    audio_available=True,
                ),
                "camera offline",
            )
            self.assertEqual(
                main._startup_device_warning_line(
                    camera_available=True,
                    audio_available=False,
                ),
                "audio offline",
            )
            self.assertIsNone(
                main._startup_device_warning_line(
                    camera_available=True,
                    audio_available=True,
                )
            )

    def test_startup_device_warning_queues_tts(self):
        import main

        lines = {"camera": ["camera offline"]}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SENSOR_WARNING_ENABLED", True))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SENSOR_WARNING_EMOTION", "curious"))
            stack.enter_context(mock.patch.object(main.config, "STARTUP_SENSOR_WARNING_LINES", lines))
            enqueue = stack.enter_context(mock.patch.object(main.speech_queue, "enqueue"))

            chosen = main._queue_startup_device_warning(
                camera_available=False,
                audio_available=True,
            )

        self.assertEqual(chosen, "camera offline")
        enqueue.assert_called_once_with(
            "camera offline",
            "curious",
            priority=1,
            tag="startup:sensor_warning",
        )

    def test_startup_device_warning_skips_when_audio_suppressed(self):
        import main

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", True))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", True))
            enqueue = stack.enter_context(mock.patch.object(main.speech_queue, "enqueue"))

            chosen = main._queue_startup_device_warning(
                camera_available=False,
                audio_available=False,
            )

        self.assertIsNone(chosen)
        enqueue.assert_not_called()

    def test_camera_reconnect_line_queues_tts(self):
        import main

        old_last = main._last_camera_reconnect_line
        main._last_camera_reconnect_line = None
        try:
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_ENABLED", True))
                stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_MIN_DOWNTIME_SECS", 1.0))
                stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_EMOTION", "happy"))
                stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_LINES", ["optics restored"]))
                stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
                stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
                stack.enter_context(mock.patch.object(main.state, "get_state", return_value=main.State.ACTIVE))
                enqueue = stack.enter_context(mock.patch.object(main.speech_queue, "enqueue"))

                chosen = main._queue_camera_reconnect_line(5.0)
        finally:
            main._last_camera_reconnect_line = old_last

        self.assertEqual(chosen, "optics restored")
        enqueue.assert_called_once_with(
            "optics restored",
            "happy",
            priority=0,
            tag="camera:reconnected",
        )

    def test_camera_reconnect_line_respects_quiet_and_downtime(self):
        import main

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_ENABLED", True))
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_MIN_DOWNTIME_SECS", 3.0))
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_LINES", ["optics restored"]))
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.state, "get_state", return_value=main.State.ACTIVE))
            enqueue = stack.enter_context(mock.patch.object(main.speech_queue, "enqueue"))

            self.assertIsNone(main._queue_camera_reconnect_line(1.0))
            enqueue.assert_not_called()

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_ENABLED", True))
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_MIN_DOWNTIME_SECS", 0.0))
            stack.enter_context(mock.patch.object(main.config, "CAMERA_RECONNECT_TTS_LINES", ["optics restored"]))
            stack.enter_context(mock.patch.object(main.config, "NO_AUDIO_MODE", False))
            stack.enter_context(mock.patch.object(main.config, "AUDIO_OUTPUT_SUPPRESSED", False))
            stack.enter_context(mock.patch.object(main.state, "get_state", return_value=main.State.QUIET))
            enqueue = stack.enter_context(mock.patch.object(main.speech_queue, "enqueue"))

            self.assertIsNone(main._queue_camera_reconnect_line(5.0))
            enqueue.assert_not_called()

    def test_question_queue_item_uses_fast_no_flush_playback_handoff(self):
        from audio import speech_queue

        opts = speech_queue._playback_handoff_options(
            "What name should I save for you?"
        )

        self.assertEqual(
            opts["post_playback_tail_secs"],
            float(__import__("config").POST_QUESTION_PLAYBACK_SUPPRESSION_SECS),
        )
        self.assertEqual(
            opts["flush_on_playback_stop"],
            bool(__import__("config").POST_QUESTION_FLUSH_AUDIO_BUFFER),
        )

    def test_social_queue_item_uses_responsive_no_flush_handoff(self):
        from audio import speech_queue

        opts = speech_queue._playback_handoff_options(
            "Got it, Bret. Nice to meet you."
        )

        # Statements now get the short tail and no flush so an immediate reply survives.
        self.assertEqual(
            opts["post_playback_tail_secs"],
            float(__import__("config").POST_SPEECH_PLAYBACK_SUPPRESSION_SECS),
        )
        self.assertEqual(
            opts["flush_on_playback_stop"],
            bool(__import__("config").POST_SPEECH_FLUSH_AUDIO_BUFFER),
        )
        self.assertFalse(opts["flush_on_playback_stop"])

    def test_last_name_prompt_without_question_mark_uses_fast_handoff(self):
        from audio import speech_queue

        opts = speech_queue._playback_handoff_options(
            "Bret, how original. Give me a last name too so the memory banks don't get confused."
        )

        self.assertEqual(
            opts["post_playback_tail_secs"],
            float(__import__("config").POST_QUESTION_PLAYBACK_SUPPRESSION_SECS),
        )
        self.assertEqual(
            opts["flush_on_playback_stop"],
            bool(__import__("config").POST_QUESTION_FLUSH_AUDIO_BUFFER),
        )

    def test_statement_queue_item_uses_responsive_no_flush_handoff(self):
        from audio import speech_queue

        opts = speech_queue._playback_handoff_options("Classic choice.")

        self.assertEqual(
            opts["post_playback_tail_secs"],
            float(__import__("config").POST_SPEECH_PLAYBACK_SUPPRESSION_SECS),
        )
        self.assertFalse(opts["flush_on_playback_stop"])

    def test_question_playback_stop_can_skip_flush_and_short_tail(self):
        from audio import echo_cancel

        with echo_cancel._lock:
            old_playing = echo_cancel._playing
            old_suppress_until = echo_cancel._suppress_until
            old_sequence_active = echo_cancel._sequence_active
            old_canceled = echo_cancel._playback_canceled
            echo_cancel._sequence_active = False
            echo_cancel._playback_canceled = False
        try:
            with (
                mock.patch("audio.stream.flush") as flush,
                mock.patch.object(echo_cancel.time, "monotonic", return_value=100.0),
            ):
                echo_cancel.set_playing(True)
                echo_cancel.set_playing(False, tail_secs=0.05, flush=False)
                flush.assert_not_called()
                self.assertTrue(echo_cancel.is_suppressed())

            with mock.patch.object(echo_cancel.time, "monotonic", return_value=100.06):
                self.assertFalse(echo_cancel.is_suppressed())
        finally:
            with echo_cancel._lock:
                echo_cancel._playing = old_playing
                echo_cancel._suppress_until = old_suppress_until
                echo_cancel._sequence_active = old_sequence_active
                echo_cancel._playback_canceled = old_canceled

    def test_tts_trims_trailing_padding_so_question_handoff_is_not_late(self):
        import numpy as np
        from audio import tts

        samplerate = 16000
        voice = np.ones(int(0.5 * samplerate), dtype=np.float32) * 0.05
        tail = np.zeros(int(0.6 * samplerate), dtype=np.float32)
        audio = np.concatenate([voice, tail])

        with (
            mock.patch.object(tts.config, "TTS_TRIM_TRAILING_SILENCE_ENABLED", True),
            mock.patch.object(tts.config, "TTS_TRIM_TRAILING_SILENCE_THRESHOLD", 0.003),
            mock.patch.object(tts.config, "TTS_TRIM_TRAILING_SILENCE_WINDOW_MS", 20),
            mock.patch.object(tts.config, "TTS_TRIM_TRAILING_SILENCE_PADDING_MS", 40),
        ):
            trimmed = tts._trim_trailing_silence(audio, samplerate)

        self.assertLess(len(trimmed), len(audio) - int(0.45 * samplerate))
        self.assertGreaterEqual(len(trimmed), len(voice))

    def test_tts_trailing_trim_leaves_short_tails_alone(self):
        import numpy as np
        from audio import tts

        samplerate = 16000
        voice = np.ones(int(0.5 * samplerate), dtype=np.float32) * 0.05
        tail = np.zeros(int(0.04 * samplerate), dtype=np.float32)
        audio = np.concatenate([voice, tail])

        with mock.patch.object(tts.config, "TTS_TRIM_TRAILING_SILENCE_ENABLED", True):
            trimmed = tts._trim_trailing_silence(audio, samplerate)

        self.assertEqual(len(trimmed), len(audio))

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

    def test_mark_startup_chime_played_skips_first_text_chime(self):
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
                queue.mark_startup_chime_played()
                queue.enqueue("Hello there.", priority=1)

            queued = sorted(queue._heap, key=lambda item: item.seq)

        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].text, "Hello there.")

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

    def test_noaudio_speech_queue_logs_text_without_tts(self):
        from audio import speech_queue

        with (
            mock.patch.object(speech_queue._SpeechQueue, "_worker", lambda self: None),
            mock.patch("config.NO_AUDIO_MODE", True),
            mock.patch("config.AUDIO_OUTPUT_SUPPRESSED", True),
            mock.patch("utils.conv_log.log_rex") as log_rex,
            mock.patch("audio.tts.speak") as tts_speak,
        ):
            queue = speech_queue._SpeechQueue()
            done = queue.enqueue("Text-only response.", priority=1)

        self.assertTrue(done.is_set())
        self.assertEqual(queue._heap, [])
        log_rex.assert_called_once_with("Text-only response.")
        tts_speak.assert_not_called()

    def test_noaudio_tts_speak_skips_elevenlabs_and_playback(self):
        from audio import tts

        started = []
        with (
            mock.patch("config.NO_AUDIO_MODE", True),
            mock.patch("config.AUDIO_OUTPUT_SUPPRESSED", True),
            mock.patch.object(tts, "_fetch_from_api") as fetch,
            mock.patch.object(tts, "_play") as play,
            mock.patch.object(tts.conv_log, "log_rex") as log_rex,
        ):
            tts.speak(
                "Hello there.",
                on_playback_start=lambda: started.append(True),
            )

        fetch.assert_not_called()
        play.assert_not_called()
        log_rex.assert_called_once_with("Hello there.")
        self.assertEqual(started, [True])

    def test_begin_user_turn_keeps_game_prompts_queued(self):
        from intelligence import interaction
        from hardware import servos

        with (
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=True),
            mock.patch.object(interaction.speech_queue, "clear_below_priority") as clear,
            # _begin_user_turn now also starts listening motion; stub it so the
            # test doesn't spawn a real servo thread on a hardware-attached box.
            mock.patch.object(servos, "start_listening_motion"),
        ):
            interaction._begin_user_turn()

        clear.assert_not_called()

    def test_submit_text_routes_gui_text_as_text_input_turn(self):
        from intelligence import interaction
        from state import State

        with (
            mock.patch.object(interaction.state_module, "get_state", return_value=State.IDLE),
            mock.patch.object(interaction.state_module, "set_state") as set_state,
            mock.patch.object(interaction, "_begin_user_turn") as begin_turn,
            mock.patch.object(interaction, "_end_user_turn") as end_turn,
            mock.patch.object(interaction, "_handle_speech_segment") as handle_segment,
        ):
            handled = interaction.submit_text("hello from the GUI")

        self.assertTrue(handled)
        set_state.assert_called_once_with(State.ACTIVE)
        begin_turn.assert_called_once()
        end_turn.assert_called_once()
        handle_segment.assert_called_once()
        self.assertEqual(
            handle_segment.call_args.kwargs["transcribed_text"],
            "hello from the GUI",
        )
        self.assertTrue(handle_segment.call_args.kwargs["text_input"])

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

    def test_real_conversation_log_is_not_clobbered_under_test_runner(self):
        # QoL: running the suite must NOT write to the live logs/conversation.log
        # (it kept overwriting/trimming a real run's transcript). Writes to the
        # DEFAULT path are suppressed under the runner; both write entry points
        # (log_rex routes straight to _append_locked, log_heard via _write).
        from utils import conv_log
        self.assertTrue(conv_log._under_test_runner())
        self.assertTrue(conv_log._writes_suppressed())
        real = conv_log._DEFAULT_LOG_PATH
        before = real.read_text(encoding="utf-8") if real.exists() else ""
        conv_log.clear_dedupe_state()
        conv_log.log_rex("MUST_NOT_HIT_REAL_LOG_rex")
        conv_log.log_heard("Tester", "MUST_NOT_HIT_REAL_LOG_heard")
        after = real.read_text(encoding="utf-8") if real.exists() else ""
        self.assertEqual(before, after)
        conv_log.clear_dedupe_state()

    def test_patched_log_path_still_writes_under_test_runner(self):
        # A test that patches _LOG_PATH to a temp file opts in to exercising the
        # writer — suppression must NOT apply there (it only guards the real log).
        from utils import conv_log
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "conversation.log"
            with mock.patch.object(conv_log, "_LOG_PATH", p):
                self.assertFalse(conv_log._writes_suppressed())
                conv_log.clear_dedupe_state()
                conv_log.log_rex("this SHOULD reach the temp log")
                self.assertIn("this SHOULD reach the temp log", p.read_text(encoding="utf-8"))
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
        self.assertTrue(policy.fast_response_expected)
        self.assertEqual(
            policy.listen_delay_secs,
            float(config.POST_QUESTION_LISTEN_DELAY_SECS),
        )
        self.assertFalse(policy.flush_buffer)

    def test_social_handoff_preserves_buffer_for_immediate_reply(self):
        from intelligence import interaction
        import config

        policy = interaction._post_tts_handoff_policy("Got it, Bret. Nice to meet you.")

        self.assertFalse(policy.asked_question)
        self.assertFalse(policy.fast_response_expected)
        self.assertEqual(
            policy.listen_delay_secs,
            float(config.POST_SPEECH_LISTEN_DELAY_SECS),
        )
        # A statement reply that began as Rex finished must not be flushed away.
        self.assertFalse(policy.flush_buffer)

    def test_statement_handoff_preserves_buffer_for_immediate_reply(self):
        from intelligence import interaction
        import config

        policy = interaction._post_tts_handoff_policy("Classic choice.")

        self.assertFalse(policy.asked_question)
        self.assertFalse(policy.fast_response_expected)
        self.assertEqual(
            policy.listen_delay_secs,
            float(config.POST_SPEECH_LISTEN_DELAY_SECS),
        )
        self.assertFalse(policy.flush_buffer)

    def test_assistant_question_detector_ignores_quoted_questions_and_tiny_openers(self):
        from intelligence import interaction

        self.assertFalse(
            interaction._assistant_asked_question(
                'Seven more hours. You must be collecting "Are we there yet?" queries.'
            )
        )
        self.assertFalse(interaction._assistant_asked_question("Seven more hours?"))
        self.assertTrue(interaction._assistant_asked_question("Do they cause chaos?"))
        self.assertTrue(interaction._assistant_asked_question("You good?"))
        self.assertTrue(interaction._assistant_asked_question("Last name?"))
        self.assertTrue(
            interaction._assistant_asked_question(
                "Bret, how original. Give me a last name too so the memory banks don't get confused."
            )
        )

    def test_apply_question_handoff_does_not_flush_stream(self):
        from intelligence import interaction

        old_floor = interaction._listen_capture_floor_at
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=100.0),
                mock.patch.object(interaction.config, "POST_TTS_CAPTURE_PREROLL_GRACE_SECS", 0.0),
                mock.patch.object(interaction.config, "POST_QUESTION_CAPTURE_PREROLL_GRACE_SECS", 0.0),
                mock.patch.object(interaction.stream, "flush") as flush,
                mock.patch.object(interaction.vad, "reset_state") as reset_vad,
            ):
                interaction._apply_post_tts_handoff(
                    "What do you do for work?",
                    source="test",
                )
        finally:
            floor = interaction._listen_capture_floor_at
            interaction._listen_capture_floor_at = old_floor

        flush.assert_not_called()
        reset_vad.assert_called_once()
        self.assertAlmostEqual(floor, 100.0)

    def test_question_handoff_allows_120ms_silent_tts_tail_preroll(self):
        from intelligence import interaction

        old_floor = interaction._listen_capture_floor_at
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=100.0),
                mock.patch.object(interaction.config, "POST_QUESTION_CAPTURE_PREROLL_GRACE_SECS", 0.12),
                mock.patch.object(interaction.stream, "flush") as flush,
                mock.patch.object(interaction.vad, "reset_state"),
            ):
                interaction._apply_post_tts_handoff(
                    "Last name?",
                    source="test",
                )
        finally:
            floor = interaction._listen_capture_floor_at
            interaction._listen_capture_floor_at = old_floor

        flush.assert_not_called()
        self.assertAlmostEqual(floor, 99.88)

    def test_social_handoff_preserves_buffer_and_uses_tail_preroll(self):
        from intelligence import interaction

        old_floor = interaction._listen_capture_floor_at
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=100.0),
                mock.patch.object(interaction.config, "POST_TTS_CAPTURE_PREROLL_GRACE_SECS", 0.12),
                mock.patch.object(interaction.stream, "flush") as flush,
                mock.patch.object(interaction.vad, "reset_state"),
            ):
                interaction._apply_post_tts_handoff(
                    "Got it, Bret. Nice to meet you.",
                    source="test",
                )
        finally:
            floor = interaction._listen_capture_floor_at
            interaction._listen_capture_floor_at = old_floor

        # A statement reply is no longer flushed away, and the capture floor reaches
        # 120ms back into the raw buffer to recover the front of an immediate reply.
        flush.assert_not_called()
        self.assertAlmostEqual(floor, 99.88)

    def test_social_sequence_end_uses_responsive_no_flush_handoff(self):
        from intelligence import interaction
        import config

        with mock.patch.object(interaction.echo_cancel, "end_sequence") as end_sequence:
            interaction._end_response_sequence_for_text("Got it, Bret. Nice to meet you.")

        end_sequence.assert_called_once_with(
            flush=False,
            tail_secs=float(config.POST_SPEECH_PLAYBACK_SUPPRESSION_SECS),
        )

    def test_post_tts_handoff_refreshes_idle_timer(self):
        from intelligence import interaction

        old_floor = interaction._listen_capture_floor_at
        try:
            interaction._last_speech_at = 10.0
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=50.0),
                mock.patch.object(interaction.stream, "flush"),
            ):
                interaction._apply_post_tts_handoff(
                    "Long Star Trek answer complete.",
                    source="test",
                )
        finally:
            interaction._listen_capture_floor_at = old_floor

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

    def test_hometown_answer_is_not_self_intro_name(self):
        from intelligence import interaction

        self.assertIsNone(interaction._extract_self_identified_name("I'm from Sacramento"))
        self.assertIsNone(interaction._extract_self_identified_name("I am in Waterford"))

    def test_filler_is_not_a_valid_name_or_transcript(self):
        from audio import transcription
        from intelligence import interaction

        self.assertIsNone(interaction._extract_introduced_name("mmm", allow_bare_name=True))
        self.assertIsNone(interaction._extract_introduced_name("mmm wait", allow_bare_name=True))
        self.assertIsNone(interaction._extract_introduced_name("have you?", allow_bare_name=True))
        self.assertTrue(transcription._is_hallucination("mmm"))
        self.assertTrue(transcription._is_hallucination("Z" * 160))
        self.assertFalse(transcription._is_hallucination("Soooo yes"))

    def test_repetition_filter_keeps_natural_repetition_but_drops_loops(self):
        from audio import transcription

        # Natural repetition of a function word in a VARIED sentence is real speech,
        # not a Whisper loop — must NOT be filtered (regression: this was discarded).
        self.assertFalse(transcription._is_hallucination(
            "I like lots of composers. I like Beethoven, I like Bach, "
            "I like Tchaikovsky, I like Stravinsky."
        ))
        # A real loop where one word dominates the utterance must still be filtered.
        self.assertTrue(transcription._is_hallucination("you you you you you you"))
        self.assertTrue(transcription._is_hallucination(
            "thank you thank you thank you thank you thank you"
        ))

    def test_subscribe_caption_hallucinations_filtered(self):
        from audio import transcription

        # Whisper hallucinates YouTube-caption boilerplate on silence/noise. The
        # abbreviated "plz subscribe" (field-logged) slipped past the old regex that
        # only matched "please subscribe" — cover the abbreviations and the bare token.
        for phrase in ("plz subscribe", "pls subscribe", "please subscribe",
                       "subscribe", "Subscribe!", "PLZ SUBSCRIBE"):
            self.assertTrue(
                transcription._is_hallucination(phrase),
                f"expected {phrase!r} to be filtered as a caption hallucination",
            )
        # Real speech that merely contains the word must NOT be filtered.
        self.assertFalse(transcription._is_hallucination("I'd subscribe to that idea"))

    def test_intro_name_trims_trailing_greeting(self):
        from intelligence import introductions

        parsed = introductions.detect("This is my sister Jennifer Hi", has_unknown_face=True)

        self.assertTrue(parsed.is_introduction)
        self.assertEqual(parsed.name, "Jennifer")
        self.assertEqual(parsed.relationship, "sister")

    def test_intro_detection_rejects_slangy_this_is_statements(self):
        from intelligence import introductions

        for text in (
            "This is fire, you know, um",
            "This is unbelievable, I'm just here talking to a whole little video",
        ):
            with self.subTest(text=text):
                parsed = introductions.detect(text, has_unknown_face=False)
                self.assertFalse(parsed.is_introduction)

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
            mock.patch("audio.tts.ensure_cached", return_value=True),
            mock.patch.object(interaction.barge_guard, "user_speaking_now", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript") as transcript,
            mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
            mock.patch.object(interaction, "_register_rex_utterance") as register,
        ):
            first = interaction._maybe_idle_outro()
            second = interaction._maybe_idle_outro()

        self.assertTrue(first)
        self.assertFalse(second)
        speak.assert_called_once_with(
            "Nobody talking now.",
            emotion="neutral",
            priority=1,
            pre_beat_ms=0,
            post_beat_ms_override=0,
            voice_settings=None,
        )
        transcript.assert_called_once_with("Rex", "Nobody talking now.")
        log_rex.assert_called_once_with("Nobody talking now.")
        register.assert_called_once_with("Nobody talking now.")
        interaction._idle_outro_spoken = False

    def test_low_memory_idle_question_asks_profile_question_once(self):
        from intelligence import interaction

        interaction._low_memory_idle_questions_spoken.clear()
        question = {"key": "job", "text": "What do you do — professionally speaking?", "depth": 1}
        # A non-VIP, sparsely-known person (the creator/VIPs are skipped — see
        # test_low_memory_idle_question_skips_creator_and_vips).
        spoken = "I don't know you well yet, Alex, What do you do — professionally speaking?"
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
                return_value={"id": 1, "name": "Alex Carter"},
            ),
            mock.patch.object(interaction.question_budget, "can_ask", return_value=True),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch("audio.tts.ensure_cached", return_value=True),
            mock.patch.object(interaction.barge_guard, "user_speaking_now", return_value=False),
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
        speak.assert_called_once_with(
            spoken,
            emotion="curious",
            priority=1,
            pre_beat_ms=0,
            post_beat_ms_override=0,
            voice_settings=None,
        )
        transcript.assert_called_once_with("Rex", spoken)
        log_rex.assert_called_once_with(spoken)
        register.assert_called_once_with(spoken)
        save_q.assert_called_once_with(1, "job", spoken, 1)
        interaction._low_memory_idle_questions_spoken.clear()

    def test_low_memory_idle_question_skips_creator_and_vips(self):
        # Rex knows the creator (and VIPs) on sight, so the generic "get to know you
        # better, what's your favorite movie?" must NOT fire on them (live-logged
        # 2026-06-20: it fired on the creator right after reminiscing about his festival).
        from intelligence import interaction

        interaction._low_memory_idle_questions_spoken.clear()
        with (
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_SECS", 10.0),
            mock.patch.object(interaction.config, "LOW_MEMORY_PROFILE_MAX_FACTS", 4),
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_profile_fact_count", return_value=1),
            mock.patch.object(interaction, "_next_profile_question") as next_q,
            mock.patch.object(
                interaction.people_memory,
                "get_person",
                return_value={"id": 1, "name": "Bret Benziger"},  # the creator
            ),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            asked = interaction._maybe_low_memory_idle_question(
                idle_for=11.0,
                effective_idle_timeout=30.0,
            )

        self.assertFalse(asked)            # creator is skipped
        speak.assert_not_called()
        next_q.assert_not_called()         # gated before any question is even selected
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

    def test_profile_fact_count_ignores_appearance_enrollment_facts(self):
        from intelligence import interaction

        facts = [
            {"category": "appearance", "key": "build", "value": "average build"},
            {"category": "appearance", "key": "hair_color", "value": "brown"},
            {"category": "appearance", "key": "notable_features", "value": "glasses"},
            {"category": "appearance", "key": "skin_color", "value": "light"},
            {"category": "job", "key": "job_title", "value": "pilot"},
        ]
        with mock.patch.object(interaction.facts_memory, "get_facts", return_value=facts):
            count = interaction._profile_fact_count(1)

        self.assertEqual(count, 1)

    def test_low_memory_idle_question_skips_rich_profiles(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_profile_fact_count", return_value=20),
            mock.patch.object(interaction, "_next_profile_question") as next_q,
        ):
            asked = interaction._maybe_low_memory_idle_question(
                idle_for=20.0,
                effective_idle_timeout=30.0,
            )

        self.assertFalse(asked)
        next_q.assert_not_called()

    def test_low_memory_idle_question_skips_fresh_directed_look_context(self):
        from intelligence import interaction

        old_context = dict(interaction._directed_look_context)
        try:
            interaction._reset_directed_look_context()
            interaction._note_directed_look_context(direction="left")
            with (
                mock.patch.object(interaction.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
                mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
                mock.patch.object(interaction, "_next_profile_question") as next_q,
            ):
                asked = interaction._maybe_low_memory_idle_question(
                    idle_for=20.0,
                    effective_idle_timeout=30.0,
                )

            self.assertFalse(asked)
            next_q.assert_not_called()
        finally:
            interaction._directed_look_context.update(old_context)

    def test_wake_word_does_not_interrupt_current_question(self):
        from intelligence import interaction

        interaction._interrupted.clear()
        interaction._wake_word_fired.clear()
        try:
            with (
                mock.patch.object(interaction, "_wake_word_recognition_gesture"),
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=True),
                mock.patch.object(interaction.consciousness, "is_waiting_for_response", return_value=True),
            ):
                interaction._on_wake_word("Hey_rex")

            self.assertFalse(interaction._interrupted.is_set())
            self.assertTrue(interaction._wake_word_fired.is_set())
        finally:
            interaction._interrupted.clear()
            interaction._wake_word_fired.clear()

    def test_wake_word_callback_triggers_recognition_gesture(self):
        from intelligence import interaction

        interaction._wake_word_fired.clear()
        try:
            with (
                mock.patch.object(interaction, "_wake_word_recognition_gesture") as gesture,
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            ):
                interaction._on_wake_word("Hey_rex")

            gesture.assert_called_once_with("Hey_rex")
            self.assertTrue(interaction._wake_word_fired.is_set())
        finally:
            interaction._wake_word_fired.clear()

    def test_sleep_state_accepts_general_and_sleep_wake_words(self):
        from intelligence import interaction

        old_state = interaction.state_module.get_state()
        interaction._wake_word_fired.clear()
        interaction.state_module.set_state(interaction.State.SLEEP)
        try:
            with mock.patch.object(
                interaction.speech_queue,
                "is_speaking",
                return_value=False,
            ):
                interaction._on_wake_word("Hey_rex")
                self.assertTrue(interaction._wake_word_fired.is_set())
                with interaction._wake_lock:
                    self.assertEqual(interaction._last_wake_word, "Hey_rex")

                interaction._wake_word_fired.clear()
                interaction._on_wake_word("wakeuprex")
                self.assertTrue(interaction._wake_word_fired.is_set())
                with interaction._wake_lock:
                    self.assertEqual(interaction._last_wake_word, "wakeuprex")
        finally:
            interaction._wake_word_fired.clear()
            interaction.state_module.set_state(old_state)

    def test_wake_word_detector_uses_general_models_as_sleep_fallbacks(self):
        from audio import wake_word
        from state import State

        old_loaded = wake_word._loaded_models
        wake_word._loaded_models = frozenset({"Hey_rex", "Yo_robot", "wakeuprex"})
        try:
            self.assertEqual(
                wake_word._active_for_state(State.SLEEP),
                frozenset({"Hey_rex", "Yo_robot", "wakeuprex"}),
            )
        finally:
            wake_word._loaded_models = old_loaded

    def test_sleep_wake_transcript_requires_explicit_rex_wake_phrase(self):
        from intelligence import interaction

        for text in [
            "wake up rex",
            "wake up, Rex.",
            "please wake up DJ Rex",
            "R3X wake up",
            "wakeuprex",
        ]:
            with self.subTest(text=text):
                self.assertTrue(interaction._is_sleep_wake_transcript(text))

        for text in [
            "wake up",
            "hey rex",
            "wake me up rex",
            "don't wake up rex",
            "I should wake up Rex later",
        ]:
            with self.subTest(text=text):
                self.assertFalse(interaction._is_sleep_wake_transcript(text))

    def test_sleep_transcription_fallback_wakes_on_phrase(self):
        import numpy as np
        from intelligence import interaction

        previous_last = interaction._last_speech_at
        try:
            with (
                mock.patch.object(
                    interaction.transcription,
                    "transcribe",
                    return_value="wake up rex",
                ),
                mock.patch.object(interaction, "_wake_from_sleep", return_value="awake") as wake,
                mock.patch.object(interaction.time, "monotonic", return_value=123.0),
            ):
                self.assertTrue(
                    interaction._wake_from_sleep_if_transcribed(
                        np.ones(1600, dtype=np.float32)
                    )
                )

            wake.assert_called_once()
            self.assertEqual(interaction._last_speech_at, 123.0)
        finally:
            interaction._last_speech_at = previous_last

    def test_sleep_transcription_fallback_ignores_non_wake_phrase(self):
        import numpy as np
        from intelligence import interaction

        with (
            mock.patch.object(
                interaction.transcription,
                "transcribe",
                return_value="thanks for watching",
            ),
            mock.patch.object(interaction, "_wake_from_sleep") as wake,
        ):
            self.assertFalse(
                interaction._wake_from_sleep_if_transcribed(
                    np.ones(1600, dtype=np.float32)
                )
            )

        wake.assert_not_called()

    def test_wake_word_recognition_gesture_filters_and_cools_down(self):
        from intelligence import interaction
        from sequences import animations

        old_last = interaction._last_wake_word_gesture_at
        interaction._last_wake_word_gesture_at = 0.0
        try:
            with (
                mock.patch.object(interaction.config, "WAKE_WORD_RECOGNITION_GESTURE_ENABLED", True),
                mock.patch.object(
                    interaction.config,
                    "WAKE_WORD_RECOGNITION_GESTURE_MODELS",
                    ["Hey_rex"],
                ),
                mock.patch.object(interaction.config, "WAKE_WORD_RECOGNITION_GESTURE_COOLDOWN_SECS", 1.25),
                mock.patch.object(interaction.state_module, "get_state", return_value=interaction.State.ACTIVE),
                mock.patch.object(interaction.time, "monotonic", side_effect=[10.0, 10.5, 12.0]),
                mock.patch.object(animations, "wake_word_ack_wave", return_value=True) as wave,
            ):
                self.assertTrue(interaction._wake_word_recognition_gesture("Hey_rex"))
                self.assertFalse(interaction._wake_word_recognition_gesture("Hey_rex"))
                self.assertTrue(interaction._wake_word_recognition_gesture("Hey_rex"))
                self.assertFalse(interaction._wake_word_recognition_gesture("wakeuprex"))
                self.assertEqual(wave.call_count, 2)
        finally:
            interaction._last_wake_word_gesture_at = old_last

    def test_question_response_uses_longer_speech_preroll(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.config, "SPEECH_PREROLL_SECS", 0.45),
            mock.patch.object(interaction.config, "POST_QUESTION_SPEECH_PREROLL_SECS", 2.0),
            mock.patch.object(interaction, "_response_wait_active", return_value=True),
        ):
            self.assertEqual(interaction._speech_preroll_secs(), 2.0)

    def test_question_response_preroll_clamps_to_post_tts_capture_floor(self):
        from intelligence import interaction

        old_floor = interaction._listen_capture_floor_at
        try:
            interaction._listen_capture_floor_at = 100.0
            with (
                mock.patch.object(interaction.config, "SPEECH_PREROLL_SECS", 0.45),
                mock.patch.object(interaction.config, "POST_QUESTION_SPEECH_PREROLL_SECS", 2.0),
                mock.patch.object(interaction.config, "AUDIO_BUFFER_SECONDS", 30),
                mock.patch.object(interaction, "_response_wait_active", return_value=True),
            ):
                capture = interaction._speech_capture_secs(
                    speech_start_mono=100.5,
                    finished_mono=102.0,
                )
        finally:
            interaction._listen_capture_floor_at = old_floor

        self.assertAlmostEqual(capture, 2.0)

    def test_non_question_speech_uses_default_preroll(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.config, "SPEECH_PREROLL_SECS", 0.45),
            mock.patch.object(interaction.config, "POST_QUESTION_SPEECH_PREROLL_SECS", 2.0),
            mock.patch.object(interaction, "_response_wait_active", return_value=False),
        ):
            self.assertEqual(interaction._speech_preroll_secs(), 0.45)

    def test_bare_wake_address_detection(self):
        from intelligence import interaction

        for text in ("Hey Rex", "hey dj-rex", "DJ Rex", "yo robot", "R3X"):
            self.assertTrue(interaction._is_bare_wake_address(text), text)

        self.assertFalse(interaction._is_bare_wake_address("Hey Rex what time is it"))
        self.assertFalse(interaction._is_bare_wake_address("Rex play jazz"))

    def test_bare_wake_from_visible_unknown_should_prompt_for_identity(self):
        from intelligence import interaction

        self.assertTrue(
            interaction._bare_wake_should_ask_visible_unknown_identity(
                person_id=None,
                identity_prompt_active=False,
                has_unknown_visible_or_recent=True,
                game_conversation_lock=False,
            )
        )
        self.assertFalse(
            interaction._bare_wake_should_ask_visible_unknown_identity(
                person_id=1,
                identity_prompt_active=False,
                has_unknown_visible_or_recent=True,
                game_conversation_lock=False,
            )
        )

    def test_visible_unknown_identity_question_opens_reply_window(self):
        from intelligence import interaction

        old_until = interaction._identity_prompt_until
        old_exchange_count = interaction._session_exchange_count
        try:
            with (
                mock.patch.object(
                    interaction.llm,
                    "get_response",
                    return_value="Hey mystery passenger, what name am I saving?",
                ) as get_response,
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.conv_memory, "add_to_transcript") as transcript,
                mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
                mock.patch.object(interaction, "_register_rex_utterance") as register,
                mock.patch.object(interaction.time, "monotonic", return_value=100.0),
            ):
                response = interaction._ask_visible_unknown_identity_question(
                    "Hey Rex",
                    recent_engagement=None,
                    group_chatter_active=False,
                    source="test",
                )

            self.assertEqual(response, "Hey mystery passenger, what name am I saving?")
            self.assertGreater(interaction._identity_prompt_until, 100.0)
            get_response.assert_called_once()
            speak.assert_called_once_with("Hey mystery passenger, what name am I saving?")
            transcript.assert_called_once_with(
                "Rex",
                "Hey mystery passenger, what name am I saving?",
            )
            log_rex.assert_called_once_with("Hey mystery passenger, what name am I saving?")
            register.assert_called_once_with("Hey mystery passenger, what name am I saving?")
        finally:
            interaction._identity_prompt_until = old_until
            interaction._session_exchange_count = old_exchange_count

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

    def test_bare_identity_name_strips_echoed_question_tail(self):
        """Rex's own '...save for you?' tail can bleed into the mic ahead of the
        answer ('for you, Bret.'). The bare-name path must still recover 'Bret'
        instead of dropping it — the failure that produced the contradictory
        'Bret, got it… what do I call you?' (live-logged 2026-06-18)."""
        from intelligence import interaction

        for garbled in ("for you, Bret.", "for you, Bret", "for you, Mary Jane"):
            extracted = interaction._extract_introduced_name(
                garbled, allow_bare_name=True
            )
            self.assertIsNotNone(extracted, garbled)
        self.assertEqual(
            interaction._extract_introduced_name("for you, Bret.", allow_bare_name=True),
            "Bret",
        )
        # A genuine comma-split two-token name is still joined, not truncated.
        self.assertEqual(
            interaction._extract_introduced_name("Mary, Jane", allow_bare_name=True),
            "Mary Jane",
        )
        self.assertEqual(
            interaction._extract_introduced_name("Mary, Jane.", allow_bare_name=True),
            "Mary Jane",
        )

    def test_first_name_helper_handles_missing_or_blank_names(self):
        from intelligence import interaction

        self.assertEqual(interaction._first_name_or(None, "friend"), "friend")
        self.assertEqual(interaction._first_name_or("   ", "friend"), "friend")
        self.assertEqual(interaction._first_name_or("Bret Benziger", "friend"), "Bret")

    def test_identity_prompt_name_reply_acknowledges_without_router_correction(self):
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction, person_specials

        old_people = interaction.world_state.get("people")
        old_until = interaction._identity_prompt_until
        old_exchange_count = interaction._session_exchange_count
        old_pending_offscreen = interaction._pending_offscreen_identify
        old_pending_face_reveal = interaction._pending_face_reveal_confirm
        old_pending_relationship = interaction._pending_post_greet_relationship[0]
        try:
            interaction.world_state.update(
                "people",
                [{
                    "id": "slot:person_1",
                    "face_id": None,
                    "voice_id": None,
                    "person_db_id": None,
                    "face_visible": True,
                }],
            )
            interaction._identity_prompt_until = interaction.time.monotonic() + 30.0
            interaction._pending_offscreen_identify = None
            interaction._pending_face_reveal_confirm = None
            interaction._pending_post_greet_relationship[0] = None

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("JT", None, None, 0.0, 0.0),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_game_suppresses_conversation",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "consume_continuation",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "classify",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "start_sequence"))
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "end_sequence"))
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "get_recent_engagement",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_identity_prompt_request",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_relationship_prompt_request",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.consciousness, "mark_engagement"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "note_person_spoke"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "clear_response_wait"))
                stack.enter_context(mock.patch.object(interaction.speech_queue, "drop_by_tag"))
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_note_voice_turn_for_group_chatter",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_audio_group_chatter_active",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_resolve_anonymous_speaker_slot",
                        return_value=("unknown_voice_1", None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_last_name_reply",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_intro_last_name_reply",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_existing_common_first_name_last_name_reply",
                        return_value=None,
                    )
                )
                enroll = stack.enter_context(
                    mock.patch.object(interaction, "_enroll_new_person", return_value=2)
                )
                retire_slot = stack.enter_context(
                    mock.patch.object(interaction, "_retire_anonymous_speaker_slot")
                )
                stack.enter_context(
                    mock.patch.object(interaction, "_dismiss_pending_consent_prompts")
                )
                speak = stack.enter_context(
                    mock.patch.object(interaction, "_speak_blocking", return_value=True)
                )
                add_transcript = stack.enter_context(
                    mock.patch.object(interaction.conv_memory, "add_to_transcript")
                )
                log_heard = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_heard")
                )
                log_rex = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_rex")
                )
                register = stack.enter_context(
                    mock.patch.object(interaction, "_register_rex_utterance")
                )
                name_update = stack.enter_context(
                    mock.patch.object(interaction, "_handle_name_update_request")
                )
                decide = stack.enter_context(
                    mock.patch.object(interaction.action_router, "decide")
                )
                interaction._handle_speech_segment(np.ones(16, dtype=np.float32))

            enroll.assert_called_once()
            self.assertEqual(enroll.call_args.args[0], "JT")
            self.assertTrue(enroll.call_args.kwargs.get("defer_face_enrollment"))
            retire_slot.assert_called_once_with(
                "unknown_voice_1",
                person_id=2,
                person_name="JT",
            )
            expected_ack = person_specials.jt_volleyball_intro_ack("JT")
            speak.assert_called_once_with(
                expected_ack,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            add_transcript.assert_any_call("JT", "JT")
            add_transcript.assert_any_call("Rex", expected_ack)
            log_heard.assert_called_once_with("JT", "JT")
            log_rex.assert_called_once_with(expected_ack)
            register.assert_called_once_with(expected_ack)
            name_update.assert_not_called()
            decide.assert_not_called()
        finally:
            interaction.world_state.update("people", old_people)
            interaction._identity_prompt_until = old_until
            interaction._session_exchange_count = old_exchange_count
            interaction._pending_offscreen_identify = old_pending_offscreen
            interaction._pending_face_reveal_confirm = old_pending_face_reveal
            interaction._pending_post_greet_relationship[0] = old_pending_relationship

    def test_identity_prompt_unparsable_reply_reasks_instead_of_contradicting(self):
        """When Rex just asked an unknown person their name and the reply has no
        usable name in it (e.g. only his own '...for you?' tail bled into the
        mic), he should gently RE-ASK — not route the turn to the LLM, which was
        handed both the transcript and the 'ask their name' agenda and answered
        'Bret, got it… what do I call you?' (live-logged 2026-06-18)."""
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction

        old_people = interaction.world_state.get("people")
        old_until = interaction._identity_prompt_until
        old_exchange_count = interaction._session_exchange_count
        old_reask = interaction._identity_reask_count
        old_pending_offscreen = interaction._pending_offscreen_identify
        old_pending_face_reveal = interaction._pending_face_reveal_confirm
        old_pending_relationship = interaction._pending_post_greet_relationship[0]
        try:
            interaction.world_state.update(
                "people",
                [{
                    "id": "slot:person_1",
                    "face_id": None,
                    "voice_id": None,
                    "person_db_id": None,
                    "face_visible": True,
                }],
            )
            interaction._identity_prompt_until = interaction.time.monotonic() + 30.0
            interaction._identity_reask_count = 0
            interaction._pending_offscreen_identify = None
            interaction._pending_face_reveal_confirm = None
            interaction._pending_post_greet_relationship[0] = None

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("for you", None, None, 0.0, 0.0),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_game_suppresses_conversation",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion, "consume_continuation",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion, "classify", return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "start_sequence"))
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "end_sequence"))
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness, "get_recent_engagement",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness, "consume_identity_prompt_request",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness, "consume_relationship_prompt_request",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.consciousness, "mark_engagement"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "note_person_spoke"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "clear_response_wait"))
                stack.enter_context(mock.patch.object(interaction.speech_queue, "drop_by_tag"))
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_note_voice_turn_for_group_chatter",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_audio_group_chatter_active", return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_resolve_anonymous_speaker_slot",
                        return_value=("unknown_voice_1", None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_handle_common_first_name_last_name_reply",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_handle_common_first_name_intro_last_name_reply",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction, "_handle_existing_common_first_name_last_name_reply",
                        return_value=None,
                    )
                )
                enroll = stack.enter_context(
                    mock.patch.object(interaction, "_enroll_new_person", return_value=2)
                )
                stack.enter_context(
                    mock.patch.object(interaction, "_dismiss_pending_consent_prompts")
                )
                speak = stack.enter_context(
                    mock.patch.object(interaction, "_speak_blocking", return_value=True)
                )
                stack.enter_context(
                    mock.patch.object(interaction.conv_memory, "add_to_transcript")
                )
                stack.enter_context(mock.patch.object(interaction.conv_log, "log_heard"))
                log_rex = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_rex")
                )
                register = stack.enter_context(
                    mock.patch.object(interaction, "_register_rex_utterance")
                )
                decide = stack.enter_context(
                    mock.patch.object(interaction.action_router, "decide")
                )
                interaction._handle_speech_segment(np.ones(16, dtype=np.float32))

            # No name was captured, so we must NOT enroll a bogus person and must
            # NOT fall through to the router/LLM (the contradictory path).
            enroll.assert_not_called()
            decide.assert_not_called()
            # Instead Rex re-asks with one of the canned re-ask lines, once.
            speak.assert_called_once()
            spoken = speak.call_args.args[0]
            self.assertIn(spoken, interaction.config.IDENTITY_PROMPT_REASK_LINES)
            log_rex.assert_called_once_with(spoken)
            register.assert_called_once_with(spoken)
            self.assertEqual(interaction._identity_reask_count, 1)
        finally:
            interaction.world_state.update("people", old_people)
            interaction._identity_prompt_until = old_until
            interaction._session_exchange_count = old_exchange_count
            interaction._identity_reask_count = old_reask
            interaction._pending_offscreen_identify = old_pending_offscreen
            interaction._pending_face_reveal_confirm = old_pending_face_reveal
            interaction._pending_post_greet_relationship[0] = old_pending_relationship

    def test_empty_transcript_from_idle_returns_to_idle(self):
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction
        from state import State

        old_state = interaction.state_module.get_state()
        try:
            interaction.state_module.set_state(State.ACTIVE)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("", None, None, 0.0, 0.0),
                    )
                )
                interaction._handle_speech_segment(
                    np.ones(16, dtype=np.float32),
                    from_idle_activation=True,
                )
            # A blank transcript (false VAD trigger / filtered hallucination) that
            # only flipped Rex to ACTIVE must drop straight back to IDLE instead of
            # camping in ACTIVE until the conversation idle timeout.
            self.assertEqual(interaction.state_module.get_state(), State.IDLE)
        finally:
            interaction.state_module.set_state(old_state)

    def test_empty_transcript_mid_conversation_stays_active(self):
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction
        from state import State

        old_state = interaction.state_module.get_state()
        try:
            interaction.state_module.set_state(State.ACTIVE)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("", None, None, 0.0, 0.0),
                    )
                )
                interaction._handle_speech_segment(
                    np.ones(16, dtype=np.float32),
                    from_idle_activation=False,
                )
            # A blank trigger DURING an active exchange must not end the session —
            # only idle-activations drop back to IDLE.
            self.assertEqual(interaction.state_module.get_state(), State.ACTIVE)
        finally:
            interaction.state_module.set_state(old_state)

    def test_pending_identity_prompt_reply_from_idle_enrolls_before_background_filter(self):
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction, person_specials

        old_people = interaction.world_state.get("people")
        old_until = interaction._identity_prompt_until
        old_exchange_count = interaction._session_exchange_count
        old_pending_offscreen = interaction._pending_offscreen_identify
        old_pending_face_reveal = interaction._pending_face_reveal_confirm
        old_pending_relationship = interaction._pending_post_greet_relationship[0]
        try:
            # Mirrors the live failure: Rex asked "who are you?", the local
            # interaction window was not armed yet, and the face tracker had no
            # current unknown box when the user answered from IDLE.
            interaction.world_state.update("people", [])
            interaction._identity_prompt_until = 0.0
            interaction._pending_offscreen_identify = None
            interaction._pending_face_reveal_confirm = None
            interaction._pending_post_greet_relationship[0] = None

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("Bret Benziger", None, None, 0.0, 0.0),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_game_suppresses_conversation",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "consume_continuation",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "classify",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "start_sequence"))
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "end_sequence"))
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "get_recent_engagement",
                        return_value=None,
                    )
                )
                consume_prompt = stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_identity_prompt_request",
                        return_value=True,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_relationship_prompt_request",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.consciousness, "mark_engagement"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "note_person_spoke"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "clear_response_wait"))
                stack.enter_context(mock.patch.object(interaction.speech_queue, "drop_by_tag"))
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_note_voice_turn_for_group_chatter",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_audio_group_chatter_active",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_has_unknown_visible_or_recent",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_resolve_anonymous_speaker_slot",
                        return_value=("unknown_voice_1", None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_pending_identity_match_confirmation",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_last_name_reply",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_intro_last_name_reply",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_existing_common_first_name_last_name_reply",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_maybe_ask_identity_match_confirmation",
                        return_value=False,
                    )
                )
                enroll = stack.enter_context(
                    mock.patch.object(interaction, "_enroll_new_person", return_value=2)
                )
                retire_slot = stack.enter_context(
                    mock.patch.object(interaction, "_retire_anonymous_speaker_slot")
                )
                stack.enter_context(
                    mock.patch.object(interaction, "_dismiss_pending_consent_prompts")
                )
                speak = stack.enter_context(
                    mock.patch.object(interaction, "_speak_blocking", return_value=True)
                )
                add_transcript = stack.enter_context(
                    mock.patch.object(interaction.conv_memory, "add_to_transcript")
                )
                log_heard = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_heard")
                )
                log_rex = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_rex")
                )
                register = stack.enter_context(
                    mock.patch.object(interaction, "_register_rex_utterance")
                )
                name_update = stack.enter_context(
                    mock.patch.object(interaction, "_handle_name_update_request")
                )
                decide = stack.enter_context(
                    mock.patch.object(interaction.action_router, "decide")
                )

                interaction._handle_speech_segment(
                    np.ones(16, dtype=np.float32),
                    from_idle_activation=True,
                )

            consume_prompt.assert_called()
            enroll.assert_called_once()
            self.assertEqual(enroll.call_args.args[0], "Bret Benziger")
            self.assertTrue(enroll.call_args.kwargs.get("defer_face_enrollment"))
            retire_slot.assert_called_once_with(
                "unknown_voice_1",
                person_id=2,
                person_name="Bret Benziger",
            )
            expected_ack = person_specials.rex_creator_intro_ack("Bret Benziger")
            speak.assert_called_once_with(
                expected_ack,
                emotion="happy",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            add_transcript.assert_any_call("Bret Benziger", "Bret Benziger")
            add_transcript.assert_any_call("Rex", expected_ack)
            log_heard.assert_called_once_with("Bret Benziger", "Bret Benziger")
            log_rex.assert_called_once_with(expected_ack)
            register.assert_called_once_with(expected_ack)
            name_update.assert_not_called()
            decide.assert_not_called()
        finally:
            interaction.world_state.update("people", old_people)
            interaction._identity_prompt_until = old_until
            interaction._session_exchange_count = old_exchange_count
            interaction._pending_offscreen_identify = old_pending_offscreen
            interaction._pending_face_reveal_confirm = old_pending_face_reveal
            interaction._pending_post_greet_relationship[0] = old_pending_relationship

    def test_existing_common_name_prompt_logs_human_turn_before_returning(self):
        from contextlib import ExitStack
        import numpy as np
        from intelligence import interaction

        old_people = interaction.world_state.get("people")
        old_exchange_count = interaction._session_exchange_count
        old_pending_offscreen = interaction._pending_offscreen_identify
        old_pending_face_reveal = interaction._pending_face_reveal_confirm
        old_pending_relationship = interaction._pending_post_greet_relationship[0]
        try:
            interaction.world_state.update("people", [
                {"person_db_id": 1, "face_id": "Bret"},
            ])
            interaction._pending_offscreen_identify = None
            interaction._pending_face_reveal_confirm = None
            interaction._pending_post_greet_relationship[0] = None

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(interaction.random, "randint", return_value=0)
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_process_audio",
                        return_value=("Nice to meet you too", 1, "Bret", 0.502, 0.0),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_game_suppresses_conversation",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "consume_continuation",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.turn_completion,
                        "classify",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "start_sequence"))
                stack.enter_context(mock.patch.object(interaction.echo_cancel, "end_sequence"))
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "get_recent_engagement",
                        return_value={"person_id": 1, "name": "Bret"},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_identity_prompt_request",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction.consciousness,
                        "consume_relationship_prompt_request",
                        return_value=None,
                    )
                )
                stack.enter_context(mock.patch.object(interaction.consciousness, "mark_engagement"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "note_person_spoke"))
                stack.enter_context(mock.patch.object(interaction.consciousness, "clear_response_wait"))
                stack.enter_context(mock.patch.object(interaction.speech_queue, "drop_by_tag"))
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_note_voice_turn_for_group_chatter",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_audio_group_chatter_active",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_has_unknown_visible_or_recent",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_resolve_anonymous_speaker_slot",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_pending_identity_match_confirmation",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_pending_prompted_name_confirmation",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_last_name_reply",
                        return_value=(None, None, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_common_first_name_intro_last_name_reply",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_handle_existing_common_first_name_last_name_reply",
                        return_value=None,
                    )
                )
                prompt = stack.enter_context(
                    mock.patch.object(
                        interaction,
                        "_maybe_prompt_existing_common_first_name",
                        return_value="Bret. Last name, before the directory panics?",
                    )
                )
                speak = stack.enter_context(
                    mock.patch.object(interaction, "_speak_blocking", return_value=True)
                )
                add_transcript = stack.enter_context(
                    mock.patch.object(interaction.conv_memory, "add_to_transcript")
                )
                log_heard = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_heard")
                )
                log_rex = stack.enter_context(
                    mock.patch.object(interaction.conv_log, "log_rex")
                )

                interaction._handle_speech_segment(np.ones(16, dtype=np.float32))

            prompt.assert_called_once()
            log_heard.assert_called_once_with("Bret", "Nice to meet you too")
            add_transcript.assert_any_call("Bret", "Nice to meet you too")
            speak.assert_called_once_with(
                "Bret. Last name, before the directory panics?",
                emotion="curious",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            log_rex.assert_called_once_with(
                "Bret. Last name, before the directory panics?"
            )
        finally:
            interaction.world_state.update("people", old_people)
            interaction._session_exchange_count = old_exchange_count
            interaction._pending_offscreen_identify = old_pending_offscreen
            interaction._pending_face_reveal_confirm = old_pending_face_reveal
            interaction._pending_post_greet_relationship[0] = old_pending_relationship

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
        self.assertEqual(
            interaction._extract_name_update("that's not Bret, I'm Daniel"),
            "Daniel",
        )
        self.assertEqual(interaction._extract_name_update("rename me to JT"), "JT")
        self.assertIsNone(interaction._extract_name_update("call me both"))
        self.assertIsNone(interaction._extract_name_update("that's not my name"))

    def test_prompted_identity_reply_preserves_comma_split_name(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._extract_introduced_name(
                "Shrek, Benziger",
                allow_bare_name=True,
            ),
            "Shrek Benziger",
        )
        self.assertTrue(
            interaction._prompted_name_reply_needs_confirmation(
                "Shrek, Benziger",
                "Shrek Benziger",
            )
        )
        self.assertFalse(
            interaction._prompted_name_reply_needs_confirmation(
                "Bret Benziger",
                "Bret Benziger",
            )
        )

    def test_prompted_identity_reply_strips_prompt_echo(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._extract_introduced_name(
                "What name should I save for you? Bret Penziker",
                allow_bare_name=True,
            ),
            "Bret Penziker",
        )

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
        self.assertIsNone(interaction._extract_last_name_reply("No", "John"))
        self.assertIsNone(
            interaction._extract_last_name_reply(
                "No, I'm not going to He has a memory Ah",
                "John",
            )
        )

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
            "No",
            "No thanks",
            "No, I'm not going to",
            "No, I'm not going to He has a memory Ah",
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

    def test_common_first_name_introduction_enrolls_single_name_for_later_prompt(self):
        from intelligence import interaction

        parsed = interaction.introductions.IntroductionParse(
            is_introduction=True,
            name="Daniel",
            relationship="acquaintance",
            subject_kind="person",
        )
        interaction._pending_common_first_name_introduction = None
        try:
            with (
                mock.patch.object(
                    interaction,
                    "_resolve_existing_visible_introduced_person",
                    return_value=None,
                ),
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
                mock.patch.object(
                    interaction,
                    "_mark_single_name_for_later_last_name",
                ) as mark_later,
            ):
                response = interaction._handle_introduction_parse(
                    parsed,
                    introducer_id=1,
                    introducer_name="Bret Benziger",
                    visible_newcomer=True,
                )

            self.assertEqual(response, "Ack Daniel.")
            enroll.assert_called_once_with(
                "Daniel",
                1,
                "Bret Benziger",
                "acquaintance",
                enroll_visible_face=True,
            )
            ack.assert_called_once_with(
                1,
                "Bret Benziger",
                3,
                "Daniel",
                "acquaintance",
                subject_kind="person",
                visible_newcomer=True,
            )
            mark_later.assert_called_once_with(3, "Daniel")
            self.assertIsNone(interaction._pending_common_first_name_introduction)
        finally:
            interaction._pending_common_first_name_introduction = None

    def test_relationship_only_introduction_opens_pending_slot(self):
        from intelligence import interaction

        parsed = interaction.introductions.IntroductionParse(
            is_introduction=True,
            name=None,
            relationship="sister",
            subject_kind="person",
            needs_name=True,
        )
        interaction._pending_introduction = None
        try:
            with mock.patch.object(
                interaction.llm,
                "get_response",
                return_value="What name am I filing for your sister?",
            ) as llm_response:
                response = interaction._handle_introduction_parse(
                    parsed,
                    introducer_id=1,
                    introducer_name="Bret Benziger",
                    visible_newcomer=False,
                )

            self.assertEqual(response, "What name am I filing for your sister?")
            self.assertIsNotNone(interaction._pending_introduction)
            self.assertEqual(interaction._pending_introduction["introducer_id"], 1)
            self.assertEqual(interaction._pending_introduction["relationship"], "sister")
            self.assertFalse(interaction._pending_introduction["visible_newcomer"])
            prompt = llm_response.call_args.args[0]
            self.assertIn("meet their sister", prompt)
            self.assertNotIn("visible newcomer", prompt)
        finally:
            interaction._pending_introduction = None

    def test_introduction_name_matches_existing_visible_person_before_enrolling(self):
        from intelligence import interaction

        parsed = interaction.introductions.IntroductionParse(
            is_introduction=True,
            name="Jennifer",
            relationship="sister",
            subject_kind="person",
        )
        interaction._pending_introduction = {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "relationship": "sister",
            "visible_newcomer": False,
            "asked_at": interaction.time.monotonic(),
        }
        try:
            with (
                mock.patch.object(
                    interaction.people_memory,
                    "find_person_by_name",
                    return_value={"id": 4, "name": "Jennifer"},
                ) as find_person,
                mock.patch.object(
                    interaction,
                    "_known_person_visible_recently",
                    return_value=True,
                ) as visible_recent,
                mock.patch.object(
                    interaction,
                    "_store_introduction_memories",
                ) as store_intro,
                mock.patch.object(
                    interaction,
                    "_intro_ack_and_followup",
                    return_value="Jennifer, welcome aboard.",
                ) as ack,
                mock.patch.object(interaction, "_enroll_introduced_person") as enroll,
            ):
                response = interaction._handle_introduction_parse(
                    parsed,
                    introducer_id=1,
                    introducer_name="Bret Benziger",
                    visible_newcomer=False,
                )

            self.assertEqual(response, "Jennifer, welcome aboard.")
            find_person.assert_called_once_with("Jennifer")
            visible_recent.assert_called_once_with(4)
            store_intro.assert_called_once_with(
                1,
                "Bret Benziger",
                4,
                "Jennifer",
                "sister",
            )
            ack.assert_called_once_with(
                1,
                "Bret Benziger",
                4,
                "Jennifer",
                "sister",
                subject_kind="person",
                visible_newcomer=True,
            )
            enroll.assert_not_called()
            self.assertIsNone(interaction._pending_introduction)
        finally:
            interaction._pending_introduction = None

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

    def test_returning_common_first_name_person_waits_for_longer_conversation(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = None
        interaction._common_first_name_prompted_this_session.clear()
        interaction._session_person_turn_counts.clear()
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

            self.assertIsNone(response)
            self.assertIsNone(interaction._pending_existing_common_first_name)
            self.assertNotIn(3, interaction._common_first_name_prompted_this_session)

            min_turns = interaction._last_name_prompt_min_person_turns()
            interaction._session_person_turn_counts[3] = min_turns
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
            interaction._session_person_turn_counts.clear()

    def test_returning_common_first_name_prompt_defers_for_commands(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = None
        interaction._common_first_name_prompted_this_session.clear()
        interaction._session_person_turn_counts.clear()
        try:
            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=False,
            ):
                response = interaction._maybe_prompt_existing_common_first_name(
                    3,
                    "Gloria",
                    current_text="What time is it?",
                )

            self.assertIsNone(response)
            self.assertIsNone(interaction._pending_existing_common_first_name)
            self.assertNotIn(3, interaction._common_first_name_prompted_this_session)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()
            interaction._session_person_turn_counts.clear()

    def test_returning_single_non_common_name_waits_for_longer_conversation(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = None
        interaction._common_first_name_prompted_this_session.clear()
        interaction._session_person_turn_counts.clear()
        try:
            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=False,
            ):
                response = interaction._maybe_prompt_existing_common_first_name(
                    7,
                    "Bret",
                )

            self.assertIsNone(response)
            self.assertIsNone(interaction._pending_existing_common_first_name)

            interaction._session_person_turn_counts[7] = interaction._last_name_prompt_min_person_turns()
            with mock.patch.object(
                interaction,
                "_has_declined_last_name",
                return_value=False,
            ):
                response = interaction._maybe_prompt_existing_common_first_name(
                    7,
                    "Bret",
                )

            self.assertIn("Bret", response)
            self.assertEqual(
                interaction._pending_existing_common_first_name["person_id"],
                7,
            )
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()
            interaction._session_person_turn_counts.clear()

    def test_last_name_ask_deferred_during_conversational_turn(self):
        # BUG-4: the ask must not barge into a substantive answer (the logged
        # "It's vodka and orange juice" hijack).
        from intelligence import interaction

        interaction._pending_existing_common_first_name = None
        interaction._common_first_name_prompted_this_session.clear()
        interaction._session_person_turn_counts.clear()
        try:
            interaction._session_person_turn_counts[7] = (
                interaction._last_name_prompt_min_person_turns()
            )
            with mock.patch.object(interaction, "_has_declined_last_name", return_value=False):
                response = interaction._maybe_prompt_existing_common_first_name(
                    7, "Bret", current_text="It's vodka and orange juice",
                )
            self.assertIsNone(response)
            self.assertIsNone(interaction._pending_existing_common_first_name)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._common_first_name_prompted_this_session.clear()
            interaction._session_person_turn_counts.clear()

    def test_unusual_last_name_requires_confirmation_before_rename(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 7, "first_name": "Bret", "asked_at": interaction.time.monotonic(),
        }
        interaction._pending_last_name_confirm = None
        try:
            with mock.patch.object(interaction.people_memory, "rename_person") as rename:
                response = interaction._handle_existing_common_first_name_last_name_reply(
                    "Bat-tigger"
                )
            rename.assert_not_called()
            self.assertIsNotNone(interaction._pending_last_name_confirm)
            self.assertEqual(interaction._pending_last_name_confirm["last_name"], "Bat-tigger")
            self.assertIn("Bat-tigger", response)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._pending_last_name_confirm = None

    def test_clear_last_name_phrase_still_renames_directly(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 7, "first_name": "Bret", "asked_at": interaction.time.monotonic(),
        }
        interaction._pending_last_name_confirm = None
        try:
            with mock.patch.object(
                interaction.people_memory, "rename_person", return_value=True
            ) as rename, mock.patch.object(interaction, "_refresh_world_state_person_name"):
                response = interaction._handle_existing_common_first_name_last_name_reply(
                    "my last name is Benziger"
                )
            rename.assert_called_once()
            self.assertEqual(rename.call_args.args[1], "Bret Benziger")
            self.assertIn("Benziger", response)
            self.assertIsNone(interaction._pending_last_name_confirm)
        finally:
            interaction._pending_existing_common_first_name = None
            interaction._pending_last_name_confirm = None

    def test_unusual_last_name_confirmed_then_renames(self):
        from intelligence import interaction

        # Bare affirmations ('yep'/'correct'/'sure') must commit the PENDING
        # surname, never become the surname themselves (was 'Bret Yep').
        for reply in ("yes that's right", "yep", "correct", "sure"):
            interaction._pending_last_name_confirm = {
                "person_id": 7, "first_name": "Bret", "last_name": "Bat-tigger",
                "asked_at": interaction.time.monotonic(),
            }
            try:
                with mock.patch.object(
                    interaction.people_memory, "rename_person", return_value=True
                ) as rename, mock.patch.object(interaction, "_refresh_world_state_person_name"):
                    interaction._handle_last_name_confirm_reply(reply)
                rename.assert_called_once()
                self.assertEqual(rename.call_args.args[1], "Bret Bat-tigger", reply)
                self.assertIsNone(interaction._pending_last_name_confirm)
            finally:
                interaction._pending_last_name_confirm = None

    def test_last_name_confirm_correction_renames_to_corrected(self):
        from intelligence import interaction

        interaction._pending_last_name_confirm = {
            "person_id": 7, "first_name": "Bret", "last_name": "Bat-tigger",
            "asked_at": interaction.time.monotonic(),
        }
        try:
            with mock.patch.object(
                interaction.people_memory, "rename_person", return_value=True
            ) as rename, mock.patch.object(interaction, "_refresh_world_state_person_name"):
                response = interaction._handle_last_name_confirm_reply(
                    "no, my last name is Benziger"
                )
            rename.assert_called_once()
            self.assertEqual(rename.call_args.args[1], "Bret Benziger")
            self.assertIsNone(interaction._pending_last_name_confirm)
        finally:
            interaction._pending_last_name_confirm = None

    def test_first_name_match_asks_before_aliasing_existing_person(self):
        from intelligence import interaction
        import numpy as np

        old_pending = interaction._pending_identity_match_confirmation
        old_exchange_count = interaction._session_exchange_count
        interaction._pending_identity_match_confirmation = None
        try:
            with (
                mock.patch.object(
                    interaction.people_memory,
                    "find_potential_person_match",
                    return_value={
                        "match_type": "first_name",
                        "person": {"id": 1, "name": "Bret Benziger"},
                        "candidate_name": "Bret",
                    },
                ),
                mock.patch.object(interaction, "_speak_blocking") as speak,
                mock.patch.object(interaction.conv_memory, "add_to_transcript"),
                mock.patch.object(interaction.conv_log, "log_rex"),
                mock.patch.object(interaction, "_register_rex_utterance"),
            ):
                asked = interaction._maybe_ask_identity_match_confirmation(
                    "Bret",
                    np.ones(16, dtype=np.float32),
                    anonymous_speaker_label="unknown_voice_1",
                )

            self.assertTrue(asked)
            prompt = speak.call_args.args[0]
            # Confirm-before-aliasing prompt names the existing person and asks.
            self.assertIn("Bret Benziger", prompt)
            self.assertIn("?", prompt)
            self.assertEqual(
                interaction._pending_identity_match_confirmation["existing_person_id"],
                1,
            )
            self.assertEqual(
                interaction._pending_identity_match_confirmation["candidate_name"],
                "Bret",
            )
        finally:
            interaction._pending_identity_match_confirmation = old_pending
            interaction._session_exchange_count = old_exchange_count

    def test_identity_match_confirmation_yes_adds_alias_to_existing_person(self):
        from intelligence import interaction
        import numpy as np

        interaction._pending_identity_match_confirmation = {
            "candidate_name": "Bret",
            "existing_person_id": 1,
            "existing_name": "Bret Benziger",
            "match_type": "first_name",
            "audio": np.ones(16, dtype=np.float32),
            "anonymous_speaker_label": "unknown_voice_1",
            "asked_at": interaction.time.monotonic(),
        }
        try:
            with (
                mock.patch.object(interaction.people_memory, "add_alias", return_value=True) as add_alias,
                mock.patch.object(interaction, "_attach_identity_sample_to_person") as attach,
                mock.patch.object(interaction, "_retire_anonymous_speaker_slot") as retire,
            ):
                response, person_id, person_name = (
                    interaction._handle_pending_identity_match_confirmation(
                        "yes",
                        np.ones(16, dtype=np.float32),
                    )
                )

            self.assertEqual(person_id, 1)
            self.assertEqual(person_name, "Bret Benziger")
            self.assertIn("Bret Benziger", response)
            add_alias.assert_called_once_with(1, "Bret", source="confirmed_first_name")
            attach.assert_called_once()
            retire.assert_called_once_with(
                "unknown_voice_1",
                person_id=1,
                person_name="Bret Benziger",
            )
            self.assertIsNone(interaction._pending_identity_match_confirmation)
        finally:
            interaction._pending_identity_match_confirmation = None

    def test_identity_match_confirmation_no_single_name_asks_for_last_name(self):
        from intelligence import interaction
        import numpy as np

        interaction._pending_identity_match_confirmation = {
            "candidate_name": "Bret",
            "existing_person_id": 1,
            "existing_name": "Bret Benziger",
            "match_type": "first_name",
            "audio": np.ones(16, dtype=np.float32),
            "asked_at": interaction.time.monotonic(),
        }
        interaction._pending_common_first_name_identity = None
        try:
            response, person_id, person_name = (
                interaction._handle_pending_identity_match_confirmation(
                    "no",
                    np.ones(16, dtype=np.float32),
                )
            )

            self.assertIsNone(person_id)
            self.assertIsNone(person_name)
            self.assertIn("last name", response)
            self.assertEqual(
                interaction._pending_common_first_name_identity["first_name"],
                "Bret",
            )
            self.assertIsNone(interaction._pending_identity_match_confirmation)
        finally:
            interaction._pending_identity_match_confirmation = None
            interaction._pending_common_first_name_identity = None

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

    def test_pending_existing_common_first_name_reply_target_sticks_to_person(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 3,
            "first_name": "Gloria",
            "asked_at": interaction.time.monotonic(),
        }
        try:
            self.assertEqual(
                interaction._pending_existing_common_first_name_reply_target("Carter"),
                (3, "Gloria"),
            )
            self.assertEqual(
                interaction._pending_existing_common_first_name_reply_target(
                    "my last name is Carter"
                ),
                (3, "Gloria"),
            )
            self.assertIsNone(
                interaction._pending_existing_common_first_name_reply_target(
                    "What time is it?"
                )
            )
            self.assertEqual(
                interaction._pending_existing_common_first_name_reply_target(
                    "No, I'm not going to He has a memory Ah"
                ),
                (3, "Gloria"),
            )
        finally:
            interaction._pending_existing_common_first_name = None

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

    def test_returning_common_first_name_implicit_refusal_is_not_saved_as_no(self):
        from intelligence import interaction

        interaction._pending_existing_common_first_name = {
            "person_id": 3,
            "first_name": "Daniel",
            "asked_at": interaction.time.monotonic(),
        }
        interaction._common_first_name_prompted_this_session.clear()
        try:
            with (
                mock.patch.object(interaction.people_memory, "rename_person") as rename,
                mock.patch.object(interaction, "_remember_last_name_declined") as remember,
            ):
                response = interaction._handle_existing_common_first_name_last_name_reply(
                    "No, I'm not going to He has a memory Ah"
                )

            self.assertIn("Daniel", response)
            rename.assert_not_called()
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
            mock.patch.object(interaction.people_memory, "find_person_by_name", return_value=None),
            mock.patch.object(interaction.people_memory, "rename_person", return_value=True) as rename,
            mock.patch.object(interaction, "_refresh_world_state_person_name") as refresh,
            mock.patch.object(
                interaction.repair_moves, "pick_recovery_line",
                return_value="I'm sure we'll have better luck next time!",
            ),
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

    def test_name_update_existing_person_does_not_rename_current_person(self):
        from intelligence import interaction
        from intelligence import repair_moves

        repair_moves.clear()
        _prev_exchange_count = interaction._session_exchange_count
        try:
            with (
                mock.patch.object(
                    interaction,
                    "_resolve_name_update_target",
                    return_value=(1, "Bret"),
                ),
                mock.patch.object(
                    interaction.people_memory,
                    "find_person_by_name",
                    return_value={"id": 3, "name": "Daniel"},
                ),
                mock.patch.object(
                    interaction,
                    "_known_person_visible_recently",
                    return_value=False,
                ),
                mock.patch.object(interaction.people_memory, "rename_person") as rename,
                mock.patch.object(interaction, "_speak_blocking") as speak,
                mock.patch.object(interaction, "_register_rex_utterance"),
                mock.patch.object(interaction.conv_memory, "add_to_transcript"),
                mock.patch.object(interaction.conv_log, "log_rex"),
            ):
                response = interaction._handle_name_update_request(
                    "That's not Bret, I'm Daniel",
                    person_id=1,
                    person_name="Bret",
                )

            # New behavior: instead of refusing, Rex asks to confirm a MERGE — the
            # voice-matched speaker is claiming to be an existing person (same human,
            # two rows). No rename happens; the merge stays pending until they confirm.
            self.assertIn("Daniel", response)
            self.assertIn("silicon wafers", response)
            rename.assert_not_called()
            speak.assert_called_once()
            self.assertIsNotNone(interaction._pending_name_merge_confirmation)
        finally:
            repair_moves.clear()
            interaction._pending_name_merge_confirmation = None
            interaction._session_exchange_count = _prev_exchange_count

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
                mock.patch.object(
                    repair_moves, "pick_recovery_line",
                    return_value="I'm sure we'll have better luck next time!",
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

        self.assertIsNone(command_parser.parse("What do you know about jazz?"))
        self.assertIsNone(command_parser.parse("No, that's wrong."))

    def test_sleep_command_must_be_standalone(self):
        from intelligence import command_parser

        accepted = [
            "go to sleep",
            "go to sleep.",
            "please go to sleep",
            "Rex go to sleep please",
            "sleep",
        ]
        rejected = [
            "I told the kids to go to sleep",
            "I should go to sleep",
            "are you going to sleep",
            "sleep is important",
        ]

        for text in accepted:
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "sleep")

        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(command_parser.parse(text))

    def test_resume_phrases_map_to_wake_up(self):
        from intelligence import command_parser

        for text in [
            "resume",
            "resume talking",
            "talk again",
            "speak again",
            "stop being quiet",
            "exit quiet mode",
        ]:
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "wake_up")

    def test_router_sleep_candidate_must_be_standalone(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="system.sleep",
            confidence=0.94,
            args={},
            reason="system state request",
        )

        self.assertEqual(interaction._router_system_command("go to sleep", decision), "sleep")
        self.assertIsNone(
            interaction._router_system_command("I told the kids to go to sleep", decision)
        )

    def test_sleep_command_speaks_then_enters_sleep_animation(self):
        from intelligence import interaction
        from sequences import animations

        old_state = interaction.state_module.get_state()
        interaction.state_module.set_state(interaction.State.ACTIVE)
        match = interaction.command_parser.CommandMatch("sleep", "exact", {})
        try:
            with (
                mock.patch.object(
                    interaction.config,
                    "SLEEP_MODE_ACKNOWLEDGMENTS",
                    ["Fine. Power nap mode."],
                ),
                mock.patch.object(interaction.random, "choice", return_value="Fine. Power nap mode."),
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.speech_queue, "clear_below_priority") as clear_queue,
                mock.patch.object(interaction.stream, "flush") as flush,
                mock.patch.object(interaction.vad, "reset_state") as reset_vad,
                mock.patch.object(animations, "sleep") as sleep_animation,
            ):
                response = interaction._execute_command(match, 1, "Bret", "go to sleep")

            self.assertEqual(response, "Fine. Power nap mode.")
            speak.assert_called_once_with("Fine. Power nap mode.", emotion="sleepy")
            clear_queue.assert_called_once_with(999)
            flush.assert_called_once()
            reset_vad.assert_called_once()
            sleep_animation.assert_called_once()
            self.assertEqual(interaction.state_module.get_state(), interaction.State.SLEEP)
        finally:
            interaction.state_module.set_state(old_state)

    def test_wake_from_sleep_animates_and_uses_sleep_wake_joke(self):
        from intelligence import interaction
        from sequences import animations

        old_state = interaction.state_module.get_state()
        interaction.state_module.set_state(interaction.State.SLEEP)
        try:
            with (
                mock.patch.object(
                    interaction.config,
                    "WAKE_FROM_SLEEP_ACKNOWLEDGMENTS",
                    ["Awake again. My warranty just flinched."],
                ),
                mock.patch.object(
                    interaction.random,
                    "choice",
                    return_value="Awake again. My warranty just flinched.",
                ),
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.speech_queue, "clear_below_priority"),
                mock.patch.object(interaction.stream, "flush"),
                mock.patch.object(interaction.vad, "reset_state"),
                mock.patch.object(animations, "wake") as wake_animation,
            ):
                response = interaction._wake_from_sleep()

            self.assertEqual(response, "Awake again. My warranty just flinched.")
            wake_animation.assert_called_once()
            speak.assert_called_once_with(
                "Awake again. My warranty just flinched.",
                emotion="happy",
                priority=2,
                pre_beat_ms=80,
                post_beat_ms_override=150,
            )
            self.assertEqual(interaction.state_module.get_state(), interaction.State.ACTIVE)
        finally:
            interaction.state_module.set_state(old_state)

    def test_wake_word_path_exits_quiet_mode(self):
        from intelligence import interaction

        old_state = interaction.state_module.get_state()
        interaction.state_module.set_state(interaction.State.QUIET)
        try:
            with mock.patch.object(interaction, "_wake_ack") as wake_ack:
                interaction._wake_from_quiet()

            self.assertEqual(interaction.state_module.get_state(), interaction.State.ACTIVE)
            wake_ack.assert_called_once()
        finally:
            interaction.state_module.set_state(old_state)

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
                mock.patch.object(
                    interaction.config,
                    "FULL_MEMORY_WIPE_ACCESS_CODE",
                    "Picard alpha 47 tango",
                ),
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._handle_pending_memory_wipe_confirmation(
                    "confirm full wipe Picard alpha 47 tango",
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

    def test_confirm_full_wipe_rejects_missing_access_code(self):
        from intelligence import interaction

        interaction._pending_memory_wipe = {
            "scope": "all",
            "person_id": None,
            "person_name": None,
            "requester_id": 4,
            "asked_at": 100.0,
        }
        try:
            with (
                mock.patch.object(interaction.time, "monotonic", return_value=105.0),
                mock.patch.object(interaction.people_memory, "delete_all_people") as delete_all,
                mock.patch.object(
                    interaction.config,
                    "FULL_MEMORY_WIPE_ACCESS_CODE",
                    "Picard alpha 47 tango",
                ),
                mock.patch.object(interaction, "_speak_blocking") as speak,
            ):
                response = interaction._handle_pending_memory_wipe_confirmation(
                    "confirm full wipe",
                    person_id=4,
                )

            self.assertIn("Access code", response)
            delete_all.assert_not_called()
            speak.assert_called_once()
            self.assertIsNone(interaction._pending_memory_wipe)
        finally:
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
            mock.patch.object(
                interaction.repair_moves, "pick_recovery_line",
                return_value="I'm sure we'll have better luck next time!",
            ),
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

    def test_forget_i_said_that_maps_to_recent_memory_discard(self):
        from intelligence import command_parser

        for text in [
            "forget I said that",
            "forget what I just said",
            "forgot I say that",
        ]:
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.command_key, "memory_boundary")
                self.assertEqual(match.args, {"scope": "recent"})

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
        self.assertFalse(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=None,
                has_unknown_visible=False,
                identity_prompt_active=False,
                text_input=True,
                text="hello?",
            )
        )
        self.assertFalse(
            interaction._should_ignore_idle_background_speech(
                from_idle_activation=True,
                person_id=None,
                has_unknown_visible=False,
                identity_prompt_active=False,
                text="look left",
            )
        )
        with mock.patch.object(interaction.wake_word, "is_ready", return_value=False):
            self.assertFalse(
                interaction._should_ignore_idle_background_speech(
                    from_idle_activation=True,
                    person_id=None,
                    has_unknown_visible=False,
                    identity_prompt_active=False,
                    text="hello?",
                )
            )

    def test_submit_text_is_ignored_while_asleep(self):
        from intelligence import interaction

        old_state = interaction.state_module.get_state()
        interaction.state_module.set_state(interaction.State.SLEEP)
        try:
            with mock.patch.object(interaction, "_handle_speech_segment") as handle:
                self.assertFalse(interaction.submit_text("hello?"))
            handle.assert_not_called()
            self.assertEqual(interaction.state_module.get_state(), interaction.State.SLEEP)
        finally:
            interaction.state_module.set_state(old_state)

    def test_pending_question_recent_attribution_survives_panned_away_face(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_has_unknown_visible_or_recent", return_value=False),
            mock.patch.object(
                interaction,
                "_latest_pending_question",
                return_value={"question_key": "job"},
            ),
        ):
            person_id, person_name, accepted = interaction._pending_question_recent_attribution(
                person_id=None,
                person_name=None,
                recent_engagement={"person_id": 1, "name": "Bret Penziger"},
                raw_best_id=1,
                speaker_score=0.548,
                text="I'm an IT Systems Administrator",
            )

        self.assertTrue(accepted)
        self.assertEqual(person_id, 1)
        self.assertEqual(person_name, "Bret Penziger")

    def test_pending_question_recent_attribution_wins_over_brief_unknown_flicker(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_has_unknown_visible_or_recent", return_value=True),
            mock.patch.object(
                interaction,
                "_latest_pending_question",
                return_value={"question_key": "hometown"},
            ),
        ):
            person_id, person_name, accepted = interaction._pending_question_recent_attribution(
                person_id=None,
                person_name=None,
                recent_engagement={"person_id": 1, "name": "Bret Benziger"},
                raw_best_id=1,
                speaker_score=0.367,
                text="I live in Sacramento, California",
            )

        self.assertTrue(accepted)
        self.assertEqual(person_id, 1)
        self.assertEqual(person_name, "Bret Benziger")

    def test_pending_question_recent_attribution_still_blocks_actual_unknown_voice(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction, "_has_unknown_visible_or_recent", return_value=True),
            mock.patch.object(
                interaction,
                "_latest_pending_question",
                return_value={"question_key": "hometown"},
            ),
        ):
            person_id, person_name, accepted = interaction._pending_question_recent_attribution(
                person_id=None,
                person_name=None,
                recent_engagement={"person_id": 1, "name": "Bret Benziger"},
                raw_best_id=None,
                speaker_score=0.0,
                text="I live in Sacramento, California",
            )

        self.assertFalse(accepted)
        self.assertIsNone(person_id)
        self.assertIsNone(person_name)

    def test_missing_unknown_slot_does_not_count_as_visible_unknown(self):
        from intelligence import interaction

        old_people = interaction.world_state.get("people")
        try:
            interaction.world_state.update("people", [
                {
                    "id": "person_2",
                    "person_db_id": None,
                    "face_id": None,
                    "face_visible": False,
                    "face_missing": True,
                    "face_box": None,
                }
            ])

            self.assertFalse(interaction._has_unknown_visible_person())
        finally:
            interaction.world_state.update("people", old_people)

    def test_vad_barge_in_is_disabled_by_default(self):
        import config
        from intelligence import interaction

        self.assertFalse(config.VAD_BARGE_IN_ENABLED)
        self.assertFalse(interaction._vad_barge_in_enabled())


class TurnCompletionTest(unittest.TestCase):
    def tearDown(self):
        from intelligence import turn_completion

        turn_completion.clear()

    def test_embedded_preposition_answers_are_complete(self):
        from intelligence import turn_completion

        complete_answers = [
            "I'm not sure who you're referring to",
            "I don't know what you're talking about",
            "That's where I'm from",
            "That is what this is for.",
        ]

        for text in complete_answers:
            with self.subTest(text=text):
                self.assertIsNone(turn_completion.classify(text))

    def test_real_to_fragments_still_hold(self):
        from intelligence import turn_completion

        for text in ("I need to", "I don't know what I need to", "I'm going to"):
            with self.subTest(text=text):
                signal = turn_completion.classify(text)
                self.assertIsNotNone(signal)

    def _hold(self, text):
        import numpy as np
        from intelligence import turn_completion as tc

        signal = tc.classify(text)
        self.assertIsNotNone(signal, f"expected {text!r} to classify incomplete")
        return tc.hold(
            text=text,
            audio_array=np.ones(16, dtype="float32"),
            raw_best_id=None,
            raw_best_name=None,
            raw_best_score=0.0,
            signal=signal,
        )

    def _consume(self, text):
        import numpy as np
        from intelligence import turn_completion as tc

        return tc.consume_continuation(
            text=text,
            audio_array=np.ones(16, dtype="float32"),
            raw_best_id=None,
            raw_best_name=None,
            raw_best_score=0.0,
        )

    def test_distinct_question_follower_not_merged(self):
        # The live repro: "What the..." held, then a separate complete question.
        from intelligence import turn_completion as tc

        self._hold("What the...")
        result = self._consume("What do you see?")
        self.assertTrue(result is None or result.get("action") != "merge")
        self.assertIsNone(tc.pending_snapshot())  # stale fragment dropped

    def test_distinct_statement_follower_not_merged(self):
        from intelligence import turn_completion as tc

        self._hold("What the...")
        result = self._consume("I already ate dinner.")
        self.assertTrue(result is None or result.get("action") != "merge")
        self.assertIsNone(tc.pending_snapshot())

    def test_genuine_continuation_still_merges(self):
        self._hold("I was going to")
        result = self._consume("the store")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "merge")
        self.assertEqual(result.get("text"), "I was going to the store")

    def test_connective_follower_still_merges(self):
        # A follower that opens with a connective is a continuation, not distinct.
        self._hold("I need to")
        result = self._consume("and then go home")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "merge")

    def test_fragment_follower_still_chains(self):
        self._hold("I need to")
        result = self._consume("go to the")  # itself incomplete -> chains
        self.assertIsNotNone(result)
        self.assertEqual(result.get("action"), "merge")

    def test_is_distinct_new_thought_contract(self):
        from intelligence import turn_completion as tc

        self.assertTrue(tc._is_distinct_new_thought("What the", "What do you see?"))
        self.assertFalse(tc._is_distinct_new_thought("I was going to", "the store"))
        self.assertFalse(tc._is_distinct_new_thought("I need to", "go to the"))


class ConversationGatingTest(unittest.TestCase):
    def test_latency_fillers_are_in_character_not_human_disfluencies(self):
        import config

        slow_ack_lines = []
        for value in config.SLOW_PATH_ACK_LINES.values():
            slow_ack_lines.extend(value)
        joined = " ".join(config.LATENCY_FILLER_LINES + slow_ack_lines).lower()

        self.assertNotRegex(joined, r"\b(?:um+|uh+|hmm+)\b")
        self.assertTrue(
            any(
                phrase in joined
                for phrase in (
                    "one sec",
                    "processing",
                    "recalibrating",
                    "memory banks",
                    "one second",
                    "let me check",
                )
            )
        )

    def test_slow_path_ack_requires_cached_line_and_marks_ttfs(self):
        from intelligence import interaction

        class Done:
            pass

        trace = interaction._new_character_loop_trace(
            "What do you see?",
            from_idle_activation=False,
            turn_start=10.0,
            raw_best_id=None,
            raw_best_name=None,
            speaker_score=0.0,
        )
        trace.transcript_ready_at = 10.25

        def fake_enqueue(*args, **kwargs):
            callback = kwargs.get("on_start")
            if callback is not None:
                callback()
            return Done()

        token = interaction._current_character_loop_trace.set(trace)
        previous_ack = interaction._last_slow_path_ack
        try:
            interaction._last_slow_path_ack = None
            with (
                mock.patch("config.SLOW_PATH_ACK_ENABLED", True),
                mock.patch("audio.tts.is_cached", return_value=True),
                mock.patch.object(interaction.random, "choice", side_effect=lambda items: items[0]),
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
                mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
                mock.patch.object(interaction.speech_queue, "enqueue", side_effect=fake_enqueue) as enqueue,
                mock.patch.object(interaction.time, "monotonic", side_effect=[10.5, 10.75]),
                mock.patch.object(interaction._log, "info"),
            ):
                self.assertTrue(interaction._try_slow_path_ack("vision"))
        finally:
            interaction._last_slow_path_ack = previous_ack
            interaction._current_character_loop_trace.reset(token)

        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[:2], ("Let me check.", "neutral"))
        self.assertEqual(enqueue.call_args.kwargs["priority"], 1)
        self.assertEqual(enqueue.call_args.kwargs["tag"], "slow_path_ack")
        self.assertEqual(trace.first_response_queued_at, 10.5)
        self.assertEqual(trace.first_response_audio_started_at, 10.75)
        self.assertEqual(trace.first_response_preview, "Let me check.")

    def test_slow_path_ack_skips_uncached_line(self):
        from intelligence import interaction

        trace = interaction._new_character_loop_trace(
            "What do you remember about me?",
            from_idle_activation=False,
            turn_start=10.0,
            raw_best_id=None,
            raw_best_name=None,
            speaker_score=0.0,
        )
        trace.transcript_ready_at = 10.25
        token = interaction._current_character_loop_trace.set(trace)
        try:
            with (
                mock.patch("config.SLOW_PATH_ACK_ENABLED", True),
                mock.patch("audio.tts.is_cached", return_value=False),
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
                mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
                mock.patch.object(interaction.speech_queue, "enqueue") as enqueue,
            ):
                self.assertFalse(interaction._try_slow_path_ack("memory"))
        finally:
            interaction._current_character_loop_trace.reset(token)

        enqueue.assert_not_called()
        self.assertIsNone(trace.first_response_queued_at)

    def test_slow_path_ack_skips_noaudio_text_mode_by_default(self):
        from intelligence import interaction

        trace = interaction._new_character_loop_trace(
            "Tell me something.",
            from_idle_activation=False,
            turn_start=10.0,
            raw_best_id=None,
            raw_best_name=None,
            speaker_score=0.0,
        )
        trace.transcript_ready_at = 10.25
        token = interaction._current_character_loop_trace.set(trace)
        previous_ack = interaction._last_slow_path_ack
        try:
            interaction._last_slow_path_ack = None
            with (
                mock.patch("config.SLOW_PATH_ACK_ENABLED", True),
                mock.patch("config.NO_AUDIO_MODE", True),
                mock.patch("config.AUDIO_OUTPUT_SUPPRESSED", True),
                mock.patch("config.SLOW_PATH_ACK_IN_TEXT_ONLY", False),
                mock.patch("audio.tts.is_cached", return_value=False) as is_cached,
                mock.patch.object(interaction.random, "choice", side_effect=lambda items: items[0]),
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
                mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
                mock.patch.object(interaction.speech_queue, "enqueue") as enqueue,
                mock.patch.object(interaction.time, "monotonic", return_value=10.5),
                mock.patch.object(interaction._log, "info"),
            ):
                self.assertFalse(interaction._try_slow_path_ack("general"))
        finally:
            interaction._last_slow_path_ack = previous_ack
            interaction._current_character_loop_trace.reset(token)

        is_cached.assert_not_called()
        enqueue.assert_not_called()

    def test_slow_path_ack_can_be_enabled_in_noaudio_without_tts_cache(self):
        from intelligence import interaction

        trace = interaction._new_character_loop_trace(
            "Tell me something.",
            from_idle_activation=False,
            turn_start=10.0,
            raw_best_id=None,
            raw_best_name=None,
            speaker_score=0.0,
        )
        trace.transcript_ready_at = 10.25
        token = interaction._current_character_loop_trace.set(trace)
        previous_ack = interaction._last_slow_path_ack
        try:
            interaction._last_slow_path_ack = None
            with (
                mock.patch("config.SLOW_PATH_ACK_ENABLED", True),
                mock.patch("config.NO_AUDIO_MODE", True),
                mock.patch("config.AUDIO_OUTPUT_SUPPRESSED", True),
                mock.patch("config.SLOW_PATH_ACK_IN_TEXT_ONLY", True),
                mock.patch("audio.tts.is_cached", return_value=False) as is_cached,
                mock.patch.object(interaction.random, "choice", side_effect=lambda items: items[0]),
                mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
                mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
                mock.patch.object(interaction.speech_queue, "enqueue") as enqueue,
                mock.patch.object(interaction.time, "monotonic", return_value=10.5),
                mock.patch.object(interaction._log, "info"),
            ):
                # Needs >= SLOW_PATH_ACK_GENERAL_MIN_WORDS (9) words to clear the
                # word-count gate added in commit b749c82; this exercises the
                # text-only/no-cache path the test actually targets.
                self.assertTrue(interaction._try_slow_path_ack(
                    "general",
                    text="Tell me a long story about the asteroid field incident please",
                ))
        finally:
            interaction._last_slow_path_ack = previous_ack
            interaction._current_character_loop_trace.reset(token)

        is_cached.assert_not_called()
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[:2], ("One sec.", "neutral"))

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

    def test_startup_named_greeting_prefixes_memory_callbacks(self):
        from intelligence import consciousness

        self.assertEqual(
            consciousness._ensure_named_startup_greeting(
                "Bret, I hear you built the interface.",
                "Bret",
            ),
            "Hey Bret, I hear you built the interface.",
        )
        self.assertEqual(
            consciousness._ensure_named_startup_greeting(
                "Major achievement on that interface.",
                "Bret",
            ),
            "Hey Bret. Major achievement on that interface.",
        )
        self.assertEqual(
            consciousness._ensure_named_startup_greeting(
                "Hey Bret, nice work on the interface.",
                "Bret",
            ),
            "Hey Bret, nice work on the interface.",
        )

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
        from intelligence import idle_behaviors
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
                mock.patch.object(idle_behaviors, "do_private_thought") as thought,
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

    def test_first_sight_greeting_retries_when_presence_gate_busy(self):
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_seen = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._last_seen.clear()
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at[1] = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [
                    {"person_db_id": 1, "face_id": "Bret Benziger"},
                ],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                suppress_proactive=False,
                interaction_busy=False,
                user_mid_sentence=False,
                likely_still_present=False,
                apparent_departure=False,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=105.0),
                mock.patch.object(consciousness.config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 0.0),
                mock.patch.object(consciousness, "_hold_startup_individual_greeting", return_value=False),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak_presence") as generate,
            ):
                consciousness._step_presence_tracking(snapshot, profile)

            self.assertNotIn(1, consciousness._greeted_this_session)
            self.assertIn(1, consciousness._first_sight_seen_at)
            self.assertNotIn(1, consciousness._visible_people)
            generate.assert_not_called()
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_seen)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_first_sight_greeting_marks_greeted_after_queue_accepts(self):
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_seen = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._last_seen.clear()
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at[1] = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [
                    {"person_db_id": 1, "face_id": "Bret Benziger"},
                ],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                suppress_proactive=False,
                interaction_busy=False,
                user_mid_sentence=False,
                likely_still_present=False,
                apparent_departure=False,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=105.0),
                mock.patch.object(consciousness.config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 0.0),
                mock.patch.object(consciousness, "_hold_startup_individual_greeting", return_value=False),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=True),
                mock.patch.object(consciousness, "_pick_due_emotional_checkin", return_value=None),
                mock.patch.object(consciousness, "_pick_birthday_window", return_value=None),
                mock.patch.object(consciousness, "_pick_due_celebration_checkin", return_value=None),
                mock.patch.object(consciousness, "_pick_milestone", return_value=None),
                mock.patch.object(consciousness, "_pick_anticipated_event", return_value=None),
                mock.patch.object(consciousness, "_pick_absence_phase", return_value=None),
                # Priority 3.5 same-day "you again" reads the live people DB; neutralize it.
                mock.patch.object(consciousness, "_same_day_return_count", return_value=0),
                mock.patch.object(consciousness, "_pick_startup_profile_question", return_value=None),
                mock.patch.object(consciousness, "_build_first_sight_mood_prompt", return_value=None),
                mock.patch.object(
                    consciousness,
                    "_build_startup_solo_greeting_prompt",
                    return_value="startup prompt",
                ),
                mock.patch.object(
                    consciousness,
                    "_generate_and_speak_presence",
                    return_value=True,
                ) as generate,
            ):
                consciousness._step_presence_tracking(snapshot, profile)

            self.assertIn(1, consciousness._greeted_this_session)
            self.assertNotIn(1, consciousness._first_sight_seen_at)
            self.assertIn(1, consciousness._visible_people)
            self.assertEqual(generate.call_args.kwargs["startup_greeting_name"], "Bret")
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_seen)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_first_sight_sparse_profile_uses_basic_profile_question(self):
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_seen = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._last_seen.clear()
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at[1] = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [
                    {"person_db_id": 1, "face_id": "Bret Benziger"},
                ],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                suppress_proactive=False,
                interaction_busy=False,
                user_mid_sentence=False,
                likely_still_present=False,
                apparent_departure=False,
            )
            question = {
                "key": "hometown",
                "text": "So where are you from?",
                "depth": 1,
            }

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=105.0),
                mock.patch.object(consciousness.config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 0.0),
                mock.patch.object(consciousness, "_hold_startup_individual_greeting", return_value=False),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=True),
                mock.patch.object(consciousness, "_pick_due_emotional_checkin", return_value=None),
                mock.patch.object(consciousness, "_pick_birthday_window", return_value=None),
                mock.patch.object(consciousness, "_pick_due_celebration_checkin", return_value=None),
                mock.patch.object(consciousness, "_pick_milestone", return_value=None),
                mock.patch.object(consciousness, "_pick_anticipated_event", return_value=None),
                mock.patch.object(consciousness, "_pick_absence_phase", return_value=None),
                # Priority 3.5 same-day "you again" reads the live people DB; neutralize it.
                mock.patch.object(consciousness, "_same_day_return_count", return_value=0),
                # Isolate from real DB state: a passed pending follow-up (priority
                # 2.5) reads memory.events directly and would otherwise outrank the
                # first-sight profile question this test is asserting.
                mock.patch("memory.events.get_pending_followups", return_value=[]),
                # The interest cold-open tier sits just above the profile question; this
                # test asserts the profile-question fallback, so neutralize it like the
                # other higher tiers above.
                mock.patch.object(consciousness, "_pick_cold_open_callback", return_value=None),
                # The profile question is for people Rex is still getting to know; the new
                # warm-greeting default outranks it for friends/creator. Force the
                # not-yet-a-friend path so this test still exercises the question fallback.
                mock.patch.object(consciousness, "_greeting_profile",
                                  return_value=("You know them a little.", False)),
                mock.patch.object(consciousness, "_pick_startup_profile_question", return_value=question),
                mock.patch.object(
                    consciousness,
                    "_generate_and_speak_presence",
                    return_value=True,
                ) as generate,
            ):
                consciousness._step_presence_tracking(snapshot, profile)

            prompt = generate.call_args.args[0]
            self.assertIn("So where are you from?", prompt)
            self.assertIn("early getting-to-know-you curiosity", prompt)
            self.assertEqual(generate.call_args.kwargs["question_key"], "hometown")
            self.assertEqual(generate.call_args.kwargs["question_depth"], 1)
            self.assertEqual(
                generate.call_args.kwargs["label"],
                "first-sight profile question for Bret Benziger",
            )
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_seen)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_first_sight_on_birthday_speaks_a_birthday_line(self):
        # END-TO-END wiring (NOT a mock of _pick_birthday_window): recognizing a known
        # person at first-sight ON their birthday must open with the birthday line. The
        # birthday is read for REAL from a 'birthday' fact = today's MM-DD, so
        # _pick_birthday_window returns 0 and the Priority-1 tier fires.
        from datetime import date
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_seen = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._last_seen.clear()
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at[1] = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [{"person_db_id": 1, "face_id": "Bret Benziger"}],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                suppress_proactive=False, interaction_busy=False,
                user_mid_sentence=False, likely_still_present=False,
                apparent_departure=False,
            )
            today_md = date.today().strftime("%m-%d")
            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=105.0),
                mock.patch.object(consciousness.config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 0.0),
                mock.patch.object(consciousness, "_hold_startup_individual_greeting", return_value=False),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=True),
                # Priority 0 (a sensitive emotional check-in) deliberately OUTRANKS the
                # birthday — "care before the bit". Neutralize it so the birthday fires.
                mock.patch.object(consciousness, "_pick_due_emotional_checkin", return_value=None),
                # Birthday read for real: a 'birthday' fact whose value is today's MM-DD.
                mock.patch("memory.facts.get_facts",
                           return_value=[{"key": "birthday", "value": today_md}]),
                mock.patch.object(consciousness, "_generate_and_speak_presence", return_value=True) as generate,
            ):
                consciousness._step_presence_tracking(snapshot, profile)

            self.assertTrue(generate.called, "no greeting fired on the birthday")
            prompt = generate.call_args.args[0]
            self.assertIn("is TODAY", prompt)
            self.assertIn("Bret", prompt)
            self.assertEqual(
                generate.call_args.kwargs["label"],
                "startup birthday (T-0) for Bret Benziger",
            )
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_seen)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def _run_birthday_vs_checkin(self, *, wins_on_day):
        """Drive first-sight on the birthday WITH a pending sensitive emotional
        check-in. Returns the greeting label that fired."""
        from datetime import date
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_seen = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._last_seen.clear()
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at[1] = 100.0
            consciousness._greeted_this_session.clear()
            snapshot = {
                "people": [{"person_db_id": 1, "face_id": "Bret Benziger"}],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                suppress_proactive=False, interaction_busy=False,
                user_mid_sentence=False, likely_still_present=False,
                apparent_departure=False,
            )
            today_md = date.today().strftime("%m-%d")
            pending_event = {"category": "health", "id": 9, "valence": -0.6}
            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=105.0),
                mock.patch.object(consciousness.config, "PRESENCE_FIRST_SIGHT_CONFIRM_SECS", 0.0),
                mock.patch.object(consciousness.config, "BIRTHDAY_WINS_ON_DAY", wins_on_day),
                mock.patch.object(consciousness, "_hold_startup_individual_greeting", return_value=False),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=True),
                mock.patch("memory.facts.get_facts",
                           return_value=[{"key": "birthday", "value": today_md}]),
                # A sensitive event IS pending (Priority 0) — it would normally win.
                mock.patch.object(consciousness, "_pick_due_emotional_checkin", return_value=pending_event),
                mock.patch.object(consciousness, "_build_emotional_checkin_prompt", return_value="checkin"),
                mock.patch.object(consciousness, "_generate_and_speak_presence", return_value=True) as generate,
            ):
                consciousness._step_presence_tracking(snapshot, profile)
            return generate.call_args.kwargs["label"] if generate.called else None
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_seen)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_birthday_outranks_emotional_checkin_on_the_actual_day(self):
        # BIRTHDAY_WINS_ON_DAY=True: on the day, "happy birthday" beats a pending
        # sensitive check-in (you reliably hear it on your birthday).
        self.assertEqual(
            self._run_birthday_vs_checkin(wins_on_day=True),
            "startup birthday (T-0) for Bret Benziger",
        )

    def test_emotional_checkin_still_wins_when_flag_off(self):
        # Flag off → care-always-first even on the day.
        self.assertEqual(
            self._run_birthday_vs_checkin(wins_on_day=False),
            "first-sight emotional check-in for Bret Benziger",
        )

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

    def test_startup_empty_room_waits_for_camera_settle(self):
        from intelligence import consciousness

        old_started = consciousness._process_started_mono
        old_seen = consciousness._startup_empty_room_seen_at
        old_fired = consciousness._startup_empty_room_fired
        old_camera = consciousness._startup_camera_first_frame_at
        old_evidence = consciousness._startup_presence_evidence_at
        old_evidence_reason = consciousness._startup_presence_evidence_reason
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._process_started_mono = 100.0
            consciousness._startup_empty_room_seen_at = 0.0
            consciousness._startup_empty_room_fired = False
            consciousness._startup_camera_first_frame_at = 100.0
            consciousness._startup_presence_evidence_at = 0.0
            consciousness._startup_presence_evidence_reason = ""
            consciousness._greeted_this_session.clear()
            snapshot = {"people": [], "crowd": {"count": 0}}
            profile = mock.Mock(
                suppress_proactive=False,
                suppress_system_comments=False,
                interaction_busy=False,
                user_mid_sentence=False,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=102.0),
                mock.patch.object(consciousness.config, "STARTUP_EMPTY_ROOM_CONFIRM_SECS", 5.0),
                mock.patch.object(consciousness.config, "STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE", False),
                mock.patch.object(consciousness, "_speak_async") as speak,
            ):
                consciousness._step_startup_empty_room_comment(snapshot, profile)

            self.assertEqual(consciousness._startup_empty_room_seen_at, 102.0)
            self.assertFalse(consciousness._startup_empty_room_fired)
            speak.assert_not_called()
        finally:
            consciousness._process_started_mono = old_started
            consciousness._startup_empty_room_seen_at = old_seen
            consciousness._startup_empty_room_fired = old_fired
            consciousness._startup_camera_first_frame_at = old_camera
            consciousness._startup_presence_evidence_at = old_evidence
            consciousness._startup_presence_evidence_reason = old_evidence_reason
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_startup_empty_room_speaks_once_after_confirm(self):
        from intelligence import consciousness

        old_started = consciousness._process_started_mono
        old_seen = consciousness._startup_empty_room_seen_at
        old_fired = consciousness._startup_empty_room_fired
        old_camera = consciousness._startup_camera_first_frame_at
        old_evidence = consciousness._startup_presence_evidence_at
        old_evidence_reason = consciousness._startup_presence_evidence_reason
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._process_started_mono = 100.0
            consciousness._startup_empty_room_seen_at = 101.0
            consciousness._startup_empty_room_fired = False
            consciousness._startup_camera_first_frame_at = 100.5
            consciousness._startup_presence_evidence_at = 0.0
            consciousness._startup_presence_evidence_reason = ""
            consciousness._greeted_this_session.clear()
            snapshot = {"people": [], "crowd": {"count": 0}}
            profile = mock.Mock(
                suppress_proactive=False,
                suppress_system_comments=False,
                interaction_busy=False,
                user_mid_sentence=False,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=107.0),
                mock.patch.object(consciousness.config, "STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE", False),
                mock.patch.object(consciousness.random, "choice", return_value="Empty startup line."),
                mock.patch.object(consciousness, "_claim_proactive_purpose", return_value="tok") as claim,
                mock.patch.object(consciousness, "_proactive_purpose_current", return_value=True),
                mock.patch.object(consciousness, "_release_proactive_purpose") as release,
                mock.patch.object(consciousness, "_speak_async", side_effect=_speak_async_spoke) as speak,
            ):
                consciousness._step_startup_empty_room_comment(snapshot, profile)

            claim.assert_called_once_with(
                "startup_empty_room",
                label="startup empty-room joke",
            )
            speak.assert_called_once()
            self.assertEqual(speak.call_args.args[0], "Empty startup line.")
            self.assertEqual(speak.call_args.kwargs["purpose"], "startup_empty_room")
            self.assertEqual(speak.call_args.kwargs["label"], "startup empty-room joke")
            self.assertTrue(consciousness._startup_empty_room_fired)
            release.assert_called_once_with("tok")
        finally:
            consciousness._process_started_mono = old_started
            consciousness._startup_empty_room_seen_at = old_seen
            consciousness._startup_empty_room_fired = old_fired
            consciousness._startup_camera_first_frame_at = old_camera
            consciousness._startup_presence_evidence_at = old_evidence
            consciousness._startup_presence_evidence_reason = old_evidence_reason
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_startup_empty_room_waits_for_presence_scan_gate(self):
        from intelligence import consciousness

        old_started = consciousness._process_started_mono
        old_seen = consciousness._startup_empty_room_seen_at
        old_fired = consciousness._startup_empty_room_fired
        old_camera = consciousness._startup_camera_first_frame_at
        old_evidence = consciousness._startup_presence_evidence_at
        old_evidence_reason = consciousness._startup_presence_evidence_reason
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._process_started_mono = 100.0
            consciousness._startup_empty_room_seen_at = 101.0
            consciousness._startup_empty_room_fired = False
            consciousness._startup_camera_first_frame_at = 100.5
            consciousness._startup_presence_evidence_at = 0.0
            consciousness._startup_presence_evidence_reason = ""
            consciousness._greeted_this_session.clear()
            snapshot = {"people": [], "crowd": {"count": 0}}
            profile = mock.Mock(
                suppress_proactive=False,
                suppress_system_comments=False,
                interaction_busy=False,
                user_mid_sentence=False,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=107.0),
                mock.patch.object(consciousness.config, "STARTUP_EMPTY_ROOM_REQUIRE_SCAN_COMPLETE", True),
                mock.patch.object(consciousness.config, "STARTUP_EMPTY_ROOM_MIN_SCAN_SECS", 12.0),
                mock.patch.object(consciousness, "_speak_async") as speak,
            ):
                consciousness._step_startup_empty_room_comment(snapshot, profile)

            speak.assert_not_called()
            self.assertEqual(consciousness._startup_empty_room_seen_at, 0.0)
        finally:
            consciousness._process_started_mono = old_started
            consciousness._startup_empty_room_seen_at = old_seen
            consciousness._startup_empty_room_fired = old_fired
            consciousness._startup_camera_first_frame_at = old_camera
            consciousness._startup_presence_evidence_at = old_evidence
            consciousness._startup_presence_evidence_reason = old_evidence_reason
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

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
        from intelligence import idle_behaviors

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
            idle_behaviors.do_empty_room_joke(snapshot)

        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "Empty room test line.")
        self.assertEqual(speak.call_args.kwargs["label"], "empty-room joke")
        release.assert_called_once_with("tok")

    def test_people_roast_prompt_stays_non_sensitive(self):
        from intelligence import consciousness
        from intelligence import idle_behaviors

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
            idle_behaviors.do_people_roast(snapshot)

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

    def test_social_frame_preserves_space_before_open_quote(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="neutral",
            max_words=60,
            max_sentences=3,
            allow_question=True,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            'Right? The tales you will tell about your drive. "And there I was, sipping soda!"',
            frame,
        )

        self.assertIn('drive. "And there I was', governed.text)
        self.assertNotIn('drive."And', governed.text)

    def test_social_frame_does_not_count_quoted_question_as_followup(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="neutral",
            max_words=60,
            max_sentences=3,
            allow_question=False,
            allow_roast="normal",
            allow_visual_comment=False,
            reason="test",
        )
        text = "You must be collecting “Are we there yet?” queries. Classic road-trip scholarship."
        governed = social_frame.govern_response(text, frame)

        self.assertEqual(governed.text, text)
        self.assertNotIn("removed_question", governed.notes)

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

    def test_presence_profile_question_is_saved_with_pool_key(self):
        from intelligence import consciousness

        with mock.patch("memory.relationships.save_question_asked") as save:
            consciousness._record_proactive_question(
                1,
                "Hey Bret. So where are you from?",
                label="first-sight profile question for Bret Benziger",
                purpose="presence_reaction",
                question_key="hometown",
                question_depth=1,
            )

        save.assert_called_once_with(
            1,
            "hometown",
            "Hey Bret. So where are you from?",
            1,
        )

    def test_jeff_benziger_special_uses_starstruck_direct_greeting(self):
        from intelligence import consciousness
        from threading import Event

        consciousness._jeff_celebrity_greeted_this_session.clear()
        consciousness._first_sight_seen_at.clear()
        consciousness._pending_jeff_celebrity_greetings.clear()
        done = Event()
        done.set()
        with (
            mock.patch.object(consciousness, "_can_jeff_celebrity_speak", return_value=True),
            mock.patch("audio.speech_queue.clear_below_priority") as clear_lower,
            mock.patch("audio.speech_queue.enqueue", return_value=done) as enqueue,
            mock.patch("memory.people.record_greeting"),
        ):
            fired = consciousness._try_fire_jeff_history_hunters_greeting(
                key=11,
                person_name="Jeff Benziger",
                person_db_id=11,
                profile=mock.Mock(),
            )

        self.assertTrue(fired)
        clear_lower.assert_called_once_with(2)
        args = enqueue.call_args.args
        self.assertIn(args[0], consciousness._JEFF_HISTORY_HUNTERS_LINES)
        self.assertEqual(args[1], "starstruck")
        self.assertEqual(enqueue.call_args.kwargs["priority"], 2)
        self.assertIn(11, consciousness._jeff_celebrity_greeted_this_session)

    def test_jeff_benziger_detection_waits_and_blocks_lower_priority_steps(self):
        from intelligence import consciousness

        consciousness._jeff_celebrity_greeted_this_session.clear()
        consciousness._first_sight_seen_at.clear()
        consciousness._pending_jeff_celebrity_greetings.clear()
        snapshot = {
            "people": [
                {"person_db_id": 11, "face_id": "Jeff Benziger"},
            ],
        }
        with mock.patch.object(consciousness.time, "monotonic", return_value=100.0):
            handled = consciousness._step_jeff_history_hunters_detection(snapshot, mock.Mock())

        self.assertTrue(handled)
        self.assertNotIn(11, consciousness._jeff_celebrity_greeted_this_session)

    def test_jeff_benziger_detection_stays_pending_when_speech_busy(self):
        from intelligence import consciousness

        consciousness._jeff_celebrity_greeted_this_session.clear()
        consciousness._first_sight_seen_at.clear()
        consciousness._pending_jeff_celebrity_greetings.clear()
        consciousness._first_sight_seen_at[11] = 90.0
        snapshot = {
            "people": [
                {"person_db_id": 11, "face_id": "Jeff Benziger"},
            ],
        }
        with (
            mock.patch.object(consciousness.time, "monotonic", return_value=100.0),
            mock.patch.object(consciousness, "_can_jeff_celebrity_speak", return_value=False),
        ):
            handled = consciousness._step_jeff_history_hunters_detection(snapshot, mock.Mock())

        self.assertTrue(handled)
        self.assertIn(11, consciousness._pending_jeff_celebrity_greetings)
        self.assertNotIn(11, consciousness._jeff_celebrity_greeted_this_session)

    def test_jeff_benziger_pending_greeting_fires_before_other_steps(self):
        from intelligence import consciousness

        consciousness._jeff_celebrity_greeted_this_session.clear()
        consciousness._pending_jeff_celebrity_greetings.clear()
        consciousness._pending_jeff_celebrity_greetings[11] = {
            "person_name": "Jeff Benziger",
            "returning": False,
            "first_seen_at": 90.0,
            "last_seen_at": 100.0,
        }
        with (
            mock.patch.object(consciousness.time, "monotonic", return_value=101.0),
            mock.patch.object(consciousness, "_try_fire_jeff_history_hunters_greeting", return_value=True) as fire,
        ):
            handled = consciousness._step_jeff_history_hunters_detection({}, mock.Mock())

        self.assertTrue(handled)
        fire.assert_called_once()

    def test_starstruck_emotion_is_more_dramatic_than_excited(self):
        from intelligence import emotion_orchestrator

        excited = emotion_orchestrator.frame_for_speech("excited")
        starstruck = emotion_orchestrator.frame_for_speech("starstruck")

        self.assertEqual(starstruck.led_style, "excited")
        self.assertGreater(starstruck.intensity, excited.intensity)
        self.assertGreater(
            starstruck.speech_motion["arm_intensity_mult"],
            excited.speech_motion["arm_intensity_mult"],
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

    def test_relationship_prompt_rejects_hallucinated_name_and_relationship(self):
        from intelligence import interaction

        rel_ctx = {
            "engaged_person_id": 1,
            "engaged_name": "Bret Benziker",
            "slot_id": "person_1",
            "asked_at": interaction.time.monotonic(),
        }

        with (
            mock.patch.object(
                interaction.llm,
                "extract_relationship_introduction",
                return_value={"name": "JT", "relationship": "partner"},
            ) as extract,
            mock.patch.object(
                interaction.people_memory,
                "find_or_create_person",
            ) as find_or_create,
            mock.patch.object(
                interaction.consciousness,
                "note_relationship_slot_handled",
            ) as handled,
        ):
            response = interaction._handle_relationship_reply(
                rel_ctx,
                "Bret Benzigert",
                1,
                "Bret Benziker",
            )

        self.assertIsNone(response)
        extract.assert_called_once()
        find_or_create.assert_not_called()
        handled.assert_called_once_with("person_1")

    def test_relationship_evidence_accepts_explicit_partner_intro(self):
        from intelligence import interaction

        self.assertTrue(
            interaction._name_supported_by_user_text(
                "JT",
                "This is my partner JT",
            )
        )
        self.assertTrue(
            interaction._relationship_supported_by_user_text(
                "partner",
                "This is my partner JT",
            )
        )

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
        self.assertIn("never pivot into a new interview topic", directive)

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
            mock.patch("audio.tts.ensure_cached", return_value=True),
            mock.patch.object(interaction.barge_guard, "user_speaking_now", return_value=False),
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

    def test_social_frame_removes_health_pun_in_no_roast_mode(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Jeff",
            purpose="identity",
            max_words=40,
            max_sentences=3,
            allow_question=True,
            allow_roast="none",
            allow_visual_comment=False,
            reason="test",
        )
        governed = social_frame.govern_response(
            "Cataracts, huh. Guess it's time for a little visual upgrade. "
            "But hey, at least you're able to see the humor in it! "
            "Who's your friend over there?",
            frame,
        )

        self.assertEqual(governed.text, "Cataracts, huh. Who's your friend over there?")
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
        self.assertIn("sharp roast", directive)
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
        self.assertIn("What are you working on today?", prompt)

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
        self.assertIn("What's on your mind?", prompt)

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

    def test_social_frame_closure_removes_escape_plan_snark(self):
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
            'Nice chatting. That’s one way to say, “I need to escape this conversation.” Good luck with your escape plan, Indiana Jones!',
            frame,
        )

        self.assertNotIn("escape", governed.text.lower())
        self.assertNotIn("Indiana Jones", governed.text)

    def test_social_frame_closure_enforces_micro_shape(self):
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
            "Always a pleasure to reboot this conversation with you. Stay out of trouble!",
            frame,
        )

        self.assertEqual(governed.text, "Always a pleasure to reboot this conversation with you.")
        self.assertNotIn("Stay out", governed.text)

    def test_social_frame_repairs_space_before_closing_quote(self):
        from intelligence import social_frame

        frame = social_frame.SocialFrame(
            addressee="Bret",
            purpose="answer",
            max_words=40,
            max_sentences=2,
            allow_question=False,
            allow_roast="light",
            allow_visual_comment=False,
            reason="test",
        )

        governed = social_frame.govern_response(
            'Your friendship score. A solid "room for improvement. " Think of it as a droid compliment.',
            frame,
        )

        self.assertIn('"room for improvement."', governed.text)
        self.assertNotIn('. "', governed.text)

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

    def test_unknown_face_identity_prompt_runs_in_active_wake_fallback(self):
        import numpy as np
        from intelligence import consciousness
        from state import State

        old_people = consciousness.world_state.get("people")
        old_signature = consciousness._last_face_feedback_signature
        old_last_identity = consciousness._last_identity_prompt_at
        old_reply_until = consciousness._identity_prompt_reply_until
        old_solo_unknown = consciousness._solo_unknown_since
        try:
            consciousness.world_state.update("people", [])
            consciousness._last_face_feedback_signature = None
            consciousness._last_identity_prompt_at = 0.0
            consciousness._pending_identity_prompt.clear()
            consciousness._identity_prompt_in_flight.clear()
            # The unknown face has already persisted past the grace window (monotonic
            # is mocked to 100.0 below); this test exercises ACTIVE-fallback gating,
            # not the startup grace itself.
            consciousness._solo_unknown_since = 1.0
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

            with (
                mock.patch("vision.face.detect_faces", return_value=[{"encoding": object()}]),
                mock.patch("vision.face.identify_face", return_value=None),
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness.state_module, "get_state", return_value=State.ACTIVE),
                # Proactive identity prompts in ACTIVE state are opt-in (default
                # off) since commit b749c82; enable it to exercise this path.
                mock.patch.object(consciousness.config, "IDENTITY_PROMPT_ALLOW_PROACTIVE_ACTIVE", True),
                mock.patch.object(consciousness.time, "monotonic", return_value=100.0),
                mock.patch.object(consciousness, "_speak_async", return_value=True) as speak,
            ):
                consciousness._step_person_recognition(frame)

            speak.assert_called_once()
            self.assertEqual(speak.call_args.kwargs.get("purpose"), "identity_prompt")
            self.assertEqual(speak.call_args.kwargs.get("label"), "identity_prompt")
            self.assertTrue(consciousness._identity_prompt_in_flight.is_set())
            self.assertEqual(consciousness._last_identity_prompt_at, 100.0)
        finally:
            consciousness.world_state.update("people", old_people)
            consciousness._last_face_feedback_signature = old_signature
            consciousness._last_identity_prompt_at = old_last_identity
            consciousness._pending_identity_prompt.clear()
            consciousness._identity_prompt_in_flight.clear()
            consciousness._identity_prompt_reply_until = old_reply_until
            consciousness._solo_unknown_since = old_solo_unknown

    def test_identity_prompt_reply_window_consumes_or_expires(self):
        from intelligence import consciousness

        old_until = consciousness._identity_prompt_reply_until
        old_pending = consciousness._pending_identity_prompt.is_set()
        old_in_flight = consciousness._identity_prompt_in_flight.is_set()
        try:
            consciousness._identity_prompt_in_flight.clear()
            consciousness._pending_identity_prompt.set()
            consciousness._identity_prompt_reply_until = 120.0

            with mock.patch.object(consciousness.time, "monotonic", return_value=110.0):
                self.assertTrue(consciousness.is_identity_prompt_waiting_for_reply())
                self.assertTrue(consciousness.consume_identity_prompt_request())

            self.assertFalse(consciousness._pending_identity_prompt.is_set())
            self.assertEqual(consciousness._identity_prompt_reply_until, 0.0)

            consciousness._pending_identity_prompt.set()
            consciousness._identity_prompt_reply_until = 120.0
            with mock.patch.object(consciousness.time, "monotonic", return_value=121.0):
                self.assertFalse(consciousness.is_identity_prompt_waiting_for_reply())

            self.assertFalse(consciousness._pending_identity_prompt.is_set())
            self.assertEqual(consciousness._identity_prompt_reply_until, 0.0)
        finally:
            consciousness._pending_identity_prompt.clear()
            consciousness._identity_prompt_in_flight.clear()
            if old_pending:
                consciousness._pending_identity_prompt.set()
            if old_in_flight:
                consciousness._identity_prompt_in_flight.set()
            consciousness._identity_prompt_reply_until = old_until

    def test_engaged_departure_stages_before_default_departure_window(self):
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_first_missing = dict(consciousness._first_missing_at)
        old_pending_departures = dict(consciousness._pending_departure_keys)
        old_confirmed_absent = dict(consciousness._confirmed_absent_at)
        old_last_snapshot = dict(consciousness._last_snapshot)
        try:
            consciousness._visible_people.clear()
            consciousness._first_missing_at.clear()
            consciousness._pending_departure_keys.clear()
            consciousness._confirmed_absent_at.clear()
            consciousness._first_missing_at[1] = 100.0
            consciousness._last_snapshot = {"people": []}
            profile = mock.Mock(
                likely_still_present=False,
                user_mid_sentence=False,
                apparent_departure=True,
            )

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=104.0),
                mock.patch.object(consciousness.config, "PRESENCE_DEPARTURE_CONFIRM_SECS", 20.0),
                mock.patch.object(consciousness.config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", 3.0),
                mock.patch.object(consciousness, "is_engaged_with", return_value=True),
                mock.patch("memory.people.get_person", return_value={"name": "Bret Benziger"}),
                mock.patch.object(consciousness, "_should_fire_presence", return_value=False),
            ):
                consciousness._step_presence_tracking({"people": []}, profile)

            self.assertIn(1, consciousness._pending_departure_keys)
            self.assertEqual(consciousness._pending_departure_keys[1][1], "Bret Benziger")
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._first_missing_at.clear()
            consciousness._first_missing_at.update(old_first_missing)
            consciousness._pending_departure_keys.clear()
            consciousness._pending_departure_keys.update(old_pending_departures)
            consciousness._confirmed_absent_at.clear()
            consciousness._confirmed_absent_at.update(old_confirmed_absent)
            consciousness._last_snapshot = old_last_snapshot

    def test_identity_bound_slot_does_not_depart_as_unknown(self):
        from intelligence import consciousness

        old_visible = set(consciousness._visible_people)
        old_last_seen = dict(consciousness._last_seen)
        old_first_missing = dict(consciousness._first_missing_at)
        old_pending_departures = dict(consciousness._pending_departure_keys)
        old_confirmed_absent = dict(consciousness._confirmed_absent_at)
        old_first_sight = dict(consciousness._first_sight_seen_at)
        old_greeted = set(consciousness._greeted_this_session)
        try:
            consciousness._visible_people.clear()
            consciousness._visible_people.add("person_1")
            consciousness._last_seen.clear()
            consciousness._last_seen["person_1"] = 100.0
            consciousness._first_missing_at.clear()
            consciousness._first_missing_at["person_1"] = 101.0
            consciousness._pending_departure_keys.clear()
            consciousness._pending_departure_keys["person_1"] = (101.0, None, None)
            consciousness._confirmed_absent_at.clear()
            consciousness._confirmed_absent_at["person_1"] = 101.0
            consciousness._first_sight_seen_at.clear()
            consciousness._greeted_this_session.add(1)

            snapshot = {
                "people": [
                    {
                        "id": "person_1",
                        "person_db_id": 1,
                        "face_id": "Bret Benziger",
                    },
                ],
                "crowd": {"count": 1},
            }
            profile = mock.Mock(
                likely_still_present=False,
                user_mid_sentence=False,
                apparent_departure=False,
            )

            with mock.patch.object(consciousness.time, "monotonic", return_value=105.0):
                consciousness._step_presence_tracking(snapshot, profile)

            self.assertNotIn("person_1", consciousness._visible_people)
            self.assertNotIn("person_1", consciousness._last_seen)
            self.assertNotIn("person_1", consciousness._first_missing_at)
            self.assertNotIn("person_1", consciousness._pending_departure_keys)
            self.assertNotIn("person_1", consciousness._confirmed_absent_at)
            self.assertIn(1, consciousness._visible_people)
        finally:
            consciousness._visible_people.clear()
            consciousness._visible_people.update(old_visible)
            consciousness._last_seen.clear()
            consciousness._last_seen.update(old_last_seen)
            consciousness._first_missing_at.clear()
            consciousness._first_missing_at.update(old_first_missing)
            consciousness._pending_departure_keys.clear()
            consciousness._pending_departure_keys.update(old_pending_departures)
            consciousness._confirmed_absent_at.clear()
            consciousness._confirmed_absent_at.update(old_confirmed_absent)
            consciousness._first_sight_seen_at.clear()
            consciousness._first_sight_seen_at.update(old_first_sight)
            consciousness._greeted_this_session.clear()
            consciousness._greeted_this_session.update(old_greeted)

    def test_weekly_recap_prompt_uses_actual_monday_part_of_day(self):
        from intelligence import consciousness

        old_weekly = set(consciousness._weekly_smalltalk_asked)
        old_check_at = consciousness._last_weekly_smalltalk_check_at
        with consciousness._engaged_lock:
            old_engaged_id = consciousness._engaged_person_id
            old_engaged_touch = consciousness._engaged_last_touch_at
        try:
            consciousness._weekly_smalltalk_asked.clear()
            consciousness._last_weekly_smalltalk_check_at = 0.0
            with consciousness._engaged_lock:
                consciousness._engaged_person_id = 1
                consciousness._engaged_last_touch_at = 90.0
            snapshot = {
                "time": {"day_of_week": "Monday", "time_of_day": "afternoon"},
                "people": [{"person_db_id": 1, "face_id": "Bret Benziger"}],
            }
            profile = mock.Mock(suppress_proactive=False, rapid_exchange=False)

            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=100.0),
                mock.patch.object(consciousness.config, "WEEKLY_SMALLTALK_CHECK_INTERVAL_SECS", 0.0),
                mock.patch.object(consciousness.config, "WEEKLY_SMALLTALK_MIN_SILENCE_SECS", 0.0),
                mock.patch.object(consciousness.config, "WEEKLY_SMALLTALK_PROBABILITY", 1.0),
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness.random, "random", return_value=0.0),
                mock.patch("memory.people.get_person", return_value={"name": "Bret Benziger"}),
                mock.patch("memory.events.get_pending_followups", return_value=[]),
                mock.patch.object(consciousness, "_generate_and_speak", return_value=True) as speak,
            ):
                consciousness._step_weekly_smalltalk(snapshot, profile)

            prompt = speak.call_args.args[0]
            self.assertIn("Monday afternoon", prompt)
            self.assertNotIn("Monday morning", prompt)
        finally:
            consciousness._weekly_smalltalk_asked.clear()
            consciousness._weekly_smalltalk_asked.update(old_weekly)
            consciousness._last_weekly_smalltalk_check_at = old_check_at
            with consciousness._engaged_lock:
                consciousness._engaged_person_id = old_engaged_id
                consciousness._engaged_last_touch_at = old_engaged_touch

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

    def test_holiday_plans_skip_minor_holidays_by_default(self):
        from intelligence import consciousness

        holiday = {
            "name": "Truman Day",
            "date": "2026-05-08",
            "days_until": 5,
            "window": "minor",
        }

        with mock.patch("config.HOLIDAY_PLANS_INCLUDE_MINOR", False):
            self.assertFalse(consciousness._holiday_plans_allowed(holiday))
        with mock.patch("config.HOLIDAY_PLANS_INCLUDE_MINOR", True):
            self.assertTrue(consciousness._holiday_plans_allowed(holiday))
        self.assertTrue(
            consciousness._holiday_plans_allowed({**holiday, "window": "major"})
        )


class PendingMusicPreferenceTest(unittest.TestCase):
    def setUp(self):
        # Order-independence: _maybe_capture_pending_qa (used by the music-preference
        # answer path) bails when any of these interaction globals is left set by an
        # earlier test in the run, which silently swallows the captured answer. Reset
        # them so these tests pass regardless of suite ordering.
        from intelligence import interaction

        interaction._pending_offscreen_identify = None
        interaction._pending_face_reveal_confirm = None
        interaction._awaiting_followup_event = None
        interaction._pending_music_offer = None

    def tearDown(self):
        from intelligence import interaction

        interaction._pending_music_offer = None

    def test_bare_music_preference_answer_is_normalized_and_not_played_immediately(self):
        from intelligence import interaction

        pending = {
            "question_key": "favorite_music",
            "question_text": "What kind of music are you into?",
        }
        answered = {"question_key": "favorite_music", "answer": "classical"}

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
                "I like classical music",
                pending_question=pending,
            )

        self.assertEqual(captured, answered)
        self.assertIn("Want me to play some classical", response)
        answer.assert_called_once_with(1, "classical")
        add_fact.assert_called_once_with(
            1,
            "preference",
            "favorite_music",
            "classical",
            "pending_qa:favorite_music",
            confidence=0.95,
        )
        speak.assert_called_once()
        self.assertEqual(
            interaction._pending_music_offer["music_query"],
            "classical",
        )

    def test_music_preference_answer_strips_trailing_music_word(self):
        from intelligence import interaction

        self.assertEqual(
            interaction._normalize_music_preference_answer("classical music"),
            "classical",
        )
        self.assertEqual(
            interaction._normalize_music_preference_answer("I'm into classic rock"),
            "classic rock",
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
            mock.patch.object(interaction, "_start_dj_after_response") as start_dj,
            mock.patch.object(interaction, "_speak_blocking") as speak,
            # Playback is deferred via threading.Timer (commit 35999e7); run the
            # timer callback inline so the test can observe dj.play synchronously.
            mock.patch.object(
                interaction.threading, "Timer",
                side_effect=lambda delay, fn: mock.Mock(start=fn),
            ),
        ):
            response = interaction._handle_pending_music_offer_reply(1, "yes")

        self.assertIn("Spinning Classical Test", response)
        handle.assert_called_once_with("classical music")
        start_dj.assert_called_once_with(track)
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

    def test_conversation_reply_general_knowledge_skips_memory_learning(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="conversation.reply",
            confidence=1.0,
            args={},
            reason="general knowledge",
        )

        self.assertTrue(
            interaction._conversation_reply_should_skip_memory_learning(
                "What do you know about jazz?",
                decision,
            )
        )
        self.assertTrue(
            interaction._conversation_reply_should_skip_memory_learning(
                "Tell me about Star Trek.",
                decision,
            )
        )
        self.assertFalse(
            interaction._conversation_reply_should_skip_memory_learning(
                "I like jazz",
                decision,
            )
        )
        self.assertFalse(
            interaction._conversation_reply_should_skip_memory_learning(
                "What do you remember about me?",
                decision,
            )
        )

    def test_vision_snapshot_blocked_response_requests_confirmation(self):
        from intelligence import action_router, interaction

        decision = action_router.ActionDecision(
            action="vision.snapshot",
            confidence=0.94,
            args={"scope": "current_view"},
            requires_confirmation=True,
            reason="privacy-sensitive scene memory",
        )

        response = interaction._router_blocked_confirmation_response(
            decision,
            "requires_confirmation",
        )

        self.assertIn("won't store a scene memory", response)
        self.assertIsNone(
            interaction._router_blocked_confirmation_response(
                decision,
                "not_in_execute_allowlist",
            )
        )

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

    def test_router_downgrades_named_day_as_date_query(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="date.query",
            confidence=0.90,
            args={},
            reason="misread holiday explanation as current date",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "What's Truman Day?",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_downgrades_ongoing_status_from_event_cancel(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="event.cancel",
            confidence=0.90,
            args={"event_hint": "driving home"},
            reason="misread ongoing trip as cancellation",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "We're still driving home",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_allows_explicit_event_cancel_with_continuation_words(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="event.cancel",
            confidence=0.90,
            args={"event_hint": "driving home"},
            reason="explicit cancellation",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "We're not driving home anymore",
            {},
        )

        self.assertEqual(routed.action, "event.cancel")

    def test_router_downgrades_pronoun_only_intro_fragment(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="identity.introduce_person",
            confidence=0.90,
            args={"name": "you"},
            reason="misread score clarification as introduction",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Me and you",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_router_downgrades_named_person_fact_from_introduction(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="identity.introduce_person",
            confidence=0.90,
            args={"name": "Jeff"},
            reason="misread fact as introduction",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "Jeff is a newspaper editor.",
            {},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLess(routed.confidence, 0.85)

    def test_nonexecuted_introduce_person_keeps_memory_learning_available(self):
        from intelligence import action_router, interaction

        intro_decision = action_router.ActionDecision(
            action="identity.introduce_person",
            confidence=0.90,
            args={"person_name": "Jeff Benziger"},
        )
        memory_decision = action_router.ActionDecision(
            action="memory.query",
            confidence=0.90,
            args={},
        )

        self.assertFalse(
            interaction._router_nonexecuted_should_suppress_memory_learning(intro_decision)
        )
        self.assertTrue(
            interaction._router_nonexecuted_should_suppress_memory_learning(memory_decision)
        )

    def test_direct_memory_learning_saves_relationship_and_named_person_fact(self):
        from intelligence import interaction

        with (
            mock.patch.object(interaction.people_memory, "find_or_create_person", return_value=(2, True)),
            mock.patch.object(interaction.people_memory, "find_person_by_name", return_value={"id": 2, "name": "Jeff Benziger"}),
            mock.patch.object(interaction.social_memory, "save_relationship") as save_relationship,
            mock.patch.object(interaction.facts_memory, "add_fact") as add_fact,
            mock.patch.object(interaction, "_record_recent_memory_candidate"),
        ):
            interaction._learn_direct_memory_from_user_text(
                "My dad is named Jeff Benziger",
                1,
            )
            interaction._learn_direct_memory_from_user_text(
                "Jeff Benziger is a newspaper editor.",
                1,
            )

        save_relationship.assert_called_once()
        rel_kwargs = save_relationship.call_args.kwargs
        self.assertEqual(rel_kwargs["from_person_id"], 1)
        self.assertEqual(rel_kwargs["to_person_id"], 2)
        self.assertEqual(rel_kwargs["relationship"], "father")
        self.assertTrue(
            any(
                call.args[:4] == (2, "job", "job_title", "newspaper editor")
                for call in add_fact.call_args_list
            )
        )

    def test_router_routes_score_query_outside_game_to_memory(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="game.answer",
            confidence=1.0,
            args={},
            reason="misread score question as game answer",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "And our score is?",
            {"active_game": False},
        )

        self.assertEqual(routed.action, "memory.query")
        self.assertGreaterEqual(routed.confidence, 0.85)

    def test_router_downgrades_game_answer_when_no_game_is_active(self):
        from intelligence import action_router

        decision = action_router.ActionDecision(
            action="game.answer",
            confidence=1.0,
            args={},
            reason="misread bare direction as game answer",
        )

        routed = action_router._apply_context_overrides(
            decision,
            "down",
            {"active_game": False},
        )

        self.assertEqual(routed.action, "conversation.reply")
        self.assertLessEqual(routed.confidence, 0.40)

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

    def test_intent_classifier_keeps_named_day_questions_general(self):
        from intelligence import intent_classifier

        named_day_questions = [
            "What's Truman Day?",
            "What is the holiday called Truman Day. I haven't heard of that before.",
            "What is Memorial Day?",
        ]
        for text in named_day_questions:
            with (
                self.subTest(text=text),
                mock.patch.object(
                    intent_classifier,
                    "_classify_with_llm",
                    return_value="query_date",
                ),
            ):
                self.assertEqual(intent_classifier.classify(text), "general")

        self.assertEqual(intent_classifier.classify("what day is it?"), "query_date")
        self.assertEqual(
            intent_classifier.classify("what's today's date?"),
            "query_date",
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
        self.assertEqual(
            intent_classifier.classify("How many times have you greeted me?"),
            "query_memory",
        )
        self.assertEqual(
            intent_classifier.classify("What's my friendship score?"),
            "query_memory",
        )
        self.assertEqual(intent_classifier.classify("Me and you"), "general")

    def test_memory_query_resolves_score_and_greeting_queries_to_current_speaker(self):
        from intelligence import memory_query

        person = {
            "id": 1,
            "name": "Bret Benziger",
            "friendship_tier": "friend",
            "familiarity_score": 0.42,
            "net_relationship_score": 0.31,
            "warmth_score": 0.55,
            "trust_score": 0.61,
            "playfulness_score": 0.33,
            "curiosity_score": 0.44,
            "antagonism_score": 0.02,
            "visit_count": 7,
            "lifetime_greeting_count": 3,
            "last_greeted_at": "2026-05-03T16:39:10+00:00",
        }
        with (
            mock.patch.object(memory_query.people_memory, "get_person", return_value=person),
            mock.patch.object(memory_query.social_memory, "summarize_for_prompt", return_value=""),
            mock.patch.object(memory_query.facts_memory, "get_prompt_facts", return_value=[]),
            mock.patch.object(memory_query.events_memory, "get_open_events", return_value=[]),
            mock.patch.object(memory_query.conv_memory, "get_conversation_history", return_value=[]),
            mock.patch.object(memory_query.emotional_events, "get_active_events", return_value=[]),
        ):
            score_target = memory_query.resolve_target("What's my friendship score?", 1)
            greeting_target = memory_query.resolve_target(
                "How many times have you greeted me?",
                1,
            )
            context = memory_query.build_context(score_target, 1)
            prompt = memory_query.build_response_prompt(
                "How many times have you greeted me?",
                context,
            )

        self.assertEqual(score_target.person_id, 1)
        self.assertEqual(greeting_target.person_id, 1)
        joined = context.as_prompt_text()
        self.assertIn("familiarity_score=0.42", joined)
        self.assertIn("net_relationship_score=0.31", joined)
        self.assertIn("lifetime_greeting_count=3", joined)
        self.assertIn("use lifetime_greeting_count", prompt)
        self.assertIn("Do not tease, insult", memory_query.build_response_prompt("What's my friendship score?", context))

    def test_memory_query_relationship_prompt_discourages_family_roasts(self):
        from intelligence import memory_query

        target = memory_query.MemoryTarget(
            person_id=11,
            name="Jeff Benziger",
            mode="relationship",
            relation_label="dad",
        )
        context = memory_query.MemoryContext(
            target=target,
            sections=[
                "Target person: Jeff Benziger (person_id=11, tier=unknown)",
                "Relationship to current speaker:\n- Bret Benziger -> Jeff Benziger: father",
            ],
            has_memory=True,
        )

        prompt = memory_query.build_response_prompt("Who is my dad?", context)

        self.assertIn("answer the relationship directly", prompt)
        self.assertIn("genetics or inheritance jokes", prompt)
        self.assertIn("rivalry speculation", prompt)

    def test_memory_query_resolves_work_and_pet_topics_to_current_speaker(self):
        from intelligence import memory_query

        person = {"id": 1, "name": "Bret Benziger", "friendship_tier": "friend"}
        facts = [
            {"category": "pet", "key": "dogs", "value": "Toby and Maxx"},
            {"category": "job", "key": "job_title", "value": "IT Systems Administrator"},
        ]
        with (
            mock.patch.object(memory_query.people_memory, "get_person", return_value=person),
            mock.patch.object(memory_query.social_memory, "summarize_for_prompt", return_value=""),
            mock.patch.object(memory_query.facts_memory, "get_prompt_facts", return_value=facts),
            mock.patch.object(
                memory_query.facts_memory,
                "format_fact_for_prompt",
                side_effect=lambda fact: f"{fact['category']}:{fact['key']}={fact['value']}",
            ),
            mock.patch.object(memory_query.events_memory, "get_open_events", return_value=[]),
            mock.patch.object(memory_query.conv_memory, "get_conversation_history", return_value=[]),
            mock.patch.object(memory_query.emotional_events, "get_active_events", return_value=[]),
        ):
            dog_target = memory_query.resolve_target("Can you tell me about my dogs?", 1)
            work_target = memory_query.resolve_target("Do you remember what I do for work?", 1)
            job_target = memory_query.resolve_target("Did I ever tell you about my job?", 1)
            context = memory_query.build_context(dog_target, 1)
            dog_prompt = memory_query.build_response_prompt("Can you tell me about my dogs?", context)
            work_prompt = memory_query.build_response_prompt("Do you remember what I do for work?", context)

        self.assertEqual(dog_target.person_id, 1)
        self.assertEqual(work_target.person_id, 1)
        self.assertEqual(job_target.person_id, 1)
        self.assertIn("the user's pets", dog_prompt)
        self.assertIn("pet:dogs=Toby and Maxx", dog_prompt)
        self.assertIn("the user's work or job", work_prompt)

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

    def test_intent_classifier_handles_memory_observations_and_self_topics(self):
        from intelligence import intent_classifier

        self.assertEqual(
            intent_classifier.classify("Thats right, I wiped your memory at some point."),
            "general",
        )
        self.assertEqual(
            intent_classifier.classify("Can you tell me about my dogs?"),
            "query_memory",
        )
        self.assertEqual(
            intent_classifier.classify("Do you remember what I do for work?"),
            "query_memory",
        )

    def test_thanks_for_asking_is_not_closure(self):
        from intelligence import end_thread, response_length

        end_thread.clear()
        closure = end_thread.note_user_turn("I'm doing okay, thanks for asking", 1)
        plan = response_length.classify("I'm doing okay, thanks for asking")

        self.assertIsNone(closure)
        self.assertNotEqual(plan.target, "micro")
        end_thread.clear()

    def test_nice_chatting_and_going_to_go_are_closure(self):
        from intelligence import end_thread, response_length

        end_thread.clear()
        closure = end_thread.note_user_turn("I'm going to go now, nice chatting.", 1)
        plan = response_length.classify("I'm going to go now, nice chatting.")

        self.assertIsNotNone(closure)
        self.assertEqual(plan.target, "micro")
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
        self.assertIn("react to the human's compliment", directive)

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

        # A specific plan is now handled by the what-if/plans branch (a concrete
        # suggestion), NOT a generic friendship-interview question.
        self.assertIn("plan", directive.lower())
        self.assertNotIn("How did you end up talking to a droid DJ?", directive)

    def test_agenda_injects_friendship_question_into_reactive_turns(self):
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

        # REACTIVE_FRIENDSHIP_QUESTIONS_ENABLED is on by default: a normal reactive
        # turn (not a user question, plan, or sensitive beat) still surfaces a
        # profile question — but now as a secondary "fold in" behind a roast-first
        # reaction, not the primary purpose.
        self.assertIn("fold in this one question", directive)
        self.assertIn("REACT first", directive)

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

    def test_scream_sound_event_uses_surprise_frame_even_when_generic_sound_reactions_disabled(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness
        from world_state import world_state

        old_snapshot = consciousness._last_snapshot
        old_startle_at = consciousness._last_startle_sound_reaction_at
        old_self_state = world_state.get("self_state")
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
            "audio_scene": {"last_sound_event": "scream"},
            "animals": [],
            "time": {},
        }
        try:
            consciousness._last_snapshot = prev
            consciousness._last_startle_sound_reaction_at = 0.0
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak", return_value=True) as speak,
                mock.patch("sequences.animations.play_body_beat") as body_beat,
                mock.patch("config.WORLD_SOUND_EVENT_REACTIONS_ENABLED", False),
                mock.patch("config.WORLD_STARTLE_SOUND_EVENT_REACTIONS_ENABLED", True),
                mock.patch("config.STARTLE_SOUND_EVENT_REACTION_COOLDOWN_SECS", 0.0),
            ):
                consciousness._step_proactive_reactions(curr, profile)
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._last_startle_sound_reaction_at = old_startle_at
            world_state.update("self_state", old_self_state)

        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[1], "surprised")
        self.assertIn("startle sound", speak.call_args.kwargs["label"])
        body_beat.assert_called_once_with("surprise_pop")

    def test_new_furry_animal_uses_surprise_frame_and_lifeform_prompt(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness
        from world_state import world_state

        old_snapshot = consciousness._last_snapshot
        old_animals = set(consciousness._animal_seen_signatures)
        old_reacted = dict(consciousness._animal_reacted_at)
        old_pending_animals = dict(consciousness._pending_animal_arrivals)
        old_self_state = world_state.get("self_state")
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
            "audio_scene": {},
            "animals": [{
                "species": "dog",
                "position": "lower right",
                "furred": True,
            }],
            "time": {},
        }
        try:
            consciousness._last_snapshot = prev
            consciousness._animal_seen_signatures.clear()
            consciousness._animal_reacted_at.clear()
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_speak_async", side_effect=_speak_async_spoke) as speak,
                mock.patch("sequences.animations.play_body_beat") as body_beat,
            ):
                consciousness._step_proactive_reactions(curr, profile)
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._animal_seen_signatures.clear()
            consciousness._animal_seen_signatures.update(old_animals)
            consciousness._animal_reacted_at.clear()
            consciousness._animal_reacted_at.update(old_reacted)
            consciousness._pending_animal_arrivals.clear()
            consciousness._pending_animal_arrivals.update(old_pending_animals)
            world_state.update("self_state", old_self_state)

        speak.assert_called_once()
        self.assertIn("small furry lifeform", speak.call_args.args[0].lower())
        self.assertEqual(speak.call_args.args[1], "surprised")
        body_beat.assert_called_once_with("surprise_pop")

    def test_new_furry_animal_stays_pending_during_startup_greeting(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness
        from world_state import world_state

        old_snapshot = consciousness._last_snapshot
        old_animals = set(consciousness._animal_seen_signatures)
        old_reacted = dict(consciousness._animal_reacted_at)
        old_pending_animals = dict(consciousness._pending_animal_arrivals)
        old_self_state = world_state.get("self_state")
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
            "audio_scene": {},
            "animals": [{
                "species": "dog",
                "position": "lower right",
                "furred": True,
            }],
            "time": {},
        }
        try:
            consciousness._last_snapshot = prev
            consciousness._animal_seen_signatures.clear()
            consciousness._animal_reacted_at.clear()
            consciousness._pending_animal_arrivals.clear()
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=True),
                mock.patch.object(consciousness, "_speak_async", return_value=True) as speak,
            ):
                consciousness._step_proactive_reactions(curr, profile)

            speak.assert_not_called()
            self.assertIn("dog:lower right", consciousness._pending_animal_arrivals)

            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_speak_async", side_effect=_speak_async_spoke) as speak,
                mock.patch("sequences.animations.play_body_beat") as body_beat,
            ):
                consciousness._step_proactive_reactions(curr, profile)
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._animal_seen_signatures.clear()
            consciousness._animal_seen_signatures.update(old_animals)
            consciousness._animal_reacted_at.clear()
            consciousness._animal_reacted_at.update(old_reacted)
            consciousness._pending_animal_arrivals.clear()
            consciousness._pending_animal_arrivals.update(old_pending_animals)
            world_state.update("self_state", old_self_state)

        speak.assert_called_once()
        self.assertIn("small furry lifeform", speak.call_args.args[0].lower())
        body_beat.assert_called_once_with("surprise_pop")

    def test_identity_prompt_wait_suppresses_generic_world_reactions(self):
        from awareness.situation import SituationProfile
        from intelligence import consciousness

        old_snapshot = consciousness._last_snapshot
        old_until = consciousness._identity_prompt_reply_until
        old_pending = consciousness._pending_identity_prompt.is_set()
        old_in_flight = consciousness._identity_prompt_in_flight.is_set()
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
        prev = {
            "crowd": {"count": 0, "count_label": "empty"},
            "people": [],
            "animals": [],
            "audio_scene": {},
            "time": {},
        }
        curr = {
            "crowd": {"count": 1, "count_label": "alone"},
            "people": [{"id": "slot:person_1", "person_db_id": None}],
            "animals": [],
            "audio_scene": {},
            "time": {},
        }
        try:
            consciousness._last_snapshot = prev
            consciousness._identity_prompt_in_flight.clear()
            consciousness._pending_identity_prompt.set()
            consciousness._identity_prompt_reply_until = 120.0
            with (
                mock.patch.object(consciousness.time, "monotonic", return_value=110.0),
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_startup_known_greeting_pending", return_value=False),
                mock.patch.object(consciousness, "_generate_and_speak") as speak,
            ):
                consciousness._step_proactive_reactions(curr, profile)
        finally:
            consciousness._last_snapshot = old_snapshot
            consciousness._identity_prompt_in_flight.clear()
            consciousness._pending_identity_prompt.clear()
            if old_pending:
                consciousness._pending_identity_prompt.set()
            if old_in_flight:
                consciousness._identity_prompt_in_flight.set()
            consciousness._identity_prompt_reply_until = old_until

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
    def test_audio_scene_detects_scream_like_startle(self):
        import numpy as np
        from audio import scene
        import config

        sr = config.AUDIO_SAMPLE_RATE
        t = np.arange(int(sr * 0.75), dtype=np.float32) / float(sr)
        audio = (0.46 * np.sin(2 * np.pi * 1500.0 * t)).astype(np.float32)

        self.assertTrue(scene._detect_scream(audio))

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

    def test_offscreen_identity_prompt_accepts_unknown_bare_initials_reply(self):
        import numpy as np
        from intelligence import interaction

        old_pending = interaction._pending_offscreen_identify
        old_exchange_count = interaction._session_exchange_count
        pending_audio = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        reply_audio = np.array([2.0, 2.0], dtype=np.float32)
        interaction._pending_offscreen_identify = {
            "audio": pending_audio,
            "asked_at": interaction.time.monotonic(),
            "prior_engaged_id": 1,
            "prior_engaged_name": "Bret Benziger",
            "overheard_text": "I can't ask on the computer if it's looking at me",
            "anonymous_speaker_label": "unknown_voice_1",
        }
        try:
            with (
                mock.patch.object(
                    interaction.llm,
                    "extract_relationship_introduction",
                    return_value={"name": None, "relationship": None},
                ),
                mock.patch.object(
                    interaction.people_memory,
                    "find_or_create_person",
                    return_value=(77, True),
                ) as find_or_create,
                mock.patch.object(
                    interaction.config,
                    "IDENTITY_VOICE_ENROLL_MIN_AUDIO_SECS",
                    0.0,
                ),
                mock.patch.object(interaction.config, "IDENTITY_VOICE_ENROLL_MIN_WORDS", 1),
                mock.patch.object(interaction.speaker_id, "enroll_voice") as enroll_voice,
                mock.patch.object(interaction.people_memory, "update_familiarity") as update_familiarity,
                mock.patch.object(interaction, "_has_unknown_visible_person", return_value=False),
                mock.patch.object(interaction, "_bind_world_state_identity") as bind_identity,
                mock.patch.object(interaction, "_retire_anonymous_speaker_slot") as retire_slot,
                mock.patch.object(
                    interaction.llm,
                    "get_response",
                    return_value="JT, welcome aboard.",
                ),
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.conv_memory, "add_to_transcript") as add_transcript,
                mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
                mock.patch.object(interaction, "_register_rex_utterance") as register,
            ):
                consumed, response = interaction._handle_pending_offscreen_identify_reply(
                    "JT",
                    person_id=None,
                    person_name=None,
                    audio_array=reply_audio,
                    anonymous_speaker_label="unknown_voice_2",
                )

            self.assertTrue(consumed)
            self.assertEqual(response, "JT, welcome aboard.")
            self.assertIsNone(interaction._pending_offscreen_identify)
            find_or_create.assert_called_once_with("JT")
            # The gate is mocked open above (min audio 0.0 / min words 1), so the
            # bare "JT" sample enrolls; the gate-defers-short-samples behavior is
            # covered separately by test_voice_enrollment_requires_longer_sample.
            enroll_voice.assert_called_once()
            self.assertEqual(enroll_voice.call_args.args[0], 77)
            np.testing.assert_array_equal(
                enroll_voice.call_args.args[1],
                np.array([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            )
            update_familiarity.assert_called_once()
            bind_identity.assert_called_once_with(77, "JT")
            retired = {call.args[0] for call in retire_slot.call_args_list}
            self.assertEqual(retired, {"unknown_voice_1", "unknown_voice_2"})
            speak.assert_called_once_with("JT, welcome aboard.")
            add_transcript.assert_called_once_with("Rex", "JT, welcome aboard.")
            log_rex.assert_called_once_with("JT, welcome aboard.")
            register.assert_called_once_with("JT, welcome aboard.")
        finally:
            interaction._pending_offscreen_identify = old_pending
            interaction._session_exchange_count = old_exchange_count

    def test_offscreen_identity_confusion_reply_repairs_and_clears(self):
        import numpy as np
        from intelligence import interaction, repair_moves

        old_pending = interaction._pending_offscreen_identify
        old_exchange_count = interaction._session_exchange_count
        repair_moves.clear()
        interaction._pending_offscreen_identify = {
            "audio": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "asked_at": interaction.time.monotonic(),
            "prior_engaged_id": 1,
            "prior_engaged_name": "Bret Benziger",
            "overheard_text": "I only got 4 hours of sleep",
            "anonymous_speaker_label": "unknown_voice_1",
            "question_text": "Who's that off-screen, Bret?",
        }
        repair_moves.note_assistant_turn("Who's that off-screen, Bret?")
        try:
            with (
                mock.patch.object(
                    interaction.llm,
                    "extract_relationship_introduction",
                    return_value={"name": None, "relationship": None},
                ),
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.conv_memory, "add_to_transcript") as add_transcript,
                mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
                mock.patch.object(interaction, "_register_rex_utterance") as register,
                mock.patch.object(interaction.people_memory, "find_or_create_person") as find_or_create,
            ):
                consumed, response = interaction._handle_pending_offscreen_identify_reply(
                    "what are you talking about?",
                    person_id=1,
                    person_name="Bret Benziger",
                    audio_array=np.array([2.0], dtype=np.float32),
                    anonymous_speaker_label=None,
                )

            self.assertTrue(consumed)
            self.assertEqual(response, "Never mind. Bad sensor read on my end.")
            self.assertIsNone(interaction._pending_offscreen_identify)
            find_or_create.assert_not_called()
            speak.assert_called_once_with(
                "Never mind. Bad sensor read on my end.",
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            add_transcript.assert_called_once_with(
                "Rex",
                "Never mind. Bad sensor read on my end.",
            )
            log_rex.assert_called_once_with("Never mind. Bad sensor read on my end.")
            register.assert_called_once_with("Never mind. Bad sensor read on my end.")
        finally:
            interaction._pending_offscreen_identify = old_pending
            interaction._session_exchange_count = old_exchange_count
            repair_moves.clear()

    def test_offscreen_identity_nevermind_cancels_before_name_extraction(self):
        import numpy as np
        from intelligence import interaction, repair_moves

        old_pending = interaction._pending_offscreen_identify
        old_exchange_count = interaction._session_exchange_count
        repair_moves.clear()
        interaction._pending_offscreen_identify = {
            "audio": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "asked_at": interaction.time.monotonic(),
            "prior_engaged_id": 1,
            "prior_engaged_name": "Bret Benziger",
            "overheard_text": "I'm gonna keep clean everything",
            "anonymous_speaker_label": "unknown_voice_1",
            "question_text": "Who's that off-screen, Bret?",
        }
        try:
            with (
                mock.patch.object(
                    interaction.llm,
                    "extract_relationship_introduction",
                    return_value={"name": "Nevermind that", "relationship": None},
                ) as extract,
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
                mock.patch.object(interaction.conv_memory, "add_to_transcript") as add_transcript,
                mock.patch.object(interaction.conv_log, "log_rex") as log_rex,
                mock.patch.object(interaction, "_register_rex_utterance") as register,
                mock.patch.object(interaction.people_memory, "find_or_create_person") as find_or_create,
                mock.patch.object(interaction.speaker_id, "enroll_voice") as enroll_voice,
            ):
                consumed, response = interaction._handle_pending_offscreen_identify_reply(
                    "Nevermind that",
                    person_id=1,
                    person_name="Bret Benziger",
                    audio_array=np.array([2.0], dtype=np.float32),
                    anonymous_speaker_label=None,
                )

            self.assertTrue(consumed)
            self.assertEqual(response, "Never mind. Bad sensor read on my end.")
            self.assertIsNone(interaction._pending_offscreen_identify)
            extract.assert_not_called()
            find_or_create.assert_not_called()
            enroll_voice.assert_not_called()
            speak.assert_called_once_with(
                "Never mind. Bad sensor read on my end.",
                emotion="neutral",
                pre_beat_ms=100,
                post_beat_ms_override=200,
            )
            add_transcript.assert_called_once_with(
                "Rex",
                "Never mind. Bad sensor read on my end.",
            )
            log_rex.assert_called_once_with("Never mind. Bad sensor read on my end.")
            register.assert_called_once_with("Never mind. Bad sensor read on my end.")
        finally:
            interaction._pending_offscreen_identify = old_pending
            interaction._session_exchange_count = old_exchange_count
            repair_moves.clear()

    def test_anonymous_speaker_slot_reuses_matching_unknown_voice(self):
        import numpy as np
        from intelligence import interaction

        audio = np.zeros(1600, dtype=np.float32)
        first_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        second_embedding = np.array([0.99, 0.08, 0.0], dtype=np.float32)

        interaction._clear_anonymous_speaker_slots()
        try:
            with mock.patch.object(
                interaction.speaker_id,
                "get_embedding",
                side_effect=[first_embedding, second_embedding],
            ):
                first_label, first_score, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=None,
                    raw_best_name=None,
                    raw_best_score=0.0,
                )
                second_label, second_score, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=None,
                    raw_best_name=None,
                    raw_best_score=0.0,
                )

            self.assertEqual(first_label, "unknown_voice_1")
            self.assertIsNone(first_score)
            self.assertEqual(second_label, "unknown_voice_1")
            self.assertIsNotNone(second_score)
            self.assertGreaterEqual(second_score, 0.74)
            self.assertEqual(len(interaction._anonymous_speaker_slots), 1)
            self.assertEqual(interaction._anonymous_speaker_slots[0].turns, 2)
        finally:
            interaction._clear_anonymous_speaker_slots()

    def test_anonymous_speaker_slot_reuses_near_match_for_same_raw_candidate(self):
        import numpy as np
        from intelligence import interaction

        audio = np.zeros(1600, dtype=np.float32)
        first_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near_embedding = np.array([0.72, 0.694, 0.0], dtype=np.float32)

        interaction._clear_anonymous_speaker_slots()
        try:
            with mock.patch.object(
                interaction.speaker_id,
                "get_embedding",
                side_effect=[first_embedding, near_embedding],
            ):
                first_label, first_score, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=1,
                    raw_best_name="Bret",
                    raw_best_score=0.52,
                )
                second_label, second_score, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=1,
                    raw_best_name="Bret",
                    raw_best_score=0.53,
                )

            self.assertEqual(first_label, "unknown_voice_1")
            self.assertIsNone(first_score)
            self.assertEqual(second_label, "unknown_voice_1")
            self.assertIsNotNone(second_score)
            self.assertGreaterEqual(second_score, 0.70)
            self.assertLess(second_score, 0.74)
            self.assertEqual(len(interaction._anonymous_speaker_slots), 1)
            self.assertEqual(interaction._anonymous_speaker_slots[0].turns, 2)
        finally:
            interaction._clear_anonymous_speaker_slots()

    def test_anonymous_speaker_slot_does_not_sticky_match_different_raw_candidate(self):
        import numpy as np
        from intelligence import interaction

        audio = np.zeros(1600, dtype=np.float32)
        first_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        near_embedding = np.array([0.72, 0.694, 0.0], dtype=np.float32)

        interaction._clear_anonymous_speaker_slots()
        try:
            with mock.patch.object(
                interaction.speaker_id,
                "get_embedding",
                side_effect=[first_embedding, near_embedding],
            ):
                first_label, _, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=1,
                    raw_best_name="Bret",
                    raw_best_score=0.52,
                )
                second_label, second_score, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=2,
                    raw_best_name="JT",
                    raw_best_score=0.53,
                )

            self.assertEqual(first_label, "unknown_voice_1")
            self.assertEqual(second_label, "unknown_voice_2")
            self.assertIsNone(second_score)
            self.assertEqual(len(interaction._anonymous_speaker_slots), 2)
        finally:
            interaction._clear_anonymous_speaker_slots()

    def test_anonymous_speaker_slot_creates_new_label_for_different_voice(self):
        import numpy as np
        from intelligence import interaction

        audio = np.zeros(1600, dtype=np.float32)
        first_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        different_embedding = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        interaction._clear_anonymous_speaker_slots()
        try:
            with mock.patch.object(
                interaction.speaker_id,
                "get_embedding",
                side_effect=[first_embedding, different_embedding],
            ):
                first_label, _, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=None,
                    raw_best_name=None,
                    raw_best_score=0.0,
                )
                second_label, _, _ = interaction._resolve_anonymous_speaker_slot(
                    audio,
                    person_id=None,
                    raw_best_id=None,
                    raw_best_name=None,
                    raw_best_score=0.0,
                )

            self.assertEqual(first_label, "unknown_voice_1")
            self.assertEqual(second_label, "unknown_voice_2")
            self.assertEqual(len(interaction._anonymous_speaker_slots), 2)
        finally:
            interaction._clear_anonymous_speaker_slots()

    def test_known_speaker_does_not_create_anonymous_slot(self):
        import numpy as np
        from intelligence import interaction

        interaction._clear_anonymous_speaker_slots()
        try:
            with mock.patch.object(interaction.speaker_id, "get_embedding") as get_embedding:
                label, score, _ = interaction._resolve_anonymous_speaker_slot(
                    np.zeros(1600, dtype=np.float32),
                    person_id=1,
                    raw_best_id=1,
                    raw_best_name="Bret",
                    raw_best_score=0.91,
                )

            self.assertIsNone(label)
            self.assertIsNone(score)
            get_embedding.assert_not_called()
            self.assertEqual(interaction._anonymous_speaker_slots, [])
        finally:
            interaction._clear_anonymous_speaker_slots()


class PostResponseMemoryExtractionTest(unittest.TestCase):
    def test_offscreen_identity_self_answer_rejects_third_person_presence(self):
        from intelligence import interaction

        self.assertFalse(
            interaction._looks_like_direct_offscreen_identity_answer(
                "Sara, is here",
                "Sara",
            )
        )
        self.assertFalse(
            interaction._looks_like_direct_offscreen_identity_answer(
                "Sarah is here",
                "Sarah",
            )
        )
        self.assertTrue(
            interaction._looks_like_direct_offscreen_identity_answer(
                "JT",
                "JT",
            )
        )
        self.assertTrue(
            interaction._looks_like_direct_offscreen_identity_answer(
                "I'm Sarah",
                "Sarah",
            )
        )

    def test_single_visible_face_does_not_override_hard_voice_match(self):
        from intelligence import interaction

        override = interaction._single_visible_face_voice_override(
            resolved_person_id=3,
            ws_person={"person_db_id": 2, "face_id": "Jeff"},
            visible_known_by_id={2: {"person_db_id": 2, "face_id": "Jeff"}},
            has_unknown_visible_or_recent=False,
            speaker_score=0.836,
            hard_threshold=0.75,
        )

        self.assertIsNone(override)

    def test_single_visible_face_can_override_weak_conflicting_voice(self):
        from intelligence import interaction

        override = interaction._single_visible_face_voice_override(
            resolved_person_id=3,
            ws_person={"person_db_id": 2, "face_id": "Jeff"},
            visible_known_by_id={2: {"person_db_id": 2, "face_id": "Jeff"}},
            has_unknown_visible_or_recent=False,
            speaker_score=0.62,
            hard_threshold=0.75,
        )

        self.assertEqual(override, (2, "Jeff"))

    def test_idle_known_offcamera_crosstalk_is_ignored_without_wake_or_command(self):
        from intelligence import interaction

        with mock.patch.object(
            interaction,
            "_known_person_visible_recently",
            return_value=False,
        ):
            self.assertTrue(
                interaction._should_ignore_idle_background_speech(
                    from_idle_activation=True,
                    person_id=3,
                    has_unknown_visible=False,
                    identity_prompt_active=False,
                    text_input=False,
                    text="the Justin shorts are good, which came out nice",
                )
            )
            self.assertFalse(
                interaction._should_ignore_idle_background_speech(
                    from_idle_activation=True,
                    person_id=3,
                    has_unknown_visible=False,
                    identity_prompt_active=False,
                    text_input=False,
                    text="Rex, how do you know so much?",
                )
            )
            self.assertFalse(
                interaction._should_ignore_idle_background_speech(
                    from_idle_activation=True,
                    person_id=3,
                    has_unknown_visible=False,
                    identity_prompt_active=False,
                    text_input=False,
                    text="Play some country music.",
                )
            )

    def test_visible_unknown_identity_prompt_ignores_obvious_crosstalk(self):
        from intelligence import interaction

        self.assertTrue(
            interaction._looks_like_background_crosstalk(
                "It was powered off channels that we can"
            )
        )
        self.assertTrue(
            interaction._looks_like_background_crosstalk(
                "when you're talking to it right"
            )
        )
        self.assertFalse(
            interaction._looks_like_background_crosstalk(
                "how do you know so much?"
            )
        )

    def test_identity_prompts_defer_to_direct_questions_and_commands(self):
        from intelligence import interaction

        self.assertTrue(interaction._turn_should_defer_identity_prompts("how do you know so much?"))
        self.assertTrue(interaction._turn_should_defer_identity_prompts("Play some country music."))
        self.assertFalse(interaction._turn_should_defer_identity_prompts("I'm Sara"))
        self.assertFalse(interaction._utterance_invites_identity_question("how do you know so much?"))
        self.assertTrue(interaction._utterance_invites_identity_question("Hey Rex"))

    def test_shutdown_requested_checks_state_and_stop_event(self):
        from intelligence import interaction
        from state import State

        interaction._stop_event.clear()
        with mock.patch.object(interaction.state_module, "get_state", return_value=State.SHUTDOWN):
            self.assertTrue(interaction._shutdown_requested())
        with mock.patch.object(interaction.state_module, "get_state", return_value=State.ACTIVE):
            self.assertFalse(interaction._shutdown_requested())
            interaction._stop_event.set()
            try:
                self.assertTrue(interaction._shutdown_requested())
            finally:
                interaction._stop_event.clear()

    def test_general_slow_path_ack_skips_short_turns_and_reply_frames(self):
        from intelligence import dialogue_act, interaction

        act = dialogue_act.DialogueActDecision(
            "answer_to_rex",
            0.90,
            "reply to last Rex turn",
            skip_action_router=True,
        )

        self.assertFalse(interaction._slow_path_ack_allowed_for_turn("general", "cataracts"))
        self.assertFalse(
            interaction._slow_path_ack_allowed_for_turn(
                "general",
                "I need to turn over where?",
            )
        )
        self.assertFalse(
            interaction._slow_path_ack_allowed_for_turn(
                "general",
                "this is a longer reply but it belongs to Rex's pending question",
                act,
            )
        )
        self.assertTrue(
            interaction._slow_path_ack_allowed_for_turn(
                "general",
                "this is a longer conversational thought that might need an actual answer",
            )
        )

    def test_voice_enrollment_requires_longer_sample(self):
        import numpy as np
        import config
        from intelligence import interaction

        short = np.zeros(int(config.AUDIO_SAMPLE_RATE * 0.25), dtype=np.float32)
        long = np.zeros(int(config.AUDIO_SAMPLE_RATE * 1.4), dtype=np.float32)

        allowed, reason = interaction._voice_enrollment_sample_allowed(
            short,
            transcript_text="Sara Ever",
        )
        self.assertFalse(allowed)
        self.assertIn("audio_too_short", reason)

        allowed, reason = interaction._voice_enrollment_sample_allowed(
            long,
            transcript_text="Sara Ever",
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "ok")

    def test_proactive_unknown_identity_prompt_does_not_fire_while_active(self):
        from intelligence import consciousness
        import state as state_module
        from state import State

        old_state = state_module.get_state()
        try:
            state_module.set_state(State.ACTIVE)
            with (
                mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
                mock.patch.object(consciousness, "_speak_async", return_value=True) as speak,
            ):
                consciousness._maybe_prompt_unknown_identity(
                    unknown_count=1,
                    known_unique=[],
                )
            speak.assert_not_called()
        finally:
            state_module.set_state(old_state)

    def test_memory_extractors_use_turn_transcript_snapshot(self):
        from intelligence import interaction

        snapshot = [
            {"speaker": "Bret", "text": "I like jazz"},
            {"speaker": "Rex", "text": "Jazz. Brave choice."},
        ]

        class ImmediateThread:
            def __init__(self, *args, **kwargs):
                self._target = kwargs.get("target")

            def start(self):
                if self._target is not None:
                    self._target()

        with (
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.events_memory, "looks_like_cancellation", return_value=False),
            mock.patch.object(interaction.consciousness, "get_pending_followup", return_value=[]),
            mock.patch.object(interaction.interoception, "record_interaction"),
            mock.patch.object(interaction.threading, "Thread", ImmediateThread),
            mock.patch.object(interaction.llm, "analyze_sentiment", return_value={}),
            mock.patch.object(interaction.friendship_patterns, "learn_from_turn"),
            mock.patch.object(interaction.conv_memory, "get_session_transcript", return_value=snapshot) as transcript,
            mock.patch.object(interaction, "_filter_forgotten_transcript", side_effect=lambda recent, _pid: list(recent)),
            mock.patch.object(interaction.llm, "extract_facts", return_value=[]) as facts,
            mock.patch.object(interaction.llm, "extract_preferences", return_value=[]) as preferences,
            mock.patch.object(interaction.llm, "extract_interests", return_value=[]) as interests,
            mock.patch.object(interaction.llm, "extract_events", return_value=[]) as events,
        ):
            interaction._post_response("I like jazz", 1, "Bret")

        transcript.assert_called_once()
        facts.assert_called_once_with(1, snapshot, person_name="Bret")
        preferences.assert_called_once_with(1, snapshot, person_name="Bret")
        interests.assert_called_once_with(1, snapshot, person_name="Bret")
        events.assert_called_once_with(1, snapshot, person_name="Bret")


class MusicMuzzleConsentTest(unittest.TestCase):
    def test_keep_it_muzzled_classifies_as_decline(self):
        from intelligence import interaction

        for text in (
            "let's keep it muzzled for now",
            "lets keep it muzzled for now",
            "keep it muzzled",
            "keep the jukebox muzzled",
            "muzzle the music",
            "no music",
            "keep it off",
            "hold off",
        ):
            self.assertIs(
                interaction._classify_consent(text),
                False,
                msg=f"expected decline for {text!r}",
            )

    def test_plain_affirmations_still_classify_as_consent(self):
        from intelligence import interaction

        for text in ("yes", "yeah", "sure", "okay", "please do", "let's do it"):
            self.assertIs(
                interaction._classify_consent(text),
                True,
                msg=f"expected consent for {text!r}",
            )

    def test_muzzle_reply_to_offer_does_not_start_music(self):
        from intelligence import interaction

        interaction._pending_music_offer = {
            "person_id": 1,
            "music_query": "classical",
            "asked_at": interaction.time.monotonic(),
        }
        with (
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction, "_start_dj_after_response") as start_dj,
        ):
            resp = interaction._handle_pending_music_offer_reply(
                1, "let's keep it muzzled for now"
            )

        self.assertIsNotNone(resp)
        self.assertIn("logged the taste", resp)
        start_dj.assert_not_called()
        self.assertIsNone(interaction._pending_music_offer)
        speak.assert_called_once()

    def test_yes_reply_to_offer_starts_music(self):
        from intelligence import interaction
        import features.dj as dj_mod

        interaction._pending_music_offer = {
            "person_id": 1,
            "music_query": "classical",
            "asked_at": interaction.time.monotonic(),
        }
        fake_track = dj_mod.TrackInfo("radio", "Classical KDFC", "http://x", "")
        with (
            mock.patch.object(dj_mod, "handle_request", return_value=fake_track),
            mock.patch.object(interaction, "_speak_blocking", return_value=True),
            mock.patch.object(interaction, "_start_dj_after_response") as start_dj,
        ):
            resp = interaction._handle_pending_music_offer_reply(1, "yeah go for it")

        self.assertIsNotNone(resp)
        start_dj.assert_called_once_with(fake_track)
        self.assertIsNone(interaction._pending_music_offer)

    def test_command_parser_maps_muzzle_to_dj_stop(self):
        from intelligence import command_parser

        for text in ("muzzle", "muzzle it", "muzzle the music", "muzzle the jukebox"):
            match = command_parser.parse(text)
            self.assertIsNotNone(match, msg=f"no match for {text!r}")
            self.assertEqual(match.command_key, "dj_stop", msg=f"for {text!r}")


class WakeWordDjBargeInTest(unittest.TestCase):
    def test_stop_dj_for_wake_stops_playback_and_clears_tail(self):
        from intelligence import interaction
        import features.dj as dj_mod

        with (
            mock.patch.object(dj_mod, "is_playing", return_value=True),
            mock.patch.object(dj_mod, "stop") as dj_stop,
            mock.patch.object(interaction.echo_cancel, "clear_suppression_tail") as clear_tail,
        ):
            stopped = interaction._stop_dj_for_wake()

        self.assertTrue(stopped)
        dj_stop.assert_called_once()
        clear_tail.assert_called_once()

    def test_stop_dj_for_wake_noop_when_not_playing(self):
        from intelligence import interaction
        import features.dj as dj_mod

        with (
            mock.patch.object(dj_mod, "is_playing", return_value=False),
            mock.patch.object(dj_mod, "stop") as dj_stop,
            mock.patch.object(interaction.echo_cancel, "clear_suppression_tail") as clear_tail,
        ):
            stopped = interaction._stop_dj_for_wake()

        self.assertFalse(stopped)
        dj_stop.assert_not_called()
        clear_tail.assert_not_called()

    def test_wake_threshold_drops_during_dj_playback(self):
        from audio import wake_word

        with (
            mock.patch.object(wake_word.config, "WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA", 0.15),
            mock.patch.object(wake_word.config, "WAKE_WORD_MIN_THRESHOLD", 0.30),
        ):
            base = wake_word._threshold("Hey_rex", dj_playing=False)
            ducked = wake_word._threshold("Hey_rex", dj_playing=True)

        self.assertAlmostEqual(base, 0.5)
        self.assertAlmostEqual(ducked, 0.35)
        self.assertLess(ducked, base)

    def test_wake_threshold_never_below_floor(self):
        from audio import wake_word

        with (
            mock.patch.object(wake_word.config, "WAKE_WORD_DJ_PLAYBACK_THRESHOLD_DELTA", 0.9),
            mock.patch.object(wake_word.config, "WAKE_WORD_MIN_THRESHOLD", 0.30),
        ):
            ducked = wake_word._threshold("Hey_rex", dj_playing=True)

        self.assertAlmostEqual(ducked, 0.30)

    def test_wake_threshold_drops_during_rex_tts_so_user_can_interrupt(self):
        # Rex's own speech masks an interrupting "hey rex" in the mic, so the bar
        # must drop while he's talking (mid-sentence barge-in). DJ takes precedence.
        from audio import wake_word

        with (
            mock.patch.object(wake_word.config, "WAKE_WORD_TTS_PLAYBACK_THRESHOLD_DELTA", 0.15),
            mock.patch.object(wake_word.config, "WAKE_WORD_MIN_THRESHOLD", 0.30),
        ):
            base = wake_word._threshold("Hey_rex")
            ducked = wake_word._threshold("Hey_rex", tts_playing=True)

        self.assertAlmostEqual(base, 0.5)
        self.assertAlmostEqual(ducked, 0.35)
        self.assertLess(ducked, base)

    def test_sd_guard_serializes_and_settles_stop(self):
        # Wake-word barge-in does a cross-thread sd.stop() + immediate replay; the
        # guard must wrap play/stop and hold a settle after a stop so the global
        # CoreAudio stream isn't re-initialized mid-teardown (the Trace/BPT crash).
        import time
        import sounddevice as sd
        from audio import sd_guard
        import config

        real_play, real_stop, installed_before = sd.play, sd.stop, sd_guard._installed
        try:
            calls = []
            sd.play = lambda *a, **k: calls.append("play")
            sd.stop = lambda *a, **k: calls.append("stop")
            sd_guard._installed = False

            with mock.patch.object(config, "AUDIO_PLAYBACK_STOP_SETTLE_SECS", 0.03):
                self.assertTrue(sd_guard.install())
                self.assertTrue(sd_guard.install())      # idempotent
                self.assertIsNot(sd.play, real_play)      # wrapped
                self.assertIsNot(sd.stop, real_stop)

                t0 = time.monotonic()
                sd.stop()
                self.assertGreaterEqual(time.monotonic() - t0, 0.025)  # settle applied
                sd.play()
                self.assertEqual(calls, ["stop", "play"])  # underlying calls reached
        finally:
            sd.play, sd.stop, sd_guard._installed = real_play, real_stop, installed_before


class StartupGreetingOpenerTest(unittest.TestCase):
    def test_startup_profile_question_disabled_by_default(self):
        from intelligence import consciousness

        # Even with a sparse profile that would otherwise yield a question, the
        # cold open must stay casual — no profile question.
        with (
            mock.patch.object(consciousness.config, "STARTUP_PROFILE_QUESTION_ENABLED", False),
            mock.patch.object(consciousness.profile_questions, "profile_fact_count", return_value=0),
            mock.patch.object(
                consciousness.profile_questions,
                "next_profile_question",
                return_value={"key": "favorite_music", "text": "What kind of music are you into?", "depth": 1},
            ) as next_q,
        ):
            self.assertIsNone(consciousness._pick_startup_profile_question(1))
            next_q.assert_not_called()

    def test_startup_profile_question_returns_question_when_enabled(self):
        from intelligence import consciousness
        from intelligence import question_budget

        question = {"key": "hometown", "text": "So where are you from?", "depth": 1}
        with (
            mock.patch.object(consciousness.config, "STARTUP_PROFILE_QUESTION_ENABLED", True),
            mock.patch.object(consciousness.config, "LOW_MEMORY_IDLE_QUESTION_ENABLED", True),
            mock.patch.object(consciousness.profile_questions, "profile_fact_count", return_value=0),
            mock.patch.object(question_budget, "can_ask", return_value=True),
            mock.patch.object(
                consciousness.profile_questions,
                "next_profile_question",
                return_value=question,
            ),
        ):
            self.assertEqual(consciousness._pick_startup_profile_question(1), question)


class HardwareAecBoundaryTest(unittest.TestCase):
    """When the ReSpeaker Lite hardware AEC is active, the post-TTS deaf window is
    shrunk so a human reply landing as Rex finishes is still captured. On any
    non-ReSpeaker machine the gate is off and the original tuning (asserted by
    PostTtsHandoffPolicyTest) is unchanged."""

    def setUp(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0

    def test_tail_shrinks_when_aec_active(self):
        from intelligence import interaction
        with mock.patch("audio.hardware_aec.is_active", return_value=True):
            self.assertEqual(interaction._reply_playback_tail_secs(True), 0.05)
            self.assertEqual(interaction._reply_playback_tail_secs(False), 0.05)

    def test_tail_unchanged_when_aec_inactive(self):
        from intelligence import interaction
        with mock.patch("audio.hardware_aec.is_active", return_value=False):
            # Statements (False) now use the same short tail as questions (True) so a
            # reply landing right after a statement isn't clipped (live 2026-06-18).
            self.assertEqual(interaction._reply_playback_tail_secs(True), 0.12)
            self.assertEqual(interaction._reply_playback_tail_secs(False), 0.12)

    def test_statement_handoff_matches_question_suppression(self):
        """Statements get the identical whisper-suppression handoff as questions:
        same post-playback attenuation tail AND the same listen-resume delay, so a
        reply spoken right after a statement keeps its opening words."""
        from intelligence import interaction
        with mock.patch("audio.hardware_aec.is_active", return_value=False):
            self.assertEqual(
                interaction._reply_playback_tail_secs(False),   # statement
                interaction._reply_playback_tail_secs(True),    # question
            )
        statement = interaction._post_tts_handoff_policy("Classic choice.")
        question = interaction._post_tts_handoff_policy("What's your favorite movie?")
        self.assertEqual(statement.listen_delay_secs, question.listen_delay_secs)
        self.assertEqual(statement.flush_buffer, question.flush_buffer)

    def test_handoff_resumes_fast_and_reaches_back_when_aec_active(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0
        with (
            mock.patch("audio.hardware_aec.is_active", return_value=True),
            mock.patch.object(interaction.time, "monotonic", return_value=100.0),
            mock.patch.object(interaction.stream, "flush"),
            mock.patch.object(interaction.vad, "reset_state"),
        ):
            interaction._apply_post_tts_handoff("The sky is blue.", source="test")
        # resume ~immediately (0.05s) and capture floor reaches 0.5s back past handoff
        self.assertAlmostEqual(interaction._listen_resume_at, 100.05)
        self.assertAlmostEqual(interaction._listen_capture_floor_at, 99.5)


class PostQuestionHandoffStickinessTest(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0
        # Assert default (non-hardware-AEC) tuning deterministically — see note in
        # PostTtsHandoffPolicyTest.setUp.
        _aec = mock.patch("audio.hardware_aec.is_active", return_value=False)
        _aec.start()
        self.addCleanup(_aec.stop)

    def tearDown(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0

    def _apply(self, text):
        from intelligence import interaction
        with (
            mock.patch.object(interaction.stream, "flush") as flush,
            mock.patch.object(interaction.vad, "reset_state"),
        ):
            interaction._apply_post_tts_handoff(text, source="test")
        return flush

    def test_question_handoff_does_not_flush(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0
        with mock.patch.object(interaction.config, "POST_QUESTION_FLUSH_AUDIO_BUFFER", False):
            flush = self._apply("What's his name?")
        flush.assert_not_called()
        self.assertFalse(interaction._post_tts_flush_needed)

    def test_trailing_statement_after_question_stays_responsive(self):
        from intelligence import interaction
        interaction._last_fast_handoff_at = 0.0
        with (
            mock.patch.object(interaction.config, "POST_QUESTION_HANDOFF_STICKY_SECS", 1.5),
            mock.patch.object(interaction.config, "POST_QUESTION_FLUSH_AUDIO_BUFFER", False),
        ):
            # Rex asks a question, then his trailing sentence is a statement.
            self._apply("What's his name?")
            flush = self._apply("Bet it's a good one.")
        # The trailing statement must NOT flush the buffer (would delete the
        # human's immediate answer to the question).
        flush.assert_not_called()
        self.assertFalse(interaction._post_tts_flush_needed)
        self.assertLessEqual(
            interaction._listen_resume_at - interaction._last_speech_at, 0.12 + 1e-6
        )

    def test_statement_outside_sticky_window_uses_normal_delay(self):
        from intelligence import interaction
        import config
        # No recent question handoff → ordinary statement. It still must NOT flush
        # (statements preserve the buffer now so an immediate reply survives), but it
        # uses the normal statement listen delay, not the shorter sticky/question one.
        interaction._last_fast_handoff_at = interaction.time.monotonic() - 100.0
        with mock.patch.object(interaction.config, "POST_QUESTION_HANDOFF_STICKY_SECS", 1.5):
            flush = self._apply("Toys are the modern navigational console.")
        flush.assert_not_called()
        self.assertFalse(interaction._post_tts_flush_needed)
        self.assertAlmostEqual(
            interaction._listen_resume_at - interaction._last_speech_at,
            float(config.POST_SPEECH_LISTEN_DELAY_SECS),
            places=6,
        )


class CrosstalkSuppressionTest(unittest.TestCase):
    def test_clear_third_party_lines_are_crosstalk(self):
        from intelligence import interaction
        for text in (
            "Love you too",
            "I love you",
            "love you too.",
            "go ahead this one babe or the far one",
            "okay, honey",
            "no, sweetheart",
            "darling, can you grab that?",
        ):
            self.assertTrue(
                interaction._looks_like_third_party_crosstalk(text),
                msg=f"expected crosstalk for {text!r}",
            )

    def test_rex_directed_and_neutral_lines_are_not_crosstalk(self):
        from intelligence import interaction
        for text in (
            "Rex, I love you",          # names Rex
            "I love this song",         # not "love you"
            "vodka and orange juice",   # ingredients
            "pass the honey",           # honey as a noun, not a vocative
            "what's your favorite movie?",
            "play some jazz",
            "I'd love you to play something upbeat",  # request to Rex, not affection
        ):
            self.assertFalse(
                interaction._looks_like_third_party_crosstalk(text),
                msg=f"did not expect crosstalk for {text!r}",
            )


class RhetoricalQuestionGuardTest(unittest.TestCase):
    def test_rhetorical_who_questions_do_not_expect_response(self):
        from intelligence import interaction
        for text in (
            "Honestly, who doesn't appreciate a droid that can sing?",
            "Who wouldn't want that?",
            "Who hasn't dreamed of piloting a star cruiser?",
        ):
            self.assertFalse(
                interaction._question_expects_response(text),
                msg=f"rhetorical question should not expect a response: {text!r}",
            )

    def test_real_questions_still_expect_response(self):
        from intelligence import interaction
        for text in (
            "What's your favorite movie?",
            "So where are you from?",
            "What's his name?",
        ):
            self.assertTrue(
                interaction._question_expects_response(text),
                msg=f"real question should expect a response: {text!r}",
            )


class EndThreadClosureNarrowingTest(unittest.TestCase):
    def setUp(self):
        from intelligence import end_thread
        end_thread.clear()

    def tearDown(self):
        from intelligence import end_thread
        end_thread.clear()

    def test_compliment_reply_does_not_arm_grace(self):
        from intelligence import end_thread
        # "Well thank you" is a reply to a compliment, not a goodbye. It must NOT
        # arm the end-of-thread grace window (which muzzled proactive banter).
        end_thread.note_assistant_turn("Keeping you from looking like a lost Womp Rat.")
        end_thread.note_user_turn("Well thank you", 1)
        self.assertFalse(end_thread.is_grace_active())

    def test_real_goodbye_still_arms_grace(self):
        from intelligence import end_thread
        end_thread.note_user_turn("alright, I gotta go", 1)
        self.assertTrue(end_thread.is_grace_active())

    def test_bare_thanks_after_question_still_closes(self):
        from intelligence import end_thread
        # A bare ack after Rex asked a question is still a legitimate soft close.
        end_thread.note_assistant_turn("Want me to queue something up?")
        end_thread.note_user_turn("thanks", 1)
        self.assertTrue(end_thread.is_grace_active())

    def test_thats_all_closes(self):
        from intelligence import end_thread
        end_thread.note_user_turn("that's all", 1)
        self.assertTrue(end_thread.is_grace_active())


class IdleBanterTest(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction, consciousness, callback_engine
        interaction._idle_banter_count = 0
        interaction._last_idle_banter_at = 0.0
        # Proactive paths hold off if another proactive line fired in the last few
        # seconds, or while Rex is awaiting an answer — clear the leaked globals so a
        # prior test can't suppress banter.
        interaction._last_proactive_line_at = 0.0
        interaction._floor_held_until = 0.0  # any prior question-bearing Rex line arms this
        interaction._idle_banter_threshold = None
        interaction._recently_banned_topics.clear()
        interaction._recent_rex_questions.clear()
        consciousness.clear_response_wait()
        # Proactive speech is now also suppressed during the "give space after a heavy
        # moment" sober window + an open grief flow. Another test may have armed those
        # globals, so clear them here (these tests are about banter, not grief).
        callback_engine.reset_state_for_tests()
        interaction._grief_flow_state.clear()
        # Isolate the in-session plans/holiday dedup from the PERSISTENT (real-DB) store:
        # these tests exercise the per-session logic and must not depend on, or write to,
        # assets/memory/people.db.
        self._plan_patches = [
            mock.patch("memory.relationships.was_proactive_asked", return_value=False),
            mock.patch("memory.relationships.mark_proactive_asked"),
        ]
        for p in self._plan_patches:
            p.start()

    def tearDown(self):
        from intelligence import interaction, consciousness, callback_engine
        for p in getattr(self, "_plan_patches", []):
            p.stop()
        interaction._idle_banter_count = 0
        interaction._last_idle_banter_at = 0.0
        interaction._last_proactive_line_at = 0.0
        interaction._floor_held_until = 0.0
        interaction._idle_banter_threshold = None
        interaction._recently_banned_topics.clear()
        interaction._recent_rex_questions.clear()
        consciousness.clear_response_wait()
        callback_engine.reset_state_for_tests()
        interaction._grief_flow_state.clear()

    def test_idle_plans_pivot_does_not_repeat_a_holiday(self):
        from intelligence import interaction, consciousness
        from awareness import holidays
        interaction._idle_plans_asked.clear()
        consciousness._holiday_plans_asked.clear()
        holiday = {
            "name": "Juneteenth National Independence Day", "when": "this Friday",
            "date": "2026-06-19", "days_until": 2, "window": "minor",
        }
        with mock.patch.object(holidays, "next_relevant_holiday", return_value=holiday):
            d1, m1 = interaction._idle_plans_directive(person_id=1)
            self.assertIn("Juneteenth", d1)
            interaction._mark_idle_plans_asked(m1)  # simulate the line actually speaking
            # Second pivot must NOT ask Juneteenth again — falls back to generic plans.
            d2, m2 = interaction._idle_plans_directive(person_id=1)
            self.assertNotIn("Juneteenth", d2)
            self.assertTrue((m2 or {}).get("generic"))
            interaction._mark_idle_plans_asked(m2)
            # Third pivot is exhausted -> empty so the caller keeps deepening the topic.
            d3, m3 = interaction._idle_plans_directive(person_id=1)
            self.assertEqual(d3, "")
            self.assertIsNone(m3)
        # Cross-path: the consciousness holiday step won't re-ask it either.
        self.assertIn((1, "2026-06-19"), consciousness._holiday_plans_asked)
        interaction._idle_plans_asked.clear()
        consciousness._holiday_plans_asked.clear()

    def test_unspoken_plans_pivot_does_not_burn_the_holiday(self):
        # The marker is recorded only when the line speaks; selecting a directive that
        # never speaks must not mark the holiday asked.
        from intelligence import interaction
        from awareness import holidays
        interaction._idle_plans_asked.clear()
        holiday = {"name": "Juneteenth National Independence Day", "when": "this Friday",
                   "date": "2026-06-19", "days_until": 2, "window": "minor"}
        with mock.patch.object(holidays, "next_relevant_holiday", return_value=holiday):
            interaction._idle_plans_directive(person_id=1)  # selected, but NOT marked
            d2, _ = interaction._idle_plans_directive(person_id=1)
            self.assertIn("Juneteenth", d2)  # still available — wasn't burned
        interaction._idle_plans_asked.clear()

    def test_banter_fires_after_silence_and_drives_conversation(self):
        from intelligence import interaction
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(interaction.llm, "get_response", return_value="So what's the dog up to?"),
            mock.patch.object(interaction.llm, "clean_response_text", side_effect=lambda s: s),
            mock.patch.object(interaction.social_frame, "build_frame"),
            mock.patch.object(
                interaction.social_frame, "govern_response",
                return_value=type("G", (), {"text": "So what's the dog up to?"})(),
            ),
            mock.patch("audio.tts.ensure_cached", return_value=True),
            mock.patch.object(interaction.barge_guard, "user_speaking_now", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript"),
            mock.patch.object(interaction.conv_log, "log_rex"),
            mock.patch.object(interaction, "_register_rex_utterance"),
            # LEGACY (inline-speak) path: pin enforce OFF so this test stays valid
            # regardless of the ACTION_GOVERNOR_ENFORCE default. The enforce path
            # (submit-a-candidate, deferred speak) is covered by the test below.
            mock.patch.object(interaction, "_governor_enforcing", return_value=False),
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
        self.assertTrue(fired)
        speak.assert_called_once()
        self.assertEqual(interaction._idle_banter_count, 1)

    def test_banter_under_enforce_submits_candidate_instead_of_speaking_inline(self):
        """Consolidation enforce mode: idle banter is an interaction-thread BYPASSER —
        under ACTION_GOVERNOR_ENFORCE it must SUBMIT a candidate (so the governor can
        arbitrate it against the consciousness tick) and arm its cooldown on submit,
        NOT speak inline. The deferred speak_fn drives the conversation only if idle
        banter wins the tick."""
        from intelligence import interaction
        from intelligence.action_governor import governor
        interaction._idle_banter_count = 0
        interaction._last_idle_banter_at = 0.0
        submitted = []
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(interaction.llm, "get_response", return_value="So what's the dog up to?"),
            mock.patch.object(interaction.llm, "clean_response_text", side_effect=lambda s: s),
            mock.patch.object(interaction.social_frame, "build_frame"),
            mock.patch.object(
                interaction.social_frame, "govern_response",
                return_value=type("G", (), {"text": "So what's the dog up to?"})(),
            ),
            mock.patch("audio.tts.ensure_cached", return_value=True),
            mock.patch.object(interaction.barge_guard, "user_speaking_now", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
            mock.patch.object(interaction.conv_memory, "add_to_transcript"),
            mock.patch.object(interaction.conv_log, "log_rex"),
            mock.patch.object(interaction, "_register_rex_utterance"),
            mock.patch.object(interaction, "_governor_enforcing", return_value=True),
            mock.patch.object(governor, "submit_external", side_effect=lambda c: submitted.append(c)),
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
            # Submitted a candidate; did NOT speak inline; armed the cooldown on submit.
            self.assertTrue(fired)
            speak.assert_not_called()
            self.assertEqual(interaction._idle_banter_count, 1)
            self.assertEqual(len(submitted), 1)
            candidate = submitted[0]
            self.assertEqual(candidate.purpose, "idle_monologue")
            self.assertEqual(candidate.source, "interaction._maybe_idle_banter")
            self.assertIsNotNone(candidate.speak_fn)
            # The governor winner runs the deferred speak_fn → THAT drives the line.
            self.assertTrue(candidate.speak_fn())
            speak.assert_called_once()

    def test_banter_submit_arms_proactive_gap_so_inline_paths_back_off(self):
        """Live bug 2026-06-17: under ENFORCE, idle banter is LLM-generated and doesn't
        speak for several seconds; in that window the inline low-memory profile question
        stacked a SECOND question. Submitting the candidate must arm the SHARED proactive
        gap immediately, so _proactive_line_recently_fired() blocks the inline paths."""
        from intelligence import interaction
        from intelligence.action_governor import governor
        interaction._idle_banter_count = 0
        interaction._last_idle_banter_at = 0.0
        interaction._last_proactive_line_at = 0.0
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(interaction, "_governor_enforcing", return_value=True),
            mock.patch.object(governor, "submit_external", side_effect=lambda c: None),
        ):
            # Before: gap not armed.
            self.assertFalse(interaction._proactive_line_recently_fired())
            fired = interaction._maybe_idle_banter(
                idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
            self.assertTrue(fired)
            # After submit (BEFORE the deferred line speaks): the shared gap is armed, so
            # an inline proactive path would back off instead of stacking a 2nd question.
            self.assertTrue(interaction._proactive_line_recently_fired())
        interaction._last_proactive_line_at = 0.0

    def test_banter_holds_until_threshold(self):
        from intelligence import interaction
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            fired = interaction._maybe_idle_banter(idle_for=3.0, effective_idle_timeout=45.0)
        self.assertFalse(fired)
        speak.assert_not_called()

    def test_banter_caps_attempts_per_stretch(self):
        from intelligence import interaction
        # At the configured cap, banter must stop (read the value so this test
        # stays correct if the cap is retuned).
        interaction._idle_banter_count = int(
            getattr(interaction.config, "IDLE_BANTER_MAX_PER_STRETCH", 2)
        )
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
        self.assertFalse(fired)
        speak.assert_not_called()

    def test_banter_respects_cooldown(self):
        from intelligence import interaction
        interaction._idle_banter_count = 1
        interaction._last_idle_banter_at = interaction.time.monotonic()  # just fired
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
        self.assertFalse(fired)
        speak.assert_not_called()

    def test_banter_skipped_when_no_one_engaged(self):
        from intelligence import interaction
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=None),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=False),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
        self.assertFalse(fired)
        speak.assert_not_called()

    def test_banter_skipped_during_end_thread_grace(self):
        from intelligence import interaction
        with (
            mock.patch.object(interaction, "_primary_session_person_id", return_value=1),
            mock.patch.object(interaction, "_directed_context_fresh", return_value=False),
            mock.patch.object(interaction, "_game_suppresses_conversation", return_value=False),
            mock.patch.object(interaction.speech_queue, "is_speaking", return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch.object(interaction.echo_cancel, "is_suppressed", return_value=False),
            mock.patch.object(interaction.end_thread, "is_grace_active", return_value=True),
            mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak,
        ):
            fired = interaction._maybe_idle_banter(idle_for=interaction.config.IDLE_BANTER_SECS + 5.0, effective_idle_timeout=45.0)
        self.assertFalse(fired)
        speak.assert_not_called()


class SpeakAsyncOnSpokeBookkeepingTest(unittest.TestCase):
    """Consolidation step 3: `on_spoke` is the cooldown/ack hook, and it must fire
    ONLY when a line actually reaches the speech queue — never on a suppressed or
    losing candidate. (Under ENFORCE, `_speak_async`'s RETURN means "submitted for
    arbitration", not "spoke"; bookkeeping that arms cooldowns or marks events
    acknowledged on the return would let a loser wrongly suppress itself.)"""

    def _patches(self, consciousness, *, can_speak):
        from audio import speech_queue  # function-local import in _do_speak
        done = mock.Mock()
        done.wait = mock.Mock(return_value=None)
        return (
            mock.patch.object(consciousness.config, "PROACTIVE_SPEECH_YIELD_ENABLED", False),
            mock.patch.object(consciousness, "_can_proactive_speak", return_value=can_speak),
            mock.patch.object(speech_queue, "enqueue", return_value=done),
            mock.patch.object(consciousness, "note_rex_utterance"),
            mock.patch.object(consciousness.conv_log, "log_rex"),
            mock.patch.object(consciousness.threading, "Thread"),  # skip the _on_done waiter
        )

    def test_on_spoke_fires_when_the_line_actually_speaks(self):
        from intelligence import consciousness

        fired = []
        with ExitStack() as stack:
            for p in self._patches(consciousness, can_speak=True):
                stack.enter_context(p)
            spoke = consciousness._speak_async(
                "Carbon-based smiling detected.",
                "happy",
                governed=False,  # legacy direct path; ENFORCE winner runs the same body
                on_spoke=lambda: fired.append(True),
            )

        self.assertTrue(spoke)
        self.assertEqual(fired, [True])  # bookkeeping ran exactly once, on the speak

    def test_on_spoke_does_not_fire_when_speech_is_suppressed(self):
        from intelligence import consciousness

        fired = []
        with ExitStack() as stack:
            for p in self._patches(consciousness, can_speak=False):
                stack.enter_context(p)
            spoke = consciousness._speak_async(
                "This line should be swallowed.",
                "neutral",
                governed=False,
                on_spoke=lambda: fired.append(True),
            )

        self.assertFalse(spoke)
        self.assertEqual(fired, [])  # a non-spoken line never arms its bookkeeping


class MicroBehaviorEnforceRoutingTest(unittest.TestCase):
    """Consolidation: the consciousness-thread micro-behaviors that used to BYPASS
    the governor (own conversation_agenda claim + a worker thread that spoke via
    `_speak_async(governed=False)`) must, under ENFORCE, instead SUBMIT a candidate
    carrying the deferred work as a `speak_fn` and take NO legacy claim — so the
    governor is the single decider. (visual_curiosity follows the same shape; small
    talk has the cleanest preconditions to drive directly.)"""

    def test_small_talk_submits_candidate_and_skips_claim_under_enforce(self):
        from intelligence import consciousness
        observed = []
        with (
            mock.patch.object(consciousness, "_governor_enforcing", return_value=True),
            mock.patch.object(consciousness, "_can_proactive_speak", return_value=True),
            mock.patch.object(
                consciousness, "_observe_governor_candidate",
                side_effect=lambda **kw: observed.append(kw) or "cg-test",
            ),
            mock.patch.object(consciousness, "_claim_proactive_purpose") as claim,
            mock.patch.object(consciousness.threading, "Thread") as Thread,
        ):
            # Empty room → open-question path (no DB lookups), still routes the same.
            consciousness._do_small_talk_question({"people": [], "time": {}})

            # ENFORCE: took NO legacy conversation_agenda claim...
            claim.assert_not_called()
            # ...and submitted exactly ONE candidate carrying a runnable speak_fn.
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0].get("purpose"), "small_talk")
            speak_fn = observed[0].get("speak_fn")
            self.assertTrue(callable(speak_fn))
            # The deferred speak_fn spawns the worker thread (slow LLM work stays off
            # the tick) — it does NOT speak inline.
            Thread.assert_not_called()
            speak_fn()
            Thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
