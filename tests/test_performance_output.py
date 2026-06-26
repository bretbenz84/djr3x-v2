import unittest
from unittest import mock


class PerformanceOutputTests(unittest.TestCase):
    def test_on_text_fires_with_generated_line_before_speaking(self):
        # Read-along: the line must reach the transcript the moment it's generated,
        # BEFORE the blocking speak — so the GUI shows it instead of waiting for TTS.
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.roast",
            prompt_contract="Roast them.",
            fallback_text="Fallback roast.",
            emotion="curious",
            delivery_style="consent_roast",
            memory_policy=performance_plan.MEMORY_DO_NOT_STORE,
        )
        order = []
        on_text = mock.Mock(side_effect=lambda t: order.append(("log", t)))
        speak = mock.Mock(side_effect=lambda *a, **k: order.append(("speak", a[0])) or True)

        output = performance_output.execute_plan(
            plan,
            generate_text=mock.Mock(return_value="Nice posture, gravity wins again."),
            speak_text=speak,
            clean_text=lambda text: text.strip(),
            on_text=on_text,
        )

        on_text.assert_called_once_with("Nice posture, gravity wins again.")
        self.assertEqual(output.text, "Nice posture, gravity wins again.")
        # log BEFORE speak.
        self.assertEqual(order[0][0], "log")
        self.assertEqual(order[1][0], "speak")

    def test_execute_plan_generates_body_beats_and_speaks(self):
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.tell_joke",
            prompt_contract="Tell one joke.",
            fallback_text="Fallback joke.",
            emotion="happy",
            body_beat="dramatic_visor_peek",
            delivery_style="quick_punchline",
            memory_policy=performance_plan.MEMORY_DO_NOT_STORE,
            pre_beat_ms=10,
            post_beat_ms=20,
        )
        play = mock.Mock()
        speak = mock.Mock(return_value=True)

        output = performance_output.execute_plan(
            plan,
            generate_text=mock.Mock(return_value="Generated joke."),
            speak_text=speak,
            play_body_beat=play,
            clean_text=lambda text: text.strip(),
        )

        self.assertEqual(output.text, "Generated joke.")
        self.assertTrue(output.completed)
        self.assertEqual(output.action, "humor.tell_joke")
        self.assertEqual(output.body_beat, "dramatic_visor_peek")
        play.assert_called_once_with("dramatic_visor_peek")
        speak.assert_called_once_with(
            "Generated joke.",
            emotion="happy",
            pre_beat_ms=10,
            post_beat_ms_override=20,
            log_text=False,
        )

    def test_quick_punchline_splits_setup_and_punchline_with_pause(self):
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.tell_joke",
            prompt_contract="Tell one joke.",
            fallback_text="Fallback joke.",
            emotion="happy",
            delivery_style="quick_punchline",
            post_beat_ms=20,
        )
        speak = mock.Mock(return_value=True)

        with mock.patch.object(
            performance_output.config,
            "JOKE_SETUP_PUNCHLINE_PAUSE_MS",
            650,
        ):
            output = performance_output.execute_plan(
                plan,
                generate_text=mock.Mock(
                    return_value="Why did the droid bring a ladder? Because the drinks were on the house."
                ),
                speak_text=speak,
                clean_text=lambda text: text.strip(),
            )

        self.assertTrue(output.completed)
        self.assertEqual(
            output.text,
            "Why did the droid bring a ladder? Because the drinks were on the house.",
        )
        self.assertEqual(speak.call_count, 2)
        speak.assert_has_calls([
            mock.call(
                "Why did the droid bring a ladder?",
                emotion="happy",
                pre_beat_ms=0,
                post_beat_ms_override=0,
                log_text=False,
            ),
            mock.call(
                "Because the drinks were on the house.",
                emotion="happy",
                pre_beat_ms=650,
                post_beat_ms_override=20,
                log_text=False,
            ),
        ])

    def test_execute_plan_uses_fallback_when_generation_fails(self):
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.free_bit",
            prompt_contract="Be funny.",
            fallback_text="Fallback bit.",
            emotion="happy",
        )

        output = performance_output.execute_plan(
            plan,
            generate_text=mock.Mock(side_effect=RuntimeError("offline")),
            speak_text=mock.Mock(return_value=True),
            clean_text=lambda text: text.strip(),
        )

        self.assertEqual(output.text, "Fallback bit.")
        self.assertTrue(output.generation_failed)

    def test_execute_plan_reports_body_beat_failure_but_still_speaks(self):
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.roast",
            prompt_contract="Roast gently.",
            fallback_text="Fallback roast.",
            emotion="curious",
            body_beat="suspicious_glance",
        )
        speak = mock.Mock(return_value=True)

        output = performance_output.execute_plan(
            plan,
            generate_text=mock.Mock(return_value="Roast line."),
            speak_text=speak,
            play_body_beat=mock.Mock(side_effect=RuntimeError("servo")),
            clean_text=lambda text: text.strip(),
        )

        self.assertEqual(output.text, "Roast line.")
        self.assertTrue(output.body_beat_failed)
        speak.assert_called_once()

    def test_execute_plan_skips_generation_when_plan_does_not_require_llm(self):
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="performance.body_beat",
            fallback_text="Physical expression logged.",
            body_beat="tiny_victory_dance",
            requires_llm=False,
        )
        generate = mock.Mock(return_value="Should not be used.")

        output = performance_output.execute_plan(
            plan,
            generate_text=generate,
            speak_text=mock.Mock(return_value=True),
            play_body_beat=mock.Mock(),
            clean_text=lambda text: text.strip(),
        )

        self.assertEqual(output.text, "Physical expression logged.")
        generate.assert_not_called()

    def test_post_line_landing_defers_beat_to_audio_end(self):
        # A comedic-landing style with a landing player defers the body beat: it does
        # NOT fire upfront, and instead rides the line's on_audio_end so it lands in
        # the post-line silence.
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.roast",
            prompt_contract="Roast.",
            fallback_text="Fallback.",
            emotion="curious",
            body_beat="suspicious_glance",
            delivery_style="consent_roast",
        )
        upfront = mock.Mock()
        landing = mock.Mock()
        captured = {}

        def speak(text, **kwargs):
            captured.update(kwargs)
            return True

        with mock.patch.object(
            performance_output.config, "PERFORMANCE_POST_LINE_BEAT_ENABLED", True
        ):
            output = performance_output.execute_plan(
                plan,
                generate_text=mock.Mock(return_value="Roast line."),
                speak_text=speak,
                play_body_beat=upfront,
                play_landing_body_beat=landing,
                clean_text=lambda t: t.strip(),
            )

        self.assertTrue(output.completed)
        upfront.assert_not_called()                       # NOT fired upfront
        self.assertIn("on_audio_end", captured)           # routed to the audio-end hook
        landing.assert_not_called()                       # not yet — only when audio ends
        captured["on_audio_end"]()                        # simulate audio ending
        landing.assert_called_once_with("suspicious_glance")

    def test_post_line_landing_attaches_to_punchline_not_setup(self):
        # For a split joke the button must land after the PUNCHLINE only, never the setup.
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.tell_joke",
            prompt_contract="Joke.",
            fallback_text="Fallback.",
            emotion="happy",
            body_beat="dramatic_visor_peek",
            delivery_style="quick_punchline",
        )
        landing = mock.Mock()
        calls = []

        def speak(text, **kwargs):
            calls.append((text, kwargs))
            return True

        with mock.patch.object(
            performance_output.config, "PERFORMANCE_POST_LINE_BEAT_ENABLED", True
        ):
            performance_output.execute_plan(
                plan,
                generate_text=mock.Mock(
                    return_value="Why did the droid cross the room? To reach the cantina."
                ),
                speak_text=speak,
                play_body_beat=mock.Mock(),
                play_landing_body_beat=landing,
                clean_text=lambda t: t.strip(),
            )

        self.assertEqual(len(calls), 2)
        _, setup_kwargs = calls[0]
        _, punch_kwargs = calls[1]
        self.assertNotIn("on_audio_end", setup_kwargs)    # setup: no button
        self.assertIn("on_audio_end", punch_kwargs)       # punchline: carries the button
        punch_kwargs["on_audio_end"]()
        landing.assert_called_once_with("dramatic_visor_peek")

    def test_post_line_landing_disabled_fires_upfront(self):
        # Kill switch off -> behaves exactly like before (beat upfront, no on_audio_end).
        from intelligence import performance_output, performance_plan

        plan = performance_plan.PerformancePlan(
            action="humor.roast",
            prompt_contract="Roast.",
            fallback_text="Fallback.",
            emotion="curious",
            body_beat="suspicious_glance",
            delivery_style="consent_roast",
        )
        upfront = mock.Mock()
        captured = {}

        def speak(text, **kwargs):
            captured.update(kwargs)
            return True

        with mock.patch.object(
            performance_output.config, "PERFORMANCE_POST_LINE_BEAT_ENABLED", False
        ):
            performance_output.execute_plan(
                plan,
                generate_text=mock.Mock(return_value="Roast line."),
                speak_text=speak,
                play_body_beat=upfront,
                play_landing_body_beat=mock.Mock(),
                clean_text=lambda t: t.strip(),
            )

        upfront.assert_called_once_with("suspicious_glance")
        self.assertNotIn("on_audio_end", captured)

    def test_execute_body_beat_event_plays_mapped_event(self):
        from intelligence import performance_output

        play = mock.Mock()

        beat = performance_output.execute_body_beat_event(
            "insult.detected",
            play_body_beat=play,
        )

        self.assertEqual(beat, "anger_flash")
        play.assert_called_once_with("anger_flash")

    def test_execute_body_beat_event_returns_none_for_unknown_or_failed_beat(self):
        from intelligence import performance_output

        self.assertIsNone(
            performance_output.execute_body_beat_event(
                "unknown.event",
                play_body_beat=mock.Mock(),
            )
        )
        self.assertIsNone(
            performance_output.execute_body_beat_event(
                "game.correct",
                play_body_beat=mock.Mock(side_effect=RuntimeError("servo")),
            )
        )


if __name__ == "__main__":
    unittest.main()
