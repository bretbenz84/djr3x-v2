import unittest
from unittest import mock


class PerformanceOutputTests(unittest.TestCase):
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
        )

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


if __name__ == "__main__":
    unittest.main()
