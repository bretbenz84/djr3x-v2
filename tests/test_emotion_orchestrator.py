import unittest


class EmotionOrchestratorTests(unittest.TestCase):
    def test_startling_animal_creates_surprise_frame(self):
        from intelligence import emotion_orchestrator

        frame = emotion_orchestrator.frame_for_event(
            "animal_detected",
            species="garden spider",
        )

        self.assertEqual(frame.affect, "surprised")
        self.assertEqual(frame.body_beat, "surprise_pop")
        self.assertEqual(frame.led_style, "excited")
        self.assertEqual(frame.motion_style, "startled")
        self.assertGreaterEqual(frame.speech_motion["visor_open_floor_frac"], 0.9)

    def test_empathy_mode_uses_shared_motion_and_word_style(self):
        from intelligence import emotion_orchestrator

        frame = emotion_orchestrator.frame_for_empathy_mode("listen")

        self.assertEqual(frame.affect, "sad")
        self.assertEqual(frame.word_style, "gentle")
        self.assertEqual(frame.motion_style, "subdued")
        self.assertGreater(frame.speech_motion["interval_scale"], 1.0)

    def test_publish_frame_mirrors_led_safe_emotion_and_rich_frame(self):
        from intelligence import emotion_orchestrator
        from world_state import world_state

        old_self_state = world_state.get("self_state")
        try:
            frame = emotion_orchestrator.frame_for_emotion("surprised")
            emotion_orchestrator.publish_frame(frame, ttl_secs=1.0)

            self_state = world_state.get("self_state")
            self.assertEqual(self_state["emotion"], "excited")
            self.assertEqual(self_state["body_state"], "startled")
            self.assertEqual(self_state["emotion_frame"]["affect"], "surprised")
            self.assertEqual(
                emotion_orchestrator.current_frame().affect,
                "surprised",
            )
        finally:
            world_state.update("self_state", old_self_state)


if __name__ == "__main__":
    unittest.main()
