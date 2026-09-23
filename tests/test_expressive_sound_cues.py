"""Exercise event -> queued accent selection without speakers or servo I/O."""
import unittest
from unittest import mock

from audio import sound_effects as sfx, speech_queue as sq
from intelligence import end_thread, performance_output, performance_plan
from tests.test_speech_generations import _bare_queue


class AccentSelectionTest(unittest.TestCase):
    def test_comedy_stances_have_distinct_accents(self):
        cases = {
            "friendly_roast": "sarcastic", "smug_superiority": "sarcastic",
            "appliance_conspiracy": "mischievous", "self_own": "embarrassed",
            "fake_system_error": "error",
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode), mock.patch.object(sfx, "play") as play:
                sfx.play_for_speech("happy", comedy_mode=mode)
                play.assert_called_once_with(expected, concurrent=True)

    def test_specific_reaction_wins_over_comedy_and_laughter(self):
        with mock.patch.object(sfx, "play") as play:
            sfx.play_for_speech("happy", effect="angry", comedy_mode="self_own",
                                text="[laughs] Some line.")
        play.assert_called_once_with("angry", concurrent=True)

    def test_laugh_delivery_uses_laugh_pool_but_happiness_stays_happy(self):
        with mock.patch.object(sfx, "play") as play:
            sfx.play_for_speech("happy", text="[laughs] You got me.")
            sfx.play_for_speech("happy", text="That's nice.")
        self.assertEqual(play.call_args_list, [mock.call("laughing", concurrent=True),
                                              mock.call("happy", concurrent=True)])
        self.assertEqual(len(sfx._stems_for("laughing")), 2)

    def test_serious_delivery_does_not_pick_comedy(self):
        with mock.patch.object(sfx, "play") as play:
            sfx.play_for_speech("sad", comedy_mode="self_own", text="[laughs] stale tag")
        play.assert_called_once_with("sad", concurrent=True)

    def test_silent_setup_and_impersonations_remain_silent(self):
        with mock.patch.object(sfx, "play") as play:
            sfx.play_for_speech("happy", effect="none", comedy_mode="self_own")
            sfx.play_for_speech("happy", effect="laughing", tag="impersonation")
        play.assert_not_called()


class PlannedAccentTest(unittest.TestCase):
    def test_joke_reserves_laughter_for_punchline(self):
        plan = performance_plan.plan_for_action("humor.tell_joke")
        speak = mock.Mock(return_value=True)
        performance_output.execute_plan(
            plan, generate_text=lambda _: "Why the ladder? The drinks were on the house.",
            speak_text=speak,
        )
        self.assertEqual([c.kwargs["sound_effect"] for c in speak.call_args_list],
                         ["none", "laughing"])

    def test_interrupted_setup_does_not_deliver_laugh(self):
        speak = mock.Mock(return_value=False)
        performance_output.execute_plan(
            performance_plan.plan_for_action("humor.tell_joke"),
            generate_text=lambda _: "Why the ladder? The drinks were on the house.",
            speak_text=speak,
        )
        self.assertEqual(speak.call_count, 1)
        self.assertEqual(speak.call_args.kwargs["sound_effect"], "none")

    def test_real_generation_failure_gets_error_instead_of_laughter(self):
        speak = mock.Mock(return_value=True)
        result = performance_output.execute_plan(
            performance_plan.plan_for_action("humor.tell_joke"),
            generate_text=mock.Mock(side_effect=RuntimeError("offline")), speak_text=speak,
        )
        self.assertTrue(result.generation_failed)
        self.assertEqual(speak.call_args.kwargs["sound_effect"], "error")

    def test_explicit_mood_performances_keep_their_specific_accent(self):
        for mood in ("confused", "embarrassed"):
            speak = mock.Mock(return_value=True)
            performance_output.execute_plan(
                performance_plan.plan_for_action("performance.mood_pose", args={"mood": mood}),
                generate_text=mock.Mock(), speak_text=speak,
            )
            self.assertEqual(speak.call_args.kwargs["sound_effect"], mood)


class QueuedAccentTest(unittest.TestCase):
    def test_event_accent_survives_enqueue_and_is_selected_only_at_playback(self):
        q = _bare_queue()
        with (mock.patch.object(sq, "_state_suppresses_output", return_value=False),
              mock.patch.object(sq, "_audio_output_suppressed", return_value=False),
              mock.patch("audio.tts.speak"), mock.patch.object(sfx, "play") as play):
            q.enqueue("Respect the droid.", "neutral", sound_effect="angry",
                      comedy_mode="self_own")
            play.assert_not_called()
            q._process_item(q._heap.pop())
        play.assert_called_once_with("angry", concurrent=True)

    def test_stale_reply_drops_its_accent_too(self):
        q = _bare_queue()
        item = sq._Item(1, 1, "Bye!", "happy", None, sq.DoneEvent(),
                        generation=sq.generation() - 1, sound_effect="goodbye")
        with (mock.patch.object(sq, "_state_suppresses_output", return_value=False),
              mock.patch.object(sfx, "play") as play):
            q._process_item(item)
        play.assert_not_called()
        self.assertEqual(item.done.dropped_reason, "stale_generation")

    def test_continuation_does_not_repeat_comedy_accent(self):
        item = sq._Item(1, 1, "The rest of the line.", "neutral", None,
                        sq.DoneEvent(), comedy_mode="self_own", suppress_audio_tag=True)
        with (mock.patch.object(sq, "_state_suppresses_output", return_value=False),
              mock.patch("audio.tts.speak"), mock.patch.object(sfx, "play") as play):
            _bare_queue()._process_item(item)
        play.assert_not_called()


class TurnCueTest(unittest.TestCase):
    def setUp(self):
        end_thread.clear()
        self.addCleanup(end_thread.clear)

    def test_goodbye_is_scoped_to_the_actual_closing_turn(self):
        from intelligence import interaction as interaction
        end_thread.note_user_turn("Bye Rex!")
        self.assertEqual(interaction._reply_sound_effect("Bye Rex!"), "goodbye")
        self.assertIsNone(interaction._reply_sound_effect("What is your favorite song?"))
        end_thread.mark_closure_spoken()
        self.assertIsNone(interaction._reply_sound_effect("Bye Rex!"))

    def test_topic_closure_after_goodbye_is_not_another_goodbye(self):
        end_thread.note_user_turn("Bye Rex!")
        end_thread.note_user_turn("Never mind.")
        self.assertFalse(end_thread.pending_farewell("Never mind."))

    def test_direct_affection_not_quoted_or_negated_affection(self):
        from intelligence import interaction as interaction
        for text in ("I love you, Rex!", "Rex, I really appreciate you.", "You're my best friend."):
            self.assertEqual(interaction._reply_sound_effect(text), "warm")
        for text in ("I don't love you.", "She said I love you.", "I love your music."):
            self.assertIsNone(interaction._reply_sound_effect(text))

    def test_reserved_speech_accent_does_not_spend_mood_cooldown(self):
        from intelligence import body_mood
        with (mock.patch.object(body_mood, "_state", dict(body_mood._state, mood="neutral")),
              mock.patch.object(body_mood, "_mood_chirp") as chirp):
            self.assertTrue(body_mood.set_mood("proud", chirp=False))
        chirp.assert_not_called()

    def test_subtle_insult_can_use_offended_mood_accent(self):
        from intelligence import body_mood
        with (mock.patch.object(body_mood, "_state", dict(body_mood._state, mood="neutral")),
              mock.patch.object(sfx, "play") as play):
            body_mood.set_mood("offended", source="layer2_insult")
        play.assert_called_once_with("angry")

    def test_repair_cues_distinguish_misunderstanding_from_own_mistake(self):
        from intelligence import interaction as interaction
        for kind, expected in (("misheard", "confused"), ("factual", "embarrassed")):
            with (mock.patch.object(interaction, "_speak_blocking") as speak,
                  mock.patch.object(interaction, "_play_event_body_beat"),
                  mock.patch.object(interaction.llm, "get_response", return_value="My mistake."),
                  mock.patch.object(interaction.repair_moves, "mark_handled"),
                  mock.patch.object(interaction.repair_moves, "should_use_better_luck_line", return_value=False)):
                interaction._generate_repair_response(
                    None, "No, I said blue.", {"kind": kind, "correction": "blue, not red"})
            self.assertEqual(speak.call_args.kwargs["sound_effect"], expected)


if __name__ == "__main__":
    unittest.main()
