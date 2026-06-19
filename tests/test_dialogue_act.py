import json
from pathlib import Path
import unittest


class DialogueActReplayTests(unittest.TestCase):
    def setUp(self):
        from intelligence import dialogue_act

        dialogue_act.clear()

    def tearDown(self):
        from intelligence import dialogue_act

        dialogue_act.clear()

    def test_misroute_replay_corpus(self):
        from intelligence import dialogue_act

        corpus_path = Path(__file__).parent / "fixtures" / "misroute_replays.json"
        cases = json.loads(corpus_path.read_text())

        for case in cases:
            with self.subTest(case=case["name"]):
                dialogue_act.clear()
                frame = case["frame"]
                dialogue_act.note_rex_turn(
                    frame["text"],
                    source=frame.get("source"),
                    topic=frame.get("topic"),
                    target_person_id=frame.get("target_person_id"),
                )
                decision = dialogue_act.classify(
                    case["utterance"],
                    {"pending": {}, "active_game": False},
                    person_id=case.get("person_id"),
                )

                self.assertEqual(decision.label, case["expected_label"])
                self.assertEqual(
                    decision.skip_action_router,
                    case["expected_skip_action_router"],
                )
                for action in case.get("blocked_actions", []):
                    self.assertIn(action, decision.blocked_actions)

    def test_active_frame_context_is_person_scoped(self):
        from intelligence import dialogue_act

        dialogue_act.note_rex_turn(
            "Bret, how did the trip go?",
            source="memory_followup",
            target_person_id=1,
        )

        self.assertIsNotNone(dialogue_act.active_frame_context(1))
        self.assertIsNone(dialogue_act.active_frame_context(2))

    def test_explicit_music_request_breaks_reply_frame(self):
        from intelligence import dialogue_act

        dialogue_act.note_rex_turn(
            "You realize this is the point where you drop some beats, right?",
            source="assistant_turn",
            target_person_id=2,
            expected_reply_types=["yes_no"],
        )

        decision = dialogue_act.classify(
            "drop some sick beats. Play some country music.",
            {"pending": {}, "active_game": False},
            person_id=2,
        )

        self.assertEqual(decision.label, "new_command")
        self.assertFalse(decision.skip_action_router)

    def test_counter_question_to_rex_keeps_answer_frame(self):
        # Rex asked something and is waiting on a reply; the user counters with their
        # OWN short question. It must stay an in-frame reply (no router hijack), not be
        # promoted to a new command.
        from intelligence import dialogue_act

        dialogue_act.note_rex_turn(
            "What got you into woodworking?",
            source="assistant_turn",
            target_person_id=3,
            expected_reply_types=["answer", "dismissal"],
        )
        decision = dialogue_act.classify(
            "Why do you ask?",
            {"pending": {}, "active_game": False},
            person_id=3,
        )
        self.assertEqual(decision.label, "answer_to_rex")
        self.assertTrue(decision.skip_action_router)

    def test_explicit_command_question_still_routes_mid_frame(self):
        # A real command phrased as a question ("what time is it?") must still break the
        # frame and route, even while Rex is awaiting a reply.
        from intelligence import dialogue_act

        dialogue_act.note_rex_turn(
            "How's the project going?",
            source="assistant_turn",
            target_person_id=3,
            expected_reply_types=["answer"],
        )
        decision = dialogue_act.classify(
            "What time is it?",
            {"pending": {}, "active_game": False},
            person_id=3,
        )
        self.assertEqual(decision.label, "new_command")
        self.assertFalse(decision.skip_action_router)

    def test_counter_question_with_no_open_frame_is_command(self):
        # No active Rex question → a wh-question is a fresh command (router runs).
        from intelligence import dialogue_act

        decision = dialogue_act.classify(
            "What can you do?",
            {"pending": {}, "active_game": False},
            person_id=3,
        )
        self.assertEqual(decision.label, "new_command")

    def test_direct_sleep_command_breaks_reply_frame(self):
        from intelligence import dialogue_act

        dialogue_act.note_rex_turn(
            "Feeling a bit less than stellar, huh?",
            source="assistant_turn",
            target_person_id=1,
            expected_reply_types=["short_answer"],
        )

        decision = dialogue_act.classify(
            "go to sleep",
            {"pending": {}, "active_game": False},
            person_id=1,
        )

        self.assertEqual(decision.label, "new_command")
        self.assertFalse(decision.skip_action_router)


if __name__ == "__main__":
    unittest.main()
