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


if __name__ == "__main__":
    unittest.main()
