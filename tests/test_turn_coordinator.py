import unittest
from unittest import mock
import numpy as np
from intelligence.turn_coordinator import CapturedTurn, PendingTurns

class PendingTurnsTest(unittest.TestCase):
    def test_finish_first_policy_preserves_input_order(self):
        queue = PendingTurns(capacity=2)
        a = CapturedTurn(np.zeros(1), 10, 11, 1)
        b = CapturedTurn(np.zeros(1), 12, 13, 1)
        self.assertTrue(queue.put(a))
        self.assertTrue(queue.put(b))
        self.assertFalse(queue.put(a))
        with mock.patch('intelligence.turn_coordinator.time.monotonic', return_value=14):
            self.assertIs(queue.pop(1), a)
            self.assertIs(queue.pop(1), b)
            self.assertIsNone(queue.pop(1))

    def test_session_reset_and_expiry_discard_old_input(self):
        queue = PendingTurns(max_age=5)
        queue.put(CapturedTurn(np.zeros(1), 10, 11, 1))
        queue.put(CapturedTurn(np.zeros(1), 10, 11, 2))
        with mock.patch('intelligence.turn_coordinator.time.monotonic', return_value=20):
            self.assertIsNone(queue.pop(2))

class QueuedQuestionTest(unittest.TestCase):
    def test_queued_words_cannot_answer_a_later_question(self):
        from intelligence import dialogue_act
        dialogue_act.clear()
        self.addCleanup(dialogue_act.clear)
        with mock.patch.object(dialogue_act.time, 'monotonic', return_value=20):
            dialogue_act.note_rex_turn('Which room is this?', target_person_id=1)
            with dialogue_act.captured_at(10):
                self.assertIsNone(dialogue_act.active_frame(person_id=1))
                self.assertIn('captured earlier', dialogue_act.queued_turn_note())
            self.assertIsNotNone(dialogue_act.active_frame(person_id=1))
            self.assertEqual(dialogue_act.queued_turn_note(), '')
