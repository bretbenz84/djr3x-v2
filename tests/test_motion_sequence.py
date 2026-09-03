import time
import unittest
from unittest import mock

from intelligence.action_router import ActionDecision
from intelligence import motion_sequence


def _decision(action, **args):
    return ActionDecision(action=action, confidence=1.0, args=args, reason="test")


class MotionSequenceTest(unittest.TestCase):
    def tearDown(self):
        motion_sequence.cancel("test cleanup", stop_base=False)
        self._wait_finished()

    def _wait_finished(self):
        deadline = time.monotonic() + 1.0
        while motion_sequence.active() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(motion_sequence.active())

    @mock.patch.object(motion_sequence, "_wait_until_settled", return_value=True)
    @mock.patch.object(motion_sequence.motion, "wait_done")
    @mock.patch.object(motion_sequence.motion_controller, "stop")
    @mock.patch.object(motion_sequence.motion_controller, "move_forward")
    @mock.patch.object(motion_sequence.motion_controller, "turn_left")
    @mock.patch.object(motion_sequence.motion_controller, "available", return_value=True)
    def test_steps_issue_in_order_after_completion(
        self, _available, turn_left, move_forward, _stop, wait_done, _settled
    ):
        events = []
        turn_left.side_effect = lambda deg: events.append(("turn", deg)) or 11
        move_forward.side_effect = lambda dist: events.append(("move", dist)) or 12
        wait_done.side_effect = lambda seq, timeout: events.append(("done", seq)) or {
            "result": "completed"
        }

        started = motion_sequence.start([
            _decision("motion.turn", direction="left", deg=90.0),
            _decision("motion.move", direction="forward", dist_m=1.524),
        ])
        self.assertTrue(started)
        self._wait_finished()
        self.assertEqual(events, [
            ("turn", 90.0), ("done", 11), ("move", 1.524), ("done", 12)
        ])

    @mock.patch.object(motion_sequence, "_wait_until_settled", return_value=True)
    @mock.patch.object(motion_sequence.motion, "wait_done", return_value={"result": "blocked"})
    @mock.patch.object(motion_sequence.motion_controller, "stop")
    @mock.patch.object(motion_sequence.motion_controller, "move_forward")
    @mock.patch.object(motion_sequence.motion_controller, "turn_left", return_value=21)
    @mock.patch.object(motion_sequence.motion_controller, "available", return_value=True)
    def test_blocked_step_aborts_remainder(
        self, _available, _turn_left, move_forward, stop, _wait_done, _settled
    ):
        self.assertTrue(motion_sequence.start([
            _decision("motion.turn", direction="left", deg=90.0),
            _decision("motion.move", direction="forward", dist_m=1.0),
        ]))
        self._wait_finished()
        move_forward.assert_not_called()
        self.assertGreaterEqual(stop.call_count, 1)

    @mock.patch.object(motion_sequence, "_wait_until_settled", return_value=True)
    @mock.patch.object(motion_sequence.motion, "wait_done")
    @mock.patch.object(motion_sequence.motion_controller, "stop")
    @mock.patch.object(motion_sequence.motion_controller, "move_forward", return_value=12)
    @mock.patch.object(motion_sequence.motion_controller, "turn_right", return_value=None)
    @mock.patch.object(motion_sequence.motion_controller, "available", return_value=True)
    def test_refused_first_step_means_no_sequence(
        self, _available, turn_right, move_forward, _stop, wait_done, _settled
    ):
        # Field 2026-09-02 23:04:34: "Turn slight right, then go forward one foot"
        # — the swing check refused the turn in the background thread while the
        # caller, told a thread had started, said "On it — 2 moves". The first
        # step is issued on the caller's thread now: refused = not started.
        started = motion_sequence.start([
            _decision("motion.turn", direction="right", deg=15.0),
            _decision("motion.move", direction="forward", dist_m=0.3),
        ])
        self.assertFalse(started)
        self.assertFalse(motion_sequence.active())
        turn_right.assert_called_once()
        move_forward.assert_not_called()
        wait_done.assert_not_called()

    @mock.patch.object(motion_sequence, "_wait_until_settled", return_value=True)
    @mock.patch.object(motion_sequence.motion, "wait_done", return_value={"result": "completed"})
    @mock.patch.object(motion_sequence.motion_controller, "stop")
    @mock.patch.object(motion_sequence.motion_controller, "move_forward", return_value=12)
    @mock.patch.object(motion_sequence.motion_controller, "turn_right", return_value=11)
    @mock.patch.object(motion_sequence.motion_controller, "available", return_value=True)
    def test_first_step_issued_once_when_it_goes(
        self, _available, turn_right, move_forward, _stop, _wait_done, _settled
    ):
        self.assertTrue(motion_sequence.start([
            _decision("motion.turn", direction="right", deg=15.0),
            _decision("motion.move", direction="forward", dist_m=0.3),
        ]))
        self._wait_finished()
        turn_right.assert_called_once()
        move_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
