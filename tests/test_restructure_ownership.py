"""Offline contracts for the remaining ownership migrations."""
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest import mock
import numpy as np

from intelligence import attribution as A, dialogue_act as D, turn_coordinator as T
from utils import turn_trace
from utils.model_usage import RequestUsage, UsageStream


class PendingExchangeTests(unittest.TestCase):
    def setUp(self):
        D.clear()
        self.addCleanup(D.clear)

    def test_only_target_can_settle_and_later_questions_stay_open(self):
        a = D.note_rex_turn("How was your trip?", target_person_id=1)
        b = D.note_rex_turn("How was work?", target_person_id=2)
        self.assertFalse(D.answer_frame(a, 2, trusted=True))
        self.assertFalse(D.answer_frame(a, 1, trusted=False))
        self.assertTrue(D.answer_frame(a, 1, trusted=True))
        self.assertFalse(a.active())
        self.assertTrue(b.active())

    def test_old_session_and_future_question_cannot_be_settled(self):
        frame = D.note_rex_turn("Ready?", target_person_id=1)
        with D.captured_at(frame.created_at - 1):
            self.assertFalse(D.answer_frame(frame, 1, trusted=True))
        D.clear()
        self.assertFalse(D.answer_frame(frame, 1, trusted=True))

    def test_earlier_target_can_answer_after_rex_addresses_another_person(self):
        first = D.note_rex_turn("How was your trip?", target_person_id=1)
        D.note_rex_turn("How was work?", target_person_id=2)
        decision = D.classify("It was excellent", person_id=1)
        self.assertIs(decision.frame, first)
        self.assertEqual(decision.label, "answer_to_rex")


class AttributionTests(unittest.TestCase):
    def test_split_only_at_silence_with_consistent_speaker_evidence(self):
        rows = [{"monotonic_at": at, "person_db_id": pid}
                for at, pid in ((1.1, 1), (1.5, 1), (2.6, 2), (2.9, 2), (4.1, 1), (4.3, 1))]
        self.assertEqual(A.sequential_boundaries([(1, 2), (2.5, 3), (4, 5)], rows), [2.25, 3.5])
        self.assertEqual(A.sequential_boundaries([(1, 5)], rows), [])

    def test_context_cannot_certify_identity(self):
        for tier in ("roster", "sticky", "known_floor"):
            ev = A.UtteranceEvidence(final_person_id=1, final_name="A", accept_tier=tier,
                                     raw_best_id=1, raw_best_score=.46, words=20, voiced_secs=5)
            self.assertIsNone(A.resolve_authoritative(ev).person_id)

    def test_strong_voice_beats_context_proposal(self):
        ev = A.UtteranceEvidence(final_person_id=2, raw_best_id=1, raw_best_name="A",
                                 raw_best_score=.9, margin=.2, required_margin=.1)
        self.assertEqual(A.resolve_authoritative(ev).person_id, 1)

    def test_aba_switches_and_overlap_abstain_without_learning_identity(self):
        for rows in ([1, 2, 1], [1, 2, 3]):
            ev = A.UtteranceEvidence(raw_best_id=1, raw_best_score=.9, margin=.2,
                visual_observations=[{"person_db_id": pid} for pid in rows])
            result = A.resolve_authoritative(ev)
            self.assertEqual(result.status, "ambiguous")
            self.assertIsNone(result.person_id)

    def test_known_off_camera_voice_and_unknown_short_interjection(self):
        ev = A.UtteranceEvidence(raw_best_id=1, raw_best_name="A", raw_best_score=.9,
                                 margin=.2, off_camera_unknown=True)
        self.assertEqual(A.resolve_authoritative(ev).person_id, 1)
        self.assertEqual(A.resolve_authoritative(A.UtteranceEvidence(words=1)).status, "unknown")


class ConcurrentCaptureTests(unittest.TestCase):
    def test_input_arrives_while_reply_is_unfinished_and_close_stops_producer(self):
        received = threading.Event()
        queue = T.PendingTurns()
        at = time.monotonic()
        turn = T.CapturedTurn(np.ones(10), at, at, 1)
        def scan(cursor):
            received.set()
            return ([turn], at + 1) if cursor == at else ([], cursor)
        with T.CaptureDuringReply(scan, at, queue, interval=.001) as producer:
            self.assertTrue(received.wait(1))
        self.assertFalse(producer.thread.is_alive())
        self.assertIs(queue.pop(1), turn)
        self.assertIsNone(queue.pop(1))

    def test_failed_scan_preserves_recovery_cursor(self):
        failed = threading.Event()
        def scan(cursor):
            failed.set()
            raise ValueError("fixture")
        with T.CaptureDuringReply(scan, 12, interval=.001) as producer:
            self.assertTrue(failed.wait(1))
        self.assertEqual(producer.cursor, 12)
        self.assertIsInstance(producer.error, ValueError)


class UsageTests(unittest.TestCase):
    def setUp(self):
        turn_trace.reset_for_tests()
        self.addCleanup(turn_trace.reset_for_tests)

    def test_final_usage_stays_with_dispatch_owner(self):
        first, token = turn_trace.begin(1)
        request = RequestUsage("fixture", "reply")
        turn_trace.end(token)
        second, token = turn_trace.begin(2)
        source = mock.Mock()
        source.__iter__ = mock.Mock(return_value=iter([
            NS(usage=None), NS(usage=NS(prompt_tokens=12, completion_tokens=3, total_tokens=15))]))
        list(UsageStream(source, request))
        self.assertEqual(first.calls["usage.reply.total_tokens"], 15)
        self.assertNotIn("usage.reply.total_tokens", second.calls)
        source.close.assert_called_once()
        turn_trace.end(token)

    def test_cancel_is_unknown_usage_not_zero_and_close_is_idempotent(self):
        request = RequestUsage("fixture", "reply")
        stream = UsageStream(iter([NS(usage=None)]), request)
        next(stream)
        stream.close()
        stream.close()
        totals = turn_trace.totals()
        self.assertEqual(totals["usage.reply.cancelled"], 1)
        self.assertEqual(totals["usage.reply.usage_unknown"], 1)
        self.assertNotIn("usage.reply.total_tokens", totals)


class MinimalPreparationTests(unittest.TestCase):
    def test_ordinary_lean_reply_does_not_build_classic_agenda(self):
        from intelligence import conversation_agenda as agenda, end_thread
        with mock.patch.object(agenda, "build_turn_plan", side_effect=AssertionError("classic agenda")), \
                mock.patch.object(end_thread, "pending_closure", return_value=None), \
                mock.patch.object(end_thread, "consume_invitation_acceptance", return_value=False):
            plan = agenda.build_lean_turn_plan("I am going camping tomorrow", 1)
        self.assertIsNotNone(plan.hard_no_question)
        self.assertLess(len(plan.directive), 300)


class ActionOwnershipTests(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_state as cs
        self.cs = cs
        cs.clear()
        self.addCleanup(cs.clear)

    def test_stop_then_late_success_does_not_revive_goal(self):
        rec = self.cs.note_action_issued(21, "turn", requested_deg=90, attempted_deg=90)
        self.cs.invalidate_running_actions("stop")
        self.cs.note_action_result(21, "completed")
        self.cs.note_action_verified(21, requested_deg=90, measured_deg=90)
        self.assertEqual(rec.status, "aborted")
        self.assertIsNone(rec.measured_deg)

    def test_request_correlates_refusal_and_none_never_means_running(self):
        from intelligence.action_result import narration_owner
        with narration_owner() as request:
            rec = self.cs.note_action_refused("turn", "swing_blocked")
            self.assertEqual(rec.request_id, request["request_id"])
            self.assertIs(request["result"], rec)
        failed = self.cs.note_action_issued(None, "turn")
        self.assertEqual(failed.status, "error")


class InputProductionAdapterTests(unittest.TestCase):
    def test_completed_capture_emitted_while_newer_speech_remains_in_progress(self):
        from intelligence import interaction as I
        audio = np.ones(160000, dtype=np.float32)
        with mock.patch.object(I.time, "monotonic", return_value=20), \
                mock.patch.object(I, "_gap_span_audio", return_value=(audio, 10)), \
                mock.patch.object(I, "_gap_voiced_runs", return_value=[(11, 12), (19, 20)]), \
                mock.patch.object(I, "_gap_first_audio_at", 0), \
                mock.patch.object(I._stop_event, "is_set", return_value=False), \
                mock.patch.object(I.echo_cancel, "last_playback_ended_at", return_value=0), \
                mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False), \
                mock.patch.object(I.speech_queue, "is_speaking", return_value=False), \
                mock.patch.object(I.output_gate, "is_busy", return_value=False), \
                mock.patch.object(I.hardware_aec, "is_active", return_value=False):
            turns, cursor = I._scan_reply_input(10)
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(cursor, 12.25)
        self.assertLess(turns[0].ended_at, 19)


class ReactiveGenerationTests(unittest.TestCase):
    def test_cancelled_generation_closes_stream_without_speaking_late_question(self):
        from intelligence import interaction as I
        closed = []
        generation = [1]
        def tokens(*args):
            try:
                generation[0] = 2
                yield "What time are you leaving?"
            finally:
                closed.append(True)
        with mock.patch.object(I.speech_queue, "generation", side_effect=lambda: generation[0]), \
                mock.patch.object(I, "_reply_token_stream", side_effect=tokens), \
                mock.patch.object(I._interrupted, "is_set", return_value=False), \
                mock.patch.object(I.empathy, "get_delivery_overrides", return_value=None), \
                mock.patch.object(I.comedy_modes, "voice_settings_for_mode", return_value=None), \
                mock.patch.object(I.speech_queue, "enqueue") as enqueue, \
                mock.patch.object(I, "_speak_blocking") as speak, \
                mock.patch.object(I, "_apply_post_tts_handoff") as handoff:
            text = I._stream_and_speak_sentences("hello", 1, NS(purpose="general"),
                NS(key="straight"), "", {}, None, None, threading.Event())
        self.assertEqual(text, "")
        self.assertEqual(closed, [True])
        enqueue.assert_not_called()
        speak.assert_not_called()
        handoff.assert_not_called()


if __name__ == "__main__":
    unittest.main()
