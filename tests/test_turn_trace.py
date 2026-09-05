"""Lean Brain phase 0 — per-turn stage stamps and model-call counts.

Covers utils.turn_trace itself (contextvar owner, background-thread fallback to
the active turn, first-stamp-wins, snapshot offsets), the counting chokepoints
(connectivity.guard_client wraps chat + responses on any client shape;
local_llm.generate and semantic._request_embedding count without a server),
the transcript turn ids, the speech-queue synthesis-start hook, and the
[character_loop] payload carrying the new blocks. No hardware, no network:
every remote call is mocked at the transport.
"""

from __future__ import annotations

import heapq
import threading
import time
import unittest
from unittest import mock

from utils import turn_trace


class TurnTraceCoreTest(unittest.TestCase):
    def setUp(self):
        turn_trace.reset_for_tests()
        self.addCleanup(turn_trace.reset_for_tests)

    def test_no_turn_means_noops(self):
        self.assertIsNone(turn_trace.current())
        self.assertFalse(turn_trace.stamp("x"))
        self.assertFalse(turn_trace.cancel("y"))
        turn_trace.count("hosted.llm")          # totals still accrue
        self.assertEqual(turn_trace.totals(), {"hosted.llm": 1})

    def test_begin_end_owns_current(self):
        tt, token = turn_trace.begin(turn_id=7)
        self.assertIs(turn_trace.current(), tt)
        self.assertEqual(tt.turn_id, 7)
        turn_trace.end(token)
        self.assertIsNone(turn_trace.current())

    def test_first_stamp_wins_unless_overwrite(self):
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        self.assertTrue(tt.stamp("model_first_token", 10.0))
        self.assertFalse(tt.stamp("model_first_token", 11.0))
        self.assertEqual(tt.stamps["model_first_token"], 10.0)
        self.assertTrue(tt.stamp("model_first_token", 12.0, overwrite=True))
        self.assertEqual(tt.stamps["model_first_token"], 12.0)

    def test_snapshot_offsets_from_origin_in_ms(self):
        tt, token = turn_trace.begin(started_at=100.0)
        self.addCleanup(turn_trace.end, token)
        tt.stamp("asr_start", 100.25)
        tt.stamp("asr_done", 100.9)
        tt.count("embed", 3)
        tt.set_value("context_chars", 4200)
        snap = tt.snapshot()
        self.assertEqual(snap["stages"], {"asr_start": 250, "asr_done": 900})
        self.assertEqual(snap["calls"], {"embed": 3})
        self.assertEqual(snap["values"], {"context_chars": 4200})
        self.assertIsNone(snap["cancel_reason"])
        # A different origin shifts every offset.
        self.assertEqual(tt.snapshot(100.5)["stages"]["asr_done"], 400)

    def test_cancel_records_first_reason_and_a_stamp(self):
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        self.assertTrue(turn_trace.cancel("wake_word_barge"))
        turn_trace.cancel("vad_barge")
        self.assertEqual(tt.cancel_reason, "wake_word_barge")
        self.assertIn("cancelled", tt.stamps)

    def test_background_thread_falls_back_to_active_turn(self):
        """threading.Thread does not inherit contextvars; the ASR/speaker-ID pair
        and the surprise classifier still have to land on the turn."""
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        seen = {}

        def _worker():
            seen["trace"] = turn_trace.current()
            turn_trace.count("hosted.llm")
            turn_trace.stamp("asr_done")

        th = threading.Thread(target=_worker)
        th.start()
        th.join(2.0)
        self.assertIs(seen["trace"], tt)
        self.assertEqual(tt.calls, {"hosted.llm": 1})
        self.assertIn("asr_done", tt.stamps)

    def test_end_clears_active_only_for_that_turn(self):
        a, tok_a = turn_trace.begin(turn_id=1)
        turn_trace.end(tok_a)
        b, tok_b = turn_trace.begin(turn_id=2)
        self.addCleanup(turn_trace.end, tok_b)
        # Ending an already-ended token must not clear the newer active turn.
        turn_trace.end(tok_a)
        self.assertIs(turn_trace.current(), b)


class _FakeCreate:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return {"ok": True}


class _FakeChatClient:
    """Shaped like an OpenAI client: .chat.completions.create and .responses.create."""

    def __init__(self, with_responses=True):
        self.chat = mock.Mock()
        self.chat.completions = mock.Mock()
        self.chat.completions.create = _FakeCreate()
        if with_responses:
            self.responses = mock.Mock()
            self.responses.create = _FakeCreate()


class GuardClientCountingTest(unittest.TestCase):
    def setUp(self):
        turn_trace.reset_for_tests()
        self.addCleanup(turn_trace.reset_for_tests)

    def test_chat_and_responses_calls_are_counted_per_turn(self):
        from intelligence import connectivity
        client = connectivity.guard_client(_FakeChatClient(), "web_search")
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        client.chat.completions.create(model="x")
        client.chat.completions.create(model="x")
        client.responses.create(input="y")
        self.assertEqual(tt.calls.get("hosted.web_search"), 2)
        self.assertEqual(tt.calls.get("hosted.web_search.responses"), 1)
        self.assertEqual(turn_trace.totals().get("hosted.web_search"), 2)

    def test_counting_survives_offline_mode_disabled(self):
        import config
        from intelligence import connectivity
        with mock.patch.object(config, "OFFLINE_MODE_ENABLED", False):
            client = connectivity.guard_client(_FakeChatClient(), "empathy")
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        client.chat.completions.create()
        self.assertEqual(tt.calls, {"hosted.empathy": 1})

    def test_client_without_responses_api_is_fine(self):
        from intelligence import connectivity
        client = connectivity.guard_client(_FakeChatClient(with_responses=False), "llm")
        tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)
        client.chat.completions.create()
        self.assertEqual(tt.calls, {"hosted.llm": 1})

    def test_real_call_result_passes_through(self):
        from intelligence import connectivity
        client = connectivity.guard_client(_FakeChatClient(), "llm")
        self.assertEqual(client.chat.completions.create(), {"ok": True})


class LocalAndEmbedCountingTest(unittest.TestCase):
    def setUp(self):
        turn_trace.reset_for_tests()
        self.addCleanup(turn_trace.reset_for_tests)
        self.tt, token = turn_trace.begin()
        self.addCleanup(turn_trace.end, token)

    def test_local_generate_counts_before_transport(self):
        from intelligence import local_llm
        resp = mock.Mock()
        resp.content = b"{}"
        resp.json.return_value = {"response": "yes"}
        with (
            mock.patch.object(local_llm, "enabled", return_value=True),
            mock.patch.object(local_llm, "wait_for_server", return_value=True),
            mock.patch.object(local_llm.requests, "post", return_value=resp) as post,
        ):
            out = local_llm.generate("q", max_tokens=2, timeout_secs=0.5)
        self.assertEqual(out, "yes")
        post.assert_called_once()
        self.assertEqual(self.tt.calls, {"local.generate": 1})

    def test_embedding_request_counts(self):
        import numpy as np
        from memory import semantic
        resp = mock.Mock()
        resp.json.return_value = {"embedding": [3.0, 4.0]}
        with mock.patch("requests.post", return_value=resp):
            vec = semantic._request_embedding("hello", timeout=0.5)
        self.assertTrue(np.allclose(vec, [0.6, 0.8]))
        self.assertEqual(self.tt.calls, {"embed": 1})


class TranscriptTurnIdTest(unittest.TestCase):
    def setUp(self):
        from memory import conversations
        self.conv = conversations
        self.conv.clear_transcript()
        self.addCleanup(self.conv.clear_transcript)

    def test_entries_carry_monotonic_ids_and_timestamps(self):
        before = time.time()
        self.conv.add_to_transcript("Bret", "one")
        self.conv.add_to_transcript("Rex", "two")
        rows = self.conv.get_session_transcript()
        self.assertEqual([r["speaker"] for r in rows], ["Bret", "Rex"])
        self.assertLess(rows[0]["turn_id"], rows[1]["turn_id"])
        self.assertGreaterEqual(rows[0]["ts"], before)
        self.assertEqual(self.conv.last_turn_id(), rows[1]["turn_id"])
        # Existing shape is preserved.
        self.assertTrue(rows[0]["learnable"])

    def test_ids_do_not_restart_after_clear(self):
        self.conv.add_to_transcript("Bret", "one")
        last = self.conv.last_turn_id()
        self.conv.clear_transcript()
        self.assertEqual(self.conv.last_turn_id(), 0)
        self.conv.add_to_transcript("Bret", "again")
        self.assertGreater(self.conv.last_turn_id(), last)


class SpeechQueueSynthHookTest(unittest.TestCase):
    def test_item_accepts_hook_and_positional_legacy_shape(self):
        from audio import speech_queue
        done = threading.Event()
        legacy = speech_queue._Item(1, 1, "hi", "neutral", None, done)
        self.assertIsNone(legacy.on_synth_start)
        fired = []
        item = speech_queue._Item(1, 2, "hi", "neutral", None, done,
                                  on_synth_start=lambda: fired.append(1))
        item.on_synth_start()
        self.assertEqual(fired, [1])
        # Heap ordering still works with the new slot present.
        heap = [item, legacy]
        heapq.heapify(heap)
        self.assertIs(heapq.heappop(heap), legacy)

    def test_enqueue_threads_hook_into_the_item(self):
        from audio import speech_queue
        queue = object.__new__(speech_queue._SpeechQueue)
        queue._lock = threading.Lock()
        queue._not_empty = threading.Condition(queue._lock)
        queue._heap = []
        queue._seq = 0
        queue._speaking = False
        queue._current_priority = -1
        queue._startup_chime_queued = True
        hook = lambda: None
        with (
            mock.patch.object(speech_queue, "_state_suppresses_output", return_value=False),
            mock.patch.object(speech_queue, "_audio_output_suppressed", return_value=False),
        ):
            queue.enqueue("hello there", priority=1, on_synth_start=hook)
        self.assertEqual(len(queue._heap), 1)
        self.assertIs(queue._heap[0].on_synth_start, hook)


class CharacterLoopPayloadTest(unittest.TestCase):
    def setUp(self):
        turn_trace.reset_for_tests()
        self.addCleanup(turn_trace.reset_for_tests)

    def test_payload_carries_stages_calls_context_and_cancel(self):
        from intelligence import interaction as I
        trace = I._CharacterLoopTrace(
            turn_id=42, utterance="hi", heard_text="hi",
            from_idle_activation=False, turn_start=1000.0,
        )
        tt = turn_trace.TurnTrace(turn_id=42, started_at=1000.0)
        tt.stamp("asr_start", 1000.1)
        tt.stamp("model_first_token", 1001.4)
        tt.count("hosted.llm", 2)
        tt.set_value("context_chars", 3000)
        tt.cancel("vad_barge")
        trace.turn_trace = tt
        I._mark_first_response_queued(trace, text="Hey.", priority=1, queued_at=1001.5)
        I._mark_first_response_audio_started(trace, started_at=1002.0)
        with mock.patch.object(I._log, "info") as info:
            I._log_character_loop_trace(trace, final_executed_path="agenda_llm",
                                        completed=True, spoken_text="Hey.")
        line = next(c.args for c in info.call_args_list if c.args and c.args[0] == "[character_loop] %s")
        import json
        payload = json.loads(line[1])
        self.assertEqual(payload["stages"]["asr_start"], 100)
        self.assertEqual(payload["stages"]["model_first_token"], 1400)
        self.assertEqual(payload["stages"]["first_response_queued"], 1500)
        self.assertEqual(payload["stages"]["audio_started"], 2000)
        self.assertEqual(payload["calls"], {"hosted.llm": 2})
        self.assertEqual(payload["context"], {"context_chars": 3000})
        self.assertEqual(payload["cancel_reason"], "vad_barge")

    def test_payload_without_turn_trace_is_unchanged(self):
        from intelligence import interaction as I
        trace = I._CharacterLoopTrace(
            turn_id=1, utterance="x", heard_text="x",
            from_idle_activation=False, turn_start=1.0,
        )
        with mock.patch.object(I._log, "info") as info:
            I._log_character_loop_trace(trace, final_executed_path="p", completed=True)
        import json
        line = next(c.args for c in info.call_args_list if c.args and c.args[0] == "[character_loop] %s")
        payload = json.loads(line[1])
        self.assertNotIn("stages", payload)
        self.assertIn("timing", payload)


if __name__ == "__main__":
    unittest.main()
