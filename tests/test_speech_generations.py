"""Lean Brain phase 5, first slice — stale proactive speech never plays, and a
cut-short reply reports what was actually delivered.

speech_queue: DoneEvent carries played/dropped truth; items may carry the speech
generation they were decided in and are dropped (at enqueue or at pop) once a
newer generation exists. interaction: _begin_user_turn and every barge-in bump
the generation; _speak_proactive stamps its line; _speak_blocking treats a dropped
line as not completed; the streaming reply returns only the delivered sentences
when interrupted. No audio: the worker's TTS call is stubbed.
"""

from __future__ import annotations

import heapq
import threading
import unittest
from unittest import mock

from audio import speech_queue as SQ


def _bare_queue():
    q = object.__new__(SQ._SpeechQueue)
    q._lock = threading.Lock()
    q._not_empty = threading.Condition(q._lock)
    q._heap = []
    q._seq = 0
    q._speaking = False
    q._last_speech_end_at = 0.0
    q._current_priority = -1
    q._current_audio_path = None
    q._startup_chime_queued = True
    return q


class DoneEventTest(unittest.TestCase):
    def test_drop_records_reason_once(self):
        ev = SQ.DoneEvent()
        self.assertFalse(ev.played)
        ev.drop("stale_generation")
        ev.drop("other")
        self.assertTrue(ev.is_set())
        self.assertEqual(ev.dropped_reason, "stale_generation")

    def test_no_audio_completion_counts_as_delivered(self):
        ev = SQ.DoneEvent()
        with mock.patch("utils.conv_log.log_rex"):
            out = SQ._complete_text_without_audio("hi", ev, None)
        self.assertIs(out, ev)
        self.assertTrue(ev.played)
        self.assertIsNone(ev.dropped_reason)


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self._saved = SQ._generation
        self.addCleanup(lambda: setattr(SQ, "_generation", self._saved))

    def test_invalidate_bumps(self):
        g0 = SQ.generation()
        g1 = SQ.invalidate_pending("test")
        self.assertEqual(g1, g0 + 1)
        self.assertEqual(SQ.generation(), g1)

    def test_stale_at_enqueue_is_dropped_unplayed(self):
        q = _bare_queue()
        g = SQ.generation()
        SQ.invalidate_pending("user_turn")
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch.object(SQ, "_audio_output_suppressed", return_value=False),
        ):
            done = q.enqueue("old news", priority=1, generation=g)
        self.assertTrue(done.is_set())
        self.assertEqual(done.dropped_reason, "stale_generation")
        self.assertFalse(done.played)
        self.assertEqual(q._heap, [])

    def test_current_generation_enqueues(self):
        q = _bare_queue()
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch.object(SQ, "_audio_output_suppressed", return_value=False),
        ):
            done = q.enqueue("fresh", priority=1, generation=SQ.generation())
        self.assertFalse(done.is_set())
        self.assertEqual(len(q._heap), 1)
        self.assertEqual(q._heap[0].generation, SQ.generation())

    def test_reply_items_without_generation_never_stale(self):
        q = _bare_queue()
        SQ.invalidate_pending("x")
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch.object(SQ, "_audio_output_suppressed", return_value=False),
        ):
            done = q.enqueue("reply", priority=1)
        self.assertFalse(done.is_set())
        self.assertIsNone(q._heap[0].generation)

    def test_drop_helpers_record_reasons(self):
        q = _bare_queue()
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch.object(SQ, "_audio_output_suppressed", return_value=False),
        ):
            a = q.enqueue("a", priority=1, tag="t")
            b = q.enqueue("b", priority=1)
        q.drop_by_tag("t")
        self.assertEqual(a.dropped_reason, "dropped_by_tag")
        q.clear_below_priority(5)
        self.assertEqual(b.dropped_reason, "cleared_below_priority")


class WorkerStaleDropTest(unittest.TestCase):
    """Run the worker's per-item body directly: a stale item is dropped at pop, a
    fresh one plays and is marked played."""

    def setUp(self):
        self._saved = SQ._generation
        self.addCleanup(lambda: setattr(SQ, "_generation", self._saved))

    def _process(self, item):
        q = _bare_queue()
        played = []
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch("audio.tts.speak", side_effect=lambda text, *a, **k: played.append(text)),
            mock.patch("audio.sound_effects.play_for_speech"),
        ):
            q._process_item(item)
        self.assertFalse(q._speaking)
        return played

    def test_stale_item_dropped_at_pop(self):
        done = SQ.DoneEvent()
        item = SQ._Item(1, 1, "stale line", "neutral", None, done, generation=SQ.generation())
        SQ.invalidate_pending("barge")
        played = self._process(item)
        self.assertEqual(played, [])
        self.assertTrue(done.is_set())
        self.assertEqual(done.dropped_reason, "stale_generation")
        self.assertFalse(done.played)

    def test_fresh_item_plays_and_is_marked(self):
        done = SQ.DoneEvent()
        item = SQ._Item(1, 1, "fresh line", "neutral", None, done, generation=SQ.generation())
        played = self._process(item)
        self.assertEqual(played, ["fresh line"])
        self.assertTrue(done.played)
        self.assertIsNone(done.dropped_reason)

    def test_tts_failure_is_not_played(self):
        done = SQ.DoneEvent()
        item = SQ._Item(1, 1, "boom", "neutral", None, done)
        q = _bare_queue()
        with (
            mock.patch.object(SQ, "_state_suppresses_output", return_value=False),
            mock.patch("audio.tts.speak", side_effect=RuntimeError("no tts")),
            mock.patch("audio.sound_effects.play_for_speech"),
        ):
            q._process_item(item)
        self.assertTrue(done.is_set())
        self.assertFalse(done.played)


class InteractionWiringTest(unittest.TestCase):
    def setUp(self):
        self._saved = SQ._generation
        self.addCleanup(lambda: setattr(SQ, "_generation", self._saved))

    def test_begin_user_turn_bumps_generation(self):
        from intelligence import interaction as I
        g0 = SQ.generation()
        with (
            mock.patch.object(I, "_situation_assessor"),
            mock.patch.object(I, "_start_listening_motion", create=True),
        ):
            try:
                I._begin_user_turn()
            except Exception:
                pass
        self.assertGreater(SQ.generation(), g0)

    def test_speak_proactive_stamps_generation(self):
        from intelligence import interaction as I
        seen = {}

        def fake_blocking(text, **kw):
            seen.update(kw)
            return True

        with (
            mock.patch.object(I, "_speak_blocking", side_effect=fake_blocking),
            mock.patch.object(I, "_text_only_mode", True),
            mock.patch.object(I, "_last_user_turn_started_at", 0.0),
            mock.patch.object(I.speech_queue, "is_speaking", return_value=False),
            mock.patch("audio.output_gate.is_busy", return_value=False),
        ):
            I._speak_proactive("hello", decided_at=10.0)
        self.assertEqual(seen.get("generation"), SQ.generation())

    def test_speak_blocking_dropped_line_is_not_completed(self):
        from intelligence import interaction as I
        ev = SQ.DoneEvent()
        ev.drop("stale_generation")
        with (
            mock.patch.object(I, "_can_speak", return_value=True),
            mock.patch.object(I.llm, "clean_response_text", side_effect=lambda t: t),
            mock.patch.object(I.motion_controller, "last_refusal", return_value=None),
            mock.patch.object(I.speech_queue, "enqueue", return_value=ev),
            mock.patch.object(I, "_apply_post_tts_handoff") as handoff,
        ):
            ok = I._speak_blocking("a line", generation=3)
        self.assertFalse(ok)
        handoff.assert_not_called()


class DeliveredTextTest(unittest.TestCase):
    def test_zip_of_spoken_and_played_events(self):
        # The exact rule _stream_and_speak_sentences applies when cut short.
        spoken = ["First.", "Second.", "Third."]
        evs = [SQ.DoneEvent(), SQ.DoneEvent(), SQ.DoneEvent()]
        evs[0].played = True
        evs[1].played = True
        evs[2].drop("cleared_below_priority")
        delivered = [p for p, ev in zip(spoken, evs) if p and getattr(ev, "played", False)]
        self.assertEqual(" ".join(delivered), "First. Second.")


if __name__ == "__main__":
    unittest.main()
