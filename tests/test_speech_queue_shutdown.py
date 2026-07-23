import heapq
import threading
import unittest
from unittest import mock

from audio import speech_queue


class SpeechQueueShutdownTests(unittest.TestCase):
    def test_cancel_all_drops_waiting_items_and_interrupts_playback(self):
        queue = object.__new__(speech_queue._SpeechQueue)
        queue._lock = threading.Lock()
        queue._not_empty = threading.Condition(queue._lock)
        queue._speaking = True

        done_a = threading.Event()
        done_b = threading.Event()
        queue._heap = [
            speech_queue._Item(1, 1, "first", "neutral", None, done_a),
            speech_queue._Item(1, 2, "second", "neutral", None, done_b),
        ]
        heapq.heapify(queue._heap)

        with (
            mock.patch("audio.echo_cancel.request_cancel") as cancel,
            mock.patch("sounddevice.stop") as stop,
        ):
            queue.cancel_all()

        self.assertEqual(queue._heap, [])
        self.assertTrue(done_a.is_set())
        self.assertTrue(done_b.is_set())
        cancel.assert_called_once_with()
        stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
