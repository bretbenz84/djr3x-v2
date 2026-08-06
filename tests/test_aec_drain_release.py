"""
The mic must re-open when Rex's SOUND ends, not when his bookkeeping does.

Field 2026-08-05 (three marked repeats in one run, logs/conversation-...23-53-30):
"Whenever I didn't see a transcription after 1-2 seconds, I repeated myself."

`echo_cancel.start_sequence()` defers every per-segment `set_playing(False)` until
`end_sequence()`, and that release sat at the END of the reply path — behind the
post-greet relationship ask, the curiosity routine, and pool-topic recording. So the
mic stayed attenuated for 1-5s AFTER Rex's last audio (measured across five runs:
medians 1.0-2.0s, max 8.0s). `_chunk_for_vad` flattened a reply spoken into that
window, VAD never fired, and the lost turn left NO trace — the capture telemetry
reported captured=6 / dropped=0 for the very run that lost three utterances, which
is what proved the loss was upstream of capture entirely.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from audio import echo_cancel, speech_queue
from intelligence import interaction as I


class IsDrainedTests(unittest.TestCase):
    """`is_drained` must distinguish "between sentences" from "genuinely finished" —
    releasing on the former would re-open the mic into Rex's own next sentence."""

    def _q(self, speaking: bool, heap: list):
        q = speech_queue._queue
        return mock.patch.object(q, "_speaking", speaking), mock.patch.object(q, "_heap", heap)

    def test_speaking_is_not_drained(self):
        a, b = self._q(True, [])
        with a, b:
            self.assertFalse(speech_queue.is_drained())

    def test_idle_with_queued_items_is_not_drained(self):
        # The between-sentences case of one streamed reply.
        a, b = self._q(False, ["next sentence"])
        with a, b:
            self.assertFalse(speech_queue.is_drained())

    def test_idle_and_empty_is_drained(self):
        a, b = self._q(False, [])
        with a, b:
            self.assertTrue(speech_queue.is_drained())


class DrainReleaseTests(unittest.TestCase):

    def setUp(self) -> None:
        echo_cancel.end_sequence(flush=False, tail_secs=0.0)
        self.addCleanup(echo_cancel.end_sequence, False, 0.0)

    def test_mid_reply_item_keeps_the_hold(self):
        echo_cancel.start_sequence()
        with mock.patch.object(speech_queue, "is_drained", return_value=False), \
             mock.patch.object(I, "_apply_post_tts_handoff"):
            I._arm_post_tts_window(mock.Mock(text="first sentence"))
        self.assertTrue(echo_cancel._sequence_active)

    def test_final_item_releases_the_hold(self):
        echo_cancel.start_sequence()
        with mock.patch.object(speech_queue, "is_drained", return_value=True), \
             mock.patch.object(I, "_apply_post_tts_handoff"):
            I._arm_post_tts_window(mock.Mock(text="last sentence."))
        self.assertFalse(echo_cancel._sequence_active)

    def test_release_does_not_wait_for_the_reply_path_bookkeeping(self):
        # The regression's shape: the hold must NOT depend on anything after the
        # audio — the queue callback alone has to be sufficient.
        echo_cancel.start_sequence()
        with mock.patch.object(speech_queue, "is_drained", return_value=True), \
             mock.patch.object(I, "_apply_post_tts_handoff"):
            I._arm_post_tts_window(mock.Mock(text="done."))
        self.assertFalse(echo_cancel._sequence_active)

    def test_flag_off_restores_release_at_end_of_turn(self):
        echo_cancel.start_sequence()
        with mock.patch.object(config, "AEC_RELEASE_ON_QUEUE_DRAIN", False), \
             mock.patch.object(speech_queue, "is_drained", return_value=True), \
             mock.patch.object(I, "_apply_post_tts_handoff"):
            I._arm_post_tts_window(mock.Mock(text="done."))
        self.assertTrue(echo_cancel._sequence_active)

    def test_a_broken_queue_probe_cannot_break_the_handoff(self):
        echo_cancel.start_sequence()
        with mock.patch.object(speech_queue, "is_drained",
                               side_effect=RuntimeError("queue gone")), \
             mock.patch.object(I, "_apply_post_tts_handoff") as handoff:
            I._arm_post_tts_window(mock.Mock(text="done."))
        handoff.assert_called_once()      # the deaf-window arming still happened

    def test_handoff_still_runs_before_the_release(self):
        echo_cancel.start_sequence()
        with mock.patch.object(speech_queue, "is_drained", return_value=True), \
             mock.patch.object(I, "_apply_post_tts_handoff") as handoff:
            I._arm_post_tts_window(mock.Mock(text="done."))
        handoff.assert_called_once()


if __name__ == "__main__":
    unittest.main()
