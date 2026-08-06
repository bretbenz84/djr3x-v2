"""
Front-clipping of a FAST reply (field 2026-08-06 00:10, dev Mac).

Two utterances lost their opening words in one conversation:

    said "I know, am I right?"   HEARD "Am I right?"
    said "Well thanks"           HEARD "Thanks."

Both were REPLIES that started the instant Rex went quiet. Two compounding causes,
both about anchoring on the wrong clock:

1. `_apply_post_tts_handoff` runs from the speech-queue done-callback, which fires
   0.5-1.5s AFTER the audio actually stopped (streamed-take cache save, sequence
   bookkeeping). It set the capture floor to `now - grace` — i.e. AFTER words the
   human had already spoken into the clean post-playback buffer. The floor now
   anchors on `echo_cancel.last_playback_ended_at()`, the real end of sound.

2. Even with the floor right, software suppression flattens a fast reply's onset,
   so VAD triggers a beat late and a preroll-sized reach-back still misses the
   first word. When VAD fires within CAPTURE_FROM_FLOOR_NEAR_SECS of the floor,
   capture starts AT the floor — everything after it is post-playback and clean.

The hardware-AEC path keeps its grace reach-back PAST the real end (residual is
~17dB down there); the software path must not reach into playback at all, where
Rex's voice is at full volume and self-transcribes.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from audio import echo_cancel
from intelligence import interaction as I


class PlaybackEndStampTests(unittest.TestCase):
    """echo_cancel must record when sound really stopped — including for segments
    whose set_playing(False) the sequence hold swallows."""

    def setUp(self) -> None:
        echo_cancel.end_sequence(flush=False, tail_secs=0.0)
        self.addCleanup(echo_cancel.end_sequence, False, 0.0)

    def test_normal_stop_is_stamped(self):
        before = echo_cancel.last_playback_ended_at()
        echo_cancel.set_playing(True)
        echo_cancel.set_playing(False, tail_secs=0.0, flush=False)
        self.assertGreater(echo_cancel.last_playback_ended_at(), before)

    def test_mid_sequence_stop_is_stamped_even_though_it_is_swallowed(self):
        # THE case that matters: the final segment of a streamed reply ends inside
        # the sequence hold, so its set_playing(False) returns early — but that
        # instant is exactly when Rex went quiet.
        echo_cancel.start_sequence()
        before = echo_cancel.last_playback_ended_at()
        echo_cancel.set_playing(False, tail_secs=0.0, flush=False)
        self.assertGreater(echo_cancel.last_playback_ended_at(), before)
        self.assertTrue(echo_cancel._sequence_active, "hold must still be intact")

    def test_accessor_is_safe_before_any_playback(self):
        self.assertIsInstance(echo_cancel.last_playback_ended_at(), float)


class CaptureFloorAnchorTests(unittest.TestCase):

    def _floor_for(self, *, aec_on: bool, real_end: float, now: float) -> float:
        with mock.patch.object(echo_cancel, "last_playback_ended_at",
                               return_value=real_end), \
             mock.patch.object(I.hardware_aec, "is_active", return_value=aec_on), \
             mock.patch.object(I.time, "monotonic", return_value=now), \
             mock.patch.object(I, "_note_rex_spoke"), \
             mock.patch.object(I.vad, "reset_state"):
            I._apply_post_tts_handoff("A statement.", source="test")
        return I._listen_capture_floor_at

    def test_software_path_anchors_on_the_real_audio_end(self):
        # The field timeline: audio ends at T, callback runs 1.1s later.
        floor = self._floor_for(aec_on=False, real_end=1000.0, now=1001.1)
        self.assertAlmostEqual(floor, 1000.0, places=2)

    def test_software_path_does_not_reach_into_playback(self):
        # Reaching back past the real end would capture Rex at FULL volume on the
        # software path and self-transcribe.
        floor = self._floor_for(aec_on=False, real_end=1000.0, now=1001.1)
        self.assertGreaterEqual(floor, 1000.0)

    def test_hardware_aec_keeps_its_grace_reach_back(self):
        floor = self._floor_for(aec_on=True, real_end=1000.0, now=1001.1)
        self.assertLess(floor, 1000.0, "AEC path should still reach back past the end")
        self.assertGreater(floor, 999.0, "but only by the tuned grace, not seconds")

    def test_falls_back_to_now_when_no_playback_recorded(self):
        floor = self._floor_for(aec_on=False, real_end=0.0, now=1001.1)
        self.assertAlmostEqual(floor, 1001.1 - 0.12, places=2)

    def test_a_stale_future_stamp_is_ignored(self):
        floor = self._floor_for(aec_on=False, real_end=9999.0, now=1001.1)
        self.assertAlmostEqual(floor, 1001.1 - 0.12, places=2)

    def test_an_old_stamp_from_a_previous_line_is_ignored(self):
        # Caught by the suite before it could ship: the queue-callback lag is
        # sub-second, so a stamp from a minute ago belongs to an EARLIER line.
        # Anchoring on it would drag the floor a minute into the past and let
        # capture reach back over unrelated audio.
        floor = self._floor_for(aec_on=False, real_end=940.0, now=1001.1)
        self.assertAlmostEqual(floor, 1001.1 - 0.12, places=2)

    def test_a_stamp_inside_the_lag_bound_is_still_used(self):
        floor = self._floor_for(aec_on=False, real_end=1000.0, now=1001.1)
        self.assertAlmostEqual(floor, 1000.0, places=2)

    def test_the_lag_bound_is_configurable(self):
        with mock.patch.object(config, "CAPTURE_FLOOR_PLAYBACK_END_MAX_LAG_SECS", 0.5):
            floor = self._floor_for(aec_on=False, real_end=1000.0, now=1001.1)
        self.assertAlmostEqual(floor, 1001.1 - 0.12, places=2)


class FastReplyWideningTests(unittest.TestCase):

    def _window(self, speech_start: float, finished: float, floor: float) -> float:
        with mock.patch.object(I, "_listen_capture_floor_at", floor), \
             mock.patch.object(I, "_speech_preroll_secs", return_value=0.45):
            return I._speech_capture_secs(speech_start, finished_mono=finished)

    def test_a_reply_just_after_the_floor_captures_from_the_floor(self):
        # VAD fires 1.4s after the floor (suppression tail delayed it); the human
        # actually started at ~0.2s. Capture must reach back to the floor.
        secs = self._window(speech_start=1001.4, finished=1003.0, floor=1000.0)
        self.assertAlmostEqual(1003.0 - secs, 1000.0, places=2)

    def test_a_mid_conversation_utterance_is_unchanged(self):
        # Far from any floor: plain speech_start - preroll, no widening.
        secs = self._window(speech_start=1030.0, finished=1033.0, floor=940.0)
        self.assertAlmostEqual(secs, 3.0 + 0.45, places=2)

    def test_widening_respects_the_near_window(self):
        with mock.patch.object(config, "CAPTURE_FROM_FLOOR_NEAR_SECS", 0.5):
            secs = self._window(speech_start=1001.4, finished=1003.0, floor=1000.0)
        self.assertAlmostEqual(secs, 1.6 + 0.45, places=2)   # preroll only

    def test_widening_can_be_disabled(self):
        with mock.patch.object(config, "CAPTURE_FROM_FLOOR_NEAR_SECS", 0.0):
            secs = self._window(speech_start=1001.4, finished=1003.0, floor=1000.0)
        self.assertAlmostEqual(secs, 1.6 + 0.45, places=2)

    def test_never_reaches_back_before_the_floor(self):
        # Preroll would reach to 1000.35-0.45=999.9; the floor must still win.
        secs = self._window(speech_start=1000.35, finished=1002.0, floor=1000.0)
        self.assertLessEqual(1002.0 - secs + 1e-9, 1000.0 + 1e-9)
        self.assertAlmostEqual(1002.0 - secs, 1000.0, places=2)

    def test_window_is_clamped_to_the_ring_buffer(self):
        secs = self._window(speech_start=1001.0, finished=1003.0, floor=1.0)
        self.assertLessEqual(secs, float(config.AUDIO_BUFFER_SECONDS))


if __name__ == "__main__":
    unittest.main()
