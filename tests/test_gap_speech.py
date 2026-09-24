"""
Thinking-gap speech recovery (GAP_SPEECH_* in config).

From a turn's endpoint until well after Rex's reply finishes, the interaction
loop is blocked inside the handler and live VAD is dead. A second line spoken
in that window ("...oh, and one more thing") lands clean in the 30s rolling
buffer and used to be erased when the post-TTS handoff stamped the capture
floor at playback end — the person read it as Rex ignoring them.

Catch-up: when the loop resumes listening, a one-shot scan sweeps
the whole blind span. A finished missed utterance is sliced from the buffer
and dispatched through the normal turn pipeline; speech still in progress is
handed to the live path as a recovered onset.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import config
from intelligence import interaction as I

SR = int(config.AUDIO_SAMPLE_RATE)


def _reset_gap_state():
    I.turn_coordinator.pending.clear()
    I._gap_watch_started_at = 0.0
    I._gap_first_audio_at = 0.0
    I._gap_recovery_floor_at = 0.0


class GapConfigTests(unittest.TestCase):
    """Pin the knob names the getattr sites reach for — a typo'd name silently
    falls back to the hardcoded default and the knob goes dead."""

    def test_knobs_exist(self):
        for name in (
            "GAP_SPEECH_RECOVERY_ENABLED",
            "GAP_CATCHUP_ENABLED",
            "GAP_SPEECH_MIN_VOICED_SECS",
            "GAP_CATCHUP_MAX_SPAN_SECS",
            "GAP_SPEECH_JOIN_GAP_SECS",
            "GAP_SPEECH_SLICE_PAD_SECS",
            "GAP_SPEECH_POST_PLAYBACK_SKIP_SECS",
            "GAP_CATCHUP_PLAYBACK_MIN_SPEECH_SECS",
            "GAP_CATCHUP_PLAYBACK_RMS_RATIO",
            "GAP_CATCHUP_PLAYBACK_MAX_VOICED_FRACTION",
            "GAP_CATCHUP_UNDER_PLAYBACK_REQUIRE_TRUSTED",
        ):
            self.assertTrue(hasattr(config, name), name)

    def test_catchup_span_fits_the_ring_buffer(self):
        self.assertLess(
            float(config.GAP_CATCHUP_MAX_SPAN_SECS),
            float(config.AUDIO_BUFFER_SECONDS),
        )


class ArmDisarmTests(unittest.TestCase):

    def setUp(self):
        _reset_gap_state()
        self.addCleanup(_reset_gap_state)

    def test_arm_stamps_the_watermark_and_clears_first_audio(self):
        I._gap_first_audio_at = 123.0
        I._arm_gap_watch()
        self.assertGreater(I._gap_watch_started_at, 0.0)
        self.assertEqual(I._gap_first_audio_at, 0.0)

    def test_disarm_clears_everything(self):
        I._arm_gap_watch()
        I._gap_first_audio_at = 5.0
        I._gap_recovery_floor_at = 6.0
        I._disarm_gap_watch()
        self.assertEqual(I._gap_watch_started_at, 0.0)
        self.assertEqual(I._gap_first_audio_at, 0.0)
        self.assertEqual(I._gap_recovery_floor_at, 0.0)

    def test_first_queue_item_stamps_first_audio_once(self):
        # The stamp bounds the CLEAN thinking gap; later sentences of the same
        # reply must not advance it.
        I._arm_gap_watch()
        with mock.patch.object(I, "_note_rex_spoke"):
            I._note_rex_spoke_item(mock.Mock(text="First sentence."))
            first = I._gap_first_audio_at
            self.assertGreater(first, 0.0)
            I._note_rex_spoke_item(mock.Mock(text="Second sentence."))
        self.assertEqual(I._gap_first_audio_at, first)

    def test_no_stamp_when_disarmed(self):
        with mock.patch.object(I, "_note_rex_spoke"):
            I._note_rex_spoke_item(mock.Mock(text="Boot line."))
        self.assertEqual(I._gap_first_audio_at, 0.0)


class GapRecoveryFloorOverrideTests(unittest.TestCase):
    """_speech_capture_secs honors the catch-up's one-shot floor override the
    same way it honors the game-barge override: the recovered utterance's onset
    predates the playback-end floor, and the floor must not clip its front."""

    def _window(self, speech_start: float, finished: float, floor: float,
                gap_floor: float) -> float:
        with mock.patch.object(I, "_listen_capture_floor_at", floor), \
             mock.patch.object(I, "_game_barge_floor_at", 0.0), \
             mock.patch.object(I, "_gap_recovery_floor_at", gap_floor), \
             mock.patch.object(I, "_speech_preroll_secs", return_value=0.45):
            return I._speech_capture_secs(speech_start, finished_mono=finished)

    def test_override_reaches_back_behind_the_playback_floor(self):
        # Reply playback ended at 1003 (floor). The person had started talking
        # at 1001.5, under/before it; the catch-up recorded 1001.3.
        secs = self._window(speech_start=1001.5, finished=1004.0,
                            floor=1003.0, gap_floor=1001.3)
        self.assertAlmostEqual(1004.0 - secs, 1001.3, places=2)

    def test_inactive_override_changes_nothing(self):
        secs = self._window(speech_start=1002.0, finished=1004.0,
                            floor=1003.0, gap_floor=0.0)
        self.assertAlmostEqual(1004.0 - secs, 1003.0, places=2)

    def test_stale_override_above_the_floor_is_ignored(self):
        secs = self._window(speech_start=1005.0, finished=1007.0,
                            floor=1000.0, gap_floor=1003.0)
        self.assertAlmostEqual(secs, 2.0 + 0.45, places=2)


class GapVoicedRunsTests(unittest.TestCase):
    """Span VAD → absolute-time runs, joined across sub-endpoint pauses."""

    def test_runs_map_to_absolute_time(self):
        audio = np.zeros(SR * 4, dtype=np.float32)
        with mock.patch.object(I.vad, "get_speech_segments",
                               return_value=[(0.5, 1.2)]):
            runs = I._gap_voiced_runs(audio, actual_start=100.0)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][0], 100.5, places=3)
        self.assertAlmostEqual(runs[0][1], 101.2, places=3)

    def test_breath_pauses_join_into_one_utterance(self):
        with mock.patch.object(I.vad, "get_speech_segments",
                               return_value=[(0.5, 1.2), (1.9, 2.6)]), \
             mock.patch.object(config, "GAP_SPEECH_JOIN_GAP_SECS", 1.2):
            runs = I._gap_voiced_runs(np.zeros(SR * 4, dtype=np.float32), 0.0)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][1], 2.6, places=3)

    def test_distinct_utterances_stay_separate(self):
        with mock.patch.object(I.vad, "get_speech_segments",
                               return_value=[(0.5, 1.2), (3.5, 4.2)]), \
             mock.patch.object(config, "GAP_SPEECH_JOIN_GAP_SECS", 1.2):
            runs = I._gap_voiced_runs(np.zeros(SR * 5, dtype=np.float32), 0.0)
        self.assertEqual(len(runs), 2)

    def test_empty_audio_is_no_runs(self):
        self.assertEqual(I._gap_voiced_runs(np.zeros(0, dtype=np.float32), 0.0), [])


class CatchUpTests(unittest.TestCase):
    """The one-shot blind-span sweep at loop resume."""

    def setUp(self):
        self.enterContext(mock.patch.object(I, "_note_voice_bearing"))
        _reset_gap_state()
        self.addCleanup(_reset_gap_state)
        self.now = 3000.0

    def test_later_finished_utterance_is_retained_in_order(self):
        result, handled = self._catch_up(armed_ago=10.0, runs=[(8.0, 7.0), (4.0, 3.0)])
        self.assertEqual(result, ("handled", None))
        with mock.patch.object(I.time, "monotonic", return_value=self.now):
            pending = I.turn_coordinator.pending.pop(I.conv_memory.transcript_version()[0])
        self.assertIsNotNone(pending)
        self.assertLess(pending.started_at, self.now - 4.0)
        self.assertGreater(pending.ended_at, self.now - 3.0)
        self.assertFalse(pending.require_trusted)

    def _catch_up(self, *, armed_ago, first_audio_ago=None, play_end_ago=None,
                  runs=(), aec_on=False, audio=None, span_secs=None):
        I._gap_watch_started_at = self.now - armed_ago
        I._gap_first_audio_at = (
            self.now - first_audio_ago if first_audio_ago is not None else 0.0
        )
        play_end = self.now - play_end_ago if play_end_ago is not None else 0.0
        if audio is None:
            secs = span_secs if span_secs is not None else armed_ago
            audio = np.zeros(int(secs * SR), dtype=np.float32)
        handled = {}

        def _handler(seg, **kwargs):
            handled["audio"] = seg
            handled["kwargs"] = kwargs

        abs_runs = [(self.now - s, self.now - e) for s, e in runs]
        with mock.patch.object(I.time, "monotonic", return_value=self.now), \
             mock.patch.object(I.hardware_aec, "is_active", return_value=aec_on), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at",
                               return_value=play_end), \
             mock.patch.object(I.stream, "get_audio_chunk", return_value=audio), \
             mock.patch.object(I, "_gap_voiced_runs", return_value=abs_runs), \
             mock.patch.object(I, "_handle_speech_segment", side_effect=_handler), \
             mock.patch.object(I, "_begin_user_turn"), \
             mock.patch.object(I, "_end_user_turn"):
            result = I._maybe_catch_up_gap_speech()
        return result, handled

    def test_disarmed_is_none(self):
        result, handled = self._catch_up(armed_ago=0.0, runs=())
        I._gap_watch_started_at = 0.0
        self.assertIsNone(I._maybe_catch_up_gap_speech())

    def test_silence_consumes_the_watch(self):
        result, handled = self._catch_up(armed_ago=8.0, runs=())
        self.assertIsNone(result)
        self.assertEqual(I._gap_watch_started_at, 0.0)

    def test_finished_clean_utterance_is_dispatched(self):
        # No playback at all this turn (silent handler) — the whole span is
        # clean; a 1s utterance spoken 5s ago must reach the turn pipeline.
        result, handled = self._catch_up(armed_ago=8.0, runs=[(6.0, 5.0)])
        self.assertEqual(result, ("handled", None))
        self.assertIn("audio", handled)
        pad = float(config.GAP_SPEECH_SLICE_PAD_SECS)
        self.assertAlmostEqual(len(handled["audio"]) / SR, 1.0 + 2 * pad, delta=0.1)

    def test_dispatch_rearms_the_watch_for_fresh_audio(self):
        self._catch_up(armed_ago=8.0, runs=[(6.0, 5.0)])
        self.assertAlmostEqual(I._gap_watch_started_at, self.now, places=3)

    def test_speech_still_in_progress_hands_a_live_onset(self):
        result, handled = self._catch_up(armed_ago=8.0, runs=[(1.5, 0.1)])
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "live")
        self.assertAlmostEqual(result[1], self.now - 1.5, places=3)
        self.assertNotIn("audio", handled)
        # The recovered onset predates the playback-end floor — the override
        # must be armed so capture reaches back to it.
        self.assertGreater(I._gap_recovery_floor_at, 0.0)
        self.assertLessEqual(I._gap_recovery_floor_at, self.now - 1.5)

    def test_no_aec_speech_under_playback_is_not_recovered(self):
        # Reply audio played 6s→1s ago; the person talked over it at 4s→2.5s
        # ago. Without hardware AEC that buffer span holds Rex at full volume —
        # physics says unrecoverable; nothing may be dispatched.
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(4.0, 2.5)])
        self.assertIsNone(result)
        self.assertNotIn("audio", handled)

    def test_no_aec_thinking_gap_speech_is_recovered(self):
        # The second line landed BEFORE playback began (7.5s→6.5s ago, audio
        # started 6s ago): clean on every machine, must dispatch.
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(7.5, 6.5)])
        self.assertEqual(result, ("handled", None))
        self.assertIn("audio", handled)

    def test_no_aec_slice_never_extends_into_playback(self):
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(7.5, 6.1)])
        self.assertEqual(result, ("handled", None))
        # Slice may pad left but must stop before first audio (6s ago).
        max_len = (8.0 - 6.0) * SR
        self.assertLessEqual(len(handled["audio"]), max_len)

    def test_aec_interjection_under_playback_is_recovered(self):
        # Robot: reply played 6s→1s ago; the person spoke a full 1.5s line over
        # it (4.5s→3.0s ago), acoustically dominant over the residual floor.
        audio = (np.random.default_rng(7).standard_normal(8 * SR) * 0.005) \
            .astype(np.float32)
        i0 = int((8.0 - 4.5) * SR)
        i1 = int((8.0 - 3.0) * SR)
        audio[i0:i1] = 0.08
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(4.5, 3.0)], aec_on=True, audio=audio)
        self.assertEqual(result, ("handled", None))
        self.assertIn("audio", handled)

    def test_aec_short_backchannel_under_playback_is_ignored(self):
        # "uh-huh" (0.5s) under Rex's reply: a human DJ wouldn't stop the show
        # for it either.
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(4.0, 3.5)], aec_on=True)
        self.assertIsNone(result)

    def test_aec_quiet_residual_shaped_run_is_ignored(self):
        # A long voiced run at the SAME level as the playback span's floor is
        # Rex's own residual, not a person.
        audio = np.full(8 * SR, 0.01, dtype=np.float32)
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(4.5, 3.0)], aec_on=True, audio=audio)
        self.assertIsNone(result)

    def test_aec_continuously_voiced_playback_span_is_ignored(self):
        # The playback span voiced nearly end-to-end IS Rex's own reply.
        audio = np.full(8 * SR, 0.08, dtype=np.float32)
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(5.9, 1.1)], aec_on=True, audio=audio)
        self.assertIsNone(result)

    def test_stale_watch_is_dropped(self):
        result, handled = self._catch_up(armed_ago=28.0, runs=[(6.0, 5.0)],
                                         span_secs=25.0)
        self.assertIsNone(result)

    def test_kill_switch(self):
        with mock.patch.object(config, "GAP_CATCHUP_ENABLED", False):
            result, handled = self._catch_up(armed_ago=8.0, runs=[(6.0, 5.0)])
        self.assertIsNone(result)
        self.assertEqual(I._gap_watch_started_at, 0.0, "watch must still be consumed")

    def test_handler_crash_is_contained(self):
        I._gap_watch_started_at = self.now - 8.0
        audio = np.zeros(8 * SR, dtype=np.float32)
        with mock.patch.object(I.time, "monotonic", return_value=self.now), \
             mock.patch.object(I.hardware_aec, "is_active", return_value=False), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at",
                               return_value=0.0), \
             mock.patch.object(I.stream, "get_audio_chunk", return_value=audio), \
             mock.patch.object(I, "_gap_voiced_runs",
                               return_value=[(self.now - 6.0, self.now - 5.0)]), \
             mock.patch.object(I, "_handle_speech_segment",
                               side_effect=RuntimeError("boom")), \
             mock.patch.object(I, "_begin_user_turn"), \
             mock.patch.object(I, "_end_user_turn") as end_turn:
            result = I._maybe_catch_up_gap_speech()
        self.assertEqual(result, ("handled", None))
        self.assertTrue(end_turn.called)


# Append these two methods to the EXISTING class CatchUpTests (4-space indent,
# after test_aec_continuously_voiced_playback_span_is_ignored). They depend on
# the _handler edit that records kwargs.

    def test_under_playback_recovery_is_trust_gated(self):
        # 2026-08-27 13:34:16 — this exact shape (a 1.5s dominant run under
        # playback) decoded "Look me what you got me." (-0.55) and Rex asked an
        # empty room "Sorry, one more time?".
        audio = (np.random.default_rng(7).standard_normal(8 * SR) * 0.005) \
            .astype(np.float32)
        i0 = int((8.0 - 4.5) * SR)
        i1 = int((8.0 - 3.0) * SR)
        audio[i0:i1] = 0.08
        result, handled = self._catch_up(
            armed_ago=8.0, first_audio_ago=6.0, play_end_ago=1.0,
            runs=[(4.5, 3.0)], aec_on=True, audio=audio)
        self.assertEqual(result, ("handled", None))
        self.assertTrue(handled["kwargs"]["require_trusted"])

    def test_clean_gap_recovery_is_not_trust_gated(self):
        # MUST KEEP WORKING: a line spoken into a silent thinking gap is heard
        # whatever the decoder thinks of it.
        result, handled = self._catch_up(armed_ago=8.0, runs=[(6.0, 5.0)])
        self.assertEqual(result, ("handled", None))
        self.assertFalse(handled["kwargs"]["require_trusted"])


# Append this class before the `if __name__ == "__main__":` guard.

class CatchUpTrustGateTests(unittest.TestCase):
    """The handler side of the gate. 2026-08-27 13:34:17 — an untrusted decode
    of a slice cut from under Rex's own playback reached the turn pipeline and
    was answered with "Sorry, one more time?", which played more audio, which
    seeded the next residual."""

    def _untrusted(self):
        return I.transcription.Transcript(
            "Look me what you got me.", avg_logprob=-0.55,
            confident=False, backend="qwen3_asr")

    def _run(self, *, require_trusted):
        with mock.patch.object(I.random, "randint", return_value=0), \
             mock.patch.object(
                 I, "_process_audio",
                 return_value=(self._untrusted(), None, None, 0.0, 0.0, 0.07)), \
             mock.patch.object(I, "_looks_like_third_party_crosstalk",
                               return_value=False), \
             mock.patch.object(I, "_looks_like_own_echo", return_value=True), \
             mock.patch.object(I, "_log_character_loop_trace"), \
             mock.patch.object(I, "_new_character_loop_trace") as trace, \
             mock.patch.object(I, "_speak_blocking", return_value=True) as speak:
            I._handle_speech_segment(
                np.ones(16, dtype=np.float32), require_trusted=require_trusted)
        return trace, speak

    def test_untrusted_under_playback_never_reaches_the_turn(self):
        trace, speak = self._run(require_trusted=True)
        trace.assert_not_called()
        speak.assert_not_called()

    def test_the_same_transcript_is_processed_normally_without_the_flag(self):
        # The gate must be the ONLY thing that changed: an untrusted transcript
        # off the live mic path still enters the pipeline as before.
        trace, _speak = self._run(require_trusted=False)
        trace.assert_called_once()


if __name__ == "__main__":
    unittest.main()
