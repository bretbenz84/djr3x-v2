"""
Thinking-gap speech recovery (GAP_SPEECH_* in config).

From a turn's endpoint until well after Rex's reply finishes, the interaction
loop is blocked inside the handler and live VAD is dead. A second line spoken
in that window ("...oh, and one more thing") lands clean in the 30s rolling
buffer and used to be erased when the post-TTS handoff stamped the capture
floor at playback end — the person read it as Rex ignoring them.

Phase 1 (merge): at the moment the reply's first sentence exists — before any
TTS is fetched or queued — the stream path scans the gap span and, if the
person spoke, unwinds via _GapSpeechDetected so the call site captures their
line and regenerates once with both lines.

Phase 2 (catch-up): when the loop resumes listening, a one-shot scan sweeps
the whole blind span. A finished missed utterance is sliced from the buffer
and dispatched through the normal turn pipeline; speech still in progress is
handed to the live path as a recovered onset.
"""

from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

import config
from intelligence import interaction as I

SR = int(config.AUDIO_SAMPLE_RATE)


def _reset_gap_state():
    I._gap_watch_started_at = 0.0
    I._gap_first_audio_at = 0.0
    I._gap_recovery_floor_at = 0.0


class GapConfigTests(unittest.TestCase):
    """Pin the knob names the getattr sites reach for — a typo'd name silently
    falls back to the hardcoded default and the knob goes dead."""

    def test_knobs_exist(self):
        for name in (
            "GAP_SPEECH_RECOVERY_ENABLED",
            "GAP_MERGE_ENABLED",
            "GAP_CATCHUP_ENABLED",
            "GAP_SPEECH_MIN_VOICED_SECS",
            "GAP_MERGE_MIN_SPAN_SECS",
            "GAP_MERGE_MAX_SPAN_SECS",
            "GAP_CATCHUP_MAX_SPAN_SECS",
            "GAP_SPEECH_JOIN_GAP_SECS",
            "GAP_SPEECH_SLICE_PAD_SECS",
            "GAP_SPEECH_POST_PLAYBACK_SKIP_SECS",
            "GAP_CATCHUP_PLAYBACK_MIN_SPEECH_SECS",
            "GAP_CATCHUP_PLAYBACK_RMS_RATIO",
            "GAP_CATCHUP_PLAYBACK_MAX_VOICED_FRACTION",
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


class ReplyGapOnsetTests(unittest.TestCase):
    """Phase-1 trigger: conservative, and never fooled by Rex's own audio."""

    def setUp(self):
        _reset_gap_state()
        self.addCleanup(_reset_gap_state)
        self.now = 2000.0
        I._gap_watch_started_at = self.now - 3.0   # 3s generation gap

    def _onset(self, *, runs, suppressed=False, busy=False, play_end=0.0,
               now=None):
        span_audio = np.zeros(SR * 3, dtype=np.float32)
        with mock.patch.object(I.time, "monotonic", return_value=now or self.now), \
             mock.patch.object(I.echo_cancel, "is_suppressed", return_value=suppressed), \
             mock.patch.object(I.output_gate, "is_busy", return_value=busy), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at",
                               return_value=play_end), \
             mock.patch.object(I.stream, "get_audio_chunk", return_value=span_audio), \
             mock.patch.object(I, "_gap_voiced_runs", return_value=runs):
            return I._reply_gap_speech_onset()

    def test_gap_speech_is_detected(self):
        onset = self._onset(runs=[(1998.0, 1999.1)])
        self.assertAlmostEqual(onset, 1998.0, places=3)

    def test_silence_is_none(self):
        self.assertIsNone(self._onset(runs=[]))

    def test_a_blip_below_the_voiced_minimum_is_ignored(self):
        self.assertIsNone(self._onset(runs=[(1998.0, 1998.2)]))

    def test_disarmed_is_none(self):
        I._gap_watch_started_at = 0.0
        self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)]))

    def test_audio_sounding_right_now_skips_the_check(self):
        # A chirp/effect is playing — the buffer tail is not the user's voice.
        self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)], suppressed=True))
        self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)], busy=True))

    def test_playback_inside_the_span_trims_the_scan(self):
        # An ack finished at 1998.5; a "run" that VAD saw before it (the ack
        # itself, at full volume on a no-AEC machine) must not trigger. The
        # trimmed scan starts after the echo skip, and the span audio is
        # requested from there.
        requested = {}

        def _chunk(secs):
            requested["secs"] = secs
            return np.zeros(int(secs * SR), dtype=np.float32)

        with mock.patch.object(I.time, "monotonic", return_value=self.now), \
             mock.patch.object(I.echo_cancel, "is_suppressed", return_value=False), \
             mock.patch.object(I.output_gate, "is_busy", return_value=False), \
             mock.patch.object(I.echo_cancel, "last_playback_ended_at",
                               return_value=1998.5), \
             mock.patch.object(I.stream, "get_audio_chunk", side_effect=_chunk), \
             mock.patch.object(I, "_gap_voiced_runs", return_value=[]):
            I._reply_gap_speech_onset()
        skip = float(config.GAP_SPEECH_POST_PLAYBACK_SKIP_SECS)
        self.assertAlmostEqual(requested["secs"], self.now - (1998.5 + skip), places=2)

    def test_a_stale_overlong_gap_is_skipped(self):
        I._gap_watch_started_at = self.now - 20.0
        self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)]))

    def test_kill_switch(self):
        with mock.patch.object(config, "GAP_SPEECH_RECOVERY_ENABLED", False):
            self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)]))
        with mock.patch.object(config, "GAP_MERGE_ENABLED", False):
            self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)]))

    def test_text_only_mode_is_inert(self):
        with mock.patch.object(I, "_text_only_mode", True):
            self.assertIsNone(self._onset(runs=[(1998.0, 1999.1)]))


class MergeGapSpeechTests(unittest.TestCase):
    """Phase-1 capture: waits out the person, transcribes, guards — and a dry
    merge returns None so the caller regenerates rather than going silent."""

    def setUp(self):
        _reset_gap_state()
        self.addCleanup(_reset_gap_state)
        I._listen_capture_floor_at = 0.0
        self.addCleanup(lambda: setattr(I, "_listen_capture_floor_at", 0.0))

    def _merge(self, *, audio, transcript, non_speech=False, echo=False):
        with mock.patch.object(I, "_accumulate_speech", return_value=audio), \
             mock.patch.object(I.transcription, "transcribe",
                               return_value=transcript), \
             mock.patch.object(I, "_is_non_speech_vocalization",
                               return_value=non_speech), \
             mock.patch.object(I, "_looks_like_own_echo", return_value=echo), \
             mock.patch.object(I.vad, "reset_state"):
            return I._merge_gap_speech(1998.0, 1997.0)

    def test_happy_path_returns_the_line(self):
        got = self._merge(audio=np.ones(SR, dtype=np.float32),
                          transcript="and also I got a new job")
        self.assertEqual(got, "and also I got a new job")

    def test_merge_raises_the_capture_floor_to_the_watermark(self):
        # Preroll must not reach back past the already-consumed line-1 audio
        # and re-swallow its tail into the merged transcript.
        self._merge(audio=np.ones(SR, dtype=np.float32), transcript="more words")
        self.assertGreaterEqual(I._listen_capture_floor_at, 1997.0)

    def test_merge_rearms_the_watch(self):
        self._merge(audio=np.ones(SR, dtype=np.float32), transcript="more words")
        self.assertGreater(I._gap_watch_started_at, 0.0)

    def test_empty_capture_is_a_dry_merge(self):
        self.assertIsNone(self._merge(audio=None, transcript="x"))
        self.assertIsNone(
            self._merge(audio=np.zeros(0, dtype=np.float32), transcript="x"))

    def test_empty_transcript_is_a_dry_merge(self):
        self.assertIsNone(
            self._merge(audio=np.ones(SR, dtype=np.float32), transcript="  "))

    def test_non_speech_vocalization_is_a_dry_merge(self):
        self.assertIsNone(self._merge(audio=np.ones(SR, dtype=np.float32),
                                      transcript="hmm", non_speech=True))

    def test_own_echo_is_a_dry_merge(self):
        # Rex's residual transcribing his own just-drafted words must never
        # become the "second line".
        self.assertIsNone(self._merge(audio=np.ones(SR, dtype=np.float32),
                                      transcript="something in my way", echo=True))


class StreamGapCheckTests(unittest.TestCase):
    """The streaming reply path unwinds via _GapSpeechDetected BEFORE anything
    is spoken or queued — and the generic stream-error handler must re-raise it
    rather than speaking the abandoned draft via the fallback."""

    def _run_stream(self, *, gap_onset, gap_check_enabled, sentences):
        enqueued = []

        def _enqueue(text, *a, **k):
            enqueued.append(text)
            done = threading.Event()
            done.set()
            return done

        done_evt = threading.Event()
        done_evt.set()
        filler = threading.Event()
        with mock.patch.object(I, "_reply_token_stream",
                               return_value=iter(sentences)), \
             mock.patch.object(I, "_reply_gap_speech_onset",
                               return_value=gap_onset), \
             mock.patch.object(I, "_prepare_stream_sentence",
                               side_effect=lambda s, f, c: s), \
             mock.patch.object(I.speech_queue, "enqueue", side_effect=_enqueue), \
             mock.patch.object(I.empathy, "get_delivery_overrides",
                               return_value=None), \
             mock.patch.object(I.conv_log, "log_rex_stream"), \
             mock.patch.object(I.conv_log, "log_rex"), \
             mock.patch.object(I.conv_log, "finish_rex_stream"), \
             mock.patch.object(I, "_apply_post_tts_handoff"), \
             mock.patch.object(config, "SELF_EMOTION_CLASSIFY_ENABLED", False), \
             mock.patch.object(config, "POST_PUNCHLINE_BEAT_MS_MAX", 0), \
             mock.patch.object(config, "LLM_STREAMING_MIN_SENTENCE_CHARS", 1):
            text = I._stream_and_speak_sentences(
                "line one", None, None, None, "", {"value": False}, None,
                None, filler, two_chunk=False,
                gap_check_enabled=gap_check_enabled,
            )
        return text, enqueued

    def test_gap_speech_unwinds_before_anything_is_queued(self):
        I._gap_watch_started_at = 500.0
        self.addCleanup(_reset_gap_state)
        with self.assertRaises(I._GapSpeechDetected):
            self._run_stream(gap_onset=501.0, gap_check_enabled=True,
                             sentences=["Hello there my friend. "])

    def test_no_gap_speech_streams_normally(self):
        text, enqueued = self._run_stream(
            gap_onset=None, gap_check_enabled=True,
            sentences=["Hello there my friend. "])
        self.assertIn("Hello there my friend.", text)
        self.assertEqual(len(enqueued), 1)

    def test_disabled_check_never_scans(self):
        with mock.patch.object(I, "_reply_gap_speech_onset") as scan:
            text, enqueued = self._run_stream(
                gap_onset=None, gap_check_enabled=False,
                sentences=["Hello there my friend. "])
            self.assertFalse(scan.called)
        self.assertEqual(len(enqueued), 1)

    def test_exception_carries_onset_and_watermark(self):
        I._gap_watch_started_at = 500.0
        self.addCleanup(_reset_gap_state)
        try:
            self._run_stream(gap_onset=501.0, gap_check_enabled=True,
                             sentences=["Hello there my friend. "])
        except I._GapSpeechDetected as gap:
            self.assertAlmostEqual(gap.onset_at, 501.0)
            self.assertAlmostEqual(gap.armed_at, 500.0)
        else:
            self.fail("expected _GapSpeechDetected")


class CatchUpTests(unittest.TestCase):
    """Phase-2: the one-shot blind-span sweep at loop resume."""

    def setUp(self):
        _reset_gap_state()
        self.addCleanup(_reset_gap_state)
        self.now = 3000.0

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


if __name__ == "__main__":
    unittest.main()
