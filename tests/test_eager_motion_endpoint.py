"""Eager motion endpointing (latency work 2026-08-02).

The 0.85s SILENCE_TIMEOUT_SECS hold applies to every turn, including explicit
drive commands where waiting is pure dead time. At 0.35s of silence a probe
transcribes the segment-so-far; a COMPLETE motion command ends the turn early
and its transcript is reused. Everything else keeps stock endpointing.
"""

import unittest
from unittest import mock

import config
from intelligence import interaction as IX
from intelligence import motion_controller


class EagerMatchTest(unittest.TestCase):
    """_eager_motion_transcript_matches: what may end a turn early."""

    def _match(self, text, moving=False):
        with mock.patch.object(motion_controller, "is_moving", return_value=moving):
            return IX._eager_motion_transcript_matches(text)

    def test_complete_motion_commands_match(self):
        for text in (
            "Turn left.",
            "turn around",
            "please turn left some",
            "Back up four feet.",
            "move forward",
            "come here",
            "Turn to your left a little and then back up four feet.",
        ):
            self.assertTrue(self._match(text), text)

    def test_ordinary_speech_does_not_match(self):
        for text in (
            "What day is it?",
            "We're not going anymore.",
            "I hate Elon Musk.",
            "you didn't give me time to answer",
            "That actually interesting work. Good morning.",
        ):
            self.assertFalse(self._match(text), text)

    def test_trailing_connective_blocks_the_cut(self):
        # The person is mid-route — never cut "turn left and ..." early.
        for text in ("turn left and", "turn left and then", "move forward then,"):
            self.assertFalse(self._match(text), text)

    def test_bare_stop_matches_only_while_moving(self):
        self.assertTrue(self._match("stop", moving=True))
        self.assertFalse(self._match("stop", moving=False))


class EagerGateTest(unittest.TestCase):
    """_eager_motion_endpoint_enabled: robot-only, base-required, kill switch."""

    def test_disabled_without_base(self):
        with mock.patch.object(motion_controller, "available", return_value=False):
            self.assertFalse(IX._eager_motion_endpoint_enabled())

    def test_kill_switch(self):
        import config
        with mock.patch.object(motion_controller, "available", return_value=True), \
             mock.patch.object(config, "MOTION_EAGER_ENDPOINT_ENABLED", False, create=True):
            self.assertFalse(IX._eager_motion_endpoint_enabled())

    def test_requires_hardware_aec_by_default(self):
        from audio import hardware_aec
        with mock.patch.object(motion_controller, "available", return_value=True), \
             mock.patch.object(hardware_aec, "is_active", return_value=False):
            self.assertFalse(IX._eager_motion_endpoint_enabled())
        with mock.patch.object(motion_controller, "available", return_value=True), \
             mock.patch.object(hardware_aec, "is_active", return_value=True):
            self.assertTrue(IX._eager_motion_endpoint_enabled())


class EagerHandoffTest(unittest.TestCase):
    def test_pop_clears_the_transcript(self):
        IX._eager_endpoint_transcript = "turn left"
        self.assertEqual(IX._pop_eager_transcript(), "turn left")
        self.assertIsNone(IX._pop_eager_transcript())

    def test_process_audio_reuses_pretranscribed(self):
        import numpy as np
        from audio import transcription, speaker_id
        with mock.patch.object(transcription, "transcribe",
                               side_effect=AssertionError("re-decoded!")), \
             mock.patch.object(speaker_id, "rank_speakers", return_value=[]):
            text, *_ = IX._process_audio(
                np.zeros(1600, dtype=np.float32), pretranscribed="turn left"
            )
        self.assertEqual(text, "turn left")


class ProbeTranscriptReuseTest(unittest.TestCase):
    """Every robot log showed two identical "[transcription] backend=qwen3_asr" lines
    per turn: the eager probe decoded the utterance after 0.35s of silence, threw the
    decode away unless it was a drive command, and the turn decoded the same audio
    again 0.3s later (queued behind the probe on MLX_LOCK). When the turn ends on the
    normal timeout with nobody having spoken since the probe captured, its decode IS
    the turn's transcript."""

    def setUp(self):
        IX._eager_endpoint_transcript = None
        self.addCleanup(setattr, IX, "_eager_endpoint_transcript", None)

    def _probe(self, text, *, done=True):
        import threading
        ev = threading.Event()
        if done:
            ev.set()
        return {"matched": False, "transcript": None, "text": text, "done": ev}

    def test_finished_probe_decode_is_adopted(self):
        from audio.transcription import Transcript
        text = Transcript("Yes, we just talked about that.", confident=True,
                          backend="qwen3_asr")
        self.assertTrue(IX._adopt_probe_transcript(self._probe(text)))
        got = IX._pop_eager_transcript()
        self.assertEqual(str(got), "Yes, we just talked about that.")
        self.assertTrue(getattr(got, "confident", None),
                        "the Transcript object (with its trust flag) must be handed over, not a bare str")

    def test_empty_or_rejected_decode_is_not_adopted(self):
        # An echo/hallucination rejection comes back as Transcript("") — the turn
        # must decode itself with the full capture, exactly as before.
        self.assertFalse(IX._adopt_probe_transcript(self._probe("")))
        self.assertFalse(IX._adopt_probe_transcript(self._probe(None)))
        self.assertIsNone(IX._pop_eager_transcript())

    def test_probe_still_running_past_the_wait_is_skipped(self):
        with mock.patch.object(config, "MOTION_EAGER_ENDPOINT_REUSE_WAIT_SECS", 0.01, create=True):
            self.assertFalse(IX._adopt_probe_transcript(self._probe("late", done=False)))
        self.assertIsNone(IX._pop_eager_transcript())

    def test_kill_switch(self):
        with mock.patch.object(config, "MOTION_EAGER_ENDPOINT_REUSE_TRANSCRIPT", False, create=True):
            self.assertFalse(IX._adopt_probe_transcript(self._probe("turn left")))
        self.assertIsNone(IX._pop_eager_transcript())

    def test_no_probe_is_a_no_op(self):
        self.assertFalse(IX._adopt_probe_transcript(None))

    def test_accumulate_adopts_the_live_probe_on_the_normal_timeout(self):
        """End-to-end through the VAD loop: speech, then silence long enough for the
        probe to start and the normal timeout to fire → the turn carries the probe's
        decode and the segment is still returned."""
        import numpy as np
        import time as _time
        from audio.transcription import Transcript
        chunk = np.zeros(int(16000 * IX._CHUNK_SECS), dtype=np.float32)
        # Two speech chunks, then silence for good.
        vad_answers = iter([True, True] + [False] * 200)
        probes: list = []

        def _start_probe(_start):
            box = self._probe(Transcript("hello there", confident=True))
            probes.append(box)
            return box

        with mock.patch.object(IX.stream, "get_audio_chunk", return_value=chunk), \
             mock.patch.object(IX.vad, "is_speech", side_effect=lambda *_a: next(vad_answers)), \
             mock.patch.object(IX, "_chunk_for_vad", side_effect=lambda c: c), \
             mock.patch.object(IX.state_module, "get_state", return_value=IX.State.ACTIVE), \
             mock.patch.object(IX._situation_assessor, "set_vad_active"), \
             mock.patch.object(IX, "_eager_motion_endpoint_enabled", return_value=True), \
             mock.patch.object(IX, "_start_eager_motion_probe", side_effect=_start_probe), \
             mock.patch.object(IX, "_speech_capture_secs", return_value=1.0), \
             mock.patch.object(config, "SILENCE_TIMEOUT_SECS", 0.20), \
             mock.patch.object(config, "MIN_SPEECH_DURATION_SECS", 0.0, create=True), \
             mock.patch.object(config, "MOTION_EAGER_ENDPOINT_SILENCE_SECS", 0.08, create=True):
            seg = IX._accumulate_speech(_time.monotonic())
        self.assertIsNotNone(seg)
        self.assertEqual(len(probes), 1, "one probe per silence run")
        self.assertEqual(str(IX._pop_eager_transcript()), "hello there")

    def test_speech_after_the_probe_makes_it_stale(self):
        """Probe starts in a mid-sentence pause, the person keeps talking, the turn
        ends later: that first decode is missing words and must NOT be adopted."""
        import numpy as np
        import time as _time
        from audio.transcription import Transcript
        chunk = np.zeros(int(16000 * IX._CHUNK_SECS), dtype=np.float32)
        # speech, pause (probe fires), speech resumes, then silence for good.
        vad_answers = iter([True, True] + [False] * 5 + [True, True] + [False] * 200)
        probes: list = []

        def _start_probe(_start):
            box = self._probe(Transcript(f"partial {len(probes)}", confident=True))
            probes.append(box)
            return box

        with mock.patch.object(IX.stream, "get_audio_chunk", return_value=chunk), \
             mock.patch.object(IX.vad, "is_speech", side_effect=lambda *_a: next(vad_answers)), \
             mock.patch.object(IX, "_chunk_for_vad", side_effect=lambda c: c), \
             mock.patch.object(IX.state_module, "get_state", return_value=IX.State.ACTIVE), \
             mock.patch.object(IX._situation_assessor, "set_vad_active"), \
             mock.patch.object(IX, "_eager_motion_endpoint_enabled", return_value=True), \
             mock.patch.object(IX, "_start_eager_motion_probe", side_effect=_start_probe), \
             mock.patch.object(IX, "_speech_capture_secs", return_value=1.0), \
             mock.patch.object(config, "SILENCE_TIMEOUT_SECS", 0.20), \
             mock.patch.object(config, "MIN_SPEECH_DURATION_SECS", 0.0, create=True), \
             mock.patch.object(config, "MOTION_EAGER_ENDPOINT_SILENCE_SECS", 0.08, create=True):
            IX._accumulate_speech(_time.monotonic())
        self.assertEqual(len(probes), 2, "a second probe for the second silence run")
        # Only the SECOND probe (which heard everything) may be adopted.
        self.assertEqual(str(IX._pop_eager_transcript()), "partial 1")


class MidErrandStopTest(unittest.TestCase):
    """_errand_stop_demanded: a stop buried in a noisy multi-sentence segment
    must count as a stop while the base is driving.

    Field 2026-08-11: "Take your bone all the way down. Take your bone. Go.
    Stop. Stop looking for me. Stop. Just stop." full-matched nothing — the
    bare-stop regex needs the whole utterance — so it routed to conversation,
    Rex SAID "Stopping." and the come-here errand kept driving.
    """

    def test_the_field_utterance_demands_a_stop(self):
        self.assertTrue(IX._errand_stop_demanded(
            "Take your bone all the way down. Take your bone. Go. Stop. "
            "Stop looking for me. Stop. Just stop."
        ))

    def test_stop_shaped_sentences_match(self):
        for text in (
            "Stop.",
            "Just stop.",
            "stop moving",
            "Stop looking for me.",
            "Please stop searching.",
            "No, stop.",
            "Quit moving.",
            "Don't move.",
            "Stay right there.",
            "Okay that's great, stop moving.",
        ):
            self.assertTrue(IX._errand_stop_demanded(text), text)

    def test_conversational_stop_words_do_not_match(self):
        for text in (
            "I can't stop laughing.",
            "We should stop by the store later.",
            "The bus stop is around the corner.",
            "Stop me if you've heard this one.",
            "It never stops raining here.",
            "",
        ):
            self.assertFalse(IX._errand_stop_demanded(text), text)

    def test_eager_endpoint_matches_a_buried_stop_while_moving(self):
        with mock.patch.object(motion_controller, "is_moving", return_value=True):
            self.assertTrue(IX._eager_motion_transcript_matches(
                "Go. Stop. Stop looking for me."
            ))
        with mock.patch.object(motion_controller, "is_moving", return_value=False):
            self.assertFalse(IX._eager_motion_transcript_matches(
                "Go. Stop. Stop looking for me."
            ))


if __name__ == "__main__":
    unittest.main()
