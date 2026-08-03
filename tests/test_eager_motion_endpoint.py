"""Eager motion endpointing (latency work 2026-08-02).

The 0.85s SILENCE_TIMEOUT_SECS hold applies to every turn, including explicit
drive commands where waiting is pure dead time. At 0.35s of silence a probe
transcribes the segment-so-far; a COMPLETE motion command ends the turn early
and its transcript is reused. Everything else keeps stock endpointing.
"""

import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
