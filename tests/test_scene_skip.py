"""
audio/scene.py must NOT analyze the mic while Rex is playing his OWN audio.

Regression for: Rex startling/laughing at his own DJ music. DJ/radio playback
holds NEITHER the speech queue nor the output_gate — only echo_cancel.set_playing(
True) — so the skip guard must also consult echo_cancel.is_suppressed(), else the
scene loop hears Rex's music and reports "music/laughter detected".
"""

import unittest
from unittest import mock

from audio import scene


class SceneSkipGuardTest(unittest.TestCase):
    def _skip(self, *, speaking=False, suppressed=False, since_release=999.0):
        with mock.patch.object(scene.speech_queue, "is_speaking", return_value=speaking), \
             mock.patch.object(scene.echo_cancel, "is_suppressed", return_value=suppressed), \
             mock.patch.object(scene.output_gate, "seconds_since_release", return_value=since_release):
            return scene._should_skip_cycle()

    def test_analyzes_room_when_quiet(self):
        self.assertFalse(self._skip())

    def test_skips_during_tts(self):
        self.assertTrue(self._skip(speaking=True))

    def test_skips_during_dj_music(self):
        # The regression: DJ playback only flips echo_cancel — must still skip.
        self.assertTrue(self._skip(suppressed=True))

    def test_skips_in_post_playback_tail(self):
        self.assertTrue(self._skip(since_release=0.0))


if __name__ == "__main__":
    unittest.main()
