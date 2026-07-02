"""Barge-yield onset recovery (field bug 2026-07-01, log 22-43-46).

When a proactive line yields because the user started talking during its ~1-2s generation gap, the
VAD loop was blocked through that gap and only notices the speech late — clipping the user's opening
words ("what are my weekend plans" → "weekend plans"). Rex wasn't playing during that window, so the
buffer holds the clean onset; the capture reaches back to it (bounded) to recover it.
"""

import unittest
from unittest import mock

import intelligence.interaction as I


class BargeRecoveredSpeechStartTest(unittest.TestCase):
    NOW = 1000.0

    def test_no_pending_marker_is_unchanged(self):
        self.assertEqual(I._barge_recovered_speech_start(self.NOW, 0.0), self.NOW)

    def test_reaches_back_to_onset(self):
        # User started 1.3s ago (during generation) → capture starts there, not "now".
        self.assertAlmostEqual(I._barge_recovered_speech_start(self.NOW, self.NOW - 1.3), self.NOW - 1.3)

    def test_clamped_to_max(self):
        # A stale/absurd onset can't reach further back than PROACTIVE_YIELD_ONSET_MAX_SECS.
        with mock.patch.object(I.config, "PROACTIVE_YIELD_ONSET_MAX_SECS", 3.0):
            self.assertAlmostEqual(
                I._barge_recovered_speech_start(self.NOW, self.NOW - 10.0), self.NOW - 3.0
            )

    def test_never_later_than_now(self):
        self.assertEqual(I._barge_recovered_speech_start(self.NOW, self.NOW + 5.0), self.NOW)

    def test_kill_switch_disables_recovery(self):
        with mock.patch.object(I.config, "PROACTIVE_YIELD_RECOVER_ONSET_ENABLED", False):
            self.assertEqual(I._barge_recovered_speech_start(self.NOW, self.NOW - 1.3), self.NOW)


if __name__ == "__main__":
    unittest.main()
