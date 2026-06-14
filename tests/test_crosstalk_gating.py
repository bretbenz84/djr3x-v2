"""
Tier 2 / item 5 — cross-talk gating. With Tier-1 single-visible attribution, an
overheard device/UI readout ("oh yeah that's definitely 30 FPS") would be pinned on
the visible person and answered/profiled. The strengthened detector marks clear,
non-Rex device readouts as background chatter so the reply path drops them.
"""

from __future__ import annotations

import unittest

from intelligence import interaction as I


class BackgroundCrosstalkDetectorTest(unittest.TestCase):
    def test_device_readouts_are_chatter(self):
        for text in ("oh yeah that's definitely 30 FPS", "that is 1080p",
                     "the frame rate looks choppy", "check the latency on that"):
            self.assertTrue(I._looks_like_background_crosstalk(text), text)

    def test_personal_or_directed_speech_is_not_chatter(self):
        for text in ("I'm at 100 percent today", "Hey Rex play some music",
                     "what do you think about that", "my favorite is 60fps gaming"):
            self.assertFalse(I._looks_like_background_crosstalk(text), text)

    def test_short_real_answers_are_not_chatter(self):
        for text in ("China", "Kebab", "yeah", "I really love astrophotography lately"):
            self.assertFalse(I._looks_like_background_crosstalk(text), text)

    def test_commands_are_never_chatter(self):
        # _speech_is_directed_to_rex (command_parser) short-circuits the detector.
        self.assertFalse(I._looks_like_background_crosstalk("shut down"))


if __name__ == "__main__":
    unittest.main()
