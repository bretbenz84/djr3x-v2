"""
tests/test_pause_command.py — the verbal pause ("rex, pause" / "one sec, be
right back") that enters QUIET mode (wake word resumes). Pure parser tests.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence import command_parser as cp


class PauseParseTest(unittest.TestCase):
    def test_pause_phrases(self):
        for phrase in (
            "pause",
            "Pause please",
            "rex, pause",
            "Hey Rex, one sec, be right back",
            "rex give me a sec, I need to make a phone call",
            "rex, be right back",
            "be right back",
            "I'll be right back",
            "brb",
            "shut up for a sec",
            "quiet for a minute",
            "hold that thought",
            "rex, hold on",
            "hey rex hang on",
        ):
            self.assertTrue(cp.parse_pause_command(phrase), phrase)

    def test_non_pause_phrases(self):
        for phrase in (
            "pause the music",
            "pause music",
            "give me a sec",                 # unaddressed → soft deferral, not a hard pause
            "one sec",                       # unaddressed
            "hold on",                       # unaddressed
            "I told my mom I'd be right back with the laundry",
            "what's the news today",
            "the pause button in the GUI is broken",
            "he paused for dramatic effect",
        ):
            self.assertFalse(cp.parse_pause_command(phrase), phrase)

    def test_parse_routes_to_quiet_mode_with_brb_flavor(self):
        m = cp.parse("Hey Rex, one sec, be right back")
        self.assertIsNotNone(m)
        self.assertEqual(m.command_key, "quiet_mode")
        self.assertEqual(m.args.get("flavor"), "brb")

    def test_pause_music_still_routes_to_dj_stop(self):
        m = cp.parse("pause the music")
        self.assertIsNotNone(m)
        self.assertEqual(m.command_key, "dj_stop")


if __name__ == "__main__":
    unittest.main()
