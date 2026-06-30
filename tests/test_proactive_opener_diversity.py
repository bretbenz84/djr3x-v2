"""Deterministic opener-diversity guard for ambient proactive chatter (field bug 2026-06-30:
Rex opened several unprompted lines with "Good" — "Good… Good… Good…").

The soft 'vary your opener' prompt rule was ignored, so a hard backstop drops a low-stakes
proactive line (idle banter / celebration check-in) whose leading word repeats one of Rex's
last few finalized openers. Salient reactions, greetings, and replies are never touched.
"""

import unittest

from intelligence import comedy_modes as cm


class OpensLikeRecentTest(unittest.TestCase):
    def setUp(self):
        cm._RECENT_OPENERS.clear()

    def test_non_adjacent_repeat_is_caught_with_lookback(self):
        cm.note_spoken_line("Good; I'd hate to recalibrate around a dead operator.")
        cm.note_spoken_line("There it is. A smile.")
        # "Good" recurs after an intervening "There" line — caught at lookback>=3.
        self.assertTrue(cm.opens_like_recent("Good—friends and a drink, respectable.", lookback=3))

    def test_distinct_opener_passes(self):
        cm.note_spoken_line("Good; hi.")
        self.assertFalse(cm.opens_like_recent("Still operating, then—good.", lookback=3))

    def test_filler_is_stripped_before_comparing(self):
        cm.note_spoken_line("Good one.")
        self.assertTrue(cm.opens_like_recent("Oh, good grief.", lookback=2))  # "Oh," stripped

    def test_empty_text_is_false(self):
        cm.note_spoken_line("Good; hi.")
        self.assertFalse(cm.opens_like_recent("", lookback=3))


class ProactiveOpenerGuardTest(unittest.TestCase):
    def setUp(self):
        cm._RECENT_OPENERS.clear()
        cm.note_spoken_line("Good; hi.")

    def test_ambient_purpose_with_repeat_is_dropped(self):
        from intelligence import interaction as ix
        self.assertTrue(ix._proactive_opener_repeats("Good, again.", "idle_monologue"))

    def test_non_ambient_purpose_never_dropped(self):
        from intelligence import interaction as ix
        # A salient reaction purpose is not in the allowlist → never dropped on opener clash.
        self.assertFalse(ix._proactive_opener_repeats("Good, a dog!", "animal_reaction"))

    def test_kill_switch(self):
        from intelligence import interaction as ix
        import config
        prev = config.PROACTIVE_OPENER_DIVERSITY_GUARD
        try:
            config.PROACTIVE_OPENER_DIVERSITY_GUARD = False
            self.assertFalse(ix._proactive_opener_repeats("Good, again.", "idle_monologue"))
        finally:
            config.PROACTIVE_OPENER_DIVERSITY_GUARD = prev


if __name__ == "__main__":
    unittest.main()
