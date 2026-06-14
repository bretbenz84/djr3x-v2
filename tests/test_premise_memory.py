"""
Premise-level anti-repeat for the main conversational reply path. Rex landed the same
"nature reminds you who's the boss" premise three times in one chat (twice back-to-back)
because the only anti-repeat on llm.stream was verbatim. premise_memory tracks the
salient CONTENT of his recent lines so the prompt can name the premises he's spent.
"""

from __future__ import annotations

import unittest

import config
from intelligence import premise_memory as pm


class PremiseKeywordTest(unittest.TestCase):
    def setUp(self):
        pm.clear()

    def tearDown(self):
        pm.clear()

    def test_stopwords_and_fillers_dropped(self):
        self.assertEqual(pm._keywords("Well, you know, it is what it is."), set())

    def test_contractions_do_not_leak_apostrophe_artifacts(self):
        kws = pm._keywords("Nature has a way of reminding us who's really in charge, doesn't it?")
        self.assertIn("nature", kws)
        self.assertIn("charge", kws)
        # No apostrophe roots like "who'" / "doesn".
        self.assertFalse(any("'" in k for k in kws))
        self.assertNotIn("doesn", kws)

    def test_light_stem_unifies_remind_forms(self):
        self.assertEqual(pm._stem("reminding"), pm._stem("reminds"))
        self.assertEqual(pm._stem("reminds"), "remind")


class AvoidDirectiveTest(unittest.TestCase):
    def setUp(self):
        pm.clear()

    def tearDown(self):
        pm.clear()

    def test_below_min_lines_is_silent(self):
        pm.note_line("Nature reminds you convenience is overrated.")
        # Only one line -> below PREMISE_ANTIREPEAT_MIN_LINES, no nudge yet.
        self.assertEqual(pm.build_avoid_directive(), "")

    def test_catches_the_field_repeat(self):
        # The actual sequence from the 02:42 log.
        pm.note_line("Nothing like a little nature to remind you that modern convenience is overrated.")
        pm.note_line("What was the highlight of your camping trip?")
        pm.note_line("Nature has a way of reminding us who's really in charge, doesn't it?")
        avoid = pm.recent_keywords()
        self.assertIn("nature", avoid)   # recurs across the window
        self.assertIn("remind", avoid)   # recurs (reminding/remind unified)
        directive = pm.build_avoid_directive()
        self.assertIn("nature", directive)
        self.assertIn("Do NOT reuse the same premise", directive)

    def test_back_to_back_repeat_is_flagged_from_last_line(self):
        pm.note_line("Let's talk about your camera setup.")
        pm.note_line("Nature has a way of reminding us who's really in charge.")
        # Next turn: the just-spoken line's keywords are in the avoid set even before
        # any cross-window recurrence, so an immediate re-roast of it is discouraged.
        avoid = pm.recent_keywords()
        self.assertIn("nature", avoid)
        self.assertIn("charge", avoid)

    def test_distinct_lines_do_not_inherit_old_premises(self):
        pm.note_line("Nature reminds you who's the boss.")
        pm.note_line("So how did your astrophotography photos turn out?")
        pm.note_line("That irony is delicious — capturing stars from a DJ booth.")
        avoid = pm.recent_keywords()
        # "nature"/"boss" fell out of the recent window's last line and never recurred.
        self.assertNotIn("nature", avoid)
        self.assertNotIn("boss", avoid)

    def test_disabled_flag_silences_it(self):
        pm.note_line("Nature reminds you convenience is overrated.")
        pm.note_line("Nature is clearly in charge here.")
        original = config.PREMISE_ANTIREPEAT_ENABLED
        try:
            config.PREMISE_ANTIREPEAT_ENABLED = False
            self.assertEqual(pm.build_avoid_directive(), "")
        finally:
            config.PREMISE_ANTIREPEAT_ENABLED = original

    def test_keyword_cap_is_respected(self):
        pm.note_line("Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo.")
        pm.note_line("Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo.")
        self.assertLessEqual(len(pm.recent_keywords()), config.PREMISE_ANTIREPEAT_MAX_KEYWORDS)

    def test_clear_resets(self):
        pm.note_line("Nature reminds you who's the boss.")
        pm.note_line("Nature is clearly in charge.")
        pm.clear()
        self.assertEqual(pm.recent_keywords(), [])
        self.assertEqual(pm.build_avoid_directive(), "")


if __name__ == "__main__":
    unittest.main()
