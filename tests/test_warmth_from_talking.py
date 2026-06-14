"""
P2 — warmth grows from talking. Previously warmth only rose from explicit compliments
(genuine_laughter and consistent_return_visit were defined in config but never wired),
so the friend score never reflected time spent. Now engaged turns + shared laughter
earn a small, CAPPED warmth bump at session end, and a genuine return visit warms Rex
up too.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import interaction
from memory import people as people_mod


class WarmthSignalDetectionTest(unittest.TestCase):
    def test_engaged_turn_needs_enough_words(self):
        self.assertTrue(interaction._text_is_engaged_turn("I went hiking this weekend"))
        self.assertFalse(interaction._text_is_engaged_turn("yeah"))
        self.assertFalse(interaction._text_is_engaged_turn("ok sure"))

    def test_laughter_detection(self):
        for t in ("hahaha", "lol that's great", "that's funny", "you got me", "😂"):
            self.assertTrue(interaction._text_is_genuine_laughter(t), t)
        for t in ("what time is it", "I like jazz", ""):
            self.assertFalse(interaction._text_is_genuine_laughter(t), t)


class WarmthTallyTest(unittest.TestCase):
    def setUp(self):
        interaction._session_warmth_signals.clear()

    def tearDown(self):
        interaction._session_warmth_signals.clear()

    def test_engaged_and_laugh_tally(self):
        interaction._note_warmth_from_talking(7, "I built a new synth this week and it sounds great")
        interaction._note_warmth_from_talking(7, "hahaha")  # short -> laugh only
        sig = interaction._session_warmth_signals[7]
        self.assertEqual(sig["engaged"], 1)
        self.assertEqual(sig["laughs"], 1)

    def test_insult_turn_is_not_engaged(self):
        interaction._note_warmth_from_talking(
            7, "you are such a useless pile of bolts", pre_classified_insult=True
        )
        self.assertEqual(interaction._session_warmth_signals[7]["engaged"], 0)


class WarmthAwardTest(unittest.TestCase):
    def setUp(self):
        interaction._session_warmth_signals.clear()

    def tearDown(self):
        interaction._session_warmth_signals.clear()

    def test_award_is_capped_per_session(self):
        # 50 engaged turns + 50 laughs should be capped, not 100x the per-unit delta.
        interaction._session_warmth_signals[7] = {"engaged": 50, "laughs": 50}
        with mock.patch.object(people_mod, "update_relationship_scores") as upd:
            interaction._award_warmth_from_talking(7)
        upd.assert_called_once()
        warmth = upd.call_args.kwargs["warmth"]
        engaged_cap = config.WARMTH_FROM_TALKING_MAX_ENGAGED_PER_SESSION
        laugh_cap = config.WARMTH_FROM_TALKING_MAX_LAUGHS_PER_SESSION
        expected = (
            config.RELATIONSHIP_INCREMENTS["engaged_turn"][1] * engaged_cap
            + config.RELATIONSHIP_INCREMENTS["genuine_laughter"][1] * laugh_cap
        )
        self.assertAlmostEqual(warmth, expected, places=6)

    def test_no_signals_no_award(self):
        with mock.patch.object(people_mod, "update_relationship_scores") as upd:
            interaction._award_warmth_from_talking(7)
        upd.assert_not_called()


class ReturnVisitWarmsUpTest(unittest.TestCase):
    """update_visit should warm + build trust on a genuine RETURN, but not the first sight."""

    def test_first_sight_only_familiarity(self):
        rows = {"visit_count": 0}
        with mock.patch.object(people_mod.db, "fetchone", return_value=rows), \
             mock.patch.object(people_mod.db, "execute"), \
             mock.patch.object(people_mod, "update_familiarity") as fam, \
             mock.patch.object(people_mod, "apply_relationship_increment") as inc:
            people_mod.update_visit(7)
        fam.assert_called_once()
        inc.assert_not_called()  # no relationship bump on the very first sighting

    def test_return_visit_warms_and_trusts(self):
        rows = {"visit_count": 3}
        with mock.patch.object(people_mod.db, "fetchone", return_value=rows), \
             mock.patch.object(people_mod.db, "execute"), \
             mock.patch.object(people_mod, "update_familiarity"), \
             mock.patch.object(people_mod, "apply_relationship_increment") as inc:
            people_mod.update_visit(7)
        kinds = {c.args[1] for c in inc.call_args_list}
        self.assertEqual(kinds, {"consistent_return_visit", "return_visit_warmth"})


class ConfigWiringTest(unittest.TestCase):
    def test_new_increments_present(self):
        for k in ("engaged_turn", "return_visit_warmth", "genuine_laughter", "consistent_return_visit"):
            self.assertIn(k, config.RELATIONSHIP_INCREMENTS)
        self.assertEqual(config.RELATIONSHIP_INCREMENTS["engaged_turn"][0], "warmth")
        self.assertEqual(config.RELATIONSHIP_INCREMENTS["return_visit_warmth"][0], "warmth")


if __name__ == "__main__":
    unittest.main()
