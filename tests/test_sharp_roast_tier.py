"""Sharp roast tier — the warmth-earned fourth antagonism level (docs §1 roast lane).

The roast governor caps intensity at none/light/normal; "sharp" is a hotter tier unlocked
ONLY by earned warmth (a close, needling friend — never strangers, minors, or sincere/sad
turns). The cap lift sharpens the PROMPT; the cruelty backstop (_CRUEL_ROAST_PAT) runs at
EVERY tier so genuinely cruel name-calling is still scrubbed, even at "sharp".
"""

import unittest
from unittest import mock

from intelligence import social_frame as sf


def _frame(allow_roast="sharp", purpose="banter", max_words=60, max_sentences=4):
    return sf.SocialFrame(
        addressee="Bret",
        purpose=purpose,
        max_words=max_words,
        max_sentences=max_sentences,
        allow_question=True,
        allow_roast=allow_roast,
        allow_visual_comment=True,
        reason="test",
    )


def _level(effective_warmth, *, target="default", empathy_mode="default",
           affect="neutral", sensitivity="none", user_text="I rewired the whole base today"):
    """_roast_level on an otherwise-'normal' turn (passes every care/boundary gate);
    person_id=None keeps it DB-free, and effective_warmth (the gate input) is passed in."""
    return sf._roast_level(None, target, empathy_mode, affect, sensitivity, user_text,
                           effective_warmth=effective_warmth)


class RoastLevelSharpTierTest(unittest.TestCase):
    def test_earned_warmth_lifts_to_sharp(self):
        self.assertEqual(_level(0.9), "sharp")            # >= 0.85 default threshold

    def test_below_threshold_stays_normal(self):
        self.assertEqual(_level(0.5), "normal")
        self.assertEqual(_level(0.0), "normal")           # strangers / no-id never sharp

    def test_boundary_beats_warmth(self):
        self.assertEqual(_level(0.95, user_text="I'll be quiet"), "none")

    def test_tender_mode_beats_warmth(self):
        self.assertEqual(_level(0.95, empathy_mode="support"), "none")

    def test_sad_affect_beats_warmth(self):
        self.assertEqual(_level(0.95, affect="sad"), "none")

    def test_heavy_sensitivity_beats_warmth(self):
        self.assertEqual(_level(0.95, sensitivity="heavy"), "none")

    def test_micro_target_caps_at_light_not_sharp(self):
        self.assertEqual(_level(0.95, target="micro"), "light")

    def test_kill_switch_disables_sharp(self):
        with mock.patch.object(sf.config, "SHARP_ROAST_TIER_ENABLED", False):
            self.assertEqual(_level(0.9), "normal")

    def test_threshold_is_configurable(self):
        with mock.patch.object(sf.config, "ANTAGONISM_TIER_CAPS_LIFT_WARMTH", 0.70):
            self.assertEqual(_level(0.72), "sharp")


class EffectiveWarmthTest(unittest.TestCase):
    def _warmth(self, person, *, minor=False):
        with mock.patch.object(sf.people_memory, "get_person", return_value=person), \
             mock.patch("intelligence.profile_questions.person_is_minor", return_value=minor):
            return sf._effective_warmth(1)

    def test_best_friend_floor_qualifies(self):
        # best_friend floor is 0.90, so even low raw warmth floors up past the threshold.
        w = self._warmth({"id": 1, "warmth_score": 0.1, "friendship_tier": "best_friend"})
        self.assertGreaterEqual(w, 0.85)

    def test_stranger_is_below_threshold(self):
        w = self._warmth({"id": 1, "warmth_score": 0.0, "friendship_tier": "stranger"})
        self.assertLess(w, 0.85)

    def test_raw_warmth_used_when_above_floor(self):
        w = self._warmth({"id": 1, "warmth_score": 0.88, "friendship_tier": "friend"})
        self.assertAlmostEqual(w, 0.88)

    def test_minor_scores_zero_even_when_warm(self):
        w = self._warmth({"id": 1, "warmth_score": 0.99, "friendship_tier": "best_friend"}, minor=True)
        self.assertEqual(w, 0.0)

    def test_none_person_id_is_zero(self):
        self.assertEqual(sf._effective_warmth(None), 0.0)


class CrueltyBackstopAtSharpTest(unittest.TestCase):
    """The harsh-word/cruelty governor must hold at EVERY tier, incl. sharp."""

    def test_name_calling_scrubbed_even_at_sharp(self):
        res = sf.govern_response("You're a pathetic idiot. Nice work though.", _frame("sharp"))
        self.assertNotIn("idiot", res.text.lower())
        self.assertNotIn("pathetic", res.text.lower())
        self.assertIn("removed_cruel_roast", res.notes)

    def test_stream_scrubs_cruelty_at_sharp(self):
        self.assertEqual(sf.govern_stream_sentence("You're a worthless loser.", _frame("sharp")), "")

    def test_cruelty_scrubbed_at_normal_too(self):
        # Net safety improvement: 'normal' had NO harsh scrub before; now it does.
        self.assertEqual(sf.govern_stream_sentence("Shut up, moron.", _frame("normal")), "")

    def test_sharp_lets_a_vivid_rib_through_that_light_drops(self):
        # The whole point of the lift: a vivid, affectionate sharp rib survives at "sharp"
        # but is scrubbed at "light" (it trips the harsh-word filter). Proves the cap moved.
        line = "Buddy, that code is a dumpster fire."
        self.assertEqual(sf.govern_stream_sentence(line, _frame("light")), "")
        self.assertIn("dumpster fire", sf.govern_stream_sentence(line, _frame("sharp")).lower())


class SharpDirectiveTextTest(unittest.TestCase):
    def test_slim_contract_emits_the_sharp_rule(self):
        c = sf.render_slim_contract(_frame("sharp", purpose="banter"))
        self.assertIn("surgical", c.lower())

    def test_sharp_still_engages_first_on_a_sincere_share(self):
        # On an interest/answer_ack turn, even sharp leads with curiosity (no sincere-mock).
        c = sf.render_slim_contract(_frame("sharp", purpose="interest"))
        self.assertIn("curiosity", c.lower())


if __name__ == "__main__":
    unittest.main()
