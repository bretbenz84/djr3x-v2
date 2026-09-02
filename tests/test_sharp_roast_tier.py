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

    def test_a_short_turn_is_not_a_soft_turn(self):
        # Owner directive 2026-09-02: brevity is length, not intensity. A micro or
        # brief plan used to cap the roast at "light" (00:30:21 that night: the
        # owner's one-line answer lost its self-own and became "Tell me more.").
        self.assertEqual(_level(0.95, target="micro"), "sharp")
        self.assertEqual(_level(0.95, target="brief"), "sharp")
        self.assertEqual(_level(0.5, target="brief"), "normal")

    def test_a_flat_arc_no_longer_pulls_punches_by_default(self):
        with mock.patch("intelligence.topic_thread.arc_reads_flat", return_value=True):
            self.assertEqual(_level(0.5), "normal")
            with mock.patch.object(sf.config, "ARC_EASES_ROAST_ON_FLOP", True):
                self.assertEqual(_level(0.5), "light")

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
    """The backstop runs at EVERY tier, incl. sharp — so it is CONTEMPT only.
    Owner directive 2026-09-02: "R3X needs to be able to insult people if it's
    funny." Insult vocabulary (idiot, pathetic, loser, disaster) is roast material
    the intensity tiers own; hatred and disgust are what no tier may say."""

    def test_insults_survive_at_sharp(self):
        res = sf.govern_response("You're a pathetic idiot. Nice work though.", _frame("sharp"))
        self.assertIn("idiot", res.text.lower())
        self.assertNotIn("removed_cruel_roast", res.notes)

    def test_insults_survive_at_normal(self):
        self.assertEqual(
            sf.govern_stream_sentence("Bold choice, you absolute disaster.", _frame("normal")),
            "Bold choice, you absolute disaster.",
        )

    def test_contempt_is_scrubbed_at_every_tier(self):
        for line in ("I hate you.", "Shut up, moron.", "You're a worthless loser.",
                     "You are a piece of garbage.", "You're a disgrace.",
                     "Honestly, you're ugly."):
            for tier in ("sharp", "normal", "light"):
                with self.subTest(line=line, tier=tier):
                    self.assertEqual(sf.govern_stream_sentence(line, _frame(tier)), "")

    def test_light_still_strips_the_harsh_words(self):
        # "light" now only comes from an explicit "likes light roasts" preference.
        self.assertEqual(sf.govern_stream_sentence("You're a pathetic idiot.", _frame("light")), "")

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



class SelfDirectedRoastTest(unittest.TestCase):
    """Field 2026-09-02 00:30:21: "Do you check the news while you're offline?" →
    "No, I just end up carrying it around in my head like an idiot with a
    subscription." The light-tier filter saw "idiot", dropped the only sentence,
    and Rex said "Tell me more." — so Bret had to ask again. A first-person line
    that never addresses a "you" is Rex roasting Rex."""

    FIELD = ("No, I just end up carrying it around in my head like an idiot "
             "with a subscription.")

    def test_field_answer_survives_the_light_tier(self):
        frame = _frame(allow_roast="light", purpose="answer")
        with mock.patch.object(sf, "_rex_last_line", return_value=""):
            got = sf.govern_response(self.FIELD, frame)
        self.assertEqual(got.text, self.FIELD)
        self.assertNotIn("fallback", got.notes)

    def test_field_answer_survives_the_no_roast_tier(self):
        frame = _frame(allow_roast="none", purpose="answer")
        with mock.patch.object(sf, "_rex_last_line", return_value=""):
            got = sf.govern_response(self.FIELD, frame)
        self.assertEqual(got.text, self.FIELD)

    def test_the_same_insult_aimed_at_the_human_is_still_removed(self):
        frame = _frame(allow_roast="light", purpose="answer")
        with mock.patch.object(sf, "_rex_last_line", return_value=""):
            got = sf.govern_response(
                "No, you just carry it around in your head like an idiot with a subscription.",
                frame,
            )
        self.assertIn("removed_sharp_roast", got.notes)

    def test_cruelty_backstop_is_contempt_and_never_self_directed(self):
        self.assertTrue(sf.contains_cruelty("Shut up, you worthless idiot."))
        self.assertFalse(sf.contains_cruelty("I'm a worthless pile of wiring today."))
        self.assertFalse(sf.contains_cruelty("You are such an idiot."))   # roast, not contempt

    def test_a_vocative_is_not_self_directed(self):
        # "genius" is aimed at the human even with no "you" in the sentence.
        self.assertFalse(sf._is_self_directed("I meant that, genius."))
        self.assertFalse(sf._is_self_directed("Correct, because sleep is when I\u2019m not doing the job, genius."))

    def test_leading_interjection_is_skipped(self):
        self.assertTrue(sf._is_self_directed("Well, I'm the idiot here."))
        self.assertTrue(sf._is_self_directed("Honestly, my wiring is a dumpster fire."))
        self.assertFalse(sf._is_self_directed("Well, the idiot here is you."))

    def test_dead_closures_are_not_exempt(self):
        frame = _frame(allow_roast="none", purpose="closure")
        self.assertTrue(sf._is_roast_sentence("I can't say I enjoyed that."))

    def test_streaming_sentence_path_agrees(self):
        frame = _frame(allow_roast="light", purpose="answer")
        self.assertEqual(sf.govern_stream_sentence(self.FIELD, frame), self.FIELD)

if __name__ == "__main__":
    unittest.main()
