"""
Tests for the "tone tracks the relationship" smaller win: a persistent
relationship-tone line derived from a person's warmth/antagonism/trust scores
(memory/people.py, each 0.0-1.0; defaults warmth=0.0, antagonism=0.0, trust=0.5).
"""

from __future__ import annotations

import unittest


def _person(**kw):
    base = {"warmth_score": 0.0, "antagonism_score": 0.0, "trust_score": 0.5, "name": "Bret"}
    base.update(kw)
    return base


class RelationshipToneTest(unittest.TestCase):
    def _tone(self, **scores):
        from intelligence import llm
        p = _person(**scores)
        return llm._relationship_tone_rule(p, p["name"])

    def test_new_neutral_person_gets_no_tone(self):
        # defaults (warmth 0.0, antagonism 0.0) -> no special tone
        self.assertEqual(self._tone(), "")

    def test_warm_friend_gets_affectionate_tone(self):
        out = self._tone(warmth_score=0.7, trust_score=0.7)
        self.assertIn("affectionate", out.lower())
        self.assertIn("warmth", out.lower())
        self.assertIn("Bret", out)

    def test_high_trust_warm_friend_mentions_trust(self):
        self.assertIn("trust", self._tone(warmth_score=0.7, trust_score=0.8).lower())

    def test_low_trust_warm_friend_omits_trust_clause(self):
        self.assertNotIn("trust", self._tone(warmth_score=0.7, trust_score=0.4).lower())

    def test_antagonist_gets_sharper_tone(self):
        out = self._tone(antagonism_score=0.6, warmth_score=0.1)
        self.assertIn("sharper", out.lower())
        self.assertIn("needle", out.lower())

    def test_mild_antagonism_below_threshold_is_neutral(self):
        self.assertEqual(self._tone(antagonism_score=0.2), "")

    def test_elevated_both_reads_as_sparring_when_antagonism_dominates(self):
        # both up but antagonism >= warmth -> sparring (sharper), not affectionate
        out = self._tone(warmth_score=0.5, antagonism_score=0.6)
        self.assertIn("sharper", out.lower())

    def test_warm_dominates_when_warmth_exceeds_antagonism(self):
        out = self._tone(warmth_score=0.7, antagonism_score=0.3)
        self.assertIn("affectionate", out.lower())

    def test_malformed_scores_are_safe(self):
        from intelligence import llm
        self.assertEqual(
            llm._relationship_tone_rule(
                {"warmth_score": None, "antagonism_score": "oops", "name": "X"}, "X"
            ),
            "",
        )

    def test_kill_switch_flag_exists_and_default_on(self):
        import config
        self.assertTrue(getattr(config, "RELATIONSHIP_TONE_ENABLED", False))

    # ── Tier-coupled warmth: a friendship TIER floors the warmth so Rex's actual
    # close friends get the warm tone even before the slow warmth_score catches up. ──

    def test_tier_floor_config_exists(self):
        import config
        floors = getattr(config, "RELATIONSHIP_TIER_WARMTH_FLOOR", None)
        self.assertIsInstance(floors, dict)
        self.assertGreaterEqual(float(floors.get("close_friend", 0.0)), 0.5)

    def test_friend_tier_gets_warm_tone_despite_low_raw_warmth(self):
        # raw warmth 0.1 would NOT cross the 0.5 warm gate on its own — the "friend"
        # tier floor (0.5) lifts it so a real friend reads warm.
        out = self._tone(warmth_score=0.1, friendship_tier="friend")
        self.assertIn("affectionate", out.lower())
        self.assertIn("warm", out.lower())

    def test_close_friend_gets_stronger_warm_tone_and_trust(self):
        out = self._tone(warmth_score=0.0, trust_score=0.4, friendship_tier="close_friend")
        self.assertIn("affectionate", out.lower())
        # Closeness earns candor even with a low trust_score.
        self.assertIn("trust", out.lower())
        self.assertIn("real ones", out.lower())

    def test_acquaintance_low_warmth_still_neutral(self):
        # acquaintance has no floor, so low raw warmth stays toneless.
        self.assertEqual(self._tone(warmth_score=0.1, friendship_tier="acquaintance"), "")

    def test_needling_close_friend_still_reads_as_sparring(self):
        # The antagonism guard uses RAW warmth/antagonism, so a needling close friend
        # gets the sparring tone, not the warm one (tier floor doesn't suppress it).
        out = self._tone(
            warmth_score=0.1, antagonism_score=0.6, friendship_tier="close_friend"
        )
        self.assertIn("sharper", out.lower())


if __name__ == "__main__":
    unittest.main()
