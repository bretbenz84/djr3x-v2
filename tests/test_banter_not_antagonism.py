"""
P3 — affectionate banter is not antagonism. In a warm, mutual-roast relationship a
playful jab-back shouldn't accrue antagonism the way a cold insult does, and it must
not pin a genuinely warm friend at the "friend" tier. apply_jab discounts the
antagonism by warmth (re-routing part to playfulness), and _compute_tier lifts the
antagonism cap once warmth is established.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from memory import people as pm


class ComputeTierWarmthReliefTest(unittest.TestCase):
    def test_antagonism_caps_a_cold_relationship(self):
        # familiarity in the close_friend band but high antagonism, no warmth -> capped.
        tier = pm._compute_tier(0.70, 0.25, 0.0)
        self.assertEqual(tier, "friend")  # 0.20 cap applies

    def test_warmth_relief_lifts_the_cap(self):
        # Same scores, but warmth at/above the relief threshold -> cap lifted.
        warm = config.ANTAGONISM_CAP_WARMTH_RELIEF
        tier = pm._compute_tier(0.70, 0.25, warm)
        self.assertEqual(tier, "close_friend")  # 0.60-0.85 familiarity band, uncapped

    def test_warmth_relief_reaches_best_friend(self):
        tier = pm._compute_tier(0.90, 0.50, 0.9)
        self.assertEqual(tier, "best_friend")

    def test_no_warmth_defaults_preserve_old_behavior(self):
        # Default warmth=0.0 keeps the original cap semantics for callers not passing it.
        self.assertEqual(pm._compute_tier(0.70, 0.45), "acquaintance")


class ApplyJabTest(unittest.TestCase):
    def test_cold_relationship_takes_full_antagonism(self):
        row = {"warmth_score": 0.1}  # below BANTER_WARMTH_THRESHOLD
        with mock.patch.object(pm.db, "fetchone", return_value=row), \
             mock.patch.object(pm, "update_relationship_scores") as upd:
            pm.apply_jab(5, "insult_mild")
        upd.assert_called_once_with(5, antagonism=config.RELATIONSHIP_INCREMENTS["insult_mild"][1])

    def test_warm_relationship_discounts_and_reroutes(self):
        full = config.RELATIONSHIP_INCREMENTS["insult_mild"][1]
        row = {"warmth_score": 1.0}  # max warmth -> max discount
        with mock.patch.object(pm.db, "fetchone", return_value=row), \
             mock.patch.object(pm, "update_relationship_scores") as upd:
            pm.apply_jab(5, "insult_mild")
        kwargs = upd.call_args.kwargs
        waived = full * config.BANTER_ANTAGONISM_DISCOUNT
        self.assertAlmostEqual(kwargs["antagonism"], full - waived, places=6)
        self.assertAlmostEqual(
            kwargs["playfulness"], waived * config.BANTER_PLAYFULNESS_SHARE, places=6
        )
        # A warm jab lands far less antagonism than a cold one.
        self.assertLess(kwargs["antagonism"], full)

    def test_discount_scales_with_warmth(self):
        full = config.RELATIONSHIP_INCREMENTS["insult_mild"][1]

        def antagonism_for(warmth):
            with mock.patch.object(pm.db, "fetchone", return_value={"warmth_score": warmth}), \
                 mock.patch.object(pm, "update_relationship_scores") as upd:
                pm.apply_jab(5, "insult_mild")
            return upd.call_args.kwargs["antagonism"]

        # More warmth -> less antagonism retained from the same jab.
        a_low = antagonism_for(0.35)
        a_high = antagonism_for(0.85)
        self.assertLess(a_high, a_low)
        self.assertLessEqual(a_low, full)

    def test_none_person_is_safe(self):
        pm.apply_jab(None, "insult_mild")  # must not raise


if __name__ == "__main__":
    unittest.main()
