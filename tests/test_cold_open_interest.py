"""
Cold-open interest selection. Rex kept opening every startup with "what's the latest
scoop on your mint chocolate chip ice cream adventures?" — a static favorite mis-stored
as a hobby, picked top by recency with no anti-repeat. The cold-open now (a) excludes
consumable-favorite / junk interests from LEADING a greeting and (b) marks the chosen
interest asked so it rotates.
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import consciousness as c
from memory import interests as interests_mem
from memory import facts as facts_mem


class ColdOpenInterestWorthyTest(unittest.TestCase):
    def test_substantive_interests_lead(self):
        for n in ("3D printing", "astrophotography", "Star Trek", "Droid Development",
                  "camping", "programming", "hair styling", "dog training"):
            self.assertTrue(c._cold_open_interest_worthy(n), n)

    def test_favorites_and_junk_are_excluded(self):
        for n in ("mint chocolate chip ice cream", "my clothes", "you now",
                  "hang out in my bed", "be in there", "favorite snack", "coffee"):
            self.assertFalse(c._cold_open_interest_worthy(n), n)


class ColdOpenCandidateFilterTest(unittest.TestCase):
    def test_ice_cream_is_not_a_cold_open_candidate(self):
        hooks = [
            {"name": "mint chocolate chip ice cream", "last_mentioned_at": "2026-06-14"},
            {"name": "astrophotography", "last_mentioned_at": "2026-06-14"},
            {"name": "you now", "last_mentioned_at": "2026-06-13"},
        ]
        with mock.patch.object(interests_mem, "get_interest_hooks", return_value=hooks), \
             mock.patch.object(facts_mem, "get_prompt_worthy_facts", return_value=[]):
            cands = c._cold_open_callback_candidates(1)
        topics = {x["topic"] for x in cands}
        self.assertIn("astrophotography", topics)
        self.assertNotIn("mint chocolate chip ice cream", topics)
        self.assertNotIn("you now", topics)


if __name__ == "__main__":
    unittest.main()
