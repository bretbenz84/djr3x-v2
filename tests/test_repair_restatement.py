"""Misheard-repair echo/duplication fixes (field bug 2026-06-30).

Two defects produced "We'll get there — recalibrating. I watch a lot of Netflix specials.
We'll get there — recalibrating." after the user re-stated content:
  (A) the recovery-line dedup missed because the LLM rendered a curly apostrophe (U+2019)
      while the constant uses a straight one (U+0027) → the preamble was appended twice;
  (B) a bare "I said X" was treated as a mishear-correction and echoed instead of answered.
"""

import unittest

from intelligence import repair_moves as r


class RecoveryLineDedupTest(unittest.TestCase):
    def test_curly_apostrophe_recovery_line_is_recognized(self):
        # The exact field string used a curly apostrophe; the dedup must still catch it.
        self.assertTrue(r._contains_recovery_line("We’ll get there — recalibrating."))

    def test_straight_apostrophe_still_recognized(self):
        self.assertTrue(r._contains_recovery_line("We'll get there — recalibrating."))

    def test_unrelated_text_not_flagged(self):
        self.assertFalse(r._contains_recovery_line("Tell me about the festival."))


class BareRestatementTest(unittest.TestCase):
    def test_plain_restatements_reroute(self):
        for t in ["I said I watch a lot of Netflix specials",
                  "I meant the blue one",
                  "um, I said pizza",
                  "I said yes"]:
            self.assertTrue(r.is_bare_restatement(t), f"should re-route: {t!r}")

    def test_contrastive_corrections_stay_in_repair(self):
        for t in ["I said blues, not jazz",
                  "I said blue, not red",
                  "That's not what I said",
                  "no, I said Tuesday"]:
            self.assertFalse(r.is_bare_restatement(t), f"should NOT re-route: {t!r}")

    def test_empty_is_false(self):
        self.assertFalse(r.is_bare_restatement(""))
        self.assertFalse(r.is_bare_restatement(None))


if __name__ == "__main__":
    unittest.main()
