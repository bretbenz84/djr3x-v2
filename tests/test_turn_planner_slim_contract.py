"""
Phase 1 ("Bet 2") — the slim per-turn contract. render_slim_contract collapses the
~40-segment, self-contradicting block into one compact contract built from the
structured SocialFrame, while preserving every per-turn decision (length, question,
roast, visual) and the machine-readable max_words token the LLM budget needs.
"""

from __future__ import annotations

import unittest

from intelligence import social_frame as sf
from intelligence import llm


def _frame(**kw):
    base = dict(
        addressee="Bret", purpose="react", max_words=36, max_sentences=2,
        allow_question=False, allow_roast="normal", allow_visual_comment=True,
        reason="test",
    )
    base.update(kw)
    return sf.SocialFrame(**base)


class SlimContractShapeTest(unittest.TestCase):
    def test_is_compact_and_carries_the_purpose(self):
        c = sf.render_slim_contract(_frame(), "Primary purpose: react to the camping comment.")
        self.assertLess(len(c.split()), 150, "slim contract must stay compact")
        self.assertIn("Primary purpose: react to the camping comment.", c)
        self.assertIn("max_words=36", c)  # machine-readable budget token preserved

    def test_default_purpose_when_absent(self):
        c = sf.render_slim_contract(_frame(), "")
        self.assertIn("Primary purpose:", c)
        self.assertIn("one specific", c)


class SlimContractDecisionsTest(unittest.TestCase):
    def test_question_permission_reflects_frame(self):
        self.assertIn("do NOT ask a question",
                      sf.render_slim_contract(_frame(allow_question=False), ""))
        self.assertIn("you may ask ONE question",
                      sf.render_slim_contract(_frame(allow_question=True), ""))

    def test_roast_levels(self):
        self.assertIn("no roasts", sf.render_slim_contract(_frame(allow_roast="none"), ""))
        self.assertIn("light", sf.render_slim_contract(_frame(allow_roast="light"), ""))
        self.assertIn("ONE sharp", sf.render_slim_contract(_frame(allow_roast="normal"), ""))

    def test_engage_first_on_sincere_share(self):
        c = sf.render_slim_contract(_frame(allow_roast="normal", purpose="interest"), "")
        self.assertIn("genuine, SPECIFIC curiosity", c)
        self.assertIn("never deflect a sincere share", c)

    def test_visual_permission(self):
        self.assertIn("GENUINELY see",
                      sf.render_slim_contract(_frame(allow_visual_comment=True), ""))
        self.assertIn("do not mention what you see",
                      sf.render_slim_contract(_frame(allow_visual_comment=False), ""))

    def test_joke_safety_always_present(self):
        c = sf.render_slim_contract(_frame(), "")
        self.assertIn("grief", c)
        self.assertIn("one joke shape", c)


class TokenBudgetTest(unittest.TestCase):
    def test_slim_contract_budget_from_max_words(self):
        c = sf.render_slim_contract(_frame(max_words=36), "")
        self.assertEqual(llm._max_tokens_for_agenda(c), int(36 * 1.7))

    def test_old_target_format_still_works(self):
        self.assertEqual(
            llm._max_tokens_for_agenda("Response length control:\n- Target: short"), 70)

    def test_empty_falls_back_to_default(self):
        self.assertEqual(llm._max_tokens_for_agenda(""), 150)


class ExtractPrimaryPurposeTest(unittest.TestCase):
    def test_extracts_the_single_purpose_line(self):
        from intelligence import interaction
        directive = (
            "Topic thread: keep continuity.\n"
            "Current topic: camping.\n"
            "Primary purpose: react to the camping comment with a specific beat.\n"
            "Comedy mode: friendly_roast."
        )
        self.assertEqual(
            interaction._extract_primary_purpose(directive),
            "Primary purpose: react to the camping comment with a specific beat.",
        )

    def test_returns_empty_when_absent(self):
        from intelligence import interaction
        self.assertEqual(interaction._extract_primary_purpose("no purpose here"), "")


class NoRoastBackhandedJabTest(unittest.TestCase):
    """Field 2026-08-27 13:37:05 — the contract said "Roast: no roasts or pointed
    teasing this turn" and the spoken reply was still "You're spared from your own
    bad timing for another minute." The prompt reached the model; the per-sentence
    no-roast scrubber is what failed to catch the backhanded-mercy shape."""

    FIELD_JAB = "You're spared from your own bad timing for another minute."
    FIELD_JAB_CURLY = "You’re spared from your own bad timing for another minute."

    def test_backhanded_mercy_is_a_roast_sentence(self):
        self.assertTrue(sf._is_roast_sentence(self.FIELD_JAB))
        self.assertTrue(sf._is_roast_sentence(self.FIELD_JAB_CURLY))

    def test_no_roast_stream_drops_it(self):
        frame = _frame(allow_roast="none", purpose="closure", allow_question=False)
        self.assertEqual(sf.govern_stream_sentence(self.FIELD_JAB, frame), "")
        self.assertEqual(sf.govern_stream_sentence(self.FIELD_JAB_CURLY, frame), "")

    def test_the_warm_beat_beside_it_survives(self):
        # Do not widen: only the jab goes, the acceptance beat stays.
        frame = _frame(allow_roast="none", purpose="companionable", allow_question=False)
        for line in ("Good.", "Stay as long as you want.", "I hear you."):
            self.assertEqual(sf.govern_stream_sentence(line, frame), line)

    def test_no_roast_contract_names_the_backhanded_shape(self):
        contract = sf.render_slim_contract(_frame(allow_roast="none"), "")
        self.assertIn("backhanded", contract.lower())
        self.assertIn("no roasts", contract.lower())

    def test_normal_and_sharp_tiers_are_untouched(self):
        # _ROAST_PAT is only consulted at the 'none' tier — a sharp turn keeps it.
        for level in ("normal", "sharp"):
            frame = _frame(allow_roast=level, allow_question=False)
            self.assertEqual(sf.govern_stream_sentence(self.FIELD_JAB, frame),
                             self.FIELD_JAB)


if __name__ == "__main__":
    unittest.main()
