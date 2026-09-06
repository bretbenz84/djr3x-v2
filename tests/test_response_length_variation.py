"""
Response-length variation: the slim contract renders max_sentences as a CEILING with a
natural one-or-two-sentence guidance (not a hard target), so Rex's statement turns vary
instead of always landing exactly two sentences. Question turns still get room for a beat
plus the one question.
"""

from __future__ import annotations

import unittest

from intelligence import social_frame


def _frame(*, allow_question, max_sentences=2, max_words=36, purpose="default_conversational_turn"):
    return social_frame.SocialFrame(
        addressee="Bret",
        purpose=purpose,
        max_words=max_words,
        max_sentences=max_sentences,
        allow_question=allow_question,
        allow_roast="normal",
        allow_visual_comment=False,
        reason="test",
    )


class SlimLengthRuleTest(unittest.TestCase):
    def test_statement_turn_allows_natural_one_or_two_sentences(self):
        rule = social_frame._slim_length_rule(_frame(allow_question=False))
        low = rule.lower()
        self.assertIn("one or two natural sentences", low)
        self.assertIn("never pad", low)
        # It is NOT phrased as a flat "write 2 sentences" target.
        self.assertNotIn("max_sentences=2", low)

    def test_question_turn_leaves_room_for_the_question(self):
        rule = social_frame._slim_length_rule(_frame(allow_question=True))
        self.assertIn("question", rule.lower())

    def test_single_sentence_cap_lands_and_stops(self):
        rule = social_frame._slim_length_rule(_frame(allow_question=False, max_sentences=1))
        self.assertIn("one sentence", rule.lower())


class RenderSlimContractTest(unittest.TestCase):
    def test_keeps_max_words_token_for_budget_parser(self):
        # llm._max_tokens_for_agenda regexes `max_words=(\d+)` — it must survive.
        out = social_frame.render_slim_contract(
            _frame(allow_question=False, max_words=36),
            primary_purpose="Primary purpose: react.",
        )
        self.assertIn("max_words=36", out)

    def test_statement_contract_encourages_brevity(self):
        out = social_frame.render_slim_contract(
            _frame(allow_question=False),
            primary_purpose="Primary purpose: react.",
        )
        self.assertIn("one or two natural sentences", out.lower())

    def test_max_tokens_for_agenda_still_parses(self):
        from intelligence import llm
        out = social_frame.render_slim_contract(
            _frame(allow_question=False, max_words=36),
            primary_purpose="Primary purpose: react.",
        )
        # Sanity: the budget parser derives a token cap from the contract (non-default).
        self.assertGreater(llm._max_tokens_for_agenda(out), 0)


if __name__ == "__main__":
    unittest.main()
