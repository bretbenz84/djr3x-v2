"""
Delivery tags are not content — the bit ledger must not compare them.

Field 2026-08-05 (run 23:44:30): FIVE lull lines dropped consecutively as
"repeats a recent bit", on five unrelated subjects —

    [curious] Hey, did you hear AWS ... take half the internet with it?
    [curious] What's the most unreasonably good thing you've eaten lately?
    Fifth visit and the pillows still think they're in charge.
    [low] Since it's late, I'm curious what your brain does ...
    Short hair again—maximum efficiency, minimum apology.

`content_tokens` tokenized the inline "[curious]" delivery tag as a CONTENT word.
At 7 characters `curious` clears BIT_LEDGER_DISTINCTIVE_LEN, and that rule is
"one shared distinctive word = same bit" — so a single stored bit containing the
word "curious" silently blocked every future line tagged [curious], whatever it
was about. The hair and pillow lines were genuine repeats and must STILL be
blocked; the news offer and the food question were not, and were lost for free.

Each drop also benches its cue for LEAN_CUE_DROP_COOLDOWN_SECS, so a false
positive costs far more than the one line.
"""

from __future__ import annotations

import unittest

import config
from intelligence import bit_ledger


class TagStrippingTests(unittest.TestCase):

    def test_delivery_tags_are_not_content_tokens(self):
        toks = bit_ledger.content_tokens("[curious] Did you hear about the outage?")
        self.assertNotIn("curious", toks)
        self.assertIn("outage", toks)

    def test_every_tag_shape_is_stripped(self):
        for tagged, tag in (
            ("[curious] a line", "curious"),
            ("[low] a line", "low"),
            ("[laughs] a line", "laughs"),
            ("[deadpan] a line", "deadpan"),
        ):
            with self.subTest(tag=tag):
                self.assertNotIn(tag, bit_ledger.content_tokens(tagged))

    def test_the_same_tag_no_longer_makes_two_lines_the_same_bit(self):
        # The exact field pair — unrelated subjects, identical tag.
        news = bit_ledger.content_tokens(
            "[curious] Hey, did you hear AWS took half the internet with it?")
        food = bit_ledger.content_tokens(
            "[curious] What's the most unreasonably good thing you've eaten lately?")
        overlap = news & food
        self.assertEqual(
            overlap, set(),
            f"unrelated lines still share tokens {overlap} — they would collide again",
        )

    def test_real_content_overlap_still_registers(self):
        # The guard must keep working: same subject, different wording.
        a = bit_ledger.content_tokens("[curious] Short hair, maximum efficiency.")
        b = bit_ledger.content_tokens("That short hair is doing a lot of work.")
        self.assertIn("hair", a & b)
        self.assertIn("short", a & b)

    def test_a_tag_word_used_as_real_content_survives(self):
        # "curious" spoken in the sentence is genuine content; only the bracketed
        # delivery tag is metadata.
        toks = bit_ledger.content_tokens("I'm curious about your week.")
        self.assertIn("curious", toks)

    def test_tokens_are_unchanged_for_untagged_lines(self):
        line = "Fifth visit and the pillows still think they're in charge."
        self.assertIn("pillows", bit_ledger.content_tokens(line))
        self.assertIn("visit", bit_ledger.content_tokens(line))

    def test_distinctive_len_rule_is_what_made_this_bite(self):
        # Documents WHY a single tag was sufficient: one shared >=7-char token
        # short-circuits the check, so tag leakage was never a near-miss.
        self.assertGreaterEqual(len("curious"),
                                int(getattr(config, "BIT_LEDGER_DISTINCTIVE_LEN", 7)))


if __name__ == "__main__":
    unittest.main()
