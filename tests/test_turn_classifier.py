"""
Tests for Bet 3's turn classifier: a single local-LLM structured read of a user
turn. The local LLM is mocked, so these make no real Ollama calls.
"""

from __future__ import annotations

import unittest
from unittest import mock

_GOOD = (
    "Topic: astrophotography\n"
    "Engagement: engaged\n"
    "Intent: share\n"
    "Sentiment: positive\n"
    "Pivot: no\n"
    "Addressee: rex"
)


class ParseTest(unittest.TestCase):
    def test_parse_well_formed(self):
        from intelligence import turn_classifier as tc
        c = tc.parse(_GOOD)
        self.assertIsNotNone(c)
        self.assertEqual(c.topic, "astrophotography")
        self.assertEqual(c.engagement, "engaged")
        self.assertEqual(c.intent, "share")
        self.assertEqual(c.sentiment, "positive")
        self.assertFalse(c.wants_pivot)
        self.assertEqual(c.addressee, "rex")

    def test_parse_validates_enums_to_defaults(self):
        from intelligence import turn_classifier as tc
        c = tc.parse(
            "Topic: -\nEngagement: super-duper\nIntent: vibing\n"
            "Sentiment: meh\nPivot: maybe\nAddressee: the wall"
        )
        self.assertIsNotNone(c)
        self.assertEqual(c.topic, "")              # "-" → empty
        self.assertEqual(c.engagement, "neutral")  # invalid → default
        self.assertEqual(c.intent, "other")
        self.assertEqual(c.sentiment, "neutral")
        self.assertFalse(c.wants_pivot)            # "maybe" → not yes
        self.assertEqual(c.addressee, "unclear")

    def test_parse_pivot_yes_and_embedded_enum(self):
        from intelligence import turn_classifier as tc
        c = tc.parse(
            "Topic: work stuff\nEngagement: low energy\nIntent: answer (short)\n"
            "Sentiment: negative\nPivot: yes, please\nAddressee: rex"
        )
        self.assertEqual(c.engagement, "low")      # picked from "low energy"
        self.assertEqual(c.intent, "answer")       # picked from "answer (short)"
        self.assertTrue(c.wants_pivot)

    def test_parse_malformed_returns_none(self):
        from intelligence import turn_classifier as tc
        self.assertIsNone(tc.parse("I think you're asking me to classify something?"))
        self.assertIsNone(tc.parse(""))
        self.assertIsNone(tc.parse("   "))


class ClassifyTest(unittest.TestCase):
    def test_disabled_returns_none_without_calling(self):
        from intelligence import turn_classifier as tc
        with (
            mock.patch.object(tc, "enabled", return_value=False),
            mock.patch("intelligence.local_llm.generate") as gen,
        ):
            self.assertIsNone(tc.classify("astrophotography"))
            gen.assert_not_called()

    def test_enabled_parses_local_llm_output(self):
        from intelligence import turn_classifier as tc
        with (
            mock.patch.object(tc, "enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", return_value=_GOOD) as gen,
        ):
            c = tc.classify("mostly nebulae and galaxies", rex_last_line="What are you into?")
            self.assertIsNotNone(c)
            self.assertEqual(c.intent, "share")
            self.assertTrue(gen.called)
            # context line is included when rex_last_line is given
            prompt = gen.call_args.args[0] if gen.call_args.args else gen.call_args.kwargs.get("prompt", "")
            self.assertIn("Rex just said", prompt)

    def test_degrades_to_none_on_backend_error(self):
        from intelligence import turn_classifier as tc
        with (
            mock.patch.object(tc, "enabled", return_value=True),
            mock.patch("intelligence.local_llm.generate", side_effect=RuntimeError("ollama down")),
        ):
            self.assertIsNone(tc.classify("hello"))   # must not raise

    def test_empty_text_returns_none(self):
        from intelligence import turn_classifier as tc
        with mock.patch.object(tc, "enabled", return_value=True):
            self.assertIsNone(tc.classify("   "))


if __name__ == "__main__":
    unittest.main()
