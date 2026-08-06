"""
News stories: say WHEN, and say it SHORT (field 2026-08-06 00:28).

1. Rex offered "did you hear about the eclipse TODAY" — the event is August 12,
   six days out. The stored story was correctly dated in BOTH the headline
   ("Total solar eclipse viewing push for August 12, 2026") and the summary, and
   the model still said "today": a news frame implies immediacy, so a headline
   handed over bare gets announced as now. Same failure `_build_anticipation_
   prompt` already fixed for remembered events, so it gets the same cure —
   compute the delta in code and STATE it.

2. "Can you tell me more about that?" produced ~90 words with "(esa.int)" read
   aloud. The system prompt said "give the COMPLETE answer ... two to four
   sentences" while the user message said "three max" — contradictory length
   rules in one prompt. The news path now carries its OWN system contract.
"""

from __future__ import annotations

import unittest
from datetime import date
from unittest import mock

import config
from awareness import current_events


class StoryDateExtractionTests(unittest.TestCase):
    TODAY = date(2026, 8, 6)

    def _d(self, text):
        return current_events.story_event_date(text, today=self.TODAY)

    def test_the_field_story_parses(self):
        self.assertEqual(
            self._d("Total solar eclipse viewing push for August 12, 2026"),
            date(2026, 8, 12),
        )

    def test_every_format_the_real_feed_produces(self):
        for text, want in (
            ("ESA program for the eclipse on August 12, 2026", date(2026, 8, 12)),
            ("impact on the Moon on August 5, 2026", date(2026, 8, 5)),
            ("Season 3 finale airs August 9", date(2026, 8, 9)),
            ("premiere for Aug. 4", date(2026, 8, 4)),
            ("Event on 12 August 2026", date(2026, 8, 12)),
            ("Scheduled 2026-08-20", date(2026, 8, 20)),
            ("the 3rd of September 2026", date(2026, 9, 3)),
        ):
            with self.subTest(text=text):
                self.assertEqual(self._d(text), want)

    def test_a_month_PREFIX_word_is_not_a_month(self):
        # Adversarial review 2026-08-06, high severity: `(jan|...|dec)[a-z]*` made
        # any word with a month prefix into a month. On an otherwise undated story
        # this REPLACED the safe "you don't know when" hedge with a confident wrong
        # date — Rex would place today's flood four months out.
        for text in (
            "Officials declared 6 counties disaster areas.",
            "Investigators found 3 separate failures in the cooling loop.",
            "Shares declined 12 percent after the ruling.",
            "The telescope has 4 decades of archived observations.",
            "12 Marines rescued from the flooded base",
            "The album marks 25 years since the band's debut.",
            "Judge declines 4 of the 5 counts",
            "20 Januarys of data show the trend",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self._d(text))

    def test_a_bare_number_before_a_month_is_not_a_day(self):
        # Version/season/list numbers, not dates: "Pixel 9 August feature drop" is
        # not August 9th. Day-first needs an ordinal, an "of", or a trailing year.
        for text in (
            "Pixel 9 August feature drop lands",
            "Season 3 September premiere confirmed",
            "Top 10 August releases you missed",
            "Windows 11 October update fixes the bug",
            "Formula 1 March testing begins",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self._d(text))

    def test_day_first_forms_with_real_date_context_still_parse(self):
        for text, want in (
            ("Event on 12 August 2026", date(2026, 8, 12)),
            ("the 3rd of September 2026", date(2026, 9, 3)),
            ("on the 12th August", date(2026, 8, 12)),
        ):
            with self.subTest(text=text):
                self.assertEqual(self._d(text), want)

    def test_a_bogus_match_no_longer_suppresses_a_real_date(self):
        # Review find: the scan returned on the FIRST regex hit, so an invalid
        # candidate ("Star Trek marks 60 years" -> mar + 60) killed the real date
        # later in the same summary and produced the "no date" clause.
        self.assertEqual(
            self._d("Star Trek marks 60 years. Celebrations culminate on September 8, 2026."),
            date(2026, 9, 8),
        )

    def test_no_date_returns_none_rather_than_guessing(self):
        self.assertIsNone(self._d("AWS outage disrupts services"))
        self.assertIsNone(self._d(""))

    def test_a_bare_date_picks_the_nearest_year(self):
        # Read in late December, "January 3" means NEXT year, not eleven months ago.
        self.assertEqual(
            current_events.story_event_date("airs January 3", today=date(2026, 12, 28)),
            date(2027, 1, 3),
        )
        self.assertEqual(
            current_events.story_event_date("aired December 28", today=date(2027, 1, 3)),
            date(2026, 12, 28),
        )

    def test_a_bare_year_is_not_mistaken_for_a_date(self):
        self.assertIsNone(self._d("the first since 1905"))

    def test_garbage_never_raises(self):
        for junk in ("February 31, 2026", "Feb 99", None, 12345):
            with self.subTest(junk=junk):
                self.assertIsNone(current_events.story_event_date(junk, today=self.TODAY))


class TimingClauseTests(unittest.TestCase):
    TODAY = date(2026, 8, 6)

    def _clause(self, headline, summary=""):
        return current_events.story_timing_clause(
            {"headline": headline, "summary": summary}, today=self.TODAY
        )

    def test_a_future_event_is_marked_not_yet_happened(self):
        c = self._clause("Total solar eclipse viewing push for August 12, 2026")
        self.assertIn("August 12, 2026", c)
        self.assertIn("in 6 days", c)
        self.assertIn("has NOT happened yet", c)

    def test_the_reflexive_today_is_banned_explicitly(self):
        c = self._clause("eclipse on August 12, 2026")
        self.assertIn('do NOT call a future event "today"', c)

    def test_today_tomorrow_yesterday_are_named(self):
        self.assertIn("TODAY", self._clause("happening August 6, 2026"))
        self.assertIn("TOMORROW", self._clause("happening August 7, 2026"))
        self.assertIn("YESTERDAY", self._clause("happened August 5, 2026"))

    def test_a_past_event_is_marked_past(self):
        c = self._clause("launched July 23, 2026")
        self.assertIn("already past", c)

    def test_an_undated_story_still_blocks_today(self):
        # No date is not licence to say "now" — a cached story can be a day old.
        c = self._clause("AWS outage disrupts services", "A power issue in Virginia.")
        self.assertIn("Do NOT say", c)
        self.assertIn("today", c)

    def test_disabled_flag_silences_it(self):
        with mock.patch.object(config, "NEWS_TIMING_CLAUSE_ENABLED", False):
            self.assertEqual(self._clause("eclipse on August 12, 2026"), "")

    def test_multiple_dates_are_listed_not_guessed_between(self):
        # "Season 3 premiered July 23 and the finale airs August 9": the FIRST date
        # is the premiere, not the finale being discussed. Asserting one date
        # confidently would be worse than the vague "today" this whole fix removes.
        c = self._clause("Season 3 premiered July 23 and the finale airs August 9")
        self.assertIn("more than one date", c)
        self.assertIn("July 23, 2026 (14 days ago)", c)
        self.assertIn("August 9, 2026 (in 3 days)", c)
        self.assertIn("do NOT state a day at all", c)
        self.assertNotIn("the event this story describes is on", c)

    def test_a_single_date_still_gets_the_confident_clause(self):
        c = self._clause("eclipse on August 12, 2026")
        self.assertIn("the event this story describes is on", c)
        self.assertNotIn("more than one date", c)

    def test_a_bare_year_does_not_count_as_a_second_date(self):
        # "the first since 1905, on August 12, 2026" is ONE event date.
        c = self._clause("the first since 1905, on August 12, 2026")
        self.assertNotIn("more than one date", c)
        self.assertIn("August 12, 2026", c)

    def test_the_same_date_written_twice_is_one_date(self):
        c = self._clause("eclipse on August 12, 2026",
                         "the total solar eclipse on August 12, 2026 will be visible")
        self.assertNotIn("more than one date", c)

    def test_bad_input_is_safe(self):
        for junk in (None, "not a dict", 42, {}):
            with self.subTest(junk=junk):
                self.assertIsInstance(current_events.story_timing_clause(junk), str)


class InstructionWiringTests(unittest.TestCase):
    STORY = {
        "headline": "Total solar eclipse viewing push for August 12, 2026",
        "summary": "ESA announced a program for the total solar eclipse on August 12, 2026.",
        "interest_topic": "astrophotography",
    }

    def _instruction(self, **cues) -> str:
        from types import SimpleNamespace as NS
        from intelligence import lean_brain
        cap = {}

        def fake(client, **kw):
            cap["m"] = kw["messages"]
            return iter([NS(choices=[NS(delta=NS(content="PASS"))])])

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake):
            lean_brain.consider_initiating(1, transcript=[], **cues)
        return cap["m"][-1]["content"]

    def test_the_generic_news_offer_carries_timing(self):
        self.assertIn("TIMING:", self._instruction(news_story=dict(self.STORY, interest_topic=None)))

    def test_the_interest_news_offer_carries_timing(self):
        self.assertIn("TIMING:", self._instruction(news_story=self.STORY))

    def test_an_undated_story_still_gets_a_clause(self):
        story = {"headline": "AWS outage disrupts services", "summary": "A power issue."}
        self.assertIn("TIMING:", self._instruction(news_story=story))

    def test_a_broken_lookup_cannot_break_the_offer(self):
        from intelligence import lean_brain
        with mock.patch.object(current_events, "story_timing_clause",
                               side_effect=RuntimeError("boom")):
            self.assertEqual(lean_brain._story_timing(self.STORY), "")


class NewsDigestContractTests(unittest.TestCase):

    def _wrapper(self):
        from intelligence import interaction as I
        return I._compose_news_search_input("Can you tell me more?", {
            "headline": "Total solar eclipse viewing push for August 12, 2026",
            "summary": "ESA announced a program for August 12, 2026.",
        })

    def test_the_wrapper_grounds_story_and_timing(self):
        w = self._wrapper()
        self.assertIn("Total solar eclipse", w)
        self.assertIn("TIMING:", w)
        self.assertTrue(w.startswith("Can you tell me more?"))

    def test_length_rules_moved_to_the_system_contract(self):
        # Stating them in the user message while the SYSTEM prompt said "give the
        # COMPLETE answer" is what produced the ~90-word reply.
        self.assertNotIn("THREE short sentences MAXIMUM", self._wrapper())
        self.assertIn("THREE SHORT SENTENCES MAXIMUM", config.WEB_SEARCH_NEWS_DIGEST_ADDENDUM)

    def test_the_digest_contract_bans_what_the_field_answer_did(self):
        a = config.WEB_SEARCH_NEWS_DIGEST_ADDENDUM
        self.assertIn("45 words", a)
        self.assertIn("(esa.int)", a)                 # the literal artifact heard
        self.assertIn("never call a future event", a)
        self.assertIn("offering to fetch more", a)

    def test_the_general_contract_is_unchanged_for_other_searches(self):
        # "What's the capital of Peru" still gets the complete-answer contract.
        self.assertIn("COMPLETE answer", config.WEB_SEARCH_PERSONA_ADDENDUM)
        self.assertNotEqual(
            config.WEB_SEARCH_PERSONA_ADDENDUM, config.WEB_SEARCH_NEWS_DIGEST_ADDENDUM
        )


class SpokenCitationTests(unittest.TestCase):

    def test_a_parenthetical_domain_is_stripped_whatever_the_tld(self):
        from intelligence.web_search import strip_links
        for cite in ("(esa.int)", "(nasa.gov)", "(apnews.com)", "(bbc.co.uk)"):
            with self.subTest(cite=cite):
                out = strip_links(f"A dramatic reunion. {cite} If skies cooperate.")
                self.assertNotIn(cite, out)
                self.assertIn("dramatic reunion", out)
                self.assertIn("skies cooperate", out)

    def test_a_numeric_parenthetical_is_not_a_domain(self):
        # Review find: `[a-z0-9-]{2,}` as the final segment ate "(7.24)" out of a
        # magnitude reading. The TLD must be alphabetic.
        from intelligence.web_search import strip_links
        for keep in ("A tremor of (7.24) magnitude hit.",
                     "Rated (9.5) by critics.",
                     "The cost is (about $3.50) all in."):
            with self.subTest(keep=keep):
                self.assertEqual(strip_links(keep), keep)

    def test_a_bare_unparenthesised_domain_is_stripped(self):
        # Review find: the paren fix only closed one shape; "According to esa.int,"
        # was still read aloud. `.int` was also missing from the TLD list.
        from intelligence.web_search import strip_links
        self.assertEqual(
            strip_links("According to esa.int, the eclipse is August 12."),
            "The eclipse is August 12.",
        )
        self.assertNotIn("reuters.com", strip_links("Per reuters.com, shares fell."))
        self.assertNotIn("bbc.co.uk", strip_links("See bbc.co.uk for more."))

    def test_a_named_source_without_a_domain_is_kept(self):
        # "According to the ESA" is normal speech, not a citation artifact.
        from intelligence.web_search import strip_links
        line = "According to the ESA, the eclipse is August 12."
        self.assertEqual(strip_links(line), line)

    def test_ordinary_parentheticals_survive(self):
        from intelligence.web_search import strip_links
        for keep in ("He said (in a nice way) it worked.",
                     "The cost is (about $3.50) all in.",
                     "Season 3 (the good one) airs Sunday."):
            with self.subTest(keep=keep):
                self.assertEqual(strip_links(keep), keep)


class CondenseTests(unittest.TestCase):
    """Shortening is a second LLM call, never a truncation — owner: 'don't just
    cut it off'. A reply cut mid-sentence sounds like Rex being interrupted."""

    def test_a_short_answer_is_left_alone(self):
        from intelligence import web_search
        with mock.patch.object(web_search._client.chat.completions, "create") as create:
            self.assertIsNone(web_search._condense("Short and sweet.", 55))
        create.assert_not_called()

    def test_a_long_answer_is_rewritten_not_cut(self):
        from types import SimpleNamespace as NS
        from intelligence import web_search
        long_text = " ".join(["word"] * 200)
        resp = NS(choices=[NS(message=NS(content="A tight three sentence version."))])
        with mock.patch.object(web_search._client.chat.completions, "create",
                               return_value=resp):
            out = web_search._condense(long_text, 55)
        self.assertEqual(out, "A tight three sentence version.")
        self.assertFalse(out.startswith(long_text[:20]), "must be rewritten, not sliced")

    def test_slack_allows_mild_overshoot(self):
        from intelligence import web_search
        with mock.patch.object(web_search._client.chat.completions, "create") as create:
            self.assertIsNone(web_search._condense(" ".join(["w"] * 70), 55))  # 70 < 55*1.4
        create.assert_not_called()

    def test_a_failed_condense_keeps_the_original(self):
        from intelligence import web_search
        with mock.patch.object(web_search._client.chat.completions, "create",
                               side_effect=RuntimeError("api down")):
            self.assertIsNone(web_search._condense(" ".join(["w"] * 200), 55))

    def test_a_rewrite_that_is_still_too_long_is_rejected(self):
        # Review find: the check was `len(out) < len(original)`, so an 85-word
        # "condense" of a 90-word answer passed — which is the actual problem.
        from types import SimpleNamespace as NS
        from intelligence import web_search
        resp = NS(choices=[NS(message=NS(content=" ".join(["w"] * 85)))])
        with mock.patch.object(web_search._client.chat.completions, "create",
                               return_value=resp):
            self.assertIsNone(web_search._condense(" ".join(["w"] * 90), 55))

    def test_a_rewrite_that_lands_on_target_is_accepted(self):
        from types import SimpleNamespace as NS
        from intelligence import web_search
        resp = NS(choices=[NS(message=NS(content=" ".join(["w"] * 50)))])
        with mock.patch.object(web_search._client.chat.completions, "create",
                               return_value=resp):
            self.assertIsNotNone(web_search._condense(" ".join(["w"] * 200), 55))

    def test_a_longer_rewrite_is_rejected(self):
        from types import SimpleNamespace as NS
        from intelligence import web_search
        resp = NS(choices=[NS(message=NS(content=" ".join(["w"] * 500)))])
        with mock.patch.object(web_search._client.chat.completions, "create",
                               return_value=resp):
            self.assertIsNone(web_search._condense(" ".join(["w"] * 200), 55))


if __name__ == "__main__":
    unittest.main()
