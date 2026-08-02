"""
Memory recall (field 2026-08-01 22:38): Bret quizzed Rex on things the DBs had
held for weeks — favorite movie, job, dog, camping, the movie watched the night
before — and Rex denied knowing ANY of it. Storage was fine; retrieval was the
failure: static topic-blind top-4 ranking, interests crowding facts out of the
unified budget, no query-time episodic recall, and ASR-shard junk ("for a job",
"lot", "d the movie") winning prompt slots. Pins all four fixes.
"""

import unittest
from unittest import mock

import config
from memory import fact_quality, recall, retrieval


class MemoryQuestionDetectorTest(unittest.TestCase):
    def test_field_run_questions_all_detected(self):
        for q in (
            "Did I tell you that I went camping?",
            "What movie did I watch last night?",
            "Can you tell me what I do for a job?",
            "What's my favorite food?",
            "What are my hobbies?",
            "Have I ever mentioned my partner?",
            "Do you know about me?",
            "Can you tell me a little bit about myself?",
            "What else do you know about me?",
            "Do you know when I went camping?",
        ):
            self.assertTrue(recall.is_memory_question(q), q)

    def test_ordinary_turns_not_detected(self):
        for q in (
            "Can you tell me about the news?",
            "I said I'm not going to move the couch anymore.",
            "My partner's name is JT.",
            "Play some music.",
            "That's all right. No thanks.",
        ):
            self.assertFalse(recall.is_memory_question(q), q)


class UtteranceTokensTest(unittest.TestCase):
    def test_question_machinery_stripped(self):
        # "did/last/night" made every episode containing "did" match the movie
        # question — three impersonation rows outranked the actual movie memory.
        tokens = recall.utterance_tokens("What movie did I watch last night?")
        self.assertEqual(tokens, {"movie", "watch"})

    def test_empty_and_garbage_safe(self):
        self.assertEqual(recall.utterance_tokens(""), set())
        self.assertEqual(recall.utterance_tokens("did you the a"), set())


class EpisodeSearchTest(unittest.TestCase):
    _ROWS = [
        {"id": 1, "created_at": "2026-07-31 03:00:00", "kind": "moment",
         "summary": "Bret told me he is going to watch 'the Odyssey' tonight.",
         "detail": "", "person_id": 1, "salience": 0.6},
        {"id": 2, "created_at": "2026-07-21 03:00:00", "kind": "moment",
         "summary": "I did an impersonation of Bret Benziger.",
         "detail": "", "person_id": 1, "salience": 0.5},
        {"id": 3, "created_at": "2026-07-20 03:00:00", "kind": "moment",
         "summary": "I did an impersonation of Bret Benziger.",
         "detail": "", "person_id": 1, "salience": 0.5},
        {"id": 4, "created_at": "2026-07-25 03:00:00", "kind": "scene",
         "summary": "A movie poster on the wall.", "detail": "",
         "person_id": None, "salience": 0.3},
    ]

    def _search(self, tokens):
        from memory import rex_db
        rows = [r for r in self._ROWS if r["kind"] != "scene"]
        with mock.patch.object(rex_db, "fetchall", return_value=rows):
            return recall.search_episodes(tokens, person_id=1, limit=3)

    def test_topic_match_finds_the_movie_episode(self):
        out = self._search({"movie", "watch"})
        self.assertTrue(out)
        self.assertIn("Odyssey", out[0]["summary"])

    def test_no_tokens_returns_nothing(self):
        self.assertEqual(recall.search_episodes(set(), person_id=1), [])

    def test_duplicate_summaries_collapse(self):
        out = self._search({"impersonation"})
        self.assertEqual(
            len([r for r in out if "impersonation" in r["summary"]]), 1
        )


class RichBlockTest(unittest.TestCase):
    def test_no_person_or_ordinary_turn_is_empty(self):
        self.assertEqual(recall.memory_question_lines(None, "What are my hobbies?"), [])
        self.assertEqual(recall.memory_question_lines(1, "Play some music."), [])

    def test_rich_block_carries_facts_interests_and_honesty_rule(self):
        with mock.patch.object(recall, "_fact_pairs",
                               return_value=["favorite movie: Mrs. Doubtfire"]), \
             mock.patch.object(recall, "_interest_pairs",
                               return_value=["camping — Mentioned going camping this summer."]), \
             mock.patch.object(recall, "_qa_pairs", return_value=[]), \
             mock.patch.object(recall, "_relationship_lines",
                               return_value=["their partner: JT"]), \
             mock.patch.object(recall, "search_episodes", return_value=[]):
            lines = recall.memory_question_lines(1, "What do you know about me?")
        joined = " ".join(lines)
        self.assertIn("MEMORY QUESTION", joined)
        self.assertIn("Mrs. Doubtfire", joined)
        self.assertIn("camping this summer", joined)
        self.assertIn("their partner: JT", joined)
        self.assertIn("never invent a memory", joined)

    def test_all_silos_empty_yields_no_block(self):
        with mock.patch.object(recall, "_fact_pairs", return_value=[]), \
             mock.patch.object(recall, "_interest_pairs", return_value=[]), \
             mock.patch.object(recall, "_qa_pairs", return_value=[]), \
             mock.patch.object(recall, "_relationship_lines", return_value=[]), \
             mock.patch.object(recall, "search_episodes", return_value=[]):
            self.assertEqual(
                recall.memory_question_lines(1, "What do you know about me?"), []
            )


class FactFloorTest(unittest.TestCase):
    def _bundle(self, n_facts=10, n_interests=20, budget=16):
        # Facts score LOW (aging), interests score HIGH — the field shape.
        facts = [{"id": i, "key": f"fact_{i}", "value": "v", "confidence": 0.9,
                  "importance": 0.85} for i in range(n_facts)]
        interests = [{"id": i, "name": f"hobby_{i}", "interest_strength": "high",
                      "confidence": 1.0} for i in range(n_interests)]
        from memory import facts as facts_db, interests as interests_db
        with mock.patch.object(facts_db, "get_prompt_worthy_facts", return_value=facts), \
             mock.patch.object(interests_db, "get_interests_for_prompt",
                               return_value=interests), \
             mock.patch.object(facts_db, "score_fact_for_prompt", return_value=0.40):
            return retrieval.retrieve_person_memory(1, budget=budget)

    def test_floor_guarantees_fact_seats(self):
        bundle = self._bundle()
        floor = int(getattr(config, "MEMORY_RETRIEVAL_MIN_FACTS", 6))
        self.assertGreaterEqual(len(bundle["facts"]), floor)
        self.assertEqual(
            len(bundle["facts"]) + len(bundle["interests"]), 16
        )

    def test_floor_never_exceeds_available_facts(self):
        bundle = self._bundle(n_facts=2)
        self.assertEqual(len(bundle["facts"]), 2)


class FragmentGateTest(unittest.TestCase):
    def test_field_junk_all_rejected(self):
        for name in ("d the movie", "for a job", "lot",
                     "your program to improve you"):
            self.assertIsNotNone(fact_quality.is_dangling_fragment(name), name)
            self.assertIsNotNone(fact_quality.reject_interest(name), name)

    def test_real_memories_all_kept(self):
        for name in ("the Odyssey", "3D printing", "people watching",
                     "mint chocolate chip ice cream", "a cappella singing",
                     "I Spy", "camping", "Star Wars",
                     "really cool website for my work"):
            self.assertIsNone(fact_quality.is_dangling_fragment(name), name)

    def test_interest_flavored_fact_values_gated(self):
        self.assertEqual(
            fact_quality.reject_fact("interest", "interest_for_a_job", "for a job"),
            "fragment_lead_word",
        )
        self.assertIsNone(
            fact_quality.reject_fact("interest", "interest_star_wars", "Star Wars")
        )


class LeanInjectionTest(unittest.TestCase):
    def test_memory_question_swaps_in_rich_block(self):
        from intelligence import lean_brain
        from memory import people
        with mock.patch.object(people, "get_person",
                               return_value={"id": 1, "name": "Bret Benziger",
                                             "friendship_tier": "acquaintance"}), \
             mock.patch.object(recall, "memory_question_lines",
                               return_value=["MEMORY QUESTION: ...", "Facts you know: x."]), \
             mock.patch.object(lean_brain, "_recent_topics", return_value=[]):
            lines = lean_brain._person_lines(1, "What do you know about me?")
        joined = " ".join(lines)
        self.assertIn("MEMORY QUESTION", joined)
        self.assertNotIn("Background you happen to know", joined)

    def test_ordinary_turn_uses_topic_ranked_background(self):
        from intelligence import lean_brain
        from memory import people
        captured = {}

        def fake_retrieve(pid, topic_tokens=None, budget=None, **kw):
            captured["tokens"] = topic_tokens
            return {"facts": [{"key": "favorite_movie", "value": "Mrs. Doubtfire"}],
                    "interests": [{"name": "the Odyssey"}]}

        with mock.patch.object(people, "get_person",
                               return_value={"id": 1, "name": "Bret Benziger",
                                             "friendship_tier": "acquaintance"}), \
             mock.patch.object(retrieval, "retrieve_person_memory",
                               side_effect=fake_retrieve), \
             mock.patch.object(lean_brain, "_recent_topics", return_value=[]):
            lines = lean_brain._person_lines(1, "I watched a great movie.")
        joined = " ".join(lines)
        self.assertIn("Mrs. Doubtfire", joined)
        self.assertIn("the Odyssey", joined)
        self.assertIn("movie", captured["tokens"])


if __name__ == "__main__":
    unittest.main()


class DateExpressionTest(unittest.TestCase):
    from datetime import date as _date
    _TODAY = _date(2026, 8, 1)

    def _parse(self, text):
        return recall.parse_date_expression(text, today=self._TODAY)

    def test_explicit_month_day_year(self):
        self.assertEqual(self._parse("what did we talk about on July 12 2026?")[:2],
                         ("2026-07-12", "2026-07-12"))

    def test_bare_month_day_resolves_to_past(self):
        # December 25 hasn't happened yet in 2026 → last year's.
        self.assertEqual(self._parse("december 25")[0], "2025-12-25")
        self.assertEqual(self._parse("july 12")[0], "2026-07-12")

    def test_relative_expressions(self):
        self.assertEqual(self._parse("earlier today")[:2], ("2026-08-01", "2026-08-01"))
        self.assertEqual(self._parse("yesterday")[:2], ("2026-07-31", "2026-07-31"))
        self.assertEqual(self._parse("last week")[:2], ("2026-07-25", "2026-07-31"))
        self.assertEqual(self._parse("last time")[0], "LAST_SESSION")

    def test_no_date_returns_none(self):
        self.assertIsNone(self._parse("what are my hobbies?"))
        self.assertIsNone(self._parse("we should talk about July sometime"))


class ConversationRecallTest(unittest.TestCase):
    _TURNS = [
        {"speaker": "Rex", "text": "Please wait while I finish loading.", "ts": "2026-07-20 20:00:00"},
        {"speaker": "Rex", "text": "Boot successful.", "ts": "2026-07-20 20:00:10"},
        {"speaker": "Bret Benziger", "text": "I finished your motor system today.", "ts": "2026-07-20 20:01:00"},
        {"speaker": "Rex", "text": "Finally — wheels I can complain about.", "ts": "2026-07-20 20:01:05"},
    ]

    def test_dated_question_injects_actual_turns(self):
        from memory import conversations as conv_db
        with mock.patch.object(conv_db, "get_logged_turns", return_value=list(self._TURNS)), \
             mock.patch.object(conv_db, "get_conversation_history", return_value=[]):
            lines = recall.conversation_recall_lines(1, "What did we talk about on July 20 2026?")
        joined = " ".join(lines)
        self.assertIn("CONVERSATION RECALL", joined)
        self.assertIn("motor system", joined)
        # Leading Rex-only boot lines are trimmed.
        self.assertNotIn("finish loading", joined)
        self.assertIn("never mention logs, records, or transcripts".lower(),
                      joined.lower())

    def test_empty_window_yields_honest_blank(self):
        from memory import conversations as conv_db
        with mock.patch.object(conv_db, "get_logged_turns", return_value=[]):
            lines = recall.conversation_recall_lines(1, "What did we talk about on July 19 2026?")
        self.assertEqual(len(lines), 1)
        self.assertIn("NOTHING", lines[0])
        self.assertIn("do not invent", lines[0])

    def test_undated_or_verbless_asks_no_block(self):
        self.assertEqual(recall.conversation_recall_lines(1, "What are my hobbies?"), [])
        self.assertEqual(recall.conversation_recall_lines(1, "Yesterday was fun."), [])

    def test_over_long_day_is_evenly_sampled(self):
        turns = [{"speaker": "Bret Benziger", "text": f"line {i}", "ts": "t"}
                 for i in range(200)]
        sampled = recall._sample_turns(turns, 40)
        self.assertEqual(len(sampled), 40)
        self.assertEqual(sampled[0]["text"], "line 0")
        self.assertGreater(int(sampled[-1]["text"].split()[1]), 150)


class DatedMentionsTest(unittest.TestCase):
    _ROWS = [
        {"day": "2026-08-01", "text": "Do you know when I went camping?"},
        {"day": "2026-07-11", "text": "Camping was fine"},
        {"day": "2026-07-11", "text": "We set up camp by the river"},
        {"day": "2026-06-18", "text": "I said I'm going camping next month"},
        {"day": "2026-07-20", "text": "I finished your motor system"},
    ]

    def _mentions(self, tokens):
        from memory import database as db
        with mock.patch.object(db, "fetchall", return_value=list(self._ROWS)):
            return recall._dated_mentions(1, tokens)

    def test_dated_mentions_found_questions_excluded(self):
        out = self._mentions({"camp"})
        joined = " ".join(out)
        self.assertIn("[2026-07-11]", joined)
        self.assertIn("[2026-06-18]", joined)
        self.assertNotIn("Do you know when", joined)   # their question isn't a mention
        self.assertNotIn("motor system", joined)       # off-topic excluded

    def test_one_line_per_day(self):
        out = self._mentions({"camp"})
        self.assertEqual(sum(1 for l in out if "[2026-07-11]" in l), 1)

    def test_no_tokens_no_mentions(self):
        self.assertEqual(recall._dated_mentions(1, set()), [])
