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
