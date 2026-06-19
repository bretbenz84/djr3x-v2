"""
Topic-relevant memory retrieval: a person's injected facts/interests are ranked against
what they JUST said (the live topic), so Rex surfaces the fitting memory because it fit —
not only the static top-N by importance.
"""

from __future__ import annotations

import unittest
from unittest import mock

from memory import facts, interests
from intelligence import topic_thread


def _fact(key, value, *, importance=0.5, confidence=0.5, category=""):
    return {
        "id": None, "key": key, "value": value, "category": category,
        "importance": importance, "confidence": confidence, "source": "explicit",
        "freshness_label": "fresh", "age_days": 1, "last_used_age_days": None,
        "decay_rate": "normal",
    }


class FactTopicOverlapTest(unittest.TestCase):
    def test_overlap_counts_matching_words(self):
        f = _fact("hobby", "rock climbing", category="sport")
        self.assertEqual(facts.fact_topic_overlap(f, {"climbing"}), 1)
        self.assertEqual(facts.fact_topic_overlap(f, {"climbing", "sport"}), 2)
        self.assertEqual(facts.fact_topic_overlap(f, {"weather"}), 0)
        self.assertEqual(facts.fact_topic_overlap(f, set()), 0)


class TopicBoostedFactRankingTest(unittest.TestCase):
    def _facts(self):
        return [
            _fact("job", "works at Boeing", importance=0.9, confidence=0.9),  # high, off-topic
            _fact("hobby", "rock climbing", importance=0.3, confidence=0.5),  # low, on-topic
        ]

    def test_static_ranking_puts_important_fact_first(self):
        with mock.patch.object(facts, "get_facts", return_value=self._facts()):
            out = facts.get_prompt_worthy_facts(1, limit=2)
            self.assertEqual(out[0]["key"], "job")

    def test_topic_relevant_fact_is_lifted_above_more_important_one(self):
        with mock.patch.object(facts, "get_facts", return_value=self._facts()):
            out = facts.get_prompt_worthy_facts(1, limit=2, topic_tokens={"climbing"})
            self.assertEqual(out[0]["key"], "hobby")

    def test_no_tokens_is_unchanged(self):
        with mock.patch.object(facts, "get_facts", return_value=self._facts()):
            a = [f["key"] for f in facts.get_prompt_worthy_facts(1, limit=2)]
            b = [f["key"] for f in facts.get_prompt_worthy_facts(1, limit=2, topic_tokens=set())]
            self.assertEqual(a, b)


class InterestRelevanceTest(unittest.TestCase):
    def test_interest_overlap(self):
        it = {"name": "rock climbing", "category": "sport", "notes": "boulders"}
        self.assertEqual(interests.interest_topic_overlap(it, {"climbing"}), 1)
        self.assertEqual(interests.interest_topic_overlap(it, {"boulders", "sport"}), 2)
        self.assertEqual(interests.interest_topic_overlap(it, set()), 0)

    def test_topic_relevant_interest_sorted_first(self):
        rows = [
            {"name": "jazz", "interest_strength": "high", "category": "", "notes": ""},
            {"name": "rock climbing", "interest_strength": "low", "category": "", "notes": ""},
        ]
        with mock.patch.object(interests, "db") as db, \
             mock.patch.object(interests, "_annotate", side_effect=lambda r: r):
            db.fetchall.return_value = rows
            out = interests.get_interests_for_prompt(1, limit=2, topic_tokens={"climbing"})
            self.assertEqual(out[0]["name"], "rock climbing")  # on-topic beats higher-strength


class TopicTokensTest(unittest.TestCase):
    def setUp(self):
        topic_thread.clear()

    def tearDown(self):
        topic_thread.clear()

    def test_tokens_from_latest_user_text(self):
        topic_thread.note_user_turn("I've been getting really into rock climbing lately")
        tokens = topic_thread.topic_tokens()
        self.assertIn("climbing", tokens)
        self.assertIn("rock", tokens)

    def test_empty_when_no_thread(self):
        self.assertEqual(topic_thread.topic_tokens(), set())


if __name__ == "__main__":
    unittest.main()
