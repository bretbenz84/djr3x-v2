"""
Phase 0.1 — input-trust gate: low-confidence ASR noise and affection statements
must NOT become steered interests or pinned topics.

Regression for the live defect (logs/djr3x-2026-06-13-14-44-51.log): the
mis-heard "I love you now" minted the steered interest "you now" and built a full
"deepen the interest thread" interview around it, and a lone "ahem" became the
pinned topic "ahem" that persisted across turns.
"""

from __future__ import annotations

import unittest


class CleanTopicSubstantivenessTest(unittest.TestCase):
    def test_pronoun_function_word_fragments_rejected(self):
        from intelligence import conversation_steering as cs
        for junk in ("you now", "me too", "it now", "you", "that this"):
            self.assertIsNone(cs._clean_topic(junk), f"{junk!r} should be rejected")

    def test_real_topics_survive(self):
        from intelligence import conversation_steering as cs
        for topic in ("art", "tea", "cars", "3d printing", "star trek",
                      "astrophotography", "mint chocolate chip"):
            self.assertEqual(cs._clean_topic(topic), topic,
                             f"{topic!r} should survive the gate")


class DetectInterestTest(unittest.TestCase):
    def test_affection_does_not_become_interest(self):
        from intelligence import conversation_steering as cs
        # The exact live repro: regex captures " you now" after "love", but the
        # substantiveness gate must reject it.
        self.assertIsNone(cs.detect_interest("I love you now"))

    def test_genuine_interest_still_detected(self):
        from intelligence import conversation_steering as cs
        self.assertEqual(cs.detect_interest("I love astrophotography"),
                         "astrophotography")
        self.assertEqual(cs.detect_interest("I'm really into 3d printing"),
                         "3d printing")


class NoteUserTurnComplimentBailTest(unittest.TestCase):
    def setUp(self):
        from intelligence import conversation_steering as cs
        cs.clear()

    def tearDown(self):
        from intelligence import conversation_steering as cs
        cs.clear()

    def test_compliment_returns_no_steering_and_no_topic(self):
        from intelligence import conversation_steering as cs
        # person_id=None avoids any DB writes; we only assert the control flow.
        self.assertIsNone(cs.note_user_turn(None, "I love you now"))
        self.assertIsNone(cs.note_user_turn(None, "you're the best"))

    def test_real_interest_still_steers(self):
        from intelligence import conversation_steering as cs
        ctx = cs.note_user_turn(None, "I love astrophotography")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.topic, "astrophotography")


class TopicThreadInterjectionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import topic_thread
        topic_thread.clear()

    def tearDown(self):
        from intelligence import topic_thread
        topic_thread.clear()

    def test_interjection_yields_no_keywords(self):
        from intelligence import topic_thread as tt
        self.assertEqual(tt._keywords("ahem"), [])
        self.assertEqual(tt._classify_topic("ahem"), ("current exchange", "light"))

    def test_ahem_is_not_pinned_as_topic(self):
        from intelligence import topic_thread as tt
        tt.note_user_turn("ahem")
        snap = tt.snapshot()
        self.assertIsNotNone(snap)
        self.assertNotEqual(snap.get("label"), "ahem")

    def test_real_keywords_still_extracted(self):
        from intelligence import topic_thread as tt
        self.assertIn("astrophotography", tt._keywords("astrophotography is great"))


class LastConversationDirectiveStripTest(unittest.TestCase):
    """S3: a baked-in 'Rex should follow up on ... ice cream' imperative in a
    stored summary must be stripped so it can't force an off-topic callback."""

    def test_strips_rex_should_clause(self):
        from intelligence import llm
        summary = (
            "Bret mentioned celebrating his birthday and described his surroundings "
            "as chaotic. He suggested playing classical music. Rex should follow up "
            "on how the birthday went and if he enjoyed ice cream amidst the chaos."
        )
        out = llm._strip_rex_directives(summary)
        self.assertNotIn("Rex should", out)
        self.assertNotIn("ice cream", out)
        self.assertIn("birthday", out)

    def test_leaves_neutral_recap_untouched(self):
        from intelligence import llm
        summary = "Bret talked about camping and his dog Max. He seemed relaxed."
        out = llm._strip_rex_directives(summary)
        self.assertIn("camping", out)
        self.assertIn("Max", out)


if __name__ == "__main__":
    unittest.main()
