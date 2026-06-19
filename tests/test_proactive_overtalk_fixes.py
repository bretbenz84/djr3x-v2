"""
BUG-1 + BUG-3 (the 2026-06-18 over-talk run): the proactive/idle layer piled
lines on during an active exchange, re-asked a question it had just asked, and
dragged a topic back ~20s after the user asked to change the subject.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import interaction as I
from memory import boundaries


class ChangeSubjectDetectionTest(unittest.TestCase):
    def test_change_subject_phrasings_detected_as_boundary(self):
        for text in (
            "Lets choose the subject",
            "let's change the subject",
            "can we talk about something else please?",
            "pick a different topic",
            "new subject",
            "can we talk about something else",
        ):
            with self.subTest(text=text):
                detected = boundaries.detect_boundary(text, fallback_topic="festival")
                self.assertIsNotNone(detected, text)
                self.assertEqual(detected["topic"], "festival")
                self.assertEqual(detected["behavior"], "mention")

    def test_genuine_interest_move_is_not_a_boundary(self):
        # A real interest declaration must NOT be swallowed as a subject change.
        for text in ("let's talk about astronomy", "I want to talk about astronomy"):
            with self.subTest(text=text):
                self.assertIsNone(boundaries.detect_boundary(text, fallback_topic="festival"))

    def test_embedded_subject_mentions_are_not_boundaries(self):
        # The bare "new subject"/"change of subject" alternations used to false-fire
        # on embedded mentions — they must not.
        for text in (
            "the new subject I'm studying is biology",
            "let us talk about the change of subject in my thesis",
            "there's a different topic I really love",
            "we had a change of subject in class today",
        ):
            with self.subTest(text=text):
                self.assertIsNone(boundaries.detect_boundary(text, fallback_topic="festival"))


class TopicBanStoreTest(unittest.TestCase):
    def setUp(self):
        I._recently_banned_topics.clear()

    def tearDown(self):
        I._recently_banned_topics.clear()

    def test_record_and_match_banned_topic(self):
        I._record_banned_topic("festival people-watching")
        active = I.recently_banned_topics()
        self.assertEqual(len(active), 1)
        self.assertTrue(I._topic_is_recently_banned("the festival was crowded"))
        self.assertFalse(I._topic_is_recently_banned("camping next month"))

    def test_generic_topic_records_no_ban(self):
        I._record_banned_topic("that")  # carries no content tokens
        self.assertEqual(I.recently_banned_topics(), [])

    def test_expired_ban_drops_out(self):
        I._record_banned_topic("festival")
        I._recently_banned_topics[0]["banned_until"] = 0.0  # force-expire
        self.assertEqual(I.recently_banned_topics(), [])

    def test_side_effects_record_ban(self):
        I._recently_banned_topics.clear()
        with mock.patch.object(I.conversation_steering, "clear"), \
             mock.patch.object(I.topic_thread, "clear"), \
             mock.patch.object(I.premise_memory, "clear"), \
             mock.patch.object(I.end_thread, "note_user_turn"):
            I._apply_topic_boundary_side_effects(1, "let's change the subject", banned_topic="festival")
        self.assertTrue(I._topic_is_recently_banned("festival crowds"))


class RecentQuestionDedupTest(unittest.TestCase):
    def setUp(self):
        I._recent_rex_questions.clear()

    def tearDown(self):
        I._recent_rex_questions.clear()

    def test_near_duplicate_question_detected(self):
        # The real-log camping re-ask shared several content words (camping,
        # setup, excuse, disappear) — caught by the 2+ shared-word rule.
        I._note_rex_question(
            "What's the part of camping that hooks you: the setup, or the excuse to disappear?"
        )
        self.assertTrue(
            I._line_duplicates_recent_question(
                "What do you need most out there: the setup, or the excuse to disappear from camping?"
            )
        )

    def test_unrelated_question_not_flagged(self):
        I._note_rex_question("What's the part of camping that actually hooks you?")
        self.assertFalse(I._line_duplicates_recent_question("What music do you like?"))
        self.assertFalse(I._line_duplicates_recent_question("Any plans this weekend?"))

    def test_single_incidental_shared_word_not_flagged(self):
        # "favorite movie" vs "favorite food" share only "favorite" — not a dup.
        I._note_rex_question("What's your favorite movie?")
        self.assertFalse(I._line_duplicates_recent_question("What's your favorite food?"))
        I._recent_rex_questions.clear()
        I._note_rex_question("What music do you play?")
        self.assertFalse(I._line_duplicates_recent_question("Do you play sports?"))

    def test_statement_is_never_a_duplicate(self):
        I._note_rex_question("What's your favorite trail?")
        self.assertFalse(I._line_duplicates_recent_question("Camping is the best."))

    def test_register_rex_question_arms_floor_hold(self):
        I._floor_held_until = 0.0
        try:
            I._register_rex_utterance("So what draws you to camping?")
            self.assertGreater(I._floor_held_until, I.time.monotonic())
        finally:
            I._floor_held_until = 0.0
            I._recent_rex_questions.clear()

    def test_no_response_recovery_does_not_shorten_reply_floor(self):
        # A reply-with-question arms the 18s POST_REPLY_QUESTION_WAIT_SECS floor;
        # the no-response-recovery arming must NOT clobber it down to 10s.
        long_hold = I.time.monotonic() + 18.0
        I._floor_held_until = long_hold
        try:
            with mock.patch.object(I, "_game_suppresses_conversation", return_value=False), \
                 mock.patch.object(I, "_question_expects_response", return_value=True), \
                 mock.patch.object(I, "_question_recovery_cooldown_secs", return_value=10.0):
                I._arm_no_response_recovery("So what draws you to camping?", 1)
            self.assertGreaterEqual(I._floor_held_until, long_hold)
        finally:
            I._floor_held_until = 0.0


class WanderRegreetSuppressionTest(unittest.TestCase):
    def test_spoken_regreet_suppressed_during_active_conversation(self):
        from intelligence import consciousness
        from awareness.situation import SituationProfile

        active = mock.Mock(spec=SituationProfile)
        active.suppress_proactive = False
        active.conversation_active = True
        active.rapid_exchange = False
        with mock.patch.object(consciousness, "_generate_and_speak_presence") as speak:
            consciousness._maybe_fire_wander_regreet({}, active)
        speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
