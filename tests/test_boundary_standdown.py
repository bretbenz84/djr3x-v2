"""Boundary stand-down fixes (field 2026-08-03 20:03).

Bret: "we don't need to bring up the website anymore" → Rex: "Understood. What's
the actual pain point right now?" — same topic, next breath. Three holes, three
fixes:
  * the boundary detector had no pattern for softened first-person-plural forms,
    so the whole harness never fired and the reply LLM freestyled;
  * when a stand-down DID fire, the ban label could be derived from the ban
    request's own words (the log shows "don't / need" banned — the website never
    was);
  * the lean brain — the live path — rendered no boundaries, passed no mute
    terms, and its proactive cues never checked bans, so diary open threads kept
    resurrecting the shut-down topic across sessions.
"""

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import config
from memory import boundaries
from memory import database as db


FIELD_SENTENCE = "I'm still working on it, but we don't need to bring up the website anymore."


class _TempDb(unittest.TestCase):
    def setUp(self):
        from setup_assets import DB_SCHEMA
        self._tmp = tempfile.TemporaryDirectory()
        self._path = Path(self._tmp.name) / "people.db"
        with sqlite3.connect(self._path) as conn:
            conn.executescript(DB_SCHEMA)
            conn.execute("INSERT INTO people (id, name) VALUES (1, 'Test User')")
        self._p = mock.patch.object(db, "_DB_FILE", self._path)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def _boundary_rows(self):
        with sqlite3.connect(self._path) as conn:
            return conn.execute(
                "SELECT behavior, topic FROM person_conversation_boundaries "
                "WHERE person_id=1 AND active=1"
            ).fetchall()


# ── Detector coverage ───────────────────────────────────────────────────────────

class SoftenedFormDetectionTest(unittest.TestCase):
    def test_field_sentence_yields_durable_named_boundary(self):
        d = boundaries.detect_boundary(FIELD_SENTENCE)
        self.assertIsNotNone(d)
        self.assertEqual(d["kind"], "boundary")
        self.assertEqual(d["behavior"], "mention")
        self.assertEqual(d["topic"], "website")

    def test_softened_named_forms(self):
        for text, topic in (
            ("no need to talk about the website", "website"),
            ("we can stop talking about my diet", "diet"),
            ("let's not talk about work", "work"),
            ("you don't have to mention the divorce anymore", "divorce"),
            ("we don't need to discuss the lawsuit", "lawsuit"),
        ):
            d = boundaries.detect_boundary(text)
            self.assertIsNotNone(d, text)
            self.assertEqual(d["kind"], "boundary", text)
            self.assertEqual(d["topic"], topic, text)

    def test_pronoun_forms_resolve_from_fallback(self):
        for text in (
            "we don't need to bring it up anymore",
            "you don't have to talk about it",
            "no need to mention that again",
            "we can stop talking about it",
        ):
            d = boundaries.detect_boundary(text, fallback_topic="website testing")
            self.assertIsNotNone(d, text)
            self.assertEqual(d["topic"], "website testing", text)

    def test_pronoun_form_without_fallback_stays_placeholder(self):
        d = boundaries.detect_boundary("we don't need to bring it up anymore")
        self.assertIsNotNone(d)
        self.assertEqual(d["topic"], "current topic")

    def test_innocent_phrasings_do_not_trigger(self):
        for text in (
            "let's talk about astronomy",
            "the new subject I'm studying is hard",
            "we need to talk about the website",
            "I don't need to work tomorrow",
            "you don't have to help me with the website",
        ):
            self.assertIsNone(boundaries.detect_boundary(text), text)


# ── Ban the right topic ─────────────────────────────────────────────────────────

class BanLabelTest(unittest.TestCase):
    def test_label_derived_from_request_is_rejected(self):
        from intelligence import interaction as I
        request = "We don't need to bring it up. We can change the subject."
        with mock.patch.object(I.topic_thread, "snapshot",
                               return_value={"label": "don't / need"}), \
             mock.patch.object(I.topic_thread, "topic_tokens",
                               return_value={"don", "need", "bring", "subject"}):
            self.assertIsNone(I._boundary_fallback_topic(exclude_text=request))

    def test_real_thread_label_survives(self):
        from intelligence import interaction as I
        request = "We don't need to bring it up. We can change the subject."
        with mock.patch.object(I.topic_thread, "snapshot",
                               return_value={"label": "website testing"}):
            self.assertEqual(
                I._boundary_fallback_topic(exclude_text=request), "website testing")

    def test_softened_standdown_reads_as_avoidant_and_keeps_thread_label(self):
        from intelligence import topic_thread as tt
        saved = tt._current
        try:
            tt._current = None
            tt.note_user_turn("The website testing is going okay I guess")
            label_before = tt._current.label
            tt.note_user_turn(FIELD_SENTENCE)
            self.assertEqual(tt._current.user_stance, "avoidant")
            self.assertEqual(tt._current.label, label_before,
                             "a stand-down must not relabel the thread with its own words")
        finally:
            tt._current = saved


class ConversationBoundaryHandlerTest(_TempDb):
    def test_offline_field_sentence_stores_boundary_and_bans_website(self):
        # OFFLINE lane since 2026-08-13: emotional.boundary migrated to the live
        # tool router, so ONLINE the durable row is the model's call and this
        # handler only applies the reversible 90s ban (pinned separately below).
        # With the link down the regex remains the only thing that records consent.
        from intelligence import interaction as I
        banned = []
        with mock.patch("intelligence.connectivity.is_offline", return_value=True), \
             mock.patch.object(I, "_record_banned_topic", side_effect=banned.append), \
             mock.patch.object(I, "_boundary_fallback_topic", return_value=None):
            reply = I._handle_conversation_boundary(1, FIELD_SENTENCE)
        self.assertIsNotNone(reply)
        self.assertIn("website", reply.lower())
        self.assertIn(("mention", "website"), self._boundary_rows())
        self.assertIn("website", [str(b) for b in banned])

    def test_offline_unresolvable_pronoun_form_stores_no_junk_row(self):
        from intelligence import interaction as I
        with mock.patch("intelligence.connectivity.is_offline", return_value=True), \
             mock.patch.object(I, "_boundary_fallback_topic", return_value=None), \
             mock.patch.object(I.llm, "get_response",
                               return_value="Fine. New subject: how was lunch?"):
            reply = I._handle_conversation_boundary(
                1, "we don't need to bring it up anymore")
        self.assertIsNotNone(reply)
        self.assertEqual(self._boundary_rows(), [],
                         "placeholder topic must not become a stored boundary")

    def test_online_boundary_is_reversible_only_and_the_row_is_the_models_call(self):
        """The durable consent write left the regex on 2026-08-13.

        The audit found this handler minting permanent rows from an INVITATION
        ("Don't ask me how I got it, long story.") and steering AWAY from a topic
        that had just been requested ("Let's talk about the topic of my
        dissertation."). Online it now applies only the reversible in-memory ban
        and hands the turn to the reply call, which decides whether to make it
        permanent via the emotional_boundary tool.
        """
        from intelligence import interaction as I
        banned = []
        with mock.patch.object(I, "_record_banned_topic", side_effect=banned.append), \
             mock.patch.object(I, "_boundary_fallback_topic", return_value=None):
            reply = I._handle_conversation_boundary(1, FIELD_SENTENCE)
        self.assertIsNone(reply, "the turn must reach the reply call")
        self.assertEqual(self._boundary_rows(), [],
                         "the durable row is the model's call now")
        self.assertIn("website", [str(b) for b in banned],
                      "the reversible ban still fires this turn")



# ── Lean wiring ─────────────────────────────────────────────────────────────────

class LeanTopicBlockedTest(_TempDb):
    def test_durable_boundary_blocks_matching_cue_text(self):
        from intelligence import interaction as I
        boundaries.add_boundary(1, "mention", "website")
        self.assertTrue(I._lean_topic_blocked(1, "whether the website testing went okay"))
        self.assertFalse(I._lean_topic_blocked(1, "how the garden project is going"))

    def test_ask_boundary_blocks_too(self):
        from intelligence import interaction as I
        boundaries.add_boundary(1, "ask", "divorce")
        self.assertTrue(I._lean_topic_blocked(1, "how the divorce is going"))

    def test_preference_etiquette_wording_never_blocks_cues(self):
        # Preference rows like "back up a few feet" / "do not talk too much" feed
        # muted_topic_terms with common words ("back", "come", "work") — those
        # must mute facts at most, never kill whole proactive lines (the workday
        # check-in literally contains "work").
        from intelligence import interaction as I
        from memory import preferences
        preferences.upsert_preference(
            1, "conversation", "boundary", "back_up",
            "do not come closer than three feet, back up",
        )
        preferences.upsert_preference(
            1, "conversation", "boundary", "work_ask", "do not ask about work stuff",
        )
        self.assertFalse(I._lean_topic_blocked(1, "So — how was work today?"))
        self.assertFalse(I._lean_topic_blocked(1, "welcome back, take a few minutes"))

    def test_fresh_topic_ban_blocks_for_anyone(self):
        from intelligence import interaction as I
        saved = list(I._recently_banned_topics)
        try:
            I._recently_banned_topics[:] = []
            I._record_banned_topic("website testing")
            self.assertTrue(I._lean_topic_blocked(None, "so, about the website..."))
            self.assertFalse(I._lean_topic_blocked(None, "seen any good movies?"))
        finally:
            I._recently_banned_topics[:] = saved

    def test_open_thread_cue_skips_banned_thread(self):
        from intelligence import interaction as I
        boundaries.add_boundary(1, "mention", "website")
        candidates = [
            {"episode_id": 5, "thread": "whether the website account provisioning got finished", "age_days": 1},
            {"episode_id": 6, "thread": "how the intern training went", "age_days": 1},
        ]
        with mock.patch("intelligence.open_threads.pending_for_person",
                        return_value=candidates):
            cue = I._lean_open_thread_cue(1)
        self.assertIsNotNone(cue)
        self.assertEqual(cue["episode_id"], 6)

    def test_open_thread_cue_none_when_all_banned(self):
        from intelligence import interaction as I
        boundaries.add_boundary(1, "mention", "website")
        candidates = [
            {"episode_id": 5, "thread": "follow up on the website progress", "age_days": 1},
        ]
        with mock.patch("intelligence.open_threads.pending_for_person",
                        return_value=candidates):
            self.assertIsNone(I._lean_open_thread_cue(1))


class LeanPromptWiringTest(_TempDb):
    def _person_lines(self, utterance="how's it going"):
        from intelligence import lean_brain
        with mock.patch("memory.episodic_recall.recent_conversation_topics",
                        return_value=[]), \
             mock.patch("memory.recall.search_episodes", return_value=[]):
            return lean_brain._person_lines(1, utterance)

    def test_stored_boundary_rendered_in_reply_prompt(self):
        boundaries.add_boundary(1, "mention", "website")
        joined = " ".join(self._person_lines())
        self.assertIn("website", joined)
        self.assertIn("consent boundaries", joined.lower())

    def test_active_ban_rendered_in_reply_prompt(self):
        from intelligence import interaction as I
        saved = list(I._recently_banned_topics)
        try:
            I._recently_banned_topics[:] = []
            I._record_banned_topic("website testing")
            joined = " ".join(self._person_lines())
        finally:
            I._recently_banned_topics[:] = saved
        self.assertIn("JUST asked to drop", joined)
        self.assertIn("website testing", joined)

    def test_mute_terms_reach_retrieval(self):
        boundaries.add_boundary(1, "mention", "website")
        captured = {}

        def fake_retrieve(person_id, **kwargs):
            captured.update(kwargs)
            return {"facts": [], "interests": []}

        from intelligence import lean_brain
        with mock.patch("memory.retrieval.retrieve_person_memory",
                        side_effect=fake_retrieve), \
             mock.patch("memory.episodic_recall.recent_conversation_topics",
                        return_value=[]), \
             mock.patch("memory.recall.search_episodes", return_value=[]):
            lean_brain._person_lines(1, "how's it going")
        self.assertIn("website", captured.get("mute_terms") or set())


if __name__ == "__main__":
    unittest.main()
