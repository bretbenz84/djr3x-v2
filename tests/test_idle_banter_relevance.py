"""
Idle-banter opinion barge-ins should be RELEVANT to the live conversation, not generic
pre-canned Rex opinions or his off-topic preoccupation. Field log 2026-06-14: while the
user talked camping/the river/Max, Rex barged in with "ranking organic snacks", "silence
is unacceptable", "dead air is like a malfunctioning hyperdrive" — all conversation-
agnostic. The volunteer directive now stays ON the live topic, and the off-topic rex_pov
preoccupation is only a cold-open fallback (no real exchange yet).
"""

from __future__ import annotations

import unittest
from unittest import mock

from intelligence import interaction as ix


class IdleBanterDirectiveTest(unittest.TestCase):
    def test_ask_user_uses_question_directive(self):
        directive, pov = ix._idle_banter_directive(True, False, "ranking organic snacks")
        self.assertIs(directive, ix._IDLE_BANTER_DIRECTIVES[0])
        self.assertFalse(pov)

    def test_ask_user_with_live_topic_deepens_thread_not_generic_interview(self):
        # Field log 2026-06-16: the user said "I'm working on your time-of-flight
        # sensors" and ~50s later idle banter asked "What's the latest project you're
        # diving into?" → "I just told you." Once a topic is open, the "spotlight on
        # the user" slot must DEEPEN that thread, never reset to the generic interview
        # directive that re-asks what they just said.
        directive, pov = ix._idle_banter_directive(True, True, "ranking organic snacks")
        self.assertFalse(pov)
        self.assertIs(directive, ix._IDLE_BANTER_LIVE_TOPIC_ASK)
        self.assertIsNot(directive, ix._IDLE_BANTER_DIRECTIVES[0])
        self.assertIn("STAY ON", directive)
        self.assertIn("NEVER re-ask", directive)
        self.assertNotIn("organic snacks", directive)  # off-topic POV not dumped

    def test_volunteer_with_live_topic_stays_on_topic_and_skips_pov(self):
        directive, pov = ix._idle_banter_directive(False, True, "ranking organic snacks")
        self.assertFalse(pov)                         # preoccupation NOT dumped
        self.assertNotIn("organic snacks", directive)  # ...so it can't go off-topic
        self.assertIn("Stay ON the subject", directive)
        self.assertIn("Do NOT change the subject", directive)

    def test_volunteer_directive_forbids_generic_dj_silence_bits(self):
        # The exact off-topic shapes from the field log are explicitly disallowed.
        directive = ix._IDLE_BANTER_DIRECTIVES[1]
        self.assertIn("music/DJ/silence/dead-air", directive)

    def test_cold_open_falls_back_to_preoccupation(self):
        directive, pov = ix._idle_banter_directive(False, False, "ranking organic snacks")
        self.assertTrue(pov)
        self.assertIn("ranking organic snacks", directive)

    def test_cold_open_without_pov_uses_on_topic_directive(self):
        directive, pov = ix._idle_banter_directive(False, False, "")
        self.assertFalse(pov)
        self.assertIs(directive, ix._IDLE_BANTER_DIRECTIVES[1])


class IdleHasLiveTopicTest(unittest.TestCase):
    def _patch_transcript(self, entries):
        return mock.patch.object(
            ix.conv_memory, "get_session_transcript", return_value=entries
        )

    def test_empty_transcript_is_not_live(self):
        with self._patch_transcript([]):
            self.assertFalse(ix._idle_has_live_topic())

    def test_single_line_is_not_live(self):
        with self._patch_transcript([{"speaker": "Bret", "text": "hi"}]):
            self.assertFalse(ix._idle_has_live_topic())

    def test_real_exchange_is_live(self):
        with self._patch_transcript([
            {"speaker": "Bret", "text": "I just got back from camping"},
            {"speaker": "Rex", "text": "Roughing it, huh?"},
        ]):
            self.assertTrue(ix._idle_has_live_topic())

    def test_only_rex_spoke_is_not_live(self):
        # A present-but-silent person Rex only greeted has no topic to deepen — so
        # re-engagement asks a getting-to-know-you question, not a fake topic follow-up.
        with self._patch_transcript([
            {"speaker": "Rex", "text": "Hey Bret, good to see you!"},
            {"speaker": "Rex", "text": "Juneteenth at home, nice."},
        ]):
            self.assertFalse(ix._idle_has_live_topic())

    def test_blank_text_entries_do_not_count(self):
        with self._patch_transcript([
            {"speaker": "Bret", "text": "   "},
            {"speaker": "Rex", "text": ""},
        ]):
            self.assertFalse(ix._idle_has_live_topic())


if __name__ == "__main__":
    unittest.main()
