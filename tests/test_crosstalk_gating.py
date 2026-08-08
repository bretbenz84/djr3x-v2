"""
Tier 2 / item 5 — cross-talk gating. With Tier-1 single-visible attribution, an
overheard device/UI readout ("oh yeah that's definitely 30 FPS") would be pinned on
the visible person and answered/profiled. The strengthened detector marks clear,
non-Rex device readouts as background chatter so the reply path drops them.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from intelligence import interaction as I


class BackgroundCrosstalkDetectorTest(unittest.TestCase):
    def test_device_readouts_are_chatter(self):
        for text in ("oh yeah that's definitely 30 FPS", "that is 1080p",
                     "the frame rate looks choppy", "check the latency on that"):
            self.assertTrue(I._looks_like_background_crosstalk(text), text)

    def test_personal_or_directed_speech_is_not_chatter(self):
        for text in ("I'm at 100 percent today", "Hey Rex play some music",
                     "what do you think about that", "my favorite is 60fps gaming"):
            self.assertFalse(I._looks_like_background_crosstalk(text), text)

    def test_short_real_answers_are_not_chatter(self):
        for text in ("China", "Kebab", "yeah", "I really love astrophotography lately"):
            self.assertFalse(I._looks_like_background_crosstalk(text), text)

    def test_commands_are_never_chatter(self):
        # _speech_is_directed_to_rex (command_parser) short-circuits the detector.
        self.assertFalse(I._looks_like_background_crosstalk("shut down"))


class ThirdPartyCrosstalkContextGateTest(unittest.TestCase):
    """Regression for the 2026-08-08 11:19 field bug: a solo owner's "I love
    you." — said straight to Rex, seconds after Rex's own line — matched the
    endearment regex and was silently dropped. The context gate must veto
    suppression in a sole-person room and inside Rex's reply window."""

    def setUp(self):
        # Default context: solo room, no unknowns, no second voice, Rex quiet
        # for ages. Individual tests override the piece they exercise.
        patchers = [
            mock.patch.object(I.world_state, "get", return_value=[{"person_db_id": 1}]),
            mock.patch.object(I, "_has_unknown_visible_or_recent", return_value=False),
            mock.patch.object(I, "_other_known_visible_recently", return_value=False),
            mock.patch.object(
                I.speech_queue, "seconds_since_last_speech", return_value=float("inf")
            ),
            mock.patch.object(I, "_anonymous_speaker_slots", []),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def test_field_case_solo_i_love_you_is_kept(self):
        # The exact field line still matches the endearment shape...
        self.assertTrue(I._looks_like_third_party_crosstalk("I love you."))
        # ...but with nobody else around, suppression is vetoed.
        self.assertFalse(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_reply_window_vetoes_even_with_second_person(self):
        with mock.patch.object(
            I.world_state, "get",
            return_value=[{"person_db_id": 1}, {"person_db_id": 2}],
        ), mock.patch.object(
            I.speech_queue, "seconds_since_last_speech", return_value=2.0
        ):
            self.assertFalse(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_two_visible_faces_allow_suppression(self):
        with mock.patch.object(
            I.world_state, "get",
            return_value=[{"person_db_id": 1}, {"person_db_id": None}],
        ):
            self.assertTrue(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_recent_unknown_face_allows_suppression(self):
        with mock.patch.object(
            I, "_has_unknown_visible_or_recent", return_value=True
        ):
            self.assertTrue(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_recent_second_voice_allows_suppression(self):
        # Off-camera partner: an anonymous voice heard a minute ago counts as
        # someone to say "love you too" to.
        slot = SimpleNamespace(last_seen_at=time.monotonic() - 60.0)
        with mock.patch.object(I, "_anonymous_speaker_slots", [slot]):
            self.assertTrue(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_stale_second_voice_does_not_count(self):
        slot = SimpleNamespace(last_seen_at=time.monotonic() - 3600.0)
        with mock.patch.object(I, "_anonymous_speaker_slots", [slot]):
            self.assertFalse(I._crosstalk_suppression_context_ok(raw_best_id=1))

    def test_other_known_face_allows_suppression_only_with_matched_voice(self):
        with mock.patch.object(
            I, "_other_known_visible_recently", return_value=True
        ):
            # Voice matched person 1, a DIFFERENT known face is around → plausible
            # third party.
            self.assertTrue(I._crosstalk_suppression_context_ok(raw_best_id=1))
            # Unmatched voice: the one visible known face is most likely the
            # speaker themselves — no second-person evidence.
            self.assertFalse(I._crosstalk_suppression_context_ok(raw_best_id=None))

    def test_require_second_person_flag_off_restores_old_behavior(self):
        with mock.patch.object(
            I.config, "CROSSTALK_REQUIRE_SECOND_PERSON", False, create=True
        ):
            self.assertTrue(I._crosstalk_suppression_context_ok(raw_best_id=1))


if __name__ == "__main__":
    unittest.main()
