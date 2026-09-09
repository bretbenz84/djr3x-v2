"""
Regressions from the 2026-08-23 18:15 "PJ run" (logs/djr3x-2026-08-23-18-15-49.log).

PJ was pre-loaded as a person (facts, alias) but had NO biometrics. His
un-enrolled voice cross-matched Bret's print at 0.60–0.75 on short turns, so:

  1. 18:16:44 — Rex asked the visible unknown "What name should I save for
     you?"; PJ's reply "Call me Playa P" (ASR: "Call me. Play a P.") scored
     0.602 on Bret's print and was attributed to BRET, so the forced-enrollment
     path never saw an unknown speaker.
  2. 18:17:43 — during the intro voice-capture window PJ's "Hello." scored
     0.751 on Bret's print and the score-only "confidently the introducer"
     guard refused to enroll him until the window expired. PJ then read as
     Bret for the rest of the night.
  3. 18:26:10 — Exudica's "Oh, nothing. I'm headed home." was parsed as a
     self-introduction and phantom person "Headed Home" was minted with her
     real voice and face.

Fixes under test: identity-prompt window demotion of sub-confident off-camera
voice matches; camera-contradiction override in the intro voice-capture guard;
ASR-period-tolerant "call me" extraction; verb-phrase name rejection.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

import config
from intelligence import interaction as I
from memory.name_validation import normalize_person_name


class IdentityPromptDemotionTest(unittest.TestCase):
    """_identity_prompt_demotes_voice_match — pure predicate for the reply window."""

    def _demotes(self, **kw):
        base = dict(
            person_id=1,                 # matched Bret's print
            speaker_score=0.602,          # PJ's field score on it
            visible_known_ids=set(),      # Bret never face-identified yet
            matched_visible_recently=False,
            unknown_visible_or_recent=True,  # PJ's unknown face in frame
            visual_speaker_pid=None,
            text="Call me. Play a P.",
            text_input=False,
        )
        base.update(kw)
        return I._identity_prompt_demotes_voice_match(**base)

    def test_pj_field_shape_demotes(self):
        # The exact 18:16:44 turn: sub-confident cross-match, matched person
        # unseen, unknown face visible, first-person reply → demote.
        self.assertTrue(self._demotes())

    def test_confident_match_is_never_demoted(self):
        # Genuine Bret landed 0.828–0.888 in the same session.
        self.assertFalse(self._demotes(speaker_score=0.828))

    def test_third_party_reply_keeps_known_speaker(self):
        # "This is PJ" is the introducer answering FOR the newcomer — the
        # describe-newcomer path needs the speaker to stay Bret so Bret's voice
        # is never bound to PJ.
        self.assertFalse(self._demotes(text="This is PJ."))

    def test_first_person_wins_over_third_party_marker(self):
        self.assertTrue(self._demotes(text="I'm PJ, this is my dog Bella."))

    def test_matched_person_on_camera_is_not_demoted(self):
        self.assertFalse(self._demotes(visible_known_ids={1}))

    def test_matched_person_recently_visible_is_not_demoted(self):
        self.assertFalse(self._demotes(matched_visible_recently=True))

    def test_visual_speaker_confirming_match_is_not_demoted(self):
        self.assertFalse(self._demotes(visual_speaker_pid=1))

    def test_no_unknown_face_means_no_demotion(self):
        self.assertFalse(self._demotes(unknown_visible_or_recent=False))

    def test_gui_text_input_is_not_demoted(self):
        self.assertFalse(self._demotes(text_input=True))




class CallMeAsrPeriodTest(unittest.TestCase):
    """ASR closes the sentence after the verb: 'Call me. Play a P.'"""

    def test_split_call_me_still_yields_a_name(self):
        name = I._extract_introduced_name("Call me. Play a P.", allow_bare_name=True)
        self.assertTrue(name)
        self.assertTrue(name.startswith("Play"))
        self.assertNotIn("call", name.lower())

    def test_clean_call_me_unchanged(self):
        self.assertEqual(
            I._extract_introduced_name("Call me Playa P", allow_bare_name=True),
            "Playa P",
        )

    def test_normalizer_survives_leading_period_after_call_me(self):
        self.assertEqual(normalize_person_name("Call me. Playa P"), "Playa P")


class HeadedHomePhantomTest(unittest.TestCase):
    """'I'm headed home' is a departure, not a self-introduction."""

    def test_headed_home_is_not_a_self_intro(self):
        self.assertIsNone(
            I._extract_self_identified_name("Oh, nothing. I'm headed home.")
        )

    def test_leaving_now_is_not_a_self_intro(self):
        self.assertIsNone(I._extract_self_identified_name("I'm leaving now."))

    def test_normalizer_rejects_headed_home(self):
        self.assertIsNone(normalize_person_name("headed home"))
        self.assertIsNone(normalize_person_name("Home"))

    def test_real_names_still_pass(self):
        # Ted / Heather / Homer share prefixes with the new stop tokens.
        self.assertEqual(I._extract_self_identified_name("I'm Ted."), "Ted")
        self.assertEqual(I._extract_self_identified_name("I'm Heather."), "Heather")
        self.assertEqual(normalize_person_name("Homer"), "Homer")


if __name__ == "__main__":
    unittest.main()
