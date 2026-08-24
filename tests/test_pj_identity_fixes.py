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


class IntroCameraContradictionTest(unittest.TestCase):
    """_intro_camera_contradicts_introducer + its effect on the capture guard."""

    def _ctx(self):
        return {
            "introducer_id": 1,
            "introducer_name": "Bret Benziger",
            "introduced_id": 7,
            "introduced_name": "PJ Thomas",
            "relationship": "friend",
            "asked_at": time.monotonic(),
        }

    def test_contradiction_requires_unknown_face(self):
        # The phantom-"Leaf" shape: no newcomer visible → never contradict.
        with mock.patch.object(I, "_has_unknown_visible_person", return_value=False), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False):
            self.assertFalse(
                I._intro_camera_contradicts_introducer(1, 0.751, True)
            )

    def test_contradiction_blocked_by_visible_introducer(self):
        with mock.patch.object(I, "_has_unknown_visible_person", return_value=True), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=True):
            self.assertFalse(
                I._intro_camera_contradicts_introducer(1, 0.751, True)
            )

    def test_contradiction_blocked_above_ceiling(self):
        # Bret's genuine "What do you know about PJ?" landed 0.888 mid-window.
        with mock.patch.object(I, "_has_unknown_visible_person", return_value=True), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False):
            self.assertFalse(
                I._intro_camera_contradicts_introducer(1, 0.888, True)
            )

    def test_field_shape_contradicts(self):
        # PJ's "Hello." at 0.751: unknown face in frame, introducer unseen.
        with mock.patch.object(I, "_has_unknown_visible_person", return_value=True), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False):
            self.assertTrue(
                I._intro_camera_contradicts_introducer(1, 0.751, True)
            )

    def test_pj_hello_enrolls_newcomer_when_camera_contradicts(self):
        # The 18:17:43 turn end-to-end through _handle_intro_voice_capture.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=True), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll, \
             mock.patch.object(I, "_bind_intro_visible_face_if_present"), \
             mock.patch.object(I.llm, "get_response", return_value="PJ! Welcome."), \
             mock.patch.object(I.consciousness, "mark_engagement"), \
             mock.patch.object(I.consciousness, "note_person_spoke"):
            resp = I._handle_intro_voice_capture(
                "Hello.",
                audio,
                person_id=1,          # mis-resolved to the introducer (Bret)
                raw_best_id=1,
                speaker_score=0.751,  # cross-match above the 0.70 guard floor
            )
        self.assertTrue(resp)
        self.assertTrue(enroll.called)
        self.assertEqual(enroll.call_args.args[0], 7)  # enrolled PJ, not Bret

    def test_confident_introducer_still_blocked_without_camera_evidence(self):
        # The Leaf-band guard must hold exactly as before when the camera does
        # not contradict (no unknown face in frame).
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=False), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_intro_voice_capture(
                "Hello.",
                audio,
                person_id=1,
                raw_best_id=1,
                speaker_score=0.751,
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)

    def test_slam_dunk_introducer_score_still_blocked_with_unknown_visible(self):
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I, "_pending_intro_voice_capture", self._ctx()), \
             mock.patch.object(I, "_has_unknown_visible_person", return_value=True), \
             mock.patch.object(I, "_known_person_visible_recently", return_value=False), \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True) as enroll:
            resp = I._handle_intro_voice_capture(
                "What do you know about PJ?",
                audio,
                person_id=1,
                raw_best_id=1,
                speaker_score=0.888,  # genuine Bret, above the override ceiling
            )
        self.assertIsNone(resp)
        self.assertFalse(enroll.called)


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
