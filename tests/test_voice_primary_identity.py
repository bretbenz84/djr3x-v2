"""
Voice-primary identity resolution.

Rex must know WHO is speaking from the VOICE — even when he can't see them
(off-camera, group, crowded room). The visible face only CORROBORATES a weak or
absent voice match; it never OVERRIDES a voice that points at someone else, and
it never captures the turn for a person the voice does not point at. An
unrecognized voice is tracked as its own off-screen/anonymous identity instead of
being pinned on whoever happens to be on camera.

This replaces the prior "single visible face wins regardless of voice" rule.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from intelligence import interaction as I


class VoicePrimaryFaceDecisionTest(unittest.TestCase):
    """The pure attribution predicate used when exactly one known face is visible.
    ws_pid = the single visible known person (id 1). Voice candidate = raw_best_id."""

    def _decide(self, **kw):
        base = dict(
            person_id=None,
            raw_best_id=None,
            speaker_score=0.0,
            ws_pid=1,
            single_visible=True,
            engaged_is_visible=True,
            unknown_visible=False,
            other_known_recently=False,
        )
        base.update(kw)
        return I._voice_primary_face_decision(**base)

    def test_voice_and_face_agree(self):
        self.assertEqual(self._decide(person_id=1, raw_best_id=1, speaker_score=0.72), "voice_agrees")

    def test_voice_matched_someone_else_wins_over_face(self):
        # Voice CONFIDENTLY matched person 2 (off-camera); person 1 is the visible
        # face. Confident voice is primary — keep person 2.
        self.assertEqual(self._decide(person_id=2, raw_best_id=2, speaker_score=0.78), "voice_over_face")

    def test_confident_voice_over_face_boundary(self):
        # Exactly at the confident threshold (0.70) the voice still wins.
        self.assertEqual(self._decide(person_id=2, raw_best_id=2, speaker_score=0.70), "voice_over_face")

    def test_marginal_voice_elsewhere_no_visual_signal_keeps_face(self):
        # THE BUG: a marginal (<0.70) match to an off-camera person while a known
        # face is visible and the active-speaker latch is empty/unavailable
        # (visual_speaker_pid=None) must NOT override the visible face. This is the
        # logged failure: Bret's deleted print landed his voice on Wade at ~0.55.
        self.assertEqual(
            self._decide(person_id=2, raw_best_id=2, speaker_score=0.55, visual_speaker_pid=None),
            "voice_weak_face_wins",
        )

    def test_marginal_voice_elsewhere_camera_confirms_visible_face_keeps_face(self):
        # Camera shows the VISIBLE face (ws=1) is the one talking — a marginal
        # match to person 2 is a near-neighbor artifact, the face anchors identity.
        self.assertEqual(
            self._decide(person_id=2, raw_best_id=2, speaker_score=0.55, visual_speaker_pid=1),
            "voice_weak_face_wins",
        )

    def test_marginal_voice_elsewhere_camera_shows_other_talker_keeps_voice(self):
        # Camera's recent on-camera talker is NOT the visible known face (it is the
        # matched person 2, who was just on camera) — the visible face is not the
        # source, so trust the off-camera voice even at a marginal score.
        self.assertEqual(
            self._decide(person_id=2, raw_best_id=2, speaker_score=0.55, visual_speaker_pid=2),
            "voice_over_face",
        )

    def test_marginal_voice_elsewhere_camera_shows_third_party_keeps_voice(self):
        # Camera's recent talker (person 3) is neither the visible face nor the
        # matched person — the visible face is still not the source; trust voice.
        self.assertEqual(
            self._decide(person_id=2, raw_best_id=2, speaker_score=0.55, visual_speaker_pid=3),
            "voice_over_face",
        )

    def test_weak_voice_leaning_to_visible_is_corroborated(self):
        # Voice gave no accepted match (person_id None) but its best candidate is
        # the visible engaged person at a weak score — corroborate.
        self.assertEqual(self._decide(person_id=None, raw_best_id=1, speaker_score=0.42), "corroborate")

    def test_weak_voice_leaning_elsewhere_is_off_screen(self):
        # The voice leans toward someone NOT in frame — do not claim the visible
        # face; treat as off-screen/unknown.
        self.assertEqual(self._decide(person_id=None, raw_best_id=2, speaker_score=0.42), "off_screen_unknown")

    def test_no_voice_signal_clean_oneonone_falls_back_to_face(self):
        # No voice candidate at all, clean 1:1 with the engaged person on camera.
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=None, engaged_is_visible=True),
            "face_only_continuity",
        )

    def test_no_voice_signal_but_not_engaged_is_off_screen(self):
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=None, engaged_is_visible=False),
            "off_screen_unknown",
        )

    def test_unknown_face_present_defers_to_intro_path(self):
        # A newcomer is/was in frame — leave unresolved for the intro/identify
        # flow rather than marking off-screen.
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=1, speaker_score=0.42, unknown_visible=True),
            "unknown_intro_path",
        )

    def test_other_known_recently_blocks_corroboration(self):
        # Another known person flickered through recently — the scene is ambiguous,
        # so a weak lean toward the visible face is not enough.
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=1, speaker_score=0.42, other_known_recently=True),
            "off_screen_unknown",
        )

    def test_unengaged_visible_face_needs_higher_floor(self):
        # Not engaged: the weak lean must clear the higher engaged-visible floor
        # (0.50), not the 0.35 match floor.
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=1, speaker_score=0.42, engaged_is_visible=False),
            "off_screen_unknown",
        )
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=1, speaker_score=0.55, engaged_is_visible=False),
            "corroborate",
        )


class VoiceprintPollutionGuardTest(unittest.TestCase):
    """A face-confirmed refresh must only append audio that BOTH the VOICE attributes
    to that person AND the camera confirms that person actually spoke. Two guards:
    (1) the voice's best candidate must be this person; (2) the visual active-speaker
    latch must confirm this person is the on-camera talker. count_biometrics is only
    reached past both guards. A visible face is NOT proof of speaking — a 3rd-party/AI
    voice (ChatGPT, TV, off-camera person) that scores onto a visible person's print
    must not poison it."""

    def setUp(self):
        I._voice_refreshed_this_session.clear()

    def test_face_confirmed_refresh_skipped_when_voice_points_elsewhere(self):
        # Guard 1: the voice's best candidate is someone else.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics") as count:
            I._maybe_auto_refresh_voice(
                1, 0.40, audio, face_confirmed=True, raw_best_id=2, visual_speaker_pid=1
            )
        self.assertFalse(count.called)  # bailed before touching the print

    def test_face_confirmed_refresh_skipped_when_visual_speaker_unconfirmed(self):
        # Guard 2 — THE POISONING FIX: voice agrees (raw_best == person) and the
        # face is visible, but the camera does NOT confirm this person is talking
        # (empty/unavailable latch). This is the ChatGPT-voice-scores-onto-Bret case.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics") as count:
            I._maybe_auto_refresh_voice(
                1, 0.72, audio, face_confirmed=True, raw_best_id=1, visual_speaker_pid=None
            )
        self.assertFalse(count.called)

    def test_face_confirmed_refresh_skipped_when_visual_speaker_is_other_person(self):
        # Guard 2: the camera shows a DIFFERENT person is the on-camera talker.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics") as count:
            I._maybe_auto_refresh_voice(
                1, 0.72, audio, face_confirmed=True, raw_best_id=1, visual_speaker_pid=2
            )
        self.assertFalse(count.called)

    def test_face_confirmed_refresh_runs_when_voice_agrees_and_camera_confirms(self):
        # Both guards pass: voice agrees AND the camera confirms this person is talking.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics", return_value=0) as count, \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True):
            I._maybe_auto_refresh_voice(
                1, 0.40, audio, face_confirmed=True, raw_best_id=1, visual_speaker_pid=1
            )
        self.assertTrue(count.called)

    def test_face_confirmed_refresh_runs_when_no_voice_candidate_and_camera_confirms(self):
        # raw_best_id None == no contradicting candidate (guard 1 ok); camera confirms.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics", return_value=0) as count, \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True):
            I._maybe_auto_refresh_voice(
                1, 0.40, audio, face_confirmed=True, raw_best_id=None, visual_speaker_pid=1
            )
        self.assertTrue(count.called)

    def test_high_voice_score_path_unaffected_by_visual_gate(self):
        # The non-face-confirmed high-score (>=0.90) path is for off-camera speakers
        # and must NOT require visual confirmation.
        audio = np.zeros(16000, dtype=np.float32)
        with mock.patch.object(I.people_memory, "count_biometrics", return_value=0) as count, \
             mock.patch.object(I, "_safe_enroll_voice", return_value=True):
            I._maybe_auto_refresh_voice(
                1, 0.95, audio, face_confirmed=False, raw_best_id=1, visual_speaker_pid=None
            )
        self.assertTrue(count.called)


class IdentityResolutionStrategyTest(unittest.TestCase):
    def _infer(self, **kw):
        base = dict(
            person_id=1,
            raw_best_id=1,
            speaker_score=0.55,
            hard_threshold=0.50,
            soft_threshold=0.60,
            sticky_accepted=False,
            ws_identified=[],
            recent_engagement=None,
            off_camera_unknown=False,
        )
        base.update(kw)
        return I._infer_identity_resolution_strategy(**base)

    def test_confident_voice_label(self):
        self.assertEqual(self._infer(speaker_score=0.74), "voice_confident")

    def test_accepted_voice_label(self):
        self.assertEqual(self._infer(speaker_score=0.55), "voice_match")

    def test_corroborated_label(self):
        self.assertEqual(self._infer(speaker_score=0.40), "voice_corroborated_by_face")

    def test_off_camera_unknown_label(self):
        self.assertEqual(
            self._infer(person_id=None, raw_best_id=None, off_camera_unknown=True),
            "off_camera_unknown",
        )

    def test_face_only_continuity_label(self):
        # Resolved to a visible person the voice did NOT point at.
        self.assertEqual(
            self._infer(
                person_id=1,
                raw_best_id=2,
                speaker_score=0.0,
                ws_identified=[{"person_db_id": 1}],
            ),
            "face_only_continuity",
        )


class VisualCorroborationTest(unittest.TestCase):
    """Commit 7: the multi-visible tie-breaker. A weak voice that leans toward a
    visible known person is accepted at a lower floor ONLY when the camera saw
    exactly that person speaking. Vision confirms; it never overrides or invents."""

    VISIBLE = {1, 2}

    def _decide(self, **kw):
        base = dict(
            raw_best_id=1,
            speaker_score=0.40,
            visible_known_ids=self.VISIBLE,
            visual_speaker_pid=1,
            floor=0.35,
        )
        base.update(kw)
        return I._visual_corroborated_speaker(**base)

    def test_voice_and_visual_agree_on_visible_person(self):
        self.assertEqual(self._decide(), 1)

    def test_visual_speaker_is_a_different_person_abstains(self):
        # Voice leans toward 1 but the camera saw 2 speaking — disagree → no attribution.
        self.assertIsNone(self._decide(raw_best_id=1, visual_speaker_pid=2))

    def test_no_visual_speaker_abstains(self):
        self.assertIsNone(self._decide(visual_speaker_pid=None))

    def test_voice_leans_to_someone_not_visible_abstains(self):
        self.assertIsNone(self._decide(raw_best_id=3, visual_speaker_pid=3))

    def test_below_floor_abstains(self):
        self.assertIsNone(self._decide(speaker_score=0.30))

    def test_no_voice_candidate_abstains(self):
        self.assertIsNone(self._decide(raw_best_id=None, visual_speaker_pid=None))


if __name__ == "__main__":
    unittest.main()
