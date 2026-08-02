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


class ShortUtteranceDetectionTest(unittest.TestCase):
    """`short_utterance` must reflect actual SPEECH, not padded buffer length.

    VAD pre/post-roll silence pads a 2-word reply past the seconds threshold, so
    word count is the padding-proof backstop that keeps a brief turn flagged short.
    """

    _SR = int(getattr(__import__("config"), "AUDIO_SAMPLE_RATE", 16000) or 16000)

    def _buf(self, secs: float) -> np.ndarray:
        return np.zeros(int(secs * self._SR), dtype=np.float32)

    def test_padded_two_word_reply_is_short_by_word_count(self):
        # Field 2026-07-23: "It's wine" sat in a >2s buffer but is 2 words.
        self.assertTrue(I._is_short_utterance(self._buf(2.4), "It's wine"))

    def test_long_many_word_utterance_is_not_short(self):
        self.assertFalse(
            I._is_short_utterance(self._buf(4.0), "happy fourth of july to everyone")
        )

    def test_brief_buffer_is_short_regardless_of_words(self):
        self.assertTrue(I._is_short_utterance(self._buf(1.0), "yep"))

    def test_long_buffer_without_transcript_is_not_short(self):
        self.assertFalse(I._is_short_utterance(self._buf(4.0), None))


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
        self.assertEqual(self._decide(person_id=1, raw_best_id=1, speaker_score=0.78), "voice_agrees")

    # ── Marginal match on the VISIBLE face (owner architecture call 2026-07-05:
    #    the camera never upgrades a marginal voice — JT's "happy 4th" at 0.628
    #    on Bret's print was credited to silently-on-camera Bret) ──────────────
    def test_marginal_match_on_visible_face_challenges_without_credibility(self):
        # 0.628 on the visible face, no continuity, no camera confirmation → ASK.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.628),
            "challenge_identity",
        )

    def test_marginal_match_passes_with_voice_continuity(self):
        # Same score, but this person's own voice was confidently matched minutes
        # ago — their voice trailing into a short turn. Attribute, never refresh.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.628,
                         voice_continuity=True),
            "voice_agrees_no_refresh",
        )

    def test_marginal_match_passes_in_ecapa_genuine_band(self):
        # First short turn of an ECAPA session (live-logged 2026-07-07: "yup, I'm
        # back" at 0.597 with the right face on camera got "who's speaking?"):
        # a genuine-band score on the visible face attributes without continuity.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.597,
                         score_genuine_band=True),
            "voice_agrees_no_refresh",
        )

    def test_marginal_match_passes_when_camera_confirms_talking(self):
        # The visual active-speaker latch positively says the visible face is the
        # one talking — the camera CAN corroborate, it just can't upgrade alone.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.628,
                         visual_speaker_pid=1),
            "voice_agrees",
        )

    def test_confident_match_needs_no_credibility(self):
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.76),
            "voice_agrees",
        )

    def test_voice_matched_someone_else_wins_over_face(self):
        # Voice CONFIDENTLY matched person 2 (off-camera); person 1 is the visible
        # face. Confident voice is primary — keep person 2.
        self.assertEqual(self._decide(person_id=2, raw_best_id=2, speaker_score=0.78), "voice_over_face")

    def test_confident_voice_over_face_boundary(self):
        # Exactly at the confident threshold (0.75) the voice still wins.
        self.assertEqual(self._decide(person_id=2, raw_best_id=2, speaker_score=0.75), "voice_over_face")

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

    # ── Short utterances (field 2026-07-18: Bret's "Yep" scored 0.332 on his
    #    OWN print — ECAPA can't score a sub-2s clip — and the session was
    #    de-personed to unknown_voice_1 with his face locked on camera) ────────
    def test_short_utterance_agreeing_with_face_wins(self):
        self.assertEqual(
            self._decide(raw_best_id=1, speaker_score=0.332, engaged_is_visible=False,
                         short_utterance=True),
            "short_face_wins",
        )

    def test_short_utterance_no_candidate_still_wins(self):
        self.assertEqual(
            self._decide(raw_best_id=None, speaker_score=0.0, engaged_is_visible=False,
                         short_utterance=True),
            "short_face_wins",
        )

    def test_short_utterance_pointing_elsewhere_stays_off_screen(self):
        # The voice actively leans toward someone NOT in frame — short or not,
        # never credit the visible face.
        self.assertEqual(
            self._decide(raw_best_id=2, speaker_score=0.40, engaged_is_visible=False,
                         short_utterance=True),
            "off_screen_unknown",
        )

    def test_short_utterance_camera_shows_other_talker_stays_off_screen(self):
        self.assertEqual(
            self._decide(raw_best_id=1, speaker_score=0.30, engaged_is_visible=False,
                         short_utterance=True, visual_speaker_pid=3),
            "off_screen_unknown",
        )

    def test_long_utterance_same_score_stays_off_screen(self):
        # A LONG clip at 0.33 is genuine evidence of a stranger — unchanged.
        self.assertEqual(
            self._decide(raw_best_id=1, speaker_score=0.332, engaged_is_visible=False,
                         short_utterance=False),
            "off_screen_unknown",
        )

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

    # ── The embedder-migration deadlock (live-logged 2026-07-06-21-15): every
    #    stored print is stale, so raw_best is None on EVERY turn; at session
    #    start engagement hasn't formed, and without attribution it never forms.
    #    The camera's active-speaker confirmation substitutes for engagement. ──
    def test_no_voice_signal_camera_confirms_talker_falls_back_to_face(self):
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=None,
                         engaged_is_visible=False, visual_speaker_pid=1),
            "face_only_continuity",
        )

    def test_no_voice_signal_camera_shows_someone_else_stays_off_screen(self):
        # The camera's recent talker is a different person — do not claim the face.
        self.assertEqual(
            self._decide(person_id=None, raw_best_id=None,
                         engaged_is_visible=False, visual_speaker_pid=2),
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


class EcapaGenuineBandTest(unittest.TestCase):
    """ECAPA-gated trust floors: under ECAPA an impostor cross-match lands ~0.25-0.45
    mapped (below the accept threshold), so an accepted match is genuine-band evidence
    and the who's-that challenges stand down. The Resemblyzer fallback — where impostors
    land 0.55-0.66 — keeps the strict 2026-07-05 guards."""

    def _backend(self, name):
        from audio import voice_score
        return mock.patch.object(voice_score, "active_backend", return_value=name)

    def test_genuine_band_requires_ecapa_backend(self):
        with self._backend("ecapa"):
            self.assertTrue(I._ecapa_genuine_band(0.597, 0.50))
        with self._backend("resemblyzer"):
            self.assertFalse(I._ecapa_genuine_band(0.597, 0.50))

    def test_genuine_band_respects_floor(self):
        with self._backend("ecapa"):
            self.assertFalse(I._ecapa_genuine_band(0.48, 0.50))
            self.assertTrue(I._ecapa_genuine_band(0.50, 0.50))

    def test_kill_switch(self):
        import config
        with self._backend("ecapa"), \
             mock.patch.object(config, "SPEAKER_ID_ECAPA_TRUST_ENABLED", False, create=True):
            self.assertFalse(I._ecapa_genuine_band(0.9, 0.50))

    def test_voice_only_challenge_stands_down_in_genuine_band(self):
        # Session-start short turn, nobody's face resolved: a genuine-band ECAPA
        # score must NOT trigger the who's-that challenge.
        I._last_voice_challenge_at = 0.0
        with self._backend("ecapa"), \
             mock.patch.object(I, "_voice_continuity_active", return_value=False), \
             mock.patch.object(I, "_someone_visible_who_isnt", return_value=True):
            self.assertFalse(I._voice_only_attribution_suspect(1, 0.60))

    def test_voice_only_challenge_still_fires_under_resemblyzer(self):
        I._last_voice_challenge_at = 0.0
        with self._backend("resemblyzer"), \
             mock.patch.object(I, "_voice_continuity_active", return_value=False), \
             mock.patch.object(I, "_someone_visible_who_isnt", return_value=True):
            self.assertTrue(I._voice_only_attribution_suspect(1, 0.60))
        I._last_voice_challenge_at = 0.0


class EnrollmentSeedsVoiceContinuityTest(unittest.TestCase):
    """A successful voice enrollment must stamp the continuity anchor: the saved
    sample IS this person's voice, ground truth. Without it, a fresh single-sample
    print scores marginal (~0.5) on the very next turn and the who's-that challenge
    fires seconds after the person said who they are (live-logged 2026-07-07:
    enrolled at 10:55:17, challenged 'who's talking?' at 10:56:05)."""

    def setUp(self):
        I._last_confident_voice_at.clear()
        self.audio = (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)

    def tearDown(self):
        I._last_confident_voice_at.clear()

    def test_successful_enrollment_activates_continuity(self):
        self.assertFalse(I._voice_continuity_active(7))
        with mock.patch.object(I, "_voice_enrollment_sample_allowed", return_value=(True, "")), \
             mock.patch.object(I.speaker_id, "enroll_voice", return_value=True):
            self.assertTrue(I._safe_enroll_voice(7, self.audio, source="new_person", confirmed=True))
        self.assertTrue(I._voice_continuity_active(7))

    def test_failed_enrollment_does_not_activate_continuity(self):
        with mock.patch.object(I, "_voice_enrollment_sample_allowed", return_value=(True, "")), \
             mock.patch.object(I.speaker_id, "enroll_voice", return_value=False):
            self.assertFalse(I._safe_enroll_voice(7, self.audio, source="new_person", confirmed=True))
        self.assertFalse(I._voice_continuity_active(7))

    def test_rejected_sample_does_not_activate_continuity(self):
        with mock.patch.object(I, "_voice_enrollment_sample_allowed", return_value=(False, "too_short")):
            self.assertFalse(I._safe_enroll_voice(7, self.audio, source="new_person", confirmed=True))
        self.assertFalse(I._voice_continuity_active(7))

    def test_marginal_next_turn_attributes_instead_of_challenging(self):
        # The end-to-end shape of the 2026-07-07 failure: freshly enrolled person is
        # the single visible face, next turn scores marginal on their own print —
        # with the enrollment-seeded anchor the decision attributes (no refresh)
        # instead of challenging identity.
        with mock.patch.object(I, "_voice_enrollment_sample_allowed", return_value=(True, "")), \
             mock.patch.object(I.speaker_id, "enroll_voice", return_value=True):
            I._safe_enroll_voice(1, self.audio, source="new_person", confirmed=True)
        decision = I._voice_primary_face_decision(
            person_id=1,
            raw_best_id=1,
            speaker_score=0.53,
            ws_pid=1,
            single_visible=True,
            engaged_is_visible=True,
            unknown_visible=False,
            other_known_recently=False,
            voice_continuity=I._voice_continuity_active(1),
        )
        self.assertEqual(decision, "voice_agrees_no_refresh")


class VoiceprintPollutionGuardTest(unittest.TestCase):
    """A face-confirmed refresh must only append audio that BOTH the VOICE attributes
    to that person AND the camera confirms that person actually spoke. Two guards:
    (1) the voice's best candidate must be this person; (2) the visual active-speaker
    latch must confirm this person is the on-camera talker. A visible face is NOT proof
    of speaking — a 3rd-party/AI voice (ChatGPT, TV, off-camera person) that scores onto
    a visible person's print must not poison it.

    EXCEPTION — bootstrap: while a print is empty/thin (below the sample floor), Guard 1 is
    relaxed so a person whose voice currently matches SOMEONE ELSE (precisely because their
    print is thin) can build one. Guard 2 still holds, so only camera-confirmed audio seeds it."""

    def setUp(self):
        I._voice_refreshed_this_session.clear()

    def _enrolled(self, *, current, **kwargs) -> bool:
        """Run _maybe_auto_refresh_voice with the async enroll executed inline; return whether
        an enrollment was actually attempted (i.e. it passed every guard)."""
        seen = []

        class _Inline:
            def __init__(self, *a, target=None, **k):
                self._t = target

            def start(self):
                if self._t:
                    self._t()

        # 3s of audible noise: passes the sample-quality gate (>=2.5s, RMS >= 0.008)
        # so these tests keep exercising the GUARDS, not the quality floor.
        audio = kwargs.pop("audio", None)
        if audio is None:
            audio = (0.05 * np.random.default_rng(0).standard_normal(48000)).astype(np.float32)
        # The refresh counts NATIVE-dimension prints (stale other-embedder rows are
        # invisible to it since the ECAPA migration).
        with mock.patch.object(I.people_memory, "count_native_voice_prints", return_value=current), \
             mock.patch.object(I, "_safe_enroll_voice", side_effect=lambda *a, **k: seen.append(1) or True), \
             mock.patch.object(I.threading, "Thread", _Inline):
            I._maybe_auto_refresh_voice(1, kwargs.pop("score", 0.5), audio, **kwargs)
        return bool(seen)

    def test_established_print_skips_when_voice_points_elsewhere(self):
        # Guard 1: an ESTABLISHED print (>= floor) is not touched when the voice matches someone else.
        self.assertFalse(self._enrolled(current=4, face_confirmed=True, raw_best_id=2, visual_speaker_pid=1))

    def test_short_shard_rejected_by_quality_gate(self):
        # A 1s VAD shard must never enter a print (dilution: measured -0.08 per-score
        # on a shard-fed centroid, 2026-07-05). Guards would otherwise pass this one.
        shard = (0.05 * np.random.default_rng(1).standard_normal(16000)).astype(np.float32)
        self.assertFalse(self._enrolled(current=4, face_confirmed=True, raw_best_id=1,
                                        visual_speaker_pid=1, audio=shard))

    def test_quiet_audio_rejected_by_quality_gate(self):
        # Long enough but near the noise floor — embeds badly, drags the centroid.
        quiet = (0.001 * np.random.default_rng(2).standard_normal(48000)).astype(np.float32)
        self.assertFalse(self._enrolled(current=4, face_confirmed=True, raw_best_id=1,
                                        visual_speaker_pid=1, audio=quiet))

    def test_thin_print_bootstraps_when_voice_points_elsewhere_but_camera_confirms(self):
        # THE FIX: 0 prints, voice matches the wrong person (that's WHY it's 0), camera confirms JT
        # is the on-camera talker -> seed the print anyway.
        self.assertTrue(self._enrolled(current=0, face_confirmed=True, raw_best_id=2, visual_speaker_pid=1))

    def test_bootstrap_still_requires_camera_confirmation(self):
        # Guard 2 is NOT bypassed by bootstrap: thin print + voice elsewhere + no camera confirm -> skip.
        self.assertFalse(self._enrolled(current=0, face_confirmed=True, raw_best_id=2, visual_speaker_pid=None))

    def test_refresh_skipped_when_visual_speaker_unconfirmed(self):
        # Guard 2 — the ChatGPT-voice-scores-onto-Bret case: voice agrees but camera doesn't confirm.
        self.assertFalse(self._enrolled(current=4, face_confirmed=True, raw_best_id=1, visual_speaker_pid=None))

    def test_refresh_skipped_when_visual_speaker_is_other_person(self):
        self.assertFalse(self._enrolled(current=4, face_confirmed=True, raw_best_id=1, visual_speaker_pid=2))

    def test_refresh_runs_when_voice_agrees_and_camera_confirms(self):
        self.assertTrue(self._enrolled(current=4, face_confirmed=True, raw_best_id=1, visual_speaker_pid=1))

    def test_refresh_runs_when_no_voice_candidate_and_camera_confirms(self):
        self.assertTrue(self._enrolled(current=4, face_confirmed=True, raw_best_id=None, visual_speaker_pid=1))

    def test_full_print_is_not_appended(self):
        # At the max-samples cap, nothing more is added regardless of guards.
        self.assertFalse(self._enrolled(current=5, face_confirmed=True, raw_best_id=1, visual_speaker_pid=1))

    def test_high_voice_score_path_unaffected_by_visual_gate(self):
        # The non-face-confirmed high-score (>=0.90) path is for off-camera speakers.
        self.assertTrue(self._enrolled(current=4, score=0.95, face_confirmed=False, raw_best_id=1, visual_speaker_pid=None))


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
        self.assertEqual(self._infer(speaker_score=0.79), "voice_confident")

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


class MouthStillVetoTest(unittest.TestCase):
    """Field 2026-08-02 12:37: JT spoke from ~20ft, cross-matched Bret's print
    at 0.455 (marginal), and silently-on-camera Bret got the credit via voice
    continuity. The active-speaker latch was EMPTY — Bret's mouth demonstrably
    wasn't moving — and that positive absence must veto the marginal accept."""

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

    def test_field_case_continuity_overridden_by_still_mouth(self):
        # JT at 0.455 on Bret's print, Bret visible + voice continuity active,
        # camera watched the mouth stay still → challenge, not attribute.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.455,
                         voice_continuity=True, visual_mouth_still=True),
            "challenge_identity",
        )

    def test_genuine_band_is_exempt(self):
        # Recalibrated 2026-08-02 13:04: the veto challenged genuine Bret at
        # 0.660 and 0.742 in one session (the detector misses real jaw motion
        # on short utterances at room distance). A genuine-band score plus the
        # face outweighs an empty latch — only sub-band scores get vetoed.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.660,
                         score_genuine_band=True, visual_mouth_still=True),
            "voice_agrees_no_refresh",
        )
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.742,
                         score_genuine_band=True, visual_mouth_still=True),
            "voice_agrees_no_refresh",
        )

    def test_confident_voice_is_exempt(self):
        # A confident voice stands on its own — a missed jaw sample must not
        # override a 0.8 match.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.80,
                         visual_mouth_still=True),
            "voice_agrees",
        )

    def test_no_veto_without_positive_absence(self):
        # Detector disabled/expired (visual_mouth_still=False): behavior unchanged.
        self.assertEqual(
            self._decide(person_id=1, raw_best_id=1, speaker_score=0.455,
                         voice_continuity=True),
            "voice_agrees_no_refresh",
        )

    def test_corroborate_becomes_challenge(self):
        # Weak lean toward the visible face would refresh the print — with a
        # still mouth that's how a stranger's audio poisons it. Ask instead.
        self.assertEqual(
            self._decide(raw_best_id=1, speaker_score=0.52,
                         visual_mouth_still=True),
            "challenge_identity",
        )

    def test_short_turn_exempt_from_veto(self):
        # One-word turns can slip between the detector's samples — the
        # short_face_wins protection stays even with an empty latch.
        self.assertEqual(
            self._decide(raw_best_id=None, speaker_score=0.30,
                         short_utterance=True, visual_mouth_still=True,
                         engaged_is_visible=False),
            "short_face_wins",
        )


if __name__ == "__main__":
    unittest.main()
