"""Voice-only attribution challenge + speaker-attribution correction.

Field log 2026-07-05: the camera was turned toward JT (face NOT recognized, real pose
visible), JT spoke, and his voice cross-matched Bret's voiceprint at 0.660 (JT has no
print of his own) — the turn was attributed to Bret, who wasn't even in frame. And
Bret's explicit correction ("That was JT speaking") was bantered at without fixing the
record. These tests pin both fixes.
"""

import unittest
from unittest import mock

import config
from intelligence import consciousness
from intelligence import interaction


def _suspect(person_id=1, score=0.660, *, voice_continuity=False, others_visible=True,
             enabled=True, last_challenge=0.0, empty_frame_challenge=True):
    """Drive _voice_only_attribution_suspect with fully mocked surroundings."""
    people = [{"person_db_id": None, "face_missing": True}] if others_visible else []
    with mock.patch.object(config, "SPEAKER_ID_UNSEEN_CHALLENGE_ENABLED", enabled, create=True), \
         mock.patch.object(config, "SPEAKER_ID_CHALLENGE_EMPTY_FRAME", empty_frame_challenge, create=True), \
         mock.patch.object(interaction, "_voice_continuity_active", return_value=voice_continuity), \
         mock.patch.object(interaction.world_state, "get", return_value=people), \
         mock.patch.object(interaction, "_last_voice_challenge_at", last_challenge):
        return interaction._voice_only_attribution_suspect(person_id, score)


class VoiceChallengeGateTest(unittest.TestCase):
    def test_field_shape_triggers_challenge(self):
        # marginal 0.660, Bret not on camera, a pose-only body visible -> challenge
        self.assertTrue(_suspect())

    def test_confident_voice_is_trusted(self):
        self.assertFalse(_suspect(score=0.75))

    def test_voice_continuity_suppresses_challenge(self):
        # Bret spoke CONFIDENTLY minutes ago; his marginal follow-up turns pass.
        # (v1 suppressed on face-recency instead — field log 2026-07-05-02-37: Bret's
        # face at session start shielded every JT turn, including 'this is JT' soft-
        # accepted as Bret at 0.483. Being in the room is not evidence of talking.)
        self.assertTrue(_suspect(voice_continuity=False))
        self.assertFalse(_suspect(voice_continuity=True))

    def test_sub_threshold_soft_accept_is_challenged(self):
        # The exact field shape: 0.483 soft-accepted (floor 0.45) with no continuity.
        self.assertTrue(_suspect(score=0.483))

    def test_empty_frame_now_challenges_by_default(self):
        # Owner call 2026-07-05: an unseen marginal match with an EMPTY frame is still
        # the cross-match shape — ask instead of silently crediting the match.
        self.assertTrue(_suspect(others_visible=False))

    def test_empty_frame_challenge_can_be_disabled(self):
        self.assertFalse(_suspect(others_visible=False, empty_frame_challenge=False))

    def test_kill_switch(self):
        self.assertFalse(_suspect(enabled=False))

    def test_cooldown_suppresses_rapid_rechallenge(self):
        import time as _time
        self.assertFalse(_suspect(last_challenge=_time.monotonic()))


class ChallengedSelfIdTest(unittest.TestCase):
    """'This is JT' from a challenged unknown voice must resolve inline — never
    'Nice to meet you, JT — who's speaking?' (field log 2026-07-05-02-51)."""

    def test_self_identifications_extract(self):
        for text, want in [("This is JT", "JT"), ("this is JT speaking", "JT"),
                           ("It's Joy", "Joy"), ("I'm JT", "JT")]:
            self.assertEqual(interaction._challenged_self_identified_name(text), want, text)

    def test_interjections_do_not_extract(self):
        for text in ("this is ridiculous", "this is amazing", "it's fine",
                     "this is so cool", "this is my house"):
            self.assertIsNone(interaction._challenged_self_identified_name(text), text)


class WarmSignatureResolveTest(unittest.TestCase):
    """A just-linked signature resolves at the warm bar; old ones need the cold bar
    (field log: JT re-became unknown_voice_2 at 0.758 fifteen seconds after naming)."""

    def _iso(self, secs_ago):
        from datetime import datetime, timezone, timedelta
        return (datetime.now(timezone.utc) - timedelta(seconds=secs_ago)).isoformat()

    def test_warm_signature_resolves_at_lower_bar(self):
        self.assertTrue(interaction._signature_resolves_to_person(0.758, self._iso(15)))

    def test_cold_signature_needs_strict_bar(self):
        self.assertFalse(interaction._signature_resolves_to_person(0.758, self._iso(172800)))
        self.assertTrue(interaction._signature_resolves_to_person(0.83, self._iso(172800)))

    def test_weak_score_never_resolves(self):
        self.assertFalse(interaction._signature_resolves_to_person(0.65, self._iso(15)))

    def test_garbage_timestamp_is_cold(self):
        self.assertFalse(interaction._signature_resolves_to_person(0.758, None))
        self.assertFalse(interaction._signature_resolves_to_person(0.758, "not-a-date"))


class AmbiguousBetweenKnownsTest(unittest.TestCase):
    """Two enrolled voices near-tied is a challenge, not Guest-N churn
    (field log 2026-07-05-21-22: JT 0.563 vs Bret 0.529, margin 0.034)."""

    def test_field_shape_is_ambiguous(self):
        self.assertTrue(interaction._voice_ambiguous_between_knowns(0.563, 0.034))

    def test_clear_winner_is_not_ambiguous(self):
        self.assertFalse(interaction._voice_ambiguous_between_knowns(0.712, 0.235))

    def test_below_threshold_is_not_ambiguous(self):
        # A sub-threshold top score is a plain unknown, not an enrolled near-tie.
        self.assertFalse(interaction._voice_ambiguous_between_knowns(0.42, 0.01))

    def test_single_enrolled_person_is_not_ambiguous(self):
        # With one print the margin is score+1 (no second candidate) — never ambiguous.
        self.assertFalse(interaction._voice_ambiguous_between_knowns(0.563, 1.563))


class SpeakerCorrectionPatternTest(unittest.TestCase):
    def _name(self, text):
        m = interaction._SPEAKER_CORRECTION_PAT.match(text.strip())
        if not m:
            return None
        if not (interaction._SPEAKER_CORRECTION_PREFIX_PAT.match(text.strip())
                or interaction._SPEAKER_CORRECTION_TAIL_PAT.search(text.strip())):
            return None
        name = (m.group("name") or "").strip().rstrip(".!,")
        if not name or name.lower() in interaction._CORRECTION_NON_NAMES:
            return None
        return name

    def test_corrections_match(self):
        self.assertEqual(self._name("That was JT speaking"), "JT")
        self.assertEqual(self._name("that was JT talking."), "JT")
        self.assertEqual(self._name("no, that wasn't me, that was JT"), "JT")
        self.assertEqual(self._name("That wasn't me, that was Joy speaking"), "Joy")

    def test_non_corrections_do_not_match(self):
        self.assertIsNone(self._name("that was great"))
        self.assertIsNone(self._name("that was JT's idea"))
        self.assertIsNone(self._name("that was Jeff"))          # ambiguous, no signal word
        self.assertIsNone(self._name("that was me talking"))
        self.assertIsNone(self._name("I was talking to JT"))


class RelabelPriorTurnTest(unittest.TestCase):
    def test_relabel_moves_attribution_and_skips_the_correction_itself(self):
        from memory import conversations as conv
        conv.clear_transcript()
        conv.add_to_transcript("Bret Benziger", "That's stretch right")     # misattributed
        conv.add_to_transcript("Rex", "A little, yeah.")
        conv.add_to_transcript("Bret Benziger", "That was JT speaking")     # the correction
        moved = conv.relabel_prior_turn("Bret Benziger", "JT",
                                        skip_text="That was JT speaking")
        self.assertTrue(moved)
        t = conv.get_session_transcript()
        self.assertEqual(t[0]["speaker"], "JT")                  # moved
        self.assertFalse(t[0]["learnable"])                      # disputed line: don't learn
        self.assertEqual(t[2]["speaker"], "Bret Benziger")       # correction untouched
        conv.clear_transcript()


if __name__ == "__main__":
    unittest.main()
