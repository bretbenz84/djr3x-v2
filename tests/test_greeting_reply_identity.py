"""The 2026-09-09 Americus reply must not lose Rex's just-spoken greeting."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from intelligence import attribution as A


TEXT = "Yeah, I'm in Americus, Georgia right now. How are you doing?"


def face(pid=1):
    return {"person_db_id": pid, "face_visible": True, "face_id": "Bret Benziger"}


def evidence(**changes):
    return replace(A.UtteranceEvidence(
        text=TEXT, words=11, voiced_secs=4.50,
        started_at=110., ended_at=114.91,
        raw_best_id=1, raw_best_name="Bret Benziger", raw_best_score=.452,
        margin=1.452, required_margin=.07, hard_threshold=.50, known_floor=.45,
        accept_tier="known_floor", scoreboard=[(1, "Bret Benziger", .452, 1)],
        visible_known_ids=[1], addressed_person_id=1, address_age_secs=10.,
        allow_short_continuity=True,
        face_observations=[{"monotonic_at": t, "faces": [face()]}
                           for t in (110., 111., 112., 113., 114., 114.91)],
    ), **changes)


class GreetingReplyTests(unittest.TestCase):
    def test_reported_voice_score_keeps_greeting_context_without_learning(self):
        result = A.resolve_authoritative(evidence())
        self.assertEqual(result.person_id, 1)
        self.assertEqual(result.basis, "reply to recently addressed visible person")
        self.assertFalse(result.as_dict()["learning_allowed"])

    def test_no_greeting_does_not_relax_long_voice_acceptance(self):
        for change in ({"addressed_person_id": None}, {"addressed_person_id": 2},
                       {"address_age_secs": None}, {"address_age_secs": 31.},
                       {"address_age_secs": -1.}, {"allow_short_continuity": False}):
            with self.subTest(change=change):
                self.assertIsNone(A.resolve_authoritative(evidence(**change)).person_id)

    def test_group_unknown_face_and_face_departure_cannot_inherit_greeting(self):
        for faces in ([face(), face(2)], [face(), {"face_visible": True}], [], [face(2)]):
            rows = list(evidence().face_observations)
            rows[2] = dict(rows[2], faces=faces)
            with self.subTest(faces=faces):
                self.assertIsNone(A.resolve_authoritative(evidence(face_observations=rows)).person_id)
        self.assertIsNone(A.resolve_authoritative(evidence(visible_known_ids=[1, 2])).person_id)
        self.assertIsNone(A.resolve_authoritative(evidence(visible_known_ids=[])).person_id)

    def test_missing_stale_sparse_or_duplicate_snapshots_fail_closed(self):
        rows = evidence().face_observations
        for change in (
            {"face_observations": []},
            {"face_observations": rows[:1]},
            {"face_observations": rows[:2] + rows[3:]},
            {"face_observations": [dict(r, monotonic_at=r["monotonic_at"]-10) for r in rows]},
            {"face_observations": [dict(r, monotonic_at=None) for r in rows]},
            {"face_observations": [rows[0]]*6},
            {"started_at": None}, {"ended_at": None},
        ):
            with self.subTest(change=change):
                self.assertIsNone(A.resolve_authoritative(evidence(**change)).person_id)

    def test_conflicting_voice_direction_and_mixed_speakers_do_not_inherit_name(self):
        for change in ({"raw_best_id": 2}, {"raw_best_score": .34}, {"margin": .01},
                       {"bearing_contradiction": True}, {"bearing_selected_pid": 2},
                       {"mixed_speakers": True}):
            with self.subTest(change=change):
                self.assertIsNone(A.resolve_authoritative(evidence(**change)).person_id)

    def test_strong_voice_still_uses_normal_recognition(self):
        result = A.resolve_authoritative(evidence(raw_best_score=.8, accept_tier="hard"))
        self.assertTrue(result.learning_allowed)
        self.assertEqual(result.basis, "strong voice")

    def test_unreliable_mouth_output_cannot_supply_or_veto_greeting_context(self):
        ev = evidence(visual_observations=[{"person_db_id": 2}, {"person_db_id": 3}])
        self.assertEqual(A.resolve_authoritative(ev).person_id, 1)
        # Sole-face evidence has to come from the independent capture history.
        ev = evidence(face_observations=[], visual_observations=evidence().face_observations)
        self.assertIsNone(A.resolve_authoritative(ev).person_id)

    def test_lean_context_explains_greeting_without_authorizing_personal_memory(self):
        from intelligence import conversation_state as CS
        self.addCleanup(CS.clear)
        CS.note_speaker_resolution(A.resolve_authoritative(evidence()).as_dict())
        text = " ".join(CS.speaker_lines())
        self.assertIn("just addressed Bret", text)
        self.assertIn("do not ask who's speaking", text)
        self.assertIn("not a verified voice", text)
        self.assertIn("or save personal facts", text)


class CaptureWiringTests(unittest.TestCase):
    def setUp(self):
        from intelligence import interaction as I, dialogue_act as DA
        self.I, self.DA = I, DA
        DA.clear(); self.addCleanup(DA.clear)
        self.enterContext(patch.object(I.time, "monotonic", return_value=115.))
        self.enterContext(patch.object(I.speaker_id, "active_backend", return_value="campplus"))
        self.enterContext(patch.object(I.speaker_id.voice_score, "match_threshold", return_value=.5))
        self.enterContext(patch.object(I, "_last_scan_secs", {"voiced": 4.5}))
        self.enterContext(patch.object(I, "_last_scan_windows", []))
        self.enterContext(patch.object(I, "_last_scan_ranked", evidence().scoreboard))
        self.enterContext(patch.object(I, "_current_turn_speaker_evidence", {}))
        self.enterContext(patch.object(I, "_utterance_observations", {
            "started_at": 110., "ended_at": 114.91, "faces": evidence().face_observations,
            "visual": [],  # The mouth model contributes nothing.
        }))
        self.transcript = self.enterContext(patch.object(I.conv_memory, "get_session_transcript", return_value=[]))
        self.frame = DA.note_rex_turn("Hey Bret, back again. Good.", source="presence_reaction",
                                     target_person_id=1)
        self.frame.created_at = 100.

    def resolve(self, previous=None):
        return self.I._resolve_turn_attribution(
            turn_id=1, text=TEXT, text_input=False, raw_best_id=1,
            raw_best_name="Bret Benziger", speaker_score=.452, speaker_margin=1.452,
            required_margin=.07, accept_tier="known_floor", identity_resolution=None,
            person_id=None, person_name=None, off_camera_unknown=True,
            visible_known_ids=[1], bearing_match=None, engaged=None, previous_speaker=previous)

    def test_latest_targeted_greeting_and_independent_capture_reach_resolver(self):
        result = self.resolve()
        self.assertEqual(result.person_id, 1)
        self.assertFalse(result.learning_allowed)

    def test_capture_before_greeting_cannot_answer_it(self):
        self.frame.created_at = 111.
        self.assertIsNone(self.resolve().person_id)

    def test_answered_expired_or_superseded_greeting_cannot_be_reused(self):
        self.frame.answered_at = 105.
        self.assertIsNone(self.resolve().person_id)
        self.frame.answered_at = None
        self.frame.created_at = 70.
        self.assertIsNone(self.resolve().person_id)
        self.frame.created_at = 100.
        self.DA.note_rex_turn("Look at that.")
        self.assertIsNone(self.resolve().person_id)

    def test_other_human_turn_breaks_the_greeting_reply_handoff(self):
        self.assertIsNone(self.resolve({"person_id": 2, "at": 105.}).person_id)
        self.transcript.return_value = [{"speaker": "unknown_voice_1",
                                         "recorded_at_monotonic": 105.}]
        self.assertIsNone(self.resolve().person_id)
