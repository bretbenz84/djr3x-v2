"""Lean Brain phase 4 (ActionResult + bounded heading alternatives) and phase 2B
(utterance evidence + shadow attribution resolver + learning gates).

No hardware: motion.send / telemetry / compass are mocked; the resolver is pure.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

import config
from intelligence import action_result as AR, attribution as AT, conversation_state as CS
from intelligence import motion_controller as mc


class ActionResultTest(unittest.TestCase):
    def test_finish_and_shrunk(self):
        r = AR.ActionResult("turn", "left 90°", seq=5, requested_deg=90.0, attempted_deg=62.0)
        self.assertEqual(r.status, "running")
        self.assertTrue(r.shrunk)
        r.finish("blocked", reason="obstacle")
        self.assertEqual(r.status, "blocked")
        self.assertIsNotNone(r.ended_at)
        self.assertFalse(r.ok)
        d = r.as_dict()
        self.assertTrue(d["shrunk"])
        self.assertEqual(d["seq"], 5)

    def test_unknown_status_kept_verbatim(self):
        r = AR.ActionResult("move")
        r.finish("weird_firmware_word")
        self.assertEqual(r.status, "weird_firmware_word")


class ConversationStateActionsTest(unittest.TestCase):
    def setUp(self):
        CS.clear()
        self.addCleanup(CS.clear)

    def test_verified_short_turn_is_partial_and_rendered(self):
        CS.note_action_issued(9, "turn", "left 90°", requested_deg=90.0, attempted_deg=90.0)
        CS.note_action_result(9, "completed")
        with mock.patch.object(config, "MOTION_COMPASS_TURN_TOLERANCE_DEG", 4.0, create=True):
            CS.note_action_verified(9, requested_deg=90.0, measured_deg=61.0)
        rec = CS.recent_actions()[0]
        self.assertEqual(rec["status"], "partial")
        line = CS.render_lines(None)[0]
        self.assertIn("landed short/long", line)
        self.assertIn("61°", line)

    def test_verified_within_tolerance_stays_completed(self):
        CS.note_action_issued(10, "turn", "left 90°", requested_deg=90.0, attempted_deg=90.0)
        CS.note_action_result(10, "completed")
        CS.note_action_verified(10, requested_deg=90.0, measured_deg=88.0)
        self.assertEqual(CS.recent_actions()[0]["status"], "completed")
        self.assertIn("compass measured 88°", CS.render_lines(None)[0])

    def test_shrunk_and_alternative_render(self):
        CS.note_action_issued(11, "turn", "left 62°", requested_deg=90.0, attempted_deg=62.0)
        self.assertIn("asked 90°, only 62°", CS.render_lines(None)[0])
        CS.clear()
        CS.note_action_issued(12, "turn", "right 180°", requested_deg=180.0, attempted_deg=-180.0,
                              alternative="asked left 180°, went right 180° because the swing that way was blocked")
        self.assertIn("went right 180°", CS.render_lines(None)[0])


def _turn_mocks(es, *, check_turn, send_seq=77, tof=None):
    """Isolate motion_controller.turn from hardware."""
    p = es.enter_context
    p(mock.patch.object(mc, "_autonomous_allowed", return_value=None))
    p(mock.patch.object(mc.motion, "telemetry", return_value={"tof_mm": tof or {"f": 2000}}))
    p(mock.patch("intelligence.motion_swing.check_turn", side_effect=check_turn))
    p(mock.patch.object(mc.motion, "send", return_value=send_seq))
    p(mock.patch.object(mc, "_calibrated_compass_yaw", return_value=None))
    p(mock.patch.object(mc, "_invalidate_turn_verification", return_value=1))
    p(mock.patch.object(mc, "_cancel_arc"))
    p(mock.patch.object(mc, "_fx_drive_loop_start"))
    p(mock.patch.object(mc, "_try_swing_escape", return_value=None))
    p(mock.patch.object(mc, "_user_commanded_fx", return_value=False))


class HeadingAlternativeTest(unittest.TestCase):
    def setUp(self):
        CS.clear()
        self.addCleanup(CS.clear)
        import contextlib
        self.es = contextlib.ExitStack()
        self.addCleanup(self.es.close)

    @staticmethod
    def _blocked_left_only(deg, tof):
        # +180 (left) is blocked; -180 (right) is clear.
        if deg > 0:
            return 0.0, "swing_blocked"
        return deg, None

    def test_flag_off_keeps_refusal(self):
        _turn_mocks(self.es, check_turn=self._blocked_left_only)
        with mock.patch.object(config, "MOTION_HEADING_ALTERNATIVES_ENABLED", False, create=True):
            seq = mc.turn(180.0, allow_reverse=True)
        self.assertIsNone(seq)
        rec = CS.recent_actions()[0]
        self.assertEqual(rec["status"], "refused")
        self.assertEqual(rec["reason"], "swing_blocked")
        self.assertIn("left 180°", rec["detail"])

    def test_flag_on_heading_goal_goes_the_other_way(self):
        _turn_mocks(self.es, check_turn=self._blocked_left_only)
        with mock.patch.object(config, "MOTION_HEADING_ALTERNATIVES_ENABLED", True, create=True):
            seq = mc.turn(180.0, allow_reverse=True)
        self.assertEqual(seq, 77)
        sent = mc.motion.send.call_args.args[0]
        self.assertEqual(sent["cmd"], "turn")
        self.assertAlmostEqual(sent["deg"], -180.0)
        rec = CS.recent_actions()[0]
        self.assertEqual(rec["requested_deg"], 180.0)
        self.assertEqual(rec["attempted_deg"], -180.0)
        self.assertIn("went right 180°", rec["alternative"])

    def test_flag_on_directional_request_never_reverses(self):
        _turn_mocks(self.es, check_turn=self._blocked_left_only)
        with mock.patch.object(config, "MOTION_HEADING_ALTERNATIVES_ENABLED", True, create=True):
            seq = mc.turn(90.0)             # "turn left" — direction is the request
        self.assertIsNone(seq)
        self.assertEqual(CS.recent_actions()[0]["status"], "refused")

    def test_alternative_must_clear_its_whole_sweep(self):
        # Both ways blocked → still refused, nothing sent.
        _turn_mocks(self.es, check_turn=lambda deg, tof: (0.0, "swing_blocked"))
        with mock.patch.object(config, "MOTION_HEADING_ALTERNATIVES_ENABLED", True, create=True):
            seq = mc.turn(180.0, allow_reverse=True)
        self.assertIsNone(seq)
        mc.motion.send.assert_not_called()

    def test_shrunk_turn_records_requested_vs_attempted(self):
        _turn_mocks(self.es, check_turn=lambda deg, tof: (62.0 if deg > 0 else deg, None))
        seq = mc.turn(90.0)
        self.assertEqual(seq, 77)
        rec = CS.recent_actions()[0]
        self.assertEqual(rec["requested_deg"], 90.0)
        self.assertEqual(rec["attempted_deg"], 62.0)
        self.assertTrue(rec["shrunk"])


class CompassVerifyRecordsTest(unittest.TestCase):
    def setUp(self):
        CS.clear()
        self.addCleanup(CS.clear)

    def test_verify_marks_partial(self):
        CS.note_action_issued(21, "turn", "left 90°", requested_deg=90.0, attempted_deg=90.0)
        CS.note_action_result(21, "completed")
        record = {"seq": 21, "desired_deg": 90.0, "rate": 60.0, "start_yaw": 10.0,
                  "epoch": mc._turn_verify_epoch, "attempt": 0}
        with (
            mock.patch.object(mc._stop, "wait", return_value=False),
            mock.patch.object(mc.motion, "connected", return_value=True),
            mock.patch.object(mc.motion, "owner", return_value="host"),
            mock.patch.object(mc, "charging", return_value=False),
            mock.patch.object(mc, "_calibrated_compass_yaw", return_value=70.0),  # only 60° turned
            mock.patch.object(config, "MOTION_COMPASS_TURN_MAX_CORRECTIONS", 0, create=True),
        ):
            mc._verify_completed_turn(record)
        rec = CS.recent_actions()[0]
        self.assertEqual(rec["status"], "partial")
        self.assertAlmostEqual(rec["measured_deg"], 60.0)


class ResolverTest(unittest.TestCase):
    def _ev(self, **kw):
        base = dict(final_person_id=1, final_name="Bret Benziger", raw_best_id=1,
                    raw_best_name="Bret", raw_best_score=0.85, margin=0.2, required_margin=0.07,
                    accept_tier="hard", words=8, voiced_secs=2.5,
                    scoreboard=[(1, "Bret", 0.85, 5), (2, "PJ", 0.65, 4)])
        base.update(kw)
        return AT.UtteranceEvidence(**base)

    def test_text_input(self):
        r = AT.resolve(AT.UtteranceEvidence(text_input=True, final_person_id=3, final_name="JT"))
        self.assertEqual(r.status, "known")
        r = AT.resolve(AT.UtteranceEvidence(text_input=True))
        self.assertEqual(r.status, "unknown")

    def test_unknown_when_nobody(self):
        r = AT.resolve(AT.UtteranceEvidence(final_person_id=None, off_camera_unknown=True))
        self.assertEqual(r.status, "unknown")
        self.assertIn("off camera", r.basis)

    def test_strong_voice_is_known(self):
        r = AT.resolve(self._ev())
        self.assertEqual(r.status, "known")
        self.assertEqual(r.conflicts, [])

    def test_voice_points_elsewhere_is_ambiguous(self):
        r = AT.resolve(self._ev(raw_best_id=2, raw_best_name="PJ", raw_best_score=0.7,
                                accept_tier=None, identity_resolution="visible_face"))
        self.assertEqual(r.status, "ambiguous")
        self.assertTrue(any("PJ" in c for c in r.conflicts))

    def test_visual_latch_disagrees(self):
        r = AT.resolve(self._ev(visual_latch_pid=2))
        self.assertEqual(r.status, "ambiguous")
        self.assertTrue(any("mouth moving" in c for c in r.conflicts))

    def test_bearing_disagrees(self):
        r = AT.resolve(self._ev(bearing_selected_pid=2))
        self.assertEqual(r.status, "ambiguous")

    def test_sticky_short_clip_is_ambiguous_but_long_is_known(self):
        short = self._ev(raw_best_id=None, raw_best_score=0.0, accept_tier="sticky", words=1,
                         voiced_secs=0.4, scoreboard=[])
        self.assertEqual(AT.resolve(short).status, "ambiguous")
        long_ = self._ev(raw_best_id=None, raw_best_score=0.0, accept_tier="sticky", words=9,
                         voiced_secs=3.0, scoreboard=[])
        self.assertEqual(AT.resolve(long_).status, "known")

    def test_thin_margin_is_ambiguous(self):
        r = AT.resolve(self._ev(raw_best_score=0.5, margin=0.02, required_margin=0.07,
                                accept_tier="roster"))
        self.assertEqual(r.status, "ambiguous")
        self.assertTrue(any("runner-up" in c for c in r.conflicts))


class SpeakerLinesTest(unittest.TestCase):
    def setUp(self):
        CS.clear()
        self.addCleanup(CS.clear)

    def test_ambiguous_and_unknown_lines(self):
        CS.note_speaker_resolution({"status": "ambiguous", "person_id": 1, "name": "Bret Benziger",
                                    "conflicts": ["the voice scored PJ at 0.70, not Bret"]})
        lines = CS.render_lines(1)
        self.assertTrue(any("SPEAKER UNCERTAIN" in l and "best guess Bret" in l for l in lines))
        CS.note_speaker_resolution({"status": "unknown", "person_id": None, "name": None, "conflicts": []})
        self.assertTrue(any("NOT someone you recognize" in l for l in CS.render_lines(None)))
        CS.note_speaker_resolution({"status": "known", "person_id": 1, "name": "Bret", "conflicts": []})
        self.assertEqual(CS.speaker_lines(), [])


class TranscriptAndLeanUncertaintyTest(unittest.TestCase):
    def test_transcript_entry_flag(self):
        from memory import conversations as conv
        conv.clear_transcript()
        self.addCleanup(conv.clear_transcript)
        conv.add_to_transcript("Bret", "hi", uncertain=True)
        conv.add_to_transcript("Rex", "yo")
        rows = conv.get_session_transcript()
        self.assertTrue(rows[0]["uncertain"])
        self.assertFalse(rows[1]["uncertain"])

    def test_lean_labels_uncertain_speakers(self):
        from intelligence import lean_brain as LB
        transcript = [
            {"speaker": "Bret Benziger", "text": "the wheels are done", "uncertain": True},
            {"speaker": "Rex", "text": "Finally."},
            {"speaker": "JT", "text": "can it climb stairs?"},
        ]
        with (
            mock.patch.object(config, "LEAN_MULTI_PARTY_ENABLED", True, create=True),
            mock.patch.object(LB, "_persona", return_value="PERSONA"),
            mock.patch.object(LB, "_person_lines", return_value=[]),
            mock.patch.object(LB, "_scene_lines", return_value=[]),
            mock.patch.object(LB, "_context_lines", return_value=[]),
            mock.patch.object(LB, "_current_speaker_display", return_value="Bret"),
            mock.patch.object(LB, "_other_participant_lines", return_value=[]),
        ):
            msgs = LB._messages("what now?", 1, transcript, None, speaker_uncertain=True)
        self.assertEqual(msgs[1]["content"], "Bret?: the wheels are done")
        self.assertEqual(msgs[3]["content"], "JT: can it climb stairs?")
        self.assertEqual(msgs[-1]["content"], "Bret?: what now?")
        self.assertIn("'?' after a speaker's name", msgs[0]["content"])


class LearningGateTest(unittest.TestCase):
    def test_passive_enroll_stands_down_when_ambiguous(self):
        import numpy as np
        from intelligence import interaction as I
        saved = I._current_turn_speaker_evidence
        self.addCleanup(setattr, I, "_current_turn_speaker_evidence", saved)
        I._current_turn_speaker_evidence = {"resolution": {"status": "ambiguous"}}
        with (
            mock.patch.object(I.config, "PASSIVE_VOICE_ENROLL_ENABLED", True),
            mock.patch.object(I, "_pending_voice_sample_capture", None),
            mock.patch.object(I, "_pending_impersonation_capture", None),
            mock.patch.object(I, "_single_visible_person_identity") as solo,
        ):
            I._maybe_passive_voice_enroll("a long enough sentence for the gate", np.zeros(16000, dtype=np.float32), 1, 1, 0.9)
        solo.assert_not_called()
        self.assertTrue(I._turn_speaker_uncertain())
        I._current_turn_speaker_evidence = {"resolution": {"status": "known"}}
        self.assertFalse(I._turn_speaker_uncertain())


if __name__ == "__main__":
    unittest.main()
