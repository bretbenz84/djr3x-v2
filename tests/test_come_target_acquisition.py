"""01:30 field regression: an absent enrollee must not make Rex search past Bret.

Exercises production attribution -> request -> scan -> acquisition -> approach,
plus losses and competing reflexes after acquisition. All hardware is mocked.
"""
import time
import unittest
from unittest import mock

import config
from intelligence import attribution, interaction as IX, motion_agency as MA
from intelligence import motion_controller as MC
from tests.test_flex_doa import _ComeFixture
from tests.test_motion_agency import _snapshot, _profile


def field_evidence(**changes):
    data = dict(words=2, voiced_secs=.51, raw_best_id=4, raw_best_name="Jeremy Thomas",
                raw_best_score=.608, margin=.16, required_margin=.07,
                hard_threshold=.50, soft_threshold=.60, accept_tier="hard",
                engaged_pid=1, previous_speaker_pid=1, previous_speaker_age_secs=24.7,
                scoreboard=[(4, "Jeremy Thomas", .608, 1), (1, "Bret", .449, 8),
                            (3, "T'Joy Jackson", .403, 2)])
    data.update(changes)
    return data


class ShortSpeakerSwitchTest(unittest.TestCase):
    def _resolve(self, **changes):
        data = field_evidence(**changes)
        data.pop("previous_speaker_age_secs")
        return attribution.resolve_authoritative(attribution.UtteranceEvidence(**data))

    def test_logged_half_second_cannot_certify_absent_jeremy(self):
        result = self._resolve()
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.person_id)
        self.assertIsNone(result.name)

    def test_confirmed_new_speaker_is_still_supported(self):
        for change in ({"raw_best_score": .82}, {"voiced_secs": 2.1, "words": 5},
                       {"visual_observations": [{"person_db_id": 4}]}):
            with self.subTest(change=change):
                self.assertEqual(self._resolve(**change).person_id, 4)


class ComeTargetAcquisitionTest(_ComeFixture):
    def setUp(self):
        super().setUp()
        self.scene = {"people": []}
        for patch in (
            mock.patch("world_state.world_state.snapshot", side_effect=lambda: self.scene),
            mock.patch.object(MA.motion_controller, "stop", return_value=9),
            mock.patch.object(MA, "no_drive_room", return_value=None),
            mock.patch.object(MA, "_start_come_drive_gaze"),
            mock.patch.object(MA, "_come_gaze_busy", return_value=False),
            mock.patch("intelligence.consciousness.suspend_face_tracking"),
            mock.patch("intelligence.consciousness.resume_face_tracking"),
            mock.patch("intelligence.consciousness.note_speaker_gaze_intent"),
            mock.patch("audio.speech_queue.enqueue"),
            mock.patch.object(IX, "_current_turn_speaker_evidence", {}),
            mock.patch.object(IX, "_cancel_motion_sequence"),
            mock.patch.object(IX, "_clear_motion_continuation"),
            mock.patch.object(IX, "_no_drive_room_decline_line", return_value=None),
            mock.patch.object(config, "MOTION_COME_NECK_SWEEP_ENABLED", False),
            mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _tick(self):
        MA.step(self.scene, _profile())

    def test_field_attribution_search_recognition_approach_and_dropout(self):
        evidence = field_evidence()
        with mock.patch.object(IX, "_last_scan_ranked", evidence["scoreboard"]), \
             mock.patch.object(IX, "_last_scan_secs", {"voiced": .51}), \
             mock.patch.object(IX, "_last_scan_windows", []), \
             mock.patch.object(IX, "_utterance_observations", {}):
            resolution = IX._resolve_turn_attribution(
                turn_id=3, text="Come here.", text_input=False, raw_best_id=4,
                raw_best_name="Jeremy Thomas", speaker_score=.608, speaker_margin=.16,
                required_margin=.07, accept_tier="hard", identity_resolution="voice_match",
                person_id=4, person_name="Jeremy Thomas", off_camera_unknown=False,
                visible_known_ids=[], bearing_match=None, engaged={"person_id": 1},
                previous_speaker={"person_id": 1, "at": time.monotonic()-24.7})
        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.person_id)
        with mock.patch.object(IX, "_recent_voice_bearing", return_value=None):
            self.assertEqual(IX._handle_router_motion_action(
                IX.action_router.classify_explicit_motion("Come here."),
                requester_person_id=resolution.person_id), "On my way.")
        self.ring.bodies = [(18.3, 3.3, 1.)]
        self._tick()
        self.assertEqual(self.turn.call_count, 1)
        # First clear camera recognition in the field at 01:31:41.
        self.scene = _snapshot(db_id=1)
        self.ring.bodies = [(155., 1.5, 1.)]
        self._tick()
        self.assertTrue(MA._requested_come["acquired"])
        self.assertEqual(MA._requested_come["requester_id"], 1)
        MA.motion_controller.stop.assert_called_once()
        self.come.assert_called_once_with(0., stop_at=config.MOTION_COME_REQUEST_STOP_AT_M)
        # Recognition flickers and the radar still offers a 155-degree leg.
        self.scene = {"people": []}
        with mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")):
            self._tick()
        self.assertFalse(MA.requested_come_active())
        self.assertEqual(self.turn.call_count, 1, "no turns after the caller was acquired")

    def test_legacy_short_voice_label_can_be_reconciled_when_bret_is_found(self):
        MA.request_come_here(person_id=4, speaker_evidence=field_evidence())
        self.scene = _snapshot(db_id=1)
        self._tick()
        self.assertEqual(MA._requested_come["requester_id"], 1)
        self.come.assert_called_once()
        self.turn.assert_not_called()

    def test_strong_different_caller_is_not_replaced_by_a_bystander(self):
        MA.request_come_here(person_id=4, speaker_evidence=field_evidence(raw_best_score=.82))
        self.scene = _snapshot(db_id=1)
        self._tick()
        self.assertFalse(MA._requested_come["acquired"])
        self.assertEqual(MA._requested_come["requester_id"], 4)
        self.come.assert_not_called()

    def test_visible_bret_with_weak_wrong_label_is_acquired_without_opening_turn(self):
        self.scene = _snapshot(db_id=1)
        self.assertTrue(MA.request_come_here(person_id=4, voice_bearing_deg=170.,
                                            speaker_evidence=field_evidence()))
        self._tick()
        self.turn.assert_not_called()
        self.come.assert_called_once()

    def test_mid_scan_acquisition_stops_motion_before_original_turn_finishes(self):
        MA.request_come_here(person_id=1, voice_bearing_deg=155.)
        self.scene = _snapshot(db_id=1)
        with mock.patch.object(MA.motion, "state", return_value="moving"), \
             mock.patch.object(MA.motion, "done_result", return_value=None):
            self._tick()
        MA.motion_controller.stop.assert_called_once()
        self.assertTrue(MA._requested_come["acquired"])
        self.assertIsNone(MA._requested_come["pending_turn_seq"])
        self.assertEqual(self.turn.call_count, 1)

    def test_post_acquisition_loss_cannot_restart_search_or_switch_people(self):
        self.scene = _snapshot(db_id=1)
        MA.request_come_here(person_id=1)
        self.scene = _snapshot(db_id=4)
        self.ring.bodies = [(155., 1.5, 1.)]
        now = time.monotonic()
        with mock.patch.object(MA, "_radar_bodies") as radar:
            MA._step_requested_come(self.scene, now)
            MA._step_requested_come(self.scene, now+4.)
            self.assertEqual(MA._requested_come["requester_id"], 1)
            MA._step_requested_come(self.scene, now+9.)
        self.assertFalse(MA.requested_come_active())
        radar.assert_not_called()
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_queued_search_and_over_here_cannot_override_acquired_target(self):
        self.scene = _snapshot(db_id=1)
        MA.request_come_here(person_id=1)
        self.assertIsNone(MA._issue_come_turn(180., time.monotonic()))
        self.assertEqual(MA.orient_to_voice(155., share=.9, reason="over_here"), "come_active")
        self.turn.assert_not_called()

    def test_recentered_neck_uses_bounded_observed_bearing_then_approaches(self):
        self._neck = 7594             # target about 27 degrees to the body's right
        self.scene = _snapshot(db_id=1)
        MA.request_come_here(person_id=1)
        self._tick()                  # park the head, retaining the observed bearing
        self._neck = 5472
        self.scene = {"people": []}    # temporarily lost while the head recenters
        self._tick()
        self.assertEqual(self.turn.call_count, 1)
        self.assertAlmostEqual(self.turn.call_args.args[0], -27.4, delta=.5)
        self.assertLessEqual(abs(self.turn.call_args.args[0]), 30.)
        self.scene = _snapshot(db_id=1)
        self._tick()
        self.come.assert_called_once()
        self.assertEqual(self.turn.call_count, 1)
        self.assertFalse(MA._requested_come["recenter_reacquire"])

    def test_acquired_alignment_cannot_turn_ninety_or_loop_indefinitely(self):
        self.scene = _snapshot(db_id=1, face_box=(1920, 400, 200, 200))
        MA.request_come_here(person_id=1)
        for _ in range(7):
            self._tick()
        self.assertEqual(self.turn.call_count, config.MOTION_COME_ALIGN_MAX_TRIES)
        self.assertTrue(all(abs(call.args[0]) <= 30 for call in self.turn.call_args_list))
        self.come.assert_not_called()
        self.assertTrue(MA._requested_come["acquired"])

    def test_short_voice_reconciliation_requires_unconflicted_recent_partner(self):
        for change in ({"previous_speaker_age_secs": 100.}, {"mixed_speakers": True},
                       {"bearing_selected_pid": 4}, {"visual_latch_pid": 4},
                       {"scoreboard": [(4, "Jeremy Thomas", .608, 1), (1, "Bret", .20, 8)]}):
            with self.subTest(change=change):
                self.scene = {"people": []}
                MA.request_come_here(person_id=4, speaker_evidence=field_evidence(**change))
                self.scene = _snapshot(db_id=1)
                self._tick()
                self.assertFalse(MA._requested_come["acquired"])
                self.come.assert_not_called()
                MA.cancel_requested_come("next case")


class CameraTurnControllerTest(unittest.TestCase):
    def test_camera_turn_has_no_delayed_compass_correction(self):
        with mock.patch.object(MC, "_autonomous_allowed", return_value=None), \
             mock.patch.object(MC.motion, "telemetry", return_value={}), \
             mock.patch("intelligence.motion_swing.check_turn", return_value=(18., None)), \
             mock.patch.object(MC, "_calibrated_compass_yaw", return_value=0.), \
             mock.patch.object(MC, "_invalidate_turn_verification", return_value=1) as invalidate, \
             mock.patch.object(MC, "_cancel_arc"), \
             mock.patch.object(MC.motion, "send", return_value=7), \
             mock.patch.object(MC, "_note_issued"), \
             mock.patch.object(MC, "_fx_drive_loop_start"), \
             mock.patch.object(MC, "_remember_turn_verification") as remember:
            self.assertEqual(MC.turn(18., verify=False, allow_escape=False), 7)
        invalidate.assert_called_once()
        remember.assert_not_called()

    def test_refused_camera_turn_cannot_queue_a_swing_escape(self):
        with mock.patch.object(MC, "_autonomous_allowed", return_value=None), \
             mock.patch.object(MC.motion, "telemetry", return_value={}), \
             mock.patch("intelligence.motion_swing.check_turn", return_value=(0., "swing_blocked")), \
             mock.patch.object(MC, "_suppressed"), \
             mock.patch.object(MC, "_try_swing_escape") as escape:
            self.assertIsNone(MC.turn(18., verify=False, allow_escape=False))
        escape.assert_not_called()
