"""2026-09-06 field runs: visible caller acquisition and holding at arrival.

Run with tools/run_lean_checks.py: all movement, speech and camera I/O is fake.
"""
import time
import unittest
from unittest import mock

import config
from intelligence import interaction as IX, motion_agency as MA
from tests.test_flex_doa import _ComeFixture
from tests.test_motion_agency import _snapshot, _profile, _CENTERED_FACE


def evidence(**overrides):
    row = dict(words=2, voiced_secs=.81, raw_best_id=3, raw_best_score=.475,
               soft_threshold=.60, scoreboard=[(3, "T'Joy", .475, 2), (1, "Bret", .426, 8)],
               engaged_pid=1, previous_speaker_pid=1, previous_speaker_age_secs=18.3)
    row.update(overrides)
    return row


class CaptureBearingRegressionTest(unittest.TestCase):
    def test_decode_wait_cannot_extend_direction_window_or_drop_audio_preroll(self):
        import numpy as np
        from hardware import flex_doa as FD
        clock = [100.0]
        segment = np.zeros(int(2.78 * config.AUDIO_SAMPLE_RATE), dtype=np.float32)
        chunk = segment[:100]
        FD._reset_for_tests()
        self.addCleanup(FD._reset_for_tests)
        # The caller spoke ahead during pre-roll. A different sound at +106
        # happens during the subsequent three-second transcription wait.
        FD._inject_for_tests([(t, 0., 0., True, 1e6, False) for t in (98.5, 98.6, 98.7)]
                             + [(t, 106., 106., True, 1e6, False)
                                for t in (100.8, 101., 101.2, 101.4, 101.6)])

        def wait(seconds):
            clock[0] += seconds

        def adopt(_probe):
            clock[0] += 3.0
            return False

        with mock.patch.object(IX.time, "monotonic", side_effect=lambda: clock[0]), \
             mock.patch.object(IX._stop_event, "is_set", return_value=False), \
             mock.patch.object(IX._stop_event, "wait", side_effect=wait), \
             mock.patch.object(IX.stream, "get_audio_chunk",
                               side_effect=lambda secs: segment if secs == 2.78 else chunk), \
             mock.patch.object(IX.vad, "is_speech", return_value=False), \
             mock.patch.object(IX, "_chunk_for_vad", side_effect=lambda c: c), \
             mock.patch.object(IX.state_module, "get_state", return_value=IX.State.ACTIVE), \
             mock.patch.object(IX._situation_assessor, "set_vad_active"), \
             mock.patch.object(IX, "_eager_motion_endpoint_enabled", return_value=True), \
             mock.patch.object(IX, "_start_eager_motion_probe", return_value={"matched": False}), \
             mock.patch.object(IX, "_adopt_probe_transcript", side_effect=adopt), \
             mock.patch.object(IX, "_speech_capture_secs", return_value=2.78), \
             mock.patch.object(config, "SILENCE_TIMEOUT_SECS", .20), \
             mock.patch.object(config, "MIN_SPEECH_DURATION_SECS", 0.), \
             mock.patch.object(config, "MOTION_EAGER_ENDPOINT_SILENCE_SECS", .08), \
             mock.patch.object(FD, "available", return_value=True), \
             mock.patch("vision.active_speaker.evidence_between", return_value=[]), \
             mock.patch.object(MA, "note_voice_bearing"), \
             mock.patch.object(IX, "_last_voice_bearing", None), \
             mock.patch.object(IX, "_utterance_observations", {}):
            result = IX._accumulate_speech(100.)
            self.assertIs(result, segment)
            voice = IX._recent_voice_bearing()
            self.assertIsNotNone(voice)
            self.assertAlmostEqual(voice["bearing_deg"], 0.)
            self.assertAlmostEqual(voice["t1"] - voice["t0"], 2.78)
            self.assertAlmostEqual(clock[0] - voice["t1"], 3.0)
            self.assertLess(voice["t0"], 98.5)


class ComeHereRegressionTest(_ComeFixture):
    def setUp(self):
        super().setUp()
        self.scene = _snapshot(face_box=_CENTERED_FACE)
        for patch in (
            mock.patch("world_state.world_state.snapshot", side_effect=lambda: self.scene),
            mock.patch("intelligence.consciousness.suspend_face_tracking"),
            mock.patch("intelligence.consciousness.note_speaker_gaze_intent"),
            mock.patch("audio.speech_queue.enqueue"),
            mock.patch.object(IX, "_current_turn_speaker_evidence", {}),
            mock.patch.object(IX, "_cancel_motion_sequence"),
            mock.patch.object(IX, "_clear_motion_continuation"),
            mock.patch.object(IX, "_no_drive_room_decline_line", return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _request(self, person_id=None, **kwargs):
        return MA.request_come_here(person_id=person_id, voice_bearing_deg=106.,
                                    voice_share=.5, **kwargs)

    def _tick(self):
        MA.step(self.scene, _profile())

    def test_known_visible_caller_beats_wrong_microphone_and_radar_bearings(self):
        self.ring.bodies = [(-179., 3.8, 1.)]
        self.assertTrue(self._request(person_id=1))
        self.turn.assert_not_called()
        self._tick()
        self.turn.assert_not_called()
        self.come.assert_called_once_with(0., stop_at=config.MOTION_COME_REQUEST_STOP_AT_M)
        self.assertIsNone(MA._requested_come["voice_world"])

    def test_field_evidence_reaches_motion_without_promoting_identity(self):
        ranked = evidence()["scoreboard"]
        with mock.patch.object(IX, "_last_scan_ranked", ranked), \
             mock.patch.object(IX, "_last_scan_secs", {"voiced": .81}), \
             mock.patch.object(IX, "_last_scan_windows", []), \
             mock.patch.object(IX, "_utterance_observations", {}):
            resolution = IX._resolve_turn_attribution(
                turn_id=3, text="Come here.", text_input=False, raw_best_id=3,
                raw_best_name="T'Joy", speaker_score=.475, speaker_margin=.05,
                required_margin=.07, accept_tier=None, identity_resolution=None,
                person_id=None, person_name=None, off_camera_unknown=True,
                visible_known_ids=[1], bearing_match=None, engaged={"person_id": 1},
                previous_speaker={"person_id": 1, "at": time.monotonic()-18.3})
        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.person_id)
        IX._current_turn_speaker_evidence["resolution"] = resolution.as_dict()
        with mock.patch.object(IX, "_recent_voice_bearing",
                               return_value={"bearing_deg": 106., "share": .5}):
            reply = IX._handle_router_motion_action(
                IX.action_router.classify_explicit_motion("Come here."))
        self.assertEqual(reply, "On my way.")
        self.assertEqual(MA._requested_come["requester_id"], 1)
        self._tick()
        self.turn.assert_not_called()
        self.come.assert_called_once()
        self.assertEqual(IX._current_turn_speaker_evidence["resolution"]["status"], "ambiguous")

    def test_weak_fallback_abstains_on_conflicting_or_stale_evidence(self):
        for changes in (
            {"mixed_speakers": True}, {"bearing_contradiction": True},
            {"bearing_selected_pid": 2}, {"visual_latch_pid": 2},
            {"raw_best_score": .8}, {"previous_speaker_pid": 2},
            {"engaged_pid": 2}, {"previous_speaker_age_secs": 1000},
            {"previous_speaker_age_secs": None}, {"scoreboard": [(1, "Bret", .2, 8)]},
            {"words": 8},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(self._request(speaker_evidence=evidence(**changes)))
                self.assertFalse(MA.requested_come_active())
                self.turn.assert_not_called()
                self.come.assert_not_called()

    def test_second_visible_person_prevents_anonymous_guess(self):
        self.scene["people"] += _snapshot(db_id=2, slot="person_2", face_box=_CENTERED_FACE)["people"]
        self.assertFalse(self._request(speaker_evidence=evidence()))
        self.turn.assert_not_called()

    def test_known_off_camera_caller_is_not_replaced_by_visible_partner(self):
        self.assertTrue(self._request(person_id=2))
        self.assertEqual(self.turn.call_args.args[0], 106.)
        self.assertEqual(MA._requested_come["requester_id"], 2)
        self.come.assert_not_called()

    def test_acquired_target_does_not_switch_to_another_person(self):
        self.assertTrue(self._request(speaker_evidence=evidence()))
        self.scene = _snapshot(db_id=2, face_box=_CENTERED_FACE)
        self._tick()
        self.come.assert_not_called()
        self.assertEqual(MA._requested_come["requester_id"], 1)

    def test_stale_head_lock_does_not_count_as_visible(self):
        self._tracking = {"locked": True, "visible": True, "lock_key": "db:1"}
        self.scene["people"][0]["face_visible"] = False
        self.assertIsNone(MA._tracked_person(self.scene, 1))
        self.assertTrue(self._request(person_id=1))
        self.assertEqual(self.turn.call_args.args[0], 106.)

    def test_front_block_holds_target_without_search_and_explains_timeout(self):
        self.assertTrue(self._request(person_id=1))
        start = time.monotonic()
        with mock.patch.object(MA.motion, "state", return_value="blocked"), \
             mock.patch("audio.speech_queue.enqueue") as say:
            MA._step_requested_come(self.scene, start, base_idle=False)
            MA._step_requested_come(self.scene, start+9, base_idle=False)
        self.assertFalse(MA.requested_come_active())
        self.assertIn("front sensors", say.call_args.args[0])
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_missing_face_geometry_never_counts_as_centered(self):
        self.scene["people"][0].pop("face_box")
        self.assertTrue(self._request(person_id=1))
        self._tick()
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_refused_radar_leg_ends_request_without_sweep_or_later_retry(self):
        self.scene = {"people": []}
        self.ring.bodies = [(-130., 4.7, 1.)]
        self.turn.return_value = None
        with mock.patch.object(MA.motion_controller, "last_refusal",
                               return_value={"line": config.MOTION_SWING_BLOCKED_LINE}):
            self.assertTrue(MA.request_come_here(person_id=1))
            self._tick()
            self.assertFalse(MA.requested_come_active())
            MA._step_requested_come(self.scene, time.monotonic()+2)
        self.turn.assert_called_once()

    def test_initial_refusal_is_returned_instead_of_on_my_way(self):
        self.scene = {"people": []}
        self.turn.return_value = None
        with mock.patch.object(IX, "_recent_voice_bearing", return_value={"bearing_deg": 106., "share": .5}), \
             mock.patch.object(MA.motion_controller, "last_refusal",
                               return_value={"line": config.MOTION_SWING_BLOCKED_LINE}):
            reply = IX._handle_router_motion_action(
                IX.action_router.classify_explicit_motion("Come here."), requester_person_id=1)
        self.assertEqual(reply, config.MOTION_SWING_BLOCKED_LINE)
        self.assertFalse(MA.requested_come_active())
        self.turn.assert_called_once()

    def test_arrival_finishes_facing_caller_without_another_advance(self):
        self.assertTrue(self._request(person_id=1))
        self._tick()
        self.scene["people"][0]["face_box"] = (1359, 400, 200, 200)
        with mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")), \
             mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.):
            self._tick()
            self.assertTrue(MA._requested_come['arrival_only'])
            self.turn.assert_called_once()
            self.assertLess(self.turn.call_args.args[0], 0.)
            self.scene["people"][0]["face_box"] = _CENTERED_FACE
            self._tick()
        self.assertFalse(MA.requested_come_active())
        self.come.assert_called_once()

    def test_obstacle_curve_restores_camera_heading_without_new_advance(self):
        with mock.patch.object(MA, "_base_yaw_deg", return_value=0.):
            self.assertTrue(self._request(person_id=1))
            self._tick()
        self.scene = {"people": []}
        self.ring.bodies = [(150., 1.5, 1.)]
        with mock.patch.object(MA, "_base_yaw_deg", return_value=25.), \
             mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")), \
             mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.):
            self._tick()
        self.assertTrue(MA._requested_come["arrival_only"])
        self.turn.assert_called_once()
        self.assertAlmostEqual(self.turn.call_args.args[0], -25.)
        self.come.assert_called_once()
        self.scene = _snapshot(db_id=1)
        with mock.patch.object(MA, "_base_yaw_deg", return_value=0.), \
             mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")), \
             mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.):
            self._tick()
        self.assertFalse(MA.requested_come_active())
        self.come.assert_called_once()

    def test_arrival_with_camera_dipped_out_of_view_does_not_start_a_search(self):
        self.assertTrue(self._request(person_id=1))
        self._tick()
        self.scene = {"people": []}
        self.ring.bodies = [(90., 1.5, 1.)]
        with mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")), \
             mock.patch("sequences.animations.travel_glance_pose") as head:
            self._tick()
        self.assertFalse(MA.requested_come_active())
        self.turn.assert_not_called()
        head.assert_not_called()
        self.come.assert_called_once()

    def test_completed_drive_is_respected_even_when_front_state_is_blocked(self):
        self.assertTrue(self._request(person_id=1))
        self._tick()
        self.scene["people"][0]["distance_zone"] = "public"
        self.scene["people"][0]["face_box"] = (1359, 400, 200, 200)
        with mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, "completed")), \
             mock.patch.object(MA.motion, "state", return_value="blocked"), \
             mock.patch.object(MA.motion, "telemetry", return_value={"tof_mm": {"fl": 610, "fr": 72}}):
            self._tick()
        self.assertFalse(MA.requested_come_active())
        self.turn.assert_not_called()
        self.come.assert_called_once()

    def test_unfinished_approach_cannot_be_interrupted_by_an_alignment_tick(self):
        self.assertTrue(self._request(person_id=1))
        self._tick()
        self.scene["people"][0]["face_box"] = (1359, 400, 200, 200)
        with mock.patch.object(MA.motion_controller, "last_come_result", return_value=(8, None)):
            self._tick()
        self.assertTrue(MA.requested_come_active())
        self.turn.assert_not_called()
        self.come.assert_called_once()
