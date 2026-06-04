import unittest
from unittest import mock

import numpy as np

from state import State

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None


class FaceTrackingTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.consciousness = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_self_state = consciousness.world_state.get("self_state")
        self.old_last_seen = consciousness._last_face_seen_at
        self.old_lock = dict(consciousness._face_tracking_lock)
        self.old_live_tracker = consciousness._face_tracking_tracker
        self.old_last_solo_identity = consciousness._last_solo_identity
        self.old_suspend_until = consciousness._face_tracking_suspended_until
        self.old_tracking_log_at = consciousness._last_face_tracking_log_at
        self.old_tracking_error_key = consciousness._face_tracking_last_error_key
        self.old_tracking_error_x = consciousness._face_tracking_last_error_x
        self.old_tracking_error_y = consciousness._face_tracking_last_error_y
        self.old_tracking_error_at = consciousness._face_tracking_last_error_at
        self.old_adaptive_head_rest = dict(consciousness._adaptive_head_rest)
        with consciousness._speaker_gaze_lock:
            self.old_speaker_gaze_intent = dict(consciousness._speaker_gaze_intent)
            consciousness._speaker_gaze_intent.clear()
        with consciousness._directed_gaze_hold_lock:
            self.old_directed_gaze_hold = dict(consciousness._directed_gaze_hold)
        consciousness.clear_directed_gaze_hold()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        c = self.consciousness
        c.world_state.update("people", self.old_people)
        c.world_state.update("self_state", self.old_self_state)
        c._last_face_seen_at = self.old_last_seen
        c._face_tracking_lock = self.old_lock
        c._face_tracking_tracker = self.old_live_tracker
        c._last_solo_identity = self.old_last_solo_identity
        c._face_tracking_suspended_until = self.old_suspend_until
        c._last_face_tracking_log_at = self.old_tracking_log_at
        c._face_tracking_last_error_key = self.old_tracking_error_key
        c._face_tracking_last_error_x = self.old_tracking_error_x
        c._face_tracking_last_error_y = self.old_tracking_error_y
        c._face_tracking_last_error_at = self.old_tracking_error_at
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update(self.old_adaptive_head_rest)
        with c._speaker_gaze_lock:
            c._speaker_gaze_intent.clear()
            c._speaker_gaze_intent.update(self.old_speaker_gaze_intent)
        with c._directed_gaze_hold_lock:
            c._directed_gaze_hold.clear()
            c._directed_gaze_hold.update(self.old_directed_gaze_hold)

    def _frame_with_patch(self, x: int, y: int) -> np.ndarray:
        rng = np.random.default_rng(1234)
        patch = rng.integers(0, 255, size=(36, 36), dtype=np.uint8)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[y:y + 36, x:x + 36] = np.repeat(patch[:, :, None], 3, axis=2)
        return frame

    def _set_servo_positions(self):
        c = self.consciousness
        self_state = c.world_state.get("self_state")
        self_state["servo_positions"] = {
            "neck": 6000,
            "headlift": 6000,
            "headtilt": 4320,
            "visor": 6000,
            "elbow": 6720,
            "hand": 6000,
            "pokerarm": 6000,
            "heroarm": 6000,
        }
        c.world_state.update("self_state", self_state)

    def test_detection_hold_clears_stale_face_geometry(self):
        c = self.consciousness
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 120, 80, 90),
            "position": (140, 165),
            "face_box_fraction": 0.1,
        }])
        c._last_face_seen_at = 100.0

        with (
            mock.patch("vision.face.detect_faces", return_value=[]),
            mock.patch.object(c.time, "monotonic", return_value=102.0),
            mock.patch.object(c.config, "FACE_DETECTION_HOLD_SECS", 6.0),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_db_id"], 1)
        self.assertFalse(people[0]["face_visible"])
        self.assertTrue(people[0]["face_missing"])
        self.assertIsNone(people[0]["face_box"])
        self.assertIsNone(people[0]["position"])

    def test_face_mood_write_replaces_stale_neutral_expression(self):
        c = self.consciousness
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "expression": "neutral",
        }])

        c._write_face_mood_to_world_state(
            1,
            {"mood": "happy", "confidence": 0.91, "notes": "smiling"},
        )

        person = c.world_state.get("people")[0]
        self.assertEqual(person["expression"], "happy")
        self.assertEqual(person["face_mood"]["mood"], "happy")

    def test_solo_identity_sticky_rejects_far_false_positive_box(self):
        c = self.consciousness
        c.world_state.update("people", [{"id": "person_1"}])
        c._last_solo_identity = (1, "Bret", 100.0, (100, 300, 80, 80))
        fake_detection = {
            "bounding_box": (900, 120, 90, 90),
            "encoding": np.zeros(128, dtype=np.float32),
            "landmarks": np.zeros((68, 2), dtype=np.int32),
        }

        with (
            mock.patch("vision.face.detect_faces", return_value=[fake_detection]),
            mock.patch("vision.face.identify_face", return_value=None),
            mock.patch.object(c.time, "monotonic", return_value=101.0),
            mock.patch.object(c, "_maybe_prompt_unknown_identity"),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertIsNone(people[0].get("person_db_id"))
        self.assertIsNone(people[0].get("face_id"))

    def test_solo_identity_sticky_accepts_nearby_unmatched_box(self):
        c = self.consciousness
        c.world_state.update("people", [{"id": "person_1"}])
        c._last_solo_identity = (1, "Bret", 100.0, (100, 300, 80, 80))
        fake_detection = {
            "bounding_box": (118, 292, 82, 82),
            "encoding": np.zeros(128, dtype=np.float32),
            "landmarks": np.zeros((68, 2), dtype=np.int32),
        }

        with (
            mock.patch("vision.face.detect_faces", return_value=[fake_detection]),
            mock.patch("vision.face.identify_face", return_value=None),
            mock.patch.object(c.time, "monotonic", return_value=101.0),
            mock.patch.object(c, "_maybe_prompt_unknown_identity"),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertEqual(people[0].get("person_db_id"), 1)
        self.assertEqual(people[0].get("face_id"), "Bret")

    def test_single_visible_face_recenters_neck_lift_and_tilt(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 160, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=200.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile") as set_profile,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        tilt_ch = c.config.SERVO_CHANNELS["headtilt"]["ch"]
        self.assertLess(updates[neck_ch], c.config.SERVO_CHANNELS["neck"]["neutral"])
        self.assertGreater(updates[lift_ch], c.config.SERVO_CHANNELS["headlift"]["neutral"])
        self.assertLess(updates[tilt_ch], c.config.SERVO_CHANNELS["headtilt"]["neutral"])
        self.assertLessEqual(
            abs(updates[neck_ch] - c.config.SERVO_CHANNELS["neck"]["neutral"]),
            c.config.FACE_TRACKING_NECK_MAX_STEP_QUS,
        )
        set_profile.assert_called_once()
        self.assertIn(neck_ch, set_profile.call_args.args[0])
        set_baseline.assert_called_once_with(
            neck=updates[neck_ch],
            lift=updates[lift_ch],
            tilt=updates[tilt_ch],
        )

    def test_adaptive_rest_learns_downward_pose_from_low_face(self):
        c = self.consciousness
        self._set_servo_positions()
        lift_neutral = c.config.SERVO_CHANNELS["headlift"]["neutral"]
        tilt_neutral = c.config.SERVO_CHANNELS["headtilt"]["neutral"]
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": lift_neutral,
            "tilt": tilt_neutral,
            "samples": 0,
            "updated_at": 0.0,
        })
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (560, 560, 160, 140),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=225.0),
            mock.patch("hardware.servos.set_servos"),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_ADAPT_ALPHA", 1.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION", 0.0),
        ):
            c._step_face_tracking(self.frame)

        self.assertLess(c._adaptive_head_rest["lift"], lift_neutral)
        self.assertGreater(c._adaptive_head_rest["tilt"], tilt_neutral)

    def test_adaptive_rest_learns_upward_pose_from_high_face(self):
        c = self.consciousness
        self._set_servo_positions()
        lift_neutral = c.config.SERVO_CHANNELS["headlift"]["neutral"]
        tilt_neutral = c.config.SERVO_CHANNELS["headtilt"]["neutral"]
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": lift_neutral,
            "tilt": tilt_neutral,
            "samples": 0,
            "updated_at": 0.0,
        })
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (560, 80, 160, 140),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=226.0),
            mock.patch("hardware.servos.set_servos"),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_ADAPT_ALPHA", 1.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION", 0.0),
        ):
            c._step_face_tracking(self.frame)

        self.assertGreater(c._adaptive_head_rest["lift"], lift_neutral)
        self.assertLess(c._adaptive_head_rest["tilt"], tilt_neutral)

    def test_speaker_gaze_center_search_uses_adaptive_vertical_rest(self):
        c = self.consciousness
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        tilt_ch = c.config.SERVO_CHANNELS["headtilt"]["ch"]
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": 5480,
            "tilt": 5010,
            "samples": 4,
            "updated_at": 100.0,
        })

        with (
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            targets = c._speaker_gaze_search_targets(0.0, 0.0)

        self.assertEqual(targets[lift_ch], 5480)
        self.assertEqual(targets[tilt_ch], 5010)

    def test_face_loss_eases_vertical_pose_toward_adaptive_rest(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": 5500,
            "tilt": 4800,
            "samples": 3,
            "updated_at": 100.0,
        })

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=260.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile") as set_profile,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS", 0.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_RETURN_MAX_STEP_QUS", 55),
        ):
            c._step_face_tracking(self.frame, [])

        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        tilt_ch = c.config.SERVO_CHANNELS["headtilt"]["ch"]
        updates = set_servos.call_args.args[0]
        self.assertEqual(updates[lift_ch], 5945)
        self.assertEqual(updates[tilt_ch], 4375)
        set_profile.assert_called_once()
        set_baseline.assert_called_once_with(neck=6000, lift=5945, tilt=4375)

    def test_recent_speaker_face_beats_larger_bystander(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [
            {
                "id": "person_1",
                "person_db_id": 1,
                "face_id": "Bret",
                "face_visible": True,
                "face_box": (100, 180, 80, 100),
            },
            {
                "id": "person_2",
                "person_db_id": 2,
                "face_id": "Other",
                "face_visible": True,
                "face_box": (900, 160, 240, 240),
            },
        ])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=200.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            c.note_speaker_gaze_intent(1, reason="speech", force_search=False)
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")
        self.assertLess(updates[neck_ch], c.config.SERVO_CHANNELS["neck"]["neutral"])

    def test_search_holds_each_waypoint_still_before_advancing(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])

        servo_mod = mock.MagicMock()
        servo_mod.manual_override_enabled.return_value = False

        with mock.patch.object(c.time, "monotonic", return_value=1000.0):
            c.note_speaker_gaze_intent(
                None, unknown_voice=True, reason="startup", force_search=True,
            )

        settle = float(c.config.SPEAKER_GAZE_SEARCH_SETTLE_SECS)
        dwell = float(c.config.SPEAKER_GAZE_SEARCH_DWELL_SECS)
        hold = settle + dwell

        # First call commits waypoint #1 — exactly one servo move issued.
        pose1 = c._step_speaker_gaze_search(servo_mod, dict(c._speaker_gaze_intent), 1000.0)
        self.assertIsNotNone(pose1)
        self.assertEqual(servo_mod.set_servos.call_count, 1)

        # Through the entire settle+dwell window the head holds still: no further
        # servo commands, and the held pose label is reported unchanged.
        for dt in (settle * 0.5, settle + dwell * 0.5, hold - 0.01):
            pose = c._step_speaker_gaze_search(
                servo_mod, dict(c._speaker_gaze_intent), 1000.0 + dt,
            )
            self.assertEqual(pose, pose1)
        self.assertEqual(servo_mod.set_servos.call_count, 1)

        # Once the dwell elapses, the next call advances to waypoint #2.
        c._step_speaker_gaze_search(
            servo_mod, dict(c._speaker_gaze_intent), 1000.0 + hold + 0.01,
        )
        self.assertEqual(servo_mod.set_servos.call_count, 2)

    def test_unknown_speech_without_face_searches_down_first(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c.note_speaker_gaze_intent(
                None,
                unknown_voice=True,
                reason="speech",
                force_search=True,
            )
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        tilt_ch = c.config.SERVO_CHANNELS["headtilt"]["ch"]
        visor_ch = c.config.SERVO_CHANNELS["visor"]["ch"]
        self.assertLess(updates[lift_ch], c.config.SERVO_CHANNELS["headlift"]["neutral"])
        self.assertGreater(updates[tilt_ch], c.config.SERVO_CHANNELS["headtilt"]["neutral"])
        self.assertEqual(updates[visor_ch], c.config.SERVO_CHANNELS["visor"]["max"])
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertTrue(tracking.get("searching"))
        # First beat of the randomized scan drops the gaze down (seated-person bias)
        # without turning the neck — label is "{horiz}_{vert}", so check the vertical.
        self.assertTrue(str(tracking.get("search_pose") or "").endswith("down"))

    def test_directed_gaze_hold_suppresses_search_and_holds_pose(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            # User just said "look down": hold the gaze, then a fresh off-camera
            # voice arrives. The hold must win — no room-scan, head stays put.
            c.hold_directed_gaze("down")
            c.note_speaker_gaze_intent(
                None,
                unknown_voice=True,
                reason="speech",
                force_search=True,
            )
            c._step_face_tracking(self.frame)

        set_servos.assert_not_called()
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertTrue(tracking.get("directed_hold"))
        self.assertFalse(tracking.get("searching"))
        # The wander stands down while a directed gaze is held.
        from sequences import animations
        self.assertTrue(animations._face_tracking_holding_gaze())

    def test_visible_face_during_hold_still_tracks(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 180, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            # Holding "look down" must not block face tracking: if Rex spots
            # someone (e.g. low in frame), he locks on and keeps watching them.
            c.hold_directed_gaze("down")
            c._step_face_tracking(self.frame)

        set_servos.assert_called_once()
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")

    def test_startup_scan_accepts_visible_face_instead_of_searching(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 180, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=400.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c.request_face_acquisition_scan(reason="startup")
            c._step_face_tracking(self.frame)

        set_servos.assert_called_once()
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertFalse(tracking.get("searching"))

    def test_face_tracking_damps_direction_reversal(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (1060, 180, 120, 120),
        }])
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 499.9}
        c._face_tracking_suspended_until = 0.0
        c._face_tracking_last_error_key = "db:1"
        c._face_tracking_last_error_x = -500.0
        c._face_tracking_last_error_y = 0.0
        c._face_tracking_last_error_at = 499.9

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=500.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        damped_step = int(c.config.FACE_TRACKING_NECK_MAX_STEP_QUS * c.config.FACE_TRACKING_REVERSAL_DAMPING)
        self.assertLessEqual(
            abs(updates[neck_ch] - c.config.SERVO_CHANNELS["neck"]["neutral"]),
            damped_step,
        )

    def test_live_tracked_edge_face_slews_neck_responsively(self):
        # Guard the responsiveness fix: an optical-flow (live_tracked) box at the
        # frame edge — the common case, since ~11/12 ticks ride optical flow — must
        # still move the neck a substantial amount per tick. Under the old tuning
        # (neck max_step 120 * live damping 0.45 = 54 qus/tick) this crawled and
        # Rex took seconds to face someone who moved. The face center sits at the
        # vertical midline so only the neck moves.
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (0, 300, 120, 120),  # far-left edge, vertically centered
            "live_tracked": True,
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        c._face_tracking_last_error_key = None
        c._face_tracking_last_error_x = None
        c._face_tracking_last_error_y = None

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        neutral = c.config.SERVO_CHANNELS["neck"]["neutral"]
        neck_move = abs(updates[neck_ch] - neutral)
        # Left-edge face → neck turns left (below neutral).
        self.assertLess(updates[neck_ch], neutral)
        # Responsive: comfortably more than the old ~54 qus/tick crawl, and never
        # beyond the per-tick step cap.
        self.assertGreaterEqual(neck_move, 150)
        self.assertLessEqual(neck_move, c.config.FACE_TRACKING_NECK_MAX_STEP_QUS)

    @unittest.skipIf(cv2 is None, "OpenCV unavailable")
    def test_live_tracking_people_advances_box_between_recognition_ticks(self):
        c = self.consciousness
        c._face_tracking_tracker = None
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (20, 30, 36, 36),
        }])

        c._live_face_tracking_people(self._frame_with_patch(20, 30))
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": False,
            "face_missing": True,
            "face_box": None,
        }])
        tracked = c._live_face_tracking_people(self._frame_with_patch(34, 39))

        self.assertTrue(tracked[0]["live_tracked"])
        self.assertTrue(tracked[0]["face_visible"])
        self.assertFalse(tracked[0]["face_missing"])
        self.assertGreater(tracked[0]["face_box"][0], 30.0)
        self.assertGreater(tracked[0]["face_box"][1], 36.0)

    def test_existing_face_lock_does_not_immediately_switch_to_other_face(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_2",
            "person_db_id": 2,
            "face_id": "Other",
            "face_visible": True,
            "face_box": (960, 180, 120, 120),
        }])
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 100.0}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=102.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_LOST_HOLD_SECS", 4.0),
        ):
            c._step_face_tracking(self.frame)

        set_servos.assert_not_called()
        set_baseline.assert_not_called()
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertEqual(tracking.get("lock_key"), "db:1")
        self.assertTrue(tracking.get("holding_lost_lock"))

    def test_recent_face_tracking_lock_suppresses_departure(self):
        c = self.consciousness
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 100.0}

        with (
            mock.patch.object(c.config, "FACE_TRACKING_LOST_HOLD_SECS", 8.0),
            mock.patch.object(c.config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", 12.0),
        ):
            self.assertTrue(c._face_tracking_recently_held_person(1, 111.0))
            self.assertFalse(c._face_tracking_recently_held_person(1, 113.0))

    def test_wander_stands_down_when_visible_face_box_exists(self):
        from sequences import animations

        with mock.patch.object(
            animations.world_state,
            "get",
            return_value=[{
                "person_db_id": 1,
                "face_visible": True,
                "face_missing": False,
                "face_box": (40, 120, 100, 140),
            }],
        ):
            self.assertTrue(animations._face_tracking_holding_gaze())


if __name__ == "__main__":
    unittest.main()
