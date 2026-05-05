import unittest
from unittest import mock

import numpy as np

from state import State


class FaceTrackingTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.consciousness = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_self_state = consciousness.world_state.get("self_state")
        self.old_last_seen = consciousness._last_face_seen_at
        self.old_lock = dict(consciousness._face_tracking_lock)
        self.old_suspend_until = consciousness._face_tracking_suspended_until
        self.old_tracking_log_at = consciousness._last_face_tracking_log_at
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        c = self.consciousness
        c.world_state.update("people", self.old_people)
        c.world_state.update("self_state", self.old_self_state)
        c._last_face_seen_at = self.old_last_seen
        c._face_tracking_lock = self.old_lock
        c._face_tracking_suspended_until = self.old_suspend_until
        c._last_face_tracking_log_at = self.old_tracking_log_at

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
