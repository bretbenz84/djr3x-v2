"""
vision/pose.py — model loading (Tasks API) and gesture classification.

History: mediapipe 0.10.x removed the legacy ``mp.solutions.pose`` solution, so the old
loader always failed and no gesture (incl. "waving") was ever published — wave-back was
dead on-device. pose.py now uses ``mp.tasks.vision.PoseLandmarker``. These tests cover the
new load guards and lock in the wave/raising_hand classification so the shadow bug (a
raised wave swallowed as "raising_hand") can't regress.
"""

import types
import unittest
from pathlib import Path
from unittest import mock

from vision import pose


def _kp(**overrides) -> dict:
    """Build a keypoint dict (name -> (x, y, visibility)). y increases downward.

    Defaults: a person facing forward, arms down at their sides (neutral).
    Override individual landmarks to pose a gesture.
    """
    base = {
        "NOSE":           (0.50, 0.25, 1.0),
        "LEFT_EYE":       (0.47, 0.23, 1.0),
        "RIGHT_EYE":      (0.53, 0.23, 1.0),
        "LEFT_EAR":       (0.44, 0.25, 1.0),
        "RIGHT_EAR":      (0.56, 0.25, 1.0),
        "LEFT_SHOULDER":  (0.40, 0.50, 1.0),
        "RIGHT_SHOULDER": (0.60, 0.50, 1.0),
        "LEFT_ELBOW":     (0.38, 0.68, 1.0),
        "RIGHT_ELBOW":    (0.62, 0.68, 1.0),
        "LEFT_WRIST":     (0.37, 0.88, 1.0),
        "RIGHT_WRIST":    (0.63, 0.88, 1.0),
        "LEFT_HIP":       (0.43, 0.78, 1.0),
        "RIGHT_HIP":      (0.57, 0.78, 1.0),
    }
    base.update(overrides)
    return base


class GestureClassificationTest(unittest.TestCase):
    def test_neutral_arms_down(self):
        self.assertEqual(pose._classify_gesture(_kp()), "neutral")

    def test_raised_hand_out_to_side_is_waving(self):
        # Right hand raised above the shoulder AND offset laterally — a greeting wave.
        kp = _kp(RIGHT_WRIST=(0.76, 0.30, 1.0), RIGHT_ELBOW=(0.70, 0.42, 1.0))
        self.assertEqual(pose._classify_gesture(kp), "waving")

    def test_wave_above_the_head_is_waving(self):
        # Wrist ABOVE the nose (enthusiastic wave). The old nose<=wrist<=shoulder band
        # rejected this; the fix must accept it.
        kp = _kp(RIGHT_WRIST=(0.78, 0.12, 1.0), RIGHT_ELBOW=(0.72, 0.30, 1.0))
        self.assertEqual(pose._classify_gesture(kp), "waving")

    def test_hand_raised_straight_up_is_raising_hand_not_waving(self):
        # Raised but directly above the shoulder (no lateral offset) → not a wave.
        kp = _kp(RIGHT_WRIST=(0.61, 0.28, 1.0), RIGHT_ELBOW=(0.61, 0.40, 1.0))
        self.assertEqual(pose._classify_gesture(kp), "raising_hand")


class LandmarkExtractionTest(unittest.TestCase):
    def test_lm_dict_reads_tasks_list_of_poses(self):
        # Tasks API: result.pose_landmarks is a LIST of per-pose landmark lists.
        Landmark = lambda x, y, v: types.SimpleNamespace(x=x, y=y, visibility=v)
        result = types.SimpleNamespace(
            pose_landmarks=[[Landmark(0.5, 0.25, 0.9), Landmark(0.4, 0.5, 0.8)]]
        )
        with mock.patch.object(pose, "_landmark_names", ["NOSE", "LEFT_SHOULDER"]):
            kp = pose._lm_dict(result)
        self.assertEqual(set(kp), {"NOSE", "LEFT_SHOULDER"})
        self.assertEqual(kp["NOSE"], (0.5, 0.25, 0.9))

    def test_lm_dict_empty_when_no_pose(self):
        self.assertEqual(pose._lm_dict(types.SimpleNamespace(pose_landmarks=[])), {})


class WorldStatePublishTest(unittest.TestCase):
    """_update_world_state must publish normalized landmarks as `pose_keypoints` so the
    GUI skeleton overlay has data, and clear them when the body leaves frame."""

    def setUp(self):
        from world_state import world_state
        self.world_state = world_state
        world_state.mutate("people", lambda _cur: [{"id": "person_1", "face_visible": True}])
        self.addCleanup(lambda: world_state.mutate("people", lambda _cur: []))

    def _detected(self):
        return [{
            "pose": "facing_forward", "gesture": "waving", "engagement": "high",
            "age_estimate": "adult", "position": (0.5, 0.5),
            "keypoints": {"NOSE": (0.5, 0.3, 0.9), "LEFT_SHOULDER": (0.4, 0.5, 0.8)},
        }]

    def test_publishes_pose_keypoints(self):
        pose._update_world_state(self._detected())
        person = self.world_state.get("people")[0]
        self.assertEqual(person["gesture"], "waving")
        self.assertEqual(person["pose_keypoints"]["NOSE"], (0.5, 0.3, 0.9))

    def test_clears_pose_keypoints_when_no_body(self):
        pose._update_world_state(self._detected())
        pose._update_world_state([])  # body left frame
        person = self.world_state.get("people")[0]
        self.assertIsNone(person["pose_keypoints"])


class WaveSpeedTest(unittest.TestCase):
    """recent_wave_speed measures how fast the raised wrist is sweeping (for mirroring)."""

    def setUp(self):
        pose._wave_motion.clear()
        self.addCleanup(pose._wave_motion.clear)

    def _feed(self, amp):
        import time
        pose._wave_motion.clear()
        t = time.monotonic()
        for i in range(6):  # 6 samples over ~1s, alternating sides by `amp`
            pose._wave_motion.append((t - (5 - i) * 0.2, 0.5 + (amp if i % 2 else -amp)))

    def test_fast_wave_measures_higher_than_slow(self):
        self._feed(0.03)
        slow = pose.recent_wave_speed()
        self._feed(0.18)
        fast = pose.recent_wave_speed()
        self.assertIsNotNone(slow)
        self.assertIsNotNone(fast)
        self.assertGreater(fast, slow)

    def test_too_few_samples_returns_none(self):
        import time
        pose._wave_motion.clear()
        pose._wave_motion.append((time.monotonic(), 0.5))
        self.assertIsNone(pose.recent_wave_speed())


class ModelLoadingTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(pose._reset_for_tests)
        pose._reset_for_tests()

    def test_disabled_flag_skips_load(self):
        with mock.patch.object(pose.config, "POSE_DETECTION_ENABLED", False):
            self.assertFalse(pose._load_model())

    def test_missing_model_file_disables_without_error(self):
        missing = Path("/nonexistent/pose_landmarker_lite.task")
        with (
            mock.patch.object(pose.config, "POSE_DETECTION_ENABLED", True),
            mock.patch.object(pose, "_model_path", return_value=missing),
            self.assertLogs("vision.pose", level="WARNING") as logs,
        ):
            self.assertFalse(pose._load_model())
        self.assertIn("model missing", "\n".join(logs.output).lower())

    def test_detect_pose_none_frame_returns_empty(self):
        self.assertEqual(pose.detect_pose(None), [])


if __name__ == "__main__":
    unittest.main()
