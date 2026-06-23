"""
Phantom-pose rejection (vision/pose._is_plausible_pose).

At num_poses>1 MediaPipe hallucinates weak skeletons onto ceiling lights / reflections.
The filter must DROP those while KEEPING real bodies — including upper-body-only (hips out
of frame) and side-on (one shoulder occluded). These lock both directions so the fix can't
silently start eating real people.
"""

import unittest
from unittest import mock

import config
from vision import pose


def _kp(**vis):
    """Build a keypoint dict. Each kwarg NAME=visibility places that landmark; shoulders
    default to a plausible width unless x-overrides are given via NAME_x."""
    coords = {
        "LEFT_SHOULDER": (0.42, 0.40),
        "RIGHT_SHOULDER": (0.58, 0.40),
        "LEFT_HIP": (0.45, 0.75),
        "RIGHT_HIP": (0.55, 0.75),
        "NOSE": (0.50, 0.25),
    }
    out = {}
    for name, v in vis.items():
        if name.endswith("_x"):
            continue
        x, y = coords.get(name, (0.5, 0.5))
        x = vis.get(name + "_x", x)
        out[name] = (x, y, float(v))
    return out


class IsPlausiblePoseTest(unittest.TestCase):
    # --- real bodies: must be KEPT ---
    def test_frontal_person_kept(self):
        self.assertTrue(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.97, RIGHT_SHOULDER=0.96, LEFT_HIP=0.9, RIGHT_HIP=0.9)))

    def test_upper_body_only_kept(self):
        # Person at a desk: shoulders strong, hips out of frame (low vis).
        self.assertTrue(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.93, RIGHT_SHOULDER=0.91, LEFT_HIP=0.05, RIGHT_HIP=0.05)))

    def test_side_on_kept(self):
        # One shoulder occluded, but a shoulder + hip form a visible torso column.
        self.assertTrue(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.88, RIGHT_SHOULDER=0.20,
                LEFT_HIP=0.82, RIGHT_HIP=0.15)))

    # --- phantoms: must be DROPPED ---
    def test_ceiling_light_phantom_dropped(self):
        # All core landmarks low-visibility — the classic blob hallucination.
        self.assertFalse(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.30, RIGHT_SHOULDER=0.28,
                LEFT_HIP=0.25, RIGHT_HIP=0.22)))

    def test_collapsed_width_blob_dropped(self):
        # Both shoulders "visible" but sitting on top of each other (near-zero width).
        self.assertFalse(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.85, RIGHT_SHOULDER=0.85,
                LEFT_SHOULDER_x=0.501, RIGHT_SHOULDER_x=0.500,
                LEFT_HIP=0.10, RIGHT_HIP=0.10)))

    def test_frame_spanning_blob_dropped(self):
        # Both shoulders "visible" but on opposite frame edges (width ~0.9) — a real torso
        # never spans the frame; the upper-width bound rejects it.
        self.assertFalse(pose._is_plausible_pose(
            _kp(LEFT_SHOULDER=0.9, RIGHT_SHOULDER=0.9,
                LEFT_SHOULDER_x=0.04, RIGHT_SHOULDER_x=0.96,
                LEFT_HIP=0.1, RIGHT_HIP=0.1)))

    def test_empty_dropped(self):
        self.assertFalse(pose._is_plausible_pose({}))

    # --- kill switch ---
    def test_filter_disabled_keeps_everything(self):
        with mock.patch.object(config, "POSE_PHANTOM_FILTER_ENABLED", False):
            self.assertTrue(pose._is_plausible_pose(
                _kp(LEFT_SHOULDER=0.1, RIGHT_SHOULDER=0.1, LEFT_HIP=0.1, RIGHT_HIP=0.1)))

    def test_threshold_is_configurable(self):
        weak = _kp(LEFT_SHOULDER=0.55, RIGHT_SHOULDER=0.55, LEFT_HIP=0.05, RIGHT_HIP=0.05)
        with mock.patch.object(config, "POSE_MIN_TORSO_VISIBILITY", 0.6):
            self.assertFalse(pose._is_plausible_pose(weak))   # 0.55 < 0.6 -> phantom
        with mock.patch.object(config, "POSE_MIN_TORSO_VISIBILITY", 0.5):
            self.assertTrue(pose._is_plausible_pose(weak))    # 0.55 >= 0.5 -> real


if __name__ == "__main__":
    unittest.main()
