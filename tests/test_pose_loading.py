import types
import unittest
from unittest import mock


class PoseLoadingTest(unittest.TestCase):
    def test_missing_legacy_mediapipe_solutions_disables_pose_without_error(self):
        from vision import pose

        fake_mp = types.SimpleNamespace(__version__="0.test")
        old_state = (
            pose._pose,
            pose._mp_pose,
            pose._mp_ok,
            pose._mp_attempted,
        )
        try:
            pose._pose = None
            pose._mp_pose = None
            pose._mp_ok = False
            pose._mp_attempted = False
            with (
                mock.patch.dict("sys.modules", {"mediapipe": fake_mp}),
                self.assertLogs("vision.pose", level="WARNING") as logs,
            ):
                self.assertFalse(pose._load_model())

            self.assertTrue(pose._mp_attempted)
            self.assertFalse(pose._mp_ok)
            self.assertIn("does not expose the legacy mp.solutions.pose API", "\n".join(logs.output))
        finally:
            (
                pose._pose,
                pose._mp_pose,
                pose._mp_ok,
                pose._mp_attempted,
            ) = old_state


if __name__ == "__main__":
    unittest.main()
