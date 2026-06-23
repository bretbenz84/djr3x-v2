"""
The "riff on the room" idle behaviors (#30) must only spend their cooldown when they
actually speak. Previously the timestamp was stamped at the top, so a blocked gate or an
empty scene burned the full 4-5 min cooldown on a no-op, making them fire far less often.
"""

import unittest
from unittest import mock

import config
from intelligence import idle_behaviors


class IdleBehaviorCooldownTest(unittest.TestCase):
    def setUp(self):
        idle_behaviors._last_live_vision_comment_at = 0.0
        idle_behaviors._last_bored_env_snark_at = 0.0

    def _capture_worker(self, fn, *args):
        """Run fn but capture the worker passed to threading.Thread instead of spawning."""
        captured = {}

        def fake_thread(target=None, **kw):
            captured["target"] = target
            return mock.MagicMock()

        with mock.patch("intelligence.idle_behaviors.threading.Thread",
                        side_effect=fake_thread):
            fn(*args)
        return captured.get("target")

    # ── live vision comment ──
    def test_live_vision_no_cooldown_when_gate_blocks(self):
        with mock.patch.object(config, "LIVE_VISION_COMMENT_COOLDOWN_SECS", 300.0):
            worker = self._capture_worker(idle_behaviors.do_live_vision_comment, {})
            self.assertIsNotNone(worker)
            with mock.patch.object(idle_behaviors, "_c") as c:
                c._can_proactive_speak.return_value = False
                worker()
        self.assertEqual(idle_behaviors._last_live_vision_comment_at, 0.0)

    def test_live_vision_cooldown_on_success(self):
        with mock.patch.object(config, "LIVE_VISION_COMMENT_COOLDOWN_SECS", 300.0):
            worker = self._capture_worker(idle_behaviors.do_live_vision_comment, {})
            with mock.patch.object(idle_behaviors, "_c") as c, \
                 mock.patch("vision.camera.get_frame", return_value=object()), \
                 mock.patch("vision.scene.describe_scene", return_value="a cluttered desk"):
                c._can_proactive_speak.return_value = True
                worker()
        self.assertGreater(idle_behaviors._last_live_vision_comment_at, 0.0)

    # ── bored environment snark ──
    def test_bored_snark_no_cooldown_when_gate_blocks(self):
        with mock.patch.object(config, "BORED_ENV_SNARK_ENABLED", True), \
             mock.patch.object(config, "BORED_ENV_SNARK_COOLDOWN_SECS", 240.0):
            worker = self._capture_worker(idle_behaviors.do_bored_environment_snark, {})
            self.assertIsNotNone(worker)
            with mock.patch.object(idle_behaviors, "_c") as c:
                c._can_proactive_speak.return_value = False
                worker()
        self.assertEqual(idle_behaviors._last_bored_env_snark_at, 0.0)

    def test_bored_snark_no_cooldown_when_scene_empty(self):
        with mock.patch.object(config, "BORED_ENV_SNARK_ENABLED", True), \
             mock.patch.object(config, "BORED_ENV_SNARK_COOLDOWN_SECS", 240.0):
            worker = self._capture_worker(idle_behaviors.do_bored_environment_snark, {})
            with mock.patch.object(idle_behaviors, "_c") as c, \
                 mock.patch("vision.camera.get_frame", return_value=object()), \
                 mock.patch("vision.scene.describe_scene_detailed", return_value={}), \
                 mock.patch("vision.scene.describe_scene", return_value=""):
                c._can_proactive_speak.return_value = True
                worker()
        self.assertEqual(idle_behaviors._last_bored_env_snark_at, 0.0)

    def test_bored_snark_cooldown_on_success(self):
        with mock.patch.object(config, "BORED_ENV_SNARK_ENABLED", True), \
             mock.patch.object(config, "BORED_ENV_SNARK_COOLDOWN_SECS", 240.0), \
             mock.patch.object(config, "BORED_ENV_SNARK_LOOK_AROUND", False):
            worker = self._capture_worker(idle_behaviors.do_bored_environment_snark, {})
            with mock.patch.object(idle_behaviors, "_c") as c, \
                 mock.patch("vision.camera.get_frame", return_value=object()), \
                 mock.patch("vision.scene.describe_scene_detailed",
                            return_value={"overall_summary": "a dull room",
                                          "notable_details": ["a chair"]}), \
                 mock.patch.object(idle_behaviors, "_bored_snark_present_name",
                                   return_value=None), \
                 mock.patch.object(idle_behaviors, "_pick_bored_env_snark_mode",
                                   return_value="complaint"), \
                 mock.patch.object(idle_behaviors, "_bored_env_snark_prompt",
                                   return_value="ugh, this room is dull"):
                c._can_proactive_speak.return_value = True
                worker()
        self.assertGreater(idle_behaviors._last_bored_env_snark_at, 0.0)


if __name__ == "__main__":
    unittest.main()
