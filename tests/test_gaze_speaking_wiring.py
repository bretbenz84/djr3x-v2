"""
#27 — the live gaze adapter used to fully suppress the engine while SPEAKING and never
populated listener_bearings, so the ~50%-on-target speaking duty-cycle and the multi-person
include-sweep (both built into gaze_engine) could never fire. These lock the wiring: during
speech the engine runs (suppressed=False) and listener_bearings/active_speaker_id are fed
from world_state.people — deterministically, without touching servos or the RNG engine.
"""

import time
import unittest
from unittest import mock

import config
from intelligence import consciousness as c
from world_state import world_state


class GazeSpeakingWiringTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "engaged": c._engaged_person_id,
            "engaged_touch": c._engaged_last_touch_at,
            "people": world_state.get("people"),
            "phase": dict(c._gaze_drive),
        }
        self.now = time.monotonic()
        c._engaged_person_id = 1
        c._engaged_last_touch_at = self.now - 1.0   # conv_active (idle < 12s)
        c._gaze_drive["phase"] = "idle"
        world_state.update("people", [
            {"person_db_id": 1, "face_visible": True, "face_box": (860, 400, 200, 200)},   # speaker
            {"person_db_id": 2, "face_visible": True, "face_box": (1500, 400, 200, 200)},  # listener (right)
        ])

    def tearDown(self):
        c._engaged_person_id = self._saved["engaged"]
        c._engaged_last_touch_at = self._saved["engaged_touch"]
        world_state.update("people", self._saved["people"])
        c._gaze_drive.clear()
        c._gaze_drive.update(self._saved["phase"])

    def _capture_inputs(self, speaking=True):
        captured = {}

        def fake_step(inp):
            captured["inp"] = inp
            return mock.Mock(drive=False, kind="on_target")

        servo_mod = mock.Mock()
        servo_mod.manual_override_enabled.return_value = False
        servo_mod.listening_motion_active.return_value = False

        with mock.patch.object(c.gaze_engine, "enabled", lambda: True), \
             mock.patch.object(c.gaze_engine, "under_test_runner", lambda: False), \
             mock.patch.object(c.gaze_engine, "step", fake_step), \
             mock.patch.object(c, "_speaker_gaze_current_intent", return_value=None), \
             mock.patch.object(c, "directed_gaze_hold_active", return_value=False), \
             mock.patch.object(c, "_face_tracking_has_fresh_lock", return_value=True):
            c._maybe_drive_gaze(servo_mod, self.now, speech_active=speaking)
        return captured.get("inp")

    def test_engine_runs_during_speech_with_listener_bearings(self):
        inp = self._capture_inputs(speaking=True)
        self.assertIsNotNone(inp, "gaze engine should be stepped (not bailed) during speech")
        self.assertTrue(inp.speaking)
        self.assertFalse(inp.suppressed, "engine must run during speech for the duty/sweep")
        self.assertEqual(inp.active_speaker_id, 1)
        # The active speaker (1) is excluded; the right-side listener (2) gets a +yaw glance.
        ids = [pid for pid, _ in inp.listener_bearings]
        self.assertEqual(ids, [2])
        self.assertGreater(inp.listener_bearings[0][1], 0.0)   # right of frame => +deg
        self.assertLessEqual(inp.listener_bearings[0][1], 22.0)  # clamped to a glance

    def test_kill_switch_restores_speech_suppression(self):
        with mock.patch.object(config, "GAZE_SPEAKING_SWEEP_ENABLED", False):
            inp = self._capture_inputs(speaking=True)
        self.assertTrue(inp.suppressed, "with the kill switch off, speech suppresses the engine")
        self.assertEqual(inp.listener_bearings, [])


if __name__ == "__main__":
    unittest.main()
