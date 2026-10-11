"""Object-directed glance: look AT the thing being asked about.

Owner request 2026-08-08: when Rex asks about something he saw (the model train
on the wall), he should turn his head toward it while asking instead of staring
at the person next to it. request_object_glance arms a glance from the object's
last detection box; _drive_object_glance runs it through the ramped gaze
stepper (away → hold → return to anchor) inside the face-tracking ownership
chain, with stall/stale deadlines so the head can never get stuck.
"""

import time
import unittest
from unittest import mock

from hardware import head_pose
from intelligence import consciousness as c


def _decoded_head(updates):
    """(pitch, tilt) a set_servos update's head pair stands for."""
    return head_pose.head_pose({ch: updates[ch] for ch in head_pose.HEAD_PAIR})


class _FakeServos:
    """Mirrors set_servos writes back into world_state.self_state.servo_positions
    (what _current_servo_position reads), like the real module does."""

    def __init__(self):
        self.writes = []

    @staticmethod
    def manual_override_enabled():
        return False

    def set_motion_profile(self, channels, speed=0, acceleration=0):
        pass

    def set_face_tracking_baseline(self, neck=None, lift=None, pitch=None, tilt=None, head=None):
        # Floats only for the head pair, like the real driver (qus here is a bug).
        for value in (pitch, tilt):
            if value is not None and abs(float(value)) > 1.5:
                raise ValueError(f"head pose fraction out of range: {value}")

    def set_servos(self, updates):
        self.writes.append(dict(updates))
        by_ch = {
            int(cfg["ch"]): name for name, cfg in c.config.SERVO_CHANNELS.items()
        }
        self_state = c.world_state.get("self_state") or {}
        positions = dict(self_state.get("servo_positions") or {})
        # The two head servos are always written together (the driver refuses one).
        present = set(head_pose.HEAD_PAIR).intersection(int(ch) for ch in updates)
        assert not present or present == set(head_pose.HEAD_PAIR), updates
        for ch, pos in updates.items():
            name = by_ch.get(int(ch))
            if name:
                positions[name] = int(pos)
        self_state["servo_positions"] = positions
        if present:
            pitch, tilt = head_pose.head_pose({ch: positions[by_ch[ch]] for ch in head_pose.HEAD_PAIR})
            self_state["head_pose"] = {"pitch": round(pitch, 3), "tilt": round(tilt, 3)}
        c.world_state.update("self_state", self_state)


class ObjectGlanceTestBase(unittest.TestCase):
    def setUp(self):
        self.old_objects = c.world_state.get("objects")
        self.old_self_state = c.world_state.get("self_state")
        self.old_gaze_drive = dict(c._gaze_drive)
        c._object_glance_release()
        self.addCleanup(self._restore)
        self_state = dict(c.world_state.get("self_state") or {})
        level = head_pose.head_pulses(0.0, 0.0)
        self_state["servo_positions"] = {
            "neck": 6000, "headlift": 6000, "visor": 5120,
            "head_left": level[head_pose.HEAD_LEFT_CH],
            "head_right": level[head_pose.HEAD_RIGHT_CH],
        }
        self_state["head_pose"] = {"pitch": 0.0, "tilt": 0.0}
        c.world_state.update("self_state", self_state)

    def _restore(self):
        c.world_state.update("objects", self.old_objects)
        c.world_state.update("self_state", self.old_self_state)
        c._gaze_drive.clear()
        c._gaze_drive.update(self.old_gaze_drive)
        c._object_glance_release()

    def _seed_object(self, label="model train", box=(1500, 200, 200, 120), age_secs=1.0):
        c.world_state.update("objects", [{
            "id": "object_1",
            "label": label,
            "position": "upper right",
            "last_seen": time.time() - age_secs,
            "confidence": 0.8,
            "source": "test",
            "box": box,
        }])


class RequestObjectGlanceTest(ObjectGlanceTestBase):
    def test_fresh_box_arms_with_correct_bearing_signs(self):
        # Box center (1600, 260) on a 1920x1080 frame: right of center and
        # high in frame -> positive yaw (right), positive pitch (up).
        self._seed_object()
        self.assertTrue(c.request_object_glance("model train", source="test"))
        self.assertEqual(c._object_glance.get("phase"), "away")
        self.assertGreater(float(c._object_glance["yaw_deg"]), 0.0)
        self.assertGreater(float(c._object_glance["pitch_deg"]), 0.0)

    def test_left_low_object_gets_negative_bearings(self):
        self._seed_object(box=(100, 800, 200, 120))
        self.assertTrue(c.request_object_glance("model train"))
        self.assertLess(float(c._object_glance["yaw_deg"]), 0.0)
        self.assertLess(float(c._object_glance["pitch_deg"]), 0.0)

    def test_stale_box_refused(self):
        self._seed_object(age_secs=120.0)
        self.assertFalse(c.request_object_glance("model train"))
        self.assertEqual(c._object_glance.get("phase"), "idle")

    def test_unseen_label_refused(self):
        self._seed_object(label="bowl")
        self.assertFalse(c.request_object_glance("model train"))

    def test_boxless_record_refused(self):
        c.world_state.update("objects", [{
            "label": "model train", "last_seen": time.time(), "confidence": 0.8,
        }])
        self.assertFalse(c.request_object_glance("model train"))

    def test_directed_hold_refuses_glance(self):
        self._seed_object()
        with mock.patch.object(c, "directed_gaze_hold_active", return_value=True):
            self.assertFalse(c.request_object_glance("model train"))

    def test_disabled_flag(self):
        self._seed_object()
        with mock.patch.object(c.config, "OBJECT_GLANCE_ENABLED", False, create=True):
            self.assertFalse(c.request_object_glance("model train"))


class DriveObjectGlanceTest(ObjectGlanceTestBase):
    def _run_until(self, servos, phase, start=1000.0, max_ticks=400, dt=0.08):
        now = start
        for _ in range(max_ticks):
            owned = c._drive_object_glance(servos, now)
            if c._object_glance.get("phase") == phase:
                return now, owned
            if not owned and c._object_glance.get("phase") == "idle":
                return now, owned
            now += dt
        self.fail(f"never reached phase {phase!r}")

    def test_full_glance_cycle_returns_to_anchor(self):
        self._seed_object()
        self.assertTrue(c.request_object_glance("model train"))
        c._object_glance["armed_at"] = 1000.0
        servos = _FakeServos()

        now, owned = self._run_until(servos, "hold", start=1000.0)
        self.assertTrue(owned)
        anchor = c._object_glance["anchor"]
        self.assertEqual(tuple(anchor), (6000, 6000, 0.0, 0.0))
        # The head actually moved toward the object (neck away from anchor).
        moved_neck = c._current_servo_position("neck")
        self.assertNotEqual(moved_neck, 6000)
        self.assertGreater(moved_neck, 6000 - 1)  # right of frame -> toward neck max
        # High in frame -> the head PITCHED up, as a pair, keeping the anchor's cock.
        head_writes = [w for w in servos.writes if head_pose.HEAD_LEFT_CH in w]
        self.assertTrue(head_writes)
        pitch, tilt = _decoded_head(head_writes[-1])
        self.assertGreater(pitch, 0.05)
        self.assertAlmostEqual(tilt, 0.0, delta=0.03)
        self.assertGreater(c._current_head_pose()[0], 0.05)

        # Hold expires -> returning -> idle, ending back at the anchor pose.
        now2, _ = self._run_until(servos, "returning", start=now + 5.0)
        for _ in range(400):
            if not c._drive_object_glance(servos, now2):
                break
            now2 += 0.08
        self.assertEqual(c._object_glance.get("phase"), "idle")
        # The return hands off to face-centering once within the tracking
        # tolerance — it need not land exactly, just back near the anchor.
        tol = int(getattr(c.config, "FACE_TRACKING_NECK_MAX_STEP_QUS", 420))
        self.assertLessEqual(abs(c._current_servo_position("neck") - 6000), tol)
        # ...and the head pitched back toward level (the anchor's pitch AND tilt).
        pitch, tilt = c._current_head_pose()
        self.assertLessEqual(abs(pitch), abs(head_pose.pitch_from_legacy_delta(tol)) + 0.01)
        self.assertAlmostEqual(tilt, 0.0, delta=0.03)

    def test_never_moved_glance_expires_quietly(self):
        self._seed_object()
        self.assertTrue(c.request_object_glance("model train"))
        c._object_glance["armed_at"] = 1000.0
        servos = _FakeServos()
        # First tick arrives long after arming (another owner held the head):
        # stand down without touching the servos.
        self.assertFalse(c._drive_object_glance(servos, 1000.0 + 30.0))
        self.assertEqual(c._object_glance.get("phase"), "idle")
        self.assertEqual(servos.writes, [])

    def test_manual_override_releases(self):
        self._seed_object()
        self.assertTrue(c.request_object_glance("model train"))
        servos = _FakeServos()
        with mock.patch.object(servos, "manual_override_enabled", create=True,
                               new=lambda: True):
            self.assertFalse(c._drive_object_glance(servos, 1000.0))
        self.assertEqual(c._object_glance.get("phase"), "idle")

    def test_idle_state_does_not_own_head(self):
        servos = _FakeServos()
        self.assertFalse(c._drive_object_glance(servos, 1000.0))
        self.assertEqual(servos.writes, [])


if __name__ == "__main__":
    unittest.main()
