"""
Tests for the idle "mind of his own" head wander (intelligence/consciousness.py):
when the conversation lulls while Rex is locked on a face, he looks around the room,
returns his gaze, and may randomly re-greet.
"""

import unittest
from unittest import mock

import config
from state import State
from intelligence import consciousness as c


class _WanderBase(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "engaged": c._engaged_last_touch_at,
            "recent": c._recent_engaged_touch_at,
            "proactive": c._last_proactive_speech_at,
            "lock": dict(c._face_tracking_lock),
            "wander": dict(c._idle_wander),
        }
        c._engaged_last_touch_at = 0.0
        c._recent_engaged_touch_at = 0.0
        c._last_proactive_speech_at = 0.0
        c._face_tracking_lock = {}
        with c._idle_wander_lock:
            c._idle_wander.update({
                "active": False, "until": 0.0, "waypoints": [], "index": 0,
                "reached_at": 0.0, "last_at": 0.0, "pending_regreet": False,
                "regreet_deadline": 0.0,
            })

    def tearDown(self):
        c._engaged_last_touch_at = self._saved["engaged"]
        c._recent_engaged_touch_at = self._saved["recent"]
        c._last_proactive_speech_at = self._saved["proactive"]
        c._face_tracking_lock = self._saved["lock"]
        with c._idle_wander_lock:
            c._idle_wander.clear()
            c._idle_wander.update(self._saved["wander"])

    def _servo(self, *, manual=False, speech=False, listening=False):
        m = mock.Mock()
        m.manual_override_enabled.return_value = manual
        m.speech_motion_active.return_value = speech
        m.listening_motion_active.return_value = listening
        return m

    def _at_pose(self, neck=6000, lift=6000, tilt=4320):
        return mock.patch.object(
            c, "_current_servo_position",
            side_effect=lambda n: {"neck": neck, "headlift": lift, "headtilt": tilt}[n],
        )


class ConversationIdleTest(_WanderBase):
    def test_zero_before_any_interaction(self):
        self.assertEqual(c._conversation_idle_secs(1000.0), 0.0)

    def test_measures_since_last_interaction(self):
        c._engaged_last_touch_at = 1000.0
        self.assertAlmostEqual(c._conversation_idle_secs(1025.0), 25.0)
        c._last_proactive_speech_at = 1020.0  # Rex spoke more recently than the user
        self.assertAlmostEqual(c._conversation_idle_secs(1025.0), 5.0)


class StartWanderTest(_WanderBase):
    def test_start_builds_route_returning_to_gaze_and_releases_lock(self):
        c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": 999.0}
        with self._at_pose(neck=6000, lift=6000, tilt=4320):
            c._start_idle_head_wander(1000.0)
        self.assertTrue(c._idle_wander["active"])
        wps = c._idle_wander["waypoints"]
        self.assertGreaterEqual(len(wps), 2)
        self.assertEqual(wps[-1], (6000, 6000, 4320))   # last waypoint = return to pre-wander gaze
        self.assertEqual(c._face_tracking_lock, {})      # let go of the lock to look away

    def test_waypoints_within_servo_limits(self):
        c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": 999.0}
        with self._at_pose():
            c._start_idle_head_wander(1000.0)
        n = config.SERVO_CHANNELS["neck"]
        for (neck, lift, tilt) in c._idle_wander["waypoints"]:
            self.assertGreaterEqual(neck, n["min"])
            self.assertLessEqual(neck, n["max"])


class DriveWanderTest(_WanderBase):
    def _set_wander(self, **over):
        base = {"active": True, "until": 2000.0, "waypoints": [(9000, 6000, 4320)],
                "index": 0, "reached_at": 0.0}
        base.update(over)
        with c._idle_wander_lock:
            c._idle_wander.update(base)

    def test_drive_steps_head_toward_waypoint(self):
        self._set_wander()
        c._engaged_last_touch_at = 1000.0  # idle (now 1100 → 100s)
        servo = self._servo()
        with self._at_pose(neck=6000):
            c._drive_idle_head_wander(servo, 1100.0)
        servo.set_servos.assert_called_once()
        updates = servo.set_servos.call_args.args[0]
        neck_ch = config.SERVO_CHANNELS["neck"]["ch"]
        self.assertIn(neck_ch, updates)
        self.assertGreater(updates[neck_ch], 6000)  # moving toward 9000
        self.assertTrue(c._idle_wander["active"])    # not done yet

    def test_drive_aborts_on_speech_without_regreet(self):
        self._set_wander()
        c._engaged_last_touch_at = 1000.0
        servo = self._servo(speech=True)
        with self._at_pose():
            c._drive_idle_head_wander(servo, 1100.0)
        self.assertFalse(c._idle_wander["active"])
        self.assertFalse(c._idle_wander["pending_regreet"])  # interrupted → no re-greet
        servo.set_servos.assert_not_called()

    def test_drive_aborts_when_conversation_resumes(self):
        self._set_wander()
        c._engaged_last_touch_at = 1099.5  # someone just spoke → idle 0.5s < 2.0
        servo = self._servo()
        with self._at_pose():
            c._drive_idle_head_wander(servo, 1100.0)
        self.assertFalse(c._idle_wander["active"])
        self.assertFalse(c._idle_wander["pending_regreet"])

    def test_drive_finishes_when_route_done_with_regreet(self):
        self._set_wander(index=1)  # index past the single waypoint
        c._engaged_last_touch_at = 1000.0
        servo = self._servo()
        with self._at_pose():
            c._drive_idle_head_wander(servo, 1100.0)
        self.assertFalse(c._idle_wander["active"])
        self.assertTrue(c._idle_wander["pending_regreet"])   # completed → eligible to re-greet

    def test_drive_advances_waypoint_after_dwell(self):
        self._set_wander(waypoints=[(6000, 6000, 4320), (9000, 6000, 4320)], index=0, reached_at=0.0)
        c._engaged_last_touch_at = 1000.0
        servo = self._servo()
        # Already AT waypoint 0 (6000,6000,4320): first drive stamps reached_at...
        with self._at_pose(neck=6000):
            c._drive_idle_head_wander(servo, 1100.0)
        self.assertGreater(c._idle_wander["reached_at"], 0.0)
        self.assertEqual(c._idle_wander["index"], 0)
        # ...after the dwell elapses, it advances to waypoint 1.
        with self._at_pose(neck=6000):
            c._drive_idle_head_wander(servo, 1100.0 + config.IDLE_HEAD_WANDER_DWELL_SECS + 0.1)
        self.assertEqual(c._idle_wander["index"], 1)


class StepDecisionTest(_WanderBase):
    def _start_ctx(self, *, idle_touch=1000.0, now=2000.0, lock=True, directed=False, **servo_kw):
        if lock:
            c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": now - 0.5}
        c._engaged_last_touch_at = idle_touch
        servo = self._servo(**servo_kw)
        return [
            mock.patch.object(c.time, "monotonic", return_value=now),
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c, "is_waiting_for_response", return_value=False),
            mock.patch.object(c, "directed_gaze_hold_active", return_value=directed),
            mock.patch.object(c, "_within_startup_group_window", return_value=False),
            mock.patch.object(c, "_startup_known_greeting_pending", return_value=False),
            mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled),
            mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active),
            mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active),
        ]

    def _run(self, patches, *, rand=0.0):
        with mock.patch.object(c.random, "random", return_value=rand), \
             mock.patch.object(c, "_start_idle_head_wander") as start:
            for p in patches:
                p.start()
            try:
                c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
            finally:
                for p in reversed(patches):
                    p.stop()
        return start

    def test_starts_when_idle_locked_and_chance_hits(self):
        start = self._run(self._start_ctx(), rand=0.0)
        start.assert_called_once()

    def test_no_start_when_chance_misses(self):
        start = self._run(self._start_ctx(), rand=0.99)
        start.assert_not_called()

    def test_no_start_when_not_idle(self):
        start = self._run(self._start_ctx(idle_touch=1995.0), rand=0.0)  # only 5s idle
        start.assert_not_called()

    def test_no_start_without_a_lock(self):
        start = self._run(self._start_ctx(lock=False), rand=0.0)
        start.assert_not_called()

    def test_no_start_while_speaking(self):
        start = self._run(self._start_ctx(speech=True), rand=0.0)
        start.assert_not_called()

    def test_no_start_within_cooldown(self):
        with c._idle_wander_lock:
            c._idle_wander["last_at"] = 1980.0  # 20s ago at now=2000 < 30s cooldown
        start = self._run(self._start_ctx(), rand=0.0)
        start.assert_not_called()

    def test_no_start_during_directed_gaze_hold(self):
        start = self._run(self._start_ctx(directed=True), rand=0.0)
        start.assert_not_called()

    def test_regreet_fires_on_relock(self):
        with c._idle_wander_lock:
            c._idle_wander.update({"pending_regreet": True, "regreet_deadline": 9999.0})
        c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": 1999.5}
        with mock.patch.object(c.time, "monotonic", return_value=2000.0), \
             mock.patch.object(c.random, "random", return_value=0.0), \
             mock.patch.object(c, "_maybe_fire_wander_regreet") as rg:
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        rg.assert_called_once()
        self.assertFalse(c._idle_wander["pending_regreet"])

    def test_regreet_skipped_keeps_looking(self):
        with c._idle_wander_lock:
            c._idle_wander.update({"pending_regreet": True, "regreet_deadline": 9999.0})
        c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": 1999.5}
        with mock.patch.object(c.time, "monotonic", return_value=2000.0), \
             mock.patch.object(c.random, "random", return_value=0.99), \
             mock.patch.object(c, "_maybe_fire_wander_regreet") as rg:
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        rg.assert_not_called()
        self.assertFalse(c._idle_wander["pending_regreet"])  # consumed either way

    def test_step_finishes_overdue_stuck_wander(self):
        # Backstop: if the face loop never finished a wander (e.g. tracking suspended /
        # frames missing), the 1Hz step ends it once it's well past its deadline.
        with c._idle_wander_lock:
            c._idle_wander.update({"active": True, "until": 1000.0})
        with mock.patch.object(c.time, "monotonic", return_value=1000.0 + 10.0), \
             mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE):
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        self.assertFalse(c._idle_wander["active"])
        self.assertFalse(c._idle_wander["pending_regreet"])  # a stall is not a clean finish

    def test_step_aborts_in_progress_wander_on_sleep(self):
        with c._idle_wander_lock:
            c._idle_wander.update({"active": True, "until": 9999.0})  # not overdue
        with mock.patch.object(c.time, "monotonic", return_value=1000.0), \
             mock.patch.object(c.state_module, "get_state", return_value=State.SLEEP):
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        self.assertFalse(c._idle_wander["active"])

    def test_pending_times_out_without_relock(self):
        with c._idle_wander_lock:
            c._idle_wander.update({"pending_regreet": True, "regreet_deadline": 1000.0})
        c._face_tracking_lock = {}  # never re-acquired
        with mock.patch.object(c.time, "monotonic", return_value=2000.0), \
             mock.patch.object(c, "_maybe_fire_wander_regreet") as rg:
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        rg.assert_not_called()
        self.assertFalse(c._idle_wander["pending_regreet"])


class RegreetSpeechTest(_WanderBase):
    def test_fires_a_presence_reaction_with_name(self):
        c._face_tracking_lock = {"key": 1, "person_id": 1, "last_seen_at": 1999.0}
        snap = {"people": [{"person_db_id": 1, "face_id": "Bret"}]}
        with mock.patch.object(c, "_generate_and_speak_presence", return_value=True) as gs:
            c._maybe_fire_wander_regreet(snap, mock.Mock(suppress_proactive=False))
        gs.assert_called_once()
        self.assertEqual(gs.call_args.kwargs.get("purpose"), "presence_reaction")

    def test_suppressed_proactive_stays_silent(self):
        with mock.patch.object(c, "_generate_and_speak_presence") as gs:
            c._maybe_fire_wander_regreet({"people": []}, mock.Mock(suppress_proactive=True))
        gs.assert_not_called()

    def test_disabled_flag_no_start(self):
        with mock.patch.object(config, "IDLE_HEAD_WANDER_ENABLED", False), \
             mock.patch.object(c, "_start_idle_head_wander") as start:
            c._step_idle_head_wander({"people": []}, mock.Mock(suppress_proactive=False))
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
