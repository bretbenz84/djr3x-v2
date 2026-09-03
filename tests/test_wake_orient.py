"""Name-call reflex: "hey Rex" from off camera turns him toward the voice —
neck within reach, base beyond it, an about-face for a call from behind, and
a full neck glance whenever the base may not turn.

    venv/bin/python -m unittest tests.test_wake_orient
"""

import time
import unittest
from unittest import mock

import config
from hardware import flex_doa
from intelligence import motion_agency as MA

_WANDER_OFF = mock.patch.object(config, "MOTION_IDLE_WANDER_ENABLED", False, create=True)
_STARTUP_OFF = mock.patch.object(config, "MOTION_STARTUP_APPROACH_ENABLED", False, create=True)


def setUpModule():
    _WANDER_OFF.start()
    _STARTUP_OFF.start()


def tearDownModule():
    _WANDER_OFF.stop()
    _STARTUP_OFF.stop()


class OrientToVoiceTest(unittest.TestCase):
    def setUp(self):
        MA.cancel_requested_come("test reset")
        MA._state.update(wake_orient_at=0.0, last_turn_at=0.0, hold_at=None,
                         traction_fails=0, no_traction_until=0.0)
        self._people = []
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch("sequences.animations.travel_glance_pose"),
            mock.patch("intelligence.consciousness.hold_directed_gaze"),
            mock.patch.object(MA, "no_drive_room", return_value=None),
            mock.patch.object(MA, "_clear_idle_wander"),
            mock.patch("world_state.world_state.get",
                       side_effect=lambda key: (self._people if key == "people" else
                                                {"servo_positions": {"neck": 5472}} if key == "self_state" else {})),
            mock.patch.object(config, "WAKE_ORIENT_REFLEX_ENABLED", True, create=True),
            mock.patch.object(config, "WAKE_ORIENT_COOLDOWN_SECS", 3.0, create=True),
        ]
        started = [p.start() for p in self._patches]
        self.turn, self.glance, self.hold = started[2], started[3], started[4]

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_within_the_neck_it_glances(self):
        self.assertEqual(MA.orient_to_voice(30.0, share=0.9), "glanced")
        self.turn.assert_not_called()
        self.glance.assert_called_once()
        self.assertEqual(self.glance.call_args[0][0], "left")          # + = left
        self.assertAlmostEqual(self.glance.call_args[1]["fraction"], 30.0 / 45.0)
        self.hold.assert_called_once()

    def test_beyond_the_neck_the_base_turns(self):
        self.assertEqual(MA.orient_to_voice(-120.0, share=0.9), "turned")
        self.glance.assert_not_called()
        self.turn.assert_called_once()
        self.assertAlmostEqual(self.turn.call_args[0][0], -120.0)       # not clamped to 60

    def test_a_call_from_behind_is_an_about_face(self):
        self.assertEqual(MA.orient_to_voice(175.0, share=0.9), "turned")
        self.assertAlmostEqual(self.turn.call_args[0][0], 175.0)

    def test_already_facing_does_nothing(self):
        self.assertEqual(MA.orient_to_voice(8.0, share=0.9), "facing")
        self.turn.assert_not_called()
        self.glance.assert_not_called()

    def test_caller_on_camera_does_nothing(self):
        # A face dead centre in the frame sits at -(yaw offset) in the base
        # frame under the calibrated lens; a voice from there is the visible person.
        self._people = [{"person_db_id": 1, "face_box": (860, 400, 200, 200), "face_visible": True}]
        offset = float(getattr(config, "VOICE_BEARING_CAM_YAW_OFFSET_DEG", 0.0))
        self.assertEqual(MA.orient_to_voice(-offset - 20.0, share=0.9), "on_camera")
        self.glance.assert_not_called()
        # ...but a voice 60° away from that face is somebody else: glance.
        self.assertEqual(MA.orient_to_voice(-offset + 40.0, share=0.9), "glanced")

    def test_no_drive_room_gets_a_full_glance_instead(self):
        with mock.patch.object(MA, "no_drive_room", return_value=("living room", "carpet")):
            self.assertEqual(MA.orient_to_voice(-120.0, share=0.9), "no_drive_glance")
        self.turn.assert_not_called()
        self.assertEqual(self.glance.call_args[0][0], "right")
        self.assertEqual(self.glance.call_args[1]["fraction"], 1.0)

    def test_dont_move_gets_a_full_glance_instead(self):
        MA.note_user_hold("test")
        self.assertEqual(MA.orient_to_voice(120.0, share=0.9), "held_glance")
        self.turn.assert_not_called()
        self.assertEqual(self.glance.call_args[0][0], "left")

    def test_refused_turn_falls_back_to_a_glance(self):
        self.turn.return_value = None
        self.assertEqual(MA.orient_to_voice(-120.0, share=0.9), "turn_refused_glance")
        self.glance.assert_called_once()

    def test_cooldown(self):
        self.assertEqual(MA.orient_to_voice(30.0, share=0.9), "glanced")
        self.assertEqual(MA.orient_to_voice(-30.0, share=0.9), "cooldown")

    def test_weak_cluster_and_disabled(self):
        self.assertEqual(MA.orient_to_voice(90.0, share=0.2), "weak")
        with mock.patch.object(config, "WAKE_ORIENT_REFLEX_ENABLED", False, create=True):
            self.assertEqual(MA.orient_to_voice(90.0, share=0.9), "disabled")
        self.glance.assert_not_called()
        self.turn.assert_not_called()


class WakeHookTest(unittest.TestCase):
    """interaction._start_wake_orient_reflex reads the DoA over the phrase and
    hands the bearing to motion_agency, stashing it as the turn's voice bearing."""

    def setUp(self):
        flex_doa._reset_for_tests()

    def tearDown(self):
        flex_doa._reset_for_tests()

    def test_bearing_over_the_phrase_drives_the_reflex(self):
        from intelligence import interaction as I
        from state import State
        now = time.monotonic()
        # 1.5 s of speech-flagged samples from the right, just before the fire.
        flex_doa._inject_for_tests([(now - 1.5 + 0.1 * i, 270.0, -90.0, True, 1.0) for i in range(15)])
        with mock.patch("intelligence.motion_agency.orient_to_voice", return_value="turned") as orient:
            worker = I._start_wake_orient_reflex("Hey_rex", State.IDLE)
            self.assertIsNotNone(worker)
            worker.join(timeout=5.0)
        orient.assert_called_once()
        self.assertAlmostEqual(orient.call_args[0][0], -90.0, delta=1.0)
        self.assertEqual(orient.call_args[1]["reason"], "wake:Hey_rex")
        self.assertIsNotNone(I._recent_voice_bearing())
        self.assertAlmostEqual(I._recent_voice_bearing()["bearing_deg"], -90.0, delta=1.0)

    def test_no_reflex_while_asleep_or_quiet(self):
        from intelligence import interaction as I
        from state import State
        self.assertIsNone(I._start_wake_orient_reflex("wakeuprex", State.SLEEP))
        self.assertIsNone(I._start_wake_orient_reflex("Hey_rex", State.QUIET))



class FieldFixes20260902Test(unittest.TestCase):
    """The 22:02 live run: radar orient undid the reflex, a spinning ring fed the
    DoA, and a 4/7-sample bearing turned him the wrong way."""

    def setUp(self):
        flex_doa._reset_for_tests()
        MA._state.update(voice_bearing_at=0.0, wake_orient_at=0.0, orient_hits=0,
                         orient_last_at=0.0, orient_visited=[], last_turn_at=0.0,
                         last_approach_at=0.0, last_flinch_at=0.0, hold_at=None,
                         traction_fails=0, no_traction_until=0.0)

    def tearDown(self):
        flex_doa._reset_for_tests()
        MA._state.update(voice_bearing_at=0.0)

    def test_samples_taken_while_the_base_moves_are_ignored(self):
        now = time.monotonic()
        flex_doa._inject_for_tests([(now - 1.0 + 0.1 * i, 105.0, 105.0, True, 1.0, True) for i in range(10)])
        self.assertIsNone(flex_doa.bearing_between(now - 1.2, now))
        flex_doa._inject_for_tests([(now - 0.5 + 0.1 * i, 30.0, 30.0, True, 1.0, False) for i in range(5)])
        res = flex_doa.bearing_between(now - 1.2, now)
        self.assertAlmostEqual(res["bearing_deg"], 30.0)
        self.assertEqual(res["n"], 5)

    def test_thin_cluster_does_not_turn(self):
        with mock.patch.object(MA.motion_controller, "turn", return_value=7) as turn, \
             mock.patch("sequences.animations.travel_glance_pose"), \
             mock.patch("intelligence.consciousness.hold_directed_gaze"):
            self.assertEqual(MA.orient_to_voice(-65.0, share=0.57, samples=2), "thin")
            turn.assert_not_called()

    def test_radar_orient_stands_down_while_a_voice_bearing_is_fresh(self):
        from tests.test_motion_agency import _profile
        body = {"bearing_deg": 120.0, "range_m": 1.5, "confidence": 0.9, "hits": 5, "frames": 8}
        patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
            mock.patch("sequences.animations.travel_glance_pose"),
            mock.patch("intelligence.consciousness.hold_directed_gaze"),
            mock.patch.object(MA, "_radar_bodies", return_value=([body], True)),
            mock.patch.object(config, "MOTION_RADAR_ORIENT_VOICE_DEFER_SECS", 20.0, create=True),
            mock.patch.object(config, "MOTION_RADAR_ORIENT_ENABLED", True, create=True),
        ]
        started = [p.start() for p in patches]
        turn = started[2]
        try:
            MA.note_voice_bearing(-10.0)
            for _ in range(4):
                MA.step({"people": []}, _profile())
            turn.assert_not_called()
            MA._state["voice_bearing_at"] = time.monotonic() - 60.0     # stale — radar may act again
            for _ in range(4):
                MA.step({"people": []}, _profile())
            turn.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_samples_during_rex_playback_are_ignored(self):
        now = time.monotonic()
        # Injected as the poller marks them while Rex plays: the exclusion flag set.
        flex_doa._inject_for_tests([(now - 1.0 + 0.1 * i, 9.0, 9.0, True, 1.0, True) for i in range(10)])
        self.assertIsNone(flex_doa.bearing_between(now - 1.2, now))

    def test_facing_is_judged_by_the_head_not_the_body(self):
        # Voice 9° left of the body while the head is parked 45° to the RIGHT:
        # the camera is 54° off the caller — glance, do not shrug.
        with mock.patch.object(MA, "_come_neck_bearing_deg", return_value=45.0), \
             mock.patch.object(MA.motion_controller, "available", return_value=True), \
             mock.patch.object(MA.motion, "state", return_value="idle"), \
             mock.patch.object(MA.motion_controller, "turn", return_value=7) as turn, \
             mock.patch("sequences.animations.travel_glance_pose") as glance, \
             mock.patch("intelligence.consciousness.hold_directed_gaze"), \
             mock.patch.object(MA, "_clear_idle_wander"), \
             mock.patch("world_state.world_state.get", return_value=[]):
            self.assertEqual(MA.orient_to_voice(9.0, share=0.9, samples=6), "glanced")
            glance.assert_called_once()
            turn.assert_not_called()
        # Same voice with the head already on it: facing.
        with mock.patch.object(MA, "_come_neck_bearing_deg", return_value=-9.0):
            MA._state["wake_orient_at"] = 0.0
            self.assertEqual(MA.orient_to_voice(9.0, share=0.9, samples=6), "facing")

    def test_busy_base_is_waited_out(self):
        states = iter(["turning", "turning", "idle", "idle", "idle"])
        with mock.patch.object(MA.motion_controller, "available", return_value=True), \
             mock.patch.object(MA.motion, "state", side_effect=lambda: next(states, "idle")), \
             mock.patch.object(MA.motion_controller, "turn", return_value=7) as turn, \
             mock.patch.object(MA, "no_drive_room", return_value=None), \
             mock.patch.object(MA, "_clear_idle_wander"), \
             mock.patch("world_state.world_state.get", return_value=[]), \
             mock.patch.object(config, "WAKE_ORIENT_BASE_WAIT_SECS", 2.0, create=True):
            self.assertEqual(MA.orient_to_voice(-120.0, share=0.9, samples=12), "turned")
            turn.assert_called_once()


class OverHereTest(unittest.TestCase):
    """"Over here" turns him toward the voice that said it (owner spec 2026-09-02)."""

    def setUp(self):
        flex_doa._reset_for_tests()
        from intelligence import interaction as I
        I._last_voice_bearing = None
        MA._state.update(wake_orient_at=0.0)

    def tearDown(self):
        flex_doa._reset_for_tests()

    def test_phrases(self):
        from intelligence import interaction as I
        for text in ("Over here.", "over here", "I'm over here", "Over here, Rex.",
                     "Hey Rex, over here!", "Here I am.", "This way.", "I'm right here",
                     "Rex, I'm over here."):
            self.assertTrue(I._over_here_phrase(text), text)
        for text in ("Come over here.", "come here", "Look over there.", "over there",
                     "Put it over here.", "I was over here yesterday.", ""):
            self.assertFalse(I._over_here_phrase(text), text)

    def test_reflex_uses_this_turns_bearing(self):
        from intelligence import interaction as I
        I._last_voice_bearing = {"bearing_deg": 120.0, "share": 0.9, "cluster_n": 9, "n": 10,
                                 "at": time.monotonic()}
        with mock.patch("intelligence.motion_agency.orient_to_voice", return_value="turned") as orient:
            worker = I._start_over_here_reflex("Over here, Rex.")
            self.assertIsNotNone(worker)
            worker.join(timeout=5.0)
        orient.assert_called_once()
        self.assertAlmostEqual(orient.call_args[0][0], 120.0)
        self.assertEqual(orient.call_args[1]["reason"], "over_here")

    def test_stale_bearing_or_other_phrase_does_nothing(self):
        from intelligence import interaction as I
        I._last_voice_bearing = {"bearing_deg": 120.0, "share": 0.9, "cluster_n": 9, "n": 10,
                                 "at": time.monotonic() - 30.0}
        self.assertIsNone(I._start_over_here_reflex("Over here."))
        I._last_voice_bearing = {"bearing_deg": 120.0, "share": 0.9, "cluster_n": 9, "n": 10,
                                 "at": time.monotonic()}
        self.assertIsNone(I._start_over_here_reflex("What time is it?"))

    def test_mid_come_search_the_turn_becomes_a_search_leg(self):
        with mock.patch.object(MA, "requested_come_active", return_value=True), \
             mock.patch.object(MA, "_issue_come_turn", return_value=7) as leg, \
             mock.patch.object(MA, "_adopt_voice_bearing_turn") as adopt, \
             mock.patch.object(MA, "_come_neck_bearing_deg", return_value=0.0), \
             mock.patch("world_state.world_state.get", return_value=[]):
            self.assertEqual(MA.orient_to_voice(150.0, share=0.9, samples=8, reason="over_here"),
                             "come_leg")
            leg.assert_called_once()
            self.assertAlmostEqual(leg.call_args[0][0], 150.0)
            adopt.assert_called_once()

class RotationDetectorTest(unittest.TestCase):
    """The DoA poller excludes samples while the base ROTATES — judged from the
    gyro yaw step, not `state != idle` (which blanked windows around the idle
    wander's 5° sways, field 2026-09-02 22:33)."""

    def setUp(self):
        flex_doa._last_yaw = None

    def _tele(self, yaw=None, state="idle", ok=True):
        t = {"state": state}
        if yaw is not None:
            t["imu"] = {"ok": ok, "yaw": yaw}
        return t

    def test_small_yaw_steps_are_not_motion(self):
        with mock.patch("hardware.motion.telemetry", side_effect=[self._tele(10.0), self._tele(10.4), self._tele(10.9)]):
            self.assertFalse(flex_doa._base_moving())   # first sample seeds
            self.assertFalse(flex_doa._base_moving())   # 0.4° step
            self.assertFalse(flex_doa._base_moving())   # 0.5° step

    def test_a_real_turn_is_motion_even_if_state_lags(self):
        with mock.patch("hardware.motion.telemetry", side_effect=[self._tele(10.0, state="idle"), self._tele(14.0, state="idle")]):
            flex_doa._base_moving()
            self.assertTrue(flex_doa._base_moving())     # 4° in one poll = 40°/s

    def test_wrap_is_handled(self):
        with mock.patch("hardware.motion.telemetry", side_effect=[self._tele(179.5), self._tele(-179.8)]):
            flex_doa._base_moving()
            self.assertFalse(flex_doa._base_moving())    # 0.7° across the wrap

    def test_without_an_imu_the_state_gate_applies(self):
        with mock.patch("hardware.motion.telemetry", side_effect=[self._tele(state="turning"), self._tele(state="idle")]):
            self.assertTrue(flex_doa._base_moving())
            self.assertFalse(flex_doa._base_moving())

    def test_no_new_wander_right_after_a_voice(self):
        from tests.test_motion_agency import _profile
        MA._state.update(wander_pending=None, wander_next_at=0.0, last_turn_at=0.0,
                         last_approach_at=0.0, last_flinch_at=0.0, no_traction_until=0.0)
        MA.note_voice_bearing(30.0)
        with mock.patch.object(MA, "_wander_clearances") as clear:
            self.assertFalse(MA._maybe_idle_wander(_profile(), time.monotonic()))
            clear.assert_not_called()        # refused before it even looked at the room
        MA._state["voice_bearing_at"] = 0.0

    def test_a_voice_drops_an_in_flight_wander(self):
        MA._state["wander_pending"] = {"at": time.monotonic(), "steps": [], "idx": 0}
        MA.note_voice_bearing(30.0)
        self.assertIsNone(MA._state.get("wander_pending"))


if __name__ == "__main__":
    unittest.main()
