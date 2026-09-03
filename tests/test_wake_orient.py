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

    def test_transcribed_bare_hey_rex_fires_the_wake_reflex(self):
        # Field 2026-09-03 11:44:45: "Hey Rex" from +168° arrived as a transcript
        # (fast-acked "I'm listening"), never as a wake-model detection, and the
        # base never moved. The heard-turn hook must turn him for that too.
        from intelligence import interaction as I
        I._last_voice_bearing = {"bearing_deg": 168.0, "share": 0.6, "cluster_n": 14, "n": 24,
                                 "at": time.monotonic()}
        with mock.patch("intelligence.motion_agency.orient_to_voice", return_value="turned") as orient:
            worker = I._start_over_here_reflex("Hey Rex.")
            self.assertIsNotNone(worker)
            worker.join(timeout=5.0)
        orient.assert_called_once()
        self.assertAlmostEqual(orient.call_args[0][0], 168.0)
        self.assertEqual(orient.call_args[1]["reason"], "wake:transcribed")

    def test_wake_reflex_switch_covers_the_transcribed_path(self):
        from intelligence import interaction as I
        I._last_voice_bearing = {"bearing_deg": 168.0, "share": 0.6, "cluster_n": 14, "n": 24,
                                 "at": time.monotonic()}
        with mock.patch.object(config, "WAKE_ORIENT_REFLEX_ENABLED", False, create=True):
            self.assertIsNone(I._start_over_here_reflex("Hey Rex."))
        with mock.patch.object(config, "OVER_HERE_REFLEX_ENABLED", False, create=True):
            self.assertIsNone(I._start_over_here_reflex("Over here."))

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


class _FakeFlex:
    """A chip whose DOA_VALUE lags the beam azimuth, like the real one."""

    def __init__(self, doa, beam_deg, energy, speech=1):
        self.doa, self.beam_deg, self.energy, self.speech = doa, beam_deg, energy, speech

    def read(self, name):
        import math as _m
        if name == "DOA_VALUE":
            return (self.doa, self.speech)
        if name == "AEC_SPENERGY_VALUES":
            return (0.0, 0.0, 0.0, self.energy)
        if name == "AEC_AZIMUTH_VALUES":
            return (0.0, 0.0, 0.0, _m.radians(self.beam_deg))
        raise KeyError(name)


class StaleHoldTest(unittest.TestCase):
    """The chip holds the previous talker's direction for ~1 s: the tail of the
    phrase decides, and the beam azimuth (which leads) is the sample while it
    carries speech energy. Field 2026-09-02 22:34:48: −1° from 7/13 while Bret
    stood 90° right."""

    def setUp(self):
        flex_doa._reset_for_tests()
        flex_doa._last_yaw = None

    def tearDown(self):
        flex_doa._reset_for_tests()

    def test_tail_of_the_phrase_outvotes_the_stale_head(self):
        now = time.monotonic()
        rows = [(now - 1.3 + 0.1 * i, 359.0, -1.0, True, 1.0, False) for i in range(7)]
        rows += [(now - 0.6 + 0.1 * i, 270.0, -90.0, True, 1.0, False) for i in range(6)]
        flex_doa._inject_for_tests(rows)
        res = flex_doa.bearing_between(now - 1.4, now)
        self.assertAlmostEqual(res["bearing_deg"], -90.0, delta=1.0)
        self.assertTrue(res["head_disagrees"])
        self.assertEqual(res["window_n"], 13)
        self.assertEqual(sorted(g[1] for g in res["clusters"]), [6, 7])   # both groups reported

    def test_beam_with_energy_is_the_sample(self):
        with mock.patch.object(flex_doa, "_dev", _FakeFlex(doa=359, beam_deg=270.0, energy=500000.0)), \
             mock.patch.object(flex_doa, "_base_moving", return_value=False), \
             mock.patch.object(flex_doa, "_self_speaking", return_value=False):
            self.assertTrue(flex_doa._poll_once())
        with flex_doa._lock:
            t, raw, bearing, speech, energy, moving, _hero, _neck = flex_doa._samples[-1]
        self.assertAlmostEqual(raw, 270.0)
        self.assertAlmostEqual(bearing, -90.0)
        self.assertTrue(speech)

    def test_beam_without_energy_falls_back_to_the_doa_register(self):
        with mock.patch.object(flex_doa, "_dev", _FakeFlex(doa=90, beam_deg=236.0, energy=0.0)), \
             mock.patch.object(flex_doa, "_base_moving", return_value=False), \
             mock.patch.object(flex_doa, "_self_speaking", return_value=False):
            self.assertTrue(flex_doa._poll_once())
        with flex_doa._lock:
            _t, raw, bearing, speech, _e, _m, _h, _nk = flex_doa._samples[-1]
        self.assertAlmostEqual(raw, 90.0)
        self.assertAlmostEqual(bearing, 90.0)


class HeroArmMountTest(unittest.TestCase):
    """The ring rides on the hero-arm section (owner 2026-09-02): idle wander
    holds that arm at neutral, and each DoA sample is corrected by the arm's
    position once the ring-yaw ratio is measured."""

    def test_idle_wander_can_hold_the_hero_arm(self):
        from sequences import animations as A
        with mock.patch.object(config, "IDLE_ARM_WANDER_HEROARM_ENABLED", False, create=True), \
             mock.patch.object(A, "_current_body_pose", return_value={7: 7500, 6: 6000}):
            targets = A._idle_arm_wander_targets()
        self.assertEqual(targets[7], A.HEROARM_NEUTRAL)
        self.assertIn(6, targets)

    def test_idle_wander_can_be_re_enabled(self):
        from sequences import animations as A
        with mock.patch.object(config, "IDLE_ARM_WANDER_HEROARM_ENABLED", True, create=True), \
             mock.patch.object(A, "_current_body_pose", return_value={7: 6000, 6: 6000}):
            targets = A._idle_arm_wander_targets()
        self.assertNotEqual(targets[7], A.HEROARM_NEUTRAL)

    def test_unmeasured_ratio_means_no_correction(self):
        with mock.patch.object(config, "FLEX_DOA_HEROARM_YAW_DEG_AT_MAX", 0.0, create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(90.0, None, 8000.0), 90.0)

    def test_ring_swung_left_reports_sources_too_far_right(self):
        # Ring 0° swung 40° left at heroarm max: a source dead ahead of the BODY
        # reads 40° right on the chip (320°); the correction restores 0°.
        with mock.patch.object(config, "FLEX_DOA_HEROARM_YAW_DEG_AT_MAX", 40.0, create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(320.0, None, 8000.0), 0.0)
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0.0, None, 6000.0), 0.0)   # neutral
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0.0, None, 7000.0), 20.0)  # half throw
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0.0, None, 4000.0), -40.0, delta=0.5)

    def test_samples_carry_the_arm_position(self):
        flex_doa._reset_for_tests()
        try:
            now = time.monotonic()
            flex_doa._inject_for_tests([(now - 0.5 + 0.1 * i, 30.0, 30.0, True, 1.0, False, 6500.0) for i in range(6)])
            res = flex_doa.bearing_between(now - 0.6, now)
            self.assertAlmostEqual(res["heroarm_qus"], 6500.0)
        finally:
            flex_doa._reset_for_tests()


class RadarTiebreakTest(unittest.TestCase):
    """A persistent radar body promotes the chip group it agrees with (22:45 /
    22:46 field cases); without agreement the chip's own pick stands."""

    def _res(self, chosen, groups):
        return {"bearing_deg": chosen, "clusters": groups, "cluster_n": 6, "n": 20, "share": 0.6}

    def _bodies(self, *bearings):
        return ([{"bearing_deg": b, "range_m": 1.8, "confidence": 1.0, "hits": 5, "frames": 8}
                 for b in bearings], True)

    def test_call_one_radar_backs_the_majority_over_the_tail(self):
        res = self._res(101.0, [(-158.0, 11), (100.0, 7), (-23.0, 4)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(-152.0)), \
             mock.patch.object(config, "WAKE_ORIENT_RADAR_TIEBREAK_ENABLED", True, create=True):
            bearing, note = MA.resolve_voice_bearing(res)
        self.assertAlmostEqual(bearing, -158.0)
        self.assertIn("backs", note)

    def test_call_three_radar_backs_the_minority(self):
        res = self._res(154.0, [(154.0, 14), (-131.0, 6), (115.0, 3)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(-121.0, 80.0)), \
             mock.patch.object(config, "WAKE_ORIENT_RADAR_TIEBREAK_ENABLED", True, create=True):
            bearing, _ = MA.resolve_voice_bearing(res)
        self.assertAlmostEqual(bearing, -131.0)

    def test_no_radar_agreement_keeps_the_chip_pick(self):
        res = self._res(154.0, [(154.0, 14), (-131.0, 6)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(20.0)):
            bearing, note = MA.resolve_voice_bearing(res)
        self.assertAlmostEqual(bearing, 154.0)
        self.assertIn("agrees with none", note)

    def test_radar_agreeing_with_the_pick_confirms_it(self):
        res = self._res(-90.0, [(-90.0, 9), (60.0, 4)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(-100.0)):
            bearing, note = MA.resolve_voice_bearing(res)
        self.assertAlmostEqual(bearing, -90.0)
        self.assertIn("agrees", note)

    def test_single_group_or_no_radar_is_untouched(self):
        res = self._res(-90.0, [(-90.0, 9)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(60.0)):
            self.assertAlmostEqual(MA.resolve_voice_bearing(res)[0], -90.0)
        res = self._res(154.0, [(154.0, 14), (-131.0, 6)])
        with mock.patch.object(MA, "_radar_bodies", return_value=([], True)):
            self.assertAlmostEqual(MA.resolve_voice_bearing(res)[0], 154.0)

    def test_tiny_groups_cannot_be_promoted(self):
        res = self._res(154.0, [(154.0, 14), (-131.0, 2)])
        with mock.patch.object(MA, "_radar_bodies", return_value=self._bodies(-125.0)):
            self.assertAlmostEqual(MA.resolve_voice_bearing(res)[0], 154.0)


class EnergyWeightedVoteTest(unittest.TestCase):
    """Owner observation 2026-09-02: right bearings came with high speech energy,
    wrong ones with low (reflections). A few strong samples outvote many weak."""

    def setUp(self):
        flex_doa._reset_for_tests()

    def tearDown(self):
        flex_doa._reset_for_tests()

    def test_strong_minority_beats_weak_majority(self):
        now = time.monotonic()
        rows = [(now - 2.0 + 0.1 * i, 154.0, 154.0, True, 20000.0, False) for i in range(14)]
        rows += [(now - 0.6 + 0.1 * i, 229.0, -131.0, True, 900000.0, False) for i in range(6)]
        flex_doa._inject_for_tests(rows)
        res = flex_doa.bearing_between(now - 2.1, now)
        self.assertAlmostEqual(res["bearing_deg"], -131.0, delta=1.0)
        self.assertEqual(res["clusters"][0][1], 6)          # heaviest group first
        self.assertGreater(res["clusters"][0][2], res["clusters"][1][2])

    def test_without_energy_the_tail_rule_still_applies(self):
        now = time.monotonic()
        rows = [(now - 1.3 + 0.1 * i, 359.0, -1.0, True, 0.0, False) for i in range(7)]
        rows += [(now - 0.6 + 0.1 * i, 270.0, -90.0, True, 0.0, False) for i in range(6)]
        flex_doa._inject_for_tests(rows)
        res = flex_doa.bearing_between(now - 1.4, now)
        self.assertAlmostEqual(res["bearing_deg"], -90.0, delta=1.0)

    def test_weighted_cluster_math(self):
        res = flex_doa.dominant_cluster([10.0, 12.0, -100.0], 20.0, weights=[1.0, 1.0, 10.0])
        self.assertAlmostEqual(res["bearing_deg"], -100.0)
        self.assertAlmostEqual(res["weight_share"], 10.0 / 12.0)
        res = flex_doa.dominant_cluster([10.0, 12.0, -100.0], 20.0)
        self.assertAlmostEqual(res["bearing_deg"], 11.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
