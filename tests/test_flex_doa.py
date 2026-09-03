"""Voice bearing from the reSpeaker Flex XVF3800's direction of arrival.

Three layers, each pinned here without hardware:

  hardware/flex_doa.py     chip angle -> base bearing (the measured convention:
                           0 ahead, 90 LEFT, 270 RIGHT), dominant-cluster
                           selection over a segment window
  motion_agency            the voice bearing as come-here evidence: below the
                           camera and explicit words, above radar alone
  consciousness            the off-camera gaze search opens toward the voice

Run per module (never `unittest discover` — see CLAUDE.md):
    venv/bin/python -m unittest tests.test_flex_doa
"""

import time
import unittest
from unittest import mock

import config
from hardware import flex_doa
from intelligence import motion_agency as MA

from tests.test_motion_agency import _FakeRing, _profile, _snapshot

_WANDER_OFF = mock.patch.object(config, "MOTION_IDLE_WANDER_ENABLED", False, create=True)
_STARTUP_OFF = mock.patch.object(config, "MOTION_STARTUP_APPROACH_ENABLED", False, create=True)


def setUpModule():
    _WANDER_OFF.start()
    _STARTUP_OFF.start()


def tearDownModule():
    _WANDER_OFF.stop()
    _STARTUP_OFF.stop()


class ChipToBaseBearingTest(unittest.TestCase):
    """Measured 2026-09-02 with the ring's printed 0° edge forward."""

    def test_measured_convention(self):
        self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0), 0.0)
        self.assertAlmostEqual(flex_doa.chip_to_base_bearing(90), 90.0)     # Rex's LEFT
        self.assertAlmostEqual(flex_doa.chip_to_base_bearing(270), -90.0)   # Rex's RIGHT
        self.assertAlmostEqual(flex_doa.chip_to_base_bearing(180), 180.0)   # behind
        self.assertAlmostEqual(flex_doa.chip_to_base_bearing(359), -1.0)

    def test_remount_knobs(self):
        with mock.patch.object(config, "FLEX_DOA_SIGN", -1.0, create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(90), -90.0)
        with mock.patch.object(config, "FLEX_DOA_FORWARD_OFFSET_DEG", 180.0, create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(180), 0.0)
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0), 180.0)

    def test_head_mount_subtracts_the_neck_yaw(self):
        # Head turned 30° to the RIGHT hears a voice dead ahead of the head:
        # in the base frame that voice is 30° to the right.
        with mock.patch.object(config, "FLEX_DOA_MOUNT", "head", create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0, neck_yaw_right_deg=30.0), -30.0)
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(90, neck_yaw_right_deg=30.0), 60.0)
        # Base mount ignores the neck.
        with mock.patch.object(config, "FLEX_DOA_MOUNT", "base", create=True):
            self.assertAlmostEqual(flex_doa.chip_to_base_bearing(0, neck_yaw_right_deg=30.0), 0.0)


class DominantClusterTest(unittest.TestCase):
    def test_the_talker_beats_the_between_words_fallback(self):
        # The 2026-09-02 "right" run: ~2/3 of samples near -90 (270 chip), the
        # rest snapping to +86 between words. Plain median drifted to 291 chip.
        rows = [-90.0, -88.0, -93.0, -90.0, -85.0, -95.0, -90.0, -90.0] + [86.0, 86.0, 86.0, 87.0]
        res = flex_doa.dominant_cluster(rows, 20.0)
        self.assertAlmostEqual(res["bearing_deg"], -90.0, delta=2.0)
        self.assertEqual(res["cluster_n"], 8)
        self.assertAlmostEqual(res["share"], 8 / 12)

    def test_straddling_the_wrap_is_fine(self):
        res = flex_doa.dominant_cluster([178.0, -179.0, 179.0, -178.0], 20.0)
        self.assertGreater(abs(res["bearing_deg"]), 177.0)
        self.assertEqual(res["cluster_n"], 4)

    def test_empty(self):
        self.assertIsNone(flex_doa.dominant_cluster([], 20.0))


class BearingBetweenTest(unittest.TestCase):
    def setUp(self):
        flex_doa._reset_for_tests()

    def tearDown(self):
        flex_doa._reset_for_tests()

    def _rows(self, t0, bearings, speech=True, step=0.1):
        return [(t0 + i * step, (b % 360.0), b, speech, 1.0) for i, b in enumerate(bearings)]

    def test_off_poller_says_nothing(self):
        self.assertIsNone(flex_doa.bearing_between(0.0, 1.0))

    def test_window_and_speech_flag(self):
        now = time.monotonic()
        flex_doa._inject_for_tests(self._rows(now - 5.0, [30.0] * 5))          # too old
        flex_doa._inject_for_tests(self._rows(now - 1.0, [-60.0] * 6))         # in window
        flex_doa._inject_for_tests(self._rows(now - 0.3, [120.0] * 6, speech=False))  # unflagged
        res = flex_doa.bearing_between(now - 1.1, now)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["bearing_deg"], -60.0, delta=1.0)
        self.assertEqual(res["n"], 6)

    def test_too_few_samples(self):
        now = time.monotonic()
        flex_doa._inject_for_tests(self._rows(now - 0.2, [40.0, 40.0]))
        self.assertIsNone(flex_doa.bearing_between(now - 0.5, now))

    def test_no_dominant_cluster(self):
        now = time.monotonic()
        # Four talkers' worth of directions, no group holds 40 %.
        flex_doa._inject_for_tests(self._rows(now - 1.0, [0.0, 90.0, 180.0, -90.0, 0.0, 90.0, 180.0, -90.0, 45.0, 135.0]))
        with mock.patch.object(config, "FLEX_DOA_MIN_CLUSTER_SHARE", 0.4, create=True):
            self.assertIsNone(flex_doa.bearing_between(now - 1.1, now))


class _ComeFixture(unittest.TestCase):
    """The RadarFirstComeTest scaffolding from tests/test_motion_agency.py."""

    def setUp(self):
        MA.cancel_requested_come("test reset")
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0, user_motion_at=0.0,
                         realign_pending_seq=None, traction_fails=0,
                         no_traction_until=0.0, hold_at=None)
        self.ring = _FakeRing()
        self._yaw = None
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch.object(MA.motion, "done_result", return_value="completed", create=True),
            mock.patch.object(MA.motion, "telemetry", side_effect=lambda: (
                {"imu": {"ok": True, "yaw": self._yaw}} if self._yaw is not None else {})),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
            mock.patch("sequences.animations.travel_glance_pose"),
            mock.patch("hardware.radar.connected", side_effect=lambda: self.ring.connected()),
            mock.patch("hardware.radar.radar_ok", side_effect=lambda: self.ring.radar_ok()),
            mock.patch("hardware.radar.recent_targets",
                       side_effect=lambda **kw: self.ring.recent_targets(**kw)),
            mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True),
            mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.0, create=True),
            mock.patch.object(config, "MOTION_COME_RADAR_SETTLE_SECS", 0.0, create=True),
            mock.patch.object(config, "MOTION_COME_RADAR_SAMPLE_SECS", 0.0, create=True),
            mock.patch.object(config, "MOTION_COME_VOICE_BEARING_ENABLED", True, create=True),
        ]
        started = [p.start() for p in self._patches]
        self.turn, self.come = started[2], started[3]
        self._tracking = {"locked": False, "visible": False}
        self._neck = 5472
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()

    def tearDown(self):
        MA.cancel_requested_come("test cleanup")
        self._ws.stop()
        for p in self._patches:
            p.stop()

    def _tick(self):
        MA.step(_snapshot(visible=False), _profile())

    def _turns(self):
        return [c.args[0] for c in self.turn.call_args_list]


class VoiceBearingComeTest(_ComeFixture):
    def test_voice_off_axis_is_the_opening_turn(self):
        self.assertTrue(MA.request_come_here(person_id=1, voice_bearing_deg=-40.0, voice_share=0.8))
        self.assertEqual(self._turns(), [-40.0])
        self.assertEqual(MA._requested_come["scan_sign"], -1.0)

    def test_voice_nearly_ahead_does_not_turn(self):
        self.assertTrue(MA.request_come_here(person_id=1, voice_bearing_deg=8.0, voice_share=0.9))
        self.assertEqual(self._turns(), [])

    def test_spoken_direction_outranks_the_voice_bearing(self):
        # "I'm behind you, come here" while the chip heard the voice off the right.
        self.assertTrue(MA.request_come_here(person_id=1, behind=True,
                                             voice_bearing_deg=-40.0, voice_share=0.8))
        self.assertEqual(self._turns(), [180.0])
        self.assertTrue(MA._requested_come["voice_used"])

    def test_weak_cluster_is_ignored(self):
        self.assertTrue(MA.request_come_here(person_id=1, voice_bearing_deg=-40.0, voice_share=0.2))
        self.assertEqual(self._turns(), [])
        self.assertIsNone(MA._requested_come["voice_bearing_deg"])

    def test_disabled_flag_restores_the_old_behavior(self):
        with mock.patch.object(config, "MOTION_COME_VOICE_BEARING_ENABLED", False, create=True):
            self.assertTrue(MA.request_come_here(person_id=1, voice_bearing_deg=-40.0, voice_share=0.9))
        self.assertEqual(self._turns(), [])

    def test_radar_body_agreeing_with_the_voice_wins_over_the_persistent_one(self):
        # Two bodies the ring likes equally well; the more confident one is at
        # +60 (left), the voice said -45 (right). Voice turn suppressed so the
        # radar step decides on its own.
        self.ring.bodies = [(60.0, 2.0, 1.0), (-45.0, 2.5, 0.8)]
        with mock.patch.object(config, "MOTION_COME_VOICE_TURN_MIN_DEG", 90.0, create=True):
            self.assertTrue(MA.request_come_here(person_id=1, voice_bearing_deg=-40.0, voice_share=0.8))
            self.assertEqual(self._turns(), [])
            self._tick()
        self.assertEqual(len(self._turns()), 1)
        self.assertAlmostEqual(self._turns()[0], -45.0, delta=3.0)

    def test_no_agreeing_body_keeps_the_radar_order(self):
        self.ring.bodies = [(60.0, 2.0, 1.0), (-120.0, 2.5, 0.8)]
        with mock.patch.object(config, "MOTION_COME_VOICE_TURN_MIN_DEG", 90.0, create=True):
            MA.request_come_here(person_id=1, voice_bearing_deg=-40.0, voice_share=0.8)
            self._tick()
        self.assertAlmostEqual(self._turns()[0], 60.0, delta=3.0)

    def test_voice_bearing_is_a_radar_sign_not_a_camera_sign(self):
        # The one trap this whole feature lives next to (tests/test_motion_face):
        # a voice on the LEFT (+) must produce a LEFT (+) turn, never be negated.
        MA.request_come_here(person_id=1, voice_bearing_deg=70.0, voice_share=0.9)
        self.assertEqual(self._turns(), [70.0])


class GazeSearchHintTest(unittest.TestCase):
    def test_plan_opens_toward_the_voice(self):
        from intelligence import consciousness as C
        with mock.patch.object(config, "MOTION_COME_NECK_HALF_SPAN_DEG", 45.0, create=True):
            plan = C._build_speaker_gaze_search_plan("off_camera_unknown", bearing_deg=-30.0)
            neck_frac, vert = plan[0]
            self.assertAlmostEqual(neck_frac, 30.0 / 45.0)   # right of centre (+ = right)
            plan = C._build_speaker_gaze_search_plan("off_camera_unknown", bearing_deg=120.0)
            self.assertEqual(plan[0][0], -1.0)                # beyond the neck: full LEFT throw
        plain = C._build_speaker_gaze_search_plan("off_camera_unknown")
        self.assertIsNone(plain[0][0])                        # no hint: the look-down opener

    def test_intent_carries_the_hint_once(self):
        from intelligence import consciousness as C
        with mock.patch.object(C, "_person_has_visible_face", return_value=False), \
             mock.patch.object(C, "_any_visible_unknown_face", return_value=False), \
             mock.patch.object(C, "_any_visible_face", return_value=False), \
             mock.patch.object(C, "directed_gaze_hold_active", return_value=False), \
             mock.patch.object(C, "_note_startup_presence_evidence"):
            C.note_speaker_gaze_intent(None, unknown_voice=True, reason="off_camera_unknown",
                                       bearing_deg=55.0)
        with C._speaker_gaze_lock:
            self.assertEqual(C._speaker_gaze_intent.get("bearing_hint_deg"), 55.0)
            self.assertTrue(C._speaker_gaze_intent.get("search_requested"))
            C._speaker_gaze_intent.clear()


if __name__ == "__main__":
    unittest.main()


class TraceTest(unittest.TestCase):
    def test_result_carries_a_per_sample_trace(self):
        flex_doa._reset_for_tests()
        try:
            now = time.monotonic()
            flex_doa._inject_for_tests([(now - 1.0 + 0.2 * i, 30.0, 30.0, True, 1e5 * (i + 1), False)
                                        for i in range(5)])
            res = flex_doa.bearing_between(now - 1.2, now)
            self.assertEqual(len(res["trace"]), 5)
            t, b, e = res["trace"][0]
            self.assertGreater(t, 0.0)                      # seconds before the window end
            self.assertEqual(b, 30)
            text = flex_doa.describe_trace(res)
            self.assertIn("+30°/0.1M", text)
            self.assertIn("+30°/0.5M", text)
        finally:
            flex_doa._reset_for_tests()
