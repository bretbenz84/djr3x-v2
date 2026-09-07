"""Weak direction polls cannot become confident movement by repetition.

Run with tools/run_lean_checks.py; fake registers and movement only.
"""
import time
import unittest
from unittest import mock

from hardware import flex_doa as FD
from intelligence import motion_agency as MA


class VoiceDirectionQualityTest(unittest.TestCase):
    def setUp(self):
        FD._reset_for_tests()
        self.addCleanup(FD._reset_for_tests)
        self.now = time.monotonic()

    def _inject(self, samples):
        # Raw register traces are deliberately retained, including rejected polls.
        FD._inject_for_tests([
            (self.now - (len(samples)-i)*.1, bearing % 360, bearing, True, energy,
             False, 6000., 5472., bearing % 360, bearing % 360)
            for i, (bearing, energy) in enumerate(samples)
        ])

    def test_logged_group_shape_cannot_become_a_left_turn(self):
        # The log preserves group means, not the individual wake polls. This is
        # a reconstruction consistent with +91 x16/.02M, -89 x1/.12M, -29 x1/.10M.
        self._inject([(91., 20000.)]*16 + [(-89., 120000.), (-29., 100000.)])
        evidence = {}
        self.assertIsNone(FD.bearing_between(self.now-2.5, self.now, diagnostics=evidence))
        self.assertEqual(evidence["eligible_n"], 2)
        self.assertEqual(evidence["window_n"], 18)
        self.assertEqual(len(evidence["sensor_trace"]), 18)
        self.assertEqual(evidence["rejected"], "too_few_speech_energy_samples")

    def test_supported_slight_right_voice_beats_repeated_weak_left_polls(self):
        self._inject([(91., 20000.)]*16 + [(-15., 120000.)]*4)
        res = FD.bearing_between(self.now-2.5, self.now)
        self.assertAlmostEqual(res["bearing_deg"], -15.)
        self.assertEqual(res["cluster_n"], 4)
        self.assertEqual(res["n"], 4)
        # A radar return at the rejected group cannot promote it back into use.
        bodies = ([{"bearing_deg": 99., "range_m": 1.5}], True)
        with mock.patch.object(MA, "_radar_bodies", return_value=bodies):
            bearing, _ = MA.resolve_voice_bearing(res)
        self.assertAlmostEqual(bearing, -15.)

    def test_silence_is_not_an_unavailable_energy_register(self):
        self._inject([(91., 0.)]*16)
        self.assertIsNone(FD.bearing_between(self.now-2.5, self.now))

    def test_only_weak_energy_cannot_fall_back_to_sample_counts(self):
        self._inject([(91., 20000.)]*16)
        self.assertIsNone(FD.bearing_between(self.now-2.5, self.now))

    def test_single_loud_poll_cannot_establish_a_direction(self):
        self._inject([(91., 1e5), (-89., 1e5), (-29., 2e6)])
        evidence = {}
        self.assertIsNone(FD.bearing_between(self.now-1., self.now, diagnostics=evidence))
        self.assertEqual(evidence["rejected"], "too_few_agreeing_samples")

    def test_radar_promoted_group_cannot_borrow_the_winners_confidence(self):
        self._inject([(-15., 1e6)]*10 + [(91., 1e5)]*4)
        res = FD.bearing_between(self.now-2., self.now)
        self.assertAlmostEqual(res["bearing_deg"], -15.)
        bodies = ([{"bearing_deg": 99., "range_m": 1.5}], True)
        with mock.patch.object(MA, "_radar_bodies", return_value=bodies), \
             mock.patch.object(MA.motion_controller, "turn") as turn, \
             mock.patch("sequences.animations.travel_glance_pose") as glance:
            bearing, _ = MA.resolve_voice_bearing(res)
            self.assertAlmostEqual(bearing, 91.)
            self.assertAlmostEqual(res["bearing_deg"], bearing)
            self.assertEqual(res["cluster_n"], 4)
            self.assertAlmostEqual(res["share"], 4/14)
            self.assertEqual(MA.orient_to_voice(bearing, share=res["share"],
                                               samples=res["cluster_n"]), "weak")
        turn.assert_not_called()
        glance.assert_not_called()

    def test_rejected_wake_drops_old_bearing_and_never_reaches_motion(self):
        from intelligence import interaction as IX
        from state import State
        self._inject([(91., 20000.)]*16 + [(-89., 120000.), (-29., 100000.)])
        with mock.patch.object(IX, "_last_voice_bearing", {"bearing_deg": 91., "t1": self.now-1}), \
             mock.patch.object(IX.conv_log, "log_wake"), \
             mock.patch.object(MA, "orient_to_voice") as orient:
            worker = IX._start_wake_orient_reflex("Hey_rex", State.ACTIVE)
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(IX._last_voice_bearing)
        orient.assert_not_called()

    def test_unavailable_energy_register_is_preserved_as_missing(self):
        from tests.test_wake_orient import _FakeFlex

        class MissingEnergy(_FakeFlex):
            def read(self, name):
                if name == "AEC_SPENERGY_VALUES":
                    raise OSError("register unavailable")
                return super().read(name)

        with mock.patch.object(FD, "_dev", MissingEnergy(doa=345, beam_deg=91, energy=0)), \
             mock.patch.object(FD, "_base_moving", return_value=False), \
             mock.patch.object(FD, "_self_speaking", return_value=False):
            self.assertTrue(FD._poll_once())
        self.assertIsNone(FD._samples[-1][4])
        self.assertAlmostEqual(FD._samples[-1][2], -15.)
