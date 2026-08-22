"""
tests/test_compass.py — hardware/compass.py math + fusion, no hardware needed.

Covers: tilt-compensation against known vectors (level / pitched / rolled),
angular wraparound in the complementary blend, the |B| magnitude gate, the
alpha-vs-current mapping, calibration round-trip, and a full fusion step
against a mocked telemetry source.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from hardware.compass import (
    Calibration,
    Compass,
    calibration_coverage_ok,
    accel_to_pitch_roll,
    alpha_for_current,
    ang_diff,
    blend_heading,
    compute_calibration,
    field_magnitude_ok,
    tilt_compensated_heading,
)


def _projection_rows(pitch: float, roll: float):
    """The three orthonormal rows implied by the module's projection formulas:
    row1/row2 are exactly the mx_h/my_h equations; row3 is the vertical."""
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    return ((cp, 0.0, sp),
            (sr * sp, cr, -sr * cp),
            (-cr * sp, sr, cr * cp))


def _body_vectors(heading_deg: float, pitch: float, roll: float, vert: float = 0.3):
    """Forward model derived FROM the module's projection matrix A (rows above):
    m_body = A^T · m_earthish, gravity_body = A^T·(0,0,1) = row3. Feeding the
    projection the same (pitch, roll) recovers the heading exactly — the test
    is self-consistent with the spec's equations by construction."""
    ch, sh = math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg))
    earth = (ch, -sh, vert)               # atan2(-my_h, mx_h) convention
    rows = _projection_rows(pitch, roll)
    m_b = [sum(rows[i][j] * earth[i] for i in range(3)) for j in range(3)]
    a_b = list(rows[2])                   # gravity in the body frame
    return m_b, a_b


class TestTiltCompensation(unittest.TestCase):
    def test_level_cardinal_headings(self):
        for h in (0.0, 90.0, 180.0, 270.0, 45.0, 315.0):
            m, _ = _body_vectors(h, 0.0, 0.0)
            got = tilt_compensated_heading(*m, pitch=0.0, roll=0.0)
            self.assertAlmostEqual(ang_diff(got, h), 0.0, places=5,
                                   msg=f"level heading {h} -> {got}")

    def test_pitched(self):
        # Pure pitch: the asin accel formulas are exact -> heading exact.
        for h in (0.0, 60.0, 200.0):
            for p_deg in (-25.0, 15.0, 30.0):
                p = math.radians(p_deg)
                m, a = _body_vectors(h, p, 0.0)
                pitch, roll = accel_to_pitch_roll(*a)
                self.assertAlmostEqual(pitch, p, places=5)
                self.assertAlmostEqual(roll, 0.0, places=5)
                got = tilt_compensated_heading(*m, pitch=pitch, roll=roll)
                self.assertAlmostEqual(ang_diff(got, h), 0.0, places=4,
                                       msg=f"h={h} pitch={p_deg} -> {got}")

    def test_rolled(self):
        # Pure roll: likewise exact.
        for h in (10.0, 135.0, 300.0):
            for r_deg in (-20.0, 25.0):
                r = math.radians(r_deg)
                m, a = _body_vectors(h, 0.0, r)
                pitch, roll = accel_to_pitch_roll(*a)
                self.assertAlmostEqual(roll, r, places=5)
                got = tilt_compensated_heading(*m, pitch=pitch, roll=roll)
                self.assertAlmostEqual(ang_diff(got, h), 0.0, places=4,
                                       msg=f"h={h} roll={r_deg} -> {got}")

    def test_pitched_and_rolled(self):
        # Combined tilt: feeding the projection the true (pitch, roll) recovers
        # the heading EXACTLY (self-consistent with the projection matrix)...
        p, r = math.radians(18.0), math.radians(-12.0)
        for h in (0.0, 77.0, 191.0, 333.0):
            m, _ = _body_vectors(h, p, r)
            got = tilt_compensated_heading(*m, pitch=p, roll=r)
            self.assertAlmostEqual(ang_diff(got, h), 0.0, places=5,
                                   msg=f"h={h} tilted -> {got}")

    def test_accel_helper_approximation_under_combined_tilt(self):
        # ...while the spec's simplified asin pitch/roll are an APPROXIMATION
        # under combined tilt — assert they stay within 1° of truth at moderate
        # angles (production uses the IMU's filtered pitch/roll anyway).
        p, r = math.radians(18.0), math.radians(-12.0)
        _, a = _body_vectors(0.0, p, r)
        pitch, roll = accel_to_pitch_roll(*a)
        self.assertAlmostEqual(math.degrees(pitch), math.degrees(p), delta=1.0)
        self.assertAlmostEqual(math.degrees(roll), math.degrees(r), delta=1.0)

    def test_declination_applied(self):
        m, _ = _body_vectors(0.0, 0.0, 0.0)
        got = tilt_compensated_heading(*m, pitch=0.0, roll=0.0, declination_deg=13.0)
        self.assertAlmostEqual(got, 13.0, places=5)

    def test_near_vertical_guard(self):
        # Pointing straight down: roll undefined — must not raise.
        pitch, roll = accel_to_pitch_roll(-1.0, 0.0, 0.0)
        self.assertAlmostEqual(pitch, math.pi / 2, places=5)
        self.assertEqual(roll, 0.0)
        pitch, roll = accel_to_pitch_roll(0.0, 0.0, 0.0)   # degenerate zero vector
        self.assertEqual((pitch, roll), (0.0, 0.0))


class TestAngularBlend(unittest.TestCase):
    def test_ang_diff_wraps(self):
        self.assertAlmostEqual(ang_diff(1.0, 359.0), 2.0)
        self.assertAlmostEqual(ang_diff(359.0, 1.0), -2.0)
        self.assertAlmostEqual(ang_diff(180.0, 0.0), 180.0)
        self.assertAlmostEqual(ang_diff(0.0, 180.0), 180.0)   # (-180,180] convention

    def test_blend_across_zero(self):
        # 350 pulled 50% toward 10 must pass THROUGH 0 -> exactly 0, not 180.
        self.assertAlmostEqual(blend_heading(350.0, 10.0, 0.5), 0.0, places=6)
        self.assertAlmostEqual(blend_heading(10.0, 350.0, 0.5), 0.0, places=6)

    def test_blend_endpoints(self):
        self.assertAlmostEqual(blend_heading(123.0, 321.0, 0.0), 123.0)
        self.assertAlmostEqual(blend_heading(123.0, 321.0, 1.0), 321.0)

    def test_blend_stays_normalized(self):
        v = blend_heading(359.0, 3.0, 0.9)
        self.assertGreaterEqual(v, 0.0)
        self.assertLess(v, 360.0)


class TestMagnitudeGate(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration(field_norm=1000.0, loaded=True)

    def test_ambient_passes(self):
        self.assertTrue(field_magnitude_ok(1000.0, 0.0, 0.0, self.cal, tolerance=0.25))
        self.assertTrue(field_magnitude_ok(0.0, 800.0, 0.0, self.cal, tolerance=0.25))

    def test_contaminated_rejected(self):
        self.assertFalse(field_magnitude_ok(2000.0, 0.0, 0.0, self.cal, tolerance=0.25))
        self.assertFalse(field_magnitude_ok(100.0, 0.0, 0.0, self.cal, tolerance=0.25))

    def test_uncalibrated_passes_everything(self):
        self.assertTrue(field_magnitude_ok(9e9, 0.0, 0.0, Calibration(), tolerance=0.25))


class TestCalibrationCoverage(unittest.TestCase):
    """A calibration is only as good as the sweep that produced it."""

    def test_a_robot_that_never_moved_is_refused(self):
        # Measured 2026-08-22: a stationary 3 s run produced offsets equal to
        # wherever the sensor was parked and an ambient |B| of 3.0 counts against a
        # true field of ~231 — and the lopsided-axis ratio test PASSED it, because
        # three evenly tiny spans are not lopsided. Installing that is worse than
        # refusing: the compass then reads plausibly and points nowhere.
        ok, note = calibration_coverage_ok(
            Calibration(field_norm=3.0, loaded=True), (6.0, 8.0, 7.0))
        self.assertFalse(ok)
        self.assertIn("3.0", note)
        self.assertIn("rotated", note)

    def test_a_real_sweep_passes_clean(self):
        ok, note = calibration_coverage_ok(
            Calibration(field_norm=231.0, loaded=True), (460.0, 440.0, 420.0))
        self.assertTrue(ok)
        self.assertEqual(note, "")

    def test_a_flat_only_spin_passes_with_a_warning(self):
        # Z barely moved: still usable, but say so.
        ok, note = calibration_coverage_ok(
            Calibration(field_norm=231.0, loaded=True), (460.0, 440.0, 30.0))
        self.assertTrue(ok)
        self.assertIn("lopsided", note)

    def test_the_floor_is_configurable(self):
        from unittest import mock
        with mock.patch.object(config, "COMPASS_CAL_MIN_FIELD_COUNTS", 500.0,
                               create=True):
            ok, _ = calibration_coverage_ok(
                Calibration(field_norm=231.0, loaded=True), (460.0, 440.0, 420.0))
        self.assertFalse(ok)


class TestAlphaVsCurrent(unittest.TestCase):
    def test_mapping(self):
        lo = config.COMPASS_CURRENT_LOW_MA
        hi = config.COMPASS_CURRENT_HIGH_MA
        a_max = config.COMPASS_ALPHA_MAX
        a_min = config.COMPASS_ALPHA_MIN
        self.assertEqual(alpha_for_current(0), a_max)
        self.assertEqual(alpha_for_current(lo), a_max)
        self.assertEqual(alpha_for_current(hi), a_min)
        self.assertEqual(alpha_for_current(hi + 5000), a_min)
        mid = alpha_for_current((lo + hi) / 2)
        self.assertAlmostEqual(mid, (a_max + a_min) / 2, places=6)
        # Sign of the current must not matter (charging is negative).
        self.assertEqual(alpha_for_current(-(hi + 100)), a_min)
        # Unknown current -> distrust.
        self.assertEqual(alpha_for_current(None), a_min)


class TestCalibration(unittest.TestCase):
    def test_offset_and_scale_recovered(self):
        # Synthetic cloud: sphere of radius 500 shifted by (100, -200, 50),
        # squashed x0.5 on z (soft iron).
        samples = []
        for i in range(36):
            for j in range(18):
                th = math.radians(i * 10)
                ph = math.radians(-85 + j * 10)
                x = 500 * math.cos(th) * math.cos(ph) + 100
                y = 500 * math.sin(th) * math.cos(ph) - 200
                z = 500 * 0.5 * math.sin(ph) + 50
                samples.append((x, y, z))
        cal = compute_calibration(samples)
        self.assertAlmostEqual(cal.offset[0], 100.0, delta=5.0)
        self.assertAlmostEqual(cal.offset[1], -200.0, delta=5.0)
        self.assertAlmostEqual(cal.offset[2], 50.0, delta=5.0)
        # z scale must be ~2x the x/y scales to round the squashed axis.
        self.assertAlmostEqual(cal.scale[2] / cal.scale[0], 2.0, delta=0.1)
        self.assertGreater(cal.field_norm, 0.0)

    def test_degenerate_cloud_rejected(self):
        flat = [(math.cos(t) * 100, math.sin(t) * 100, 7.0) for t in
                [i * 0.1 for i in range(100)]]
        with self.assertRaises(ValueError):
            compute_calibration(flat)

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError):
            compute_calibration([(1.0, 2.0, 3.0)] * 5)


class TestFusion(unittest.TestCase):
    """Full Compass.update() steps against a mocked telemetry source."""

    def _make(self, frames, cal=None):
        it = iter(frames)
        last = {"f": None}
        def source():
            try:
                last["f"] = next(it)
            except StopIteration:
                pass
            return last["f"]
        return Compass(telemetry_source=source,
                       calibration=cal or Calibration(field_norm=1000.0, loaded=True))

    @staticmethod
    def _frame(mag_heading_deg, yaw, ma, mag_norm=1000.0):
        # Level robot; body-frame field for the requested magnetic heading,
        # pre-scaled to the calibrated norm. Declination forced to 0 via config
        # monkeypatch in setUp.
        h = math.radians(mag_heading_deg)
        return {
            "mag": {"ok": True, "x": mag_norm * math.cos(h), "y": -mag_norm * math.sin(h), "z": 0.0},
            "imu": {"ok": True, "pitch": 0.0, "roll": 0.0, "yaw": yaw},
            "batt_ma": ma,
        }

    def setUp(self):
        self._decl = config.COMPASS_DECLINATION_DEG
        config.COMPASS_DECLINATION_DEG = 0.0

    def tearDown(self):
        config.COMPASS_DECLINATION_DEG = self._decl

    def test_first_sample_anchors(self):
        c = self._make([self._frame(90.0, yaw=0.0, ma=100)])
        c.update()
        self.assertAlmostEqual(c.get_heading(), 90.0, places=4)
        self.assertAlmostEqual(c.get_fused_yaw(), 90.0, places=4)

    def test_gyro_carries_under_load(self):
        # High current: alpha=ALPHA_MIN=0 -> fused follows gyro deltas only,
        # even though the mag heading is garbage (simulated 40° off).
        frames = [self._frame(90.0, yaw=0.0, ma=100),
                  self._frame(130.0, yaw=10.0, ma=9000),
                  self._frame(130.0, yaw=20.0, ma=9000)]
        c = self._make(frames)
        for _ in frames:
            c.update()
        self.assertAlmostEqual(c.get_fused_yaw(), 110.0, places=3)
        self.assertEqual(c.status()["alpha"], config.COMPASS_ALPHA_MIN)

    def test_mag_reanchors_at_idle(self):
        # Idle current: repeated updates pull the fused yaw toward the mag
        # heading across the 0/360 wrap (fused 350 -> mag 10).
        frames = [self._frame(350.0, yaw=0.0, ma=100)]
        frames += [self._frame(10.0, yaw=0.0, ma=100)] * 200
        c = self._make(frames)
        for _ in frames:
            c.update()
        self.assertAlmostEqual(ang_diff(c.get_fused_yaw(), 10.0), 0.0, delta=1.0)

    def test_contaminated_sample_counted_and_ignored(self):
        frames = [self._frame(90.0, yaw=0.0, ma=100),
                  self._frame(180.0, yaw=0.0, ma=100, mag_norm=4000.0),  # |B| blown
                  ]
        c = self._make(frames)
        for _ in frames:
            c.update()
        self.assertEqual(c.status()["rejected"], 1)
        self.assertAlmostEqual(c.get_fused_yaw(), 90.0, places=3)   # unmoved

    def test_overflow_flag_skips(self):
        c = self._make([{"mag": {"ok": True, "ovl": True, "x": 1, "y": 1, "z": 1},
                         "imu": {"ok": True, "pitch": 0, "roll": 0, "yaw": 0},
                         "batt_ma": 100}])
        c.update()
        self.assertIsNone(c.get_heading())


if __name__ == "__main__":
    unittest.main()
