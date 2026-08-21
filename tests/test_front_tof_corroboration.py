"""The front cross-check must not be the thing it is checking.

Firmware min-combines the 8x8 matrix into fl/fr (tof.cpp: `out.fl =
tof_front_combine(out.fl, mfl)`, nearest wins), so a matrix phantom simply IS the
published front reading. `_radial_front_m()` was documented and used as the
independent second opinion against exactly that failure — while reading fl/fr.
It was cross-checking the phantom against itself.

Field 2026-08-20 (logs/djr3x-2026-08-20-20-06-34.log): 773 front zone_blocks, 100%
`front`, 87% of them with the base parked, and five flinch retreats — ~1.5 m of
unrequested reverse — off a channel that dipped to 0.07-0.11 m from >0.6 m.

Firmware now publishes fl_radial/fr_radial (the radial pair BEFORE the overlay).
Two distinct questions now have two distinct functions:
  _front_clearance_m  — "is there room to drive forward?"  → conservative fl/fr
  _radial_front_m     — "is that return corroborated?"      → radial only, or None
"""

import unittest
from unittest import mock

import config
from intelligence import motion_agency as ma


def _tele(**tof):
    return {"tof_mm": tof}


class RadialIndependenceTests(unittest.TestCase):
    def setUp(self):
        ma._radial_front_fallback_warned = False

    def test_radial_ignores_the_matrix_phantom(self):
        """The exact field shape: matrix says 80 mm, radial says the room is open."""
        with mock.patch.object(ma.motion, "telemetry",
                               return_value=_tele(fl=80, fr=80,
                                                  fl_radial=4000, fr_radial=4000)):
            self.assertAlmostEqual(ma._radial_front_m(), 4.0, places=3)
            self.assertAlmostEqual(ma._front_clearance_m(), 0.08, places=3)

    def test_clearance_still_takes_the_nearest_return(self):
        """Clearance WANTS the conservative pair — a phantom that turns out real
        must still stop him."""
        with mock.patch.object(ma.motion, "telemetry",
                               return_value=_tele(fl=300, fr=80,
                                                  fl_radial=4000, fr_radial=4000)):
            self.assertAlmostEqual(ma._front_clearance_m(), 0.08, places=3)

    def test_old_firmware_reports_no_independent_reading(self):
        """Silently falling back to fl/fr would restore the false confidence this
        whole change exists to remove."""
        with (
            mock.patch.object(ma.motion, "telemetry", return_value=_tele(fl=80, fr=80)),
            mock.patch.object(ma._log, "warning") as warn,
        ):
            self.assertIsNone(ma._radial_front_m())
            self.assertAlmostEqual(ma._front_clearance_m(), 0.08, places=3)
            warn.assert_called_once()
            # ...and only once, not every tick.
            ma._radial_front_m()
            warn.assert_called_once()

    def test_matrix_only_build_reports_no_independent_reading(self):
        """Firmware publishes -1 for the radial pair when no radial array is wired."""
        with mock.patch.object(ma.motion, "telemetry",
                               return_value=_tele(fl=80, fr=80,
                                                  fl_radial=-1, fr_radial=-1)):
            self.assertIsNone(ma._radial_front_m())


class FlinchCorroborationTests(unittest.TestCase):
    def test_open_radial_vetoes_the_retreat(self):
        with (
            mock.patch.object(config, "MOTION_FLINCH_REQUIRE_CORROBORATION", True),
            mock.patch.object(ma, "_radial_front_m", return_value=4.0),
        ):
            self.assertFalse(ma._flinch_corroborated(),
                             "reversed away from something only the matrix can see")

    def test_near_radial_allows_the_retreat(self):
        with (
            mock.patch.object(config, "MOTION_FLINCH_REQUIRE_CORROBORATION", True),
            mock.patch.object(ma, "_radial_front_m", return_value=0.22),
        ):
            self.assertTrue(ma._flinch_corroborated(),
                            "a real shin registers on both sensors — must still flinch")

    def test_missing_radial_fails_open(self):
        """Old firmware must behave exactly as before, not gain a blind spot."""
        with (
            mock.patch.object(config, "MOTION_FLINCH_REQUIRE_CORROBORATION", True),
            mock.patch.object(ma, "_radial_front_m", return_value=None),
        ):
            self.assertTrue(ma._flinch_corroborated())

    def test_kill_switch_restores_old_behaviour(self):
        with (
            mock.patch.object(config, "MOTION_FLINCH_REQUIRE_CORROBORATION", False),
            mock.patch.object(ma, "_radial_front_m", return_value=4.0),
        ):
            self.assertTrue(ma._flinch_corroborated())

    def test_veto_is_logged_but_throttled(self):
        ma._flinch_state["last_veto_log_at"] = 0.0
        with (
            mock.patch.object(ma, "_radial_front_m", return_value=4.0),
            mock.patch.object(ma._log, "info") as info,
        ):
            ma._log_uncorroborated_flinch(0.08, 1000.0, "approach")
            ma._log_uncorroborated_flinch(0.08, 1001.0, "approach")   # inside cooldown
            self.assertEqual(info.call_count, 1, "a held reflex must be visible, once")
            ma._log_uncorroborated_flinch(0.08, 1100.0, "approach")   # past cooldown
            self.assertEqual(info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
