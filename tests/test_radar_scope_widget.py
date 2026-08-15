"""
RadarRingWidget (Motivator Control radar scope) tests: the polar mapping obeys
the project bearing convention (front = up, + = left/CCW, docs/motion_protocol.md
§4), wire targets normalize/skip exactly like hardware/radar.py, staleness and
the latch display gate correctly, and a target actually changes the rendered
scope. Runs headless via the Qt 'offscreen' platform; skips cleanly if PySide6 /
a Qt platform isn't available.
"""

from __future__ import annotations

import os
import re
import time
import unittest
from pathlib import Path

# Force headless BEFORE any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from gui.dashboard import RadarRingWidget
    _GUI_OK = True
except Exception:  # pragma: no cover - environment without a usable Qt platform
    _GUI_OK = False


def _tel(targets=None, *, ok=True, up=3, sens=None, errs=0, age=0.0):
    """A radar telemetry frame as hardware.radar.telemetry() returns it."""
    return {
        "v": 1,
        "type": "telemetry",
        "radar": {"ok": ok, "up": up, "targets": list(targets or [])},
        "sens": sens if sens is not None else [
            {"ok": True, "frames": 100, "bad": 0, "drop": 0} for _ in range(3)
        ],
        "errs": errs,
        "rx_monotonic": time.monotonic() - age,
    }


_WIRE_TARGET = {"b": 137.2, "r": 4.10, "c": 0.82, "s": -0.30, "m": 6}


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class PolarMathTest(unittest.TestCase):
    """Front = up, + bearing = left/CCW, range clamped to the 8 m full scale."""

    def _pt(self, bearing, rng):
        return RadarRingWidget._polar_px(100.0, 100.0, 10.0, 80.0, 8.0, bearing, rng)

    def test_bearing_zero_is_straight_up(self):
        p = self._pt(0.0, 8.0)
        self.assertAlmostEqual(p.x(), 100.0, places=5)
        self.assertAlmostEqual(p.y(), 10.0, places=5)     # body_r + reach above center

    def test_positive_bearing_is_screen_left(self):
        p = self._pt(90.0, 4.0)                            # half range -> r = 10 + 40
        self.assertAlmostEqual(p.x(), 50.0, places=5)
        self.assertAlmostEqual(p.y(), 100.0, places=5)

    def test_negative_bearing_is_screen_right(self):
        p = self._pt(-60.0, 2.0)                           # front-right mount direction
        self.assertGreater(p.x(), 100.0)
        self.assertLess(p.y(), 100.0)                      # forward quarter -> above center

    def test_rear_mount_bearing_is_straight_down(self):
        # The rear module's boresight is +180 (the wrapped value) — it must plot
        # straight below the body, and -180 must land on the same pixel.
        p = self._pt(180.0, 4.0)                           # half range -> r = 10 + 40
        self.assertAlmostEqual(p.x(), 100.0, places=5)
        self.assertAlmostEqual(p.y(), 150.0, places=5)
        q = self._pt(-180.0, 4.0)
        self.assertAlmostEqual(q.x(), p.x(), places=5)
        self.assertAlmostEqual(q.y(), p.y(), places=5)

    def test_range_clamps_at_full_scale(self):
        far = self._pt(0.0, 20.0)
        edge = self._pt(0.0, 8.0)
        self.assertAlmostEqual(far.y(), edge.y(), places=5)


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class SetStateTest(unittest.TestCase):
    def setUp(self):
        self.w = RadarRingWidget()

    def test_wire_targets_normalize_and_malformed_are_skipped(self):
        self.w.set_state(_tel([_WIRE_TARGET, {"b": "bogus"}]), [], None, True)
        self.assertTrue(self.w._fresh)
        self.assertEqual(len(self.w._targets), 1)
        t = self.w._targets[0]
        self.assertAlmostEqual(t["bearing_deg"], 137.2)
        self.assertAlmostEqual(t["range_m"], 4.10)
        self.assertAlmostEqual(t["confidence"], 0.82)
        self.assertAlmostEqual(t["speed_mps"], -0.30)
        self.assertEqual(t["sensors"], 6)
        self.assertEqual(self.w._latched, [])              # live frame wins
        self.assertEqual(len(self.w._sens), 3)

    def test_stale_telemetry_shows_nothing(self):
        self.w.set_state(_tel([_WIRE_TARGET], age=5.0), [], None, True)
        self.assertFalse(self.w._fresh)
        self.assertEqual(self.w._targets, [])
        self.assertFalse(self.w._ok)

    def test_latch_display_only_when_live_frame_is_empty(self):
        latched = [{"bearing_deg": 10.0, "range_m": 2.0, "confidence": 0.7,
                    "speed_mps": 0.0, "sensors": 1}]
        self.w.set_state(_tel([]), latched, None, True)
        self.assertEqual(len(self.w._latched), 1)
        # A live frame suppresses the latch display again.
        self.w.set_state(_tel([_WIRE_TARGET]), latched, None, True)
        self.assertEqual(self.w._latched, [])

    def test_mounts_come_from_hello_with_pins_fallback(self):
        self.assertEqual(self.w._mounts, RadarRingWidget._FALLBACK_MOUNTS)
        hello = {"sensors": [{"mount": 0, "cfg": True}, {"mount": 90, "cfg": False}]}
        self.w.set_state(_tel([]), [], hello, True)
        self.assertEqual(self.w._mounts, (0.0, 90.0))
        self.assertEqual(self.w._cfg, (True, False))
        self.w.set_state(_tel([]), [], None, True)
        self.assertEqual(self.w._mounts, RadarRingWidget._FALLBACK_MOUNTS)

    def test_fallback_mounts_mirror_the_firmware_pin_table(self):
        # The fallback is what the scope draws until the board's hello arrives
        # (and in --demo). It must be the SAME ring pins.h describes, or the
        # wedges lie about where the modules point until the link is up.
        pins_h = (Path(__file__).resolve().parents[1]
                  / "firmware" / "djr3x_radar" / "pins.h").read_text()
        table = pins_h.split("RADAR_SENSORS[RADAR_SENSOR_COUNT] = {", 1)[1].split("};", 1)[0]
        mounts = tuple(
            float(m.group(1))
            for m in re.finditer(r"^\s*\{\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(-?[\d.]+)f?\s*\}",
                                 table, re.MULTILINE)
        )
        self.assertEqual(len(mounts), 3, table)
        self.assertEqual(mounts, RadarRingWidget._FALLBACK_MOUNTS)
        # And it IS the two-forward / one-rear ring: pair at ±60°, lone at 180°.
        self.assertEqual(sorted(mounts), [-60.0, 60.0, 180.0])

    def test_disconnected_and_clear_reset(self):
        self.w.set_state(_tel([_WIRE_TARGET]), [], None, True)
        self.w.set_state(None, [], None, False)
        self.assertFalse(self.w._connected)
        self.assertEqual(self.w._targets, [])
        self.w.set_state(_tel([_WIRE_TARGET]), [], None, True)
        self.w.clear()
        self.assertFalse(self.w._connected)
        self.assertEqual(self.w._targets, [])
        self.assertEqual(self.w._sens, [])

    def test_conf_color_bands(self):
        self.assertEqual(RadarRingWidget._conf_color(0.9).green(), 200)   # confident
        self.assertEqual(RadarRingWidget._conf_color(0.4).red(), 240)     # marginal
        self.assertEqual(RadarRingWidget._conf_color(0.1).red(), 120)     # barely there


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt offscreen platform unavailable")
class RenderSmokeTest(unittest.TestCase):
    """A target must change the painted scope (dot + label), not just the chips —
    compare the central band with the chip row cropped off."""

    def _scope_band(self, widget):
        img = widget.grab().toImage()
        return img.copy(QRect(0, 30, img.width(), img.height() - 100))

    def test_live_target_renders_ink(self):
        w = RadarRingWidget()
        w.resize(400, 460)
        w.set_state(_tel([]), [], None, True)
        empty = self._scope_band(w)
        w.set_state(_tel([{"b": 0.0, "r": 4.0, "c": 0.9, "s": 0.4, "m": 1}]), [], None, True)
        self.assertNotEqual(self._scope_band(w), empty)

    def test_latched_target_renders_hollow_ink(self):
        w = RadarRingWidget()
        w.resize(400, 460)
        w.set_state(_tel([]), [], None, True)
        empty = self._scope_band(w)
        w.set_state(_tel([]), [{"bearing_deg": -45.0, "range_m": 3.0, "confidence": 0.6,
                               "speed_mps": 0.0, "sensors": 2}], None, True)
        self.assertNotEqual(self._scope_band(w), empty)

    def test_no_link_renders_without_error(self):
        w = RadarRingWidget()
        w.resize(320, 360)
        w.clear()
        self.assertFalse(w.grab().toImage().isNull())
