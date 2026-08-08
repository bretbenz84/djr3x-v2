"""
GUI chest-LED mirror (owner 2026-08-08: the avatar was missing the three chest
LED pods entirely). hardware/leds_chest.send_command mirrors mode-level state to
the gui bridge BEFORE the enabled/connected checks (avatar shows intent on a
dev Mac with no Arduino), and gui/rex_avatar.chest_render_state maps that state
to render parameters that re-create the firmware pattern per mode.

Avatar construction tests run headless via the Qt 'offscreen' platform and skip
cleanly if PySide6 / a Qt platform isn't available (same pattern as
test_memory_banks_gui).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui.state_bridge import gui_bridge
from hardware import leds_chest

try:
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    from gui.rex_avatar import RexAvatar
    _GUI_OK = True
except Exception:  # pragma: no cover - environment without a usable Qt platform
    _GUI_OK = False

from gui.rex_avatar import chest_render_state


def _chest_snapshot() -> dict:
    return gui_bridge.get_snapshot()["chest_led_state"]


class CommandMirrorTest(unittest.TestCase):
    """leds_chest commands land in the bridge even with the hardware disabled."""

    def setUp(self) -> None:
        self._enabled = mock.patch.object(leds_chest, "CHEST_LEDS_ENABLED", False)
        self._enabled.start()
        self.addCleanup(self._enabled.stop)
        gui_bridge.update_chest_led_state(mode="off", emotion="")
        self.addCleanup(lambda: gui_bridge.update_chest_led_state(mode="off", emotion=""))

    def test_mode_commands_mirror(self) -> None:
        for call, mode in (
            (leds_chest.startup, "startup"),
            (leds_chest.idle, "idle"),
            (leds_chest.active, "active"),
            (leds_chest.sleep, "sleep"),
            (leds_chest.off, "off"),
            (leds_chest.fade_off, "fadeoff"),
        ):
            with self.subTest(mode=mode):
                call()
                self.assertEqual(_chest_snapshot()["mode"], mode)

    def test_speak_carries_emotion(self) -> None:
        leds_chest.speak("Sad")
        state = _chest_snapshot()
        self.assertEqual(state["mode"], "speak")
        self.assertEqual(state["emotion"], "sad")

    def test_charge_carries_soc_and_charging(self) -> None:
        leds_chest.charge_status(55, True)
        state = _chest_snapshot()
        self.assertEqual(state["mode"], "charge")
        self.assertEqual(state["soc"], 55)
        self.assertTrue(state["charging"])

    def test_compliment_flash_keeps_mode(self) -> None:
        leds_chest.idle()
        before = _chest_snapshot()["flash_at"]
        leds_chest.compliment_flash()
        state = _chest_snapshot()
        self.assertEqual(state["mode"], "idle")
        self.assertGreater(state["flash_at"], before)

    def test_next_pattern_changes_nothing(self) -> None:
        leds_chest.idle()
        leds_chest.next_pattern()
        self.assertEqual(_chest_snapshot()["mode"], "idle")


class RenderStateTest(unittest.TestCase):
    """chest_render_state: pure mode → render-parameter mapping."""

    def _state(self, **kw) -> dict:
        base = {"mode": "idle", "emotion": None, "soc": None, "charging": False,
                "flash_at": 0.0, "updated_at": 1000.0}
        base.update(kw)
        return base

    def test_off_is_dark(self) -> None:
        self.assertFalse(chest_render_state(self._state(mode="off"), 1000.0)["on"])

    def test_idle_flickers(self) -> None:
        p = chest_render_state(self._state(mode="idle"), 1000.0)
        self.assertTrue(p["on"])
        self.assertGreater(p["rate"], 0)
        self.assertIsNone(p["fill"])

    def test_idle_uses_firmware_palettes(self) -> None:
        # Ladder = SmallLEDColors (dim red/white/blue, weighted 3:4:2); squares =
        # BlockLEDColors (red/white/gold/blue) — straight from chest_nano.ino.
        p = chest_render_state(self._state(mode="idle"), 1000.0)
        self.assertEqual(len(p["ladder"]), 9)
        self.assertEqual(len(p["squares"]), 4)

    def test_speak_sad_is_blue_and_slow(self) -> None:
        sad = chest_render_state(self._state(mode="speak", emotion="sad"), 1000.0)
        excited = chest_render_state(self._state(mode="speak", emotion="excited"), 1000.0)
        for r, g, b in sad["ladder"]:
            self.assertGreater(b, r)                               # blue-dominant
        for r, g, b in excited["ladder"]:
            self.assertGreater(r, b)                               # red-orange
        self.assertLess(sad["rate"], excited["rate"])

    def test_speak_angry_is_all_red(self) -> None:
        p = chest_render_state(self._state(mode="speak", emotion="angry"), 1000.0)
        for palette in (p["ladder"], p["squares"]):
            for r, g, b in palette:
                self.assertGreater(r, max(g, b))

    def test_charge_fills_ladder_to_soc_with_gauge_gradient(self) -> None:
        p = chest_render_state(self._state(mode="charge", soc=55, charging=True), 1000.0)
        self.assertAlmostEqual(p["fill"], 0.55)
        self.assertTrue(p["gauge"])
        self.assertTrue(p["charging"])
        from gui.rex_avatar import _chest_gauge_color
        empty, full = _chest_gauge_color(0.0), _chest_gauge_color(1.0)
        self.assertGreater(empty[0], empty[2])                     # red at the bottom
        self.assertGreater(full[2], full[0])                       # blue at the top

    def test_fadeoff_decays_to_dark(self) -> None:
        st = self._state(mode="fadeoff", updated_at=1000.0)
        self.assertGreater(chest_render_state(st, 1001.0)["brightness"], 0.5)
        self.assertFalse(chest_render_state(st, 1005.0)["on"])

    def test_flash_window(self) -> None:
        st = self._state(flash_at=1000.0)
        self.assertTrue(chest_render_state(st, 1001.0)["flash"])
        self.assertFalse(chest_render_state(st, 1003.0)["flash"])


@unittest.skipUnless(_GUI_OK, "PySide6 / Qt platform unavailable")
class AvatarIngestionTest(unittest.TestCase):
    def test_snapshot_updates_chest_state_and_paints(self) -> None:
        avatar = RexAvatar()
        avatar.set_snapshot({"chest_led_state": {"mode": "speak", "emotion": "happy",
                                                 "updated_at": 1.0}})
        self.assertEqual(avatar._chest_state["mode"], "speak")
        # Render once offscreen so the new drawing path actually executes.
        from PySide6.QtGui import QImage, QPainter
        image = QImage(430, 400, QImage.Format.Format_ARGB32)
        avatar.resize(430, 400)
        avatar.render(image)


if __name__ == "__main__":
    unittest.main()
