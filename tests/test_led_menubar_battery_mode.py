"""Battery Meter Mode in the LED Control menu bar console.

The console used to hold both Arduino ports for as long as Rex was off, which
silently starved the battery menu bar app of the boards it needs to paint the
chest charge gauge and the mouth glow (owner 2026-08-04: the robot sat dark on
the charger). These cover the ownership toggle that fixes it.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_REPO = Path(__file__).resolve().parent.parent


def _load_console():
    spec = importlib.util.spec_from_file_location(
        "rex_led_menubar_test", _REPO / "tools" / "rex_led_menubar.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatteryModePersistenceTest(unittest.TestCase):
    def setUp(self):
        self.console = _load_console()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.console._MODE_STATE_PATH = Path(self._tmp.name) / "state" / "mode.json"

    def test_defaults_to_battery_mode_with_no_state_file(self):
        # The safe default: an install that has never touched the toggle leaves
        # the charge gauge visible instead of blanking it.
        self.assertTrue(self.console._load_battery_mode())
        self.assertTrue(self.console.battery_mode())

    def test_toggle_round_trips_through_the_state_file(self):
        self.console.set_battery_mode(False)
        self.assertFalse(self.console.battery_mode())
        self.assertEqual(
            json.loads(self.console._MODE_STATE_PATH.read_text()),
            {"battery_mode": False},
        )
        self.assertFalse(self.console._load_battery_mode())

        self.console.set_battery_mode(True)
        self.assertTrue(self.console._load_battery_mode())

    def test_corrupt_state_file_falls_back_to_battery_mode(self):
        self.console._MODE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.console._MODE_STATE_PATH.write_text("{not json")
        self.assertTrue(self.console._load_battery_mode())

    def test_unwritable_state_dir_does_not_crash_the_toggle(self):
        self.console._MODE_STATE_PATH = Path("/proc/nope/mode.json")
        with mock.patch.object(self.console.log, "warning") as warn:
            self.console.set_battery_mode(False)
        self.assertFalse(self.console.battery_mode())   # in-memory switch still took
        warn.assert_called_once()


class ZoneQueueGateTest(unittest.TestCase):
    def setUp(self):
        self.console = _load_console()
        self.zone = self.console._Zone("chest", "ARDUINO_CHEST_PORT",
                                       synth_speak_levels=False)

    def _drain(self):
        with self.zone._lock:
            queued = self.zone._queue[:]
            self.zone._queue.clear()
        return queued

    def test_clicks_are_dropped_only_when_the_board_is_unreachable(self):
        for mode in ("dormant", "no_port"):
            self.zone._set(mode, "")
            self.zone.enqueue("IDLE")
            self.assertEqual(self._drain(), [], mode)

    def test_click_during_boot_or_mode_flip_is_held_not_dropped(self):
        # "battery" is the state the zone is in at the instant a button click
        # flips the mode; "connecting" is the ~2 s Arduino reboot after an open.
        for mode in ("battery", "connecting", "live"):
            self.zone._set(mode, "")
            self.zone.enqueue("IDLE")
            self.assertEqual(self._drain(), ["IDLE"], mode)

    def test_queue_is_bounded_so_a_missing_board_cannot_pile_up(self):
        self.zone._set("connecting", "")
        for i in range(self.console._MAX_QUEUED + 5):
            self.zone.enqueue(f"CMD{i}")
        queued = self._drain()
        self.assertEqual(len(queued), self.console._MAX_QUEUED)
        self.assertEqual(queued[-1], f"CMD{self.console._MAX_QUEUED + 4}")


class StatusLineTest(unittest.TestCase):
    def setUp(self):
        self.console = _load_console()

    def test_battery_mode_row_says_who_owns_the_board(self):
        self.assertEqual(self.console._status_line("battery", "battery meter owns this board"),
                         "🔋 battery meter owns this board")
        self.assertEqual(self.console._status_line("live", "live on /dev/x"),
                         "● live on /dev/x")
        self.assertIn("Rex is running", self.console._status_line("dormant", ""))


if __name__ == "__main__":
    unittest.main()
