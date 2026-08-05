"""Charger edge cues while main.py is down and the battery companion owns serial."""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock


_REPO = Path(__file__).resolve().parent.parent


def _load_meter():
    spec = importlib.util.spec_from_file_location(
        "rex_battery_menubar_test", _REPO / "tools" / "rex_battery_menubar.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChargerSoundTest(unittest.TestCase):
    def setUp(self):
        self.meter = _load_meter()
        self.meter._reset_charger_transition_baseline()

    def _frame(self, charging, mv):
        return json.dumps({
            "type": "telemetry",
            "charging": charging,
            "batt_mv": mv,
            "batt_ma": 0,
            "batt_soc": 100,
            "state": "idle",
        }).encode()

    def test_initial_state_is_silent_then_edges_play_once(self):
        with mock.patch("rex_supervisor._play_charger_effect") as play:
            self.meter._handle_line(self._frame(False, 13400))
            play.assert_not_called()
            self.meter._handle_line(self._frame(True, 14200))
            self.meter._handle_line(self._frame(True, 14200))
            self.meter._handle_line(self._frame(False, 13400))
        self.assertEqual(play.call_args_list, [mock.call(True), mock.call(False)])

    def test_voltage_fallback_detects_full_pack_on_charger(self):
        with mock.patch("rex_supervisor._play_charger_effect") as play:
            self.meter._handle_line(self._frame(False, 13400))
            self.meter._handle_line(self._frame(False, 14200))
        play.assert_called_once_with(True)


class ChestChargeGaugeTest(unittest.TestCase):
    def setUp(self):
        self.meter = _load_meter()
        self.meter._chest_charge_state = None
        self.meter._chest_retry_at = 0.0
        self.meter._chest_sent_at = 0.0

    def test_soc_maps_to_twentyfour_visible_levels(self):
        # Ceiling mapping, matching the Nano's 24-pixel contiguous meter.
        self.assertEqual(self.meter._charge_level(0), 0)
        self.assertEqual(self.meter._charge_level(1), 1)
        self.assertEqual(self.meter._charge_level(50), 12)
        self.assertEqual(self.meter._charge_level(100), 24)
        self.assertIsNone(self.meter._charge_level(None))

    def test_sync_sends_only_on_level_or_charger_change(self):
        with mock.patch.object(self.meter, "_send_chest_command", return_value=True) as send:
            self.meter._sync_chest_charge("live", 50, False)
            self.meter._sync_chest_charge("live", 49, False)  # same visible level (12)
            self.meter._sync_chest_charge("live", 49, True)   # animation changes
            self.meter._sync_chest_charge("live", 55, True)   # next visible level
        self.assertEqual(
            send.call_args_list,
            [
                mock.call("CHARGE:50:0"),
                mock.call("CHARGE:49:1"),
                mock.call("CHARGE:55:1"),
            ],
        )

    def test_stale_baseline_repaints_after_refresh_window(self):
        # A reflashed/rebooted Nano forgets its mode; the periodic refresh
        # repaints even when the visible state never changed.
        with mock.patch.object(self.meter, "_send_chest_command", return_value=True) as send:
            self.meter._sync_chest_charge("live", 50, False)
            self.meter._sync_chest_charge("live", 50, False)  # deduped
            self.meter._chest_sent_at -= self.meter.CHEST_REFRESH_S + 1
            self.meter._sync_chest_charge("live", 50, False)  # refresh fires
        self.assertEqual(send.call_count, 2)

    def test_dormant_state_resets_baseline_without_writing(self):
        self.meter._chest_charge_state = (4, True)
        with mock.patch.object(self.meter, "_send_chest_command") as send:
            self.meter._sync_chest_charge("dormant", 50, True)
        send.assert_not_called()
        self.assertIsNone(self.meter._chest_charge_state)

    def test_failed_send_repaints_as_soon_as_the_port_frees(self):
        # The LED Control console holding the chest port made every open here
        # fail; the meter must not then dedup the retry away once it lets go.
        with mock.patch.object(self.meter, "_send_chest_command", return_value=True):
            self.meter._sync_chest_charge("live", 50, True)
        with mock.patch.object(self.meter, "_send_chest_command", return_value=False) as send:
            self.meter._sync_chest_charge("live", 60, True)
        send.assert_called_once_with("CHARGE:60:1")
        self.assertIsNone(self.meter._chest_charge_state)

        self.meter._chest_retry_at = 0.0     # backoff elapsed; port is free now
        with mock.patch.object(self.meter, "_send_chest_command", return_value=True) as send:
            self.meter._sync_chest_charge("live", 60, True)
        send.assert_called_once_with("CHARGE:60:1")

    def test_open_failure_warns_once_per_outage(self):
        board = self.meter._BOARD_CHEST
        with mock.patch.object(self.meter.log, "warning") as warn:
            for _ in range(3):
                self.meter._warn_board_unreachable(
                    board, "/dev/cu.usbserial-1420",
                    OSError("[Errno 16] Resource busy"))
        self.assertEqual(warn.call_count, 1)
        self.assertIn("LED Control", warn.call_args[0][0] % warn.call_args[0][1:])

        self.meter._clear_board_warning(board)
        with mock.patch.object(self.meter.log, "warning") as warn:
            self.meter._warn_board_unreachable(board, "/dev/cu.usbserial-1420",
                                               OSError("nope"))
        self.assertEqual(warn.call_count, 1)

    def test_unopenable_port_retries_fast_but_a_failed_write_backs_off(self):
        # A failed open reset nothing, and the delay is how long the gauge stays
        # wrong after another app releases the port. A failed WRITE means the
        # open already reset the Nano, so don't strobe it.
        board = self.meter._BOARD_CHEST
        self.meter._warn_board_unreachable(board, "/dev/null", OSError("busy"))
        self.assertEqual(self.meter._retry_delay(board),
                         self.meter._PORT_BUSY_RETRY_SECS)
        self.meter._clear_board_warning(board)      # open succeeded this time
        self.assertEqual(self.meter._retry_delay(board),
                         self.meter._WRITE_FAIL_RETRY_SECS)


class MouthChargeColorTest(unittest.TestCase):
    def setUp(self):
        self.meter = _load_meter()
        self.meter._mouth_charge_state = None
        self.meter._mouth_retry_at = 0.0

    def test_requested_soc_color_bands(self):
        expected = {
            0: 0, 25: 0,
            26: 1, 50: 1,
            51: 2, 75: 2,
            76: 3, 90: 3,
            91: 4, 100: 4,
        }
        for soc, band in expected.items():
            self.assertEqual(self.meter._mouth_soc_band(soc), band, soc)

    def test_failed_send_repaints_as_soon_as_the_port_frees(self):
        with mock.patch.object(self.meter, "_send_mouth_command", return_value=True):
            self.meter._sync_mouth_charge("live", 91, True)
        with mock.patch.object(self.meter, "_send_mouth_command", return_value=False):
            self.meter._sync_mouth_charge("live", 40, True)
        self.assertIsNone(self.meter._mouth_charge_state)

        self.meter._mouth_retry_at = 0.0
        with mock.patch.object(self.meter, "_send_mouth_command", return_value=True) as send:
            self.meter._sync_mouth_charge("live", 40, True)
        send.assert_called_once_with("CHARGE:40")

    def test_charging_sends_color_command_only_when_band_changes(self):
        with mock.patch.object(self.meter, "_send_mouth_command", return_value=True) as send:
            self.meter._sync_mouth_charge("live", 24, True)
            self.meter._sync_mouth_charge("live", 25, True)
            self.meter._sync_mouth_charge("live", 26, True)
            self.meter._sync_mouth_charge("live", 91, True)
            self.meter._sync_mouth_charge("live", 91, False)
        self.assertEqual(
            send.call_args_list,
            [
                mock.call("CHARGE:24"),
                mock.call("CHARGE:26"),
                mock.call("CHARGE:91"),
                mock.call("OFF"),
            ],
        )

    def test_running_controller_never_gets_charge_command(self):
        with mock.patch.object(self.meter, "_send_mouth_command") as send:
            self.meter._sync_mouth_charge("dormant", 100, True)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
