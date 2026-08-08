"""Spoken charge-cable declarations (field 2026-08-07: the flinch-on-the-cord
rollback). "You're unplugged" / "I plugged you in" relays the operator's word
to the firmware charging latch via chg_assert — the escape hatch for charger
states the battery gauge cannot measure (a finished supply at ~0 mA on a full
pack reads identical to unplugged). The unplug path is contradicted while
charge current visibly flows, and mutes the host voltage backstop so surface
charge can't re-lock the wheels the operator just freed.
"""

import time
import unittest
from unittest import mock

from intelligence import interaction as I
from intelligence import motion_controller as MC


class CharginDeclarationParseTest(unittest.TestCase):
    def test_unplug_declarations(self):
        for text in (
            "You're unplugged.",
            "you are unplugged now",
            "I unplugged you.",
            "I've just unplugged you",
            "The cable's off.",
            "charger is disconnected",
            "You're off the charger.",
        ):
            with self.subTest(text=text):
                self.assertIs(I._charging_declaration(text), False)

    def test_plug_declarations(self):
        for text in (
            "You're plugged in.",
            "I plugged you in.",
            "I've plugged you back in",
            "you're back on the charger",
            "You're charging now.",
        ):
            with self.subTest(text=text):
                self.assertIs(I._charging_declaration(text), True)

    def test_ordinary_speech_is_not_a_declaration(self):
        for text in (
            "Are you charging?",
            "Why are you plugged in?",
            "Move forward two feet.",
            "The charger for my phone died.",
            "No, I did not.",
            "",
        ):
            with self.subTest(text=text):
                self.assertIsNone(I._charging_declaration(text))


class ChargingDeclarationHandlerTest(unittest.TestCase):
    def test_unplug_relays_assert_off(self):
        with mock.patch.object(I.motion_controller, "available", return_value=True), \
             mock.patch.object(I.motion_controller, "status",
                               return_value={"batt_ma": 120}), \
             mock.patch.object(I.motion_controller, "charge_assert") as ca:
            line = I._handle_charging_declaration(False)
        ca.assert_called_once_with(False)
        self.assertIn("wheels", line.lower())

    def test_unplug_contradicted_by_visible_charge_current(self):
        # + = discharging, so -600 mA is definite inflow: the cable is
        # demonstrably attached and the word is refused, not relayed.
        with mock.patch.object(I.motion_controller, "available", return_value=True), \
             mock.patch.object(I.motion_controller, "status",
                               return_value={"batt_ma": -600}), \
             mock.patch.object(I.motion_controller, "charge_assert") as ca:
            line = I._handle_charging_declaration(False)
        ca.assert_not_called()
        self.assertIn("current", line.lower())

    def test_plug_relays_assert_on_without_meter_veto(self):
        with mock.patch.object(I.motion_controller, "available", return_value=True), \
             mock.patch.object(I.motion_controller, "charge_assert") as ca:
            line = I._handle_charging_declaration(True)
        ca.assert_called_once_with(True)
        self.assertIn("locked", line.lower())

    def test_no_base_connected(self):
        with mock.patch.object(I.motion_controller, "available", return_value=False), \
             mock.patch.object(I.motion_controller, "charge_assert") as ca:
            I._handle_charging_declaration(False)
        ca.assert_not_called()


class HostChargingAssertMuteTest(unittest.TestCase):
    """charging()'s voltage backstop yields to a fresh operator unplug assert."""

    def setUp(self):
        self._save = (MC._charging_last_true_at, MC._charge_asserted_off_at)
        MC._charging_last_true_at = 0.0
        MC._charge_asserted_off_at = 0.0

    def tearDown(self):
        MC._charging_last_true_at, MC._charge_asserted_off_at = self._save

    def test_voltage_backstop_locks_without_assert(self):
        with mock.patch.object(MC.motion, "telemetry",
                               return_value={"charging": False, "batt_mv": 13800}):
            self.assertTrue(MC.charging())

    def test_assert_off_mutes_voltage_backstop(self):
        with mock.patch.object(MC.motion, "connected", return_value=True), \
             mock.patch.object(MC.motion, "send", return_value=7):
            MC.charge_assert(False)
        with mock.patch.object(MC.motion, "telemetry",
                               return_value={"charging": False, "batt_mv": 13800}):
            self.assertFalse(MC.charging())

    def test_firmware_charging_rearms_the_backstop(self):
        with mock.patch.object(MC.motion, "connected", return_value=True), \
             mock.patch.object(MC.motion, "send", return_value=7):
            MC.charge_assert(False)
        with mock.patch.object(MC.motion, "telemetry",
                               return_value={"charging": True, "batt_mv": 14200}):
            self.assertTrue(MC.charging())      # fw sees the charger: locked...
        self.assertEqual(MC._charge_asserted_off_at, 0.0)   # ...and assert dropped

    def test_assert_on_locks_immediately(self):
        with mock.patch.object(MC.motion, "connected", return_value=True), \
             mock.patch.object(MC.motion, "send", return_value=7):
            MC.charge_assert(True)
        with mock.patch.object(MC.motion, "telemetry",
                               return_value={"charging": False, "batt_mv": 13200}):
            # fw hasn't ticked yet; the host-side sticky stamp already locks.
            self.assertTrue(MC.charging())

    def test_expired_mute_restores_the_backstop(self):
        MC._charge_asserted_off_at = time.monotonic() - 4000.0   # long past the mute
        with mock.patch.object(MC.motion, "telemetry",
                               return_value={"charging": False, "batt_mv": 13800}):
            self.assertTrue(MC.charging())


if __name__ == "__main__":
    unittest.main()
