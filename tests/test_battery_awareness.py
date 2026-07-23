"""
Battery awareness (owner spec 2026-07-07, 12.8V 4S LiFePO4 pack via INA226).

LiFePO4's discharge curve is flat, so the module only claims honest bands
(charging/nominal/low/critical), grumbles ONCE per downward crossing per session
and only with an audience, and motion_agency stops volunteering approaches while
critical. Fully dormant when the firmware reports batt_mv=-1 (no sensor wired).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import config
from intelligence import battery_awareness as BA


def _profile(**over):
    base = dict(user_mid_sentence=False, interaction_busy=False)
    base.update(over)
    return SimpleNamespace(**base)


def _reset():
    BA._last_tier = None
    BA._announced_tiers = set()
    BA._pending_announce = None
    BA._last_read_mv = -1
    BA._last_spoke_at = 0.0
    BA._last_charging = None
    BA._pending_charging_line = False


class TierMappingTest(unittest.TestCase):
    def test_lifepo4_bands(self):
        self.assertEqual(BA.tier_for_mv(13600), "charging")   # on the charger
        self.assertEqual(BA.tier_for_mv(13100), "nominal")    # the long plateau
        self.assertEqual(BA.tier_for_mv(12700), "low")        # the knee (~20%)
        self.assertEqual(BA.tier_for_mv(12100), "critical")   # near BMS cutoff

    def test_unknown_voltage_is_none(self):
        self.assertIsNone(BA.tier_for_mv(-1))
        self.assertIsNone(BA.tier_for_mv(0))

    def test_hysteresis_blocks_sag_flapping(self):
        # Sitting at 12.98V after being LOW: 30mV above the nominal floor is
        # inside the 100mV hysteresis — stays low instead of flapping.
        self.assertEqual(BA.tier_for_mv(12980, previous="low"), "low")
        # A real recovery (charger) clears the band decisively.
        self.assertEqual(BA.tier_for_mv(13200, previous="low"), "nominal")


class StepBehaviorTest(unittest.TestCase):
    def setUp(self):
        _reset()

    def tearDown(self):
        _reset()

    def _step(self, mv, *, people=None, speak_ok=True):
        snapshot = {"people": people if people is not None
                    else [{"face_visible": True}]}
        with (
            mock.patch.object(BA, "current_mv", return_value=mv),
            mock.patch("intelligence.speech_engine.speak_async",
                       return_value=speak_ok) as speak,
        ):
            BA.step(snapshot, _profile())
        return speak

    def test_first_reading_baselines_silently(self):
        speak = self._step(12700)   # boots already low: baseline, no remark
        speak.assert_not_called()

    def test_downward_crossing_grumbles_once(self):
        self._step(13100)                   # baseline nominal
        speak = self._step(12700)           # nominal -> low
        speak.assert_called_once()
        self.assertEqual(speak.call_args.kwargs.get("purpose"), "battery_status")
        speak2 = self._step(12690)          # still low: no repeat
        speak2.assert_not_called()

    def test_crossing_with_empty_room_waits_for_audience(self):
        self._step(13100)
        speak = self._step(12700, people=[])     # crossing, nobody there
        speak.assert_not_called()
        speak2 = self._step(12700)               # someone shows up -> latched grumble
        speak2.assert_called_once()

    def test_no_sensor_is_fully_dormant(self):
        speak = self._step(-1)
        speak.assert_not_called()
        self.assertIsNone(BA._last_tier)

    def test_critical_flag_gates_motion(self):
        with mock.patch.object(BA, "current_mv", return_value=12100):
            self.assertTrue(BA.battery_critical())
        with mock.patch.object(BA, "current_mv", return_value=13100):
            self.assertFalse(BA.battery_critical())
        with mock.patch.object(BA, "current_mv", return_value=-1):
            self.assertFalse(BA.battery_critical())   # unknown = no opinion

    def test_charger_edges_play_matching_effects_once(self):
        telemetry = {"charging": False, "batt_mv": 13400}
        with (
            mock.patch("hardware.motion.telemetry", side_effect=lambda: dict(telemetry)),
            mock.patch("audio.sound_effects.play") as play,
            mock.patch("intelligence.speech_engine.speak_async", return_value=True),
            mock.patch.object(config, "MOTION_CHARGER_NOTICE_DEBOUNCE_SECS", 0.0, create=True),
        ):
            BA._step_charging({}, _profile())  # startup baseline is silent
            play.assert_not_called()
            telemetry.update(charging=True, batt_mv=14200)
            BA._step_charging({}, _profile())
            play.assert_called_once_with("charger_connected", force=True)
            BA._step_charging({}, _profile())  # stable state does not repeat
            self.assertEqual(play.call_count, 1)
            telemetry.update(charging=False, batt_mv=13400)
            BA._step_charging({}, _profile())
        self.assertEqual(
            play.call_args_list[-1],
            mock.call("charger_disconnected", force=True),
        )

    def test_charger_voltage_fallback_plays_connected_effect(self):
        readings = [
            {"charging": False, "batt_mv": 13400},
            {"charging": False, "batt_mv": 14200},
        ]
        with (
            mock.patch("hardware.motion.telemetry", side_effect=readings),
            mock.patch("audio.sound_effects.play") as play,
            mock.patch("intelligence.speech_engine.speak_async", return_value=True),
            mock.patch.object(config, "MOTION_CHARGER_NOTICE_DEBOUNCE_SECS", 0.0, create=True),
        ):
            BA._step_charging({}, _profile())
            BA._step_charging({}, _profile())
        play.assert_called_once_with("charger_connected", force=True)

    def test_charger_notice_debounced_against_a_flap(self):
        # A brief voltage-sag flap (unplug then re-plug within the debounce) must NOT
        # announce anything — the transition never persisted.
        telemetry = {"charging": True, "batt_mv": 14200}
        BA._last_charging = None
        BA._chg_candidate = None
        with (
            mock.patch("hardware.motion.telemetry", side_effect=lambda: dict(telemetry)),
            mock.patch("audio.sound_effects.play") as play,
            mock.patch("intelligence.speech_engine.speak_async", return_value=True),
            mock.patch.object(config, "MOTION_CHARGER_NOTICE_DEBOUNCE_SECS", 12.0, create=True),
        ):
            BA._step_charging({}, _profile())          # baseline: charging
            telemetry.update(charging=False, batt_mv=13750)   # sag flaps it "off"
            BA._step_charging({}, _profile())          # candidate armed, not stable
            telemetry.update(charging=True, batt_mv=14200)    # recovers within debounce
            BA._step_charging({}, _profile())
            play.assert_not_called()                   # never announced the flap


class MotionGateTest(unittest.TestCase):
    def test_agency_approach_declines_when_critical(self):
        from intelligence import motion_agency as MA
        MA._state.update(neck_hits=0, far_hits=3, last_turn_at=0.0,
                         last_approach_at=0.0)
        with (
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "come") as come,
            mock.patch.object(MA, "_tracked_person",
                              return_value={"id": "person_1",
                                            "distance_zone": "public"}),
            mock.patch.object(MA, "neck_offset_fraction", return_value=0.0),
            mock.patch("intelligence.battery_awareness.battery_critical",
                       return_value=True),
        ):
            MA.step({"people": []}, _profile(suppress_proactive=False))
        come.assert_not_called()
        self.assertEqual(MA._state["far_hits"], 0)   # counter reset, not deferred


if __name__ == "__main__":
    unittest.main()
