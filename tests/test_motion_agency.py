"""
Autonomous base motion (owner spec 2026-07-06): turn to face the tracked person
(neck offset = body misalignment signal) and approach someone at public distance
(`come`, ToF-guarded by the firmware). Decision layer only — these tests pin the
gating, confirmation counters, cooldowns, and turn-direction math.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import config
from intelligence import motion_agency as MA


def _profile(**over):
    base = dict(user_mid_sentence=False, suppress_proactive=False,
                interaction_busy=False)
    base.update(over)
    return SimpleNamespace(**base)


def _snapshot(distance_zone="social", slot="person_1"):
    return {"people": [{"id": slot, "person_db_id": 1,
                        "distance_zone": distance_zone, "face_visible": True}]}


class MotionAgencyTest(unittest.TestCase):
    def setUp(self):
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0)
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
        ]
        self.available, self.state, self.turn, self.come = [p.start() for p in self._patches]
        # Tracked person: locked+visible on slot person_1.
        self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()
        self._neck = 6000  # neutral (SERVO_CHANNELS neck: 1984/9984/6000)

    def tearDown(self):
        self._ws.stop()
        for p in self._patches:
            p.stop()

    def _tick(self, n=1, zone="social", profile=None):
        for _ in range(n):
            MA.step(_snapshot(distance_zone=zone), profile or _profile())

    # ── realign ────────────────────────────────────────────────────────────────

    def test_neck_parked_right_turns_base_right(self):
        # Neck at +40% of half-span (6000 + 0.4*3984 ≈ 7594) for 2 ticks.
        self._neck = 7594
        self._tick(2)
        self.turn.assert_called_once()
        deg = self.turn.call_args[0][0]
        self.assertLess(deg, 0)          # + neck frac (Rex's right) -> CW/negative turn
        self.assertGreaterEqual(abs(deg), 10.0)

    def test_single_tick_does_not_turn(self):
        self._neck = 7594
        self._tick(1)
        self.turn.assert_not_called()

    def test_centered_neck_never_turns(self):
        self._tick(5)
        self.turn.assert_not_called()

    def test_turn_cooldown_blocks_immediate_second_turn(self):
        self._neck = 7594
        self._tick(2)               # fires
        self._tick(2)               # still within cooldown
        self.assertEqual(self.turn.call_count, 1)

    def test_invert_flag_flips_direction(self):
        with mock.patch.object(config, "MOTION_FACE_TURN_INVERT", True, create=True):
            self._neck = 7594
            self._tick(2)
        self.assertGreater(self.turn.call_args[0][0], 0)

    # ── approach ───────────────────────────────────────────────────────────────

    def test_sustained_public_distance_triggers_come(self):
        self._tick(4, zone="public")
        self.come.assert_called_once()

    def test_brief_public_distance_does_not(self):
        self._tick(3, zone="public")
        self._tick(1, zone="social")   # counter resets
        self._tick(3, zone="public")
        self.come.assert_not_called()

    def test_not_facing_them_blocks_approach(self):
        self._neck = 7594  # 40% off-center — realign wins first, approach counter idle
        self._tick(6, zone="public")
        self.come.assert_not_called()

    def test_approach_cooldown(self):
        self._tick(4, zone="public")
        self._tick(4, zone="public")
        self.assertEqual(self.come.call_count, 1)

    # ── gates ──────────────────────────────────────────────────────────────────

    def test_mid_sentence_freezes_everything(self):
        self._neck = 7594
        self._tick(4, profile=_profile(user_mid_sentence=True))
        self._tick(4, zone="public", profile=_profile(user_mid_sentence=True))
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_suppress_proactive_blocks_approach_not_realign(self):
        prof = _profile(suppress_proactive=True)
        self._tick(6, zone="public", profile=prof)
        self.come.assert_not_called()
        self._neck = 7594
        self._tick(2, profile=prof)
        self.turn.assert_called_once()   # realigning to face someone is not speech-like

    def test_moving_base_defers(self):
        self.state.return_value = "moving"
        self._neck = 7594
        self._tick(4, zone="public")
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_no_tracked_person_resets(self):
        self._tracking = {"locked": False, "visible": False}
        self._tick(6, zone="public")
        self.come.assert_not_called()

    def test_master_kill_switch(self):
        with mock.patch.object(config, "AUTONOMOUS_MOTION_ENABLED", False, create=True):
            self._neck = 7594
            self._tick(4, zone="public")
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_disconnected_base_is_silent(self):
        self.available.return_value = False
        self._tick(4, zone="public")
        self.come.assert_not_called()


class TurnMathTest(unittest.TestCase):
    def test_proportional_and_clamped(self):
        self.assertAlmostEqual(MA._turn_degrees_for(0.5), -30.0)
        self.assertAlmostEqual(MA._turn_degrees_for(-0.5), 30.0)
        self.assertAlmostEqual(MA._turn_degrees_for(1.5), -60.0)   # clamped to max
        self.assertAlmostEqual(MA._turn_degrees_for(0.05), -10.0)  # floored to min


if __name__ == "__main__":
    unittest.main()
