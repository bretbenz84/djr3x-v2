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
        MA.cancel_requested_come("test reset")
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0)
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
        ]
        self.available, self.state, self.turn, self.come, self.battery = [
            p.start() for p in self._patches
        ]
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
        MA.cancel_requested_come("test cleanup")
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

    # ── explicit requested come ───────────────────────────────────────────────

    def test_requested_come_scans_until_a_person_is_visible(self):
        self._tracking = {"locked": False, "visible": False}
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.turn.assert_called_once_with(config.MOTION_COME_SEARCH_TURN_DEG)
        self.come.assert_not_called()
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_approaches_visible_centered_person_at_one_meter(self):
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.come.assert_called_once_with(
            0.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )
        self.assertFalse(MA.requested_come_active())

    def test_requested_come_matches_recognized_db_lock(self):
        self._tracking = {"locked": True, "visible": True, "lock_key": "db:1"}
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.come.assert_called_once()
        self.turn.assert_not_called()

    def test_requested_come_aligns_before_approaching(self):
        self._neck = 7594
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.turn.assert_called_once()
        self.come.assert_not_called()

        self._neck = 6000
        self._tick()
        self.come.assert_called_once_with(
            0.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )

    def test_requested_come_stops_after_full_search(self):
        self._tracking = {"locked": False, "visible": False}
        with mock.patch.object(config, "MOTION_COME_SEARCH_MAX_TURNS", 2, create=True), \
             mock.patch.object(config, "MOTION_COME_REACQUIRE_GRACE_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(3)
        self.assertEqual(self.turn.call_count, 2)
        self.assertFalse(MA.requested_come_active())

    def test_requested_come_scan_waits_out_reacquire_grace(self):
        # A chassis turn swings the camera; the person "vanishes" for a beat. Within
        # the grace window the search WAITS instead of stacking more turns (the
        # 2026-07-21 bookshelf spiral).
        self._tracking = {"locked": False, "visible": False}
        self.assertTrue(MA.request_come_here())
        self._tick(1)                              # scan turn 1 (no prior turn -> no grace)
        self.assertEqual(self.turn.call_count, 1)
        self._tick(4)                              # still inside the 3 s grace -> all waits
        self.assertEqual(self.turn.call_count, 1)
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_scan_sweeps_alternating_sides(self):
        # Sweep pattern (sign alternates, magnitude grows): +45, -90, +135 — net
        # offsets +45, -45, +90 around the last-known side, not a one-way spiral.
        self._tracking = {"locked": False, "visible": False}
        with mock.patch.object(config, "MOTION_COME_REACQUIRE_GRACE_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(3)
        rels = [c.args[0] for c in self.turn.call_args_list]
        self.assertEqual(rels, [45.0, -90.0, 135.0])

    def test_requested_come_align_seeds_sweep_side_and_grace(self):
        # Person on the left (+ align turn), then lost: the sweep must start back
        # toward that side, and only after the re-acquire grace.
        self._neck = 4400                          # far left -> positive align turn
        self.assertTrue(MA.request_come_here())
        self._tick(1)                              # align turn issued
        self.assertEqual(self.turn.call_count, 1)
        align_deg = self.turn.call_args[0][0]
        self.assertGreater(align_deg, 0)
        self._tracking = {"locked": False, "visible": False}
        self._tick(2)                              # inside grace -> no scan yet
        self.assertEqual(self.turn.call_count, 1)
        with mock.patch.object(config, "MOTION_COME_REACQUIRE_GRACE_SECS", 0.0, create=True):
            self._tick(1)                          # grace over -> first sweep turn
        self.assertEqual(self.turn.call_count, 2)
        self.assertGreater(self.turn.call_args[0][0], 0)   # starts toward the last-known side


class TurnMathTest(unittest.TestCase):
    def test_proportional_and_clamped(self):
        self.assertAlmostEqual(MA._turn_degrees_for(0.5), -30.0)
        self.assertAlmostEqual(MA._turn_degrees_for(-0.5), 30.0)
        self.assertAlmostEqual(MA._turn_degrees_for(1.5), -60.0)   # clamped to max
        self.assertAlmostEqual(MA._turn_degrees_for(0.05), -10.0)  # floored to min


class MinValidTest(unittest.TestCase):
    def test_mm_to_m_and_invalid_sentinel(self):
        self.assertAlmostEqual(MA._min_valid_m(820, 900), 0.82)   # nearest of two
        self.assertAlmostEqual(MA._min_valid_m(-1, 900), 0.90)    # -1 = error, skipped
        self.assertAlmostEqual(MA._min_valid_m(900, "x"), 0.90)   # junk skipped, valid kept
        self.assertAlmostEqual(MA._min_valid_m(0, 900), 0.0)      # 0 mm is a valid near read
        self.assertIsNone(MA._min_valid_m(-1, -1))                # both dead -> None
        self.assertIsNone(MA._min_valid_m(None, "x"))             # all junk -> None
        self.assertIsNone(MA._min_valid_m())                      # no readings -> None
        self.assertAlmostEqual(MA._min_valid_m(4000, 3500), 3.50)  # clear reads are valid


class FlinchTest(unittest.TestCase):
    """The reflexive front-intrusion back-off (owner request 2026-07-20).

    Approach detection now needs MOTION_FLINCH_CONFIRM_TICKS (default 2) consecutive
    intruding ticks, so the fixtures drive one far tick then two close ticks."""

    def setUp(self):
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0, last_flinch_at=0.0)
        MA._reset_flinch()
        MA._flinch_state["last_corner_log_at"] = 0.0
        self.addCleanup(MA._reset_flinch)
        # tof_mm mutated per test to script an approach; telemetry reads it live.
        self._tof = {"fl": 4000, "fr": 4000, "rl": 4000, "rr": 4000}
        self._state_val = "idle"
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", side_effect=lambda: self._state_val),
            mock.patch.object(MA.motion, "telemetry",
                              side_effect=lambda: {"tof_mm": dict(self._tof)}),
            mock.patch.object(MA.motion_controller, "move", return_value=9),
            # Keep the social behaviors inert (no tracked person) so only flinch acts.
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch("world_state.world_state.get", return_value={}),
        ]
        (self.available, self.state, self.telemetry, self.move,
         self.turn, self.come, self.ws) = [p.start() for p in self._patches]

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _tick(self, front_mm=None, rear_mm=None, fl_mm=None, fr_mm=None,
              rl_mm=None, rr_mm=None, state=None, profile=None):
        if front_mm is not None:
            self._tof["fl"] = self._tof["fr"] = front_mm
        if fl_mm is not None:
            self._tof["fl"] = fl_mm
        if fr_mm is not None:
            self._tof["fr"] = fr_mm
        if rear_mm is not None:
            self._tof["rl"] = self._tof["rr"] = rear_mm
        if rl_mm is not None:
            self._tof["rl"] = rl_mm
        if rr_mm is not None:
            self._tof["rr"] = rr_mm
        if state is not None:
            self._state_val = state
        MA.step({"people": []}, profile or _profile())

    def _approach(self, close_mm=250, rear_mm=None, far_mm=1500, confirm=2, profile=None):
        """One far tick to seat the baseline, then `confirm` close ticks."""
        self._tick(front_mm=far_mm, rear_mm=rear_mm, profile=profile)
        for _ in range(confirm):
            self._tick(front_mm=close_mm, rear_mm=rear_mm, profile=profile)

    # ── firing ────────────────────────────────────────────────────────────────

    def test_front_approach_backs_up(self):
        self._tick(front_mm=1500)      # far — seats the "came from" baseline
        self._tick(front_mm=250)       # intruding tick 1 -> confirm not yet met
        self.move.assert_not_called()
        self._tick(front_mm=250)       # intruding tick 2 -> flinch
        self.move.assert_called_once()
        dist, speed = self.move.call_args[0]
        self.assertLess(dist, 0)                              # reverse
        self.assertAlmostEqual(abs(dist), 0.30, places=2)    # full backup, rear open
        self.assertAlmostEqual(speed, config.MOTION_FLINCH_SPEED_MS, places=3)

    def test_person_at_35_cm_stays_outside_new_trigger_zone(self):
        self._tick(front_mm=1500)
        self._tick(front_mm=350)
        self._tick(front_mm=350)
        self.move.assert_not_called()

    def test_slow_creep_still_flinches(self):
        # ~5 cm/tick creep to contact — the case the old fixed-window version missed.
        for mm in (600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 90, 80):
            self._tick(front_mm=mm)
        self.assertTrue(self.move.called)
        self.assertLess(self.move.call_args[0][0], 0)

    def test_static_clutter_one_side_does_not_mask_other_side_approach(self):
        # fr pinned by a static object at 0.30 m; fl sees a real walk-up.
        self._tick(fl_mm=1500, fr_mm=300)
        self._tick(fl_mm=1500, fr_mm=300)   # baselines settle: fl~1.5, fr frozen 0.30
        self._tick(fl_mm=250, fr_mm=200)    # fl intrudes (1/2)
        self._tick(fl_mm=250, fr_mm=200)    # fl intrudes (2/2) -> flinch
        self.move.assert_called_once()

    # ── false-positive rejection ────────────────────────────────────────────────

    def test_static_close_object_never_flinches(self):
        for _ in range(6):
            self._tick(front_mm=250)   # always close, never CLOSING
        self.move.assert_not_called()

    def test_single_near_glitch_does_not_fire(self):
        self._tick(front_mm=1500)
        self._tick(front_mm=60)        # one spurious near frame (1 tick < confirm)
        self._tick(front_mm=1500)      # cleared again -> confirm counter resets
        self.move.assert_not_called()

    def test_far_flyer_does_not_fake_an_approach(self):
        # Static person at 0.25 m with one dropout-to-max frame: the capped baseline
        # drift means the flyer can't manufacture a large drop.
        for mm in (250, 250, 3500, 250, 250):
            self._tick(front_mm=mm)
        self.move.assert_not_called()

    def test_subfloor_front_read_is_ignored(self):
        self._tick(front_mm=1500)
        self._tick(front_mm=30)        # below MOTION_FLINCH_MIN_VALID_M -> noise
        self.assertEqual(MA._flinch_state["hits"], 0)
        self.move.assert_not_called()

    def test_slow_drift_below_drop_threshold_does_not_flinch(self):
        # Jitters 0.44 <-> 0.40 m near the trigger: under _TRIGGER but the closure
        # off the (adapting) baseline never reaches _APPROACH_DROP_M.
        for mm in (440, 400, 430, 400, 420, 400):
            self._tick(front_mm=mm)
        self.move.assert_not_called()

    # ── rear-limited retreat ─────────────────────────────────────────────────────

    def test_rear_wall_limits_backup_distance(self):
        self._approach(rear_mm=500)                          # only 0.5 m behind
        self.move.assert_called_once()
        self.assertAlmostEqual(abs(self.move.call_args[0][0]), 0.20, places=2)  # 0.5-0.3

    def test_backup_uses_nearer_rear_of_pair(self):
        # rl open, rr close: the nearer rear (rr) must cap the retreat.
        self._tick(front_mm=1500, rl_mm=4000, rr_mm=500)
        self._tick(front_mm=250, rl_mm=4000, rr_mm=500)
        self._tick(front_mm=250, rl_mm=4000, rr_mm=500)
        self.move.assert_called_once()
        self.assertAlmostEqual(abs(self.move.call_args[0][0]), 0.20, places=2)

    def test_cornered_holds_and_does_not_stamp_cooldown(self):
        self._approach(rear_mm=320)                          # wall ~0.32 m behind
        self.move.assert_not_called()
        self.assertEqual(MA._state["last_flinch_at"], 0.0)   # hold must NOT burn cooldown
        # rear clears the next tick -> he should be free to flinch immediately.
        self._tick(front_mm=250, rear_mm=1500)
        self.move.assert_called_once()

    def test_blind_rear_holds_no_token_step(self):
        # Both rear sensors dead: the firmware rear stop also fails open, so hold.
        self._approach(rear_mm=-1)
        self.move.assert_not_called()

    # ── firmware BLOCKED (fast / very close) ─────────────────────────────────────

    def test_blocked_close_read_without_approach_baseline_holds(self):
        self._tick(front_mm=80, state="blocked")
        self.move.assert_not_called()

    def test_blocked_after_observed_approach_backs_off(self):
        self._tick(front_mm=1500, state="idle")
        self._tick(front_mm=80, state="blocked")
        self._tick(front_mm=80, state="blocked")
        self.move.assert_called_once()
        self.assertLess(self.move.call_args[0][0], 0)

    def test_blocked_but_front_clear_does_not_reverse(self):
        # Blocked, but the front is clear (block is rear/side): the front >= trigger
        # guard must short-circuit even with the rear WIDE OPEN — so a reverse is not
        # issued from a non-front block. (Pins the metre-vs-mm fusion fix.)
        self._tick(fl_mm=1500, fr_mm=1500, rl_mm=1500, rr_mm=1500, state="blocked")
        self.move.assert_not_called()

    def test_blocked_on_rear_wall_does_not_reverse_into_it(self):
        # Realistic rear block (wall ~0.09 m behind), front clear -> hold.
        self._tick(fl_mm=1500, fr_mm=1500, rl_mm=90, rr_mm=90, state="blocked")
        self.move.assert_not_called()

    def test_blocked_front_but_cornered_holds(self):
        self._tick(fl_mm=80, fr_mm=80, rl_mm=300, rr_mm=300, state="blocked")
        self.move.assert_not_called()

    # ── gates ────────────────────────────────────────────────────────────────────

    def test_cooldown_blocks_second_flinch(self):
        self._approach()                # fires
        self._approach()                # within cooldown -> blocked
        self.assertEqual(self.move.call_count, 1)

    def test_fires_mid_sentence_by_default(self):
        self._approach(profile=_profile(user_mid_sentence=True))
        self.move.assert_called_once()  # a reflex ignores the speech freeze

    def test_long_sentence_defers_then_fires_without_baseline_decay(self):
        # Finding #1: a long gated stretch must not let the far baseline roll off.
        with mock.patch.object(config, "MOTION_FLINCH_ALLOW_MID_SENTENCE", False, create=True):
            prof = _profile(user_mid_sentence=True)
            self._tick(front_mm=1500, profile=prof)
            for _ in range(8):                     # crowded the whole long sentence
                self._tick(front_mm=250, profile=prof)
            self.move.assert_not_called()          # deferred throughout
        self._tick(front_mm=250)                   # sentence ends -> still fires
        self.move.assert_called_once()

    def test_suppressed_move_fires_when_gate_reopens(self):
        # Finding #1 via the paused/gamepad path: move() suppressed, baseline frozen.
        self.move.return_value = None
        self._tick(front_mm=1500)
        for _ in range(8):
            self._tick(front_mm=250)               # attempted but suppressed each tick
        self.assertTrue(self.move.called)
        self.move.return_value = 9                 # gate reopens
        self._tick(front_mm=250)                   # cooldown never stamped -> fires
        self.assertLess(self.move.call_args[0][0], 0)

    def test_kill_switch_disables_flinch(self):
        with mock.patch.object(config, "MOTION_FLINCH_ENABLED", False, create=True):
            self._approach()
        self.move.assert_not_called()

    def test_moving_base_never_flinches_and_clears_state(self):
        self._tick(front_mm=1500)                # seats baseline while idle
        self._state_val = "moving"
        self._tick(front_mm=350)                 # busy -> no flinch, state dropped
        self.move.assert_not_called()
        self.assertIsNone(MA._flinch_state["baseline"]["fl"])
        self.assertEqual(MA._flinch_state["hits"], 0)
        # Back to idle at close range with NO prior "far": can't fake an approach.
        self._state_val = "idle"
        self._tick(front_mm=350)
        self._tick(front_mm=350)
        self.move.assert_not_called()

    def test_multiframe_far_dropout_does_not_fake_approach(self):
        # Two consecutive dropout-to-max frames must not inflate the baseline enough to
        # fire on a static object (the baseline only rises after a confirmed re-open).
        for mm in (350, 350, 3500, 3500, 350, 350):
            self._tick(front_mm=mm)
        self.move.assert_not_called()

    def test_blocked_subfloor_front_holds_without_valid_approach_evidence(self):
        self._tick(fl_mm=30, fr_mm=25, rl_mm=4000, rr_mm=4000, state="blocked")
        self.move.assert_not_called()

    def test_blocked_unreadable_front_holds_when_rear_clear(self):
        self._tick(fl_mm=-1, fr_mm=-1, rl_mm=4000, rr_mm=4000, state="blocked")
        self.move.assert_not_called()

    def test_master_kill_resets_baseline(self):
        # Someone walks up while autonomy is OFF -> must NOT read as an approach when
        # it comes back on.
        self._tick(front_mm=1500)                # baseline seats ~1.5 while live
        with mock.patch.object(config, "AUTONOMOUS_MOTION_ENABLED", False, create=True):
            self._tick(front_mm=350)             # disabled: person walks up and stands
            self._tick(front_mm=350)
        self.assertIsNone(MA._flinch_state["baseline"]["fl"])  # dropped on disable
        self._tick(front_mm=350)                 # re-enabled: seats fresh at 0.35
        self._tick(front_mm=350)
        self.move.assert_not_called()


if __name__ == "__main__":
    unittest.main()
