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


def _snapshot(distance_zone="social", slot="person_1", visible=True, face_box=None,
              db_id=1):
    """`visible=False` means NOBODY is on camera. The come-here search keys off
    world_state.people (a head lock is head behavior, not visibility), so clearing
    only face_tracking no longer simulates an empty room."""
    if not visible:
        return {"people": []}
    person = {"id": slot, "person_db_id": db_id,
              "distance_zone": distance_zone, "face_visible": True}
    if face_box is not None:
        person["face_box"] = face_box
    return {"people": [person]}


# Realign arming fixtures (owner spec 2026-07-31: neck first, wheels last).
# Neck 9600 on the 1984/9984/6000 channel -> +0.90 of half-span (sweep exhausted);
# the face box centre at x=1760 on the default 1920-wide frame -> +0.83 of the
# half-width (extreme right of frame), same side as the neck.
_EXHAUSTED_NECK_RIGHT = 9600
_EDGE_FACE_RIGHT = (1650, 400, 220, 220)
_EDGE_FACE_LEFT = (50, 400, 220, 220)
_CENTERED_FACE = (860, 400, 200, 200)


class MotionAgencyTest(unittest.TestCase):
    def setUp(self):
        MA.cancel_requested_come("test reset")
        # user_motion_at MUST be reset: motion_sequence.start() calls
        # motion_agency.note_user_motion(), so any earlier test that ran a route
        # (tests/test_motion.py) leaves the realign stand-down armed and every
        # realign assertion below silently fails.
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0, user_motion_at=0.0,
                         realign_pending_seq=None, traction_fails=0,
                         no_traction_until=0.0, hold_at=None)
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
            # Come-search turns resolve instantly (the firmware `done` is what the
            # dwell/settle windows key off); the align settle is zeroed so the
            # single-tick fixtures keep their cadence — the dwell/settle behaviors
            # get their own explicit tests.
            mock.patch.object(MA.motion, "done_result", return_value="completed",
                              create=True),
            mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.0,
                              create=True),
        ]
        (self.available, self.state, self.turn, self.come, self.battery,
         self.done_result, _) = [p.start() for p in self._patches]
        # Tracked person: locked+visible on slot person_1.
        self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
        self._visible = True          # world_state.people also shows them
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()
        self._neck = 6000  # neutral (SERVO_CHANNELS neck: 1984/9984/6000)
        self._face_box = None

    def _arm_realign(self):
        """Meet BOTH realign conditions: neck sweep exhausted to Rex's right AND
        the face at the extreme right edge of frame."""
        self._neck = _EXHAUSTED_NECK_RIGHT
        self._face_box = _EDGE_FACE_RIGHT

    def tearDown(self):
        MA.cancel_requested_come("test cleanup")
        self._ws.stop()
        for p in self._patches:
            p.stop()

    def _tick(self, n=1, zone="social", profile=None):
        for _ in range(n):
            MA.step(_snapshot(distance_zone=zone, visible=self._visible,
                              face_box=self._face_box),
                    profile or _profile())

    def _verdicts(self, *results):
        """Feed firmware `done` results back for the realign turn seq (always 7)."""
        seq = iter(results)
        return mock.patch.object(MA.motion, "done_result",
                                 side_effect=lambda _s: next(seq, None), create=True)

    # ── realign ────────────────────────────────────────────────────────────────

    def test_exhausted_neck_and_edge_face_turn_base_right(self):
        # Neck at +90% of half-span (sweep exhausted) AND face at the extreme right
        # edge of frame, held for 2 ticks -> the base finally turns.
        self._arm_realign()
        self._tick(2)
        self.turn.assert_called_once()
        deg = self.turn.call_args[0][0]
        self.assertLess(deg, 0)          # + neck frac (Rex's right) -> CW/negative turn
        self.assertGreaterEqual(abs(deg), 10.0)

    def test_moderate_neck_offset_never_turns_the_base(self):
        # The old trigger (neck ≥ 30% off-neutral) fired constantly. The neck servo
        # is the primary tracker now: 40% off-neutral is just the neck doing its job,
        # even with the face at the frame edge — no wheels.
        self._neck = 7594                       # +40% of half-span
        self._face_box = _EDGE_FACE_RIGHT
        self._tick(5)
        self.turn.assert_not_called()

    def test_exhausted_neck_with_centered_face_does_not_turn(self):
        # Neck at its limit but face-tracking is still holding the face in the middle
        # of frame: the camera has them, so the wheels stay put.
        self._neck = _EXHAUSTED_NECK_RIGHT
        self._face_box = _CENTERED_FACE
        self._tick(5)
        self.turn.assert_not_called()

    def test_exhausted_neck_without_a_face_box_does_not_turn(self):
        # No face-box measurement -> no proof the face is at the frame edge -> hold.
        self._neck = _EXHAUSTED_NECK_RIGHT
        self._face_box = None
        self._tick(5)
        self.turn.assert_not_called()

    def test_neck_pinned_with_face_moderately_off_centre_turns(self):
        # THE FIELD FAILURE (2026-07-31 20:14): neck pinned at its hard minimum
        # (1984, frac -1.0) with the face 38% left of centre (x=594 on 1920) — the
        # neck could not reach further, but the old 0.70 "extreme edge" bar meant
        # the base never turned. Any sustained same-side offset past tracking
        # jitter must fire once the sweep is exhausted.
        self._neck = 1984
        self._face_box = (494, 400, 200, 200)   # centre x=594 -> frac -0.38
        self._tick(2)
        self.turn.assert_called_once()
        self.assertGreater(self.turn.call_args[0][0], 0)   # person left -> CCW/left turn

    def test_face_escaping_the_opposite_edge_does_not_turn(self):
        # Neck parked right but the face at the LEFT edge: the neck can still sweep
        # toward them — a base turn keyed off the neck would rotate the wrong way.
        self._neck = _EXHAUSTED_NECK_RIGHT
        self._face_box = _EDGE_FACE_LEFT
        self._tick(5)
        self.turn.assert_not_called()

    # ── no traction (carpet) ───────────────────────────────────────────────────
    # The firmware aborts a turn that makes no physical yaw progress; two in a row
    # means the wheels are scrubbing, so the social behaviors stand down.

    def test_two_aborted_turns_stand_down_autonomous_motion(self):
        self._arm_realign()
        with self._verdicts("aborted", "aborted"), \
             mock.patch.object(MA, "_emit_traction_notice"):
            self._tick(2)                                  # turn 1
            MA._state["last_turn_at"] = 0.0                # skip the cooldown
            self._tick(2)                                  # verdict 1 + turn 2
            MA._state["last_turn_at"] = 0.0
            self._tick(4)                                  # verdict 2 -> latched
            before = self.turn.call_count
            MA._state["last_turn_at"] = 0.0
            self._tick(6)
        self.assertEqual(self.turn.call_count, before, "kept grinding after standing down")
        self.assertGreater(MA._state["no_traction_until"], 0.0)

    def test_one_aborted_turn_does_not_latch(self):
        # A comms loss aborts finite commands with the same code — one is not enough.
        self._arm_realign()
        with self._verdicts("aborted", "completed"):
            self._tick(2)
            MA._state["last_turn_at"] = 0.0
            self._tick(2)
            MA._state["last_turn_at"] = 0.0
            self._tick(2)
        self.assertEqual(MA._state["no_traction_until"], 0.0)
        self.assertGreaterEqual(self.turn.call_count, 2)

    def test_completed_turn_clears_the_fail_streak(self):
        self._arm_realign()
        MA._state["traction_fails"] = 1
        with self._verdicts("completed"):
            self._tick(2)
            MA._state["last_turn_at"] = 0.0
            self._tick(2)
        self.assertEqual(MA._state["traction_fails"], 0)

    def test_traction_notice_is_spoken_once(self):
        self._arm_realign()
        with self._verdicts("aborted", "aborted"), \
             mock.patch.object(MA, "_emit_traction_notice") as notice:
            self._tick(2)
            MA._state["last_turn_at"] = 0.0
            self._tick(2)
            MA._state["last_turn_at"] = 0.0
            self._tick(6)
        self.assertEqual(notice.call_count, 1)

    def test_stop_does_not_clear_the_traction_streak(self):
        # Field 2026-07-25 (14:30 run): a realign turn aborted (streak 1), the owner
        # said "Don't move", and note_user_motion() wiped the streak — so the carpet
        # detector never reached its threshold and he tried again 49 s later. Being
        # told to stop is the opposite of evidence that the wheels found grip.
        MA._state["traction_fails"] = 1
        MA.note_user_hold("user said stop")
        self.assertEqual(MA._state["traction_fails"], 1)

    # ── "don't move" is a standing instruction ─────────────────────────────────

    def test_hold_outlasts_the_steering_standdown(self):
        # 49 s after "Stopping." he was realigning again: a stop only armed the 45 s
        # steering window. The hold latches instead.
        self._arm_realign()
        MA.note_user_hold()
        with mock.patch.object(config, "MOTION_USER_MOTION_STANDDOWN_SECS", 45.0, create=True):
            MA._state["user_motion_at"] = time.monotonic() - 3600.0   # long expired
            self._tick(6)
        self.turn.assert_not_called()

    def test_hold_expires_when_an_expiry_is_configured(self):
        self._arm_realign()
        MA.note_user_hold()
        MA._state["hold_at"] = time.monotonic() - 120.0
        with mock.patch.object(config, "MOTION_STOP_STANDDOWN_SECS", 60.0, create=True):
            self._tick(2)
        self.turn.assert_called_once()

    def test_a_later_move_command_releases_the_hold(self):
        self._arm_realign()
        MA.note_user_hold()
        MA.note_user_motion()
        MA._state["user_motion_at"] = 0.0        # only the hold is under test here
        self._tick(2)
        self.turn.assert_called_once()

    def test_come_here_releases_the_hold(self):
        MA.note_user_hold()
        MA._state["no_traction_until"] = time.monotonic() + 300.0
        MA.request_come_here()
        self.assertIsNone(MA._state["hold_at"])
        self.assertEqual(MA._state["no_traction_until"], 0.0)

    # ── per-room "don't drive here" ────────────────────────────────────────────

    def _room_rule(self, name="workshop", reason="carpet", on=True):
        belief = {"name": name, "place_id": 1, "score": 0.9, "since_ts": 0.0,
                  "no_drive": on, "no_drive_reason": reason}
        return mock.patch("perception.place_service.current_place",
                          return_value=belief, create=True)

    def test_no_drive_room_blocks_realign(self):
        self._arm_realign()
        with self._room_rule():
            self._tick(6)
        self.turn.assert_not_called()

    def test_no_drive_room_reports_the_room_and_reason(self):
        with self._room_rule():
            self.assertEqual(MA.no_drive_room(), ("workshop", "carpet"))

    def test_a_room_without_the_rule_still_realigns(self):
        self._arm_realign()
        with self._room_rule(on=False):
            self._tick(2)
        self.turn.assert_called_once()

    def test_come_here_is_refused_in_a_no_drive_room(self):
        # The one case come-here does NOT override: "come here" is nearly always said
        # from inside the very room the owner flagged.
        with self._room_rule():
            self.assertFalse(MA.request_come_here())
        self.assertFalse(MA.requested_come_active())

    def test_the_room_rule_can_be_disabled_wholesale(self):
        self._arm_realign()
        with mock.patch.object(config, "MOTION_ROOM_NO_DRIVE_ENABLED", False, create=True), \
                self._room_rule():
            self._tick(2)
        self.turn.assert_called_once()

    def test_voice_command_clears_the_traction_latch(self):
        # Explicit commands are never gated: the owner may have carried him to tile.
        MA._state["no_traction_until"] = time.monotonic() + 300.0
        MA._state["traction_fails"] = 3
        MA.note_user_motion()
        self.assertEqual(MA._state["no_traction_until"], 0.0)
        self.assertEqual(MA._state["traction_fails"], 0)

    def test_single_tick_does_not_turn(self):
        self._arm_realign()
        self._tick(1)
        self.turn.assert_not_called()

    def test_centered_neck_never_turns(self):
        self._tick(5)
        self.turn.assert_not_called()

    def test_turn_cooldown_blocks_immediate_second_turn(self):
        self._arm_realign()
        self._tick(2)               # fires
        self._tick(2)               # still within cooldown
        self.assertEqual(self.turn.call_count, 1)

    def test_invert_flag_flips_direction(self):
        with mock.patch.object(config, "MOTION_FACE_TURN_INVERT", True, create=True):
            self._arm_realign()
            self._tick(2)
        self.assertGreater(self.turn.call_args[0][0], 0)

    # ── approach ───────────────────────────────────────────────────────────────

    def test_sustained_public_distance_triggers_come(self):
        self._tick(4, zone="public")
        self.come.assert_called_once()

    def test_spontaneous_approach_stops_a_respectful_meter_out(self):
        # An uninvited drive must not use the explicit come-here's closer stop.
        self._tick(4, zone="public")
        self.come.assert_called_once_with(0.0, stop_at=config.MOTION_APPROACH_STOP_AT_M)

    def test_close_front_tof_vetoes_a_public_zone_approach(self):
        # THE FIELD FAILURE (2026-07-31): on the wide-angle lens a face 3-4 ft away
        # reads under the 30% "public" width fraction, so face size said "far" about
        # someone within arm's reach and Rex drove at them. The front ToF saw the
        # truth the whole time — anything nearer than MOTION_APPROACH_MIN_START_M
        # means they are NOT far, and the approach must not even arm.
        with mock.patch.object(MA.motion, "telemetry", return_value={
                "tof_mm": {"fl": 1100, "fr": 1200, "rl": 4000, "rr": 4000}}):
            self._tick(8, zone="public")
        self.come.assert_not_called()

    def test_open_floor_front_tof_allows_the_approach(self):
        with mock.patch.object(MA.motion, "telemetry", return_value={
                "tof_mm": {"fl": 3500, "fr": 3500, "rl": 4000, "rr": 4000}}):
            self._tick(4, zone="public")
        self.come.assert_called_once()

    def test_missing_front_tof_fails_open(self):
        # No usable front reading (sensor error): the gate must not silently kill
        # the behavior — the firmware's own obstacle stop still guards the drive.
        with mock.patch.object(MA.motion, "telemetry", return_value={
                "tof_mm": {"fl": -1, "fr": -1, "rl": 4000, "rr": 4000}}):
            self._tick(4, zone="public")
        self.come.assert_called_once()

    def test_brief_public_distance_does_not(self):
        self._tick(3, zone="public")
        self._tick(1, zone="social")   # counter resets
        self._tick(3, zone="public")
        self.come.assert_not_called()

    def test_not_facing_them_blocks_approach(self):
        self._neck = 7594  # 40% off-center — not facing them, approach counter idle
        self._tick(6, zone="public")
        self.come.assert_not_called()

    def test_approach_cooldown(self):
        self._tick(4, zone="public")
        self._tick(4, zone="public")
        self.assertEqual(self.come.call_count, 1)

    # ── gates ──────────────────────────────────────────────────────────────────

    def test_mid_sentence_freezes_everything(self):
        self._arm_realign()
        self._tick(4, profile=_profile(user_mid_sentence=True))
        self._tick(4, zone="public", profile=_profile(user_mid_sentence=True))
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_suppress_proactive_blocks_approach_not_realign(self):
        prof = _profile(suppress_proactive=True)
        self._tick(6, zone="public", profile=prof)
        self.come.assert_not_called()
        self._arm_realign()
        self._tick(2, profile=prof)
        self.turn.assert_called_once()   # realigning to face someone is not speech-like

    def test_moving_base_defers(self):
        self.state.return_value = "moving"
        self._arm_realign()
        self._tick(4, zone="public")
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_no_tracked_person_resets(self):
        self._tracking = {"locked": False, "visible": False}
        self._visible = False
        self._tick(6, zone="public")
        self.come.assert_not_called()

    def test_master_kill_switch(self):
        with mock.patch.object(config, "AUTONOMOUS_MOTION_ENABLED", False, create=True):
            self._arm_realign()
            self._tick(4, zone="public")
        self.turn.assert_not_called()
        self.come.assert_not_called()

    def test_sleep_resets_detectors_and_blocks_all_autonomy(self):
        from state import State
        MA._state["neck_hits"] = 2
        MA._state["far_hits"] = 3
        MA._flinch_state["hits"] = 5
        with mock.patch.object(
            MA.state_module, "get_state", return_value=State.SLEEP
        ):
            self._tick(1, zone="public")
        self.turn.assert_not_called()
        self.come.assert_not_called()
        self.assertEqual(MA._state["neck_hits"], 0)
        self.assertEqual(MA._state["far_hits"], 0)
        self.assertEqual(MA._flinch_state["hits"], 0)

    def test_disconnected_base_is_silent(self):
        self.available.return_value = False
        self._tick(4, zone="public")
        self.come.assert_not_called()

    # ── explicit requested come ───────────────────────────────────────────────

    def test_requested_come_scans_until_a_person_is_visible(self):
        self._tracking = {"locked": False, "visible": False}
        self._visible = False        # nobody on camera at all
        self.assertTrue(MA.request_come_here())
        self._tick()
        # Scan legs sweep at the dedicated (slower) rate so the sighting sampler
        # can catch a face the camera crosses mid-turn.
        self.turn.assert_called_once_with(
            config.MOTION_COME_SEARCH_TURN_DEG,
            rate=config.MOTION_COME_SCAN_RATE_DEG_S,
        )
        self.come.assert_not_called()
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_approaches_visible_centered_person_at_one_meter(self):
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.come.assert_called_once_with(
            0.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )
        # The errand now stays ALIVE across the drive so a transient obstruction
        # (a dog crossing) can be waited out and retried; it ends on the firmware
        # reporting the drive completed. See ComeResumesAfterBlockTest.
        self.assertTrue(MA.requested_come_active())

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
        self._visible = False        # nobody on camera
        with mock.patch.object(config, "MOTION_COME_SEARCH_MAX_TURNS", 2, create=True), \
             mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(3)
        self.assertEqual(self.turn.call_count, 2)
        self.assertFalse(MA.requested_come_active())

    def test_requested_come_dwells_after_a_scan_turn_completes(self):
        # After a scan leg's firmware `done` lands, the camera must DWELL (default
        # 3 s, settled) before the search may conclude "nobody this way" — the
        # detect→identify pipeline needs still frames (field 2026-08-11: back-to-
        # back legs left ~1 s of still camera and Rex swept right past the owner).
        self._tracking = {"locked": False, "visible": False}
        self._visible = False        # nobody on camera at all
        self.assertTrue(MA.request_come_here())
        self._tick(1)                              # scan turn 1 (no prior turn -> no wait)
        self.assertEqual(self.turn.call_count, 1)
        self._tick(4)                              # done landed; dwell holds every tick
        self.assertEqual(self.turn.call_count, 1)
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_waits_for_the_turn_done_before_deciding(self):
        # While our own turn is still executing (no `done` yet), no scan/align
        # decision may be made at all — the camera is swinging.
        self._tracking = {"locked": False, "visible": False}
        self._visible = False
        self.done_result.return_value = None       # turn never reports back...
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(1)
            self.assertEqual(self.turn.call_count, 1)
            self._tick(4)                          # in flight -> every tick waits
            self.assertEqual(self.turn.call_count, 1)
            self.done_result.return_value = "completed"
            self._tick(1)                          # done landed, dwell 0 -> next leg
        self.assertEqual(self.turn.call_count, 2)

    def test_requested_come_scan_sweeps_alternating_sides(self):
        # Sweep pattern (sign alternates, magnitude grows): +45, -90, +135 — net
        # offsets +45, -45, +90 around the last-known side, not a one-way spiral.
        self._tracking = {"locked": False, "visible": False}
        self._visible = False        # nobody on camera
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(3)
        rels = [c.args[0] for c in self.turn.call_args_list]
        self.assertEqual(rels, [45.0, -90.0, 135.0])

    def test_requested_come_align_seeds_sweep_side_and_dwell(self):
        # Person on the left (+ align turn), then lost: the sweep must start back
        # toward that side, and only after the settled-camera dwell.
        self._neck = 4400                          # far left -> positive align turn
        self.assertTrue(MA.request_come_here())
        self._tick(1)                              # align turn issued
        self.assertEqual(self.turn.call_count, 1)
        align_deg = self.turn.call_args[0][0]
        self.assertGreater(align_deg, 0)
        self._tracking = {"locked": False, "visible": False}
        self._visible = False        # nobody on camera
        self._tick(2)                              # inside the dwell -> no scan yet
        self.assertEqual(self.turn.call_count, 1)
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self._tick(1)                          # dwell over -> first sweep turn
        self.assertEqual(self.turn.call_count, 2)
        self.assertGreater(self.turn.call_args[0][0], 0)   # starts toward the last-known side

    def test_align_measurement_waits_out_the_settle_window(self):
        # After an align turn completes, the neck is still re-centring the same
        # error the base just corrected — measuring mid-slew produced the
        # +15/-37/+37 oscillation that never read "centered" (field 2026-08-11).
        self._neck = 7594
        with mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 5.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(1)                          # align turn 1
            self.assertEqual(self.turn.call_count, 1)
            self._tick(3)                          # done landed, settle holds
            self.assertEqual(self.turn.call_count, 1)
            self.come.assert_not_called()
            self.assertTrue(MA.requested_come_active())

    def test_requested_come_targets_the_requester_not_the_first_face(self):
        # JT (db 2) is plainly visible and centered; Bret (db 1) asked. Rex must
        # NOT deliver himself to JT — he keeps searching (owner spec 2026-08-11).
        self._tracking = {"locked": False, "visible": False}
        self.assertTrue(MA.request_come_here(person_id=1))
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            for _ in range(2):
                MA.step(_snapshot(db_id=2), _profile())
        self.come.assert_not_called()
        self.assertEqual(self.turn.call_count, 2)      # still sweeping
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_head_lock_on_the_wrong_person_is_ignored(self):
        # The head happens to be tracking db 1, but db 2 asked: the lock must not
        # steer the come-here at the wrong person.
        self._tracking = {"locked": True, "visible": True, "lock_key": "db:1"}
        self.assertTrue(MA.request_come_here(person_id=2))
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self._tick(1)                              # snapshot person is db 1
        self.come.assert_not_called()
        self.assertEqual(self.turn.call_count, 1)      # scan, not approach
        self.assertTrue(MA.requested_come_active())

    def test_requested_come_approaches_the_requester_once_found(self):
        self._tracking = {"locked": True, "visible": True, "lock_key": "db:1"}
        self.assertTrue(MA.request_come_here(person_id=1))
        self._tick(1)
        self.come.assert_called_once_with(
            0.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )

    def test_requester_sighting_restarts_the_giveup_clock(self):
        # Seen 40 s into a 45 s budget: the clock restarts from the sighting, so
        # the errand must NOT die of old age while he is actively working the
        # alignment (field 2026-08-11: "no person found" 5 s after aligning).
        self._tracking = {"locked": False, "visible": False}
        self._visible = False
        self.assertTrue(MA.request_come_here())
        MA._requested_come["started_at"] = time.monotonic() - 100.0   # way past timeout
        MA._requested_come["last_seen_at"] = time.monotonic() - 1.0   # but just seen
        self._tick(1)
        self.assertTrue(MA.requested_come_active(), "a fresh sighting must keep it alive")
        MA._requested_come["last_seen_at"] = time.monotonic() - 100.0
        self._tick(1)
        self.assertFalse(MA.requested_come_active())

    # ── fused alignment bearing (field 2026-08-11: ±12-45° oscillation) ────────

    def test_head_locked_with_face_off_centre_still_aligns(self):
        # Neck at neutral but the face far right of frame: the old head-locked
        # neck-only read said "centered" and launched the drive at nothing. The
        # fused bearing (neck + face) sees the true offset and aligns first.
        self._neck = 6000
        self._face_box = _EDGE_FACE_RIGHT
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.turn.assert_called_once()
        self.come.assert_not_called()

    def test_opposing_neck_and_face_offsets_cancel_and_he_approaches(self):
        # Mid-slew snapshot: head is +0.40 of half-span right (neck 7594) while
        # the face still sits left of frame centre by the same bearing. Either
        # signal alone reads a big error and over-corrects (the sign-flipping
        # align loop); the sum reads ~0 and the approach launches.
        self._neck = 6867                        # +0.40 of the 3488 half-span (neutral 5472)
        self._face_box = (476, 400, 200, 200)    # centre 576 -> -0.40 of half-width
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.come.assert_called_once_with(
            0.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )

    def test_align_turns_that_never_settle_hand_the_residual_to_come(self):
        # After MOTION_COME_ALIGN_MAX_TRIES align turns the residual bearing goes
        # to the firmware as the `come` heading instead of a fourth base turn —
        # alignment must not starve the approach forever (field 2026-08-11: four
        # minutes of align turns, zero re-approaches).
        self._neck = 5472                        # exactly neutral
        self._face_box = (1148, 400, 200, 200)   # centre 1248 -> +0.30 offset
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.turn.assert_called_once()           # try 1: a normal align turn
        self.assertEqual(MA._requested_come["align_turns"], 1)
        MA._requested_come["align_turns"] = int(config.MOTION_COME_ALIGN_MAX_TRIES)
        self._tick()
        self.assertEqual(self.turn.call_count, 1, "no further align turns")
        self.come.assert_called_once_with(
            -18.0, stop_at=config.MOTION_COME_REQUEST_STOP_AT_M
        )                                        # -0.30 * MOTION_FACE_TURN_MAX_DEG

    def test_a_lost_sighting_resets_the_align_try_counter(self):
        self._neck = 6000
        self._face_box = (1148, 400, 200, 200)
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.assertEqual(MA._requested_come["align_turns"], 1)
        self._visible = False
        self._tracking = {"locked": False, "visible": False}
        self._tick()
        self.assertEqual(MA._requested_come["align_turns"], 0)


class RequestedComeFieldFixTest(unittest.TestCase):
    """Fixes for the 2026-07-23 'come here just spins' session: mid-turn sightings
    are remembered and turned back toward; sweep legs rotate the short way; a front
    zone flap no longer kills the search; an explicit come preempts exploration."""

    def setUp(self):
        MA.cancel_requested_come("test reset")
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
            mock.patch.object(MA.motion, "done_result", return_value="completed",
                              create=True),
            mock.patch.object(config, "MOTION_COME_ALIGN_SETTLE_SECS", 0.0,
                              create=True),
        ]
        (self.available, self.state, self.turn, self.come, self.battery,
         self.done_result, _) = [p.start() for p in self._patches]
        self._tracking = {"locked": False, "visible": False}
        self._visible = False         # this class exercises the SEARCH path: empty room
        self._neck = 6000
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()

    def tearDown(self):
        MA.cancel_requested_come("test cleanup")
        self._ws.stop()
        for p in self._patches:
            p.stop()

    def _tick(self, n=1):
        for _ in range(n):
            MA.step(_snapshot(visible=self._visible), _profile())

    def test_midturn_sighting_turns_back_instead_of_sweeping_on(self):
        # Scan turn 1 issued; DURING the turn (base moving) the camera sweeps past
        # the person and face tracking locks briefly, off to Rex's right. Once lost
        # again, the search must turn back toward that side — not take sweep leg 2.
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(1)
            self.assertEqual(self.turn.call_count, 1)  # sweep leg 1
            # Mid-turn sighting: base busy, person visible, neck parked right.
            self.state.return_value = "moving"
            self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
            self._visible = True
            self._neck = 7594
            self._tick(1)                              # sampler records; step defers
            self.assertEqual(self.turn.call_count, 1)
            # Lock lost again, base settled.
            self.state.return_value = "idle"
            self._tracking = {"locked": False, "visible": False}
            self._visible = False        # nobody on camera
            self._tick(1)
        self.assertEqual(self.turn.call_count, 2)
        resight = self.turn.call_args[0][0]
        self.assertAlmostEqual(abs(resight), config.MOTION_COME_RESIGHT_TURN_DEG)
        self.assertLess(resight, 0)                    # back toward the right side
        self.assertTrue(MA.requested_come_active())

    def test_sweep_legs_rotate_the_short_way(self):
        # Later sweep legs used to issue -225/-270 relative spins; same net heading
        # must now be reached the short way (a command is never > 180 deg).
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True), \
             mock.patch.object(config, "MOTION_COME_SEARCH_MAX_TURNS", 6, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(6)
        rels = [c.args[0] for c in self.turn.call_args_list]
        # raw pattern would be +45,-90,+135,-180,+225,-270; the last two wrap to
        # the equivalent short rotations -135 and +90.
        self.assertEqual(rels, [45.0, -90.0, 135.0, -180.0, -135.0, 90.0])
        self.assertTrue(all(abs(r) <= 180.0 for r in rels))

    def test_front_zone_block_does_not_cancel_the_search(self):
        # Turning away from a block is firmware-legal; a front flap must not kill
        # the search (it only defers the forward approach).
        self.state.return_value = "blocked"
        with mock.patch.object(config, "MOTION_COME_SCAN_DWELL_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick(1)
            self.assertTrue(MA.requested_come_active())
            self.assertEqual(self.turn.call_count, 1)   # scan turn still issued
            # Person found and centered while blocked: hold, don't approach yet.
            self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
            self._visible = True
            self._tick(1)
            self.come.assert_not_called()
            self.assertTrue(MA.requested_come_active())
            # Front clears -> approach starts.
            self.state.return_value = "idle"
            self._tick(1)
        self.come.assert_called_once()
        self.assertTrue(MA.requested_come_active(), "alive until the drive reports back")

    def test_come_approaches_a_visible_face_without_a_head_lock(self):
        # THE FIELD FAILURE (2026-07-24, owner ~9 ft away): "my face was detected in
        # the GUI when I said come here, but he just turned left and right then
        # around and never came anywhere." A short "Come here." scored 0.232, was
        # ruled an off-camera stranger, and the gaze SEARCH that followed pulled the
        # head off his face — breaking the lock come-here used as its only
        # visibility signal. world_state.people still had him the whole time.
        self._tracking = {"locked": False, "visible": False}   # lock pulled away
        self._visible = True                                   # but plainly on camera
        self._neck = 6000                                      # head centred: no steer
        self.assertTrue(MA.request_come_here())
        self._tick(1)
        self.turn.assert_not_called()          # nothing to align: don't sweep the room
        self.come.assert_called_once()         # go to them
        self.assertTrue(MA.requested_come_active(), "alive until the drive reports back")

    def test_come_aligns_off_the_face_when_the_head_is_not_on_them(self):
        # Same situation but the face sits far to Rex's right in frame. With no head
        # lock the neck says "centred", so the align turn has to come from the face's
        # position in frame or he would drive straight past them.
        self._tracking = {"locked": False, "visible": False}
        self._visible = True
        self._neck = 6000
        snap = {"people": [{"id": "person_1", "person_db_id": 1,
                            "distance_zone": "social", "face_visible": True,
                            "face_box": (1500, 400, 200, 200)}]}   # right of centre
        self.assertTrue(MA.request_come_here())
        MA.step(snap, _profile())
        self.turn.assert_called_once()
        self.assertLess(self.turn.call_args[0][0], 0, "face on the right -> CW turn")

    def test_request_come_stops_active_exploration(self):
        with mock.patch("intelligence.exploration.active", return_value=True), \
             mock.patch("intelligence.exploration.stop") as stop:
            self.assertTrue(MA.request_come_here())
        stop.assert_called_once()
        self.assertTrue(MA.requested_come_active())


class ComeResumesAfterBlockTest(unittest.TestCase):
    """A come-here errand must survive being stopped short.

    Field 2026-07-24: "If he gets blocked by my dog walking in front of it, he stops
    and tells me so. But if my dog moves out of the way he should keep trying to come
    to the speaker." The errand used to end the instant `come` was sent, so anything
    that interrupted the drive ended the whole thing.
    """

    def setUp(self):
        MA.cancel_requested_come("test reset")
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0, user_motion_at=0.0)
        self._result = None          # what the firmware says the last `come` did
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", side_effect=lambda: self._state_val),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch.object(MA.motion_controller, "last_come_result",
                              side_effect=lambda: (8, self._result)),
            mock.patch("intelligence.battery_awareness.battery_critical", return_value=False),
        ]
        (self.available, self.state, self.turn, self.come,
         self.last_result, self.battery) = [p.start() for p in self._patches]
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        self.addCleanup(lambda: MA.cancel_requested_come("test cleanup"))
        self._state_val = "idle"
        self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
        self._neck = 6000            # centred: no align turn, straight to the approach
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()
        self.addCleanup(self._ws.stop)

    def _tick(self, n=1):
        for _ in range(n):
            MA.step(_snapshot(), _profile())

    def test_dog_walks_through_then_leaves_and_he_resumes(self):
        with mock.patch.object(config, "MOTION_COME_RETRY_GAP_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            self._tick()                          # launch
            self.assertEqual(self.come.call_count, 1)
            self.assertTrue(MA.requested_come_active(), "errand must stay alive")

            # Dog steps in: firmware blocks the drive.
            self._result, self._state_val = "blocked", "blocked"
            self._tick(3)
            self.assertEqual(self.come.call_count, 1, "must not butt at the obstruction")
            self.assertTrue(MA.requested_come_active(), "a block must not end the errand")

            # Dog leaves: the base is free again.
            self._state_val = "idle"
            self._tick()
            self.assertEqual(self.come.call_count, 2, "he must resume once the path clears")

    def test_arrival_ends_the_errand(self):
        self.assertTrue(MA.request_come_here())
        self._tick()
        self.assertEqual(self.come.call_count, 1)
        self._result = "completed"                # firmware reached the stop distance
        self._tick()
        self.assertFalse(MA.requested_come_active(), "arriving must end the errand")
        self.assertEqual(self.come.call_count, 1, "no re-launch after arriving")

    def test_completed_while_still_far_is_a_stopped_short_retry(self):
        # Firmware "completed" = it stopped stop_at short of the nearest front
        # return — and a phantom floor return (mis-calibrated matrix ToF)
        # completes the drive seconds in while the requester still reads PUBLIC
        # distance across the room (field 2026-08-11: `come` done in 3 s, 261
        # front zone_blocks that session). Visibly-far + completed = the sensor
        # is lying; retry instead of declaring arrival.
        with mock.patch.object(config, "MOTION_COME_RETRY_GAP_SECS", 0.0, create=True):
            self.assertTrue(MA.request_come_here())
            MA.step(_snapshot(distance_zone="public"), _profile())
            self.assertEqual(self.come.call_count, 1)
            self._result = "completed"
            MA.step(_snapshot(distance_zone="public"), _profile())
        self.assertTrue(MA.requested_come_active(),
                        "a phantom arrival must not end the errand")
        self.assertEqual(self.come.call_count, 2, "he must try again")

    def test_still_driving_is_left_alone(self):
        self.assertTrue(MA.request_come_here())
        self._tick()
        self._result = None                       # in flight
        self._tick(3)
        self.assertEqual(self.come.call_count, 1, "must not re-issue mid-drive")
        self.assertTrue(MA.requested_come_active())

    def test_a_permanent_obstruction_gives_up(self):
        with mock.patch.object(config, "MOTION_COME_RETRY_GAP_SECS", 0.0, create=True), \
             mock.patch.object(config, "MOTION_COME_MAX_APPROACHES", 3, create=True):
            self.assertTrue(MA.request_come_here())
            for _ in range(12):
                self._tick()
                self._result = "blocked"          # never actually clears
        self.assertLessEqual(self.come.call_count, 3)
        self.assertFalse(MA.requested_come_active(), "must not retry forever")


class UserMotionStanddownTest(unittest.TestCase):
    """After an explicit voice motion command, realign must NOT rotate the body
    back toward the face (field 2026-07-23: "turn right a little" -> -45, realign
    +30 toward the speaker 13 s later)."""

    def setUp(self):
        MA.cancel_requested_come("test reset")
        MA._state.update(neck_hits=0, far_hits=0, last_turn_at=0.0,
                         last_approach_at=0.0, user_motion_at=0.0, hold_at=None)
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
        self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
        self._visible = True
        self._neck = _EXHAUSTED_NECK_RIGHT   # sweep exhausted right
        self._face_box = _EDGE_FACE_RIGHT    # face at frame edge — realign would fire
                                             # after 2 confirm ticks
        self._ws = mock.patch(
            "world_state.world_state.get",
            side_effect=lambda key: (
                {"face_tracking": self._tracking,
                 "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {}),
        )
        self._ws.start()

    def tearDown(self):
        MA._state["user_motion_at"] = 0.0
        MA._state["hold_at"] = None
        self._ws.stop()
        for p in self._patches:
            p.stop()

    def _tick(self, n=1):
        for _ in range(n):
            MA.step(_snapshot(visible=self._visible, face_box=self._face_box),
                    _profile())

    def test_realign_stands_down_after_user_motion(self):
        MA.note_user_motion()
        self._tick(4)
        self.turn.assert_not_called()

    def test_realign_resumes_after_window_expires(self):
        MA.note_user_motion()
        MA._state["user_motion_at"] = time.monotonic() - (
            float(config.MOTION_USER_MOTION_STANDDOWN_SECS) + 1.0
        )
        self._tick(2)   # confirm ticks
        self.turn.assert_called_once()

    def test_flinch_reflex_ignores_the_standdown(self):
        # note_user_motion must not silence the safety reflex path.
        MA.note_user_motion()
        with mock.patch.object(MA, "_maybe_flinch", return_value=True) as flinch:
            self._tick(1)
        flinch.assert_called_once()

    def test_hold_does_not_silence_the_flinch_reflex(self):
        # A reflex is not a social behavior: someone crowding him still gets a
        # back-off even while he has been told to stay put.
        MA.note_user_hold()
        with mock.patch.object(MA, "_maybe_flinch", return_value=True) as flinch:
            self._tick(1)
        flinch.assert_called_once()

    def test_hold_blocks_realign_with_no_expiry(self):
        MA.note_user_hold()
        MA._state["hold_at"] = time.monotonic() - 86400.0   # a day ago; 0 = never expires
        self._tick(4)
        self.turn.assert_not_called()

    def test_come_here_ignores_the_standdown(self):
        MA.note_user_motion()
        self.assertTrue(MA.request_come_here())
        self._tick(1)
        # Person visible far right -> the come align turn still fires.
        self.turn.assert_called_once()


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
                         last_approach_at=0.0, last_flinch_at=0.0,
                         user_motion_at=0.0)
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
            # Pin the flinch thresholds the fixtures are calibrated to (250 mm intrusion,
            # 1.5 m baseline) so these logic tests don't move when the SHIPPING defaults
            # are re-tuned (raised 2026-07-23 to make the reflex harder to trigger).
            mock.patch.object(config, "MOTION_FLINCH_TRIGGER_M", 0.26, create=True),
            mock.patch.object(config, "MOTION_FLINCH_APPROACH_DROP_M", 0.20, create=True),
            mock.patch.object(config, "MOTION_FLINCH_CONFIRM_TICKS", 5, create=True),
            mock.patch.object(config, "MOTION_FLINCH_COOLDOWN_SECS", 6.0, create=True),
            # Pinned too, so "sub-floor read" keeps meaning what the fixtures assume
            # (30 mm is noise at a 50 mm floor). The SHIPPING floor is lower — a foot
            # at 1 inch is a real object — and is guarded by FlinchShippingDefaultsTest.
            mock.patch.object(config, "MOTION_FLINCH_MIN_VALID_M", 0.05, create=True),
        ]
        (self.available, self.state, self.telemetry, self.move, self.turn,
         self.come, self.ws, *_cfg) = [p.start() for p in self._patches]

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

    def _approach(self, close_mm=250, rear_mm=None, far_mm=1500, confirm=None, profile=None):
        """One far tick to seat the baseline, then `confirm` close ticks."""
        if confirm is None:
            confirm = config.MOTION_FLINCH_CONFIRM_TICKS
        self._tick(front_mm=far_mm, rear_mm=rear_mm, profile=profile)
        for _ in range(confirm):
            self._tick(front_mm=close_mm, rear_mm=rear_mm, profile=profile)

    # ── firing ────────────────────────────────────────────────────────────────

    def test_front_approach_backs_up(self):
        self._tick(front_mm=1500)      # far — seats the "came from" baseline
        for _ in range(config.MOTION_FLINCH_CONFIRM_TICKS - 1):
            self._tick(front_mm=250)
            self.move.assert_not_called()
        self._tick(front_mm=250)       # sustained intrusion -> flinch
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

    def test_slow_creep_does_not_trigger_hardened_reflex(self):
        # A gradual range drift is deliberately insufficient now; the hardened reflex
        # requires a sustained close intrusion off a meaningfully open baseline.
        for mm in (600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 90, 80):
            self._tick(front_mm=mm)
        self.move.assert_not_called()

    def test_static_clutter_one_side_does_not_mask_other_side_approach(self):
        # fr pinned by a static object at 0.30 m; fl sees a real walk-up.
        self._tick(fl_mm=1500, fr_mm=300)
        self._tick(fl_mm=1500, fr_mm=300)   # baselines settle: fl~1.5, fr frozen 0.30
        for _ in range(config.MOTION_FLINCH_CONFIRM_TICKS):
            self._tick(fl_mm=250, fr_mm=200)
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
        for _ in range(config.MOTION_FLINCH_CONFIRM_TICKS):
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
        for _ in range(config.MOTION_FLINCH_CONFIRM_TICKS):
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


class FlinchShippingDefaultsTest(unittest.TestCase):
    """FlinchTest pins its own thresholds so the LOGIC stays testable while the
    shipping numbers are tuned. That is exactly how the reflex silently died: the
    2026-07-23 hardening left the SHIPPING config with no feasible window at all,
    and every logic test still passed.

    A flinch needs, at the same instant:
        MIN_VALID <= d < TRIGGER      and      d <= baseline - DROP
    With the hardened values (TRIGGER 0.18, DROP 0.26, MIN_VALID 0.05) and a foot
    hovering ~0.30 m out, that demanded d <= 0.04 while discarding everything under
    0.05 — unfireable. Field 2026-07-24: "I moved my foot from about 1 foot to 1 inch
    from the front ToF array and he did not back up."
    """

    def _cfg(self, name, default=None):
        return float(getattr(config, name, default))

    def test_a_reachable_distance_window_exists(self):
        trigger = self._cfg("MOTION_FLINCH_TRIGGER_M")
        drop = self._cfg("MOTION_FLINCH_APPROACH_DROP_M")
        min_valid = self._cfg("MOTION_FLINCH_MIN_VALID_M")
        # Someone standing a foot away who then crowds in is the ordinary case.
        baseline = 0.30
        highest_firing_d = baseline - drop
        self.assertGreaterEqual(
            highest_firing_d, min_valid,
            f"no feasible window: DROP={drop} needs d<={highest_firing_d:.3f} m but "
            f"MIN_VALID={min_valid} discards anything under it",
        )
        self.assertLess(min_valid, trigger, "the noise floor swallows the trigger radius")

    def test_drop_is_not_wider_than_the_trigger_radius(self):
        # If DROP >= TRIGGER the reflex can only ever fire for a target that started
        # outside the trigger AND ended near zero — the window collapses.
        self.assertLess(self._cfg("MOTION_FLINCH_APPROACH_DROP_M"),
                        self._cfg("MOTION_FLINCH_TRIGGER_M"))

    def test_confirm_window_is_a_reflex_not_a_wait(self):
        ticks = self._cfg("MOTION_FLINCH_CONFIRM_TICKS")
        interval = self._cfg("CONSCIOUSNESS_LOOP_INTERVAL_SECS", 1.0)
        self.assertGreaterEqual(ticks, 2, "single-frame noise must not fire it")
        self.assertLessEqual(ticks * interval, 4.0,
                             f"{ticks} ticks x {interval}s is a wait, not a reflex")

    def test_an_inch_away_is_a_real_reading_not_noise(self):
        # 1 inch = 0.0254 m. The reflex matters MOST when something is closest, so
        # the noise floor must sit below a genuine very-close object.
        self.assertLess(self._cfg("MOTION_FLINCH_MIN_VALID_M"), 0.0254)


class FlinchEndToEndDefaultsTest(unittest.TestCase):
    """Drive the real _maybe_flinch with the SHIPPING defaults through the owner's
    exact scenario: idle at ~1 ft, then a foot at ~1 inch, held."""

    def setUp(self):
        MA._reset_flinch()
        MA._state.update(last_flinch_at=0.0, user_motion_at=0.0)
        self.addCleanup(MA._reset_flinch)
        self._front_m = 0.30
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "move", return_value=9),
            mock.patch.object(MA.motion, "telemetry", side_effect=lambda: {
                "tof_mm": {"fl": int(self._front_m * 1000), "fr": int(self._front_m * 1000),
                           "rl": 4000, "rr": 4000}}),
        ]
        self.available, self.state, self.move, self.telemetry = [
            p.start() for p in self._patches
        ]
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def _run(self, approach_m, hold_ticks, standoff_m=0.30):
        prof = _profile()
        self._front_m = standoff_m
        for _ in range(5):                       # establish the open-distance baseline
            MA._maybe_flinch(prof, time.monotonic(), "idle")
        self._front_m = approach_m
        for tick in range(1, hold_ticks + 1):
            if MA._maybe_flinch(prof, time.monotonic(), "idle"):
                return tick
        return None

    def test_foot_at_one_inch_backs_him_off(self):
        fired = self._run(0.025, hold_ticks=8)
        self.assertIsNotNone(fired, "the reflex never fired on a foot at 1 inch")
        self.assertLessEqual(fired, 4, "a reflex must not take more than a few seconds")
        self.move.assert_called()
        self.assertLess(self.move.call_args[0][0], 0, "back-off must move BACKWARD")

    def test_static_clutter_never_fires(self):
        # A wall parked at 20 cm the whole time has no closing trend.
        self.assertIsNone(self._run(0.20, hold_ticks=12, standoff_m=0.20))
        self.move.assert_not_called()

    def test_a_momentary_poke_does_not_fire(self):
        self.assertIsNone(self._run(0.025, hold_ticks=1))
