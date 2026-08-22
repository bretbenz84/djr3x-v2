""""Turn to face me" — one base turn onto the requester's bearing, then stop.

The whole feature is a sign chain, and a sign chain is where this class of thing
dies. Two frames meet here and they are OPPOSITE by design:

    camera bearing (_come_bearing_deg)   + = Rex's RIGHT
    radar bearing  (_radar_bodies)       + = LEFT/CCW, already the turn frame
    turn command   (motion_controller)   + = LEFT/CCW

so the camera number is negated exactly once (_come_turn_for_bearing) and the radar
number is not negated at all. Send a radar bearing through the camera's converter
and Rex turns exactly as far the wrong way. Most of this file exists to pin that.

Run per module (never `unittest discover` — see CLAUDE.md):

    venv/bin/python -m unittest tests.test_motion_face
"""

import unittest
from unittest import mock

import config
from intelligence import action_router as ar
from intelligence import motion_agency as MA
from intelligence import tool_router


# The idle wander rolls real dice and the startup approach arms on import; both
# steer the neck, which is the bearing this feature reads.
_WANDER_OFF = mock.patch.object(config, "MOTION_IDLE_WANDER_ENABLED", False,
                                create=True)
_STARTUP_OFF = mock.patch.object(config, "MOTION_STARTUP_APPROACH_ENABLED", False,
                                 create=True)


def setUpModule():
    _WANDER_OFF.start()
    _STARTUP_OFF.start()


def tearDownModule():
    _WANDER_OFF.stop()
    _STARTUP_OFF.stop()


# Servo neutral for the neck (SERVO_CHANNELS neck: 1984 / 8960 / 5472).
_NECK_NEUTRAL = 5472

REAL_COMMANDS = (
    "face me", "turn to face me", "turn and face me", "turn towards me",
    "turn toward me", "point yourself at me", "face me please", "rex, face me",
    "can you turn to face me",
    # Literal drive commands that merely share the verb — these must KEEP working.
    "face the other way", "face forward", "face away", "face my way",
    "face your left", "face your right",
    # A robot at a party facing the crowd is a real request, and "come face me"
    # leads with a motion verb, so it is a come command whatever else it is.
    "face the crowd", "come face me",
    # THE REGRESSION SET. Every one of these was admitted before the figurative
    # widening and refused after the first draft of it — 14 of 14 — because the
    # "face me <tail>" clause listed bare prepositions and adverbs instead of the
    # thing that actually makes the phrase figurative, the arena. "face me again"
    # is the likeliest utterance of the entire feature: he turns a little short and
    # you say it. And because _MOTION_FIGURATIVE_RE is searched over the WHOLE
    # utterance for EVERY motion action, the over-block leaked sideways — "come
    # here and face me for a sec" stopped being a come command.
    "face me again", "turn and face me again", "face me for a second",
    "face me one more time", "rex face me over here", "face me at the table",
    "turn to face me on the couch", "turn around and face the crowd",
    "come here and face me for a sec", "turn to face me for a moment",
    "face me over there", "back up two feet then turn and face me for a second",
)
# "face" is the widest verb in the motion evidence gate — bare, with no object
# restriction, because "face me" IS the command. Measured 2026-08-22 before the
# guard was widened: 14 of these 23 cleared the gate.
FIGURATIVE_DECOYS = (
    "face the music", "face the facts", "face reality", "face it",
    "face your fears", "face the day", "face the consequences",
    "face the challenge", "face the truth", "face facts",
    "face off", "face up to it", "face palm", "face me in chess",
    "face me like a man", "face me on the court", "face me in the final",
    "face us at the tournament", "face me like a real droid",
    "you'll have to face me eventually", "he had to face the wall",
    "she faced the camera", "let's face it, that's wrong",
    "time to face the problem",
)


def _person(db_id=1, face_box=None):
    return {"person_db_id": db_id, "id": "person_1", "name": "Bret",
            "face_visible": True, "face_box": face_box}


def _snapshot(person=None):
    return {"people": [person] if person else []}


class CatalogWiringTest(unittest.TestCase):
    """Each of these fails OPEN into conversation if missed — the whole reason the
    new-executable-action checklist exists."""

    def test_spec_is_executable_motion(self):
        spec = {s.key: s for s in ar.ACTION_SPECS}["motion.face"]
        self.assertEqual(spec.category, "motion")
        self.assertTrue(spec.executable)
        self.assertIn("motion.face", ar.EXECUTABLE_ACTIONS)

    def test_schema_exists_and_takes_no_arguments(self):
        # A spec with no _TOOL_DEFS entry is a KeyError on EVERY reply call, not a
        # quiet gap — tool_schemas() indexes the table unguarded.
        schema = tool_router.tool_schema_for("motion.face")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["function"]["parameters"]["properties"], {})

    def test_allowlist_and_live_set_carry_the_key(self):
        self.assertIn("motion.face", config.ACTION_ROUTER_EXECUTE_ACTIONS)
        self.assertIn("motion.face", config.TOOL_ROUTER_LIVE_ACTIONS)
        self.assertIn("motion.face", tool_router.live_actions())

    def test_it_is_not_dispatched_as_a_single_motion_action(self):
        # _MOTION_ACTIONS dispatches to _handle_router_motion_action, which has no
        # face arm and returns None — Rex answers with a classic reply and never
        # moves. It must stay OUT of that set and IN the drive set (for the no-base
        # verbal denial).
        from intelligence import interaction
        self.assertNotIn("motion.face", interaction._MOTION_ACTIONS)
        self.assertIn("motion.face", interaction._MOTION_DRIVE_ACTIONS)

    def test_the_evidence_gate_covers_the_key(self):
        # missing_required_evidence_reason ends in a bare `return None`, so an
        # unlisted action gets NO evidence requirement at all, and
        # motion_command_refusal_reason short-circuits the same way.
        self.assertIn("motion.face", ar._MOTION_TOOL_ACTIONS)
        decision = ar.ActionDecision(action="motion.face", confidence=1.0,
                                     args={}, reason="test")
        self.assertIsNone(
            ar.missing_required_evidence_reason("turn to face me", decision))
        self.assertEqual(
            ar.missing_required_evidence_reason("face me in chess", decision),
            "missing_motion_command_evidence")

    def test_motion_stays_out_of_the_owned_actions_demotion(self):
        self.assertEqual(
            [a for a in ar.TOOL_ROUTER_OWNED_ACTIONS if a.startswith("motion")], [])


class FigurativeFaceGateTest(unittest.TestCase):
    """The gate is the only thing between an idiom and the wheels here."""

    def test_real_commands_are_admitted(self):
        for text in REAL_COMMANDS:
            self.assertIsNone(
                ar.motion_command_refusal_reason(text, "motion.face"), text)

    def test_every_figurative_face_is_refused(self):
        for text in FIGURATIVE_DECOYS:
            self.assertEqual(
                ar.motion_command_refusal_reason(text, "motion.face"),
                "missing_motion_command_evidence", text)

    def test_the_widening_also_covers_the_existing_turn_lane(self):
        # The same hole was open on motion.turn, which has admitted these all along.
        for text in FIGURATIVE_DECOYS:
            self.assertEqual(
                ar.motion_command_refusal_reason(text, "motion.turn"),
                "missing_motion_command_evidence", text)

    def test_look_at_me_stays_out_of_the_wheels(self):
        # "look" is deliberately absent from the motion verb set. That omission is
        # the whole guard keeping the "look at ..." family a head gesture, and it
        # must not be relaxed to catch "turn and look at me".
        for text in ("look at me", "look at me go", "look over here"):
            self.assertEqual(
                ar.motion_command_refusal_reason(text, "motion.face"),
                "missing_motion_command_evidence", text)


class _BearingTestBase(unittest.TestCase):
    """Drives motion_agency.face_requester with the world faked around it."""

    def setUp(self):
        self.turns: list = []
        self.snapshot = _snapshot(_person())
        self.tracking = {"locked": True, "visible": True, "lock_key": "db:1"}
        self.neck = _NECK_NEUTRAL
        MA._state.update(last_turn_at=0.0, last_approach_at=0.0, last_flinch_at=0.0,
                         no_traction_until=0.0, traction_fails=0)
        MA._state.pop("face_me_radar_reason", None)
        MA.cancel_requested_come("test setup")
        self._stack = []
        for patcher in (
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion_controller, "turn",
                              side_effect=lambda d, rate=None: self.turns.append(
                                  (round(d, 3), rate)) or 7),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch("world_state.world_state.snapshot",
                       side_effect=lambda: self.snapshot),
            mock.patch("world_state.world_state.get",
                       side_effect=lambda key: (
                           {"face_tracking": self.tracking,
                            "servo_positions": {"neck": self.neck},
                            "frame_size": {"width": 1920, "height": 1080}}
                           if key == "self_state" else {})),
            mock.patch.object(MA, "_radar_bodies", return_value=([], True)),
        ):
            self._stack.append(patcher)
            patcher.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for patcher in reversed(self._stack):
            try:
                patcher.stop()
            except Exception:
                pass

    def _face_at(self, frac):
        """Put the requester's face box at `frac` of the half-width off centre
        (+ = Rex's right), neck parked so frame centre IS the nose."""
        width, box_w = 1920.0, 100.0
        centre = width / 2.0 + frac * (width / 2.0)
        self.snapshot = _snapshot(_person(face_box=[centre - box_w / 2.0, 400.0,
                                                    box_w, 120.0]))


class CameraBearingSignTest(_BearingTestBase):
    """+ camera bearing = Rex's RIGHT, and a right-hand person needs a NEGATIVE
    (clockwise) turn. This is the single inversion in the whole system."""

    def test_a_person_on_the_right_draws_a_negative_turn(self):
        self._face_at(0.9)                       # far right of frame
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertLess(deg, 0.0, "right of frame must turn CW (negative)")
        self.assertEqual(self.turns[0][0], deg)

    def test_a_person_on_the_left_draws_a_positive_turn(self):
        self._face_at(-0.9)
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertGreater(deg, 0.0, "left of frame must turn CCW (positive)")

    def test_the_turn_carries_the_configured_rate(self):
        self._face_at(0.9)
        MA.face_requester(1)
        self.assertEqual(self.turns[0][1],
                         float(config.MOTION_FACE_ME_TURN_RATE_DEG_S))

    def test_a_centred_face_says_so_and_never_turns(self):
        # The dead-band must run BEFORE _come_turn_for_bearing, whose
        # MOTION_FACE_TURN_MIN_DEG floor maps a 0 deg bearing to a +10 deg turn and
        # a +3 to a -10 — i.e. "you're already looking at me" becomes a swing AWAY.
        for frac in (0.0, 0.05, -0.05, 0.2, -0.2):
            self.turns.clear()
            self._face_at(frac)
            deg, reason = MA.face_requester(1)
            self.assertEqual(reason, "already_facing", frac)
            self.assertIsNone(deg, frac)
            self.assertEqual(self.turns, [], frac)

    def test_the_floor_trap_this_dead_band_exists_for(self):
        # Pinned so the day someone "simplifies" the dead-band away, the reason is
        # in the failure message.
        self.assertEqual(MA._come_turn_for_bearing(0.0), 10.0)
        self.assertEqual(MA._come_turn_for_bearing(3.0), -10.0)

    def test_the_neck_yaw_is_part_of_the_bearing(self):
        # Face centred in frame but the head is cranked right: the person IS to the
        # right, and a face-only reading would call that "already facing you".
        cfg = config.SERVO_CHANNELS["neck"]
        self.neck = int(cfg["neutral"] + 0.8 * (cfg["max"] - cfg["neutral"]))
        self._face_at(0.0)
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertLess(deg, 0.0)


class RadarBearingSignTest(_BearingTestBase):
    """Radar bearings are ALREADY + = left/CCW and go into turn() raw. Negating
    them would turn Rex exactly as far the wrong way."""

    def _bodies(self, *bodies):
        return mock.patch.object(MA, "_radar_bodies", return_value=(list(bodies), True))

    def _body(self, bearing, rng=2.0, conf=0.9, hits=4):
        return {"bearing_deg": bearing, "range_m": rng, "confidence": conf,
                "hits": hits, "frames": 5}

    def setUp(self):
        super().setUp()
        self.snapshot = _snapshot()        # nobody on camera -> the radar lane
        self.tracking = {"locked": False, "visible": False, "lock_key": ""}

    def test_a_radar_bearing_is_not_negated(self):
        with self._bodies(self._body(+40.0)):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertEqual(deg, 40.0, "radar + = left/CCW already; negating mirrors it")
        # A turn stamps last_turn_at, and the quiet window then blocks the next
        # radar read — his own rotation smears the bearings. Clear it to test the
        # mirror case; the window itself is pinned below.
        MA._state["last_turn_at"] = 0.0
        with self._bodies(self._body(-40.0)):
            self.turns.clear()
            deg, _ = MA.face_requester(1)
        self.assertEqual(deg, -40.0)

    def test_a_second_face_me_waits_out_his_own_rotation(self):
        with self._bodies(self._body(+40.0)):
            self.assertEqual(MA.face_requester(1)[1], "turned")
            self.assertEqual(MA.face_requester(1)[1], "no_bearing")

    def test_someone_behind_him_is_reachable(self):
        # The camera lane inherits a 60 deg cap through _come_turn_for_bearing, which
        # is right for a bearing that can only be inside the frame. Behind him is the
        # single likeliest reason to say "turn to face me", so the radar lane gets a
        # wider cap — 60 would leave him at a wall announcing he had faced you.
        with self._bodies(self._body(+170.0)):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertEqual(deg, 170.0)

    def test_more_than_one_plausible_body_refuses(self):
        # Radar reports a body, never WHOSE. Come-here can take the best guess
        # because it then LOOKS and marks the bearing spent; a one-shot has no
        # look-and-retry, so ambiguity is a refusal.
        with self._bodies(self._body(+40.0), self._body(-120.0)):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "ambiguous")
        self.assertIsNone(deg)
        self.assertEqual(self.turns, [])

    def test_the_near_ghost_floor_drops_the_shell_echo(self):
        # The firmware's own range gate leaks a ~0.5 m self-return that is present in
        # EVERY frame, so it scores maximum hits and sorts first.
        with self._bodies(self._body(+170.0, rng=0.5), self._body(-30.0, rng=2.0)):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")
        self.assertEqual(deg, -30.0)

    def test_low_confidence_and_far_returns_are_dropped(self):
        with self._bodies(self._body(+40.0, conf=0.05)):
            self.assertEqual(MA.face_requester(1)[1], "no_bearing")
        with self._bodies(self._body(+40.0, rng=99.0)):
            self.assertEqual(MA.face_requester(1)[1], "no_bearing")

    def test_a_recent_maneuver_of_his_own_blocks_the_radar_read(self):
        # Bearings smear across his own rotation.
        MA._state["last_turn_at"] = MA.time.monotonic()
        with self._bodies(self._body(+40.0)):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "no_bearing")
        self.assertIsNone(deg)

    def test_nothing_at_all_refuses_rather_than_guessing(self):
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "no_bearing")
        self.assertIsNone(deg)
        self.assertEqual(self.turns, [])


class RequesterIdentityTest(_BearingTestBase):
    """Camera first, and identity-resolved — the one thing radar cannot do."""

    def test_someone_elses_face_does_not_satisfy_the_request(self):
        # The JT-on-the-couch failure: with two people in the room, "face me" must
        # go to whoever SAID it. Person 2 is visible; person 1 asked.
        self.snapshot = _snapshot(_person(db_id=2, face_box=[1700, 400, 100, 120]))
        self.tracking = {"locked": True, "visible": True, "lock_key": "db:2"}
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "no_bearing")
        self.assertEqual(self.turns, [])

    def test_an_unknown_voice_never_widens_to_any_known_face(self):
        self._face_at(0.9)
        deg, reason = MA.face_requester(None)
        self.assertEqual(reason, "no_bearing")
        self.assertEqual(self.turns, [])


class AgencyGateTest(_BearingTestBase):
    def test_a_running_come_here_is_left_alone(self):
        self._face_at(0.9)
        with mock.patch.object(MA, "requested_come_active", return_value=True):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "come_active")
        self.assertEqual(self.turns, [])

    def test_a_moving_base_is_busy(self):
        self._face_at(0.9)
        with mock.patch.object(MA.motion, "state", return_value="moving"):
            self.assertEqual(MA.face_requester(1)[1], "busy")

    def test_a_traction_latch_declines(self):
        self._face_at(0.9)
        MA._state["no_traction_until"] = MA.time.monotonic() + 60.0
        try:
            self.assertEqual(MA.face_requester(1)[1], "traction")
        finally:
            MA._state["no_traction_until"] = 0.0

    def test_the_master_flag_disables_it(self):
        self._face_at(0.9)
        with mock.patch.object(config, "MOTION_FACE_ME_ENABLED", False, create=True):
            self.assertEqual(MA.face_requester(1)[1], "disabled")
        self.assertEqual(self.turns, [])

    def test_a_base_that_refuses_the_turn_reports_suppressed(self):
        # The swing check (a spin into the shelf behind him), a ToF block, manual
        # override — turn() returns None and nothing moved, so say nothing rather
        # than announcing a turn that did not happen.
        self._face_at(0.9)
        with mock.patch.object(MA.motion_controller, "turn", return_value=None):
            deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "suppressed")
        self.assertIsNone(deg)

    def test_the_turn_is_stamped_as_voice_commanded(self):
        # Two things hang off the stamp. motion_controller._suppressed only speaks
        # its refusal ("I'd clip something behind me") when a human asked out loud,
        # so without it a swing-blocked face-me is silent at BOTH ends — and turning
        # toward someone BEHIND him is the likeliest swing block there is. It also
        # keeps the drive whir at commanded volume instead of the autonomous half.
        self._face_at(0.9)
        with mock.patch.object(MA.motion_controller,
                               "note_user_commanded_motion") as stamp:
            MA.face_requester(1)
        stamp.assert_called_once()

    def test_the_stamp_lands_before_the_turn_is_issued(self):
        # Order matters: _suppressed reads the stamp DURING turn(), so stamping
        # afterwards would be the same silence.
        order = []
        self._face_at(0.9)
        with mock.patch.object(MA.motion_controller, "note_user_commanded_motion",
                               side_effect=lambda: order.append("stamp")), \
                mock.patch.object(MA.motion_controller, "turn",
                                  side_effect=lambda d, rate=None: order.append("turn") or 7):
            MA.face_requester(1)
        self.assertEqual(order, ["stamp", "turn"])

    def test_it_does_not_stand_realign_down(self):
        # The whole point. A commanded turn normally calls note_user_motion(), which
        # suppresses the autonomous realign for MOTION_USER_STEERING_SECS — so
        # "turn to face me" used to stop the one behavior that would have faced you.
        # This turn points him AT the person, which is exactly what realign wants to
        # finish; the same reasoning already exempts the "I'm behind you" turns.
        self._face_at(0.9)
        MA._state["user_motion_at"] = 0.0
        MA.face_requester(1)
        self.assertEqual(float(MA._state.get("user_motion_at") or 0.0), 0.0)

    def test_it_lifts_a_standing_hold_the_way_come_here_does(self):
        self._face_at(0.9)
        MA.note_user_hold("test")
        deg, reason = MA.face_requester(1)
        self.assertEqual(reason, "turned")


class InteractionHandlerTest(unittest.TestCase):
    """The spoken half: which line, and whether the audit may call it a maneuver."""

    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        self._stack = []
        for patcher in (
            mock.patch.object(interaction.motion_controller, "available",
                              return_value=True),
            mock.patch.object(interaction.motion_controller, "charging",
                              return_value=False),
            mock.patch("intelligence.motion_agency.no_drive_room", return_value=None),
        ):
            self._stack.append(patcher)
            patcher.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for patcher in reversed(self._stack):
            try:
                patcher.stop()
            except Exception:
                pass

    def _with(self, result):
        return mock.patch("intelligence.motion_agency.face_requester",
                          return_value=result)

    def test_a_turn_speaks_and_counts_as_a_maneuver(self):
        with self._with((-35.0, "turned")):
            line, turned = self.interaction._handle_face_requester(1)
        self.assertIn(line, config.MOTION_FACE_ME_TURNING_LINES)
        self.assertTrue(turned)

    def test_every_refusal_speaks_but_is_not_a_maneuver(self):
        for reason, pool in (
            ("already_facing", config.MOTION_FACE_ME_ALREADY_LINES),
            ("ambiguous", config.MOTION_FACE_ME_AMBIGUOUS_LINES),
            ("no_bearing", config.MOTION_FACE_ME_NO_BEARING_LINES),
        ):
            with self._with((None, reason)):
                line, turned = self.interaction._handle_face_requester(1)
            self.assertIn(line, pool, reason)
            self.assertFalse(turned, reason)

    def test_internal_states_say_nothing_at_all(self):
        # busy / traction / suppressed / disabled are not the human's business —
        # the turn falls through to conversation rather than narrating them.
        for reason in ("busy", "traction", "suppressed", "disabled"):
            with self._with((None, reason)):
                line, turned = self.interaction._handle_face_requester(1)
            self.assertIsNone(line, reason)
            self.assertFalse(turned, reason)

    def test_charging_locks_the_wheels_before_any_sensor_is_read(self):
        with mock.patch.object(self.interaction.motion_controller, "charging",
                               return_value=True), \
                mock.patch("intelligence.motion_agency.face_requester") as face:
            line, turned = self.interaction._handle_face_requester(1)
        face.assert_not_called()
        self.assertEqual(line, "I'm plugged in and charging. Wheels stay locked.")
        self.assertFalse(turned)

    def test_a_no_drive_room_declines_with_the_way_out_stated(self):
        with mock.patch("intelligence.motion_agency.no_drive_room",
                        return_value=("workshop", "carpet")), \
                mock.patch("intelligence.motion_agency.face_requester") as face:
            line, turned = self.interaction._handle_face_requester(1)
        face.assert_not_called()
        self.assertIn("not to drive in the workshop", line)
        self.assertIn("you can drive in here", line)

    def test_no_base_is_a_no_op_here(self):
        # The dispatcher speaks the no-base denial; the executor must not pretend.
        with mock.patch.object(self.interaction.motion_controller, "available",
                               return_value=False):
            self.assertEqual(self.interaction._handle_face_requester(1), (None, False))


class DispatchTest(unittest.TestCase):
    """The organic path: the reply call choosing motion_face."""

    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        self.spoken: list[str] = []
        interaction._tool_routed_path.clear()
        self._stack = []
        for patcher in (
            mock.patch.object(interaction, "_speak_blocking",
                              side_effect=lambda line, **kw: self.spoken.append(line)),
            mock.patch.object(interaction.motion_controller, "available",
                              return_value=True),
            mock.patch.object(interaction.motion_controller, "charging",
                              return_value=False),
            mock.patch("intelligence.motion_agency.no_drive_room", return_value=None),
            mock.patch.object(interaction, "_turn_transcript_trusted",
                              return_value=True),
            mock.patch.object(interaction.llm, "get_response",
                              return_value="fallback words"),
            mock.patch("intelligence.motion_agency.face_requester",
                       return_value=(-35.0, "turned")),
        ):
            self._stack.append(patcher)
            patcher.start()
        self.addCleanup(self._stop)

    def _stop(self):
        for patcher in reversed(self._stack):
            try:
                patcher.stop()
            except Exception:
                pass

    def _route(self, text):
        return self.interaction._execute_tool_routed_action(
            "motion.face", {}, text, 1)

    def test_a_face_command_turns_and_stamps_the_path(self):
        resp = self._route("turn to face me")
        self.assertIn(resp, config.MOTION_FACE_ME_TURNING_LINES)
        self.assertEqual(self.interaction._consume_tool_routed_path(),
                         "tool_router.motion.face")

    def test_the_evidence_gate_refuses_every_idiom(self):
        for text in FIGURATIVE_DECOYS:
            resp = self._route(text)
            self.assertEqual(resp, "fallback words", text)
            self.assertIsNone(self.interaction._consume_tool_routed_path())

    def test_a_shaky_transcript_declines_to_conversation(self):
        with mock.patch.object(self.interaction, "_turn_transcript_trusted",
                               return_value=False):
            self.assertEqual(self._route("turn to face me"), "fallback words")

    def test_the_allowlist_is_a_real_kill_switch(self):
        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS) - {"motion.face"}
        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", allowed):
            self.assertEqual(self._route("turn to face me"), "fallback words")

    def test_no_base_says_so_out_loud(self):
        with mock.patch.object(self.interaction.motion_controller, "available",
                               return_value=False):
            resp = self._route("turn to face me")
        self.assertIn(resp, config.MOTION_NO_BASE_DENIAL_LINES)


if __name__ == "__main__":
    unittest.main()
