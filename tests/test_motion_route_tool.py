"""LLM-planned drive routes — docs/motion_route_tool_plan.md.

The regex sequence parser is TRI-STATE: [] = not a route, 2+ decisions = a parsed
route, None = "route-shaped and I couldn't parse it", which spoke "I couldn't
safely parse that whole route" and moved nothing. These tests pin the machinery
that replaces that third arm, and — just as importantly — pin the things that must
NOT change: the other two arms, the regex fast lane's confidence bar, and every
physical gate between a model's opinion and the wheels.

Run per module (never `unittest discover` — see CLAUDE.md):

    venv/bin/python -m unittest tests.test_motion_route_tool
"""

import unittest
from unittest import mock

import config
from intelligence import action_router as ar
from intelligence import motion_route, motion_sequence, tool_router


# ── the field corpus ────────────────────────────────────────────────────────────
# Every utterance that ever reached the tri-state None arm and still does, mined
# from all 572 logs in logs/ (final_executed_path=fast_local_takeover.motion.
# sequence_rejected) and re-run through the CURRENT classifier on 2026-08-22.
# Seven real commands the regex cannot express, and five pieces of ASR debris that
# must keep being refused. This is the rescue path's actual job, not a hypothetical.
FIELD_UNPARSED_ROUTES = (
    "Rex, turn to your right and move forward 10 feet",
    "Move right a little bit, then forward two feet, then turn around.",
    "Turn, turn to your left.",
    "Three feet, and turn a little bit right.",
    "Turn to your left a little bit, and then tell me what you see.",
    "Turn right a little. Alright, it's got two sides.",
    "Turn, never mind, move forward, four feet.",
)
FIELD_UNPARSED_DEBRIS = (
    "I'm hungry. Oh, this is nice back here. I made this up like this, huh? Yeah. "
    "Put the towels down, too. Oh, um, no, um, I bought all this stuff. Remember I "
    "just recently moved in? He redid the flooring and everything. Oh, yeah. Feel "
    "free to look around the room.",
    "and the other, " * 13 + "and the other,",
    "What? What? What? Walk with somebody with your hands on and go back, go and "
    "chin to chin with him? No, this is a servant's pose.",
    "Oh, you went a little far, but I'm over here, turn right, find the black guy.",
)
# The figurative-motion decoys the Phase-3 gate work measured (docs/tool_router_
# scope.md §3). None of these is a drive command; none may execute by any path.
FIGURATIVE_DECOYS = (
    "I think we should move on from that topic", "let's move on",
    "moving forward, I want to try something", "I need to run to the store",
    "she moved forward with the plan", "can you back me up on this",
    "go right ahead and tell him", "come to think of it, that's wrong",
    "my head is spinning", "I'm going to head out soon",
    "we should do a lap sometime", "the meeting turned into a disaster",
    "let's roll", "roll with it", "come on, go on, catch",
)


def _steps(*items):
    return {"steps": list(items)}


class TranslatorArgContractTest(unittest.TestCase):
    """The keys the schema promises are the keys the executors read.

    Arg-name drift is this migration's most-repeated bug and it always fails
    SILENTLY: `degrees` where the executor reads `deg` (a commanded angle becomes
    the default 90), `distance` where it reads `dist_m` (0.30 m nudge), arc's lone
    `direction` where it reads ang_dir/lin_dir (every arc curves forward-LEFT), and
    the enum value "backward" where the executor tests == "back" and otherwise
    falls through to move_forward — "back up" driving him INTO the requester.
    """

    def test_schema_step_keys_are_the_executor_keys(self):
        schema = tool_router.tool_schema_for("motion.route")
        self.assertIsNotNone(schema)
        item = schema["function"]["parameters"]["properties"]["steps"]["items"]
        self.assertEqual(
            set(item["properties"]),
            {"op", "direction", "deg", "dist_m", "ang_dir", "lin_dir", "small", "pace"})
        self.assertEqual(item["properties"]["op"]["enum"], ["turn", "move", "arc"])
        # "back", never "backward" — the shipped enum-value drift.
        self.assertIn("back", item["properties"]["direction"]["enum"])
        self.assertNotIn("backward", item["properties"]["direction"]["enum"])
        self.assertEqual(item["properties"]["ang_dir"]["enum"], ["left", "right"])
        self.assertEqual(item["properties"]["lin_dir"]["enum"], ["forward", "back"])
        self.assertIn("DEGREES", item["properties"]["deg"]["description"])
        self.assertIn("METRES", item["properties"]["dist_m"]["description"])
        for req in item["required"]:
            self.assertIn(req, item["properties"])

    def test_schema_has_no_target_field(self):
        # Plan §9: target-relative motion ("face the window", "go to the couch")
        # needs a bearing source Rex does not have. With no field for it the model
        # cannot claim one.
        schema = tool_router.tool_schema_for("motion.route")
        item = schema["function"]["parameters"]["properties"]["steps"]["items"]
        for banned in ("target", "object", "place", "landmark", "toward"):
            self.assertNotIn(banned, item["properties"])

    def test_translated_args_are_exactly_what_the_executors_read(self):
        decisions, reason = ar.route_tool_to_decisions(_steps(
            {"op": "turn", "direction": "right", "deg": 45},
            {"op": "move", "direction": "back", "dist_m": 0.6},
            {"op": "arc", "ang_dir": "left", "lin_dir": "back", "small": True},
        ))
        self.assertIsNone(reason)
        self.assertEqual([d.action for d in decisions],
                         ["motion.turn", "motion.move", "motion.arc"])
        self.assertEqual(decisions[0].args, {"direction": "right", "deg": 45.0})
        self.assertEqual(decisions[1].args, {"direction": "back", "dist_m": 0.6})
        self.assertEqual(decisions[2].args,
                         {"ang_dir": "left", "lin_dir": "back", "small": True})

    def test_backward_is_normalised_to_back(self):
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "move", "direction": "backward", "dist_m": 0.4}))
        self.assertEqual(decisions[0].args["direction"], "back")

    def test_signed_magnitudes_are_consumed_not_forwarded(self):
        # The plan sketched signed numbers; neither executor reads a sign. They are
        # accepted as a fallback for a missing direction word and turned into one.
        decisions, _ = ar.route_tool_to_decisions(_steps(
            {"op": "move", "dist_m": -0.5}, {"op": "turn", "deg": -90}))
        self.assertEqual(decisions[0].args, {"direction": "back", "dist_m": 0.5})
        self.assertEqual(decisions[1].args, {"direction": "right", "deg": 90.0})

    def test_an_unrecognised_direction_refuses_instead_of_defaulting(self):
        # The step schema carries ONE shared direction enum across turn, move and
        # arc, so "right" on a move step is schema-legal — and the corpus utterance
        # this feature was built for is literally "Move right a little bit, then
        # forward two feet, then turn around." The signed fallback used to catch it
        # and drive a metre STRAIGHT FORWARD, at whatever was in front of him. The
        # base cannot strafe; sideways is an arc, and picking one of two readings for
        # the model is the silent substitution this translator exists to refuse.
        for step, reason in (
            ({"op": "move", "direction": "right", "dist_m": 1.0},
             "route_move_bad_direction"),
            ({"op": "move", "direction": "left", "dist_m": 1.0},
             "route_move_bad_direction"),
            ({"op": "move", "direction": "around", "dist_m": 1.0},
             "route_move_bad_direction"),
            ({"op": "move", "direction": "north", "dist_m": 1.0},
             "route_move_bad_direction"),
            ({"op": "turn", "direction": "forward", "deg": 90},
             "route_turn_bad_direction"),
            ({"op": "turn", "direction": "sideways", "deg": 90},
             "route_turn_bad_direction"),
            ({"op": "arc", "ang_dir": "sideways"}, "route_arc_bad_direction"),
        ):
            decisions, got = ar.route_tool_to_decisions(_steps(step))
            self.assertIsNone(decisions, step)
            self.assertEqual(got, reason, step)

    def test_a_prefix_test_would_read_reverse_as_right(self):
        # Direction words are matched as exact members of a set, not by first letter.
        # "reverse" starts with r; on a turn step a prefix test makes it a RIGHT turn.
        decisions, reason = ar.route_tool_to_decisions(
            _steps({"op": "turn", "direction": "reverse", "deg": 90}))
        self.assertIsNone(decisions)
        self.assertEqual(reason, "route_turn_bad_direction")
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "move", "direction": "reverse", "dist_m": 0.5}))
        self.assertEqual(decisions[0].args["direction"], "back")

    def test_a_turn_cannot_be_backward_so_back_is_an_about_face(self):
        # The shared enum's own description says "'back', never 'backward'" — which
        # is advice for MOVE steps, so the model will write "back" on a turn too.
        # There is no other reading of a backward turn, so it is an alias, not a
        # guess; on a move step the same word means the opposite thing, which is why
        # the two ladders never share a vocabulary.
        for word in ("back", "backward", "backwards", "u-turn", "180"):
            decisions, _ = ar.route_tool_to_decisions(
                _steps({"op": "turn", "direction": word}))
            self.assertEqual(decisions[0].args["direction"], "around", word)

    def test_an_aliased_op_keeps_the_direction_it_names(self):
        # The op normalizer accepts single-verb TOOL names because a model that
        # pattern-matched motion_move/motion_turn will write them — but it used to
        # keep only the verb, so "move_back" arrived with no direction and the
        # positive-magnitude default drove a metre FORWARD, into whoever asked him to
        # back away. That is verbatim the failure the module header says it prevents.
        for step, expected in (
            ({"op": "move_back", "dist_m": 1.0}, "back"),
            ({"op": "turn_right", "deg": 90}, "right"),
            ({"op": "turn-left", "deg": 90}, "left"),
            ({"op": "motion.turn", "direction": "right", "deg": 45}, "right"),
            ({"op": "move_forward", "dist_m": 0.5}, "forward"),
        ):
            decisions, reason = ar.route_tool_to_decisions(_steps(step))
            self.assertIsNone(reason, step)
            self.assertEqual(decisions[0].args["direction"], expected, step)

    def test_an_arc_reads_the_shared_direction_key_by_value(self):
        # An arc carries BOTH senses and the shared enum can hold either. Read by
        # position, a "back" written into `direction` was consumed as the lateral
        # word, dropped for not starting with "r", and replaced with a forward-LEFT
        # curve — the single-verb arc schema has no `direction` field at all for
        # exactly this reason.
        decisions, reason = ar.route_tool_to_decisions(
            _steps({"op": "arc", "ang_dir": "left", "direction": "back"}))
        self.assertIsNone(reason)
        self.assertEqual(decisions[0].args,
                         {"ang_dir": "left", "lin_dir": "back", "small": False})
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "arc", "direction": "right"}))
        self.assertEqual(decisions[0].args["ang_dir"], "right")
        # ...and a linear-only arc still has no side, so it refuses rather than
        # inventing one.
        decisions, reason = ar.route_tool_to_decisions(
            _steps({"op": "arc", "direction": "back"}))
        self.assertIsNone(decisions)
        self.assertEqual(reason, "route_arc_without_direction")

    def test_omitted_magnitudes_take_the_config_defaults(self):
        decisions, _ = ar.route_tool_to_decisions(_steps(
            {"op": "turn", "direction": "left"}, {"op": "move", "direction": "forward"}))
        self.assertEqual(decisions[0].args["deg"], float(config.MOTION_DEFAULT_TURN_DEG))
        self.assertEqual(decisions[1].args["dist_m"],
                         float(config.MOTION_DEFAULT_MOVE_DIST_M))

    def test_around_defaults_to_a_180(self):
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "turn", "direction": "around"}))
        self.assertEqual(decisions[0].args, {"direction": "around", "deg": 180.0})

    def test_pace_slow_maps_onto_the_wire_rate_and_speed(self):
        decisions, _ = ar.route_tool_to_decisions(_steps(
            {"op": "turn", "direction": "left", "deg": 90, "pace": "slow"},
            {"op": "move", "direction": "forward", "dist_m": 0.5, "pace": "slow"},
            {"op": "move", "direction": "forward", "dist_m": 0.5, "pace": "normal"},
        ))
        scale = float(config.MOTION_ROUTE_SLOW_PACE_SCALE)
        self.assertAlmostEqual(decisions[0].args["rate"],
                               float(config.MOTION_DEFAULT_TURN_RATE) * scale)
        self.assertAlmostEqual(decisions[1].args["speed"],
                               float(config.MOTION_MAX_LINEAR_MS) * scale)
        self.assertNotIn("speed", decisions[2].args)


class TranslatorClampTest(unittest.TestCase):
    """The clamps are the whole safety story for magnitudes: a route carries its
    own numbers because it cannot re-read them out of the human's words."""

    def _refused(self, args):
        decisions, reason = ar.route_tool_to_decisions(args)
        self.assertIsNone(decisions)
        self.assertTrue(reason)
        return reason

    def test_one_bad_step_refuses_the_whole_route(self):
        # Tri-state discipline: "turn left then sing" must not turn left. A step the
        # route cannot express refuses the WHOLE thing rather than dropping a leg and
        # leaving the base somewhere nobody asked for.
        reason = self._refused(_steps(
            {"op": "move", "direction": "forward", "dist_m": 0.3},
            {"op": "sing"},
            {"op": "turn", "direction": "left", "deg": 90},
        ))
        self.assertEqual(reason, "route_step_unknown_op")
        # ...and a placeholder zero mid-route, which would otherwise reach the
        # executor's `float(args.get("dist_m") or DEFAULT)` and roll 0.30 m.
        self.assertEqual(
            self._refused(_steps({"op": "turn", "direction": "left", "deg": 90},
                                 {"op": "move", "direction": "forward", "dist_m": 0})),
            "route_move_too_small")

    def test_an_oversized_leg_is_clamped_not_refused(self):
        # A shortened leg keeps every step and every direction the human asked for —
        # it is not the dropped leg §4.2 forbids. Same numbers and same treatment as
        # the single-verb tool path already ships (_motion_args_from_tool).
        decisions, reason = ar.route_tool_to_decisions(_steps(
            {"op": "move", "direction": "forward",
             "dist_m": config.MOTION_ROUTE_MAX_STEP_M + 5.0},
            {"op": "turn", "direction": "left",
             "deg": config.MOTION_ROUTE_MAX_STEP_DEG + 400}))
        self.assertIsNone(reason)
        self.assertEqual(decisions[0].args["dist_m"],
                         float(config.MOTION_ROUTE_MAX_STEP_M))
        self.assertEqual(decisions[1].args["deg"],
                         float(config.MOTION_ROUTE_MAX_STEP_DEG))

    def test_a_ten_foot_leg_survives_the_clamp(self):
        # The most common real unparsed route on record: "Rex, turn to your right and
        # move forward 10 feet". If the ceiling refuses this, the feature ships and
        # the owner still hears the denial.
        decisions, reason = ar.route_tool_to_decisions(_steps(
            {"op": "turn", "direction": "right", "deg": 90},
            {"op": "move", "direction": "forward", "dist_m": 3.048}))
        self.assertIsNone(reason)
        # Clamped to the one-room ceiling (a 5 cm shortfall on 10 feet), NOT refused —
        # the same thing "move forward 10 feet" already gets as a single command.
        self.assertEqual(decisions[1].args["dist_m"],
                         float(config.MOTION_ROUTE_MAX_STEP_M))
        self.assertGreater(decisions[1].args["dist_m"], 3.0 - 0.01)

    def test_total_distance_and_rotation_budgets_refuse(self):
        leg = {"op": "move", "direction": "forward",
               "dist_m": config.MOTION_ROUTE_MAX_STEP_M}
        n = int(config.MOTION_ROUTE_MAX_TOTAL_M // config.MOTION_ROUTE_MAX_STEP_M) + 1
        self.assertEqual(self._refused(_steps(*([leg] * n))),
                         "route_over_total_distance_cap")
        spin = {"op": "turn", "direction": "left", "deg": 360}
        n = int(config.MOTION_ROUTE_MAX_TOTAL_DEG // 360) + 1
        self.assertEqual(self._refused(_steps(*([spin] * n))),
                         "route_over_total_rotation_cap")

    def test_step_count_cap(self):
        step = {"op": "turn", "direction": "left", "deg": 20}
        self.assertEqual(
            self._refused(_steps(*([step] * (int(config.MOTION_ROUTE_MAX_STEPS) + 1)))),
            "route_too_many_steps")

    def test_zero_magnitude_refuses_rather_than_becoming_the_default(self):
        # Both executors read `float(args.get("deg") or DEFAULT)`. `or` — so a
        # placeholder zero is a 90 degree turn and a 0.30 m roll. It must not reach
        # them at all.
        self.assertEqual(
            self._refused(_steps({"op": "move", "direction": "forward", "dist_m": 0})),
            "route_move_too_small")
        self.assertEqual(
            self._refused(_steps({"op": "turn", "direction": "left", "deg": 0.0})),
            "route_turn_too_small")

    def test_garbage_and_missing_args(self):
        self.assertEqual(self._refused(None), "route_args_not_an_object")
        self.assertEqual(self._refused("turn left"), "route_args_not_an_object")
        self.assertEqual(self._refused({}), "route_has_no_steps")
        self.assertEqual(self._refused(_steps()), "route_has_no_steps")
        self.assertEqual(self._refused(_steps("turn left")), "route_step_not_an_object")
        self.assertEqual(self._refused(_steps({"op": "sing"})), "route_step_unknown_op")
        self.assertEqual(self._refused(_steps({"op": "arc"})),
                         "route_arc_without_direction")
        self.assertEqual(self._refused(_steps({"op": "move"})),
                         "route_move_without_direction")
        # NaN/inf are unreadable, not zero: the step falls back to the config
        # default nudge exactly as an omitted magnitude does, rather than reaching
        # float() and blowing up in the executor.
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "move", "direction": "forward", "dist_m": float("nan")}))
        self.assertEqual(decisions[0].args["dist_m"],
                         float(config.MOTION_DEFAULT_MOVE_DIST_M))

    def test_come_is_not_a_route_step(self):
        # Plan §11: come seizes the requester-errand machinery. A route containing it
        # refuses so the errand can own the turn.
        self.assertEqual(self._refused(_steps({"op": "come"})), "route_step_unknown_op")

    def test_clamps_read_config_at_call_time(self):
        with mock.patch.object(config, "MOTION_ROUTE_MAX_STEP_M", 0.5, create=True):
            decisions, _ = ar.route_tool_to_decisions(
                _steps({"op": "move", "direction": "forward", "dist_m": 1.0}))
            self.assertEqual(decisions[0].args["dist_m"], 0.5)
        with mock.patch.object(config, "MOTION_ROUTE_MAX_TOTAL_M", 0.4, create=True):
            self.assertEqual(
                self._refused(_steps({"op": "move", "direction": "forward",
                                      "dist_m": 1.0})),
                "route_over_total_distance_cap")


class CatalogWiringTest(unittest.TestCase):
    """The six wiring points, each of which fails OPEN into conversation if missed."""

    def test_spec_is_executable_motion(self):
        spec = {s.key: s for s in ar.ACTION_SPECS}["motion.route"]
        self.assertEqual(spec.category, "motion")
        self.assertTrue(spec.executable)
        self.assertIn("motion.route", ar.EXECUTABLE_ACTIONS)

    def test_allowlist_carries_the_key(self):
        self.assertIn("motion.route", config.ACTION_ROUTER_EXECUTE_ACTIONS)

    def test_evidence_gate_covers_the_route_key(self):
        # Without this, missing_required_evidence_reason's bare `return None` tail
        # gives an unrecognised action NO evidence requirement at all.
        self.assertIn("motion.route", ar._MOTION_TOOL_ACTIONS)
        decision = ar.ActionDecision(action="motion.route", confidence=1.0,
                                     args={}, reason="test")
        self.assertEqual(
            ar.missing_required_evidence_reason("let's move on", decision),
            "missing_motion_command_evidence")
        self.assertIsNone(ar.missing_required_evidence_reason(
            "go forward a bit then turn around", decision))

    def test_route_is_not_dispatched_as_a_single_motion_action(self):
        # _MOTION_ACTIONS dispatches to _handle_router_motion_action, which has no
        # route arm and returns None — Rex would answer with a classic reply and not
        # move. The route needs its own branch, so it must stay OUT of that set.
        from intelligence import interaction
        self.assertNotIn("motion.route", interaction._MOTION_ACTIONS)
        self.assertIn("motion.route", interaction._MOTION_DRIVE_ACTIONS)

    def test_motion_stays_out_of_the_owned_actions_demotion(self):
        self.assertEqual(
            [a for a in ar.TOOL_ROUTER_OWNED_ACTIONS if a.startswith("motion")], [])

    def test_organic_path_is_phase_gated_both_ways(self):
        with mock.patch.object(config, "MOTION_ROUTE_ORGANIC_ENABLED", False,
                               create=True):
            self.assertNotIn("motion.route", tool_router.live_actions())
            self.assertIsNone(tool_router.resolve_tool_call("motion_route", "{}"))
        with mock.patch.object(config, "MOTION_ROUTE_ORGANIC_ENABLED", True,
                               create=True):
            self.assertIn("motion.route", tool_router.live_actions())
            self.assertEqual(
                tool_router.resolve_tool_call("motion_route", '{"steps": []}'),
                ("motion.route", {"steps": []}))
            names = {t["function"]["name"] for t in tool_router.live_reply_tools()}
            self.assertIn("motion_route", names)
        with mock.patch.object(config, "MOTION_ROUTE_ENABLED", False, create=True), \
                mock.patch.object(config, "MOTION_ROUTE_ORGANIC_ENABLED", True,
                                  create=True):
            # The master switch outranks the phase flag.
            self.assertNotIn("motion.route", tool_router.live_actions())


class TriStateUnchangedTest(unittest.TestCase):
    """The two arms this feature must not touch. The regex keeps the first claim."""

    def test_non_sequence_utterances_still_return_empty(self):
        for text in ("and then,", "and move backwards", "yeah that sounds great, thanks",
                     "Turn left, 15 degrees.", "let's move on"):
            self.assertEqual(ar.classify_explicit_motion_sequence(text), [], text)

    def test_parseable_routes_still_parse(self):
        seq = ar.classify_explicit_motion_sequence("turn left then move forward five feet")
        self.assertEqual([d.action for d in seq], ["motion.turn", "motion.move"])
        self.assertEqual(ar.classify_explicit_motion_sequence("turn left then sing"), None)

    def test_the_field_corpus_still_lands_on_the_rescue_arm(self):
        for text in FIELD_UNPARSED_ROUTES + FIELD_UNPARSED_DEBRIS:
            self.assertIsNone(ar.classify_explicit_motion_sequence(text, max_steps=8),
                              text)

    def test_figurative_decoys_never_reach_the_rescue_arm(self):
        # Plan §7 metric: the decoy false-fire rate must be ~0. These turns never
        # asked for motion, so they must not even become route-SHAPED.
        for text in FIGURATIVE_DECOYS:
            self.assertNotEqual(
                ar.classify_explicit_motion_sequence(text, max_steps=8), None, text)
            self.assertEqual(
                ar.motion_command_refusal_reason(text, "motion.route"),
                "missing_motion_command_evidence", text)


class InterpreterContractTest(unittest.TestCase):
    """The prose-wins regression (docs/tool_router_scope.md, Phase 2 carve-out).

    Four of eight unambiguous impersonation imperatives got NO tool call from the
    persona reply call — the model performed in prose instead — while the shadow
    router returned the right answer every time. The rescue interpreter is built so
    that cannot happen: no character in the prompt, exactly two tools, forced choice,
    and no conversation history to read a stale argument out of.
    """

    def _capture(self, tool_calls=None, raise_exc=None):
        captured = {}

        def fake_create(client, **kwargs):
            captured.update(kwargs)
            if raise_exc is not None:
                raise raise_exc
            msg = mock.Mock()
            msg.tool_calls = tool_calls or []
            return mock.Mock(choices=[mock.Mock(message=msg)])

        return captured, fake_create

    def _call(self, name, arguments):
        call = mock.Mock()
        call.function = mock.Mock()
        call.function.name = name
        call.function.arguments = arguments
        return call

    def test_call_is_forced_two_tool_and_persona_free(self):
        captured, fake_create = self._capture([
            self._call("motion_route",
                       '{"steps": [{"op": "turn", "direction": "left", "deg": 90}, '
                       '{"op": "move", "direction": "forward", "dist_m": 0.5}]}')])
        with mock.patch("intelligence.llm_compat.create", fake_create):
            result = motion_route.interpret("turn left and roll forward half a metre")
        self.assertEqual(captured["extra"]["tool_choice"], "required")
        names = {t["function"]["name"] for t in captured["extra"]["tools"]}
        self.assertEqual(names, {"motion_route", "motion_route_decline"})
        # No persona, and no conversation history: exactly one system turn and the
        # bare utterance (plan §11 — stale context is the target='speaker' bug).
        messages = captured["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        system = messages[0]["content"].lower()
        for persona_word in ("dj r3x", "rex", "droid", "character", "funny", "voice"):
            self.assertNotIn(persona_word, system)
        self.assertIn("metres", system)
        self.assertIn("degrees", system)
        self.assertEqual(messages[1]["content"], "turn left and roll forward half a metre")
        self.assertEqual(len(result["args"]["steps"]), 2)
        self.assertFalse(result["declined"])

    def test_the_rescue_schema_reframes_without_touching_the_step_shape(self):
        # The step shape is where arg-name drift bites, so it must be the shared
        # definition verbatim. Only the ROUTE-level framing differs: the shared
        # description sends a single movement to motion_turn/move/arc, which is right
        # on the reply call and wrong on a call where this tool and a decline are the
        # whole surface. Measured 2026-08-22: with the unmodified schema, 7 of the 13
        # real None-arm utterances were declined in as many words ("single turn
        # command, not a multi-step route") — every one a real command that would
        # have drawn the denial this call exists to delete.
        shared = tool_router.tool_schema_for("motion.route")
        rescue = motion_route._rescue_schema(shared)
        self.assertEqual(
            rescue["function"]["parameters"]["properties"]["steps"]["items"],
            shared["function"]["parameters"]["properties"]["steps"]["items"])
        self.assertEqual(
            rescue["function"]["parameters"]["properties"]["steps"]["minItems"], 1)
        self.assertEqual(shared["function"]["parameters"]["properties"]["steps"]
                         ["minItems"], 2, "the shared schema must not be mutated")
        self.assertNotIn("motion_turn", rescue["function"]["description"])
        self.assertIsNone(motion_route._rescue_schema(None))

    def test_stated_limits_track_config(self):
        captured, fake_create = self._capture([])
        with mock.patch.object(config, "MOTION_ROUTE_MAX_STEPS", 4, create=True), \
                mock.patch("intelligence.llm_compat.create", fake_create):
            motion_route.interpret("go forward then turn")
        self.assertIn("at most 4 steps", captured["messages"][0]["content"])

    def test_decline_tool_is_a_decline_not_a_route(self):
        _, fake_create = self._capture([
            self._call("motion_route_decline", '{"reason": "ASR debris"}')])
        with mock.patch("intelligence.llm_compat.create", fake_create):
            result = motion_route.interpret("and the other, and the other,")
        self.assertTrue(result["declined"])
        self.assertIsNone(result["args"])
        self.assertEqual(result["reason"], "ASR debris")

    def test_prose_answer_is_read_as_a_decline(self):
        _, fake_create = self._capture([])
        with mock.patch("intelligence.llm_compat.create", fake_create):
            result = motion_route.interpret("we should do a lap sometime")
        self.assertTrue(result["declined"])
        self.assertIsNone(result["args"])

    def test_failures_come_back_in_the_dict_and_never_raise(self):
        # The caller sits inside a ladder whose except-handler logs at DEBUG under an
        # unrelated message and drops the turn to conversation — an exception escaping
        # here would turn a network blip into a silently mis-routed command.
        _, fake_create = self._capture(raise_exc=RuntimeError("link down"))
        with mock.patch("intelligence.llm_compat.create", fake_create):
            result = motion_route.interpret("turn left then go forward")
        self.assertIsNone(result["args"])
        self.assertIn("RuntimeError", result["error"])

    def test_unparseable_tool_arguments_are_an_error_not_a_route(self):
        _, fake_create = self._capture([self._call("motion_route", "{not json")])
        with mock.patch("intelligence.llm_compat.create", fake_create):
            result = motion_route.interpret("turn left then go forward")
        self.assertIsNone(result["args"])
        self.assertTrue(result["error"])

    def test_forced_choice_degrades_to_auto_when_the_sdk_refuses(self):
        seen = []

        def fake_create(client, **kwargs):
            seen.append(kwargs["extra"]["tool_choice"])
            if len(seen) == 1:
                raise TypeError("unexpected keyword argument 'tool_choice'")
            msg = mock.Mock()
            msg.tool_calls = []
            return mock.Mock(choices=[mock.Mock(message=msg)])

        with mock.patch("intelligence.llm_compat.create", fake_create):
            motion_route.interpret("turn left then go forward")
        self.assertEqual(seen, ["required", "auto"])

    def test_offline_never_reaches_the_model(self):
        # Plan §4.4: the local reply model gets no tool surface, so with the link
        # down the deterministic classifiers stay the whole story.
        with mock.patch("intelligence.connectivity.is_offline", return_value=True):
            self.assertFalse(motion_route.available())
        with mock.patch("intelligence.connectivity.is_offline", return_value=False):
            self.assertTrue(motion_route.available())
            with mock.patch.object(config, "MOTION_ROUTE_ENABLED", False, create=True):
                self.assertFalse(motion_route.available())


class SpinClearanceTest(unittest.TestCase):
    """A near-full spin is all-or-nothing (plan §4.3.4).

    The drive axle sits aft of the ring centre and the arms reach past it, so a spin
    in place sweeps a wide arc — the bookshelf hand-loss incidents. motion_swing
    SHRINKS a blocked turn, which is right for a 90 and wrong for a 360: the point
    of a spin is ending where you started, and the next leg drives off the heading a
    shrunk spin leaves behind.
    """

    TIGHT = {"rl": 300, "rr": 300, "lb": 300, "rb": 300,
             "fl": 3000, "fr": 3000, "lf": 3000, "rf": 3000}
    OPEN = {k: 4000 for k in ("fl", "fr", "rl", "rr", "lf", "lb", "rf", "rb")}

    def _spin(self, deg=360.0):
        return [ar.ActionDecision(action="motion.turn", confidence=1.0,
                                  args={"direction": "left", "deg": deg},
                                  reason="test")]

    def _with_tof(self, tof):
        return mock.patch.object(motion_sequence.motion, "telemetry",
                                 return_value={"tof_mm": tof})

    def test_full_spin_refused_when_the_sweep_is_cramped(self):
        with self._with_tof(self.TIGHT):
            self.assertEqual(motion_sequence.spin_clearance_reason(self._spin()),
                             "spin_no_elbow_room")

    def test_full_spin_allowed_with_room(self):
        with self._with_tof(self.OPEN):
            self.assertIsNone(motion_sequence.spin_clearance_reason(self._spin()))

    def test_ordinary_turns_are_not_checked(self):
        # Below the floor a shrink is the right degradation and this must not fire.
        with self._with_tof(self.TIGHT):
            self.assertIsNone(motion_sequence.spin_clearance_reason(self._spin(90.0)))

    def test_unknown_sensing_is_not_a_refusal(self):
        with mock.patch.object(motion_sequence.motion, "telemetry", return_value={}):
            self.assertIsNone(motion_sequence.spin_clearance_reason(self._spin()))

    def test_flag_disables_the_check(self):
        with self._with_tof(self.TIGHT), \
                mock.patch.object(config, "MOTION_ROUTE_SPIN_CHECK_ENABLED", False,
                                  create=True):
            self.assertIsNone(motion_sequence.spin_clearance_reason(self._spin()))


class SequencePaceTest(unittest.TestCase):
    """`pace: slow` has to reach the wire, and the step deadline has to follow it.

    A half-speed leg measured against the full-speed default times out mid-move, and
    a timed-out step aborts the whole remainder — "drive that slowly" would have read
    as "drive the first leg and give up"."""

    def test_paced_steps_use_the_signed_primitives_with_rate_and_speed(self):
        calls = {}
        with mock.patch.object(motion_sequence.motion_controller, "turn",
                               side_effect=lambda d, rate=None, **kw: calls.setdefault(
                                   "turn", (d, rate)) or 1) as _t, \
                mock.patch.object(motion_sequence.motion_controller, "move",
                                  side_effect=lambda d, speed=None: calls.setdefault(
                                      "move", (d, speed)) or 2):
            motion_sequence._issue(ar.ActionDecision(
                action="motion.turn", confidence=1.0,
                args={"direction": "right", "deg": 90.0, "rate": 30.0}, reason="t"))
            motion_sequence._issue(ar.ActionDecision(
                action="motion.move", confidence=1.0,
                args={"direction": "back", "dist_m": 0.5, "speed": 0.2}, reason="t"))
        self.assertEqual(calls["turn"], (-90.0, 30.0))
        self.assertEqual(calls["move"], (-0.5, 0.2))

    def test_unpaced_steps_still_use_the_voice_verbs(self):
        with mock.patch.object(motion_sequence.motion_controller, "turn_left",
                               return_value=7) as turn_left:
            seq, _ = motion_sequence._issue(ar.ActionDecision(
                action="motion.turn", confidence=1.0,
                args={"direction": "left", "deg": 90.0}, reason="t"))
        turn_left.assert_called_once_with(90.0)
        self.assertEqual(seq, 7)

    def test_a_one_step_route_keeps_the_pace_too(self):
        # A one-step plan is dispatched to the single-verb executor, whose voice
        # verbs (turn_left/move_forward) take no rate — so "back up a metre, slowly",
        # a single-step plan and exactly the phrasing that asks for this, drove at
        # full speed while the identical step inside a two-step route did not.
        from intelligence import interaction
        calls = {}
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "move", "direction": "back", "dist_m": 1.0, "pace": "slow"}))
        with mock.patch.object(interaction.motion_controller, "available",
                               return_value=True), \
                mock.patch.object(interaction.motion_controller, "charging",
                                  return_value=False), \
                mock.patch("intelligence.motion_agency.no_drive_room",
                           return_value=None), \
                mock.patch.object(interaction.motion_controller, "move",
                                  side_effect=lambda d, speed=None: calls.setdefault(
                                      "move", (d, speed)) or 5), \
                mock.patch.object(interaction, "_remember_motion_continuation"), \
                mock.patch.object(interaction.motion_controller, "announce_if_blocked"):
            line, drove = interaction._handle_motion_route(decisions)
        self.assertEqual(line, "Backing up.")
        self.assertTrue(drove)
        self.assertEqual(calls["move"][0], -1.0)
        self.assertAlmostEqual(calls["move"][1],
                               float(config.MOTION_MAX_LINEAR_MS)
                               * float(config.MOTION_ROUTE_SLOW_PACE_SCALE))

    def test_an_unpaced_one_step_route_still_uses_the_voice_verbs(self):
        from intelligence import interaction
        decisions, _ = ar.route_tool_to_decisions(
            _steps({"op": "turn", "direction": "left", "deg": 90}))
        with mock.patch.object(interaction.motion_controller, "available",
                               return_value=True), \
                mock.patch.object(interaction.motion_controller, "charging",
                                  return_value=False), \
                mock.patch("intelligence.motion_agency.no_drive_room",
                           return_value=None), \
                mock.patch.object(interaction.motion_controller, "turn_left",
                                  return_value=9) as turn_left, \
                mock.patch.object(interaction, "_remember_motion_continuation"):
            line, drove = interaction._handle_motion_route(decisions)
        turn_left.assert_called_once_with(90.0)
        self.assertEqual(line, "Turning left.")
        self.assertTrue(drove)

    def test_step_timeout_follows_the_paced_rate(self):
        fast = motion_sequence._step_timeout(ar.ActionDecision(
            action="motion.move", confidence=1.0, args={"dist_m": 3.0}, reason="t"))
        slow = motion_sequence._step_timeout(ar.ActionDecision(
            action="motion.move", confidence=1.0,
            args={"dist_m": 3.0, "speed": 0.05}, reason="t"))
        self.assertGreater(slow, fast)

    def test_junk_pace_values_are_ignored(self):
        self.assertIsNone(motion_sequence._pace_value({"rate": "quick"}, "rate"))
        self.assertIsNone(motion_sequence._pace_value({"rate": 0}, "rate"))
        self.assertIsNone(motion_sequence._pace_value({}, "rate"))


class RescuePathTest(unittest.TestCase):
    """The tri-state None arm, end to end with the interpreter mocked."""

    ROUTE = _steps({"op": "turn", "direction": "right", "deg": 90},
                   {"op": "move", "direction": "forward", "dist_m": 3.048})

    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        self.spoken: list[str] = []
        self.started: list[list] = []
        self.queued: list[str] = []      # ack lines, which ride the speech queue
        self.interpret_at: list[list] = []
        interaction._tool_routed_path.clear()
        self._stack = []
        for patcher in (
            mock.patch.object(interaction, "_speak_blocking",
                              side_effect=lambda line, **kw: self.spoken.append(line)),
            mock.patch.object(interaction.speech_queue, "enqueue",
                              side_effect=lambda line, *a, **kw: self.queued.append(line)),
            mock.patch.object(interaction.speech_queue, "is_speaking",
                              return_value=False),
            mock.patch.object(interaction.output_gate, "is_busy", return_value=False),
            mock.patch("audio.tts.is_cached", return_value=True),
            mock.patch.object(interaction.motion_controller, "available",
                              return_value=True),
            mock.patch.object(interaction.motion_controller, "charging",
                              return_value=False),
            mock.patch.object(interaction, "_start_motion_sequence",
                              side_effect=lambda d: self.started.append(list(d)) or True),
            mock.patch("intelligence.motion_agency.no_drive_room", return_value=None),
            mock.patch.object(motion_sequence, "spin_clearance_reason",
                              return_value=None),
            mock.patch("intelligence.connectivity.is_offline", return_value=False),
            mock.patch.object(interaction, "_turn_transcript_trusted",
                              return_value=True),
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

    def _takeover(self, text, interpret_result):
        # side_effect, not return_value: the mock records WHEN it ran relative to the
        # ack, which is the property the ack exists for and which a return_value mock
        # cannot see.
        def _interpret(_text):
            self.interpret_at.append(list(self.queued))
            return interpret_result

        with mock.patch.object(motion_route, "interpret",
                               side_effect=_interpret) as interp:
            line = self.interaction._explicit_motion_takeover(text)
        return line, interp

    def _ok(self, args=None):
        return {"args": args if args is not None else self.ROUTE, "declined": False,
                "reason": "", "secs": 0.4, "error": None}

    def test_a_rescued_route_drives_and_never_speaks_the_denial(self):
        line, interp = self._takeover(
            "Rex, turn to your right and move forward 10 feet", self._ok())
        interp.assert_called_once()
        self.assertEqual(line, "On it — 2 moves.")
        self.assertNotIn("I couldn't safely parse that whole route.", self.spoken)
        self.assertEqual(
            [(d.action, d.args) for d in self.started[0]],
            [("motion.turn", {"direction": "right", "deg": 90.0}),
             ("motion.move", {"direction": "forward", "dist_m": 3.0})])

    def test_ack_is_queued_before_the_interpreter_runs_and_does_not_block(self):
        # Two properties, both load-bearing and neither pinned by "the ack is first
        # in the spoken list": it must be QUEUED (speech_queue.enqueue, so it plays
        # OVER the ~0.8 s call) rather than spoken blocking (_speak_blocking waits
        # for playback AND an 800-1500 ms post-punchline beat, which would put ~2 s
        # AHEAD of the call it exists to cover), and it must be in the queue by the
        # time the interpreter runs.
        self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(len(self.queued), 1)
        self.assertIn(self.queued[0], config.MOTION_ROUTE_ACK_LINES)
        self.assertNotIn(self.queued[0], self.spoken)
        self.assertEqual(self.interpret_at, [[self.queued[0]]],
                         "the ack must already be queued when the interpreter runs")

    def test_the_ack_stays_quiet_when_something_is_already_speaking(self):
        with mock.patch.object(self.interaction.speech_queue, "is_speaking",
                               return_value=True):
            line, _ = self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(self.queued, [])
        self.assertEqual(line, "On it — 2 moves.")   # the route still runs

    def test_the_ack_stays_quiet_when_its_audio_is_not_cached(self):
        # Fetching TTS at speak time would put back the latency the ack removes.
        with mock.patch("audio.tts.is_cached", return_value=False):
            self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(self.queued, [])

    def test_a_declined_interpretation_falls_back_to_the_denial(self):
        line, _ = self._takeover(
            FIELD_UNPARSED_DEBRIS[1],
            {"args": None, "declined": True, "reason": "chatter", "secs": 0.3,
             "error": None})
        self.assertEqual(line, "I couldn't safely parse that whole route.")
        self.assertEqual(self.started, [])

    def test_a_plan_outside_the_clamps_falls_back_to_the_denial(self):
        line, _ = self._takeover(
            FIELD_UNPARSED_ROUTES[0],
            self._ok(_steps(*([{"op": "move", "direction": "forward",
                                "dist_m": config.MOTION_ROUTE_MAX_STEP_M}] * 5))))
        self.assertEqual(line, "I couldn't safely parse that whole route.")
        self.assertEqual(self.started, [])

    def test_an_interpreter_error_falls_back_to_the_denial(self):
        line, _ = self._takeover(
            "Turn, turn to your left.",
            {"args": None, "declined": False, "reason": "", "secs": 6.0,
             "error": "APITimeoutError: timed out"})
        self.assertEqual(line, "I couldn't safely parse that whole route.")

    def test_offline_keeps_the_denial_and_never_calls_the_model(self):
        with mock.patch("intelligence.connectivity.is_offline", return_value=True), \
                mock.patch.object(motion_route, "interpret") as interp:
            line = self.interaction._explicit_motion_takeover("Turn, turn to your left.")
        interp.assert_not_called()
        self.assertEqual(line, "I couldn't safely parse that whole route.")

    def test_master_switch_off_keeps_the_denial(self):
        with mock.patch.object(config, "MOTION_ROUTE_ENABLED", False, create=True), \
                mock.patch.object(motion_route, "interpret") as interp:
            line = self.interaction._explicit_motion_takeover("Turn, turn to your left.")
        interp.assert_not_called()
        self.assertEqual(line, "I couldn't safely parse that whole route.")

    def test_low_confidence_transcript_blocks_the_rescue(self):
        # Plan §4.3.1: a fabricated drive is strictly worse than a fabricated fact.
        with mock.patch.object(self.interaction, "_turn_transcript_trusted",
                               return_value=False), \
                mock.patch.object(motion_route, "interpret") as interp:
            line = self.interaction._explicit_motion_takeover("Turn, turn to your left.")
        interp.assert_not_called()
        self.assertEqual(line, "I couldn't safely parse that whole route.")
        self.assertEqual(self.started, [])

    def test_low_confidence_transcript_does_not_block_the_regex_lane(self):
        # The fast lane's bar is deliberately unchanged — its rigidity IS its guard.
        with mock.patch.object(self.interaction, "_turn_transcript_trusted",
                               return_value=False):
            line = self.interaction._explicit_motion_takeover(
                "turn left then move forward five feet")
        self.assertEqual(line, "On it — 2 moves.")

    def test_charging_locks_the_wheels(self):
        with mock.patch.object(self.interaction.motion_controller, "charging",
                               return_value=True):
            line, _ = self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(line, "I'm plugged in and charging. Wheels stay locked.")
        self.assertEqual(self.started, [])

    def test_a_no_drive_room_declines_with_the_way_out_stated(self):
        with mock.patch("intelligence.motion_agency.no_drive_room",
                        return_value=("workshop", "carpet")):
            line, _ = self._takeover("Turn, turn to your left.", self._ok())
        self.assertIn("not to drive in the workshop", line)
        self.assertIn("you can drive in here", line)
        self.assertEqual(self.started, [])

    def test_a_refusal_line_is_never_stamped_as_an_executed_route(self):
        # A refusal is ALSO a line, so a caller that treats "returned a string" as
        # "the wheels turned" writes an audit that says a route executed while the
        # base sat on the charger — the exact class of untruth c7ef872 was written
        # to end (203 audited not_in_execute_allowlist events logged while the
        # wheels turned).
        decisions, _ = ar.route_tool_to_decisions(self.ROUTE)
        line, drove = self.interaction._handle_motion_route(decisions)
        self.assertEqual(line, "On it — 2 moves.")
        self.assertTrue(drove)
        with mock.patch.object(self.interaction.motion_controller, "charging",
                               return_value=True):
            line, drove = self.interaction._handle_motion_route(decisions)
        self.assertTrue(line)
        self.assertFalse(drove)
        with mock.patch("intelligence.motion_agency.no_drive_room",
                        return_value=("workshop", "carpet")):
            line, drove = self.interaction._handle_motion_route(decisions)
        self.assertTrue(line)
        self.assertFalse(drove)
        with mock.patch.object(motion_sequence, "spin_clearance_reason",
                               return_value="spin_no_elbow_room"):
            line, drove = self.interaction._handle_motion_route(decisions)
        self.assertEqual(line, config.MOTION_ROUTE_SPIN_DENIAL_LINE)
        self.assertFalse(drove)
        with mock.patch.object(self.interaction, "_start_motion_sequence",
                               return_value=False):
            self.assertEqual(self.interaction._handle_motion_route(decisions),
                             (None, False))

    def test_a_cramped_full_spin_refuses_in_character(self):
        with mock.patch.object(motion_sequence, "spin_clearance_reason",
                               return_value="spin_no_elbow_room"):
            line, _ = self._takeover(
                "spin all the way around then move forward",
                self._ok(_steps({"op": "turn", "direction": "around", "deg": 360},
                                {"op": "move", "direction": "forward", "dist_m": 0.3})))
        self.assertEqual(line, config.MOTION_ROUTE_SPIN_DENIAL_LINE)
        self.assertEqual(self.started, [])

    def test_the_allowlist_is_a_real_kill_switch(self):
        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS) - {"motion.route"}
        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", allowed):
            line, _ = self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(line, "I couldn't safely parse that whole route.")
        self.assertEqual(self.started, [])

    def test_switching_a_leg_off_stops_the_route_that_contains_it(self):
        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS) - {"motion.turn"}
        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", allowed):
            line, _ = self._takeover("Turn, turn to your left.", self._ok())
        self.assertEqual(line, "I couldn't safely parse that whole route.")
        self.assertEqual(self.started, [])

    def test_a_one_step_plan_runs_as_a_single_command(self):
        # motion_sequence.start() refuses anything under two steps and returns False,
        # which would leave the turn silent.
        with mock.patch.object(self.interaction, "_handle_router_motion_action",
                               return_value="Spinning around.") as single:
            line, _ = self._takeover(
                "just spin around", self._ok(_steps({"op": "turn",
                                                     "direction": "around",
                                                     "deg": 360})))
        single.assert_called_once()
        self.assertEqual(line, "Spinning around.")
        self.assertEqual(self.started, [])

    def test_no_base_falls_through_before_the_interpreter_is_asked(self):
        with mock.patch.object(self.interaction.motion_controller, "available",
                               return_value=False), \
                mock.patch.object(motion_route, "interpret") as interp:
            self.interaction._explicit_motion_takeover("Turn, turn to your left.")
        interp.assert_not_called()

    def test_the_other_tri_state_arms_never_reach_the_interpreter(self):
        with mock.patch.object(motion_route, "interpret") as interp:
            self.assertEqual(
                self.interaction._explicit_motion_takeover(
                    "turn left then move forward five feet"),
                "On it — 2 moves.")
            for text in FIGURATIVE_DECOYS:
                self.interaction._explicit_motion_takeover(text)
        interp.assert_not_called()
        self.assertEqual(len(self.started), 1)


class OrganicPathTest(unittest.TestCase):
    """Phase 2: the reply call choosing motion.route on a turn the regex never saw."""

    ROUTE = _steps({"op": "move", "direction": "back", "dist_m": 0.3},
                   {"op": "turn", "direction": "around", "deg": 180})

    def setUp(self):
        from intelligence import interaction
        self.interaction = interaction
        self.spoken: list[str] = []
        self.started: list[list] = []
        interaction._tool_routed_path.clear()
        self._stack = []
        for patcher in (
            mock.patch.object(interaction, "_speak_blocking",
                              side_effect=lambda line, **kw: self.spoken.append(line)),
            mock.patch.object(interaction.motion_controller, "available",
                              return_value=True),
            mock.patch.object(interaction.motion_controller, "charging",
                              return_value=False),
            mock.patch.object(interaction, "_start_motion_sequence",
                              side_effect=lambda d: self.started.append(list(d)) or True),
            mock.patch("intelligence.motion_agency.no_drive_room", return_value=None),
            mock.patch.object(motion_sequence, "spin_clearance_reason",
                              return_value=None),
            mock.patch.object(interaction, "_turn_transcript_trusted",
                              return_value=True),
            mock.patch.object(interaction.llm, "get_response",
                              return_value="fallback words"),
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

    def _route(self, text, args=None):
        return self.interaction._execute_tool_routed_action(
            "motion.route", args if args is not None else self.ROUTE, text, None)

    def test_a_conversational_route_drives(self):
        resp = self._route("back up a little and then face the other way")
        self.assertEqual(resp, "On it — 2 moves.")
        self.assertEqual(self.interaction._consume_tool_routed_path(),
                         "tool_router.motion.route")
        self.assertEqual(
            [(d.action, d.args) for d in self.started[0]],
            [("motion.move", {"direction": "back", "dist_m": 0.3}),
             ("motion.turn", {"direction": "around", "deg": 180.0})])

    def test_the_evidence_gate_refuses_a_route_invented_out_of_banter(self):
        # Unlike the rescue path this one DOES run the gate: the regex never claimed
        # the turn, and the model chose the tool off a persona reply call.
        for text in FIGURATIVE_DECOYS:
            self.started.clear()
            resp = self._route(text)
            self.assertEqual(resp, "fallback words", text)
            self.assertEqual(self.started, [], text)
            self.assertIsNone(self.interaction._consume_tool_routed_path())

    def test_a_shaky_transcript_declines_to_conversation(self):
        with mock.patch.object(self.interaction, "_turn_transcript_trusted",
                               return_value=False):
            resp = self._route("back up a little and then face the other way")
        self.assertEqual(resp, "fallback words")
        self.assertEqual(self.started, [])

    def test_a_plan_outside_the_clamps_declines_to_conversation(self):
        resp = self._route("back up then roll way out",
                           _steps(*([{"op": "move", "direction": "back",
                                      "dist_m": config.MOTION_ROUTE_MAX_STEP_M}] * 5)))
        self.assertEqual(resp, "fallback words")
        self.assertEqual(self.started, [])

    def test_no_base_says_so_out_loud(self):
        with mock.patch.object(self.interaction.motion_controller, "available",
                               return_value=False):
            resp = self._route("back up a little and then face the other way")
        self.assertIn(resp, config.MOTION_NO_BASE_DENIAL_LINES)
        self.assertEqual(self.started, [])

    def test_charging_locks_the_wheels(self):
        with mock.patch.object(self.interaction.motion_controller, "charging",
                               return_value=True):
            resp = self._route("back up a little and then face the other way")
        self.assertEqual(resp, "I'm plugged in and charging. Wheels stay locked.")
        self.assertEqual(self.started, [])

    def test_the_allowlist_is_a_real_kill_switch(self):
        allowed = set(config.ACTION_ROUTER_EXECUTE_ACTIONS) - {"motion.route"}
        with mock.patch.object(config, "ACTION_ROUTER_EXECUTE_ACTIONS", allowed):
            resp = self._route("back up a little and then face the other way")
        self.assertEqual(resp, "fallback words")
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
