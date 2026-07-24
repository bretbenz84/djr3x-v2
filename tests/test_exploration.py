"""Tests for room exploration mode (intelligence/exploration.py + wiring).

Mock motion + mock vision + mock speech: no serial, no camera, no OpenAI, no DB
writes. The FSM worker is driven SYNCHRONOUSLY (calling _run_session directly) for
determinism — no thread-join races.

Run in isolation (the full discover run is known-flaky):
    ./venv/bin/python -m unittest tests.test_exploration -v
"""

import time
import unittest
from unittest import mock

import config
from intelligence import action_router as ar
from intelligence import exploration as ex


# ── helpers ───────────────────────────────────────────────────────────────────


def _cand(name, score, view="center", category="art", boring=False):
    return {
        "name": name, "view": view, "category": category,
        "interest": score, "score": score, "boring": boring,
        "riff_hook": f"{name} detail", "novelty": "",
    }


def _appraisal(cand):
    return {"top": cand, "candidates": [cand], "open_direction": "none", "floor_hazard": ""}


def _new_session(state="exploring", appraise_ok=True):
    sess = ex._Session(person_id=7, person_name="Bret", source="invite")
    sess.state = state
    # Locomotion is gated on a real vision read from the previous stop ("never drive
    # blind"); default True so leg tests exercise the drive path. Production starts False.
    sess.last_appraise_ok = appraise_ok
    return sess


# ── 1. Invitation classifier ──────────────────────────────────────────────────


class ClassifierTests(unittest.TestCase):
    POSITIVE = [
        "feel free to explore the room",
        "why don't you look around a little",
        "explore your environment",
        "make yourself at home",
        "go ahead and explore",
        "wander around a bit",
        "check out the place",
        "take a look around",
        "roam around the room",
        "have a look around",
        "go explore the room",
        "take a lap around the room",
        "scope out the room",
        "explore your new home",
        "take a stroll around the room",
        "hey rex, look around a little",
        "you can wander around",
        "be my guest and look around",
    ]
    NEGATIVE = [
        "look around and tell me what you see",
        "look around for my keys",
        "look to your left",
        "turn around",
        "what do you see",
        "look left and describe it",
        "this room is worth exploring someday",
        "search the room for the cat",
        "describe the room",
        "move forward",
        "explore new marketing strategies",
        "i want to explore my feelings",
        # ── review-confirmed false positives that MUST decline ──
        # First-person answers (not addressed to Rex):
        "I love to just wander around the city",
        "I like to look around a little",
        "we usually explore the area on weekends",
        "I want to check out the place",
        "Sometimes I just roam around the neighborhood",
        # Negations (opposite of intent):
        "Don't look around, focus on me",
        "no need to look around",
        "please do not look around",
        "don't explore the room",
        # Third-party narration:
        "the dog likes to roam around the yard",
        "the kids love to wander around the house",
        # Search errands (incl. non-look verbs + pronoun objects):
        "scan the room for my keys",
        "survey the room for survivors",
        "scope out the room for exits",
        "look around for it",
        "look around for him",
        # Bare/figurative "explore":
        "there is so much to explore",
        "we have a lot of ground to explore",
        "i want to explore",
        "let me explore",
        "explore your feelings",
        "explore your options",
    ]

    def test_positive_phrases_fire(self):
        for t in self.POSITIVE:
            d = ar.classify_explicit_exploration(t)
            self.assertIsNotNone(d, f"should classify: {t!r}")
            self.assertEqual(d.action, "motion.explore", f"wrong action for {t!r}")

    def test_negative_phrases_decline(self):
        for t in self.NEGATIVE:
            d = ar.classify_explicit_exploration(t)
            self.assertIsNone(d, f"should NOT classify: {t!r} -> {d and d.action}")

    def test_spec_registered(self):
        self.assertIn("motion.explore", ar.EXECUTABLE_ACTIONS)
        self.assertIn("motion.explore", ar._VALID_ACTIONS)
        self.assertEqual(ar.ACTION_CATEGORIES["motion.explore"], "motion")

    def test_directed_look_still_declines(self):
        # The existing "look around and tell me what you see" path must keep it.
        self.assertIsNone(ar.classify_explicit_exploration("look around and tell me what you see"))
        # A motion command is claimed by the motion classifier first, not here.
        self.assertIsNone(ar.classify_explicit_exploration("turn around"))


# ── 2. Turn consumption while active ──────────────────────────────────────────


class TurnConsumptionTests(unittest.TestCase):
    def setUp(self):
        self._stop = mock.patch("intelligence.motion_controller.stop", return_value=1)
        self._stop.start()
        ex._session = _new_session()

    def tearDown(self):
        ex._session = None
        self._stop.stop()

    def test_stop_word_ends_mode(self):
        line = ex.handle_user_turn("stop", 7)
        self.assertIn(line, config.EXPLORE_ABORT_LINES)
        self.assertTrue(ex._session.aborting())
        self.assertEqual(ex._session.abort_reason, "user_recall")

    def test_recall_phrase_ends_mode(self):
        line = ex.handle_user_turn("okay come back here", 7)
        self.assertIsNotNone(line)
        self.assertTrue(ex._session.aborting())

    def test_encouragement_continues(self):
        line = ex.handle_user_turn("keep going", 7)
        self.assertIn(line, config.EXPLORE_ENCOURAGE_ACK_LINES)
        self.assertFalse(ex._session.aborting())
        self.assertEqual(ex._session.state, "exploring")

    def test_real_question_pauses_and_releases(self):
        line = ex.handle_user_turn("what's that painting made of?", 7)
        self.assertIsNone(line)  # released to normal routing
        self.assertEqual(ex._session.state, "paused")
        self.assertFalse(ex._session.aborting())

    def test_second_question_while_paused_ends_mode(self):
        ex.handle_user_turn("what's that?", 7)          # -> paused
        line = ex.handle_user_turn("actually play some music", 7)
        self.assertIsNone(line)
        self.assertTrue(ex._session.aborting())
        self.assertEqual(ex._session.abort_reason, "user_engaged")

    def test_encouragement_resumes_a_paused_walk(self):
        ex.handle_user_turn("what's that?", 7)          # -> paused
        line = ex.handle_user_turn("keep going", 7)
        self.assertIn(line, config.EXPLORE_ENCOURAGE_ACK_LINES)
        self.assertEqual(ex._session.state, "exploring")

    def test_explicit_drive_command_ends_the_walk_and_releases(self):
        # Manual takeover: "turn right" during a walk must END it (not pause) and
        # release the turn so the normal motion path executes it — pausing meant
        # the walk resumed 4 s later and turned wherever ITS plan pointed (field
        # 2026-07-23: "I tell it to turn right, it turns left").
        line = ex.handle_user_turn("turn to your right a little", 7)
        self.assertIsNone(line)                          # released to routing
        self.assertTrue(ex._session.aborting())
        self.assertEqual(ex._session.abort_reason, "user_motion_command")

    def test_move_command_ends_the_walk_too(self):
        line = ex.handle_user_turn("move forward three feet", 7)
        self.assertIsNone(line)
        self.assertTrue(ex._session.aborting())


# ── 3. FSM end-to-end (synchronous, mocked seams) ─────────────────────────────


class FsmTests(unittest.TestCase):
    def setUp(self):
        self.spoken = []

        def _rec(sess, text, **kw):
            self.spoken.append(text)
            sess.lines_spoken += 1
            return True  # _speak's contract: True = line actually enqueued

        # Silence + no motion + no vision + no LLM — the loop runs on canned appraisals.
        self._patches = [
            mock.patch.object(ex, "_hold_head", lambda s: None),
            mock.patch.object(ex, "_release_head", lambda: None),
            mock.patch.object(ex, "_glance", lambda v: None),
            mock.patch.object(ex, "_travel_one_leg", lambda s: None),
            mock.patch.object(ex, "_survey", lambda s: [("center", object())]),
            mock.patch.object(ex, "_check_can_continue", return_value=True),
            mock.patch.object(ex, "_announce", lambda s: None),
            mock.patch.object(ex, "_generate", return_value="A witty line."),
            mock.patch.object(ex, "_speak", _rec),
        ]
        for p in self._patches:
            p.start()
        ex._session = None

    def tearDown(self):
        for p in self._patches:
            p.stop()
        ex._session = None

    def _run_with_appraisals(self, appraisals):
        seq = list(appraisals)

        def _fake_appraise(sess, views):
            if seq:
                return seq.pop(0)
            return _appraisal(_cand("nothing", 0.05, boring=True))

        with mock.patch.object(ex, "_appraise", _fake_appraise), \
                mock.patch.object(ex, "_seed_topic") as seed, \
                mock.patch.object(ex, "_record_episode") as episode, \
                mock.patch.object(ex, "_bank_callback"):
            sess = _new_session(state="announce")
            ex._session = sess
            ex._run_session(sess)
            return sess, seed, episode

    def test_never_fixates_on_first_stop(self):
        # A wildly interesting first stop must NOT end the walk (min-stops guard).
        interesting0 = _appraisal(_cand("mural", 0.95))
        interesting1 = _appraisal(_cand("neon sign", 0.9))
        sess, seed, _ = self._run_with_appraisals([interesting0, interesting1])
        # Fixation happened at the SECOND stop (stops_done == 2), not the first.
        self.assertEqual(sess.stops_done, 2)
        self.assertIsNotNone(sess.best)
        seed.assert_called_once()

    def test_fixates_when_threshold_met_after_min_stops(self):
        dull = _appraisal(_cand("a chair", 0.1, boring=True))
        star = _appraisal(_cand("oil painting", 0.9))
        sess, seed, episode = self._run_with_appraisals([dull, star])
        self.assertEqual(sess.best["name"], "oil painting")
        self.assertEqual(sess.stops_done, 2)
        seed.assert_called_once()

    def test_boring_room_winds_down_without_fixation(self):
        boring = [_appraisal(_cand(f"chair {i}", 0.1, boring=True)) for i in range(config.EXPLORE_MAX_STOPS)]
        sess, seed, episode = self._run_with_appraisals(boring)
        # No fixation on junk — the walk wound down.
        seed.assert_not_called()
        episode.assert_not_called()
        self.assertGreaterEqual(sess.stops_done, 2)

    def test_fallback_fixates_on_best_so_far_at_budget(self):
        # Nothing crosses 0.75, but a 0.6 clears the 0.55 fallback at budget end.
        mids = [_appraisal(_cand("odd trinket", 0.6))] + [
            _appraisal(_cand(f"lamp {i}", 0.2)) for i in range(config.EXPLORE_MAX_STOPS)
        ]
        sess, seed, _ = self._run_with_appraisals(mids)
        seed.assert_called_once()
        self.assertEqual(sess.best["name"], "odd trinket")


# ── 4. Scoring ────────────────────────────────────────────────────────────────


class ScoringTests(unittest.TestCase):
    def test_boring_label_is_clamped(self):
        cands = [_cand("chair", 0.95, category="object")]
        # strip pre-set score so _score_candidates recomputes
        for c in cands:
            c.pop("score", None)
            c.pop("boring", None)
        with mock.patch.object(ex, "_label_sightings", return_value={}):
            ex._score_candidates(cands)
        self.assertTrue(cands[0]["boring"])
        self.assertLessEqual(cands[0]["score"], config.EXPLORE_BORING_MAX_SCORE)

    def test_novelty_boost_for_new_label(self):
        cands = [_cand("antique telescope", 0.6, category="object")]
        for c in cands:
            c.pop("score", None)
            c.pop("boring", None)
        with mock.patch.object(ex, "_label_sightings", return_value={}):
            ex._score_candidates(cands)
        self.assertFalse(cands[0]["boring"])
        self.assertGreater(cands[0]["score"], 0.6)  # boosted

    def test_no_boost_for_established_label(self):
        cands = [_cand("plant", 0.6, category="decor")]
        for c in cands:
            c.pop("score", None)
            c.pop("boring", None)
        with mock.patch.object(ex, "_label_sightings", return_value={"plant": 9, "decor": 9}):
            ex._score_candidates(cands)
        self.assertAlmostEqual(cands[0]["score"], 0.6, places=6)

    def test_cand_key_dedup(self):
        a = _cand("Red Vase", 0.8, category="Decor")
        b = _cand("red vase", 0.4, category="decor")
        self.assertEqual(ex._cand_key(a), ex._cand_key(b))


# ── 5. Floor ownership ────────────────────────────────────────────────────────


class FloorOwnershipTests(unittest.TestCase):
    def setUp(self):
        ex._session = _new_session()

    def tearDown(self):
        ex._session = None

    def test_can_proactive_speak_denied_while_active(self):
        from intelligence import speech_engine
        self.assertTrue(ex.active())
        self.assertFalse(speech_engine.can_proactive_speak())
        self.assertFalse(speech_engine.can_proactive_speak(salient=True))
        self.assertFalse(speech_engine.can_proactive_speak(reactive=True))

    def test_motion_agency_stands_down(self):
        from intelligence import motion_agency

        class _Prof:
            user_mid_sentence = False
            suppress_proactive = False
            interaction_busy = False

        with mock.patch("intelligence.motion_controller.available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn") as turn, \
                mock.patch("intelligence.motion_controller.come") as come:
            motion_agency.step({"people": []}, _Prof())
            turn.assert_not_called()
            come.assert_not_called()

    def test_active_false_when_no_session(self):
        ex._session = None
        self.assertFalse(ex.active())


# ── 6. Locomotion (mocked motion, no serial) ──────────────────────────────────


class LocomotionTests(unittest.TestCase):
    def setUp(self):
        ex._session = None

    def test_travel_issues_turn_and_move(self):
        sess = _new_session()
        sess.last_open_direction = "left"
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn", return_value=11) as turn, \
                mock.patch("intelligence.motion_controller.move", return_value=12) as move, \
                mock.patch.object(ex, "_tof_mm", return_value={"fl": 2000, "fr": 1900, "lf": 2500}), \
                mock.patch("hardware.motion.wait_done", return_value={"result": "completed", "odom": {"x": 0, "y": 0, "theta": 0}}):
            ex._travel_one_leg(sess)
        turn.assert_called_once()
        move.assert_called_once()
        self.assertEqual(sess.legs_done, 1)
        self.assertEqual(sess.blocked_legs, 0)

    def test_blocked_leg_marks_dead_heading(self):
        sess = _new_session()
        sess.last_open_direction = "center"  # no turn, straight move
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn", return_value=11), \
                mock.patch("intelligence.motion_controller.move", return_value=12), \
                mock.patch.object(ex, "_tof_mm", return_value={"fl": 1800, "fr": 1700}), \
                mock.patch("hardware.motion.wait_done", return_value={"result": "blocked", "odom": {"x": 0, "y": 0, "theta": 0}}):
            ex._travel_one_leg(sess)
        self.assertEqual(sess.blocked_legs, 1)

    def test_floor_hazard_vetoes_forward_leg(self):
        sess = _new_session()
        sess.last_open_direction = "center"
        sess.last_floor_hazard = "cables across the floor"
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn", return_value=11), \
                mock.patch("intelligence.motion_controller.move", return_value=12) as move, \
                mock.patch("hardware.motion.wait_done", return_value={"result": "completed", "odom": {}}):
            ex._travel_one_leg(sess)
        move.assert_not_called()  # forward leg vetoed by the hazard hint
        self.assertEqual(sess.last_floor_hazard, "")  # cleared after use

    def test_travel_noop_without_base(self):
        sess = _new_session()
        with mock.patch.object(ex, "base_available", return_value=False), \
                mock.patch("intelligence.motion_controller.move") as move:
            ex._travel_one_leg(sess)
        move.assert_not_called()

    def test_travel_noop_when_locomotion_disabled(self):
        sess = _new_session()
        with mock.patch.object(config, "EXPLORE_LOCOMOTION_ENABLED", False), \
                mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn") as turn:
            ex._travel_one_leg(sess)
        turn.assert_not_called()

    def test_never_drives_blind_without_a_vision_read(self):
        # last_appraise_ok False (dead camera / vision error) => hold position, no leg.
        sess = _new_session(appraise_ok=False)
        sess.last_open_direction = "center"
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn") as turn, \
                mock.patch("intelligence.motion_controller.move") as move:
            ex._travel_one_leg(sess)
        turn.assert_not_called()
        move.assert_not_called()

    def test_pause_before_move_cancels_forward_leg(self):
        # A pause that lands during the turn must cancel the forward move.
        sess = _new_session()
        sess.last_open_direction = "left"

        def _turn_then_pause(*a, **k):
            sess.state = "paused"   # user interrupt lands during the turn
            return 11

        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn", side_effect=_turn_then_pause), \
                mock.patch("intelligence.motion_controller.move") as move, \
                mock.patch.object(ex, "_tof_mm", return_value={"fl": 900, "fr": 850, "lf": 2000}), \
                mock.patch("hardware.motion.wait_done", return_value={"result": "completed", "odom": {}}):
            ex._travel_one_leg(sess)
        move.assert_not_called()  # forward leg cancelled by the pause

    def test_mobile_session_requires_real_wandering_before_fixation(self):
        sess = _new_session()
        sess.had_base = True
        sess.stops_done = 6
        sess.best = _cand("neon sign", 0.95)
        with mock.patch.object(config, "EXPLORE_MIN_LEGS_BEFORE_FIXATE", 3):
            sess.legs_done = 2
            self.assertFalse(ex._should_fixate(sess))
            sess.legs_done = 3
            self.assertTrue(ex._should_fixate(sess))

    def test_headonly_session_does_not_wait_for_impossible_legs(self):
        sess = _new_session()
        sess.had_base = False
        sess.stops_done = 2
        sess.legs_done = 0
        sess.best = _cand("neon sign", 0.95)
        self.assertTrue(ex._should_fixate(sess))

    def test_heading_and_distance_come_from_tof_clearance(self):
        sess = _new_session()
        tof = {"fl": 900, "fr": 850, "lf": 2200, "lb": 800,
               "rl": 700, "rr": 700, "rb": 800, "rf": 600}
        with mock.patch.object(ex, "_tof_mm", return_value=tof), \
                mock.patch("hardware.motion.telemetry", return_value={"odom": {"theta": 0.0}}):
            self.assertEqual(ex._plan_leg_heading(sess), 67.5)
            # min(front)=0.85; (0.85 - 0.45) * 0.65 = 0.26 m
            self.assertAlmostEqual(ex._plan_leg_distance(), 0.26, places=2)

    def test_distance_refuses_close_or_missing_front_tof(self):
        with mock.patch.object(ex, "_tof_mm", return_value={"fl": 500, "fr": 480}):
            self.assertIsNone(ex._plan_leg_distance())
        with mock.patch.object(ex, "_tof_mm", return_value={}):
            self.assertIsNone(ex._plan_leg_distance())

    def test_open_front_prefers_straight_corridor(self):
        sess = _new_session()
        tof = {"fl": 2500, "fr": 2400, "lf": 1200, "rf": 1200}
        with mock.patch.object(ex, "_tof_mm", return_value=tof), \
                mock.patch("hardware.motion.telemetry", return_value={"odom": {"theta": 0.0}}):
            self.assertEqual(ex._plan_leg_heading(sess), 0.0)

    def test_clear_floor_language_is_not_a_hazard(self):
        for text in ("None", "Looks clear ahead", "The floor ahead looks clear.", ""):
            self.assertEqual(ex._normalize_floor_hazard(text), "")
        self.assertEqual(ex._normalize_floor_hazard("cables across the floor"),
                         "cables across the floor")

    def test_generic_mess_prose_no_longer_grounds_the_robot(self):
        # Field 2026-07-23: "Some clutter around the kitchen area." vetoed EVERY
        # forward leg — look-around panned and never drove. Solid clutter is the
        # ToF's job; only ToF-blind dangers (cables/steps/liquids) may veto.
        for text in (
            "Some clutter around the kitchen area.",
            "A few objects near the couch",
            "Shoes and a backpack on the floor",
            "boxes stacked to the left",
        ):
            self.assertEqual(ex._normalize_floor_hazard(text), "", text)

    def test_tof_blind_dangers_still_veto(self):
        for text in (
            "a power cord runs across the path",
            "step down into the sunken living room",
            "wet spill near the doorway",
            "broken glass by the table",
            "charger cable on the rug",
        ):
            self.assertEqual(ex._normalize_floor_hazard(text), text, text)

    def test_opening_turn_answers_the_invite_with_the_base(self):
        # First stop: no forward move (no vision read yet), but a chassis turn of at
        # least EXPLORE_OPENING_TURN_MIN_DEG fires so the invite visibly moves him.
        sess = _new_session()
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch.object(ex, "_plan_leg_heading", return_value=0.0), \
                mock.patch.object(ex, "_wait_leg_done", return_value="completed"), \
                mock.patch("intelligence.motion_controller.turn", return_value=5) as turn:
            ex._opening_turn(sess)
        turn.assert_called_once()
        self.assertGreaterEqual(
            abs(turn.call_args[0][0]),
            float(config.EXPLORE_OPENING_TURN_MIN_DEG),
        )

    def test_opening_turn_skips_without_base(self):
        sess = _new_session()
        with mock.patch.object(ex, "base_available", return_value=False), \
                mock.patch("intelligence.motion_controller.turn") as turn:
            ex._opening_turn(sess)
        turn.assert_not_called()

    def test_travel_gaze_spans_the_turn_and_move(self):
        sess = _new_session()
        sess.last_open_direction = "left"
        gaze_handle = object()
        events = []

        def _turn(*args, **kwargs):
            events.append("turn")
            return 11

        def _move(*args, **kwargs):
            events.append("move")
            return 12

        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch.object(
                    ex, "_start_travel_gaze",
                    side_effect=lambda s: events.append("gaze_start") or gaze_handle,
                ), \
                mock.patch.object(
                    ex, "_stop_travel_gaze",
                    side_effect=lambda h: events.append("gaze_stop"),
                ) as stop_gaze, \
                mock.patch("intelligence.motion_controller.turn", side_effect=_turn), \
                mock.patch("intelligence.motion_controller.move", side_effect=_move), \
                mock.patch.object(ex, "_tof_mm", return_value={"fl": 2000, "fr": 1900, "lf": 2500}), \
                mock.patch("hardware.motion.wait_done", return_value={"result": "completed", "odom": {}}):
            ex._travel_one_leg(sess)

        self.assertEqual(events, ["gaze_start", "turn", "move", "gaze_stop"])
        stop_gaze.assert_called_once_with(gaze_handle)


# ── 7. Invite dispatch (interaction wiring) ───────────────────────────────────


class InviteDispatchTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def test_invite_starts_session_and_stays_silent(self):
        from intelligence import interaction
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch.object(ex, "start", return_value=True) as start:
            resp = interaction._handle_explore_invite(
                "look around a little", person_id=7, person_name="Bret",
            )
        start.assert_called_once()
        self.assertTrue(interaction._is_silent_command_response(resp))

    def test_no_base_speaks_denial(self):
        from intelligence import interaction
        with mock.patch.object(ex, "base_available", return_value=False), \
                mock.patch.object(config, "EXPLORE_HEADONLY_FALLBACK_ENABLED", False), \
                mock.patch.object(interaction, "_speak_blocking", return_value=True) as speak:
            resp = interaction._handle_explore_invite(
                "explore the room", person_id=7, person_name="Bret",
            )
        self.assertIn(resp, config.EXPLORE_NO_BASE_LINES)
        speak.assert_called_once()

    def test_refused_start_releases_turn(self):
        from intelligence import interaction
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch.object(ex, "start", return_value=False):
            resp = interaction._handle_explore_invite(
                "explore", person_id=7, person_name="Bret",
            )
        self.assertIsNone(resp)  # released to normal routing


# ── 8. active()/status hygiene ────────────────────────────────────────────────


class LifecycleTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def test_ttl_expiry_frees_floor(self):
        sess = _new_session()
        sess.created_at = time.monotonic() - (config.EXPLORE_STEP_TTL_SECS + 5)
        ex._session = sess
        self.assertFalse(ex.active())

    def test_status_shape(self):
        ex._session = _new_session()
        st = ex.status()
        self.assertTrue(st["active"])
        self.assertEqual(st["state"], "exploring")
        self.assertIn("stops_done", st)


# ── 9. Motion-agency stand-down (with a REAL maneuver-triggering snapshot) ────


class _Prof:
    user_mid_sentence = False
    suppress_proactive = False
    interaction_busy = False


def _tracked_snapshot():
    return {"people": [{"id": "person_1", "person_db_id": 1,
                        "distance_zone": "social", "face_visible": True}]}


class MotionAgencyRealStandDownTests(unittest.TestCase):
    """The empty-people test is vacuous (no maneuver possible). This drives a real
    realign trigger (locked person + big neck offset) so the assertion actually
    proves exploration.active() suppresses a maneuver that WOULD otherwise fire."""

    def setUp(self):
        from intelligence import motion_agency as MA
        self.MA = MA
        MA._reset("neck_hits", "far_hits")
        MA._state["last_turn_at"] = 0.0
        self._tracking = {"locked": True, "visible": True, "lock_key": "slot:person_1"}
        self._neck = 8500  # far off neutral 6000 -> |frac| ~0.63 >= MOTION_FACE_NECK_FRACTION
        self._patches = [
            mock.patch.object(MA.motion_controller, "available", return_value=True),
            mock.patch.object(MA.motion, "state", return_value="idle"),
            mock.patch.object(MA.motion_controller, "turn", return_value=7),
            mock.patch.object(MA.motion_controller, "come", return_value=8),
            mock.patch("world_state.world_state.get", side_effect=lambda key: (
                {"face_tracking": self._tracking, "servo_positions": {"neck": self._neck}}
                if key == "self_state" else {})),
        ]
        for p in self._patches:
            p.start()
        self.turn = self.MA.motion_controller.turn

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.MA._reset("neck_hits", "far_hits")
        ex._session = None

    def test_control_maneuver_fires_when_not_exploring(self):
        # Sanity: the harness really does trigger a realign turn when idle.
        ex._session = None
        for _ in range(3):
            self.MA.step(_tracked_snapshot(), _Prof())
        self.turn.assert_called()

    def test_standdown_suppresses_maneuver_while_exploring(self):
        ex._session = _new_session()
        self.assertTrue(ex.active())
        for _ in range(3):
            self.MA.step(_tracked_snapshot(), _Prof())
        self.turn.assert_not_called()


# ── 10. Worker abort honoring (real _check_can_continue, no external takeover) ─


class WorkerAbortHonoringTests(unittest.TestCase):
    """FsmTests mocks _check_can_continue away; these drive the REAL guard so a
    regression that stops honoring abort/gamepad/battery/game is caught."""

    def setUp(self):
        self._patches = [
            mock.patch.object(ex, "_hold_head", lambda s: None),
            mock.patch.object(ex, "_release_head", lambda: None),
            mock.patch.object(ex, "_glance", lambda v: None),
            mock.patch.object(ex, "_travel_one_leg", lambda s: None),
            mock.patch.object(ex, "_survey", lambda s: [("center", object())]),
            mock.patch.object(ex, "_appraise", lambda s, v: _appraisal(_cand("thing", 0.5))),
            mock.patch.object(ex, "_generate", return_value="line"),
            mock.patch.object(ex, "_speak", lambda s, t, **k: None),
            mock.patch.object(ex, "_seed_topic", lambda s, c: None),
            mock.patch.object(ex, "_record_episode", lambda s, c: None),
            mock.patch.object(ex, "_bank_callback", lambda s, c: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        ex._session = None

    def test_preset_abort_exits_immediately(self):
        sess = _new_session(state="announce")
        sess.abort.set()
        ex._run_session(sess)
        self.assertEqual(sess.stops_done, 0)
        self.assertFalse(sess.fixated)

    def test_gamepad_owner_manual_aborts(self):
        sess = _new_session(state="announce")
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("hardware.motion.owner", return_value="manual"):
            ex._run_session(sess)
        self.assertTrue(sess.aborting())
        self.assertEqual(sess.abort_reason, "manual_override")
        self.assertFalse(sess.fixated)

    def test_battery_critical_aborts(self):
        sess = _new_session(state="announce")
        with mock.patch("intelligence.battery_awareness.battery_critical", return_value=True):
            ex._run_session(sess)
        self.assertTrue(sess.aborting())
        self.assertEqual(sess.abort_reason, "battery_critical")

    def test_game_started_aborts(self):
        sess = _new_session(state="announce")
        with mock.patch("features.games.is_active", return_value=True):
            ex._run_session(sess)
        self.assertTrue(sess.aborting())
        self.assertEqual(sess.abort_reason, "game_started")


# ── 11. Vision safety caps ("never wander blind") ─────────────────────────────


class VisionCapTests(unittest.TestCase):
    def setUp(self):
        self._patches = [
            mock.patch.object(ex, "_hold_head", lambda s: None),
            mock.patch.object(ex, "_release_head", lambda: None),
            mock.patch.object(ex, "_glance", lambda v: None),
            mock.patch.object(ex, "_generate", return_value="line"),
            mock.patch.object(ex, "_speak", lambda s, t, **k: None),
            mock.patch.object(ex, "_seed_topic", lambda s, c: None),
            mock.patch.object(ex, "_record_episode", lambda s, c: None),
            mock.patch.object(ex, "_bank_callback", lambda s, c: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        ex._session = None

    def test_repeated_vision_failure_ends_walk(self):
        # Frames present but every appraisal call fails -> vision_failures climbs to the
        # cap and the loop ends without fixation (never keeps firing calls blindly).
        with mock.patch.object(ex, "_survey", lambda s: [("center", object())]), \
                mock.patch.object(ex, "_appraise_call", return_value=None), \
                mock.patch.object(ex, "_travel_one_leg", lambda s: None), \
                mock.patch.object(ex, "_seed_topic") as seed:
            sess = _new_session(state="announce")
            ex._run_session(sess)
        self.assertGreaterEqual(sess.vision_failures, config.EXPLORE_VISION_MAX_FAILURES)
        self.assertFalse(sess.fixated)
        seed.assert_not_called()

    def test_blind_camera_never_drives_and_ends(self):
        # No frames at all: counts as failures AND the blind-gate blocks every forward
        # leg. Camera blindness is only DISCOVERED at the first survey, so the one
        # rotation-in-place opening turn (issued before it, ToF-guided, firmware-safe)
        # is permitted — but no forward translation may ever happen.
        with mock.patch.object(ex, "_survey", lambda s: []), \
                mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("intelligence.motion_controller.turn", return_value=None) as turn, \
                mock.patch("intelligence.motion_controller.move") as move:
            sess = _new_session(state="announce")
            ex._run_session(sess)
        move.assert_not_called()
        self.assertLessEqual(turn.call_count, 1)  # at most the opening rotation
        self.assertGreaterEqual(sess.vision_failures, config.EXPLORE_VISION_MAX_FAILURES)
        self.assertFalse(sess.fixated)


# ── 12. start() / can_start() refusal logic ───────────────────────────────────


class StartRefusalTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def test_can_start_disabled(self):
        with mock.patch.object(config, "EXPLORE_ENABLED", False):
            self.assertEqual(ex.can_start(), "disabled")

    def test_can_start_game_active(self):
        with mock.patch("features.games.is_active", return_value=True):
            self.assertEqual(ex.can_start(), "game_active")

    def test_can_start_battery_critical(self):
        with mock.patch("intelligence.battery_awareness.battery_critical", return_value=True):
            self.assertEqual(ex.can_start(), "battery_critical")

    def test_can_start_already_active(self):
        ex._session = _new_session()
        self.assertEqual(ex.can_start(), "already_active")

    def test_start_refuses_with_no_base_and_no_fallback(self):
        with mock.patch.object(ex, "base_available", return_value=False), \
                mock.patch.object(config, "EXPLORE_HEADONLY_FALLBACK_ENABLED", False):
            started = ex.start(7, "Bret", source="invite")
        self.assertFalse(started)
        self.assertIsNone(ex._session)  # no worker thread spawned

    def test_start_refuses_when_already_active(self):
        ex._session = _new_session()
        with mock.patch.object(ex, "base_available", return_value=True):
            started = ex.start(9, "JT", source="invite")
        self.assertFalse(started)


# ── 13. Terminal motion states (disconnect / fault / estop abort) ─────────────


class TerminalMotionStateTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def _sess_with_base(self):
        sess = _new_session()
        sess.had_base = True
        return sess

    def test_base_disconnect_aborts(self):
        sess = self._sess_with_base()
        with mock.patch.object(ex, "base_available", return_value=False):
            self.assertFalse(ex._check_can_continue(sess))
        self.assertTrue(sess.aborting())
        self.assertEqual(sess.abort_reason, "base_disconnected")

    def test_firmware_estop_aborts(self):
        sess = self._sess_with_base()
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("hardware.motion.owner", return_value="auto"), \
                mock.patch("hardware.motion.state", return_value="estop"):
            self.assertFalse(ex._check_can_continue(sess))
        self.assertTrue(sess.aborting())
        self.assertEqual(sess.abort_reason, "base_estop")

    def test_firmware_fault_aborts(self):
        sess = self._sess_with_base()
        with mock.patch.object(ex, "base_available", return_value=True), \
                mock.patch("hardware.motion.owner", return_value="auto"), \
                mock.patch("hardware.motion.state", return_value="fault"):
            self.assertFalse(ex._check_can_continue(sess))
        self.assertEqual(sess.abort_reason, "base_fault")

    def test_headonly_session_ignores_disconnect(self):
        # A session that never had a base (head-only fallback) must not abort on
        # "disconnect" — there was nothing to disconnect.
        sess = _new_session()
        sess.had_base = False
        with mock.patch.object(ex, "base_available", return_value=False):
            self.assertTrue(ex._check_can_continue(sess))
        self.assertFalse(sess.aborting())


# ── 14. Tether origin captured at session start ───────────────────────────────


class TetherOriginTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def test_origin_is_session_start_not_first_leg_destination(self):
        # Odometry reads (2.0, 3.0) at session start; even with zero legs driven the
        # worker captures that as the tether origin BEFORE the explore loop runs.
        sess = _new_session(state="announce")
        with mock.patch.object(ex, "_hold_head", lambda s: None), \
                mock.patch.object(ex, "_release_head", lambda: None), \
                mock.patch.object(ex, "_announce", lambda s: None), \
                mock.patch.object(ex, "_explore_loop", lambda s: True), \
                mock.patch.object(ex, "_handoff", lambda s: None), \
                mock.patch.object(ex, "_current_xy", return_value=(2.0, 3.0)):
            ex._run_session(sess)
        self.assertEqual(sess.start_xy, (2.0, 3.0))


# ── 15. Fixation persists ONLY when the beat was delivered ────────────────────


class FixationDeliveryTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def _fixate(self, speak_ok):
        sess = _new_session()
        with mock.patch.object(ex, "_glance", lambda v: None), \
                mock.patch.object(ex, "_generate", return_value="THIS is art."), \
                mock.patch.object(ex, "_speak", return_value=speak_ok):
            ex._fixate(sess, _cand("oil painting", 0.9))
        return sess

    def test_delivered_fixation_persists(self):
        sess = self._fixate(speak_ok=True)
        self.assertTrue(sess.fixated)

    def test_dropped_fixation_line_does_not_persist(self):
        # The user started talking / enqueue failed -> Rex never said the beat, so
        # no fixation may be seeded into topic/memory at handoff.
        sess = self._fixate(speak_ok=False)
        self.assertFalse(sess.fixated)


# ── 16. Person candidates: fail-closed minor/identity gate ────────────────────


class PersonGateTests(unittest.TestCase):
    def _score(self, allowed):
        cands = [_cand("person by the window", 0.9, category="person")]
        for c in cands:
            c.pop("score", None)
            c.pop("boring", None)
        with mock.patch.object(ex, "_label_sightings", return_value={}), \
                mock.patch.object(ex, "_person_candidate_allowed", return_value=allowed):
            ex._score_candidates(cands)
        return cands[0]

    def test_blocked_person_is_clamped_and_can_never_fixate(self):
        c = self._score(allowed=False)
        self.assertTrue(c["boring"])
        self.assertTrue(c.get("person_blocked"))
        self.assertLessEqual(c["score"], config.EXPLORE_BORING_MAX_SCORE)
        # boring=True also blocks the fixation gate (_should_fixate rejects boring).

    def test_allowed_person_scores_normally(self):
        c = self._score(allowed=True)
        self.assertFalse(c["boring"])
        self.assertGreaterEqual(c["score"], 0.9)

    def test_gate_fails_closed_on_unknown_face(self):
        # An unidentified visible person (no person_db_id) -> False.
        with mock.patch("world_state.world_state.get", return_value=[
                {"id": "person_1", "face_visible": True}]):
            self.assertFalse(ex._person_candidate_allowed())

    def test_gate_fails_closed_on_minor(self):
        with mock.patch("world_state.world_state.get", return_value=[
                {"id": "person_1", "face_visible": True, "person_db_id": 5}]), \
                mock.patch("vision.face.visible_known_people", return_value=[(5, "Kid")]), \
                mock.patch("intelligence.profile_questions.person_is_minor", return_value=True):
            self.assertFalse(ex._person_candidate_allowed())

    def test_gate_fails_closed_on_error(self):
        with mock.patch("world_state.world_state.get", side_effect=RuntimeError("boom")):
            self.assertFalse(ex._person_candidate_allowed())

    def test_gate_allows_known_adult(self):
        with mock.patch("world_state.world_state.get", return_value=[
                {"id": "person_1", "face_visible": True, "person_db_id": 7}]), \
                mock.patch("vision.face.visible_known_people", return_value=[(7, "Bret")]), \
                mock.patch("intelligence.profile_questions.person_is_minor", return_value=False):
            self.assertTrue(ex._person_candidate_allowed())

    def test_person_directive_forbids_appearance_remarks(self):
        cand = _cand("person by the window", 0.9, category="person")
        sess = _new_session()
        riff = ex._riff_directive(sess, cand, boring=False)
        fix = ex._fixate_directive(sess, cand, ask=True)
        for d in (riff, fix):
            self.assertIn("NO remarks about their body", d)
        # Non-person subjects carry no person clause.
        obj = ex._riff_directive(sess, _cand("oil painting", 0.9), boring=False)
        self.assertNotIn("NO remarks about their body", obj)


# ── 17. Ownership TTL always outlives the configured session duration ─────────


class OwnershipTtlTests(unittest.TestCase):
    def tearDown(self):
        ex._session = None

    def test_long_configured_session_keeps_ownership_past_flat_ttl(self):
        # duration 400 > flat TTL 240: at 300s elapsed the session must STILL own
        # the floor (the old flat TTL released it mid-run).
        sess = _new_session()
        sess.created_at = time.monotonic() - 300.0
        ex._session = sess
        with mock.patch.object(config, "EXPLORE_MAX_DURATION_SECS", 400.0), \
                mock.patch.object(config, "EXPLORE_STEP_TTL_SECS", 240.0):
            self.assertTrue(ex.active())

    def test_wedged_session_still_expires(self):
        sess = _new_session()
        sess.created_at = time.monotonic() - 500.0  # past duration+60 and TTL
        ex._session = sess
        with mock.patch.object(config, "EXPLORE_MAX_DURATION_SECS", 400.0), \
                mock.patch.object(config, "EXPLORE_STEP_TTL_SECS", 240.0):
            self.assertFalse(ex.active())


if __name__ == "__main__":
    unittest.main()
