import unittest
from unittest import mock

import numpy as np

from hardware import head_pose
from state import State


def _decoded_head(updates):
    """(pitch, tilt) the head pair in a set_servos update stands for."""
    return head_pose.head_pose({ch: updates[ch] for ch in head_pose.HEAD_PAIR})


def _level_positions(neck=6000, lift=6000, pitch=0.0, tilt=0.0):
    pair = head_pose.head_pulses(pitch, tilt)
    return {
        "neck": neck,
        "headlift": lift,
        "head_left": pair[head_pose.HEAD_LEFT_CH],
        "head_right": pair[head_pose.HEAD_RIGHT_CH],
        "visor": 5732,
        "elbow": 6720,
        "hand": 6000,
        "pokerarm": 6000,
        "heroarm": 6000,
    }

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None


class FaceTrackingTests(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness

        self.consciousness = consciousness
        self.old_people = consciousness.world_state.get("people")
        self.old_self_state = consciousness.world_state.get("self_state")
        self.old_last_seen = consciousness._last_face_seen_at
        self.old_lock = dict(consciousness._face_tracking_lock)
        self.old_live_tracker = consciousness._face_tracking_tracker
        self.old_last_solo_identity = consciousness._last_solo_identity
        self.old_suspend_until = consciousness._face_tracking_suspended_until
        self.old_tracking_log_at = consciousness._last_face_tracking_log_at
        self.old_tracking_error_key = consciousness._face_tracking_last_error_key
        self.old_tracking_error_x = consciousness._face_tracking_last_error_x
        self.old_tracking_error_y = consciousness._face_tracking_last_error_y
        self.old_tracking_error_at = consciousness._face_tracking_last_error_at
        # Jump-guard reference state leaks across tests if not isolated (a rejected
        # jump leaves last/pending center set), making the suite order-dependent —
        # save + reset it so each test starts from a clean lock-free state.
        self.old_tracking_last_center = consciousness._face_tracking_last_center
        self.old_tracking_pending_center = consciousness._face_tracking_pending_center
        consciousness._face_tracking_last_center = None
        consciousness._face_tracking_pending_center = None
        # The persistent reversal-damping deadlines compare against mocked clocks that
        # move backward between tests — reset them so damping can't leak across tests.
        self.old_damped_x_until = consciousness._face_tracking_damped_x_until
        self.old_damped_y_until = consciousness._face_tracking_damped_y_until
        consciousness._face_tracking_damped_x_until = 0.0
        consciousness._face_tracking_damped_y_until = 0.0
        self.old_adaptive_head_rest = dict(consciousness._adaptive_head_rest)
        with consciousness._speaker_gaze_lock:
            self.old_speaker_gaze_intent = dict(consciousness._speaker_gaze_intent)
            consciousness._speaker_gaze_intent.clear()
        with consciousness._directed_gaze_hold_lock:
            self.old_directed_gaze_hold = dict(consciousness._directed_gaze_hold)
        consciousness.clear_directed_gaze_hold()
        self.old_process_started_mono = consciousness._process_started_mono
        self.old_presence_evidence_at = consciousness._startup_presence_evidence_at
        self.old_fallback_started = consciousness._startup_presence_fallback_started
        self.old_fallback_active = consciousness._startup_presence_fallback_active
        self.old_attentive_cock = dict(consciousness._attentive_cock)
        consciousness.reset_attentive_cock()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        c = self.consciousness
        c.world_state.update("people", self.old_people)
        c.world_state.update("self_state", self.old_self_state)
        c._last_face_seen_at = self.old_last_seen
        c._face_tracking_lock = self.old_lock
        c._face_tracking_tracker = self.old_live_tracker
        c._last_solo_identity = self.old_last_solo_identity
        c._face_tracking_suspended_until = self.old_suspend_until
        c._last_face_tracking_log_at = self.old_tracking_log_at
        c._face_tracking_last_error_key = self.old_tracking_error_key
        c._face_tracking_last_error_x = self.old_tracking_error_x
        c._face_tracking_last_error_y = self.old_tracking_error_y
        c._face_tracking_last_error_at = self.old_tracking_error_at
        c._face_tracking_last_center = self.old_tracking_last_center
        c._face_tracking_pending_center = self.old_tracking_pending_center
        c._face_tracking_damped_x_until = self.old_damped_x_until
        c._face_tracking_damped_y_until = self.old_damped_y_until
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update(self.old_adaptive_head_rest)
        with c._speaker_gaze_lock:
            c._speaker_gaze_intent.clear()
            c._speaker_gaze_intent.update(self.old_speaker_gaze_intent)
        with c._directed_gaze_hold_lock:
            c._directed_gaze_hold.clear()
            c._directed_gaze_hold.update(self.old_directed_gaze_hold)
        c._process_started_mono = self.old_process_started_mono
        c._startup_presence_evidence_at = self.old_presence_evidence_at
        c._startup_presence_fallback_started = self.old_fallback_started
        c._startup_presence_fallback_active = self.old_fallback_active
        c._attentive_cock.clear()
        c._attentive_cock.update(self.old_attentive_cock)

    def _frame_with_patch(self, x: int, y: int) -> np.ndarray:
        rng = np.random.default_rng(1234)
        patch = rng.integers(0, 255, size=(36, 36), dtype=np.uint8)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[y:y + 36, x:x + 36] = np.repeat(patch[:, :, None], 3, axis=2)
        return frame

    def _set_servo_positions(self, *, pitch=0.0, tilt=0.0):
        """Park the head at neck 6000 / lift 6000 with the two-servo head at (pitch, tilt),
        publishing both the named pulses and the driver's head_pose like hardware/servos."""
        c = self.consciousness
        self_state = c.world_state.get("self_state")
        self_state["servo_positions"] = _level_positions(pitch=pitch, tilt=tilt)
        self_state["head_pose"] = {"pitch": pitch, "tilt": tilt}
        c.world_state.update("self_state", self_state)

    def test_detection_hold_clears_stale_face_geometry(self):
        c = self.consciousness
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 120, 80, 90),
            "position": (140, 165),
            "face_box_fraction": 0.1,
        }])
        c._last_face_seen_at = 100.0

        with (
            mock.patch("vision.face.detect_faces", return_value=[]),
            mock.patch.object(c.time, "monotonic", return_value=102.0),
            mock.patch.object(c.config, "FACE_DETECTION_HOLD_SECS", 6.0),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_db_id"], 1)
        self.assertFalse(people[0]["face_visible"])
        self.assertTrue(people[0]["face_missing"])
        self.assertIsNone(people[0]["face_box"])
        self.assertIsNone(people[0]["position"])

    def test_face_mood_write_replaces_stale_neutral_expression(self):
        c = self.consciousness
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "expression": "neutral",
        }])

        c._write_face_mood_to_world_state(
            1,
            {"mood": "happy", "confidence": 0.91, "notes": "smiling"},
        )

        person = c.world_state.get("people")[0]
        self.assertEqual(person["expression"], "happy")
        self.assertEqual(person["face_mood"]["mood"], "happy")

    def test_solo_identity_sticky_rejects_far_false_positive_box(self):
        c = self.consciousness
        c.world_state.update("people", [{"id": "person_1"}])
        c._last_solo_identity = (1, "Bret", 100.0, (100, 300, 80, 80))
        fake_detection = {
            "bounding_box": (900, 120, 90, 90),
            "encoding": np.zeros(128, dtype=np.float32),
            "landmarks": np.zeros((68, 2), dtype=np.int32),
        }

        with (
            mock.patch("vision.face.detect_faces", return_value=[fake_detection]),
            mock.patch("vision.face.identify_face", return_value=None),
            mock.patch.object(c.time, "monotonic", return_value=101.0),
            mock.patch.object(c, "_maybe_prompt_unknown_identity"),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertIsNone(people[0].get("person_db_id"))
        self.assertIsNone(people[0].get("face_id"))

    def test_solo_identity_sticky_accepts_nearby_unmatched_box(self):
        c = self.consciousness
        c.world_state.update("people", [{"id": "person_1"}])
        c._last_solo_identity = (1, "Bret", 100.0, (100, 300, 80, 80))
        fake_detection = {
            "bounding_box": (118, 292, 82, 82),
            "encoding": np.zeros(128, dtype=np.float32),
            "landmarks": np.zeros((68, 2), dtype=np.int32),
        }

        with (
            mock.patch("vision.face.detect_faces", return_value=[fake_detection]),
            mock.patch("vision.face.identify_face", return_value=None),
            mock.patch.object(c.time, "monotonic", return_value=101.0),
            mock.patch.object(c, "_maybe_prompt_unknown_identity"),
        ):
            c._step_person_recognition(self.frame)

        people = c.world_state.get("people")
        self.assertEqual(people[0].get("person_db_id"), 1)
        self.assertEqual(people[0].get("face_id"), "Bret")

    def test_single_visible_face_recenters_neck_lift_and_pitch(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 160, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=200.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile") as set_profile,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        # Corrections are relative to the STARTING pose (_set_servo_positions: neck 6000),
        # not the configured neutral — a left-of-center face pulls the neck below 6000.
        self.assertLess(updates[neck_ch], 6000)
        # The face sits above center, but the lift starts (6000) above the eye-level
        # match ceiling — so the lift HOLDS (no update) and the head's PITCH alone
        # supplies the upward gaze. Matching never stretches the neck above the ceiling.
        self.assertNotIn(lift_ch, updates)
        # Both head servos are written together; decoded, the head pitched UP and
        # kept its tilt (straight: no attentive cock is scheduled on the first tick).
        self.assertTrue(set(head_pose.HEAD_PAIR).issubset(updates))
        pitch, tilt = _decoded_head(updates)
        self.assertGreater(pitch, 0.0)
        self.assertLessEqual(pitch, abs(head_pose.pitch_from_legacy_delta(
            c.config.FACE_TRACKING_TILT_MAX_STEP_QUS)) + 0.01)
        self.assertAlmostEqual(tilt, 0.0, delta=0.02)
        self.assertLessEqual(
            abs(updates[neck_ch] - 6000),
            c.config.FACE_TRACKING_NECK_MAX_STEP_QUS * c.config.FACE_TRACKING_EDGE_BOOST_MULT,
        )
        # Two profile writes: neck+lift at tracking speed, BOTH head servos on their own
        # softer profile (the old lone tilt channel's speed/acceleration knobs).
        self.assertEqual(set_profile.call_count, 2)
        self.assertIn(neck_ch, set_profile.call_args_list[0].args[0])
        self.assertEqual(set(set_profile.call_args_list[1].args[0]), set(head_pose.HEAD_PAIR))
        self.assertEqual(set_profile.call_args_list[1].kwargs["speed"],
                         c.config.FACE_TRACKING_TILT_SERVO_SPEED)
        set_baseline.assert_called_once()
        kwargs = set_baseline.call_args.kwargs
        self.assertEqual(kwargs["neck"], updates[neck_ch])
        self.assertEqual(kwargs["lift"], 6000)  # held — no lift update above the match ceiling
        self.assertAlmostEqual(kwargs["pitch"], pitch, delta=0.01)   # fractions, never qus
        self.assertAlmostEqual(kwargs["tilt"], tilt, delta=0.01)

    def _run_one_face_tracking_tick(self, *, face_box, neck_start=6000, speech=False,
                                    rail_damp=True):
        """Run a single fresh face-tracking tick and return the set_servos updates dict."""
        c = self.consciousness
        self_state = c.world_state.get("self_state")
        self_state["servo_positions"] = _level_positions(neck=neck_start)
        self_state["head_pose"] = {"pitch": 0.0, "tilt": 0.0}
        c.world_state.update("self_state", self_state)
        c.world_state.update("people", [{
            "id": "person_1", "person_db_id": 1, "face_id": "Bret",
            "face_visible": True, "face_box": face_box,
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        c._face_tracking_last_error_key = None
        c._face_tracking_last_error_x = 0.0
        c._face_tracking_last_error_y = 0.0
        c._face_tracking_last_error_at = 0.0
        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch("hardware.servos.listening_motion_active", return_value=False),
            mock.patch("hardware.servos.speech_motion_active", return_value=speech),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_RAIL_DAMP_ENABLED", rail_damp),
        ):
            c._step_face_tracking(self.frame)
        return set_servos.call_args.args[0] if set_servos.called else {}

    def test_neck_saturated_at_rail_helper(self):
        c = self.consciousness
        cfg = c.config.SERVO_CHANNELS["neck"]
        hi, lo = int(cfg["max"]), int(cfg["min"])
        # Pinned at the high rail and still demanding more → saturated.
        self.assertTrue(c._neck_saturated_at_rail(hi, hi, cfg))
        self.assertTrue(c._neck_saturated_at_rail(hi - 10, hi, cfg))
        self.assertTrue(c._neck_saturated_at_rail(lo, lo, cfg))           # low rail
        # Pinned high but the target wants to come BACK toward centre → free to move.
        self.assertFalse(c._neck_saturated_at_rail(hi, 6000, cfg))
        self.assertFalse(c._neck_saturated_at_rail(6000, 9000, cfg))      # mid-range
        with mock.patch.object(c.config, "FACE_TRACKING_RAIL_DAMP_ENABLED", False):
            self.assertFalse(c._neck_saturated_at_rail(hi, hi, cfg))

    def test_speech_calms_neck_centering(self):
        # Face far to the right → strong neck correction; speaking should soften it.
        far_right = (940, 300, 120, 120)  # center (1000, 360) in a 1280x720 frame
        neck_ch = self.consciousness.config.SERVO_CHANNELS["neck"]["ch"]
        neutral = int(self.consciousness.config.SERVO_CHANNELS["neck"]["neutral"])
        off = self._run_one_face_tracking_tick(face_box=far_right, speech=False)
        on = self._run_one_face_tracking_tick(face_box=far_right, speech=True)
        off_delta = abs(off.get(neck_ch, neutral) - neutral)
        on_delta = abs(on.get(neck_ch, neutral) - neutral)
        self.assertGreater(off_delta, 0)
        self.assertLess(on_delta, off_delta)  # speaking softens the head correction

    def test_rail_damp_holds_pinned_neck(self):
        # Neck nearly pinned at its max, face far right → centering wants the rail.
        far_right = (1140, 300, 120, 120)  # center (1200, 360)
        neck_ch = self.consciousness.config.SERVO_CHANNELS["neck"]["ch"]
        near_max = int(self.consciousness.config.SERVO_CHANNELS["neck"]["max"]) - 34
        damped = self._run_one_face_tracking_tick(face_box=far_right, neck_start=near_max, rail_damp=True)
        undamped = self._run_one_face_tracking_tick(face_box=far_right, neck_start=near_max, rail_damp=False)
        self.assertNotIn(neck_ch, damped)   # held: no jitter into the rail
        self.assertIn(neck_ch, undamped)    # without damping it crawls into the rail

    def test_active_wander_drives_instead_of_centering(self):
        c = self.consciousness
        self._set_servo_positions()
        with c._idle_wander_lock:
            c._idle_wander["active"] = True
        try:
            with (
                mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
                mock.patch.object(c.time, "monotonic", return_value=400.0),
                mock.patch("hardware.servos.listening_motion_active", return_value=False),
                mock.patch("hardware.servos.speech_motion_active", return_value=False),
                mock.patch("hardware.servos.set_servos") as set_servos,
                mock.patch.object(c, "_drive_idle_head_wander") as drive,
            ):
                c._step_face_tracking(self.frame)
            drive.assert_called_once()       # wander drives the head...
            set_servos.assert_not_called()   # ...instead of the normal centering
        finally:
            with c._idle_wander_lock:
                c._idle_wander["active"] = False

    def test_active_wander_drives_even_without_a_frame(self):
        # BLOCKER regression: the wander must drive (and self-finish) even when frame is
        # None, so a camera hiccup can't strand the head looking away.
        c = self.consciousness
        with c._idle_wander_lock:
            c._idle_wander["active"] = True
        try:
            with (
                mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
                mock.patch.object(c.time, "monotonic", return_value=400.0),
                mock.patch.object(c, "_drive_idle_head_wander") as drive,
            ):
                c._step_face_tracking(None)  # no frame
            drive.assert_called_once()
        finally:
            with c._idle_wander_lock:
                c._idle_wander["active"] = False

    def test_adaptive_rest_learns_downward_pose_from_low_face(self):
        c = self.consciousness
        self._set_servo_positions()
        lift_neutral = c.config.SERVO_CHANNELS["headlift"]["neutral"]
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": lift_neutral,
            "pitch": 0.0,
            "samples": 0,
            "updated_at": 0.0,
        })
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (560, 560, 160, 140),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=225.0),
            mock.patch("hardware.servos.set_servos"),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_ADAPT_ALPHA", 1.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION", 0.0),
        ):
            c._step_face_tracking(self.frame)

        # The low face teaches a rest BELOW the starting pose (the fixture parks
        # the head at 6000). With the 2026-08-19 neutral retune (3600), the
        # learned rest clamps at neutral + REST_MAX_LIFT_OFFSET and may sit
        # above neutral — comparing against neutral was fixture rot.
        self.assertLess(c._adaptive_head_rest["lift"], 6000)
        self.assertLess(c._adaptive_head_rest["pitch"], 0.0)   # a rest pitched DOWN

    def test_adaptive_rest_learns_upward_pose_from_high_face(self):
        c = self.consciousness
        self._set_servo_positions()
        lift_neutral = c.config.SERVO_CHANNELS["headlift"]["neutral"]
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": lift_neutral,
            "pitch": 0.0,
            "samples": 0,
            "updated_at": 0.0,
        })
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (560, 80, 160, 140),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=226.0),
            mock.patch("hardware.servos.set_servos"),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_ADAPT_ALPHA", 1.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_MIN_FACE_AREA_FRACTION", 0.0),
        ):
            c._step_face_tracking(self.frame)

        # A high face teaches an upward PITCH rest, but the LIFT rest is capped at the
        # eye-level match ceiling — Rex looks up with his eyes, not a stretched neck.
        ceiling = c._lift_match_ceiling_qus()
        self.assertIsNotNone(ceiling)
        self.assertLessEqual(c._adaptive_head_rest["lift"], ceiling)
        self.assertGreater(c._adaptive_head_rest["pitch"], 0.0)

    def _run_vertical_tick(self, *, face_y, lift_start, ceiling_frac=None):
        """One tracking tick against a vertically-offset face; returns set_servos updates."""
        c = self.consciousness
        self._set_servo_positions()
        self_state = c.world_state.get("self_state")
        self_state["servo_positions"]["headlift"] = lift_start
        c.world_state.update("self_state", self_state)
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (580, face_y, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE))
            stack.enter_context(mock.patch.object(c.time, "monotonic", return_value=250.0))
            set_servos = stack.enter_context(mock.patch("hardware.servos.set_servos"))
            stack.enter_context(mock.patch("hardware.servos.set_motion_profile"))
            stack.enter_context(mock.patch("hardware.servos.set_face_tracking_baseline"))
            stack.enter_context(mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True))
            if ceiling_frac is not None:
                stack.enter_context(mock.patch.object(
                    c.config, "FACE_TRACKING_LIFT_MATCH_CEILING_FRAC", ceiling_frac
                ))
            c._step_face_tracking(self.frame)
        return set_servos.call_args.args[0] if set_servos.call_args else {}

    def test_lift_matching_capped_at_ceiling_high_face(self):
        # Standing adult (face high in frame) with the lift already at the match
        # ceiling: the lift must NOT rise further; the head's pitch supplies the upward gaze.
        c = self.consciousness
        ceiling = c._lift_match_ceiling_qus()
        updates = self._run_vertical_tick(face_y=60, lift_start=ceiling)
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        self.assertNotIn(lift_ch, updates)  # held at the ceiling
        self.assertTrue(set(head_pose.HEAD_PAIR).issubset(updates))
        pitch, tilt = _decoded_head(updates)
        self.assertGreater(pitch, 0.0)              # pitched UP
        self.assertAlmostEqual(tilt, 0.0, delta=0.02)

    def test_low_face_pitches_the_head_down_and_keeps_the_tilt(self):
        # Owner spec: a face low in the frame pitches the head DOWN; the tilt is left
        # where the attentive-cock schedule has it (here: an active cock of +0.2 that the
        # head already sits at — face following must not disturb it).
        c = self.consciousness
        c._attentive_cock.update({"next_at": 1.0, "until": 10_000.0, "tilt": 0.2})
        self._set_servo_positions(pitch=0.0, tilt=0.2)
        c.world_state.update("people", [{
            "id": "person_1", "person_db_id": 1, "face_id": "Bret",
            "face_visible": True, "face_box": (580, 560, 120, 120),   # center (640, 620): low
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=250.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            c._step_face_tracking(self.frame)
        updates = set_servos.call_args.args[0]
        self.assertTrue(set(head_pose.HEAD_PAIR).issubset(updates))
        pitch, tilt = _decoded_head(updates)
        cap = abs(head_pose.pitch_from_legacy_delta(c.config.FACE_TRACKING_TILT_MAX_STEP_QUS))
        self.assertLess(pitch, 0.0)                        # pitched DOWN...
        self.assertGreaterEqual(pitch, -cap - 0.01)        # ...by at most the per-tick cap
        self.assertAlmostEqual(tilt, 0.2, delta=0.02)      # the cock is kept
        self.assertAlmostEqual(set_baseline.call_args.kwargs["tilt"], 0.2, delta=1e-6)
        self.assertLess(set_baseline.call_args.kwargs["pitch"], 0.0)

    def test_pitch_follows_the_degrees_formula(self):
        # One tick, no caps biting: degrees = deg(pitch) - gain * share * (err_y/cy) * 29,
        # then the smoothing alpha, i.e. the V3 program's vertical law.
        c = self.consciousness
        updates = self._run_vertical_tick(face_y=400, lift_start=6000)   # center y 460 of 720
        error_y = 460 - 360.0
        gain = float(c.config.FACE_TRACKING_VERTICAL_GAIN)
        share = float(c.config.FACE_TRACKING_TILT_SHARE)
        alpha = 1.0 - float(c.config.TRACKING_SMOOTHING_FACTOR)
        degrees = 0.0 - gain * share * (error_y / 360.0) * 29.0
        expected = alpha * head_pose.pitch_from_degrees(degrees)
        cap = abs(head_pose.pitch_from_legacy_delta(c.config.FACE_TRACKING_TILT_MAX_STEP_QUS))
        expected = max(-cap, expected)
        pitch, _tilt = _decoded_head(updates)
        self.assertLess(expected, 0.0)
        self.assertAlmostEqual(pitch, expected, delta=0.01)

    def test_attentive_cock_schedule_cocks_then_straightens(self):
        # Every 10-25 s the head settles into a 0.15-0.3 cock to a random side for
        # 3-8 s, then straightens; random is pinned so the schedule is deterministic.
        c = self.consciousness
        c.reset_attentive_cock()
        draws = iter([12.0, 0.25, 4.0, 15.0, 0.18, 5.0])
        with (
            mock.patch.object(c.random, "uniform", side_effect=lambda a, b: next(draws)),
            mock.patch.object(c.random, "choice", side_effect=lambda seq: seq[-1]),
        ):
            self.assertEqual(c._attentive_cock_tilt(100.0), 0.0)     # books the first cock
            self.assertAlmostEqual(c._attentive_cock["next_at"], 112.0)
            self.assertEqual(c._attentive_cock_tilt(111.9), 0.0)
            self.assertAlmostEqual(c._attentive_cock_tilt(112.0), 0.25)   # cock begins (right)
            self.assertAlmostEqual(c._attentive_cock["until"], 116.0)
            self.assertAlmostEqual(c._attentive_cock_tilt(115.9), 0.25)   # held
            self.assertEqual(c._attentive_cock_tilt(116.0), 0.0)         # straightens...
            self.assertAlmostEqual(c._attentive_cock["next_at"], 131.0)  # ...and rebooks
            self.assertEqual(c._attentive_cock_tilt(125.0), 0.0)
            self.assertAlmostEqual(c._attentive_cock_tilt(131.0), 0.18)
        self.assertEqual(c.ATTENTIVE_COCK_INTERVAL_SECS, (10.0, 25.0))
        self.assertEqual(c.ATTENTIVE_COCK_HOLD_SECS, (3.0, 8.0))
        self.assertEqual(c.ATTENTIVE_COCK_TILT, (0.15, 0.3))

    def test_locked_face_eases_the_tilt_toward_the_attentive_cock(self):
        # A centered face (no centering error) with an active cock: the only servo write
        # is the head pair, tilting toward the cock one slew step per tick, pitch untouched.
        c = self.consciousness
        c._attentive_cock.update({"next_at": 1.0, "until": 10_000.0, "tilt": -0.25})
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1", "person_db_id": 1, "face_id": "Bret",
            "face_visible": True, "face_box": (580, 300, 120, 120),   # center (640, 360)
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=250.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
        ):
            c._step_face_tracking(self.frame)
        updates = set_servos.call_args.args[0]
        self.assertEqual(set(updates), set(head_pose.HEAD_PAIR))
        pitch, tilt = _decoded_head(updates)
        self.assertAlmostEqual(pitch, 0.0, delta=0.02)
        self.assertLess(tilt, -0.02)                                # cocking to Rex's left
        self.assertGreaterEqual(tilt, -c.ATTENTIVE_COCK_STEP - 0.02)  # one slew step
        self.assertAlmostEqual(set_baseline.call_args.kwargs["tilt"], -c.ATTENTIVE_COCK_STEP, delta=1e-6)

    def test_camera_moved_counts_both_head_servos_and_the_lift(self):
        # "The head moved, the camera view changed" counts the lift and BOTH head
        # channels (pitch or cock) through head_pose.HEAD_SMALL, not only the neck.
        c = self.consciousness
        key = 777
        level = head_pose.head_pulses(0.0, 0.0)
        head_then = {1: 3600, 2: level[2], 11: level[11]}
        c._last_seen_pose[key] = (0.0, 0.0, dict(head_then))
        c._camera_moved_since.pop(key, None)
        try:
            def moved_with(changes=None):
                head_now = dict(head_then)
                head_now.update(changes or {})
                with mock.patch.object(c, "_camera_pose", return_value=(0.0, 0.0, head_now)):
                    result = c._camera_moved_since_seen(key, 1000.0)
                c._camera_moved_since.pop(key, None)
                return result
            self.assertFalse(moved_with())
            self.assertTrue(moved_with({11: head_then[11] + head_pose.HEAD_SMALL[11] + 1}))   # channel 11
            self.assertFalse(moved_with({11: head_then[11] + head_pose.HEAD_SMALL[11] - 1}))
            self.assertTrue(moved_with({2: head_then[2] - head_pose.HEAD_SMALL[2] - 1}))      # channel 2
            self.assertTrue(moved_with({1: head_then[1] + head_pose.HEAD_SMALL[1] + 1}))      # the lift
            # A cock alone (both servos the same way) reframes too.
            cocked = head_pose.head_pulses(0.0, 0.6)
            self.assertTrue(moved_with({2: cocked[2], 11: cocked[11]}))
            # The old two-field pose (neck deg, base yaw deg) still works unchanged.
            c._last_seen_pose[key] = (0.0, 0.0)
            with mock.patch.object(c, "_camera_pose", return_value=(40.0, 0.0)):
                self.assertTrue(c._camera_moved_since_seen(key, 1000.0))
            c._camera_moved_since.pop(key, None)
            with mock.patch.object(c, "_camera_pose", return_value=(1.0, 0.0, head_then)):
                self.assertFalse(c._camera_moved_since_seen(key, 1000.0))
        finally:
            c._last_seen_pose.pop(key, None)
            c._camera_moved_since.pop(key, None)

    def test_camera_pose_publishes_the_head_pulses(self):
        c = self.consciousness
        self._set_servo_positions(pitch=0.3, tilt=-0.1)
        pair = head_pose.head_pulses(0.3, -0.1)
        with mock.patch("intelligence.motion_agency._come_neck_bearing_deg", return_value=5.0), \
             mock.patch("intelligence.motion_agency._base_yaw_deg", return_value=None):
            neck, yaw, head = c._camera_pose()
        self.assertEqual((neck, yaw), (5.0, None))
        self.assertEqual(head, {1: 6000, 2: pair[2], 11: pair[11]})

    def test_lift_matching_below_ceiling_may_still_rise_to_it(self):
        # From well below the ceiling, an above-center face may raise the lift — but
        # only up to the ceiling, never past it.
        c = self.consciousness
        ceiling = c._lift_match_ceiling_qus()
        start = ceiling - 120
        updates = self._run_vertical_tick(face_y=60, lift_start=start)
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        self.assertIn(lift_ch, updates)
        self.assertGreater(updates[lift_ch], start)
        self.assertLessEqual(updates[lift_ch], ceiling)

    def test_lift_matching_downward_is_unrestricted(self):
        # A low face (child / seated person) still pulls the lift DOWN freely.
        c = self.consciousness
        updates = self._run_vertical_tick(face_y=560, lift_start=6000)
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        self.assertIn(lift_ch, updates)
        self.assertLess(updates[lift_ch], 6000)

    def test_lift_ceiling_disabled_restores_upward_matching(self):
        # frac >= 1.0 turns the ceiling off: a high face raises the lift as before.
        c = self.consciousness
        updates = self._run_vertical_tick(face_y=60, lift_start=6000, ceiling_frac=1.0)
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        self.assertIn(lift_ch, updates)
        self.assertGreater(updates[lift_ch], 6000)

    def test_speaker_gaze_center_search_uses_adaptive_vertical_rest(self):
        c = self.consciousness
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        # A rest INSIDE the ±REST_MAX_LIFT_OFFSET clamp about neutral (3600 since
        # the 2026-08-19 retune) passes through untouched; the old fixture value
        # 5480 now clamps and asserted the clamp, not the pass-through.
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": 4400,
            "pitch": -0.4,
            "samples": 4,
            "updated_at": 100.0,
        })

        with (
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            targets = c._speaker_gaze_search_targets(0.0, 0.0, tilt=0.0)

        self.assertEqual(targets[lift_ch], 4400)
        pitch, tilt = _decoded_head(targets)
        self.assertAlmostEqual(pitch, -0.4, delta=0.01)   # the rest pitch passes through
        self.assertAlmostEqual(tilt, 0.0, delta=0.01)

    def test_speaker_gaze_search_down_end_and_random_tilt(self):
        # The "down" end is pitch -SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION, never past the
        # adaptive rest toward UP; the visor sits at the capture pose (inverted visor: NOT
        # the channel max, which is now fully down); a scan carries a +/-0.3 cock.
        c = self.consciousness
        visor_ch = c.config.SERVO_CHANNELS["visor"]["ch"]
        frac = float(c.config.SPEAKER_GAZE_SEARCH_DOWN_TILT_FRACTION)
        self.assertAlmostEqual(frac, 0.65)
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({"lift": 3600, "pitch": 0.0, "samples": 4, "updated_at": 100.0})
        with (
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            down = c._speaker_gaze_search_targets(0.0, 1.0, tilt=0.0)
            with mock.patch.object(c.random, "choice", side_effect=lambda seq: seq[0]):
                left = c._speaker_gaze_search_targets(0.0, 0.0)
            with mock.patch.object(c.random, "choice", side_effect=lambda seq: seq[-1]):
                right = c._speaker_gaze_search_targets(0.0, 0.0)
            # A rest already pitched further down than the fraction is kept (never pulled up).
            c._adaptive_head_rest.update({"pitch": -0.7})
            deeper = c._speaker_gaze_search_targets(0.0, 1.0, tilt=0.0)
        self.assertAlmostEqual(_decoded_head(down)[0], -frac, delta=0.01)
        self.assertEqual(down[visor_ch], int(c.config.VISOR_CAMERA_CAPTURE_QUS))
        self.assertNotEqual(down[visor_ch], int(c.config.SERVO_CHANNELS["visor"]["max"]))
        self.assertAlmostEqual(_decoded_head(left)[1], -c.SEARCH_TILT, delta=0.01)
        self.assertAlmostEqual(_decoded_head(right)[1], c.SEARCH_TILT, delta=0.01)
        self.assertAlmostEqual(_decoded_head(left)[0], 0.0, delta=0.01)
        self.assertAlmostEqual(_decoded_head(deeper)[0], -0.7, delta=0.01)
        for targets in (down, left, right, deeper):
            self.assertTrue(set(head_pose.HEAD_PAIR).issubset(targets))

    def test_face_loss_eases_vertical_pose_toward_adaptive_rest(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        c._adaptive_head_rest.clear()
        c._adaptive_head_rest.update({
            "lift": 5500,
            "pitch": -0.4,
            "samples": 3,
            "updated_at": 100.0,
        })

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=260.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile") as set_profile,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_ADAPTIVE_REST_ENABLED", True),
            mock.patch.object(c.config, "FACE_TRACKING_REST_RETURN_AFTER_LOST_SECS", 0.0),
            mock.patch.object(c.config, "FACE_TRACKING_REST_RETURN_MAX_STEP_QUS", 55),
        ):
            c._step_face_tracking(self.frame, [])

        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        updates = set_servos.call_args.args[0]
        self.assertEqual(updates[lift_ch], 5945)
        # The pitch eases toward the -0.4 rest one capped step (55 old q-us -> ~0.046)
        # at a time, both head servos written together, the tilt staying straight.
        step = abs(head_pose.pitch_from_legacy_delta(c.config.FACE_TRACKING_REST_RETURN_MAX_STEP_QUS))
        pitch, tilt = _decoded_head(updates)
        self.assertAlmostEqual(pitch, -step, delta=0.01)
        self.assertAlmostEqual(tilt, 0.0, delta=0.01)
        set_profile.assert_called_once()
        set_baseline.assert_called_once()
        kwargs = set_baseline.call_args.kwargs
        self.assertEqual((kwargs["neck"], kwargs["lift"]), (6000, 5945))
        self.assertAlmostEqual(kwargs["pitch"], -step, delta=1e-6)
        self.assertEqual(kwargs["tilt"], 0.0)

    def test_recent_speaker_face_beats_larger_bystander(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [
            {
                "id": "person_1",
                "person_db_id": 1,
                "face_id": "Bret",
                "face_visible": True,
                "face_box": (100, 180, 80, 100),
            },
            {
                "id": "person_2",
                "person_db_id": 2,
                "face_id": "Other",
                "face_visible": True,
                "face_box": (900, 160, 240, 240),
            },
        ])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=200.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
            mock.patch.object(c.config, "FACE_TRACKING_VERTICAL_ENABLED", True),
        ):
            c.note_speaker_gaze_intent(1, reason="speech", force_search=False)
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")
        # Relative to the STARTING neck pose (6000), the speaker's left-side face pulls left.
        self.assertLess(updates[neck_ch], 6000)

    def test_search_holds_each_waypoint_still_before_advancing(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])

        servo_mod = mock.MagicMock()
        servo_mod.manual_override_enabled.return_value = False

        with mock.patch.object(c.time, "monotonic", return_value=1000.0):
            c.note_speaker_gaze_intent(
                None, unknown_voice=True, reason="startup", force_search=True,
            )

        settle = float(c.config.SPEAKER_GAZE_SEARCH_SETTLE_SECS)
        dwell = float(c.config.SPEAKER_GAZE_SEARCH_DWELL_SECS)
        hold = settle + dwell

        # First call commits waypoint #1 — exactly one servo move issued.
        pose1 = c._step_speaker_gaze_search(servo_mod, dict(c._speaker_gaze_intent), 1000.0)
        self.assertIsNotNone(pose1)
        self.assertEqual(servo_mod.set_servos.call_count, 1)

        # Through the entire settle+dwell window the head holds still: no further
        # servo commands, and the held pose label is reported unchanged.
        for dt in (settle * 0.5, settle + dwell * 0.5, hold - 0.01):
            pose = c._step_speaker_gaze_search(
                servo_mod, dict(c._speaker_gaze_intent), 1000.0 + dt,
            )
            self.assertEqual(pose, pose1)
        self.assertEqual(servo_mod.set_servos.call_count, 1)

        # Once the dwell elapses, the next call advances to waypoint #2.
        c._step_speaker_gaze_search(
            servo_mod, dict(c._speaker_gaze_intent), 1000.0 + hold + 0.01,
        )
        self.assertEqual(servo_mod.set_servos.call_count, 2)

    def test_greeting_height_maps_vertical_to_lift(self):
        c = self.consciousness
        lift_cfg = c.config.SERVO_CHANNELS["headlift"]
        neutral = int(lift_cfg["neutral"])

        low_lift, low_pitch = c._greeting_height_targets("low")
        high_lift, high_pitch = c._greeting_height_targets("high")
        center_lift, center_pitch = c._greeting_height_targets("center")
        # "low" pitches the head down with the lift drop, "high" up a touch, "center" level.
        self.assertEqual((low_pitch, high_pitch, center_pitch), (-0.5, 0.4, 0.0))

        # "Well below/above" is measured against the travel actually available on
        # each side of neutral, so retuning the channel limits can't make these
        # thresholds unreachable (the min moved to 650 us on 2026-08-23).
        down_travel = neutral - int(lift_cfg["min"])
        up_travel = int(lift_cfg["max"]) - neutral
        self.assertLess(low_lift, neutral - down_travel * 0.5)     # head drops well below neutral
        self.assertGreater(high_lift, neutral + up_travel * 0.5)   # head rises well above neutral
        self.assertEqual(center_lift, neutral)
        self.assertGreaterEqual(low_lift, int(lift_cfg["min"]))
        self.assertLessEqual(high_lift, int(lift_cfg["max"]))

    def test_apply_greeting_height_pins_low_and_holds(self):
        c = self.consciousness
        lift_ch = int(c.config.SERVO_CHANNELS["headlift"]["ch"])
        neutral = int(c.config.SERVO_CHANNELS["headlift"]["neutral"])

        with (
            mock.patch.object(c.time, "monotonic", return_value=500.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
        ):
            c._apply_greeting_height("low")

        set_servos.assert_called_once()
        targets = set_servos.call_args.args[0]
        self.assertLess(targets[lift_ch], neutral)  # head went low
        self.assertTrue(set(head_pose.HEAD_PAIR).issubset(targets))   # the pair, together
        self.assertAlmostEqual(_decoded_head(targets)[0], -0.5, delta=0.01)   # pitched down
        self.assertTrue(set_baseline.called)
        self.assertLess(set_baseline.call_args.kwargs.get("lift"), neutral)
        self.assertEqual(set_baseline.call_args.kwargs.get("pitch"), -0.5)
        # The head is pinned so it holds still long enough for dlib to lock.
        self.assertTrue(c.directed_gaze_hold_active(500.0))

    def test_presence_fallback_trigger_spawns_once_when_room_empty(self):
        c = self.consciousness
        c._process_started_mono = 1000.0
        c._last_face_seen_at = 0.0
        c._startup_presence_evidence_at = 0.0
        c._startup_presence_fallback_started = False
        c._startup_presence_fallback_active = False
        snapshot = {"people": [], "crowd": {"count": 0}}

        # 20s in: past the dlib scan window (13.5+0.5) and inside the startup window.
        with (
            mock.patch.object(c.time, "monotonic", return_value=1020.0),
            mock.patch.object(c, "threading") as threading_mod,
        ):
            c._step_startup_presence_fallback_trigger(snapshot)
            threading_mod.Thread.assert_called_once()
            self.assertTrue(c._startup_presence_fallback_started)
            self.assertTrue(c._startup_presence_fallback_active)

            # Second call must not spawn a second worker.
            c._step_startup_presence_fallback_trigger(snapshot)
            threading_mod.Thread.assert_called_once()

    def test_empty_room_comment_blocked_while_fallback_active(self):
        c = self.consciousness
        old_fired = c._startup_empty_room_fired
        c._startup_empty_room_fired = False
        c._startup_presence_fallback_active = True
        try:
            with mock.patch.object(c, "_speak_async") as speak:
                c._step_startup_empty_room_comment(
                    {"people": [], "crowd": {"count": 0}}, mock.MagicMock(),
                )
            speak.assert_not_called()
        finally:
            c._startup_empty_room_fired = old_fired

    def test_presence_fallback_trigger_skips_when_face_seen(self):
        c = self.consciousness
        c._process_started_mono = 1000.0
        c._last_face_seen_at = 1010.0   # dlib already proved presence
        c._startup_presence_evidence_at = 0.0
        c._startup_presence_fallback_started = False
        c._startup_presence_fallback_active = False
        snapshot = {"people": [], "crowd": {"count": 0}}

        with (
            mock.patch.object(c.time, "monotonic", return_value=1020.0),
            mock.patch.object(c, "threading") as threading_mod,
        ):
            c._step_startup_presence_fallback_trigger(snapshot)
            threading_mod.Thread.assert_not_called()
        self.assertFalse(c._startup_presence_fallback_started)

    def test_unknown_speech_without_face_searches_down_first(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c.note_speaker_gaze_intent(
                None,
                unknown_voice=True,
                reason="speech",
                force_search=True,
            )
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        lift_ch = c.config.SERVO_CHANNELS["headlift"]["ch"]
        visor_ch = c.config.SERVO_CHANNELS["visor"]["ch"]
        self.assertLess(updates[lift_ch], c.config.SERVO_CHANNELS["headlift"]["neutral"])
        pitch, tilt = _decoded_head(updates)
        self.assertLess(pitch, 0.0)                                   # looking DOWN
        self.assertAlmostEqual(abs(tilt), c.SEARCH_TILT, delta=0.01)  # with the scan's cock
        self.assertEqual(updates[visor_ch], int(c.config.VISOR_CAMERA_CAPTURE_QUS))
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertTrue(tracking.get("searching"))
        # First beat of the randomized scan drops the gaze down (seated-person bias)
        # without turning the neck — label is "{horiz}_{vert}", so check the vertical.
        self.assertTrue(str(tracking.get("search_pose") or "").endswith("down"))

    def test_directed_gaze_hold_suppresses_search_and_holds_pose(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            # User just said "look down": hold the gaze, then a fresh off-camera
            # voice arrives. The hold must win — no room-scan, head stays put.
            c.hold_directed_gaze("down")
            c.note_speaker_gaze_intent(
                None,
                unknown_voice=True,
                reason="speech",
                force_search=True,
            )
            c._step_face_tracking(self.frame)

        set_servos.assert_not_called()
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertTrue(tracking.get("directed_hold"))
        self.assertFalse(tracking.get("searching"))
        # The wander stands down while a directed gaze is held.
        from sequences import animations
        self.assertTrue(animations._face_tracking_holding_gaze())

    def test_visible_face_during_hold_still_tracks(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 180, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            # Holding "look down" must not block face tracking: if Rex spots
            # someone (e.g. low in frame), he locks on and keeps watching them.
            c.hold_directed_gaze("down")
            c._step_face_tracking(self.frame)

        set_servos.assert_called_once()
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")

    def test_startup_scan_accepts_visible_face_instead_of_searching(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (100, 180, 120, 120),
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=400.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c.request_face_acquisition_scan(reason="startup")
            c._step_face_tracking(self.frame)

        set_servos.assert_called_once()
        self.assertEqual(c._face_tracking_lock.get("key"), "db:1")
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertFalse(tracking.get("searching"))

    def test_face_tracking_damps_direction_reversal(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (1060, 180, 120, 120),
        }])
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 499.9}
        c._face_tracking_suspended_until = 0.0
        c._face_tracking_last_error_key = "db:1"
        c._face_tracking_last_error_x = -500.0
        c._face_tracking_last_error_y = 0.0
        c._face_tracking_last_error_at = 499.9

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=500.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        damped_step = int(c.config.FACE_TRACKING_NECK_MAX_STEP_QUS * c.config.FACE_TRACKING_REVERSAL_DAMPING)
        # A sign-reversed error must move at most the damped step off the STARTING pose
        # (6000) — and the edge boost must not re-inflate a reversal-damped cap.
        self.assertLessEqual(abs(updates[neck_ch] - 6000), damped_step)

    def test_live_tracked_edge_face_slews_neck_responsively(self):
        # Guard the responsiveness fix: an optical-flow (live_tracked) box at the
        # frame edge — the common case, since ~11/12 ticks ride optical flow — must
        # still move the neck a substantial amount per tick. Under the old tuning
        # (neck max_step 120 * live damping 0.45 = 54 qus/tick) this crawled and
        # Rex took seconds to face someone who moved. The face center sits at the
        # vertical midline so only the neck moves.
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (0, 300, 120, 120),  # far-left edge, vertically centered
            "live_tracked": True,
        }])
        c._face_tracking_lock = {}
        c._face_tracking_suspended_until = 0.0
        c._face_tracking_last_error_key = None
        c._face_tracking_last_error_x = None
        c._face_tracking_last_error_y = None

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=300.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_face_tracking_baseline"),
        ):
            c._step_face_tracking(self.frame)

        updates = set_servos.call_args.args[0]
        neck_ch = c.config.SERVO_CHANNELS["neck"]["ch"]
        start = 6000  # the starting neck pose from _set_servo_positions
        neck_move = abs(updates[neck_ch] - start)
        # Left-edge face → neck turns left (below the starting pose).
        self.assertLess(updates[neck_ch], start)
        # Responsive: comfortably more than the old ~54 qus/tick crawl, and never
        # beyond the per-tick step cap — which at the frame edge is scaled up by
        # the edge boost (2026-07-31: a flat cap made a lateral re-face take ~5s).
        self.assertGreaterEqual(neck_move, 150)
        self.assertLessEqual(
            neck_move,
            c.config.FACE_TRACKING_NECK_MAX_STEP_QUS
            * float(getattr(c.config, "FACE_TRACKING_EDGE_BOOST_MULT", 1.0)),
        )

    @unittest.skipIf(cv2 is None, "OpenCV unavailable")
    def test_live_tracking_people_advances_box_between_recognition_ticks(self):
        c = self.consciousness
        c._face_tracking_tracker = None
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": True,
            "face_box": (20, 30, 36, 36),
        }])

        c._live_face_tracking_people(self._frame_with_patch(20, 30))
        c.world_state.update("people", [{
            "id": "person_1",
            "person_db_id": 1,
            "face_id": "Bret",
            "face_visible": False,
            "face_missing": True,
            "face_box": None,
        }])
        tracked = c._live_face_tracking_people(self._frame_with_patch(34, 39))

        self.assertTrue(tracked[0]["live_tracked"])
        self.assertTrue(tracked[0]["face_visible"])
        self.assertFalse(tracked[0]["face_missing"])
        self.assertGreater(tracked[0]["face_box"][0], 30.0)
        self.assertGreater(tracked[0]["face_box"][1], 36.0)

    def test_existing_face_lock_does_not_immediately_switch_to_other_face(self):
        c = self.consciousness
        self._set_servo_positions()
        c.world_state.update("people", [{
            "id": "person_2",
            "person_db_id": 2,
            "face_id": "Other",
            "face_visible": True,
            "face_box": (960, 180, 120, 120),
        }])
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 100.0}
        c._face_tracking_suspended_until = 0.0

        with (
            mock.patch.object(c.state_module, "get_state", return_value=State.ACTIVE),
            mock.patch.object(c.time, "monotonic", return_value=102.0),
            mock.patch("hardware.servos.set_servos") as set_servos,
            mock.patch("hardware.servos.set_face_tracking_baseline") as set_baseline,
            mock.patch.object(c.config, "FACE_TRACKING_LOST_HOLD_SECS", 4.0),
        ):
            c._step_face_tracking(self.frame)

        set_servos.assert_not_called()
        set_baseline.assert_not_called()
        tracking = c.world_state.get("self_state").get("face_tracking") or {}
        self.assertEqual(tracking.get("lock_key"), "db:1")
        self.assertTrue(tracking.get("holding_lost_lock"))

    def test_recent_face_tracking_lock_suppresses_departure(self):
        c = self.consciousness
        c._face_tracking_lock = {"key": "db:1", "person_id": 1, "last_seen_at": 100.0}

        with (
            mock.patch.object(c.config, "FACE_TRACKING_LOST_HOLD_SECS", 8.0),
            mock.patch.object(c.config, "PRESENCE_ENGAGED_DEPARTURE_CONFIRM_SECS", 12.0),
        ):
            self.assertTrue(c._face_tracking_recently_held_person(1, 111.0))
            self.assertFalse(c._face_tracking_recently_held_person(1, 113.0))

    def test_wander_stands_down_when_visible_face_box_exists(self):
        from sequences import animations

        with mock.patch.object(
            animations.world_state,
            "get",
            return_value=[{
                "person_db_id": 1,
                "face_visible": True,
                "face_missing": False,
                "face_box": (40, 120, 100, 140),
            }],
        ):
            self.assertTrue(animations._face_tracking_holding_gaze())


if __name__ == "__main__":
    unittest.main()
