"""
Tests for mood-driven body language:
  • intelligence/body_mood.py — the pure sustained-mood state + posture mapping
  • personality.is_obvious_compliment — the layer-1 compliment pre-check
  • performance_plan — compliment/amusement event → body beat
  • consciousness._mood_rest_bias / _step_adaptive_head_rest_return / _step_mood_expression
    — the head-rest bias + visor expression, gated + hardware-safe
"""

import unittest
from unittest import mock

import config
from hardware import head_pose
from intelligence import body_mood, performance_plan, personality


class BodyMoodStateTest(unittest.TestCase):
    def setUp(self):
        body_mood.clear()
        # Deterministic clock + no ambient fallback so decay lands on neutral.
        self._t = [1000.0]
        self._clock = mock.patch.object(body_mood, "_now", lambda: self._t[0])
        self._clock.start()
        self._ambient = mock.patch.object(config, "BODY_MOOD_AMBIENT_FALLBACK_ENABLED", False)
        self._ambient.start()

    def tearDown(self):
        self._clock.stop()
        self._ambient.stop()
        body_mood.clear()

    def test_disabled_is_neutral_and_inert(self):
        with mock.patch.object(config, "BODY_LANGUAGE_MOOD_ENABLED", False):
            self.assertFalse(body_mood.set_mood("proud"))
            self.assertEqual(body_mood.current_mood(), ("neutral", 0.0))
            self.assertEqual(body_mood.head_bias(), (0, 0.0))
            self.assertIsNone(body_mood.visor_target())

    def test_canonical_mood_and_aliases(self):
        self.assertEqual(body_mood.canonical_mood("complimented"), "proud")
        self.assertEqual(body_mood.canonical_mood("insulted"), "offended")
        self.assertEqual(body_mood.canonical_mood("DELIGHTED"), "giddy")
        self.assertIsNone(body_mood.canonical_mood("banana"))
        self.assertIsNone(body_mood.canonical_mood(""))

    def test_set_and_read(self):
        self.assertTrue(body_mood.set_mood("complimented", intensity=1.0, ttl=10))
        mood, intensity = body_mood.current_mood()
        self.assertEqual(mood, "proud")
        self.assertAlmostEqual(intensity, 1.0, places=3)

    def test_mood_chirp_fires_on_fresh_transition_only(self):
        with mock.patch("audio.sound_effects.play") as play:
            self.assertTrue(body_mood.set_mood("complimented"))   # -> proud
            play.assert_called_once_with("proud")
            play.reset_mock()
            # re-setting the SAME mood must not re-chirp (no fresh transition)
            body_mood.set_mood("praised")                          # also -> proud
            play.assert_not_called()

    def test_amused_mood_chirps_laughing(self):
        with mock.patch("audio.sound_effects.play") as play:
            body_mood.set_mood("funny")                            # -> amused
            play.assert_called_once_with("laughing")

    def test_uncurated_mood_is_silent(self):
        with mock.patch("audio.sound_effects.play") as play:
            body_mood.set_mood("sad")                              # no curated chirp
            play.assert_not_called()

    def test_unknown_mood_rejected(self):
        self.assertFalse(body_mood.set_mood("banana"))
        self.assertFalse(body_mood.set_mood("neutral"))

    def test_linear_decay_to_neutral(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=10)
        self._t[0] += 5.0
        self.assertAlmostEqual(body_mood.current_mood()[1], 0.5, places=2)
        self._t[0] += 6.0  # past ttl
        self.assertEqual(body_mood.current_mood(), ("neutral", 0.0))

    def test_weaker_mood_does_not_stomp_a_strong_one(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        # A clearly weaker, different mood is refused...
        self.assertFalse(body_mood.set_mood("curious", intensity=0.5))
        self.assertEqual(body_mood.current_mood()[0], "proud")
        # ...but a comparably strong one wins.
        self.assertTrue(body_mood.set_mood("curious", intensity=1.0))
        self.assertEqual(body_mood.current_mood()[0], "curious")

    def test_head_bias_signs(self):
        # headlift: + = up (qus); head pitch: a FRACTION of the two-servo head's travel,
        # + = chin up, - = chin down (the old inverted tilt-servo sign is gone).
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        lift, pitch = body_mood.head_bias()
        self.assertGreater(lift, 0)   # head raised
        self.assertGreater(pitch, 0)  # chin up
        self.assertLessEqual(abs(pitch), 1.0)
        for mood in ("angry", "offended"):
            body_mood.set_mood(mood, intensity=1.0, ttl=60)
            self.assertGreater(body_mood.head_bias()[1], 0, mood)   # chin up
        for mood in ("sad", "thinking"):
            body_mood.set_mood(mood, intensity=1.0, ttl=60)
            lift, pitch = body_mood.head_bias()
            self.assertLess(pitch, 0, mood)                          # chin down
        body_mood.set_mood("sad", intensity=1.0, ttl=60)
        self.assertLess(body_mood.head_bias()[0], 0)                 # head drooped

    def test_head_bias_pitch_is_the_old_tilt_delta_converted(self):
        # Each pose's pitch is the OLD tilt-servo delta through pitch_from_legacy_delta,
        # written once as a literal: proud was -180 (chin up), sad +260 (chin down).
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        self.assertAlmostEqual(body_mood.head_bias()[1], head_pose.pitch_from_legacy_delta(-180), delta=0.01)
        body_mood.set_mood("sad", intensity=1.0, ttl=60)
        self.assertAlmostEqual(body_mood.head_bias()[1], head_pose.pitch_from_legacy_delta(260), delta=0.01)

    def test_head_bias_scales_with_intensity(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        full = body_mood.head_bias()
        body_mood.set_mood("proud", intensity=0.5, ttl=60)
        half = body_mood.head_bias()
        self.assertAlmostEqual(half[0], full[0] / 2, delta=2)

    def test_head_scale_config_zeroes_bias(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        with mock.patch.object(config, "BODY_MOOD_HEAD_SCALE", 0.0):
            self.assertEqual(body_mood.head_bias(), (0, 0.0))

    def test_visor_never_past_the_camera_clear_edge(self):
        # Inverted visor: LOW pulses raise it, HIGH pulses lower it. The camera-clear
        # EDGE is the highest pulse a mood may reach; the interpolation must never pass it.
        edge = body_mood.visor_camera_clear_max()
        self.assertEqual(edge, int(config.BODY_MOOD_VISOR_CAMERA_CLEAR_QUS))
        self.assertEqual(edge, 5520)
        self.assertFalse(hasattr(body_mood, "visor_lens_clear_floor"))
        for m in ("proud", "giddy", "sad", "bored", "annoyed", "suspicious", "surprised", "angry"):
            for inten in (1.0, 0.6, 0.26):
                body_mood.set_mood(m, intensity=inten, ttl=60)
                target = body_mood.visor_target()
                self.assertIsNotNone(target, (m, inten))
                self.assertLessEqual(target, edge, (m, inten))        # never past the edge
                self.assertGreaterEqual(target, int(config.SERVO_CHANNELS["visor"]["min"]), (m, inten))

    def test_anger_and_offense_narrow_the_visor_to_the_edge(self):
        # Insult/anger reads as a SQUINT: the visor lowers to exactly the camera-clear
        # edge (the most-closed safe position), never an open "glare".
        edge = body_mood.visor_camera_clear_max()
        for m in ("offended", "angry", "insulted", "mad", "suspicious"):
            body_mood.set_mood(m, intensity=1.0, ttl=60)
            self.assertEqual(body_mood.visor_target(), edge, m)
        # A positive mood opens it (a LOWER pulse than the awake neutral) — the contrast.
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        self.assertLess(body_mood.visor_target(), int(config.SERVO_CHANNELS["visor"]["neutral"]))
        self.assertLess(body_mood.visor_target(), edge)

    def test_squint_relaxes_toward_neutral_as_emotion_decays(self):
        neutral = int(config.SERVO_CHANNELS["visor"]["neutral"])   # 5120, awake
        edge = body_mood.visor_camera_clear_max()                  # 5520
        for mood in ("angry", "suspicious", "offended"):
            body_mood.clear()
            body_mood.set_mood(mood, intensity=1.0, ttl=10)
            self.assertEqual(body_mood.visor_target(), edge)
            self._t[0] += 5
            self.assertAlmostEqual(body_mood.visor_target(), (neutral + edge) / 2, delta=2)
        # Subdued moods lower the visor from the awake neutral but never past the edge;
        # a pleased mood lifts it (a LOWER pulse on the inverted servo). Each keeps its
        # September offset from the awake visor.
        for mood in ("sad", "bored", "thinking"):
            body_mood.clear()
            body_mood.set_mood(mood, intensity=1.0, ttl=10)
            self.assertGreater(body_mood.visor_target(), neutral)
            self.assertLess(body_mood.visor_target(), edge)
        for mood in ("happy", "proud"):
            body_mood.clear()
            body_mood.set_mood(mood, intensity=1.0, ttl=10)
            self.assertLess(body_mood.visor_target(), neutral)
            self.assertGreaterEqual(body_mood.visor_target(), int(config.SERVO_CHANNELS["visor"]["min"]))
        # Surprise throws it fully UP (the visor's lowest pulse).
        body_mood.clear()
        body_mood.set_mood("surprised", intensity=1.0, ttl=10)
        self.assertEqual(body_mood.visor_target(), int(config.SERVO_CHANNELS["visor"]["min"]))

    def test_idle_visor_drifts_in_relaxed_range(self):
        low, high = (int(v) for v in config.SERVO_IDLE_VISOR_RANGE_QUS)
        self.assertEqual((low, high), (4770, 5470))
        targets = [body_mood.idle_visor_target(t / 10) for t in range(140)]
        self.assertGreater(max(targets) - min(targets), 300)
        self.assertEqual(min(targets), low)
        self.assertEqual(max(targets), high)
        self.assertLessEqual(max(targets), body_mood.visor_camera_clear_max())
        # Smooth: a 7 s period over a 350 q-us amplitude moves ~31 q-us per 0.1 s at most.
        self.assertLess(max(abs(a - b) for a, b in zip(targets, targets[1:])), 40)
        with mock.patch.object(config, "SERVO_IDLE_VISOR_ENABLED", False):
            self.assertIsNone(body_mood.idle_visor_target(1.0))

    def test_visor_none_below_min_intensity(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=10)
        self._t[0] += 9.0  # intensity ~0.1, below the 0.25 visor floor
        self.assertIsNone(body_mood.visor_target())

    def test_breathing_emotion_mapping(self):
        body_mood.set_mood("giddy", intensity=1.0, ttl=60)
        self.assertEqual(body_mood.breathing_emotion(), "excited")
        body_mood.set_mood("sad", intensity=1.0, ttl=60)
        self.assertEqual(body_mood.breathing_emotion(), "sad")
        body_mood.set_mood("suspicious", intensity=1.0, ttl=60)
        self.assertIsNone(body_mood.breathing_emotion())

    def test_idle_beat_is_a_real_beat_name(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        beat = body_mood.idle_beat()
        self.assertEqual(beat, "proud_dj_pose")
        self.assertIn(beat, performance_plan.BODY_BEAT_NAMES)

    def test_idle_beat_none_when_faint(self):
        body_mood.set_mood("proud", intensity=0.3, ttl=60)  # below 0.4 idle floor
        self.assertIsNone(body_mood.idle_beat())

    def test_clear_resets(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        body_mood.clear()
        self.assertEqual(body_mood.current_mood(), ("neutral", 0.0))


class BodyMoodAmbientFallbackTest(unittest.TestCase):
    def setUp(self):
        body_mood.clear()

    def tearDown(self):
        body_mood.clear()

    def test_ambient_mood_from_emotion_frame(self):
        # With no event mood, a happy emotion frame yields a mild 'happy' posture.
        fake_frame = mock.Mock(affect="happy")
        with mock.patch("intelligence.emotion_orchestrator.current_frame", return_value=fake_frame):
            with mock.patch.object(config, "BODY_MOOD_AMBIENT_INTENSITY", 0.4):
                mood, intensity = body_mood.current_mood()
        self.assertEqual(mood, "happy")
        self.assertAlmostEqual(intensity, 0.4, places=3)

    def test_ambient_disabled_is_neutral(self):
        fake_frame = mock.Mock(affect="happy")
        with mock.patch("intelligence.emotion_orchestrator.current_frame", return_value=fake_frame):
            with mock.patch.object(config, "BODY_MOOD_AMBIENT_FALLBACK_ENABLED", False):
                self.assertEqual(body_mood.current_mood(), ("neutral", 0.0))


class ComplimentDetectionTest(unittest.TestCase):
    def test_obvious_compliments(self):
        for t in ("you're amazing", "Good job, Rex!", "that was brilliant", "I love you buddy"):
            self.assertTrue(personality.is_obvious_compliment(t), t)

    def test_everyday_compliments_aimed_at_rex(self):
        # Regression for a real robot run: "You're a nice robot" produced no reaction
        # because "nice"/"nice robot" wasn't recognized.
        for t in (
            "You're a nice robot",
            "you are a nice robot",
            "good robot",
            "you're so cool",
            "I like you",
            "you're cute",
            "what a good boy",
        ):
            self.assertTrue(personality.is_obvious_compliment(t), t)

    def test_non_compliments(self):
        # Ambiguous bare words must NOT false-trigger out of context.
        for t in ("what's the weather", "", "you're an idiot", "play some music",
                  "nice weather today", "cool, let's go", "that's a great question"):
            self.assertFalse(personality.is_obvious_compliment(t), t)


class PerformancePlanEventBeatTest(unittest.TestCase):
    def test_compliment_event_maps_to_proud_pose(self):
        self.assertEqual(performance_plan.body_beat_for_event("compliment.detected"), "proud_dj_pose")

    def test_amusement_event_maps_to_giddy(self):
        self.assertEqual(performance_plan.body_beat_for_event("amusement.detected"), "giddy_wiggle")


class ConsciousnessMoodRestBiasTest(unittest.TestCase):
    def setUp(self):
        body_mood.clear()

    def tearDown(self):
        body_mood.clear()

    def test_rest_bias_zero_when_disabled(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        with mock.patch.object(config, "BODY_LANGUAGE_MOOD_ENABLED", False):
            self.assertEqual(c._mood_rest_bias(), (0, 0.0))

    def test_rest_bias_clamped_to_config_offsets(self):
        from intelligence import consciousness as c
        # Force an absurdly large head bias; _mood_rest_bias must clamp it. The pitch
        # clamp knob is still the old tilt servo's q-us (320) converted to a pitch magnitude.
        with mock.patch.object(body_mood, "head_bias", return_value=(99999, -99.0)):
            with mock.patch.object(body_mood, "enabled", return_value=True):
                lift, pitch = c._mood_rest_bias()
        self.assertEqual(lift, int(config.BODY_MOOD_REST_MAX_LIFT_OFFSET_QUS))
        max_pitch = abs(head_pose.pitch_from_legacy_delta(config.BODY_MOOD_REST_MAX_TILT_OFFSET_QUS))
        self.assertAlmostEqual(pitch, -max_pitch, delta=1e-6)
        self.assertAlmostEqual(max_pitch, 0.27, delta=0.01)
        with mock.patch.object(body_mood, "head_bias", return_value=(0, 99.0)):
            with mock.patch.object(body_mood, "enabled", return_value=True):
                self.assertAlmostEqual(c._mood_rest_bias()[1], max_pitch, delta=1e-6)

    def test_rest_return_pitches_the_head_pair_with_the_mood(self):
        # A proud mood (chin up) composes onto the rest pitch: the rest return writes
        # BOTH head servos together, one capped step toward the biased pitch.
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        servo = mock.Mock()
        servo.manual_override_enabled.return_value = False
        servo.speech_motion_active.return_value = False
        servo.listening_motion_active.return_value = False
        with mock.patch.object(c, "_current_head_pose", return_value=(0.0, 0.0)), \
             mock.patch.object(c, "_current_head_pulses", return_value=head_pose.head_pulses(0.0, 0.0)):
            self.assertTrue(c._step_adaptive_head_rest_return(servo, now=1000.0, lost_age_secs=5.0))
        updates = servo.set_servos.call_args.args[0]
        self.assertTrue(set(head_pose.HEAD_PAIR).issubset(updates))
        pitch, tilt = head_pose.head_pose({ch: updates[ch] for ch in head_pose.HEAD_PAIR})
        step = abs(head_pose.pitch_from_legacy_delta(config.FACE_TRACKING_REST_RETURN_MAX_STEP_QUS))
        self.assertGreater(pitch, 0.0)
        self.assertAlmostEqual(pitch, step, delta=0.01)
        self.assertAlmostEqual(tilt, 0.0, delta=0.01)
        kwargs = servo.set_face_tracking_baseline.call_args.kwargs
        self.assertAlmostEqual(kwargs["pitch"], step, delta=1e-6)
        self.assertEqual(kwargs["tilt"], 0.0)

    def test_rest_return_composes_mood_bias_into_head_target(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)  # head up (lift > 0)
        servo = mock.Mock()
        servo.manual_override_enabled.return_value = False
        servo.speech_motion_active.return_value = False
        servo.listening_motion_active.return_value = False
        lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
        # The matching rest base is capped at the eye-level match ceiling; the mood
        # bias is EXPRESSIVE and composes on top of that capped base — from the
        # ceiling, a proud mood still raises the head.
        ceiling = int(c._lift_match_ceiling_qus())
        with mock.patch.object(c, "_current_servo_position", return_value=ceiling):
            moved = c._step_adaptive_head_rest_return(servo, now=1000.0, lost_age_secs=5.0)
        self.assertTrue(moved)
        servo.set_servos.assert_called_once()
        updates = servo.set_servos.call_args.args[0]
        # Proud → head raised above the (capped) rest base. The base is the
        # neutral capped at the match ceiling; since the 2026-08-19 retune the
        # neutral (3600) sits BELOW the ceiling, so the base IS the neutral —
        # asserting "> ceiling" was only true while neutral >= ceiling.
        rest_base = min(int(config.SERVO_CHANNELS["headlift"]["neutral"]), ceiling)
        self.assertIn(lift_ch, updates)
        self.assertGreater(updates[lift_ch], rest_base)

    def test_rest_return_noop_when_no_mood_and_no_learned_rest(self):
        from intelligence import consciousness as c
        servo = mock.Mock()
        servo.manual_override_enabled.return_value = False
        servo.speech_motion_active.return_value = False
        servo.listening_motion_active.return_value = False
        # No mood, no adaptive samples → nothing to express.
        self.assertFalse(c._step_adaptive_head_rest_return(servo, now=1000.0, lost_age_secs=5.0))
        servo.set_servos.assert_not_called()


class ConsciousnessMoodExpressionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness as c
        body_mood.clear()
        c._mood_owns_visor = False
        c._last_mood_breathing = None

    def tearDown(self):
        from intelligence import consciousness as c
        body_mood.clear()
        c._mood_owns_visor = False
        c._last_mood_breathing = None

    def _servo(self, *, manual=False, speech=False, listening=False):
        servo = mock.Mock()
        servo.manual_override_enabled.return_value = manual
        servo.speech_motion_active.return_value = speech
        servo.listening_motion_active.return_value = listening
        return servo

    def test_visor_set_toward_mood_when_idle(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        servo = self._servo()
        visor_ch = int(config.SERVO_CHANNELS["visor"]["ch"])
        with mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled), \
             mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active), \
             mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active), \
             mock.patch("hardware.servos.set_servo") as set_servo, \
             mock.patch("hardware.servos.set_motion_profile"), \
             mock.patch("hardware.servos.set_breathing_emotion"), \
             mock.patch.object(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", False):
            c._step_mood_expression({}, mock.Mock())
        set_servo.assert_called_once()
        ch, pos = set_servo.call_args.args[0], set_servo.call_args.args[1]
        self.assertEqual(ch, visor_ch)
        # Inverted visor: proud OPENS it, i.e. a lower pulse than the awake neutral (5120).
        self.assertLess(pos, int(config.SERVO_CHANNELS["visor"]["neutral"]))
        self.assertGreaterEqual(pos, int(config.SERVO_CHANNELS["visor"]["min"]))

    def test_visor_not_touched_during_speech(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        servo = self._servo(speech=True)
        with mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled), \
             mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active), \
             mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active), \
             mock.patch("hardware.servos.set_servo") as set_servo, \
             mock.patch.object(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", False):
            c._step_mood_expression({}, mock.Mock())
        set_servo.assert_not_called()

    def test_idle_yields_to_listening_and_scripted_head_gestures(self):
        from intelligence import consciousness as c
        for listening, gesture in ((True, False), (False, True)):
            with (mock.patch("hardware.servos.listening_motion_active", return_value=listening),
                  mock.patch("hardware.servos.head_gesture_active", return_value=gesture),
                  mock.patch("hardware.servos.manual_override_enabled", return_value=False),
                  mock.patch("hardware.servos.set_servo") as move,
                  mock.patch.object(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", False)):
                c._step_mood_expression({}, mock.Mock())
                move.assert_not_called()

    def test_idle_yields_to_body_sequence(self):
        from intelligence import consciousness as c
        with (mock.patch("sequences.animations.body_motion_active", return_value=True),
              mock.patch("hardware.servos.set_servo") as move):
            c._step_mood_expression({}, mock.Mock())
            move.assert_not_called()

    def test_manual_override_blocks_all(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        servo = self._servo(manual=True)
        with mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled), \
             mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active), \
             mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active), \
             mock.patch("hardware.servos.set_servo") as set_servo:
            c._step_mood_expression({}, mock.Mock())
        set_servo.assert_not_called()

    def test_disabled_is_noop(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        with mock.patch.object(config, "BODY_LANGUAGE_MOOD_ENABLED", False), \
             mock.patch("hardware.servos.set_servo") as set_servo:
            c._step_mood_expression({}, mock.Mock())
        set_servo.assert_not_called()

    def test_visor_released_to_relaxed_idle_when_mood_ends(self):
        from intelligence import consciousness as c
        servo = self._servo()
        visor_ch = int(config.SERVO_CHANNELS["visor"]["ch"])
        edge = int(body_mood.visor_camera_clear_max())
        self.assertEqual(edge, 5520)
        relaxed = int(config.SERVO_CHANNELS["visor"]["neutral"])
        breathing = mock.Mock()
        patches = [
            mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled),
            mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active),
            mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_breathing_emotion", breathing),
            mock.patch.object(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", False),
            mock.patch.object(body_mood, "idle_visor_target", return_value=relaxed),
            mock.patch.object(config, "BODY_MOOD_AMBIENT_FALLBACK_ENABLED", False),
        ]
        for p in patches:
            p.start()
        try:
            # Tick 1: active proud mood → owns the visor + sets excited breathing.
            body_mood.set_mood("proud", intensity=1.0, ttl=60)
            with mock.patch("hardware.servos.set_servo") as set_servo:
                c._step_mood_expression({}, mock.Mock())
                self.assertTrue(c._mood_owns_visor)
                pos = set_servo.call_args.args[1]
                self.assertLessEqual(pos, edge)              # never past the camera-clear edge
            breathing.assert_any_call("excited")
            breathing.reset_mock()
            # Tick 2: mood gone → relaxed idle, with breathing returned to neutral.
            body_mood.clear()
            with mock.patch("hardware.servos.set_servo") as set_servo:
                c._step_mood_expression({}, mock.Mock())
                set_servo.assert_called_once_with(visor_ch, relaxed)
                self.assertLessEqual(set_servo.call_args.args[1], edge)
                self.assertFalse(c._mood_owns_visor)
            breathing.assert_called_once_with("neutral")
            breathing.reset_mock()
            # Tick 3: idle continues, breathing is not redundantly reasserted.
            with mock.patch("hardware.servos.set_servo") as set_servo:
                c._step_mood_expression({}, mock.Mock())
                set_servo.assert_called_once_with(visor_ch, relaxed)
            breathing.assert_not_called()
        finally:
            for p in reversed(patches):
                p.stop()


if __name__ == "__main__":
    unittest.main()
