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
            self.assertEqual(body_mood.head_bias(), (0, 0))
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
        # headlift: + = up; headtilt: - = chin up (inverted)
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        lift, tilt = body_mood.head_bias()
        self.assertGreater(lift, 0)   # head raised
        self.assertLess(tilt, 0)      # chin up
        body_mood.set_mood("sad", intensity=1.0, ttl=60)
        lift, tilt = body_mood.head_bias()
        self.assertLess(lift, 0)      # head drooped
        self.assertGreater(tilt, 0)   # chin down

    def test_head_bias_scales_with_intensity(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        full = body_mood.head_bias()
        body_mood.set_mood("proud", intensity=0.5, ttl=60)
        half = body_mood.head_bias()
        self.assertAlmostEqual(half[0], full[0] / 2, delta=2)

    def test_head_scale_config_zeroes_bias(self):
        body_mood.set_mood("proud", intensity=1.0, ttl=60)
        with mock.patch.object(config, "BODY_MOOD_HEAD_SCALE", 0.0):
            self.assertEqual(body_mood.head_bias(), (0, 0))

    def test_visor_never_below_lens_clear_floor(self):
        floor = body_mood.visor_lens_clear_floor()
        # Sweep moods AND intensities — the interpolation must never dip below the floor.
        for m in ("proud", "giddy", "sad", "bored", "annoyed", "suspicious", "surprised"):
            for inten in (1.0, 0.6, 0.26):
                body_mood.set_mood(m, intensity=inten, ttl=60)
                target = body_mood.visor_target()
                self.assertIsNotNone(target, (m, inten))
                self.assertGreaterEqual(target, floor, (m, inten))  # lens-clear floor
                self.assertLessEqual(target, 6976, (m, inten))      # max open

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
            self.assertEqual(c._mood_rest_bias(), (0, 0))

    def test_rest_bias_clamped_to_config_offsets(self):
        from intelligence import consciousness as c
        # Force an absurdly large head bias; _mood_rest_bias must clamp it.
        with mock.patch.object(body_mood, "head_bias", return_value=(99999, -99999)):
            with mock.patch.object(body_mood, "enabled", return_value=True):
                lift, tilt = c._mood_rest_bias()
        self.assertEqual(lift, int(config.BODY_MOOD_REST_MAX_LIFT_OFFSET_QUS))
        self.assertEqual(tilt, -int(config.BODY_MOOD_REST_MAX_TILT_OFFSET_QUS))

    def test_rest_return_composes_mood_bias_into_head_target(self):
        from intelligence import consciousness as c
        body_mood.set_mood("proud", intensity=1.0, ttl=60)  # head up (lift > 0)
        servo = mock.Mock()
        servo.manual_override_enabled.return_value = False
        servo.speech_motion_active.return_value = False
        servo.listening_motion_active.return_value = False
        lift_ch = int(config.SERVO_CHANNELS["headlift"]["ch"])
        neutral_lift = int(config.SERVO_CHANNELS["headlift"]["neutral"])
        with mock.patch.object(c, "_current_servo_position", return_value=neutral_lift):
            moved = c._step_adaptive_head_rest_return(servo, now=1000.0, lost_age_secs=5.0)
        self.assertTrue(moved)
        servo.set_servos.assert_called_once()
        updates = servo.set_servos.call_args.args[0]
        # Proud → head raised above neutral on the rest pose.
        self.assertIn(lift_ch, updates)
        self.assertGreater(updates[lift_ch], neutral_lift)

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
        self.assertGreaterEqual(pos, 6400)  # lens-clear

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

    def test_visor_released_to_lens_clear_floor_when_mood_ends(self):
        # SAFETY REGRESSION GUARD: when a mood decays the visor must be released to the
        # LENS-CLEAR FLOOR (6400 / VISOR_HALF), never to the servo neutral (6000) which
        # sits below the floor and would partially cover the camera lens.
        from intelligence import consciousness as c
        servo = self._servo()
        visor_ch = int(config.SERVO_CHANNELS["visor"]["ch"])
        floor = int(body_mood.visor_lens_clear_floor())
        self.assertEqual(floor, 6400)
        self.assertGreater(floor, int(config.SERVO_CHANNELS["visor"]["neutral"]))  # 6400 > 6000
        breathing = mock.Mock()
        patches = [
            mock.patch("hardware.servos.manual_override_enabled", servo.manual_override_enabled),
            mock.patch("hardware.servos.speech_motion_active", servo.speech_motion_active),
            mock.patch("hardware.servos.listening_motion_active", servo.listening_motion_active),
            mock.patch("hardware.servos.set_motion_profile"),
            mock.patch("hardware.servos.set_breathing_emotion", breathing),
            mock.patch.object(config, "BODY_MOOD_IDLE_GESTURE_ENABLED", False),
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
                self.assertGreaterEqual(pos, floor)          # never below the lens floor
            breathing.assert_any_call("excited")
            breathing.reset_mock()
            # Tick 2: mood gone → visor RELEASED to the lens-clear floor (not 6000), and
            # breathing released back to neutral, each exactly once.
            body_mood.clear()
            with mock.patch("hardware.servos.set_servo") as set_servo:
                c._step_mood_expression({}, mock.Mock())
                set_servo.assert_called_once_with(visor_ch, floor)
                self.assertGreaterEqual(set_servo.call_args.args[1], floor)
                self.assertFalse(c._mood_owns_visor)
            breathing.assert_called_once_with("neutral")
            breathing.reset_mock()
            # Tick 3: fully released → no redundant visor or breathing commands.
            with mock.patch("hardware.servos.set_servo") as set_servo:
                c._step_mood_expression({}, mock.Mock())
                set_servo.assert_not_called()
            breathing.assert_not_called()
        finally:
            for p in reversed(patches):
                p.stop()


if __name__ == "__main__":
    unittest.main()
