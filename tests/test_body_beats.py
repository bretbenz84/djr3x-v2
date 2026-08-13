import unittest
from unittest import mock


class BodyBeatAnimationTest(unittest.TestCase):
    def tearDown(self):
        from sequences import animations
        animations._last_spontaneous_beat_at = 0.0

    def _play_and_capture(self, beat_name):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }
        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.random, "choice", return_value=1),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle"),
            mock.patch.object(animations.servos, "resume_arm_idle"),
        ):
            self.assertTrue(animations.play_body_beat(beat_name, async_=False))
        return moves

    def test_neck_center_uses_configured_range_midpoint(self):
        from sequences import animations

        neck_cfg = animations.config.SERVO_CHANNELS["neck"]
        expected_midpoint = (int(neck_cfg["min"]) + int(neck_cfg["max"])) // 2

        self.assertEqual(animations.NECK_CENTER, expected_midpoint)

    def test_offended_recoil_uses_inverted_upward_headtilt(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.random, "choice", return_value=1),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle"),
            mock.patch.object(animations.servos, "resume_arm_idle"),
        ):
            self.assertTrue(animations.play_body_beat("insult", async_=False))

        first_move = moves[0]
        self.assertEqual(first_move[1], animations.HEADLIFT_HIGH)
        self.assertEqual(first_move[2], animations.HEADTILT_UP)
        self.assertLess(first_move[2], animations.HEADTILT_NEUTRAL)
        self.assertEqual(first_move[3], animations.VISOR_OPEN)

    def test_visor_squint_is_halfway_between_neutral_and_closed(self):
        # The angry glower depth: the visor drops to ~halfway between neutral and
        # fully-closed — genuinely over the eyes, below the lens-clear floor.
        from sequences import animations

        halfway = (animations.VISOR_NEUTRAL + animations.VISOR_CLOSED) // 2
        self.assertAlmostEqual(animations.VISOR_SQUINT, halfway, delta=40)
        self.assertLess(animations.VISOR_SQUINT, animations.VISOR_NEUTRAL)    # below neutral
        self.assertGreater(animations.VISOR_SQUINT, animations.VISOR_CLOSED)  # not fully closed
        self.assertLess(animations.VISOR_SQUINT, animations.VISOR_HALF)       # below the camera floor

    def test_anger_flash_drops_visor_over_the_eyes(self):
        # The beat insults route to (anger_flash) must drop the visor DOWN to the
        # squint — below the lens-clear floor — so an insult reads as a glower.
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.random, "choice", return_value=1),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle"),
            mock.patch.object(animations.servos, "resume_arm_idle"),
        ):
            self.assertTrue(animations.play_body_beat("anger_flash", async_=False))

        first_move = moves[0]
        self.assertEqual(first_move[3], animations.VISOR_SQUINT)        # visor dropped to the squint
        self.assertLess(first_move[3], animations.VISOR_HALF)           # below the lens-clear floor

    def test_named_body_beats_are_registered(self):
        from sequences import animations

        self.assertEqual(
            set(animations.body_beat_names()),
            {
                "agreement_nod",
                "anger_flash",
                "disagreement_shake",
                "disbelief_stare",
                "dramatic_visor_peek",
                "disgust_recoil",
                "giddy_wiggle",
                "happy_bounce",
                "offended_recoil",
                "proud_dj_pose",
                "sad_droop",
                "surprise_pop",
                "suspicious_glance",
                "thinking_tilt",
                "tiny_victory_dance",
                "eye_roll",
                "double_take",
                "mic_drop",
                "spit_take",
            },
        )

    def test_surprise_pop_opens_visor_like_raised_eyebrows(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        snapshot = {
            0: animations.NECK_CENTER,
            1: animations.HEADLIFT_NEUTRAL,
            2: animations.HEADTILT_NEUTRAL,
            3: animations.VISOR_HALF,
        }

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
        ):
            self.assertTrue(animations.play_body_beat("surprise", async_=False))

        first_move = moves[0]
        self.assertEqual(first_move[1], animations.HEADLIFT_HIGH)
        # The heavy head must NOT be whipped on the tilt rod for surprise — the
        # pop is expressed by full lift + full visor with the headtilt parked.
        self.assertNotIn(2, first_move)
        self.assertEqual(first_move[3], animations.VISOR_OPEN)

    def test_wake_word_ack_wave_moves_hand_and_elbow_together(self):
        from sequences import animations

        moves = []
        snapshot = {
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.servos, "pause_arm_idle") as pause,
            mock.patch.object(animations.servos, "resume_arm_idle") as resume,
        ):
            self.assertTrue(animations.wake_word_ack_wave(count=2, async_=False))

        pause.assert_called_once()
        resume.assert_called_once()
        self.assertEqual(
            moves[0],
            {
                7: animations.HEROARM_FORWARD,
                4: animations.ELBOW_NEUTRAL,
                5: animations.HAND_NEUTRAL,
            },
        )
        self.assertEqual(moves[1], {4: animations.ELBOW_UP, 5: animations.HAND_RIGHT})
        self.assertEqual(moves[2], {4: animations.ELBOW_DOWN, 5: animations.HAND_LEFT})
        self.assertEqual(moves[3], {4: animations.ELBOW_UP, 5: animations.HAND_RIGHT})
        self.assertEqual(moves[4], {4: animations.ELBOW_DOWN, 5: animations.HAND_LEFT})
        self.assertEqual(moves[-1], snapshot)

    def test_wave_back_gesture_sweeps_wrist_full_travel_count_times(self):
        from sequences import animations

        # Every commanded position, from EITHER servo entry point: the wrist and the
        # elbow now move together, so the gesture batches them through set_servos()
        # rather than issuing one set_servo() per channel. Record both so the assertions
        # below test the motion, not the call shape.
        sets = []        # (channel, position)
        speeds = []      # (channel, speed) from set_speed
        snapshot = {
            4: animations.ELBOW_NEUTRAL,
            5: animations.HAND_NEUTRAL,
            7: animations.HEROARM_NEUTRAL,
        }
        hand_min = animations.config.SERVO_CHANNELS["hand"]["min"]
        hand_max = animations.config.SERVO_CHANNELS["hand"]["max"]
        elbow_max = animations.config.SERVO_CHANNELS["elbow"]["max"]
        elbow_hi = min(elbow_max, animations.ELBOW_UP + animations.config.WAVE_BACK_ELBOW_WAVE_QUS)
        default_speed = animations.config.SERVO_DEFAULT_SPEED

        with (
            mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
            mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
            mock.patch.object(animations.time, "sleep", return_value=None),
            mock.patch.object(animations.servos, "set_servo", side_effect=lambda ch, pos: sets.append((ch, pos))),
            mock.patch.object(animations.servos, "set_servos",
                              side_effect=lambda targets: sets.extend(targets.items())),
            mock.patch.object(animations.servos, "set_speed", side_effect=lambda ch, sp: speeds.append((ch, sp))),
            mock.patch.object(animations.servos, "set_acceleration"),
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "pause_arm_idle") as pause,
            mock.patch.object(animations.servos, "resume_arm_idle") as resume,
        ):
            self.assertTrue(animations.wave_back_gesture(count=4, async_=False))

        pause.assert_called_once()
        resume.assert_called_once()
        # The wrist (ch5) is driven to BOTH full travel limits exactly `count` times.
        self.assertEqual(sum(1 for ch, pos in sets if ch == 5 and pos == hand_max), 4)
        self.assertEqual(sum(1 for ch, pos in sets if ch == 5 and pos == hand_min), 4)
        # The elbow (ch4) bobs in sync with the wrist — one lift per outbound sweep.
        self.assertEqual(sum(1 for ch, pos in sets if ch == 4 and pos == elbow_hi), 4)
        # Arm raised (elbow up + hero arm forward) before the wave.
        self.assertIn((4, animations.ELBOW_UP), sets)
        self.assertIn((7, animations.HEROARM_FORWARD), sets)
        # The wrist channel was sped up then restored to the default speed.
        self.assertTrue(any(ch == 5 and sp != default_speed for ch, sp in speeds))
        self.assertEqual(speeds[-1], (7, default_speed))  # last action restores defaults
        self.assertTrue(any(ch == 5 and sp == default_speed for ch, sp in speeds))
        # Arm lowered back to the captured pose at the end.
        move_to.assert_called_once()
        self.assertEqual(move_to.call_args.args[0], snapshot)

    def test_half_period_override_speeds_up_or_slows_the_wrist(self):
        from sequences import animations

        snapshot = {4: animations.ELBOW_NEUTRAL, 5: animations.HAND_NEUTRAL, 7: animations.HEROARM_NEUTRAL}
        hand_max = animations.config.SERVO_CHANNELS["hand"]["max"]

        def wrist_speed_for(hp):
            # One ordered log of both call kinds. The wave speed can't be picked off as
            # "the first non-default speed on ch5" any more: the raise phase sets its own
            # (half_period-independent) glide speed on every arm channel first. The wave
            # speed is the last one set on the wrist before it actually sweeps.
            events = []  # ("speed", ch, value) | ("pos", ch, value)
            with (
                mock.patch.object(animations._state_module, "get_state", return_value=animations._State.ACTIVE),
                mock.patch.object(animations, "_current_body_pose", return_value=snapshot),
                mock.patch.object(animations.time, "sleep", return_value=None),
                mock.patch.object(animations.servos, "set_servo",
                                  side_effect=lambda ch, pos: events.append(("pos", ch, pos))),
                mock.patch.object(animations.servos, "set_servos",
                                  side_effect=lambda targets: events.extend(
                                      ("pos", ch, pos) for ch, pos in targets.items())),
                mock.patch.object(animations.servos, "set_speed",
                                  side_effect=lambda ch, sp: events.append(("speed", ch, sp))),
                mock.patch.object(animations.servos, "set_acceleration"),
                mock.patch.object(animations.servos, "move_to"),
                mock.patch.object(animations.servos, "pause_arm_idle"),
                mock.patch.object(animations.servos, "resume_arm_idle"),
            ):
                self.assertTrue(animations.wave_back_gesture(count=1, half_period=hp, async_=False))
            first_sweep = next(
                i for i, (kind, ch, val) in enumerate(events)
                if kind == "pos" and ch == 5 and val == hand_max
            )
            return next(
                val for kind, ch, val in reversed(events[:first_sweep])
                if kind == "speed" and ch == 5
            )

        # A shorter half-period (faster wave) ⇒ higher Maestro speed than a longer one.
        self.assertGreater(wrist_speed_for(0.18), wrist_speed_for(0.48))

    def test_sleep_animation_uses_shutdown_rest_pose(self):
        from sequences import animations

        moves = []

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations.leds_chest, "sleep") as chest_sleep,
            mock.patch.object(animations.leds_head, "sleep") as head_sleep,
            mock.patch.object(animations.servos, "pause_arm_idle") as pause_arm,
            mock.patch.object(animations.servos, "set_motion_profile") as profile,
            mock.patch.object(animations.servos, "latch_sleep_pose") as latch,
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.time, "sleep", return_value=None),
        ):
            animations.sleep()

        chest_sleep.assert_called_once()
        head_sleep.assert_called_once()
        pause_arm.assert_called_once()
        profile.assert_called_once()
        latch.assert_called_once()
        self.assertEqual(moves[0], {3: animations.VISOR_CLOSED})
        self.assertEqual(
            moves[1],
            {
                0: animations.NECK_CENTER,
                1: animations.HEADLIFT_FLOOR,
                2: animations.HEADTILT_DOWN,
                # Re-asserted so a racing end_speech_motion can't leave it open
                # (field 2026-08-13).
                3: animations.VISOR_CLOSED,
                4: animations.ELBOW_NEUTRAL,
                5: animations.HAND_NEUTRAL,
                6: animations.POKERARM_NEUTRAL,
                7: animations.HEROARM_NEUTRAL,
            },
        )

    def test_shutdown_animation_centers_neck_at_configured_midpoint(self):
        from sequences import animations

        moves = []
        neck_cfg = animations.config.SERVO_CHANNELS["neck"]
        expected_midpoint = (int(neck_cfg["min"]) + int(neck_cfg["max"])) // 2

        def record_move(targets, **_kwargs):
            moves.append(dict(targets))

        with (
            mock.patch.object(animations.servos, "stop_breathing") as stop_breathing,
            mock.patch.object(animations.servos, "move_to", side_effect=record_move),
            mock.patch.object(animations.leds_head, "fade_off") as head_fade_off,
            mock.patch.object(animations.leds_chest, "fade_off") as chest_fade_off,
            mock.patch.object(animations.time, "sleep", return_value=None),
        ):
            animations.shutdown()

        stop_breathing.assert_called_once()
        # Visor, neck, head-lift and head-tilt all droop together in ONE move_to
        # so the droid powers down in a single motion (not visor→tilt→lift).
        self.assertEqual(len(moves), 1)
        self.assertEqual(
            moves[0],
            {
                3: animations.VISOR_CLOSED,
                0: expected_midpoint,
                1: animations.HEADLIFT_FLOOR,
                2: animations.HEADTILT_DOWN,
            },
        )
        # Shutdown fades the LEDs out (lifelike power-down), not an instant off.
        head_fade_off.assert_called_once()
        chest_fade_off.assert_called_once()

    def test_wake_animation_restores_active_pose_and_arm_idle(self):
        from sequences import animations

        with (
            mock.patch.object(animations.leds_chest, "active") as chest_active,
            mock.patch.object(animations.leds_head, "active") as head_active,
            mock.patch.object(animations.leds_head, "set_eye_color") as eye_color,
            mock.patch.object(animations.servos, "move_to") as move_to,
            mock.patch.object(animations.servos, "resume_arm_idle") as resume_arm,
        ):
            animations.wake()

        chest_active.assert_called_once()
        move_to.assert_called_once_with(
            {
                1: animations.HEADLIFT_NEUTRAL,
                2: animations.HEADTILT_NEUTRAL,
                3: animations.VISOR_HALF,
            },
            step_us=35,
            step_delay=0.02,
        )
        head_active.assert_called_once()
        eye_color.assert_called_once_with(255, 200, 0)
        resume_arm.assert_called_once()


    def test_eye_roll_brow_lifts_the_visor_and_looks_up(self):
        # Eyes don't move — the visor is the brow. An eye-roll lifts the visor wide
        # (brow up) and tips the chin up ("ugh, really").
        from sequences import animations

        first = self._play_and_capture("eye_roll")[0]
        self.assertEqual(first[3], animations.VISOR_OPEN)       # brow-lift
        self.assertEqual(first[2], animations.HEADTILT_UP)      # look up
        self.assertLess(first[2], animations.HEADTILT_NEUTRAL)  # (inverted: up = lower value)

    def test_double_take_glances_away_then_snaps_back_with_visor_pop(self):
        from sequences import animations

        moves = self._play_and_capture("double_take")
        self.assertNotEqual(moves[0][0], animations.NECK_CENTER)   # first: glance away
        snap = moves[1]                                            # then: SNAP back
        self.assertEqual(snap[0], animations.NECK_CENTER)
        self.assertEqual(snap[3], animations.VISOR_OPEN)          # visor pop
        self.assertEqual(snap[1], animations.HEADLIFT_HIGH)
        self.assertNotIn(2, snap)  # headtilt parked — no whip on the tilt rod

    def test_mic_drop_presents_then_drops_the_hero_arm(self):
        from sequences import animations

        moves = self._play_and_capture("mic_drop")
        self.assertEqual(moves[0][7], animations.HEROARM_FORWARD)  # present the mic
        self.assertEqual(moves[1][7], animations.HEROARM_BACK)     # drop it

    def test_spit_take_recoils_with_a_wide_visor(self):
        from sequences import animations

        first = self._play_and_capture("spit_take")[0]
        self.assertEqual(first[1], animations.HEADLIFT_HIGH)  # sharp recoil up-and-back
        self.assertEqual(first[3], animations.VISOR_OPEN)     # eyes wide (visor)
        self.assertNotIn(2, first)  # headtilt parked — no whip on the tilt rod

    def test_spontaneous_beat_frequency_cooldown(self):
        from sequences import animations

        clock = [1000.0]
        with (
            mock.patch.object(animations.time, "monotonic", lambda: clock[0]),
            mock.patch.object(animations.config, "COMEDY_BEAT_MIN_GAP_SECS", 10.0),
        ):
            animations._last_spontaneous_beat_at = 0.0
            self.assertTrue(animations.spontaneous_beat_allowed())
            animations.note_spontaneous_beat()                        # fired at t=1000
            self.assertFalse(animations.spontaneous_beat_allowed())   # immediately: blocked
            clock[0] += 11.0
            self.assertTrue(animations.spontaneous_beat_allowed())    # gap elapsed: allowed

    def test_spontaneous_play_is_gated_but_explicit_is_not(self):
        from sequences import animations

        # A self-directed beat is suppressed while on cooldown...
        with mock.patch.object(animations, "spontaneous_beat_allowed", return_value=False):
            self.assertFalse(animations.play_body_beat("eye_roll", spontaneous=True, async_=False))
            # ...but an explicit (non-spontaneous) call ignores the cooldown entirely.
            with mock.patch.object(animations, "_run_body_beat", return_value=True) as run:
                self.assertTrue(animations.play_body_beat("eye_roll", spontaneous=False, async_=False))
                run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
