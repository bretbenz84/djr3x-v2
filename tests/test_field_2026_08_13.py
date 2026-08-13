"""Field regressions from the 2026-08-13 10:26 run
(logs/conversation-2026-08-13-10-26-13.log).

Four user-visible failures in one three-minute session:

  * "That's not funny." shipped identity.name_correction at 0.95 (the
    "that's not X" branch accepted any word outside a 21-word stoplist) and
    Rex asked for "the right name" out of nowhere;
  * "Go to sleep." was blocked by the system.sleep evidence regex — the ASR's
    trailing period read as extra content, so the first command got a
    conversational ack ("Going quiet.") and no sleep;
  * the sleep that DID land raced the ack clip's end_speech_motion: the
    speech-end release drove the visor back to neutral around the sleep glide
    and nothing ever re-closed it;
  * the same pet was announced twice under two species labels — covered in
    tests/test_animal_returns.py (CrossSpeciesFlipTest).

Plus one from the 11:27 run (logs/conversation-2026-08-13-11-27-02.log): Rex
asked "what's the most annoying part of calibrating those sensors?" and the
ANSWER to his own question was parsed as a memory correction.
"""

import unittest
from unittest import mock

from intelligence import action_router as AR


class SleepEvidenceTest(unittest.TestCase):
    def _reason(self, text):
        return AR.missing_required_evidence_reason(
            text, AR.ActionDecision(action="system.sleep", confidence=0.9))

    def test_transcript_punctuation_is_not_extra_content(self):
        for text in ("Go to sleep.", "go to sleep!", "Shut down.",
                     "Wake up.", "Be quiet…", "Power off."):
            self.assertIsNone(self._reason(text), text)

    def test_bare_commands_still_pass(self):
        for text in ("go to sleep", "shut down", "wake up"):
            self.assertIsNone(self._reason(text), text)

    def test_extra_words_still_blocked(self):
        for text in ("Go to sleep now please", "sleep is great",
                     "I wish you would go to sleep"):
            self.assertEqual(
                self._reason(text), "missing_system_mode_evidence", text)

    def test_punctuated_sleep_passes_the_legacy_turn_policy_gate(self):
        from intelligence import command_parser
        from intelligence import interaction as I

        match = command_parser.parse("Go to sleep.")
        self.assertIsNotNone(match)
        self.assertEqual(match.command_key, "sleep")
        self.assertIsNone(I._legacy_command_execution_block_reason(
            match, text="Go to sleep.", context={}))


class ThatsNotFunnyTest(unittest.TestCase):
    def test_lowercase_adjectives_are_not_name_corrections(self):
        for text in ("That's not funny.", "That's not cool.",
                     "That's not fair.", "that's not nice",
                     "That's not working."):
            self.assertIsNone(AR.classify_explicit_control(text), text)

    def test_real_corrections_still_route(self):
        for text, name in (("that's not Bret, I'm Daniel", "Daniel"),
                           ("call me JT", "JT"),
                           ("my name is Daniel", "Daniel")):
            decision = AR.classify_explicit_control(text)
            self.assertIsNotNone(decision, text)
            self.assertEqual(decision.action, "identity.name_correction", text)
            self.assertEqual(decision.args.get("name"), name, text)
        for text in ("That's not Brad", "that's not my name",
                     "you got my name wrong"):
            decision = AR.classify_explicit_control(text)
            self.assertIsNotNone(decision, text)
            self.assertEqual(decision.action, "identity.name_correction", text)

    def test_llm_route_evidence_gate_matches(self):
        decision = AR.ActionDecision(
            action="identity.name_correction", confidence=0.95)
        self.assertEqual(
            AR.missing_required_evidence_reason("That's not funny.", decision),
            "missing_identity_name_evidence")
        self.assertIsNone(
            AR.missing_required_evidence_reason("That's not Brad", decision))


class SleepPoseRaceTest(unittest.TestCase):
    """The sleep glide must survive the ack clip's end_speech_motion firing
    around it, and the reached pose must hold until wake."""

    def setUp(self):
        from hardware import servos
        servos._sleep_latch.clear()
        servos._manual_override.clear()
        self.addCleanup(servos._sleep_latch.clear)

    def test_sleep_resets_profile_reasserts_visor_and_latches(self):
        from sequences import animations as A

        moves, profiles = [], []
        with mock.patch.object(A.servos, "move_to",
                               side_effect=lambda t, **k: moves.append(dict(t))), \
             mock.patch.object(A.servos, "set_motion_profile",
                               side_effect=lambda chans, **k: profiles.append(list(chans))), \
             mock.patch.object(A.servos, "latch_sleep_pose") as latch, \
             mock.patch.object(A.servos, "pause_arm_idle"), \
             mock.patch.object(A.leds_chest, "sleep"), \
             mock.patch.object(A.leds_head, "sleep"), \
             mock.patch.object(A.time, "sleep"):
            A.sleep()

        latch.assert_called_once()
        self.assertTrue(profiles, "sleep must clear stale slow speed caps")
        self.assertTrue(set(A.config.HEAD_CHANNELS) <= set(profiles[0]))
        visor_ch = 3
        self.assertEqual(moves[0], {visor_ch: A.VISOR_CLOSED})
        self.assertEqual(moves[-1][visor_ch], A.VISOR_CLOSED,
                         "the final move must re-assert the visor closed so a "
                         "racing writer can't leave it half-open")

    def test_wake_releases_the_sleep_latch(self):
        from hardware import servos
        from sequences import animations as A

        servos.latch_sleep_pose()
        with mock.patch.object(A.servos, "move_to"), \
             mock.patch.object(A.servos, "resume_arm_idle"), \
             mock.patch.object(A.leds_chest, "active"), \
             mock.patch.object(A.leds_head, "active"), \
             mock.patch.object(A.leds_head, "set_eye_color"):
            A.wake()
        self.assertFalse(servos._sleep_latch.is_set())

    def test_shutdown_releases_the_sleep_latch(self):
        # This morning's run powered down FROM sleep — the shutdown droop must
        # not be frozen by the sleep latch.
        from hardware import servos
        from sequences import animations as A

        servos.latch_sleep_pose()
        with mock.patch.object(A.servos, "set_manual_override_enabled"), \
             mock.patch.object(A.servos, "stop_breathing"), \
             mock.patch.object(A.servos, "set_motion_profile"), \
             mock.patch.object(A.servos, "move_to"), \
             mock.patch.object(A.servos, "latch_shutdown_pose"), \
             mock.patch.object(A.leds_head, "fade_off"), \
             mock.patch.object(A.leds_chest, "fade_off"), \
             mock.patch.object(A.time, "sleep"):
            A.shutdown()
        self.assertFalse(servos._sleep_latch.is_set())

    def test_sleep_latch_blocks_program_writes(self):
        from hardware import servos

        servos.latch_sleep_pose()
        self.assertTrue(servos._program_servo_updates_blocked())
        servos.release_sleep_latch()
        self.assertFalse(servos._program_servo_updates_blocked())

    def test_end_speech_motion_holds_pose_when_not_awake(self):
        from hardware import servos

        servos._speech_active.set()
        with mock.patch.object(servos, "_automatic_motion_allowed",
                               return_value=False), \
             mock.patch.object(servos, "set_servos") as writes, \
             mock.patch.object(servos, "set_motion_profile") as profile, \
             mock.patch.object(servos, "resume_arm_idle") as resume, \
             mock.patch.object(servos, "set_breathing_emotion"):
            servos.end_speech_motion()
        writes.assert_not_called()
        profile.assert_not_called()
        resume.assert_not_called()
        self.assertFalse(servos._speech_active.is_set(),
                         "bookkeeping must still run")

    def test_end_speech_motion_normal_path_unchanged(self):
        from hardware import servos

        servos._speech_active.set()
        try:
            with mock.patch.object(servos, "_automatic_motion_allowed",
                                   return_value=True), \
                 mock.patch.object(servos, "SERVOS_ENABLED", True), \
                 mock.patch.object(servos, "set_servos") as writes, \
                 mock.patch.object(servos, "set_motion_profile"), \
                 mock.patch.object(servos, "set_breathing_emotion"):
                servos.end_speech_motion()
            writes.assert_called_once()
            visor_ch = servos._channel("visor")
            self.assertEqual(
                writes.call_args[0][0][visor_ch],
                servos.config.SERVO_CHANNELS["visor"]["neutral"],
                "awake, speech end still releases the visor to neutral")
        finally:
            servos._speech_active.clear()


class ActuallyIsNotACorrectionTest(unittest.TestCase):
    """11:30:42 — Rex asked "What's the most annoying part of calibrating those
    sensors — the math, or the fact that you have to trust your own hands?" and
    Bret answered it. "Actually, ..." matched the bare discourse-marker branch
    of _parse_memory_correct_fact, so Rex replied "Corrected. I now have Bret
    Benziger as the most annoying part: figuring out where to mount them on your
    body." and wrote `the_most_annoying_part` into his person_facts.

    The dialogue-act guard that names memory_correct_fact as un-promotable could
    not help: the reply was 15 words, over _looks_like_contextual_reply's
    12-word cap, so the turn came through as general_chat.
    """

    FIELD_UTTERANCE = (
        "Actually, the most annoying part is figuring out where to mount "
        "them on your body."
    )

    def test_answering_rex_is_not_a_memory_correction(self):
        from intelligence import command_parser
        self.assertIsNone(command_parser.parse(self.FIELD_UTTERANCE))

    def test_actually_needs_evidence_beyond_the_marker(self):
        from intelligence import command_parser
        for text in (
            "Actually, the deadline is Friday.",
            "Actually, the hardest part is the wiring.",
            "No, the meeting is at three.",
            "Actually, they changed the deadline on him.",
            "Nope, the whole thing is a mess.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(command_parser.parse(text))

    def test_real_corrections_still_route(self):
        from intelligence import command_parser
        for text in (
            "That's wrong, Daniel's last name is Smith.",   # lead-in IS evidence
            "Actually, call me Bret Michael.",              # explicit rename
            "Actually, Daniel's last name is Smith.",       # named-person fact
            "Nope, Daniel is a pilot.",
            "No, her name is Sarah.",
        ):
            with self.subTest(text=text):
                match = command_parser.parse(text)
                self.assertIsNotNone(match, text)
                self.assertEqual(match.command_key, "memory_correct_fact")

    def test_generic_write_rejects_a_topical_noun_phrase(self):
        """Second layer: even reached directly, "<clause> is <clause>" must not
        become a person attribute — it falls through to the elaboration reply."""
        from intelligence import interaction as I

        detail = ("the most annoying part is figuring out where to mount "
                  "them on your body")
        spoken = []
        with (
            mock.patch.object(I.facts_memory, "apply_fact_correction") as correct,
            mock.patch.object(I.llm, "get_response",
                              return_value="Mounting brackets. The eternal enemy."),
            mock.patch.object(I, "_speak_blocking",
                              side_effect=lambda t, *a, **k: spoken.append(t) or True),
            mock.patch.object(I, "_extract_memory_statement_target",
                              return_value=(1, "Bret Benziger", detail, False)),
        ):
            resp = I._execute_memory_correct_fact_command(
                {"correction": detail}, 1, "Bret Benziger")

        correct.assert_not_called()
        self.assertNotIn("Corrected", resp)
        self.assertIn("Mounting brackets", resp)

    def test_generic_write_still_stores_a_real_attribute(self):
        from intelligence import interaction as I

        with (
            mock.patch.object(I.facts_memory, "apply_fact_correction") as correct,
            mock.patch.object(I.repair_moves, "add_better_luck_line",
                              side_effect=lambda line: line),
            mock.patch.object(I, "_speak_blocking"),
            mock.patch.object(I, "_extract_memory_statement_target",
                              return_value=(7, "Daniel", "job is engineer", True)),
        ):
            resp = I._execute_memory_correct_fact_command(
                {"correction": "Daniel's job is engineer"}, 1, "Bret Benziger")

        correct.assert_called_once()
        self.assertEqual(correct.call_args.args[1], "job")
        self.assertEqual(correct.call_args.args[2], "engineer")
        self.assertIn("Corrected", resp)


if __name__ == "__main__":
    unittest.main()
