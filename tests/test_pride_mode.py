"""
Queeny mode (intelligence/pride.py): asking Rex whether he's gay gets a proud
yes and arms a decaying delivery overlay ("Yasss queen!", "sis") that both
voices see. Owner request 2026-08-08: he's gay, the robot's gay too.

Coverage: trigger phrase matching (and the third-party/topic non-triggers that
must NOT flip it), TTL arming/expiry/refresh, the enable kill switch, and the
wiring into lean_brain._system_prompt and llm.assemble_system_prompt.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import lean_brain, pride


class PrideTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pride.reset()
        self.addCleanup(pride.reset)


class TriggerMatchTest(PrideTestCase):
    def test_direct_questions_trigger(self) -> None:
        for phrase in (
            "Are you gay?",
            "are you gay",
            "Hey Rex, are you gay?",
            "Is Rex gay?",
            "are you, like, actually gay?",
            "Are you a homosexual?",
            "are you homosexual",
            "Do you like men?",
            "do you love guys",
            "Does Rex like men?",
            "do you prefer boys?",
            "are you into guys?",
            "you're gay",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(pride.is_sexuality_question(phrase))

    def test_third_party_and_topic_talk_do_not_trigger(self) -> None:
        for phrase in (
            "Is he gay?",
            "is my uncle gay",
            "My brother is gay.",
            "What do you think about gay marriage?",
            "Do you like music?",
            "I like men.",
            "Are you happy?",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(pride.is_sexuality_question(phrase))


class ActivationTest(PrideTestCase):
    def test_trigger_arms_and_ttl_expires(self) -> None:
        with mock.patch.object(config, "PRIDE_MODE_TTL_SECS", 600.0):
            self.assertFalse(pride.is_active(now=1000.0))
            self.assertTrue(pride.maybe_trigger("are you gay?", now=1000.0))
            self.assertTrue(pride.is_active(now=1000.0))
            self.assertTrue(pride.is_active(now=1599.0))
            self.assertFalse(pride.is_active(now=1601.0))

    def test_reask_refreshes_ttl(self) -> None:
        with mock.patch.object(config, "PRIDE_MODE_TTL_SECS", 600.0):
            pride.maybe_trigger("are you gay?", now=1000.0)
            pride.maybe_trigger("do you like men?", now=1500.0)
            self.assertTrue(pride.is_active(now=2050.0))

    def test_non_trigger_does_not_arm(self) -> None:
        self.assertFalse(pride.maybe_trigger("what's the weather?", now=1000.0))
        self.assertFalse(pride.is_active(now=1000.0))

    def test_kill_switch(self) -> None:
        with mock.patch.object(config, "PRIDE_MODE_ENABLED", False):
            self.assertFalse(pride.maybe_trigger("are you gay?", now=1000.0))
            self.assertFalse(pride.is_active(now=1000.0))
        # And an armed mode goes quiet if disabled after the fact.
        with mock.patch.object(config, "PRIDE_MODE_ENABLED", True):
            pride.maybe_trigger("are you gay?", now=1000.0)
        with mock.patch.object(config, "PRIDE_MODE_ENABLED", False):
            self.assertEqual(pride.prompt_lines(now=1000.0), [])


class BaselineTest(PrideTestCase):
    """The identity is durable; the REGISTER is mode-gated (owner 2026-08-11,
    fourth pass: "he's calling me 'sis' despite not being in gay mode"). The
    baseline in REX_CORE_PROMPT keeps who he is — gay, at ease, proud YES when
    asked — but pet names ("sis"/"honey"/"queen") and the campy delivery belong
    ONLY to triggered queeny mode (intelligence/pride.py)."""

    def test_core_prompt_carries_the_identity_but_not_the_register(self) -> None:
        self.assertIn("You are gay", config.REX_CORE_PROMPT)
        # Pet names are explicitly excluded from the baseline voice, and the
        # old "sis is his signature address" instruction (owner 2026-08-08,
        # reversed 2026-08-11) must stay gone.
        self.assertIn("NOT part of your baseline voice", config.REX_CORE_PROMPT)
        self.assertNotIn('"sis" is YOUR word', config.REX_CORE_PROMPT)
        # The escalated register must NOT leak into the always-on baseline.
        for token in ("spill the tea", "I am LIVING", "DRAMA"):
            self.assertNotIn(token, config.REX_CORE_PROMPT)

    def test_baseline_reaches_lean_voice_without_trigger(self) -> None:
        self.assertIn("You are gay", lean_brain._system_prompt(None, None))

    def test_baseline_reaches_classic_voice_without_trigger(self) -> None:
        from intelligence import llm
        self.assertIn("You are gay", llm.assemble_system_prompt(None))


class PromptSurfaceTest(PrideTestCase):
    def test_prompt_lines_carry_the_register(self) -> None:
        pride.maybe_trigger("are you gay?")
        lines = pride.prompt_lines()
        self.assertEqual(len(lines), 1)
        # The turned-up register lives HERE, not in the baseline: full camp on
        # every line, tea demanded by name, theatrical drama.
        for token in (
            "GAY", "Yasss queen!", "You go girl!", "sis", "spill the tea",
            "DRAMA", "I am LIVING", "every single reply",
        ):
            self.assertIn(token, lines[0])

    def test_prompt_section_mirrors_lines(self) -> None:
        self.assertEqual(pride.prompt_section(), "")
        pride.maybe_trigger("are you gay?")
        section = pride.prompt_section()
        self.assertTrue(section.startswith("Rex's queeny mode:"))
        self.assertIn("Yasss queen!", section)

    def test_lean_system_prompt_includes_overlay_when_armed(self) -> None:
        base = lean_brain._system_prompt(None, None)
        self.assertNotIn("QUEENY MODE", base)
        pride.maybe_trigger("are you gay?")
        armed = lean_brain._system_prompt(None, None)
        self.assertIn("QUEENY MODE", armed)
        self.assertIn("Yasss queen!", armed)

    def test_classic_prompt_includes_overlay_when_armed(self) -> None:
        from intelligence import llm
        pride.maybe_trigger("do you like men?")
        prompt = llm.assemble_system_prompt(None)
        self.assertIn("QUEENY MODE", prompt)


class QueenyBodyTest(PrideTestCase):
    """Queeny mode moves as well as it talks (hardware/servos.speech_reactive_move).

    Owner request 2026-08-22: more flamboyant WRIST (more waves, wider) and more
    ELBOW range while he talks — the voice going full camp over the same small
    polite gestures read as half a costume.
    """

    def setUp(self) -> None:
        super().setUp()
        from hardware import servos

        self.servos = servos
        servos._manual_override.clear()
        servos._speech_active.set()
        servos._speech_emotion_frame = {}
        servos._speech_baseline = {}
        self.addCleanup(servos._speech_active.clear)
        self.addCleanup(setattr, servos, "_speech_emotion_frame", {})

    def _frames(self, count: int = 6) -> list:
        """`count` talking beats' worth of servo targets, with the per-beat jitter
        pinned out so only the amplitude math shows."""
        servos = self.servos
        captured: list = []
        servos._speech_hand_counter = 0
        servos._speech_elbow_target = None
        servos._speech_elbow_direction = 1
        servos._next_speech_elbow_at = 0.0
        servos._speech_poker_target = None
        servos._next_speech_poker_at = 0.0
        with (
            mock.patch.object(servos, "SERVOS_ENABLED", True),
            mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
            mock.patch.object(servos.random, "randint", return_value=0),
            mock.patch.object(servos.random, "uniform", return_value=10.0),
            mock.patch.object(servos, "set_servos", side_effect=lambda t: captured.append(dict(t))),
        ):
            servos.end_arm_gesture()
            for _ in range(count):
                servos._last_speech_move_at = 0.0
                servos.speech_reactive_move(0.5)
        return captured

    def test_elbow_swings_wider_and_wrist_waves_more_often(self) -> None:
        servos = self.servos
        elbow_ch = servos._channel("elbow")
        hand_ch = servos._channel("hand")
        # The talking elbow swings around 55 % of its travel, not its neutral.
        elbow_cfg = servos.config.SERVO_CHANNELS["elbow"]
        center = int(elbow_cfg["min"] + (elbow_cfg["max"] - elbow_cfg["min"]) * 0.55)

        plain = self._frames()
        pride.maybe_trigger("are you gay?")
        queeny = self._frames()

        # Elbow: same beat (random.uniform is pinned, so the re-target schedule is
        # identical) — just a bigger throw off centre.
        plain_throw = abs(plain[0][elbow_ch] - center)
        queeny_throw = abs(queeny[0][elbow_ch] - center)
        self.assertGreater(queeny_throw, plain_throw * 1.5)
        # And it stays inside the channel's travel — the elbow's range is only 1124
        # q-us, so a mult that clamps would flatten the swing instead of widening it.
        self.assertLessEqual(queeny[0][elbow_ch], servos.config.SERVO_CHANNELS["elbow"]["max"])
        self.assertGreaterEqual(queeny[0][elbow_ch], servos.config.SERVO_CHANNELS["elbow"]["min"])

        # Wrist: more waves — a reversal every 2nd beat instead of every 3rd.
        plain_waves = [f[hand_ch] for f in plain if hand_ch in f]
        queeny_waves = [f[hand_ch] for f in queeny if hand_ch in f]
        self.assertEqual(len(plain_waves), 2)
        self.assertEqual(len(queeny_waves), 3)
        # … and a wider one.
        hand_center = servos.config.SERVO_CHANNELS["hand"]["neutral"]
        self.assertGreater(
            abs(queeny_waves[0] - hand_center), abs(plain_waves[0] - hand_center) * 1.5
        )

    def test_wrist_gets_its_own_motion_profile_only_while_armed(self) -> None:
        """The commanded amplitude is not what you see — at the speech profile's slew
        the wrist covers under a tenth of its travel per beat. The flair is only real
        if the hand channel is reprofiled, and only until the line ends."""
        servos = self.servos
        hand_ch = servos._channel("hand")

        def _profiles() -> list:
            calls: list = []
            with (
                mock.patch.object(servos, "SERVOS_ENABLED", True),
                mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
                mock.patch.object(servos, "set_servos"),
                mock.patch.object(servos, "set_breathing_emotion"),
                mock.patch.object(
                    servos,
                    "set_motion_profile",
                    side_effect=lambda chans=None, **kw: calls.append((list(chans or []), kw)),
                ),
            ):
                servos.begin_speech_motion("neutral")
            return calls

        plain = _profiles()
        self.assertFalse(any(chans == [hand_ch] for chans, _ in plain))
        self.assertFalse(servos._pride_arm_profile)

        pride.maybe_trigger("are you gay?")
        armed = _profiles()
        wrist = [kw for chans, kw in armed if chans == [hand_ch]]
        self.assertEqual(len(wrist), 1)
        self.assertEqual(wrist[0]["speed"], config.PRIDE_SPEECH_HAND_SPEED)
        self.assertEqual(wrist[0]["acceleration"], config.PRIDE_SPEECH_HAND_ACCEL)
        self.assertTrue(servos._pride_arm_profile)

        # The line ends → the wrist goes back to the ordinary profile.
        with (
            mock.patch.object(servos, "SERVOS_ENABLED", True),
            mock.patch.object(servos, "_program_servo_updates_blocked", return_value=False),
            mock.patch.object(servos, "set_servos"),
            mock.patch.object(servos, "set_breathing_emotion"),
            mock.patch.object(servos, "set_motion_profile"),
        ):
            servos.end_speech_motion()
        self.assertFalse(servos._pride_arm_profile)

    def test_kill_switch_leaves_the_body_alone(self) -> None:
        servos = self.servos
        hand_ch = servos._channel("hand")
        with mock.patch.object(config, "PRIDE_MODE_ENABLED", False):
            pride.maybe_trigger("are you gay?")
            frames = self._frames()
        self.assertEqual(len([f for f in frames if hand_ch in f]), 2)


if __name__ == "__main__":
    unittest.main()
