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


if __name__ == "__main__":
    unittest.main()
