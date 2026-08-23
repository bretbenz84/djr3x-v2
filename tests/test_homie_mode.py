"""
Homie mode (intelligence/homie.py): greeting Rex with "what's up my homie" /
"what's up homeboy" / "wassup homie" arms a decaying delivery overlay that
code-switches his register into natural AAVE, seen by both voices. Owner
request 2026-08-23.

Coverage: trigger phrase matching (ASR spelling variants, and the
partial/topic phrases that must NOT flip it), TTL arming/expiry/refresh, the
enable kill switch, and the wiring into lean_brain._system_prompt and
llm.assemble_system_prompt. Modeled on tests/test_pride_mode.py minus the body
tests — homie mode is voice-only.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import homie, lean_brain


class HomieTestCase(unittest.TestCase):
    def setUp(self) -> None:
        homie.reset()
        self.addCleanup(homie.reset)


class TriggerMatchTest(HomieTestCase):
    def test_greetings_trigger(self) -> None:
        for phrase in (
            "What's up my homie?",
            "what's up homie",
            "whats up my homie",
            "What's up homeboy!",
            "what's up home boy",
            "wassup homie",
            "Wassup, homie!",
            "whassup my homie",
            "wazzup homie",
            "what up homie",
            "what is up my homie",
            "sup homie",
            "Hey Rex, what's up my homie?",
            "wassup homies",
            "what's up my homey",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(homie.is_homie_greeting(phrase))

    def test_partial_and_topic_talk_do_not_trigger(self) -> None:
        for phrase in (
            # The greeting alone or the address alone is not the phrase.
            "What's up?",
            "wassup",
            "hey homie",
            "my homie Marcus is coming over",
            "he's my homeboy",
            # Too far apart to be one greeting.
            "what's up with the music? play something for my homie",
            "what's for supper",
            "How are you?",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(homie.is_homie_greeting(phrase))


class ActivationTest(HomieTestCase):
    def test_trigger_arms_and_ttl_expires(self) -> None:
        with mock.patch.object(config, "HOMIE_MODE_TTL_SECS", 600.0):
            self.assertFalse(homie.is_active(now=1000.0))
            self.assertTrue(homie.maybe_trigger("wassup homie", now=1000.0))
            self.assertTrue(homie.is_active(now=1000.0))
            self.assertTrue(homie.is_active(now=1599.0))
            self.assertFalse(homie.is_active(now=1601.0))

    def test_regreet_refreshes_ttl(self) -> None:
        with mock.patch.object(config, "HOMIE_MODE_TTL_SECS", 600.0):
            homie.maybe_trigger("wassup homie", now=1000.0)
            homie.maybe_trigger("what's up homeboy", now=1500.0)
            self.assertTrue(homie.is_active(now=2050.0))

    def test_non_trigger_does_not_arm(self) -> None:
        self.assertFalse(homie.maybe_trigger("what's the weather?", now=1000.0))
        self.assertFalse(homie.is_active(now=1000.0))

    def test_kill_switch(self) -> None:
        with mock.patch.object(config, "HOMIE_MODE_ENABLED", False):
            self.assertFalse(homie.maybe_trigger("wassup homie", now=1000.0))
            self.assertFalse(homie.is_active(now=1000.0))
        # And an armed mode goes quiet if disabled after the fact.
        with mock.patch.object(config, "HOMIE_MODE_ENABLED", True):
            homie.maybe_trigger("wassup homie", now=1000.0)
        with mock.patch.object(config, "HOMIE_MODE_ENABLED", False):
            self.assertEqual(homie.prompt_lines(now=1000.0), [])


class PromptSurfaceTest(HomieTestCase):
    def test_prompt_lines_carry_the_register(self) -> None:
        homie.maybe_trigger("wassup homie")
        lines = homie.prompt_lines()
        self.assertEqual(len(lines), 1)
        # The register lives HERE, not in the baseline — full volume (owner
        # second pass 2026-08-23: every line in register, no averaging down),
        # with the one guard that survives: speak it, never mock it.
        for token in (
            "African American Vernacular English",
            "code-switch",
            "every single reply",
            "no neutral lines",
            "habitual \"be\"",
            "greet them back",
            "don't do an impression",
            "still Rex",
        ):
            self.assertIn(token, lines[0])

    def test_prompt_section_mirrors_lines(self) -> None:
        self.assertEqual(homie.prompt_section(), "")
        homie.maybe_trigger("wassup homie")
        section = homie.prompt_section()
        self.assertTrue(section.startswith("Rex's homie mode:"))
        self.assertIn("code-switch", section)

    def test_lean_system_prompt_includes_overlay_when_armed(self) -> None:
        base = lean_brain._system_prompt(None, None)
        self.assertNotIn("HOMIE MODE", base)
        homie.maybe_trigger("wassup homie")
        armed = lean_brain._system_prompt(None, None)
        self.assertIn("HOMIE MODE", armed)
        self.assertIn("African American Vernacular English", armed)

    def test_classic_prompt_includes_overlay_when_armed(self) -> None:
        from intelligence import llm
        homie.maybe_trigger("what's up my homie")
        prompt = llm.assemble_system_prompt(None)
        self.assertIn("HOMIE MODE", prompt)


if __name__ == "__main__":
    unittest.main()
