"""
Rex's self-state actually REACHES the model — both voices.

This is the load-bearing half of the 2026-08-05 "how are you?" fix. A day mood that
never lands in the prompt changes nothing, and there is a live precedent for exactly
that failure: intelligence/rex_pov.py injects its preoccupation ONLY into
llm.assemble_system_prompt, so under LEAN_BRAIN_ENABLED (the default) it never reaches
a direct reply at all.

So these tests assert the injection at both seams:
  * lean_brain._system_prompt — the LIVE path, covering replies AND directives
    (greetings, proactive lines) under ONE VOICE, and
  * llm.assemble_system_prompt — the classic fallback + the web-search prompt,
plus the REX_CORE_PROMPT rule that bans a status report as a wellbeing answer.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import greeting_cadence, lean_brain, rex_mood

_TEST_SEEDS = [
    {"id": "test-mood", "label": "cantankerous", "valence": -0.2, "energy": 0.6,
     "line": "Contrary, and enjoying it.", "fits": ["any"]},
]


class _InjectionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            mock.patch.object(config, "REX_MOOD_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_SEEDS", _TEST_SEEDS),
            mock.patch.object(rex_mood, "_SIGNALS", ()),
        ]
        for p in self._patches:
            p.start()
        rex_mood.clear()

    def tearDown(self) -> None:
        rex_mood.clear()
        for p in self._patches:
            p.stop()


class LeanBrainInjectionTests(_InjectionTestCase):
    """lean_brain is the PRIMARY voice (config.LEAN_BRAIN_ENABLED ships True) — if the
    mood is missing here, the feature does not exist in production."""

    def test_mood_reaches_the_lean_system_prompt(self):
        rex_mood.ensure_today()
        prompt = lean_brain._system_prompt(None, None)
        self.assertIn("YOUR OWN STATE TODAY", prompt)
        self.assertIn("cantankerous", prompt)

    def test_mood_lands_in_the_shared_builder_not_only_the_proactive_block(self):
        # _situation_block feeds ONLY consider_initiating. If the mood lived there, a
        # DIRECT "how are you?" would get no self-state — the exact bug being fixed.
        rex_mood.ensure_today()
        self.assertTrue(lean_brain._mood_lines())
        self.assertIn("YOUR OWN STATE TODAY", lean_brain._system_prompt(None, None))

    def test_directive_path_gets_the_mood_too(self):
        # stream_directive (greetings, proactive lines) routes through _messages ->
        # _system_prompt, so greeting text is colored by the same state.
        rex_mood.ensure_today()
        msgs = lean_brain._messages(
            "You see Bret — greet him.", None, [], None, label_current_speaker=False,
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("YOUR OWN STATE TODAY", msgs[0]["content"])

    def test_reply_path_gets_the_mood(self):
        rex_mood.ensure_today()
        msgs = lean_brain._messages("How are you?", None, [], None)
        self.assertIn("YOUR OWN STATE TODAY", msgs[0]["content"])

    def test_disabled_mood_adds_nothing(self):
        with mock.patch.object(config, "REX_MOOD_ENABLED", False):
            self.assertEqual(lean_brain._mood_lines(), [])
            self.assertNotIn("YOUR OWN STATE TODAY",
                             lean_brain._system_prompt(None, None))

    def test_a_broken_mood_module_cannot_break_a_reply(self):
        with mock.patch.object(rex_mood, "prompt_lines", side_effect=RuntimeError("boom")):
            self.assertEqual(lean_brain._mood_lines(), [])
            self.assertIsInstance(lean_brain._system_prompt(None, None), str)

    def test_wellbeing_suppression_reaches_the_lean_prompt(self):
        with mock.patch.object(greeting_cadence, "suppression_line",
                               return_value="You ALREADY asked them how they're doing."):
            prompt = lean_brain._system_prompt(7, None)
        self.assertIn("ALREADY asked", prompt)

    def test_a_broken_cadence_module_cannot_break_a_reply(self):
        with mock.patch.object(greeting_cadence, "suppression_line",
                               side_effect=RuntimeError("db gone")):
            self.assertEqual(lean_brain._cadence_lines(7), [])
            self.assertIsInstance(lean_brain._system_prompt(7, None), str)


class ClassicPromptInjectionTests(_InjectionTestCase):
    """The classic prompt is still the reply-path fallback on lean errors and the base
    for web-search replies — it must not answer "how are you" without the mood either."""

    def test_mood_section_reaches_assemble_system_prompt(self):
        from intelligence import llm
        rex_mood.ensure_today()
        with mock.patch.object(llm.conv_db, "get_session_transcript", return_value=[]):
            prompt = llm.assemble_system_prompt(person_id=None)
        self.assertIn("Rex's own state today:", prompt)
        self.assertIn("cantankerous", prompt)

    def test_mood_section_is_omitted_when_disabled(self):
        from intelligence import llm
        with (mock.patch.object(config, "REX_MOOD_ENABLED", False),
              mock.patch.object(llm.conv_db, "get_session_transcript", return_value=[])):
            prompt = llm.assemble_system_prompt(person_id=None)
        self.assertNotIn("Rex's own state today:", prompt)


class PersonaRuleTests(unittest.TestCase):
    """config.REX_CORE_PROMPT is shared by BOTH voices, which is where a persona rule
    belongs (see CONTEXT.md: "keep new persona/taste rules in REX_CORE_PROMPT")."""

    def test_persona_forbids_the_status_report_answer(self):
        prompt = config.REX_CORE_PROMPT
        self.assertIn("normal parameters", prompt)
        self.assertIn("NEVER answer a question about your wellbeing", prompt)

    def test_persona_names_the_reciprocal_forms(self):
        # The owner's actual sequence: Rex asks -> "I'm good, how about you?" -> Rex
        # reports normal parameters. The reciprocal has to be named or it reads as a
        # different question from "how are you?".
        prompt = config.REX_CORE_PROMPT
        for phrase in ("how about you?", "and yourself?", "how's your day?"):
            self.assertIn(phrase, prompt)

    def test_persona_still_allows_the_droid_tic_elsewhere(self):
        # "systems nominal" stays a legitimate verbal tic — it is banned only as a
        # WELLBEING answer, not deleted from his vocabulary.
        self.assertIn('"systems nominal"', config.REX_CORE_PROMPT)

    def test_persona_demands_a_fresh_answer_each_time(self):
        # Otherwise the fix just relocates the repetition from one canned line to another.
        prompt = config.REX_CORE_PROMPT
        self.assertIn("never recite it verbatim", prompt)
        self.assertIn("never give the same answer twice in one day", prompt)


if __name__ == "__main__":
    unittest.main()
