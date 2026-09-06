"""Keep layered conversation prompts consistent with the approved Rex character."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import config
from intelligence import lean_brain as lean, llm, social_frame as sf
from tests.test_sharp_roast_tier import _frame


class ConversationalPersonaTests(unittest.TestCase):
    def assert_no_roast_pressure(self, prompt):
        for phrase in ('FUNNY WITH TEETH', 'Roasting is a DEFAULT lens', 'Kid gloves are WRONG',
                       'zero mercy', 'no holding back', 'ROAST-LEAN', 'surgical',
                       'more cutting rib', 'uses real knowledge against them'):
            self.assertNotIn(phrase.lower(), prompt.lower())

    def test_lean_assembled_creator_prompt_preserves_attention_and_boundaries(self):
        from memory import people, boundaries
        person = {'id': 1, 'name': 'Bret Benziger', 'friendship_tier': 'best_friend'}
        with ExitStack() as stack:
            stack.enter_context(patch.object(people, 'get_person', return_value=person))
            stack.enter_context(patch.object(boundaries, 'summarize_for_prompt',
                                             return_value='Do not joke about the website.'))
            for helper in ('_scene_lines', '_room_belief_lines', '_mood_lines', '_pride_lines',
                           '_homie_lines', '_taste_lines', '_reaction_lines', '_cadence_lines', '_context_lines'):
                stack.enter_context(patch.object(lean, helper, return_value=[]))
            prompt = lean._system_prompt(1, {}, user_text="I'm into GPT six Astra.")
        self.assertTrue(prompt.startswith(config.REX_CORE_PROMPT))
        self.assertIn('not a standing request for a roast', prompt)
        self.assertIn('Do not joke about the website.', prompt)
        self.assertIn('An ordinary answer can be a complete and successful turn', prompt)
        self.assert_no_roast_pressure(prompt)

    def test_known_people_and_classic_relationships_do_not_escalate_to_roasting(self):
        from memory import people
        for tier in ('acquaintance', 'friend', 'close_friend', 'best_friend'):
            with self.subTest(tier=tier):
                person = {'id': 2, 'name': 'Jeremy Thomas', 'friendship_tier': tier,
                          'warmth_score': .95, 'antagonism_score': 0., 'trust_score': .8}
                with patch.object(people, 'get_person', return_value=person):
                    lines = '\n'.join(lean._person_lines(2, 'I like volleyball.'))
                self.assertIn('does not mean treating them more harshly', lines)
                self.assert_no_roast_pressure(lines + llm._TIER_ROAST_STYLE[tier]
                                              + llm._relationship_tone_rule(person, 'JT'))

    def test_both_contract_formats_preserve_optional_teasing(self):
        for tier in ('normal', 'sharp'):
            frame = _frame(tier)
            for prompt in (sf.build_directive(frame), sf.render_slim_contract(frame)):
                self.assert_no_roast_pressure(prompt)
                self.assertIn('ordinary answer needs no punchline' if tier == 'normal'
                              else 'never requires a harsher joke', prompt)

    def test_comedy_overlays_cannot_require_a_bit_on_every_turn(self):
        from intelligence import comedy_modes as comedy
        for key, mode in comedy._MODES.items():
            if key == 'straight':
                continue
            with self.subTest(mode=key):
                self.assertIn('This style is optional', comedy.build_directive(mode))
                self.assertIn('skip the bit', comedy.build_slim_directive(mode))

    def test_classic_assembled_prompt_uses_approved_character(self):
        prompt = llm.assemble_system_prompt(None)
        self.assertTrue(prompt.startswith(config.REX_CORE_PROMPT))
        self.assert_no_roast_pressure(prompt)


if __name__ == '__main__':
    unittest.main()
