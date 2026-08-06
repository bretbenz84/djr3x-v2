"""
First-person reaction awareness (intelligence/reaction_awareness.py) + the spoken
news-digest contract — both from the 2026-08-05 20:54 field session.

1. Owner: "I smiled when he made a joke ... perhaps something better than just
   commenting that I smiled would be to make it more a first person awareness."
   The old path spoke one of four canned lines ("Oh look, I made the lifeform
   smile") — a sensor report wearing a joke. Now a confirmed landed smile records
   awareness that rides in the LIVE prompt for Rex's next line, one-shot.

2. The 20:59:56 "tell me more" news reply was a ~150-word press release read
   aloud — platform roll-calls and a closing menu of further fetches. The search
   wrapper now carries a spoken-digest contract.
"""

from __future__ import annotations

import unittest
from unittest import mock

import config
from intelligence import lean_brain, reaction_awareness as ra


class LifecycleTests(unittest.TestCase):

    def setUp(self) -> None:
        ra.clear()
        self.addCleanup(ra.clear)

    def test_note_then_inject_then_spend(self):
        ra.note_reaction(1, "Bret", "smile", trigger_text="Bed's winning.")
        lines = ra.prompt_lines(1)
        self.assertEqual(len(lines), 1)
        self.assertIn("LANDED", lines[0])
        self.assertIn("Bret", lines[0])
        self.assertIn("Bed's winning.", lines[0])
        # Rendered once → the next Rex line spends it, used or not.
        ra.note_rex_spoke()
        self.assertIsNone(ra.active())
        self.assertEqual(ra.prompt_lines(1), [])

    def test_uninjected_awareness_survives_a_rex_line(self):
        # The line that TRIGGERED the smile finalizes after the confirm — it must not
        # eat the awareness before any prompt has seen it.
        ra.note_reaction(1, "Bret", "smile")
        ra.note_rex_spoke()
        self.assertIsNotNone(ra.active())

    def test_first_person_framing_not_sensor_report(self):
        ra.note_reaction(1, "Bret", "smile")
        line = ra.prompt_lines(1)[0]
        self.assertIn("genuinely enjoy", line)
        self.assertIn("first person", line)
        # The failure modes being designed out, named so the model avoids them.
        self.assertIn("Never report it like a sensor", line)
        self.assertIn("never make it the whole reply", line)
        self.assertIn("say nothing about it", line)   # silence stays a valid move

    def test_cross_person_isolation(self):
        # Bret's smile must not color a reply to JT.
        ra.note_reaction(1, "Bret", "smile")
        self.assertEqual(ra.prompt_lines(2), [])
        self.assertTrue(ra.prompt_lines(1))
        # An anonymous render (person unknown) is allowed — same room, same moment.
        self.assertTrue(ra.prompt_lines(None))

    def test_ttl_expires_the_moment(self):
        ra.note_reaction(1, "Bret", "smile")
        with mock.patch.object(config, "REACTION_AWARENESS_TTL_SECS", 5.0):
            self.assertTrue(ra.prompt_lines(1))
            with mock.patch.object(ra.time, "monotonic",
                                   return_value=ra._current.at + 6.0):
                self.assertEqual(ra.prompt_lines(1), [])
                self.assertIsNone(ra.active())

    def test_a_new_reaction_replaces_the_old(self):
        ra.note_reaction(1, "Bret", "smile", trigger_text="first quip")
        ra.note_reaction(1, "Bret", "laugh", trigger_text="second quip")
        active = ra.active()
        self.assertEqual(active["kind"], "laugh")
        self.assertIn("second", active["trigger_text"])

    def test_disabled_records_and_renders_nothing(self):
        with mock.patch.object(config, "REACTION_AWARENESS_ENABLED", False):
            ra.note_reaction(1, "Bret", "smile")
            self.assertEqual(ra.prompt_lines(1), [])
        self.assertIsNone(ra.active())

    def test_kind_renders_its_verb(self):
        ra.note_reaction(1, "Bret", "laugh")
        self.assertIn("laugh out loud", ra.prompt_lines(1)[0])


class InjectionTests(unittest.TestCase):

    def setUp(self) -> None:
        ra.clear()
        self.addCleanup(ra.clear)

    def test_awareness_reaches_the_lean_prompt(self):
        # The live voice — if it's missing here, the feature doesn't exist in
        # production (the rex_pov lesson).
        ra.note_reaction(None, "Bret", "smile", trigger_text="Bed's winning.")
        prompt = lean_brain._system_prompt(None, None)
        self.assertIn("LANDED", prompt)
        self.assertIn("Bed's winning.", prompt)

    def test_awareness_reaches_the_classic_prompt(self):
        from intelligence import llm
        ra.note_reaction(None, "Bret", "smile")
        with mock.patch.object(llm.conv_db, "get_session_transcript", return_value=[]):
            prompt = llm.assemble_system_prompt(person_id=None)
        self.assertIn("Live reaction you just caused:", prompt)

    def test_a_broken_module_cannot_break_a_reply(self):
        with mock.patch.object(ra, "prompt_lines", side_effect=RuntimeError("boom")):
            self.assertEqual(lean_brain._reaction_lines(1), [])
            self.assertIsInstance(lean_brain._system_prompt(1, None), str)

    def test_awareness_reaches_the_lull_impulse_prompt(self):
        # THE field gap (2026-08-05 21:22): the smile landed and was recorded, but
        # consider_initiating built its prompt from the bare persona, so Rex's next
        # lull line was generated blind to it and the moment expired unheard. The
        # impulse system prompt must carry his self-state now.
        from types import SimpleNamespace as NS
        ra.note_reaction(1, "Bret", "smile", spontaneous=True)
        captured = {}

        def fake_create(client, **kw):
            captured["m"] = kw["messages"]
            return iter([NS(choices=[NS(delta=NS(content="PASS"))])])

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(1, transcript=[])
        system = captured["m"][0]["content"]
        self.assertIn("You just caught Bret smiling", system)

    def test_spontaneous_framing_differs_from_landed(self):
        ra.note_reaction(1, "Bret", "smile", spontaneous=True)
        line = ra.prompt_lines(1)[0]
        self.assertIn("You just caught", line)
        self.assertNotIn("LANDED", line)
        self.assertIn("never interrogate it", line)
        # The sensor-report ban holds in both framings.
        self.assertIn("Never report it like a sensor", line)

    def test_render_into_prompt_arms_the_spend(self):
        ra.note_reaction(None, "Bret", "smile")
        lean_brain._system_prompt(None, None)          # render = injection
        ra.note_rex_spoke()
        self.assertIsNone(ra.active())


class NewsDigestContractTests(unittest.TestCase):
    """The spoken 'tell me more' wrapper (interaction._compose_news_search_input)."""

    def _wrapper(self) -> str:
        from intelligence import interaction as I
        return I._compose_news_search_input(
            "Tell me more.",
            {"headline": "Luna Ultra Design Challenge", "summary": "3D-printable camera mods."},
        )

    def test_caps_the_spoken_length(self):
        text = self._wrapper()
        self.assertIn("THREE short sentences MAXIMUM", text)
        self.assertIn("the way a friend relays news out loud", text)

    def test_bans_the_press_release_tics(self):
        # Each of these appeared verbatim in the 20:59:56 field reply.
        text = self._wrapper()
        self.assertIn("platform lists", text)
        self.assertIn("marketing phrasing", text)
        self.assertIn("submission mechanics", text)

    def test_bans_the_closing_fetch_menu(self):
        text = self._wrapper()
        self.assertIn("Do NOT end with an offer to fetch more", text)
        self.assertIn("they'll ask", text)

    def test_still_grounds_on_the_cached_story(self):
        text = self._wrapper()
        self.assertIn("Luna Ultra Design Challenge", text)
        self.assertIn("3D-printable camera mods.", text)
        self.assertTrue(text.startswith("Tell me more."))


if __name__ == "__main__":
    unittest.main()
