"""
Rex volunteering his own day, unprompted (owner 2026-08-05: "real people do that.
It doesn't have to happen all the time, but perhaps on a random roll?").

The first pass gave him a mood but only ever revealed it under interrogation, which
is still a lookup table — a person mentions their day without being asked. This is
the volunteering half: a lull cue that offers the mood as ONE dry aside.

Everything here is about keeping that LIFELIKE rather than turning it into a new
daily ritual — which would just be the original "he always says the same thing"
complaint wearing a different hat. Hence: notable days only, once per day (persisted,
so a reboot doesn't re-arm it), friends only, and a random roll.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import interaction as I
from intelligence import lean_brain, rex_mood
from tests._lean_impulse_state import reset_impulse_state

_NOTABLE = {"id": "worn-out", "label": "worn", "valence": -0.6, "energy": 0.2,
            "line": "Worn down, and awake for all of it.", "fits": ["any"]}
_BLAND = {"id": "even", "label": "even", "valence": 0.05, "energy": 0.5,
          "line": "Perfectly ordinary.", "fits": ["any"]}


class _CueTestCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_impulse_state(self)
        self._patches = [
            mock.patch.object(config, "REX_MOOD_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_SHARE_ENABLED", True),
            mock.patch.object(config, "REX_MOOD_SHARE_PROBABILITY", 1.0),   # always roll in
            mock.patch.object(config, "REX_MOOD_SEEDS", [_NOTABLE]),
            mock.patch.object(config, "REX_MOOD_DRIFT_MIN_INTERVAL_SECS", 0.0),
            mock.patch.object(rex_mood, "_SIGNALS", ()),
        ]
        for p in self._patches:
            p.start()
        rex_mood.clear()

    def tearDown(self) -> None:
        rex_mood.clear()
        for p in self._patches:
            p.stop()

    def _person(self, tier="close_friend", name="Bret Benziger"):
        return mock.patch.object(
            I.people_memory, "get_person",
            return_value={"id": 1, "name": name, "friendship_tier": tier},
        )


class GateTests(_CueTestCase):

    def test_a_friend_on_a_notable_day_gets_the_share(self):
        with self._person():
            cue = I._lean_mood_share_cue(1)
        self.assertIsNotNone(cue)
        self.assertEqual(cue["label"], "worn")

    def test_the_roll_can_decline(self):
        # "It doesn't have to happen all the time" — probability 0 must never fire.
        with self._person(), mock.patch.object(config, "REX_MOOD_SHARE_PROBABILITY", 0.0):
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_a_bland_day_is_never_volunteered(self):
        with self._person(), mock.patch.object(config, "REX_MOOD_SEEDS", [_BLAND]):
            rex_mood.clear()
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_an_acquaintance_does_not_get_your_day(self):
        for tier in ("acquaintance", "stranger", ""):
            with self.subTest(tier=tier):
                rex_mood.clear()
                with self._person(tier=tier, name="Someone Else"):
                    self.assertIsNone(I._lean_mood_share_cue(1))

    def test_the_creator_always_qualifies_regardless_of_computed_tier(self):
        with self._person(tier="acquaintance"), \
             mock.patch.object(I.person_specials, "is_rex_creator", return_value=True):
            self.assertIsNotNone(I._lean_mood_share_cue(1))

    def test_once_per_session(self):
        with self._person():
            self.assertIsNotNone(I._lean_mood_share_cue(1))
            I._lean_mood_shared_this_session = True
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_already_answered_how_are_you_spends_the_unprompted_share(self):
        # The redundancy that started all of this: he told you when you asked, so he
        # must not then announce it. This gate is per-DAY and persisted.
        with self._person():
            self.assertIsNotNone(I._lean_mood_share_cue(1))
            rex_mood.note_spoken()
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_disabled_flag_stops_it(self):
        with self._person(), mock.patch.object(config, "REX_MOOD_SHARE_ENABLED", False):
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_no_person_no_share(self):
        self.assertIsNone(I._lean_mood_share_cue(None))

    def test_a_broken_lookup_never_breaks_the_impulse(self):
        with mock.patch.object(I.people_memory, "get_person",
                               side_effect=RuntimeError("db gone")):
            self.assertIsNone(I._lean_mood_share_cue(1))
        with self._person(), mock.patch.object(rex_mood, "share_cue",
                                               side_effect=RuntimeError("boom")):
            self.assertIsNone(I._lean_mood_share_cue(1))


class InstructionTests(_CueTestCase):

    def _instruction(self, cue) -> str:
        captured = {}

        def fake_create(client, **kw):
            captured["m"] = kw["messages"]
            return iter([NS(choices=[NS(delta=NS(content="PASS"))])])

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(None, transcript=[], mood_share=cue)
        return captured["m"][-1]["content"]

    def test_instruction_carries_the_mood_and_forbids_a_recital(self):
        rex_mood.ensure_today()
        text = self._instruction(rex_mood.share_cue())
        self.assertIn("worn", text)
        self.assertIn("Worn down, and awake for all of it.", text)
        self.assertIn("NOT a script", text)
        self.assertIn("say it fresh", text)

    def test_instruction_bans_systems_language(self):
        # "I am currently operating in a degraded mood state" is the failure mode.
        rex_mood.ensure_today()
        text = self._instruction(rex_mood.share_cue())
        for banned in ("'mood'", "'state'", "'status'", "'parameters'", "'diagnostic'"):
            self.assertIn(banned, text)

    def test_instruction_is_a_statement_not_a_question(self):
        rex_mood.ensure_today()
        text = self._instruction(rex_mood.share_cue())
        self.assertIn("Do NOT ask them anything", text)
        self.assertIn("offhand aside", text)
        self.assertIn("not a bid for sympathy", text)
        self.assertIn("ONE short line", text)

    def test_instruction_includes_the_reason_when_the_day_gave_one(self):
        def newsy(now=None, allow_blocking=False):
            return (("chewing",), "you've had a thing rattling around all day")

        with mock.patch.object(rex_mood, "_SIGNALS", (("news", newsy),)):
            rex_mood.clear()
            rex_mood.ensure_today()
            text = self._instruction(rex_mood.share_cue())
        self.assertIn("rattling around all day", text)

    def test_a_cue_with_no_reason_still_renders_cleanly(self):
        rex_mood.ensure_today()
        cue = dict(rex_mood.share_cue(), because="", shade="")
        text = self._instruction(cue)
        self.assertIn("How you've actually been today: worn.", text)
        self.assertNotIn("—  ", text)
        self.assertNotIn(", .", text)


class LadderPriorityTests(_CueTestCase):
    """Placement: below everything about THEM (asking after someone's weekend beats
    talking about yourself), above generic news."""

    def _dispatch(self, **cues) -> str:
        captured = {}

        def fake_create(client, **kw):
            captured["m"] = kw["messages"]
            return iter([NS(choices=[NS(delta=NS(content="PASS"))])])

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(None, transcript=[], **cues)
        return captured["m"][-1]["content"]

    def test_mood_share_beats_generic_news(self):
        rex_mood.ensure_today()
        text = self._dispatch(
            mood_share=rex_mood.share_cue(),
            news_story={"headline": "Some headline", "summary": "..."},
        )
        self.assertIn("How you've actually been today", text)
        self.assertNotIn("Some headline", text)

    def test_anything_about_them_beats_the_mood_share(self):
        rex_mood.ensure_today()
        for cue_name, cue in (
            ("weekend_plans", {"when": "this weekend"}),
            ("interest_discovery", {"known": "3D printing"}),
            ("workday_checkin", {"kind": "day"}),
            ("event_followup", {"event_name": "the interview"}),
            ("celebration", {"description": "they got the job"}),
        ):
            with self.subTest(cue=cue_name):
                text = self._dispatch(mood_share=rex_mood.share_cue(), **{cue_name: cue})
                self.assertNotIn("How you've actually been today", text)

    def test_the_ladder_marks_mood_share_as_the_winning_cue(self):
        # _winning_kind drives the drop-cooldown bookkeeping; a cue missing from that
        # tuple silently can't be benched when its line gets dropped.
        import inspect
        src = inspect.getsource(I._maybe_lean_impulse)
        self.assertIn('("mood_share", mood_share)', src)


class ShippedDefaultsTests(unittest.TestCase):

    def test_probability_leaves_room_for_it_not_to_happen(self):
        p = float(config.REX_MOOD_SHARE_PROBABILITY)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0, "a certainty would be a scheduled broadcast, not a mood")

    def test_notability_bar_excludes_the_blandest_moods(self):
        seeds = list(config.REX_MOOD_SEEDS)
        mi = float(config.REX_MOOD_SHARE_MIN_INTENSITY)
        lo = float(config.REX_MOOD_SHARE_LOW_ENERGY)
        hi = float(config.REX_MOOD_SHARE_HIGH_ENERGY)
        notable = [s for s in seeds
                   if abs(s["valence"]) >= mi or s["energy"] <= lo or s["energy"] >= hi]
        # Some days he mentions it, some days he doesn't — both must be reachable.
        self.assertGreater(len(notable), 0)
        self.assertLess(len(notable), len(seeds))

    def test_tiers_exclude_acquaintances(self):
        tiers = {str(t).lower() for t in config.REX_MOOD_SHARE_MIN_TIERS}
        self.assertNotIn("acquaintance", tiers)
        self.assertNotIn("stranger", tiers)
        self.assertIn("friend", tiers)


if __name__ == "__main__":
    unittest.main()
