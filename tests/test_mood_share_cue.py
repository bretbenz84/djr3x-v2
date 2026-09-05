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


class AddendumConflictTests(_CueTestCase):
    """Review find 2026-08-05: the mood-share instruction ends "You MUST give the one
    line; do not reply PASS", and the generic low-energy/no-questions addenda append
    "just reply PASS — PASS is genuinely good here" straight after. A contradictory
    instruction pair resolves unpredictably; the share now gets tailored addenda."""

    def _instruction(self, **kw) -> str:
        captured = {}

        def fake_create(client, **k):
            captured["m"] = k["messages"]
            return iter([NS(choices=[NS(delta=NS(content="PASS"))])])

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            lean_brain.consider_initiating(None, transcript=[], **kw)
        return captured["m"][-1]["content"]

    def test_low_energy_share_never_contradicts_itself(self):
        rex_mood.ensure_today()
        text = self._instruction(mood_share=rex_mood.share_cue(), low_energy=True)
        self.assertIn("do not reply PASS", text)
        self.assertNotIn("PASS is a genuinely", text)
        self.assertNotIn("just reply PASS", text)
        # It still adapts to the tired room instead of ignoring it.
        self.assertIn("low-key", text)
        self.assertIn("needs no response", text)

    def test_no_questions_share_never_contradicts_itself(self):
        rex_mood.ensure_today()
        text = self._instruction(mood_share=rex_mood.share_cue(), no_questions=True)
        self.assertIn("do not reply PASS", text)
        self.assertNotIn("observation, or PASS", text)

    def test_other_cues_keep_the_generic_addenda(self):
        text = self._instruction(low_energy=True)
        self.assertIn("PASS is a genuinely", text)


class DropBenchTests(_CueTestCase):

    def test_a_benched_mood_share_is_not_reoffered(self):
        # _strike_lean_cue records the bench when a generated share gets dropped
        # (near-duplicate, banned topic); the builder must actually consult it or the
        # bench does nothing and the ladder regenerates the same doomed line.
        import time as _time
        with self._person():
            self.assertIsNotNone(I._lean_mood_share_cue(1))
            I._strike_lean_cue("mood_share")
            self.assertIsNone(I._lean_mood_share_cue(1))
            I._lean_cue_cooldowns["mood_share"] = _time.monotonic() - 1.0
            self.assertIsNotNone(I._lean_mood_share_cue(1))


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
        # Phase 3: the ladder became _collect_lean_cue_candidates (spec table) and
        # the spend site binds the chosen kind by string in _maybe_lean_impulse.
        self.assertIn('("mood_share",', inspect.getsource(I._collect_lean_cue_candidates))
        self.assertIn('chosen == "mood_share"', inspect.getsource(I._maybe_lean_impulse))


class GreetingAsideTests(_CueTestCase):
    """Owner 2026-08-05: "Can his opening line to a recognized person ever offer up
    his current emotional state? Or does it only fire during lulls?" It was lull-only
    — the greeting received the mood but the day-mood bullet ends with "don't
    announce your mood unprompted", so it only tinted his tone. The hello now carries
    an optional aside on the same gates, sharing the one-per-day spend."""

    def setUp(self) -> None:
        super().setUp()
        from intelligence import consciousness
        self.c = consciousness
        p = mock.patch.object(config, "REX_MOOD_GREETING_ASIDE_PROBABILITY", 1.0)
        p.start()
        self.addCleanup(p.stop)

    def _person(self, tier="close_friend", name="Bret Benziger"):
        from memory import people as pm
        return mock.patch.object(
            pm, "get_person",
            return_value={"id": 1, "name": name, "friendship_tier": tier},
        )

    def test_plain_hello_branches_carry_the_aside(self):
        for label in (
            "first-sight warm greeting for Bret",
            "startup same-day return (#2) for Bret",
            "startup quick-return (recent) for Bret",
            "startup cadence (streak) for Bret",
            "startup long-absence for Bret",
            "startup recent-return for Bret",
        ):
            with self.subTest(label=label):
                self.assertTrue(self.c._greeting_allows_mood_aside(label))

    def test_branches_about_THEM_never_carry_it(self):
        # Turning a birthday or a grief check-in toward his own day is exactly the
        # self-absorption the mood gates exist to prevent.
        for label in (
            "startup birthday (T-0) for Bret",
            "startup celebration check-in for Bret",
            "startup emotional check-in for Bret",
            "startup milestone for Bret",
            "startup anticipation for Bret",
            "startup followup for Bret",
            "startup continuity (thing) for Bret",
        ):
            with self.subTest(label=label):
                self.assertFalse(self.c._greeting_allows_mood_aside(label))

    def test_the_snap_quick_return_never_carries_it(self):
        # Its whole contract is under eight words and no additions.
        self.assertFalse(
            self.c._greeting_allows_mood_aside("startup quick-return (snap) for Bret"))
        self.assertTrue(
            self.c._greeting_allows_mood_aside("startup quick-return (recent) for Bret"))

    def test_the_clause_overrides_the_standing_no_announce_rule(self):
        # The system prompt's day-mood bullet says "don't announce your mood
        # unprompted" on EVERY call; without an explicit override the greeting
        # aside and that rule contradict each other.
        rex_mood.ensure_today()
        with self._person():
            clause = self.c._greeting_mood_aside(1)
        self.assertIn("OVERRIDES", clause)
        self.assertIn("don't announce your mood", clause)

    def test_the_clause_keeps_the_hello_primary(self):
        rex_mood.ensure_today()
        with self._person():
            clause = self.c._greeting_mood_aside(1)
        self.assertIn("hello FIRST, aside second", clause)
        self.assertIn("do not let it replace the greeting", clause)
        self.assertIn("NOT a script", clause)
        # Same systems-language ban as the lull share.
        for banned in ('"mood"', '"state"', '"status"'):
            self.assertIn(banned, clause)

    def test_gates_match_the_lull_share(self):
        rex_mood.ensure_today()
        # Roll can decline.
        with self._person(), mock.patch.object(
            config, "REX_MOOD_GREETING_ASIDE_PROBABILITY", 0.0
        ):
            self.assertEqual(self.c._greeting_mood_aside(1), "")
        # Acquaintances don't get your day.
        with self._person(tier="acquaintance", name="Someone Else"):
            self.assertEqual(self.c._greeting_mood_aside(1), "")
        # A bland day has nothing worth mentioning.
        with self._person(), mock.patch.object(config, "REX_MOOD_SEEDS", [_BLAND]):
            rex_mood.clear()
            self.assertEqual(self.c._greeting_mood_aside(1), "")
        # Feature flag.
        with self._person(), mock.patch.object(
            config, "REX_MOOD_GREETING_ASIDE_ENABLED", False
        ):
            self.assertEqual(self.c._greeting_mood_aside(1), "")

    def test_the_creator_qualifies_regardless_of_tier(self):
        rex_mood.ensure_today()
        with self._person(tier="acquaintance"), \
             mock.patch.object(self.c.person_specials, "is_rex_creator", return_value=True):
            self.assertNotEqual(self.c._greeting_mood_aside(1), "")

    def test_greeting_and_lull_share_ONE_spend_per_day(self):
        # The point of sharing the persisted `spoken` flag: hearing about his day at
        # the hello must mean not hearing about it again in a lull that evening.
        rex_mood.ensure_today()
        with self._person():
            self.assertNotEqual(self.c._greeting_mood_aside(1), "")
            rex_mood.note_spoken()                      # the greeting was dispatched
            self.assertEqual(self.c._greeting_mood_aside(1), "")
            self.assertIsNone(I._lean_mood_share_cue(1))

    def test_bad_person_id_is_safe(self):
        for bad in (None, "Bret"):
            self.assertEqual(self.c._greeting_mood_aside(bad), "")

    def test_a_broken_lookup_never_breaks_the_greeting(self):
        rex_mood.ensure_today()
        with mock.patch.object(rex_mood, "share_cue", side_effect=RuntimeError("boom")), \
             self._person():
            self.assertEqual(self.c._greeting_mood_aside(1), "")

    def test_the_ladder_appends_it_last_and_spends_on_dispatch(self):
        # Structural guard: the clause must be appended AFTER the wellbeing-ask
        # clause (so it is the last word on mood), gated on the label whitelist,
        # and spent inside the `if queued:` side-effect block.
        import inspect
        src = inspect.getsource(self.c._step_presence_tracking)
        self.assertIn("_greeting_allows_mood_aside(label)", src)
        self.assertIn("_greeting_mood_aside(person_db_id)", src)
        self.assertIn("_mood_aside_used", src)
        self.assertLess(src.index("_wellbeing_ask_clause(person_db_id)"),
                        src.index("_greeting_mood_aside(person_db_id)"))


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
