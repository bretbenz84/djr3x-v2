"""Evening workday check-in cue (owner 2026-08-03): on a weekday evening after
WORKDAY_CHECKIN_START_HOUR, a known repeat visitor gets ONE "how was work
today?" (profession stored) or "how was your day?" (no profession) — once per
person per day, durable across restarts, and deliberately not every day (one
probability roll per day)."""

import unittest
from datetime import datetime
from types import SimpleNamespace as NS
from unittest import mock

import config
from intelligence import interaction
from intelligence import lean_brain
from tests._lean_impulse_state import reset_impulse_state


def _one_chunk_stream(text):
    return [NS(choices=[NS(delta=NS(content=text))])]


_TUE_EVENING = datetime(2026, 8, 4, 18, 30)   # Tuesday 6:30pm
_TUE_NOON = datetime(2026, 8, 4, 12, 0)
_TUE_LATE = datetime(2026, 8, 4, 23, 30)
_SAT_EVENING = datetime(2026, 8, 8, 18, 30)   # Saturday


class _CueCase(unittest.TestCase):
    def setUp(self):
        reset_impulse_state(self)
        self._asked = mock.patch.object(
            interaction.rel_memory, "was_proactive_asked", return_value=False
        )
        self._asked.start()
        self.addCleanup(self._asked.stop)
        self._facts = mock.patch.object(
            interaction.facts_memory, "get_facts",
            return_value=[{"category": "job", "key": "job_title", "value": "trainer"}],
        )
        self._facts.start()
        self.addCleanup(self._facts.stop)
        self._roll = mock.patch.object(interaction.random, "random", return_value=0.0)
        self._roll.start()
        self.addCleanup(self._roll.stop)
        self._session = mock.patch.object(
            interaction.conv_memory, "get_session_transcript", return_value=[]
        )
        self._session.start()
        self.addCleanup(self._session.stop)

    def _cue(self, now=_TUE_EVENING, person_id=1):
        return interaction._lean_workday_checkin_cue(person_id, now=now)


class WorkdayCheckinGatesTest(_CueCase):
    def test_weekday_evening_with_profession_offers_work_variant(self):
        cue = self._cue()
        self.assertIsNotNone(cue)
        self.assertEqual(cue["kind"], "work")
        self.assertEqual(cue["profession"], "trainer")
        self.assertEqual(cue["topic_key"], "workday_checkin:2026-08-04")

    def test_no_profession_offers_day_variant(self):
        with mock.patch.object(interaction.facts_memory, "get_facts", return_value=[]):
            cue = self._cue()
        self.assertIsNotNone(cue)
        self.assertEqual(cue["kind"], "day")

    def test_before_five_pm_does_not_fire(self):
        self.assertIsNone(self._cue(now=_TUE_NOON))

    def test_after_end_hour_does_not_fire(self):
        self.assertIsNone(self._cue(now=_TUE_LATE))

    def test_weekend_does_not_fire(self):
        self.assertIsNone(self._cue(now=_SAT_EVENING))

    def test_already_asked_today_does_not_fire(self):
        with mock.patch.object(
            interaction.rel_memory, "was_proactive_asked", return_value=True
        ):
            self.assertIsNone(self._cue())

    def test_unknown_person_does_not_fire(self):
        self.assertIsNone(self._cue(person_id=None))

    def test_kill_switch(self):
        with mock.patch.object(config, "WORKDAY_CHECKIN_ENABLED", False, create=True):
            self.assertIsNone(self._cue())

    def test_failed_probability_roll_sits_out_the_whole_day(self):
        rolls = []

        def _roll():
            rolls.append(1)
            return 0.99   # above WORKDAY_CHECKIN_PROBABILITY (0.8) — sit out

        with mock.patch.object(interaction.random, "random", side_effect=_roll):
            self.assertIsNone(self._cue())
            self.assertIsNone(self._cue())   # later lull, same day
        self.assertEqual(len(rolls), 1, "one roll per (person, day), memoized")

    def test_low_confidence_inferred_profession_falls_back_to_day_variant(self):
        # job_title='trainer' at conf 0.55 (inferred from Bret training the
        # robot) produced "How was work today, trainer?" — field 2026-08-07.
        with mock.patch.object(
            interaction.facts_memory, "get_facts",
            return_value=[{"category": "job", "key": "job_title",
                          "value": "trainer", "confidence": 0.55}],
        ):
            cue = self._cue()
        self.assertIsNotNone(cue)
        self.assertEqual(cue["kind"], "day")

    def test_confident_profession_still_offers_work_variant(self):
        with mock.patch.object(
            interaction.facts_memory, "get_facts",
            return_value=[{"category": "job", "key": "job_title",
                          "value": "welder", "confidence": 0.95}],
        ):
            cue = self._cue()
        self.assertEqual(cue["kind"], "work")
        self.assertEqual(cue["profession"], "welder")

    def test_work_already_discussed_this_session_does_not_fire(self):
        # "I just got home from work" → Rex asked about it → the evening cue
        # must not re-ask "how was work today?" minutes later (field 2026-08-07).
        with mock.patch.object(
            interaction.conv_memory, "get_session_transcript",
            return_value=[{"speaker": "Bret Benziger",
                          "text": "I just got home from work."}],
        ):
            self.assertIsNone(self._cue())

    def test_profession_from_key_when_category_differs(self):
        with mock.patch.object(
            interaction.facts_memory, "get_facts",
            return_value=[{"category": "other", "key": "profession", "value": "welder"}],
        ):
            cue = self._cue()
        self.assertEqual(cue["kind"], "work")
        self.assertEqual(cue["profession"], "welder")


class WorkdayCheckinInstructionTest(unittest.TestCase):
    def _captured_instruction(self, cue):
        captured = []

        def fake_create(client, **kwargs):
            captured.append(kwargs["messages"][-1]["content"])
            return _one_chunk_stream("How was work today, chief?")

        with mock.patch.object(lean_brain.llm_compat, "create", side_effect=fake_create):
            line = lean_brain.consider_initiating(
                person_id=None, transcript=[], workday_checkin=cue,
            )
        return line, captured[0]

    def test_work_variant_mentions_profession_and_demands_the_question(self):
        line, instruction = self._captured_instruction(
            {"topic_key": "workday_checkin:2026-08-04", "kind": "work",
             "profession": "trainer"}
        )
        self.assertTrue(line.endswith("?"))
        self.assertIn("how was work today?", instruction)
        self.assertIn("trainer", instruction)
        self.assertIn("You MUST ask it", instruction)

    def test_day_variant_asks_about_their_day(self):
        _line, instruction = self._captured_instruction(
            {"topic_key": "workday_checkin:2026-08-04", "kind": "day",
             "profession": ""}
        )
        self.assertIn("how was your day?", instruction)
        self.assertNotIn("their work", instruction)


if __name__ == "__main__":
    unittest.main()
