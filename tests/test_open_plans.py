"""Inject open plans into the live reply.

_build_person_context read emotional events but never the calendar, so mid-conversation
Rex didn't know you have a thing tomorrow. _open_plans_prompt_line surfaces the next 1-2
DATED near-term events as background awareness (restraint rule), skipping any the
proactive anticipation path already raised.
"""

import unittest
from datetime import date, timedelta
from unittest import mock


def _ev(name, days, eid=1):
    return {"id": eid, "event_name": name, "event_date": (date.today() + timedelta(days=days)).isoformat()}


class OpenPlansPromptLineTest(unittest.TestCase):
    def _line(self, events, *, anticipated=()):
        from intelligence import llm

        with mock.patch("memory.events.get_upcoming_events", return_value=list(events)), \
             mock.patch.object(llm, "_open_plan_anticipated",
                               side_effect=lambda pid, eid: eid in anticipated):
            return llm._open_plans_prompt_line(1)

    def test_disabled_returns_empty(self):
        from intelligence import llm

        with mock.patch.object(llm.config, "OPEN_PLANS_IN_REPLY_ENABLED", False):
            self.assertEqual(self._line([_ev("dentist", 1)]), "")

    def test_no_events_returns_empty(self):
        self.assertEqual(self._line([]), "")

    def test_tomorrow_is_surfaced_with_restraint_rule(self):
        line = self._line([_ev("the dentist", 1)])
        self.assertIn("the dentist (tomorrow)", line)
        self.assertIn("background awareness", line)
        self.assertIn("do", line.lower())  # the "do NOT lead with it" restraint

    def test_today(self):
        self.assertIn("the concert (today)", self._line([_ev("the concert", 0)]))

    def test_future_dated_uses_the_date(self):
        d = (date.today() + timedelta(days=4)).isoformat()
        self.assertIn(f"the trip (on {d})", self._line([_ev("the trip", 4)]))

    def test_beyond_window_excluded(self):
        self.assertEqual(self._line([_ev("far thing", 100)]), "")

    def test_past_event_excluded(self):
        self.assertEqual(self._line([_ev("yesterday", -2)]), "")

    def test_undated_excluded(self):
        self.assertEqual(self._line([{"id": 1, "event_name": "someday", "event_date": None}]), "")

    def test_already_anticipated_is_skipped(self):
        self.assertEqual(self._line([_ev("dentist", 1, eid=7)], anticipated={7}), "")

    def test_capped_to_max(self):
        evs = [_ev(f"thing{i}", i + 1, eid=i) for i in range(5)]
        line = self._line(evs)  # default OPEN_PLANS_MAX = 2
        self.assertIn("thing0", line)
        self.assertIn("thing1", line)
        self.assertNotIn("thing2", line)  # capped — the 3rd is dropped


class EventRecentlyAnticipatedTest(unittest.TestCase):
    def test_membership_and_bad_input(self):
        from intelligence import consciousness as cns

        cns._anticipated_events.add((3, 9))
        try:
            self.assertTrue(cns.event_recently_anticipated(3, 9))
            self.assertFalse(cns.event_recently_anticipated(3, 10))
            self.assertFalse(cns.event_recently_anticipated(None, 9))
        finally:
            cns._anticipated_events.discard((3, 9))


if __name__ == "__main__":
    unittest.main()
