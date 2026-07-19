"""Cross-session anticipation throttle (the "Juneteenth every launch" fix).

consciousness._pick_anticipated_event must skip an upcoming event whose anticipated_at
(when REX last spoke an anticipation — stamped by events.mark_anticipated) is within
ANTICIPATION_REPEAT_COOLDOWN_HOURS, so Rex doesn't greet with the same plan on every
startup. It must NOT key on mentioned_at — that's the human's own mention, and throttling
on it meant a plan mentioned at 1 AM was never anticipated all day (field 2026-07-18:
the river float).
"""

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import config
from intelligence import consciousness as c


def _event(mentioned_hours_ago: float = 1.0,
           anticipated_hours_ago: float | None = None) -> dict:
    def _iso(h):
        return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()
    return {
        "id": 7,
        "event_name": "relaxing at home",
        "event_date": (date.today() + timedelta(days=2)).isoformat(),
        "mentioned_at": _iso(mentioned_hours_ago),
        "anticipated_at": _iso(anticipated_hours_ago) if anticipated_hours_ago is not None else None,
    }


class AnticipationCooldownTest(unittest.TestCase):
    def setUp(self):
        c._anticipated_events.clear()

    def tearDown(self):
        c._anticipated_events.clear()

    def test_recently_anticipated_event_is_skipped(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20), \
             mock.patch("memory.events.get_upcoming_events",
                        return_value=[_event(anticipated_hours_ago=1.0)]):
            self.assertIsNone(c._pick_anticipated_event(1))

    def test_never_anticipated_event_is_not_throttled_by_its_mention(self):
        # The river-float bug: mentioned 1h ago, never anticipated → must fire.
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20), \
             mock.patch("memory.events.get_upcoming_events",
                        return_value=[_event(mentioned_hours_ago=1.0)]):
            picked = c._pick_anticipated_event(1)
            self.assertIsNotNone(picked)
            self.assertEqual(picked["event_name"], "relaxing at home")

    def test_event_eligible_again_after_cooldown(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20), \
             mock.patch("memory.events.get_upcoming_events",
                        return_value=[_event(anticipated_hours_ago=48.0)]):
            picked = c._pick_anticipated_event(1)
            self.assertIsNotNone(picked)
            self.assertEqual(picked["event_name"], "relaxing at home")

    def test_cooldown_zero_disables_the_throttle(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 0), \
             mock.patch("memory.events.get_upcoming_events",
                        return_value=[_event(anticipated_hours_ago=0.1)]):
            self.assertIsNotNone(c._pick_anticipated_event(1))


if __name__ == "__main__":
    unittest.main()
