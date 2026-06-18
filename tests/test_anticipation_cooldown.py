"""Cross-session anticipation throttle (the "Juneteenth every launch" fix).

consciousness._pick_anticipated_event must skip an upcoming event whose mentioned_at is
within ANTICIPATION_REPEAT_COOLDOWN_HOURS, so Rex doesn't greet with the same plan on
every startup. mentioned_at is refreshed by events.mark_anticipated on each anticipation.
"""

import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

import config
from intelligence import consciousness as c


def _event(mentioned_hours_ago: float) -> dict:
    when = (datetime.now(timezone.utc) - timedelta(hours=mentioned_hours_ago)).isoformat()
    return {
        "id": 7,
        "event_name": "relaxing at home",
        "event_date": (date.today() + timedelta(days=2)).isoformat(),
        "mentioned_at": when,
    }


class AnticipationCooldownTest(unittest.TestCase):
    def setUp(self):
        c._anticipated_events.clear()

    def tearDown(self):
        c._anticipated_events.clear()

    def test_recently_mentioned_event_is_skipped(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20), \
             mock.patch("memory.events.get_upcoming_events", return_value=[_event(1.0)]):
            self.assertIsNone(c._pick_anticipated_event(1))

    def test_event_eligible_again_after_cooldown(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 20), \
             mock.patch("memory.events.get_upcoming_events", return_value=[_event(48.0)]):
            picked = c._pick_anticipated_event(1)
            self.assertIsNotNone(picked)
            self.assertEqual(picked["event_name"], "relaxing at home")

    def test_cooldown_zero_disables_the_throttle(self):
        with mock.patch.object(config, "ANTICIPATION_REPEAT_COOLDOWN_HOURS", 0), \
             mock.patch("memory.events.get_upcoming_events", return_value=[_event(0.1)]):
            self.assertIsNotNone(c._pick_anticipated_event(1))


if __name__ == "__main__":
    unittest.main()
