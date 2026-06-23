"""
World-reaction dedupe (#16). When several world-changes co-occur in one tick (notable
date + time-of-day + weather), only ONE is chosen by random.choice and spoken. The bug:
every candidate marked itself acknowledged at APPEND time, so the un-chosen ones were
permanently swallowed — Rex never voiced commentary he'd silently "used up". The fix
defers the dedupe-consume until after selection, keyed to the chosen trigger.
"""

import types
import unittest
from unittest import mock

import config
from intelligence import consciousness as c


class WorldReactionDedupeTest(unittest.TestCase):
    def setUp(self):
        self._saved = (
            set(c._acknowledged_dates),
            set(c._acknowledged_tod),
            set(c._acknowledged_weather_signatures),
            c._last_snapshot,
        )
        c._acknowledged_dates.clear()
        c._acknowledged_tod.clear()
        c._acknowledged_weather_signatures.clear()
        # Prior tick: morning, no notable date — so the snapshot below is a real change.
        c._last_snapshot = {"time": {"time_of_day": "morning", "notable_date": None},
                            "weather": {"available": False}, "people": []}

    def tearDown(self):
        dates, tod, weather, snap = self._saved
        c._acknowledged_dates.clear(); c._acknowledged_dates.update(dates)
        c._acknowledged_tod.clear(); c._acknowledged_tod.update(tod)
        c._acknowledged_weather_signatures.clear(); c._acknowledged_weather_signatures.update(weather)
        c._last_snapshot = snap

    def _run_with_two_triggers(self, pick_label_substr):
        """Drive one tick where a notable-date AND a time-of-day change co-occur, and
        force random.choice to pick whichever trigger's label contains pick_label_substr."""
        snapshot = {"time": {"time_of_day": "afternoon", "notable_date": "Christmas"},
                    "weather": {"available": False}, "people": []}
        profile = types.SimpleNamespace(suppress_proactive=False, rapid_exchange=False)

        def choose(seq):
            for item in seq:
                if pick_label_substr in (item.get("label") or ""):
                    return item
            return seq[0]

        with mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_startup_known_greeting_pending", return_value=False), \
             mock.patch.object(c, "is_identity_prompt_waiting_for_reply", return_value=False), \
             mock.patch.object(c, "_stage_animal_arrivals"), \
             mock.patch.object(c, "_fire_pending_animal_arrival_reaction", return_value=False), \
             mock.patch.object(c, "_generate_and_speak"), \
             mock.patch.object(c.random, "choice", side_effect=choose), \
             mock.patch.object(config, "TIME_OF_DAY_REACTIONS_ENABLED", True), \
             mock.patch.object(config, "WEATHER_PROACTIVE_REACTIONS_ENABLED", False):
            c._step_proactive_reactions(snapshot, profile)

    def test_unchosen_trigger_stays_retryable(self):
        # Choose the time-of-day trigger; the notable-date one must NOT be acknowledged.
        self._run_with_two_triggers("time of day")
        self.assertIn("afternoon", c._acknowledged_tod)        # chosen -> consumed
        self.assertNotIn("Christmas", c._acknowledged_dates)   # un-chosen -> can fire next tick

    def test_chosen_trigger_is_consumed(self):
        # Choose the notable-date trigger; the time-of-day one must NOT be acknowledged.
        self._run_with_two_triggers("notable date")
        self.assertIn("Christmas", c._acknowledged_dates)      # chosen -> consumed
        self.assertNotIn("afternoon", c._acknowledged_tod)     # un-chosen -> can fire next tick


if __name__ == "__main__":
    unittest.main()
