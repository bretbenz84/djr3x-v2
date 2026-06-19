"""
Time-of-day rollover reaction: when the part of day changes (morning → afternoon →
evening → night → late_night), Rex makes one spontaneous remark, at most once per
transition per session. Mirrors the weather/notable-date change-detection blocks in
_step_proactive_reactions.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import config


def _profile():
    # _step_proactive_reactions only reads suppress_proactive / rapid_exchange.
    return types.SimpleNamespace(suppress_proactive=False, rapid_exchange=False)


class TimeOfDayReactionTest(unittest.TestCase):
    def setUp(self):
        from intelligence import consciousness
        self.c = consciousness
        consciousness._acknowledged_tod.clear()
        self._saved_snapshot = consciousness._last_snapshot

    def tearDown(self):
        self.c._last_snapshot = self._saved_snapshot
        self.c._acknowledged_tod.clear()

    def _run(self, prev_tod, curr_tod):
        """Drive one proactive tick with prev/curr part-of-day, returning the captured
        _generate_and_speak call kwargs (or None if nothing fired)."""
        c = self.c
        captured = {}

        def _fake_speak(prompt, emotion, *, purpose="world_reaction", label="", metadata=None):
            captured["prompt"] = prompt
            captured["label"] = label
            captured["emotion"] = emotion

        c._last_snapshot = {"time": {"time_of_day": prev_tod}}
        snapshot = {"time": {"time_of_day": curr_tod}}

        with mock.patch.object(c, "_stage_animal_arrivals"), \
             mock.patch.object(c, "_fire_pending_animal_arrival_reaction", return_value=False), \
             mock.patch.object(c, "_startup_known_greeting_pending", return_value=False), \
             mock.patch.object(c, "is_identity_prompt_waiting_for_reply", return_value=False), \
             mock.patch.object(c, "_can_proactive_speak", return_value=True), \
             mock.patch.object(c, "_generate_and_speak", side_effect=_fake_speak):
            c._step_proactive_reactions(snapshot, _profile())
        return captured or None

    def test_rollover_fires_a_reaction(self):
        out = self._run("afternoon", "evening")
        self.assertIsNotNone(out)
        self.assertIn("evening", out["label"])
        self.assertIn("evening", out["prompt"].lower())

    def test_late_night_label_is_humanized(self):
        out = self._run("night", "late_night")
        self.assertIsNotNone(out)
        self.assertIn("late night", out["prompt"].lower())

    def test_no_rollover_no_reaction(self):
        self.assertIsNone(self._run("morning", "morning"))

    def test_fires_only_once_per_transition_per_session(self):
        self.assertIsNotNone(self._run("afternoon", "evening"))
        # Same transition again in the same session → already acknowledged, no re-fire.
        self.assertIsNone(self._run("afternoon", "evening"))

    def test_disabled_flag_suppresses(self):
        with mock.patch.object(config, "TIME_OF_DAY_REACTIONS_ENABLED", False):
            self.assertIsNone(self._run("afternoon", "evening"))


if __name__ == "__main__":
    unittest.main()
