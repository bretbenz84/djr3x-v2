"""
Give-space-after-heavy guard: after a heavy/grief disclosure (or while a grief flow is
open), Rex must NOT proactively re-engage — no idle banter, plans, or snark that could
re-probe the topic the user stepped back from. He still RESPONDS when spoken to.

Regression: 2026-06-18 live run — the user named a late parent and declined to discuss
it; ~30s later proactive idle banter asked "what's one thing about your mother you still
hear in your own head?", violating the just-set boundary.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock


class RecentlyHeavyWindowTest(unittest.TestCase):
    def setUp(self):
        from intelligence import callback_engine as cb
        self.cb = cb
        cb.reset_state_for_tests()

    def tearDown(self):
        self.cb.reset_state_for_tests()

    def test_not_heavy_initially(self):
        self.assertFalse(self.cb.recently_heavy())

    def test_heavy_after_note(self):
        self.cb.note_heavy_moment()
        self.assertTrue(self.cb.recently_heavy())

    def test_window_expires(self):
        import config
        self.cb.note_heavy_moment()
        window = float(getattr(config, "CALLBACK_SUPPRESS_AFTER_HEAVY_SECS", 1800.0))
        self.cb._last_heavy_at = time.monotonic() - (window + 5.0)
        self.assertFalse(self.cb.recently_heavy())


class SuppressProactiveAfterHeavyTest(unittest.TestCase):
    def setUp(self):
        from intelligence import callback_engine as cb
        from intelligence import interaction
        self.cb = cb
        self.interaction = interaction
        cb.reset_state_for_tests()
        interaction._grief_flow_state.clear()

    def tearDown(self):
        self.cb.reset_state_for_tests()
        self.interaction._grief_flow_state.clear()

    def test_false_when_calm(self):
        self.assertFalse(self.interaction._suppress_proactive_after_heavy(1))

    def test_true_during_sober_window(self):
        self.cb.note_heavy_moment()
        self.assertTrue(self.interaction._suppress_proactive_after_heavy(1))

    def test_true_while_grief_flow_active(self):
        self.interaction._grief_flow_state[1] = {
            "started_at": time.monotonic(), "step": "awaiting_consent",
        }
        self.assertTrue(self.interaction._suppress_proactive_after_heavy(1))

    def test_grief_flow_is_person_scoped(self):
        self.interaction._grief_flow_state[1] = {
            "started_at": time.monotonic(), "step": "awaiting_consent",
        }
        # Another person, no heavy moment → not suppressed.
        self.assertFalse(self.interaction._suppress_proactive_after_heavy(2))


class CanProactiveSpeakGivesSpaceTest(unittest.TestCase):
    """can_proactive_speak suppresses NON-salient proactive speech during the sober
    window (idle banter / plans / snark), so Rex gives space after a heavy moment."""

    def setUp(self):
        from intelligence import callback_engine as cb
        cb.reset_state_for_tests()

    def tearDown(self):
        from intelligence import callback_engine as cb
        cb.reset_state_for_tests()

    def test_nonsalient_suppressed_in_sober_window(self):
        from intelligence import speech_engine as se
        from intelligence import callback_engine as cb
        cb.note_heavy_moment()
        with mock.patch("intelligence.consciousness._can_speak", return_value=True), \
             mock.patch("intelligence.interaction.tell_about_flow_active", return_value=False), \
             mock.patch("intelligence.interaction.onboarding_flow_active", return_value=False):
            self.assertFalse(se.can_proactive_speak(salient=False))


if __name__ == "__main__":
    unittest.main()
