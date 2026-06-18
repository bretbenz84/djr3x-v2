"""
Explicit-goodbye exit: when the user says a genuine sign-off and then leaves the
camera view, Rex should treat the conversation as closed and stop trying to
re-engage an empty room (no departure quip, no idle banter / monologue) until
they come back.

Regression for the 2026-06-17 session where Bret said "I've gotta go now",
walked off-camera, and Rex fired a departure quip and two idle-banter lines at
the empty room before Bret had to return and manually say "Shut down."
"""

import unittest
from unittest import mock

from intelligence import end_thread


class FarewellDepartureTests(unittest.TestCase):
    def setUp(self):
        end_thread.clear()

    def tearDown(self):
        end_thread.clear()

    def test_goodbye_then_departure_closes_conversation(self):
        end_thread.note_user_turn(
            "Well, it's been nice talking to you. I've gotta go now."
        )
        self.assertTrue(end_thread.recent_farewell())
        self.assertTrue(end_thread.note_farewell_departure())
        self.assertTrue(end_thread.is_conversation_closed())
        # Every inline proactive path backs off on this flag.
        self.assertTrue(end_thread.is_grace_active())
        # While closed nobody's there — allow nothing, not even check-ins.
        self.assertFalse(end_thread.can_proactive_purpose("idle_monologue"))
        self.assertFalse(end_thread.can_proactive_purpose("presence_reaction"))
        self.assertFalse(end_thread.can_proactive_purpose("emotional_checkin"))

    def test_topic_closure_does_not_arm_latch(self):
        # A topic landing ("never mind") is a soft close, not a goodbye — stepping
        # off-camera after it should NOT make Rex go dormant.
        end_thread.note_user_turn("Anyway, never mind.")
        self.assertFalse(end_thread.recent_farewell())
        self.assertFalse(end_thread.note_farewell_departure())
        self.assertFalse(end_thread.is_conversation_closed())

    def test_departure_without_recent_goodbye_does_not_close(self):
        # No farewell on record → a plain camera departure keeps normal behavior
        # (the quip still fires; only the latch is gated here).
        self.assertFalse(end_thread.note_farewell_departure())
        self.assertFalse(end_thread.is_conversation_closed())

    def test_presence_return_lifts_dormancy(self):
        end_thread.note_user_turn("see you later")
        end_thread.note_farewell_departure()
        self.assertTrue(end_thread.is_conversation_closed())
        end_thread.note_presence_return()
        self.assertFalse(end_thread.is_conversation_closed())
        self.assertFalse(end_thread.recent_farewell())
        self.assertTrue(end_thread.can_proactive_purpose("idle_monologue"))

    def test_new_substantive_turn_clears_latch(self):
        end_thread.note_user_turn("gotta go")
        end_thread.note_farewell_departure()
        self.assertTrue(end_thread.is_conversation_closed())
        # They came back and said something real → Rex is live again.
        end_thread.note_user_turn("Actually I wanted to show you my telescope")
        self.assertFalse(end_thread.is_conversation_closed())
        self.assertFalse(end_thread.recent_farewell())

    def test_goodbye_without_departure_keeps_soft_grace_only(self):
        # Said bye but still sitting there: soft grace only — care still allowed,
        # idle filler still suppressed (unchanged end-of-thread behavior).
        end_thread.note_user_turn("nice talking to you")
        self.assertTrue(end_thread.recent_farewell())
        self.assertFalse(end_thread.is_conversation_closed())
        self.assertTrue(end_thread.is_grace_active())
        self.assertTrue(end_thread.can_proactive_purpose("emotional_checkin"))
        self.assertFalse(end_thread.can_proactive_purpose("idle_monologue"))

    def test_closed_latch_self_expires_after_cap(self):
        end_thread.note_user_turn("bye")
        end_thread.note_farewell_departure()
        self.assertTrue(end_thread.is_conversation_closed())
        cap = end_thread._farewell_closed_max_secs()
        real = end_thread.time.monotonic
        with mock.patch.object(
            end_thread.time, "monotonic", lambda: real() + cap + 1.0
        ):
            self.assertFalse(end_thread.is_conversation_closed())
            self.assertFalse(end_thread.is_grace_active())

    def test_stale_goodbye_does_not_close_on_late_departure(self):
        # A goodbye that's already aged out of the window shouldn't latch closed
        # when a much-later departure happens.
        end_thread.note_user_turn("catch you later")
        window = end_thread._farewell_window_secs()
        real = end_thread.time.monotonic
        with mock.patch.object(
            end_thread.time, "monotonic", lambda: real() + window + 1.0
        ):
            self.assertFalse(end_thread.recent_farewell())
            self.assertFalse(end_thread.note_farewell_departure())
            self.assertFalse(end_thread.is_conversation_closed())


if __name__ == "__main__":
    unittest.main()
