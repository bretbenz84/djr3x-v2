"""Regression test for the event-follow-up "did not happen" resolver.

When Rex asks "how did X go?" and the user says they never went / didn't go,
the follow-up must resolve (so it stops being re-injected into the agenda),
rather than being held open as a repair and re-asked every turn. This pins the
denial detector that drives that resolution.
"""

import unittest
from unittest import mock

from intelligence import interaction


class FollowupDidNotHappenTest(unittest.TestCase):
    def test_denials_resolve_followup(self):
        for text in [
            "I never went to that",
            "I'm not going to that, I didn't go to that",
            "I never went",
            "I didn't go",
            "I did not go",
            "I never made it",
            "I didn't end up going",
            "Couldn't make it",
            "I wasn't able to go",
            "that never happened",
        ]:
            with self.subTest(text=text):
                self.assertTrue(interaction._followup_event_did_not_happen(text))

    def test_real_answers_do_not_resolve(self):
        for text in [
            "It was amazing",
            "I'm doing great",
            "I went and it was fun",
            "Not going to lie, it was incredible",
            "what are you talking about?",
            "Best concert of my life",
        ]:
            with self.subTest(text=text):
                self.assertFalse(interaction._followup_event_did_not_happen(text))

    def test_empty(self):
        self.assertFalse(interaction._followup_event_did_not_happen(""))
        self.assertFalse(interaction._followup_event_did_not_happen(None))


class MemoryFollowupCadenceTest(unittest.TestCase):
    """The moderate clamp: at most one proactive follow-up per conversational lull.

    Pins the gap/cooldown/flat gates and the per-session anti-repeat that together
    stop Rex from running down a checklist of remembered events turn-after-turn
    (the Disneyland → swimming → Disneyland-again interrogation from the live run).
    """

    def setUp(self):
        # Snapshot + reset the module-level cadence state around each test.
        self._saved = (
            interaction._last_followup_exchange,
            interaction._last_followup_at,
            set(interaction._fired_followup_event_ids),
        )
        interaction._last_followup_exchange = -(10**9)
        interaction._last_followup_at = 0.0
        interaction._fired_followup_event_ids = set()

    def tearDown(self):
        (
            interaction._last_followup_exchange,
            interaction._last_followup_at,
            interaction._fired_followup_event_ids,
        ) = self._saved

    def _allows(self, exchange, monotonic=1000.0, flat=False):
        with mock.patch.object(interaction, "_conversation_exchange_count", return_value=exchange), \
             mock.patch.object(interaction.topic_thread, "arc_reads_flat", return_value=flat), \
             mock.patch.object(interaction.time, "monotonic", return_value=monotonic):
            return interaction._memory_followup_cadence_allows()

    def test_first_followup_allowed(self):
        # Nothing has fired yet → the first follow-up of the session may fire.
        self.assertTrue(self._allows(exchange=2))

    def test_blocked_within_exchange_gap(self):
        interaction._last_followup_exchange = 10
        interaction._last_followup_at = 0.0  # cooldown disabled, isolate the gap gate
        # gap = 12 - 10 = 2 < FOLLOWUP_MIN_GAP_EXCHANGES (5)
        self.assertFalse(self._allows(exchange=12))

    def test_allowed_past_exchange_gap(self):
        interaction._last_followup_exchange = 10
        interaction._last_followup_at = 0.0
        # gap = 16 - 10 = 6 >= 5
        self.assertTrue(self._allows(exchange=16))

    def test_blocked_within_cooldown(self):
        interaction._last_followup_exchange = 10
        interaction._last_followup_at = 100.0
        # gap satisfied (6), but only 30s < FOLLOWUP_COOLDOWN_SECS (60) elapsed
        self.assertFalse(self._allows(exchange=16, monotonic=130.0))

    def test_allowed_past_cooldown(self):
        interaction._last_followup_exchange = 10
        interaction._last_followup_at = 100.0
        # gap satisfied AND 100s >= 60s cooldown
        self.assertTrue(self._allows(exchange=16, monotonic=200.0))

    def test_flat_room_suppresses(self):
        # Even with a huge gap and no cooldown, a flat-reading room blocks.
        self.assertFalse(self._allows(exchange=999, flat=True))

    def test_session_reset_clears_state(self):
        interaction._last_followup_exchange = 50
        interaction._last_followup_at = 500.0
        interaction._fired_followup_event_ids = {4, 5}
        # Transcript shrank (new session) → state resets and a follow-up may fire.
        self.assertTrue(self._allows(exchange=2))
        self.assertEqual(interaction._fired_followup_event_ids, set())
        self.assertEqual(interaction._last_followup_at, 0.0)

    def test_note_fired_records_cadence_and_antirepeat(self):
        with mock.patch.object(interaction, "_conversation_exchange_count", return_value=8), \
             mock.patch.object(interaction.time, "monotonic", return_value=500.0):
            interaction._note_memory_followup_fired(4)
        self.assertEqual(interaction._last_followup_exchange, 8)
        self.assertEqual(interaction._last_followup_at, 500.0)
        self.assertIn(4, interaction._fired_followup_event_ids)

    def test_note_fired_ignores_none_event_id(self):
        with mock.patch.object(interaction, "_conversation_exchange_count", return_value=8), \
             mock.patch.object(interaction.time, "monotonic", return_value=500.0):
            interaction._note_memory_followup_fired(None)
        self.assertEqual(interaction._fired_followup_event_ids, set())


if __name__ == "__main__":
    unittest.main()
