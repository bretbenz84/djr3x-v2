"""When the person sets a boundary asking Rex to stop bringing up a topic
("do not ask about my back pain"), proactive check-ins about matching remembered
events must be MUTED — otherwise the celebration/emotional greeting keeps leading
with it even though they explicitly said to stop. Pins
`boundaries.apply_detected_boundary` -> `emotional_events.mute_matching_positive_events`.
"""

import unittest
from unittest import mock


class BoundaryMutesMatchingEventsTest(unittest.TestCase):
    def _apply(self, detected, *, kill_switch=True):
        from memory import boundaries, emotional_events
        with mock.patch.object(boundaries, "add_boundary", return_value=7), \
             mock.patch.object(boundaries, "deactivate_boundary"), \
             mock.patch.object(emotional_events, "mute_matching_positive_events",
                               return_value=[{"id": 2}]) as mute, \
             mock.patch.object(boundaries, "_boundary_mutes_events", return_value=kill_switch):
            result = boundaries.apply_detected_boundary(1, {"action": "add", **detected})
        return result, mute

    def test_ask_boundary_mutes_matching_events(self):
        result, mute = self._apply({"behavior": "ask", "topic": "back pain"})
        mute.assert_called_once()
        self.assertEqual(mute.call_args.args[0], 1)            # person_id
        self.assertEqual(mute.call_args.args[1], "back pain")  # topic
        self.assertEqual(result["behavior"], "ask")

    def test_mention_boundary_also_mutes(self):
        _, mute = self._apply({"behavior": "mention", "topic": "the trip"})
        mute.assert_called_once()

    def test_roast_boundary_does_not_mute_events(self):
        # "Don't roast me" is about teasing, not a topic to forget.
        _, mute = self._apply({"behavior": "roast", "topic": "anything"})
        mute.assert_not_called()

    def test_kill_switch_disables_mute(self):
        _, mute = self._apply({"behavior": "ask", "topic": "back pain"}, kill_switch=False)
        mute.assert_not_called()

    def test_clear_action_does_not_mute(self):
        from memory import boundaries, emotional_events
        with mock.patch.object(emotional_events, "mute_matching_positive_events") as mute, \
             mock.patch.object(boundaries, "deactivate_boundary"):
            boundaries.apply_detected_boundary(
                1, {"action": "clear", "behavior": "ask", "topic": "back pain"})
        mute.assert_not_called()


class ReconcileExistingBoundariesTest(unittest.TestCase):
    """Existing (prior-session) boundaries must still mute matching events —
    reconcile_event_mutes is called before picking a startup celebration."""

    def _reconcile(self, boundaries_rows, *, kill_switch=True):
        from memory import boundaries, emotional_events
        with mock.patch.object(boundaries, "get_boundaries", return_value=boundaries_rows), \
             mock.patch.object(emotional_events, "mute_matching_positive_events",
                               return_value=[{"id": 2}]) as mute, \
             mock.patch.object(boundaries, "_boundary_mutes_events", return_value=kill_switch):
            total = boundaries.reconcile_event_mutes(1)
        return total, mute

    def test_active_ask_boundary_is_reconciled(self):
        total, mute = self._reconcile([{"behavior": "ask", "topic": "back pain"}])
        mute.assert_called_once_with(1, "back pain", reason="boundary: ask back pain")
        self.assertEqual(total, 1)

    def test_roast_and_empty_topic_boundaries_are_skipped(self):
        _, mute = self._reconcile([
            {"behavior": "roast", "topic": "anything"},
            {"behavior": "ask", "topic": ""},
        ])
        mute.assert_not_called()

    def test_kill_switch_disables_reconcile(self):
        total, mute = self._reconcile([{"behavior": "ask", "topic": "back pain"}],
                                      kill_switch=False)
        mute.assert_not_called()
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
