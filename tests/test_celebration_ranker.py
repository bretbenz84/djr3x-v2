"""
Tests for the "what's worth bringing up" cold-open ranker: among the celebration
candidates that pass the _celebration_worth_leading_with gate, Rex leads with the
BEST one — ranked by did-they-invite-it (dominant) x recency x concreteness —
rather than just the first/most-recent that happens to pass.

The gate itself (_celebration_worth_leading_with) is covered by
test_conversation_revamp.CelebrationColdOpenGateTest and is unchanged here.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest import mock


def _ev(description, *, age_days=1.0, invited=False):
    when = datetime.utcnow() - timedelta(days=age_days)
    return {
        "description": description,
        "mentioned_at": when.strftime("%Y-%m-%d %H:%M:%S"),
        "person_invited_topic": 1 if invited else 0,
    }


class LeadScoreTest(unittest.TestCase):
    def _score(self, ev):
        from intelligence import consciousness
        return consciousness._celebration_lead_score(ev)

    def test_invited_outscores_uninvited_all_else_equal(self):
        base = "won the regional volleyball championship"
        self.assertGreater(
            self._score(_ev(base, age_days=5, invited=True)),
            self._score(_ev(base, age_days=5, invited=False)),
        )

    def test_more_recent_outscores_older_all_else_equal(self):
        base = "won the regional volleyball championship"
        self.assertGreater(
            self._score(_ev(base, age_days=1, invited=False)),
            self._score(_ev(base, age_days=20, invited=False)),
        )

    def test_concrete_milestone_outscores_borderline(self):
        concrete = _ev("won the regional volleyball championship", age_days=3, invited=True)
        borderline = _ev("had a decent enough day at the office", age_days=3, invited=True)
        self.assertGreater(self._score(concrete), self._score(borderline))

    def test_invited_bonus_outweighs_a_recency_edge(self):
        # an invited older milestone beats a slightly-more-recent uninvited one
        invited_old = _ev("got the new job at the observatory", age_days=12, invited=True)
        recent_uninvited = _ev("won the regional volleyball championship", age_days=1, invited=False)
        self.assertGreater(self._score(invited_old), self._score(recent_uninvited))


class PickRankerTest(unittest.TestCase):
    def _pick(self, candidates):
        from intelligence import consciousness
        with mock.patch(
            "memory.emotional_events.get_startup_celebrations", return_value=candidates
        ):
            return consciousness._pick_due_celebration_checkin(1)

    def test_picks_invited_concrete_over_recent_uninvited(self):
        recent_uninvited = _ev("won the regional volleyball championship", age_days=1, invited=False)
        invited = _ev("got the new job at the observatory", age_days=12, invited=True)
        # order in the list must NOT decide it — the ranker does
        picked = self._pick([recent_uninvited, invited])
        self.assertEqual(picked["description"], "got the new job at the observatory")

    def test_picks_most_recent_when_both_uninvited_concrete(self):
        recent = _ev("won the regional championship", age_days=1, invited=False)
        older = _ev("launched the new product line", age_days=18, invited=False)
        picked = self._pick([older, recent])
        self.assertEqual(picked["description"], "won the regional championship")

    def test_kill_switch_restores_first_worthy_pick(self):
        import config
        from intelligence import consciousness
        recent_uninvited = _ev("won the regional volleyball championship", age_days=1, invited=False)
        invited = _ev("got the new job at the observatory", age_days=12, invited=True)
        with mock.patch(
            "memory.emotional_events.get_startup_celebrations",
            return_value=[recent_uninvited, invited],
        ):
            with mock.patch.object(config, "PRESENCE_CELEBRATION_RANK_ENABLED", False):
                picked = consciousness._pick_due_celebration_checkin(1)
        # rank off -> first that passes the gate (list order), not the best
        self.assertEqual(picked["description"], "won the regional volleyball championship")

    def test_no_worthy_candidate_returns_none(self):
        vague = _ev("the speaker feels proud of their problem-solving skills", age_days=2)
        self.assertIsNone(self._pick([vague]))

    def test_non_int_person_returns_none(self):
        from intelligence import consciousness
        self.assertIsNone(consciousness._pick_due_celebration_checkin(None))


if __name__ == "__main__":
    unittest.main()
