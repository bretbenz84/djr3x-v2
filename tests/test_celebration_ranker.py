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


class StartupCelebrationCooldownTest(unittest.TestCase):
    """Once a celebration has led a greeting it must NOT re-lead every restart.
    `get_startup_celebrations` suppresses an acknowledged event for
    PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS (cross-process), instead of the old
    within-process-only `last_acknowledged_at < process_started_iso` dedup."""

    def _captured_query(self, cooldown_days):
        from memory import emotional_events as ee
        captured = {}

        def fake_fetchall(query, params):
            captured["query"] = query
            captured["params"] = params
            return []

        with mock.patch.object(ee.config, "PRESENCE_CELEBRATION_RELEAD_COOLDOWN_DAYS", cooldown_days), \
             mock.patch.object(ee.db, "fetchall", side_effect=fake_fetchall):
            ee.get_startup_celebrations(1, "2026-01-01 00:00:00", limit=3)
        return captured

    def test_cooldown_applies_a_cross_process_window(self):
        cap = self._captured_query(14)
        self.assertIn("datetime('now'", cap["query"])
        self.assertIn("-14 days", cap["params"])
        # the per-process iso is NOT used when the cooldown is active
        self.assertNotIn("2026-01-01 00:00:00", cap["params"])

    def test_cooldown_zero_restores_per_process_behavior(self):
        cap = self._captured_query(0)
        self.assertIn("2026-01-01 00:00:00", cap["params"])
        self.assertNotIn("-0 days", cap["params"])


class ColdOpenCallbackRankerTest(unittest.TestCase):
    """The cold-open ranker EXTENDED across facts/interests: when nothing higher
    applies, Rex leads with the best remembered interest/fact, scored by the SAME
    invited × recency × concreteness lead-score as celebrations."""

    def test_lead_score_prefers_invited_then_recency(self):
        from intelligence import consciousness as c
        invited_recent = {"invited": True, "recency_iso": "2026-06-04 12:00:00", "text": "astrophotography"}
        not_invited = {"invited": False, "recency_iso": "2026-06-04 12:00:00", "text": "astrophotography"}
        invited_old = {"invited": True, "recency_iso": "2025-01-01 12:00:00", "text": "astrophotography"}
        self.assertGreater(c._cold_open_lead_score(invited_recent), c._cold_open_lead_score(not_invited))
        self.assertGreater(c._cold_open_lead_score(invited_recent), c._cold_open_lead_score(invited_old))

    def test_candidates_include_interests_and_activity_facts_drop_favorites(self):
        from intelligence import consciousness as c
        interests = [
            {"name": "astrophotography", "last_mentioned_at": "2026-06-04 12:00:00"},
            {"name": "birdwatching", "last_mentioned_at": "2025-01-01 12:00:00"},
        ]
        facts = [
            {"category": "favorite", "value": "mint chocolate chip ice cream",
             "source": "explicit", "freshness_label": "fresh"},  # NOT cold-open material
            {"category": "project", "value": "building a robot DJ", "source": "explicit",
             "last_mentioned_at": "2026-06-04 12:00:00", "freshness_label": "fresh"},
        ]
        with (
            mock.patch("memory.interests.get_interest_hooks", return_value=interests),
            mock.patch("memory.facts.get_prompt_worthy_facts", return_value=facts),
        ):
            cands = c._cold_open_callback_candidates(1)
        topics = {x["topic"] for x in cands}
        self.assertIn("astrophotography", topics)            # interest in
        self.assertIn("building a robot DJ", topics)         # project fact in
        self.assertNotIn("mint chocolate chip ice cream", topics)  # favorite excluded
        # The recent astrophotography interest outranks the older birdwatching.
        self.assertEqual(max(cands, key=c._cold_open_lead_score)["topic"], "astrophotography")

    def test_pick_returns_best_and_respects_flag(self):
        from intelligence import consciousness as c
        interests = [{"name": "astrophotography", "last_mentioned_at": "2026-06-04 12:00:00"}]
        with (
            mock.patch("memory.interests.get_interest_hooks", return_value=interests),
            mock.patch("memory.facts.get_prompt_worthy_facts", return_value=[]),
        ):
            picked = c._pick_cold_open_callback(1)
            self.assertEqual(picked["topic"], "astrophotography")
            with mock.patch.object(c.config, "COLD_OPEN_INTEREST_RANK_ENABLED", False):
                self.assertIsNone(c._pick_cold_open_callback(1))

    def test_pick_returns_none_when_no_candidates(self):
        from intelligence import consciousness as c
        with (
            mock.patch("memory.interests.get_interest_hooks", return_value=[]),
            mock.patch("memory.facts.get_prompt_worthy_facts", return_value=[]),
        ):
            self.assertIsNone(c._pick_cold_open_callback(1))


if __name__ == "__main__":
    unittest.main()
