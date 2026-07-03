"""memory/trends.py — cross-session cadence + recurring-topic awareness.

Computed purely from existing rows (people + per-session conversations); the tests
drive it with synthetic histories and assert the human-shaped outputs: streaks,
frequency, the 2–60-day medium gap (previously uncovered by any greeting hook),
recurring topics, and the once-per-day cache.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from memory import trends


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _history(days_ago_list, topics=""):
    now = datetime.now(timezone.utc)
    return [
        {"session_date": _iso(now - timedelta(days=d)), "topics": topics}
        for d in days_ago_list
    ]


class TrendsTest(unittest.TestCase):
    def setUp(self):
        trends._cache.clear()
        self.now = datetime.now(timezone.utc)

    def _stats(self, *, person=None, history=None):
        person = person or {"visit_count": 10, "first_seen": _iso(self.now - timedelta(days=90)),
                            "last_seen": _iso(self.now - timedelta(days=1))}
        with mock.patch.object(trends.people_db, "get_person", return_value=person), \
             mock.patch.object(trends.conv_db, "get_conversation_history", return_value=history or []):
            return trends.visit_stats(1)

    def test_streak_detection(self):
        s = self._stats(history=_history([0, 1, 2, 5, 9]))
        self.assertEqual(s["streak_days"], 3)   # today, yesterday, day before

    def test_streak_allows_yesterday_anchor(self):
        # Visits yesterday + day before (none yet today) still count as a live streak.
        s = self._stats(history=_history([1, 2]))
        self.assertEqual(s["streak_days"], 2)

    def test_frequency_windows(self):
        s = self._stats(history=_history([0, 1, 3, 6, 12, 20, 40]))
        self.assertEqual(s["sessions_7d"], 4)
        self.assertEqual(s["sessions_30d"], 6)

    def test_recurring_topics_need_three_distinct_days(self):
        history = (
            _history([1], topics="volleyball, cooking")
            + _history([4], topics="volleyball")
            + _history([9], topics="volleyball, movies")
            + _history([12], topics="cooking")
        )
        s = self._stats(history=history)
        self.assertEqual(s["recurring_topics"], ["volleyball"])  # 3 days; cooking only 2

    def test_cadence_hook_streak_wins(self):
        trends._cache.clear()
        with mock.patch.object(trends, "visit_stats", return_value={
            "total_visits": 12, "days_known": 90, "sessions_7d": 5,
            "sessions_30d": 9, "streak_days": 2, "gap_days": 0.2, "recurring_topics": [],
        }):
            kind, phrase = trends.cadence_hook(1)
        self.assertEqual(kind, "streak")
        self.assertIn("third day in a row", phrase)

    def test_cadence_hook_medium_gap_fills_the_hole(self):
        # 2–60 days had NO greeting hook (recent_return < 48h, long_absence >= 60d).
        with mock.patch.object(trends, "visit_stats", return_value={
            "total_visits": 12, "days_known": 90, "sessions_7d": 0,
            "sessions_30d": 1, "streak_days": 0, "gap_days": 15.0, "recurring_topics": [],
        }):
            kind, phrase = trends.cadence_hook(1)
        self.assertEqual(kind, "medium_gap")
        self.assertIn("week", phrase)

    def test_cadence_hook_none_for_ordinary_visit(self):
        with mock.patch.object(trends, "visit_stats", return_value={
            "total_visits": 4, "days_known": 30, "sessions_7d": 1,
            "sessions_30d": 3, "streak_days": 0, "gap_days": 1.5, "recurring_topics": [],
        }):
            self.assertIsNone(trends.cadence_hook(1))

    def test_prompt_summary_compact_and_gated(self):
        with mock.patch.object(trends, "visit_stats", return_value={
            "total_visits": 22, "days_known": 120, "sessions_7d": 4,
            "sessions_30d": 10, "streak_days": 0, "gap_days": 0.5,
            "recurring_topics": ["volleyball", "cooking"],
        }):
            line = trends.summarize_for_prompt(1)
        self.assertIn("visit #23", line)
        self.assertIn("volleyball", line)
        self.assertLess(len(line.split()), 70)
        # a stranger with no story produces nothing
        with mock.patch.object(trends, "visit_stats", return_value={
            "total_visits": 1, "days_known": 0, "sessions_7d": 1,
            "sessions_30d": 1, "streak_days": 0, "gap_days": 0.1, "recurring_topics": [],
        }):
            self.assertEqual(trends.summarize_for_prompt(1), "")

    def test_stats_cached_per_day(self):
        person = {"visit_count": 5, "first_seen": _iso(self.now), "last_seen": _iso(self.now)}
        with mock.patch.object(trends.people_db, "get_person", return_value=person) as gp, \
             mock.patch.object(trends.conv_db, "get_conversation_history", return_value=[]):
            trends.visit_stats(1)
            trends.visit_stats(1)
        self.assertEqual(gp.call_count, 1)


if __name__ == "__main__":
    unittest.main()
