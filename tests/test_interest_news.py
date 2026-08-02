"""
tests/test_interest_news.py — interest-tailored lull news (2026-08-02):
per-topic daily cache in awareness/current_events.py plus the lean cue that
prefers a person's interest stories over the generic headline pool, and the
interest-discovery ask. No network: fetches are stubbed.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from awareness import current_events


def _story(h="SNW episode drops", s="A new episode aired this week."):
    return {"headline": h, "summary": s, "topic": "tv", "mentioned": False}


class InterestNewsCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = config.CURRENT_EVENTS_PATH
        config.CURRENT_EVENTS_PATH = str(Path(self._tmp.name) / "current_events.json")
        current_events._interest_fetches_today.update(date=None, count=0)
        current_events._interest_refresh_inflight.clear()

    def tearDown(self):
        config.CURRENT_EVENTS_PATH = self._orig_path
        self._tmp.cleanup()

    def test_refresh_fetches_and_pick_returns_unmentioned(self):
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            return_value=[_story()],
        ) as fetch:
            current_events.refresh_interest_news(["Star Trek"])
        fetch.assert_called_once_with("star trek")
        picked = current_events.pick_interest_story(["star trek"])
        self.assertIsNotNone(picked)
        topic, story = picked
        self.assertEqual(topic, "star trek")
        self.assertEqual(story["headline"], "SNW episode drops")

    def test_same_day_refresh_is_cached(self):
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            return_value=[_story()],
        ) as fetch:
            current_events.refresh_interest_news(["volleyball"])
            current_events.refresh_interest_news(["Volleyball"])   # case-normalized
        fetch.assert_called_once()

    def test_daily_budget_caps_fetches(self):
        with (
            mock.patch.object(config, "INTEREST_NEWS_MAX_TOPICS_PER_DAY", 2, create=True),
            mock.patch.object(
                current_events, "_fetch_interest_news_via_web_search",
                return_value=[_story()],
            ) as fetch,
        ):
            current_events.refresh_interest_news(["a1", "a2", "a3"])
        self.assertEqual(fetch.call_count, 2)

    def test_mark_mentioned_spends_the_story(self):
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            return_value=[_story()],
        ):
            current_events.refresh_interest_news(["star trek"])
        topic, story = current_events.pick_interest_story(["star trek"])
        current_events.mark_interest_story_mentioned(topic, story)
        self.assertIsNone(current_events.pick_interest_story(["star trek"]))

    def test_stale_entry_is_not_served(self):
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            return_value=[_story()],
        ):
            current_events.refresh_interest_news(["star trek"])
        data = current_events._load()
        old = (datetime.now() - timedelta(hours=50)).isoformat(timespec="seconds")
        data["interest_news"]["star trek"]["fetched_at"] = old
        current_events._save(data)
        self.assertIsNone(current_events.pick_interest_story(["star trek"]))

    def test_failed_fetch_keeps_previous_entry(self):
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            return_value=[_story()],
        ):
            current_events.refresh_interest_news(["star trek"])
        # Next-day refresh fails — yesterday's stories survive.
        data = current_events._load()
        data["interest_news"]["star trek"]["date"] = "2000-01-01"
        current_events._save(data)
        current_events._interest_fetches_today.update(date=None, count=0)
        with mock.patch.object(
            current_events, "_fetch_interest_news_via_web_search",
            side_effect=RuntimeError("boom"),
        ):
            current_events.refresh_interest_news(["star trek"])
        self.assertIsNotNone(current_events.pick_interest_story(["star trek"]))


class LeanInterestCueTest(unittest.TestCase):
    def setUp(self):
        import intelligence.interaction as I
        self.I = I
        self._saved = I._lean_news_mentioned_this_session
        I._lean_news_mentioned_this_session = False

    def tearDown(self):
        self.I._lean_news_mentioned_this_session = self._saved

    def test_news_cue_prefers_interest_story(self):
        I = self.I
        with (
            mock.patch.object(I, "_person_interest_topics",
                              return_value=["star trek"]),
            mock.patch("awareness.current_events.start_interest_refresh"),
            mock.patch("awareness.current_events.pick_interest_story",
                       return_value=("star trek", _story())),
            mock.patch("awareness.current_events.pick_story") as general,
        ):
            cue = I._lean_news_cue(1)
        general.assert_not_called()
        self.assertEqual(cue["interest_topic"], "star trek")

    def test_news_cue_falls_back_to_general(self):
        I = self.I
        with (
            mock.patch.object(I, "_person_interest_topics", return_value=[]),
            mock.patch("awareness.current_events.pick_story",
                       return_value=_story("World news", "A thing happened.")),
        ):
            cue = I._lean_news_cue(None)
        self.assertNotIn("interest_topic", cue or {})
        self.assertEqual(cue["headline"], "World news")

    def test_interest_discovery_cue_fires_for_sparse_catalogue(self):
        I = self.I
        with (
            mock.patch("memory.interests.get_interests_for_prompt",
                       return_value=[{"name": "astrophotography"}]),
            mock.patch.object(I.rel_memory, "was_proactive_asked",
                              return_value=False),
        ):
            cue = I._lean_interest_discovery_cue(1)
        self.assertIsNotNone(cue)
        self.assertIn("astrophotography", cue["known"])
        self.assertTrue(cue["topic_key"].startswith("interest_discovery:"))

    def test_interest_discovery_stands_down_when_catalogue_rich(self):
        I = self.I
        rows = [{"name": f"hobby{i}"} for i in range(5)]
        with mock.patch("memory.interests.get_interests_for_prompt",
                        return_value=rows):
            self.assertIsNone(I._lean_interest_discovery_cue(1))

    def test_interest_discovery_respects_durable_mark(self):
        I = self.I
        with (
            mock.patch("memory.interests.get_interests_for_prompt",
                       return_value=[]),
            mock.patch.object(I.rel_memory, "was_proactive_asked",
                              return_value=True),
        ):
            self.assertIsNone(I._lean_interest_discovery_cue(1))

    def test_unknown_person_gets_no_discovery_ask(self):
        self.assertIsNone(self.I._lean_interest_discovery_cue(None))


if __name__ == "__main__":
    unittest.main()
